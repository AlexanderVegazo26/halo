// Token vocabulary trie, byte matcher and token-mask engine (TRD §25).

#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "halo/core/error.h"
#include "halo/sampling/structured.h"
#include "halo/tokenizer/tokenizer.h"
#include "sampling/grammar_impl.h"

namespace halo::sampling {

using detail::kDead;
using detail::Runtime;

// ---------------------------------------------------------------------------------------
// TokenVocab
// ---------------------------------------------------------------------------------------

std::shared_ptr<const TokenVocab> TokenVocab::build(const tokenizer::Tokenizer& tok) {
    const std::size_t n = tok.vocab_size();
    std::vector<std::string> pieces(n);
    std::vector<bool> included(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        const auto id = static_cast<std::int32_t>(i);
        pieces[i] = tok.token_to_piece(id);
        included[i] = tok.token_type(id) == tokenizer::TokenType::Normal && !pieces[i].empty();
    }
    return from_pieces(std::move(pieces), std::move(included));
}

std::shared_ptr<const TokenVocab> TokenVocab::from_pieces(std::vector<std::string> pieces,
                                                          std::vector<bool> included) {
    HALO_CHECK(pieces.size() == included.size(), ErrorCode::Api, "TokenVocab: pieces/included size mismatch");
    HALO_CHECK(pieces.size() < (std::size_t{1} << 31), ErrorCode::Api, "TokenVocab: vocabulary too large");
    auto v = std::make_shared<TokenVocab>(PrivateKey{});
    v->pieces_ = std::move(pieces);
    v->included_ = std::move(included);
    std::vector<std::int32_t> ids;
    for (std::size_t i = 0; i < v->pieces_.size(); ++i) {
        if (v->included_[i]) {
            HALO_CHECK(!v->pieces_[i].empty(), ErrorCode::Api, "TokenVocab: token {} has an empty piece", i);
            HALO_CHECK(v->pieces_[i].size() < 65535, ErrorCode::Api, "TokenVocab: token {} piece too long", i);
            ids.push_back(static_cast<std::int32_t>(i));
        }
    }
    const auto& P = v->pieces_;
    // std::string compares bytes as unsigned char, which is the trie's byte order.
    std::sort(ids.begin(), ids.end(), [&](std::int32_t a, std::int32_t b) {
        const int c = P[static_cast<std::size_t>(a)].compare(P[static_cast<std::size_t>(b)]);
        return c != 0 ? c < 0 : a < b;
    });
    std::vector<std::uint32_t> path;  // node index per depth-1
    std::vector<std::uint32_t> tok_node;
    const std::string* prev = nullptr;
    for (const auto id : ids) {
        const std::string& p = P[static_cast<std::size_t>(id)];
        std::size_t lcp = 0;
        if (prev != nullptr) {
            while (lcp < p.size() && lcp < prev->size() && p[lcp] == (*prev)[lcp]) ++lcp;
        }
        lcp = std::min(lcp, path.size());
        while (path.size() > lcp) {
            v->subtree_end_[path.back()] = static_cast<std::uint32_t>(v->byte_.size());
            path.pop_back();
        }
        for (std::size_t d = lcp; d < p.size(); ++d) {
            path.push_back(static_cast<std::uint32_t>(v->byte_.size()));
            v->byte_.push_back(static_cast<std::uint8_t>(p[d]));
            v->depth_.push_back(static_cast<std::uint16_t>(d + 1));
            v->subtree_end_.push_back(0);
        }
        tok_node.push_back(path.back());
        v->toks_.push_back(id);
        v->max_depth_ = std::max(v->max_depth_, p.size());
        prev = &p;
    }
    while (!path.empty()) {
        v->subtree_end_[path.back()] = static_cast<std::uint32_t>(v->byte_.size());
        path.pop_back();
    }
    // tok_node is non-decreasing (a token ends at the newest node, or at the same node as
    // an identical previous piece), so toks_ is already grouped by node.
    v->tok_begin_.assign(v->byte_.size() + 1, 0);
    for (const auto nd : tok_node) ++v->tok_begin_[nd + 1];
    std::partial_sum(v->tok_begin_.begin(), v->tok_begin_.end(), v->tok_begin_.begin());
    return v;
}

bool TokenVocab::included(std::int32_t id) const {
    return id >= 0 && static_cast<std::size_t>(id) < pieces_.size() && included_[static_cast<std::size_t>(id)];
}

const std::string& TokenVocab::piece(std::int32_t id) const {
    HALO_CHECK(id >= 0 && static_cast<std::size_t>(id) < pieces_.size(), ErrorCode::Api,
               "TokenVocab: token id {} out of range", id);
    return pieces_[static_cast<std::size_t>(id)];
}

// ---------------------------------------------------------------------------------------
// ByteMatcher
// ---------------------------------------------------------------------------------------

ByteMatcher::ByteMatcher(const Grammar& g) : rt_(std::make_unique<Runtime>(g.data())), state_(rt_->initial()) {}
ByteMatcher::ByteMatcher(ByteMatcher&&) noexcept = default;
ByteMatcher& ByteMatcher::operator=(ByteMatcher&&) noexcept = default;
ByteMatcher::~ByteMatcher() = default;

bool ByteMatcher::accept_bytes(std::string_view bytes) {
    std::int32_t s = state_;
    for (const char c : bytes) {
        s = rt_->step(s, static_cast<std::uint8_t>(c));
        if (s == kDead) return false;
    }
    state_ = s;
    return true;
}
bool ByteMatcher::is_accepting() const { return rt_->accepting(state_); }
bool ByteMatcher::can_continue() const { return rt_->can_continue(state_); }
void ByteMatcher::reset() { state_ = rt_->initial(); }

// ---------------------------------------------------------------------------------------
// TokenMatcher
// ---------------------------------------------------------------------------------------

namespace {
constexpr std::size_t kCacheSlots = 64;
// Interned states before the runtime is rebuilt (each costs a 1 KiB transition row).
constexpr std::size_t kMaxStates = std::size_t{1} << 14;
}  // namespace

struct TokenMatcher::Cache {
    std::unordered_map<std::int32_t, std::size_t> slot_of;
    std::vector<std::vector<std::uint64_t>> slots;
    std::vector<std::int32_t> slot_state;
    std::size_t next = 0;

    void clear() {
        slot_of.clear();
        std::fill(slot_state.begin(), slot_state.end(), kDead);
        next = 0;
    }
};

TokenMatcher::TokenMatcher(const Grammar& g, std::shared_ptr<const TokenVocab> vocab, std::vector<std::int32_t> eos_ids)
    : grammar_(g), vocab_(std::move(vocab)), eos_(std::move(eos_ids)) {
    HALO_CHECK(vocab_ != nullptr, ErrorCode::Api, "TokenMatcher: vocabulary required");
    HALO_CHECK(!eos_.empty(), ErrorCode::Api, "TokenMatcher: at least one EOS token id is required");
    for (const auto e : eos_) {
        HALO_CHECK(e >= 0 && static_cast<std::size_t>(e) < vocab_->vocab_size(), ErrorCode::Api,
                   "TokenMatcher: EOS id {} outside the vocabulary", e);
        HALO_CHECK(!vocab_->included(e), ErrorCode::Api, "TokenMatcher: EOS id {} is a normal text token", e);
    }
    rt_ = std::make_unique<Runtime>(grammar_.data());
    state_ = rt_->initial();
    cache_ = std::make_unique<Cache>();
    cache_->slots.assign(kCacheSlots, {});
    cache_->slot_state.assign(kCacheSlots, kDead);
    dfs_states_.assign(vocab_->max_depth_ + 1, kDead);
}

TokenMatcher::TokenMatcher(TokenMatcher&&) noexcept = default;
TokenMatcher& TokenMatcher::operator=(TokenMatcher&&) noexcept = default;
TokenMatcher::~TokenMatcher() = default;

void TokenMatcher::maybe_compact() {
    if (rt_->num_states() <= kMaxStates) return;
    const auto m = rt_->materialize(state_);
    rt_ = std::make_unique<Runtime>(grammar_.data());
    state_ = rt_->intern(m);
    cache_->clear();
    ++stats_.compactions;
}

void TokenMatcher::compute_trie_mask(std::vector<std::uint64_t>& out) {
    const TokenVocab& v = *vocab_;
    out.assign((v.vocab_size() + 63) / 64, 0);
    if (finished_) return;
    const std::size_t n = v.byte_.size();
    dfs_states_[0] = state_;
    std::size_t i = 0;
    std::uint64_t visited = 0;
    while (i < n) {
        const std::size_t d = v.depth_[i];
        const std::int32_t t = rt_->step(dfs_states_[d - 1], v.byte_[i]);
        if (t == kDead) {
            i = v.subtree_end_[i];
            continue;
        }
        dfs_states_[d] = t;
        for (std::uint32_t k = v.tok_begin_[i]; k < v.tok_begin_[i + 1]; ++k) {
            const auto id = static_cast<std::size_t>(v.toks_[k]);
            out[id >> 6U] |= std::uint64_t{1} << (id & 63U);
        }
        ++visited;
        ++i;
    }
    stats_.trie_nodes_visited += visited;
    if (rt_->accepting(state_)) {
        for (const auto e : eos_) out[static_cast<std::size_t>(e) >> 6U] |= std::uint64_t{1} << (static_cast<unsigned>(e) & 63U);
    }
}

std::span<const std::uint64_t> TokenMatcher::mask() {
    ++stats_.masks;
    if (finished_) {
        scratch_.assign((vocab_->vocab_size() + 63) / 64, 0);
        return scratch_;
    }
    maybe_compact();
    if (const auto it = cache_->slot_of.find(state_); it != cache_->slot_of.end()) {
        ++stats_.cache_hits;
        return cache_->slots[it->second];
    }
    const std::size_t slot = cache_->next;
    cache_->next = (cache_->next + 1) % kCacheSlots;
    if (cache_->slot_state[slot] != kDead) cache_->slot_of.erase(cache_->slot_state[slot]);
    compute_trie_mask(cache_->slots[slot]);
    cache_->slot_state[slot] = state_;
    cache_->slot_of.emplace(state_, slot);
    return cache_->slots[slot];
}

std::span<const std::uint64_t> TokenMatcher::mask_uncached() {
    maybe_compact();
    compute_trie_mask(scratch_);
    return scratch_;
}

std::vector<std::uint64_t> TokenMatcher::mask_naive() {
    const TokenVocab& v = *vocab_;
    std::vector<std::uint64_t> out((v.vocab_size() + 63) / 64, 0);
    if (finished_) return out;
    for (std::size_t id = 0; id < v.vocab_size(); ++id) {
        if (!v.included(static_cast<std::int32_t>(id))) continue;
        std::int32_t s = state_;
        for (const char c : v.piece(static_cast<std::int32_t>(id))) {
            s = rt_->step_uncached(s, static_cast<std::uint8_t>(c));
            if (s == kDead) break;
        }
        if (s != kDead) out[id >> 6U] |= std::uint64_t{1} << (id & 63U);
    }
    if (rt_->accepting(state_)) {
        for (const auto e : eos_) out[static_cast<std::size_t>(e) >> 6U] |= std::uint64_t{1} << (static_cast<unsigned>(e) & 63U);
    }
    return out;
}

bool TokenMatcher::is_allowed(std::int32_t id) {
    if (finished_) return false;
    if (std::find(eos_.begin(), eos_.end(), id) != eos_.end()) return rt_->accepting(state_);
    if (!vocab_->included(id)) return false;
    std::int32_t s = state_;
    for (const char c : vocab_->piece(id)) {
        s = rt_->step(s, static_cast<std::uint8_t>(c));
        if (s == kDead) return false;
    }
    return true;
}

void TokenMatcher::accept_token(std::int32_t id) {
    HALO_CHECK(!finished_, ErrorCode::Api, "structured output: token {} after EOS", id);
    if (std::find(eos_.begin(), eos_.end(), id) != eos_.end()) {
        HALO_CHECK(rt_->accepting(state_), ErrorCode::Api, "structured output: EOS before the JSON is complete");
        finished_ = true;
        return;
    }
    HALO_CHECK(vocab_->included(id), ErrorCode::Api, "structured output: token {} is not allowed (special/unused)", id);
    maybe_compact();
    std::int32_t s = state_;
    for (const char c : vocab_->piece(id)) {
        s = rt_->step(s, static_cast<std::uint8_t>(c));
        HALO_CHECK(s != kDead, ErrorCode::Api, "structured output: token {} is not allowed by the grammar here", id);
    }
    state_ = s;
}

bool TokenMatcher::is_complete() const { return !finished_ && rt_->accepting(state_); }

void TokenMatcher::reset() {
    state_ = rt_->initial();
    finished_ = false;
}

MatcherSnapshot TokenMatcher::snapshot() const {
    auto m = rt_->materialize(state_);
    return MatcherSnapshot{std::move(m.stacks), m.accepting, finished_};
}

void TokenMatcher::restore(const MatcherSnapshot& s) {
    const std::int32_t st = rt_->intern(detail::Materialized{s.stacks, s.accepting});
    HALO_CHECK(st != kDead, ErrorCode::Api, "structured output: invalid snapshot");
    state_ = st;
    finished_ = s.finished;
}

MaskStats TokenMatcher::stats() const {
    MaskStats s = stats_;
    s.states = rt_->num_states();
    return s;
}

}  // namespace halo::sampling
