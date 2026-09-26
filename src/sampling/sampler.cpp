// Sampler chain (TRD §23). Semantics are documented in include/halo/sampling/sampler.h.

#include "halo/sampling/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "halo/core/error.h"
#include "halo/tokenizer/tokenizer.h"

namespace halo::sampling {

namespace {

bool finite(float x) noexcept { return std::isfinite(x); }
bool in01(float x) noexcept { return x >= 0.0f && x <= 1.0f; }  // false for NaN

/// Canonical "better" order: larger logit first, ties to the lower id.
bool by_logit_desc(const chain::Cand& a, const chain::Cand& b) noexcept {
    return a.logit != b.logit ? a.logit > b.logit : a.id < b.id;
}
bool by_id(const chain::Cand& a, const chain::Cand& b) noexcept { return a.id < b.id; }

double max_logit(const std::vector<chain::Cand>& c) {
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& x : c) m = std::max(m, x.logit);
    return m;
}

bool mask_has(std::span<const std::uint64_t> mask, std::int32_t id) noexcept {
    // Ids past the TokenVocab (padded logits rows) are never allowed; bits past its
    // vocab_size inside the last word are zero.
    const auto u = static_cast<std::uint32_t>(id);
    if ((u >> 6U) >= mask.size()) return false;
    return ((mask[u >> 6U] >> (u & 63U)) & 1U) != 0;
}

}  // namespace

// ---------------------------------------------------------------------------------------
// Parameter helpers
// ---------------------------------------------------------------------------------------

void validate(const SamplingParams& p) {
    HALO_CHECK(finite(p.temperature) && p.temperature >= 0.0f, ErrorCode::Api,
               "sampling: temperature must be finite and >= 0 (0 = greedy), got {}", p.temperature);
    HALO_CHECK(p.top_k >= 0, ErrorCode::Api, "sampling: top_k must be >= 0 (0 = disabled), got {}", p.top_k);
    HALO_CHECK(in01(p.top_p), ErrorCode::Api, "sampling: top_p must be in [0, 1]");
    HALO_CHECK(in01(p.min_p), ErrorCode::Api, "sampling: min_p must be in [0, 1]");
    HALO_CHECK(in01(p.typical_p), ErrorCode::Api, "sampling: typical_p must be in [0, 1]");
    HALO_CHECK(finite(p.repetition_penalty) && p.repetition_penalty > 0.0f, ErrorCode::Api,
               "sampling: repetition_penalty must be finite and > 0");
    HALO_CHECK(finite(p.presence_penalty), ErrorCode::Api, "sampling: presence_penalty must be finite");
    HALO_CHECK(finite(p.frequency_penalty), ErrorCode::Api, "sampling: frequency_penalty must be finite");
    HALO_CHECK(p.penalty_last_n >= -1, ErrorCode::Api, "sampling: penalty_last_n must be >= -1 (-1 = all)");
    HALO_CHECK(!(p.json_schema && p.json_object), ErrorCode::Api,
               "sampling: json_schema and json_object are mutually exclusive");
}

bool penalties_active(const SamplingParams& p) noexcept {
    return p.penalty_last_n != 0 &&
           (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f || p.frequency_penalty != 0.0f);
}

bool is_structured(const SamplingParams& p) noexcept { return p.json_schema.has_value() || p.json_object; }

bool exact_prefilter_ok(const SamplingParams& p, std::size_t k) noexcept {
    if (k == 0 || penalties_active(p) || is_structured(p)) return false;
    if (p.greedy()) return true;
    return p.top_k >= 1 && static_cast<std::size_t>(p.top_k) <= k;
}

// ---------------------------------------------------------------------------------------
// Chain steps
// ---------------------------------------------------------------------------------------

namespace chain {

std::vector<std::pair<std::int32_t, std::int32_t>> window_counts(std::span<const std::int32_t> history, int last_n) {
    std::vector<std::pair<std::int32_t, std::int32_t>> out;
    if (last_n == 0 || history.empty()) return out;
    const std::size_t n = last_n < 0 ? history.size() : std::min(history.size(), static_cast<std::size_t>(last_n));
    std::vector<std::int32_t> w(history.end() - static_cast<std::ptrdiff_t>(n), history.end());
    std::sort(w.begin(), w.end());
    for (std::size_t i = 0; i < w.size();) {
        std::size_t j = i;
        while (j < w.size() && w[j] == w[i]) ++j;
        out.emplace_back(w[i], static_cast<std::int32_t>(j - i));
        i = j;
    }
    return out;
}

void apply_penalties(std::vector<Cand>& cands, const SamplingParams& p,
                     std::span<const std::pair<std::int32_t, std::int32_t>> counts) {
    if (!penalties_active(p)) return;
    const auto rep = static_cast<double>(p.repetition_penalty);
    const auto freq = static_cast<double>(p.frequency_penalty);
    const auto pres = static_cast<double>(p.presence_penalty);
    for (const auto& [id, c] : counts) {
        const auto it = std::lower_bound(cands.begin(), cands.end(), id,
                                         [](const Cand& x, std::int32_t v) { return x.id < v; });
        if (it == cands.end() || it->id != id) continue;
        double& l = it->logit;
        l = l <= 0.0 ? l * rep : l / rep;
        l -= static_cast<double>(c) * freq + pres;
    }
}

void apply_temperature(std::vector<Cand>& cands, float temperature) {
    const auto t = static_cast<double>(temperature);
    for (auto& c : cands) c.logit /= t;
}

void apply_top_k(std::vector<Cand>& cands, int k) {
    if (k <= 0 || static_cast<std::size_t>(k) >= cands.size()) return;
    const auto kk = static_cast<std::ptrdiff_t>(k);
    std::nth_element(cands.begin(), cands.begin() + kk - 1, cands.end(), by_logit_desc);
    cands.resize(static_cast<std::size_t>(k));
}

std::vector<double> probabilities(const std::vector<Cand>& cands) {
    std::vector<double> p(cands.size());
    if (cands.empty()) return p;
    const double m = max_logit(cands);
    double s = 0.0;
    for (std::size_t i = 0; i < cands.size(); ++i) {
        p[i] = std::exp(cands[i].logit - m);
        s += p[i];
    }
    for (auto& x : p) x /= s;
    return p;
}

void apply_typical(std::vector<Cand>& cands, float p) {
    if (p >= 1.0f || cands.size() <= 1) return;
    std::sort(cands.begin(), cands.end(), by_logit_desc);  // canonical order for the sums
    const double m = cands.front().logit;
    double s = 0.0;
    for (const auto& c : cands) s += std::exp(c.logit - m);
    const double log_s = std::log(s);
    std::vector<double> logp(cands.size()), prob(cands.size());
    double entropy = 0.0;
    for (std::size_t i = 0; i < cands.size(); ++i) {
        logp[i] = cands[i].logit - m - log_s;
        prob[i] = std::exp(logp[i]);
        entropy -= prob[i] * logp[i];
    }
    std::vector<std::size_t> order(cands.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::vector<double> score(cands.size());
    for (std::size_t i = 0; i < cands.size(); ++i) score[i] = std::fabs(-logp[i] - entropy);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return score[a] != score[b] ? score[a] < score[b] : cands[a].id < cands[b].id;
    });
    const auto target = static_cast<double>(p);
    double cum = 0.0;
    std::size_t keep = order.size();
    for (std::size_t i = 0; i < order.size(); ++i) {
        cum += prob[order[i]];
        if (cum > target) {
            keep = i + 1;
            break;
        }
    }
    std::vector<Cand> out;
    out.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) out.push_back(cands[order[i]]);
    cands = std::move(out);
}

void apply_top_p(std::vector<Cand>& cands, float p) {
    if (p >= 1.0f || cands.size() <= 1) return;
    std::sort(cands.begin(), cands.end(), by_logit_desc);
    const auto prob = probabilities(cands);
    const auto target = static_cast<double>(p);
    double cum = 0.0;
    std::size_t keep = cands.size();
    for (std::size_t i = 0; i < cands.size(); ++i) {
        cum += prob[i];
        if (cum >= target) {
            keep = i + 1;
            break;
        }
    }
    cands.resize(keep);
}

void apply_min_p(std::vector<Cand>& cands, float p) {
    if (p <= 0.0f || cands.empty()) return;
    const double threshold = max_logit(cands) + std::log(static_cast<double>(p));
    std::erase_if(cands, [&](const Cand& c) { return c.logit < threshold; });
}

std::int32_t greedy(const std::vector<Cand>& cands) {
    HALO_CHECK(!cands.empty(), ErrorCode::Api, "sampling: no candidate to choose from");
    return std::min_element(cands.begin(), cands.end(), by_logit_desc)->id;
}

std::int32_t draw(std::vector<Cand>& cands, double u) {
    HALO_CHECK(!cands.empty(), ErrorCode::Api, "sampling: no candidate to choose from");
    std::sort(cands.begin(), cands.end(), by_id);
    const double m = max_logit(cands);
    double total = 0.0;
    for (const auto& c : cands) total += std::exp(c.logit - m);
    const double target = u * total;
    double cum = 0.0;
    std::int32_t last_positive = cands.front().id;
    for (const auto& c : cands) {
        const double w = std::exp(c.logit - m);
        if (w <= 0.0) continue;
        cum += w;
        last_positive = c.id;
        if (cum > target) return c.id;
    }
    return last_positive;  // u * total rounded up to the full sum
}

double uniform01(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11U) * 0x1.0p-53;
}

}  // namespace chain

// ---------------------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------------------

Sampler::Sampler(const SamplingParams& params, std::size_t vocab_size, std::size_t emit_vocab_size,
                 std::uint64_t seed, std::unique_ptr<TokenMatcher> matcher)
    : params_(params),
      vocab_size_(vocab_size),
      emit_vocab_size_(emit_vocab_size),
      seed_(seed),
      rng_(seed),
      matcher_(std::move(matcher)) {}

Sampler::Sampler(Sampler&&) noexcept = default;
Sampler& Sampler::operator=(Sampler&&) noexcept = default;
Sampler::~Sampler() = default;

Sampler Sampler::create(const SamplingParams& params, const tokenizer::Tokenizer* tokenizer, SamplerConfig cfg) {
    validate(params);
    std::size_t vocab = cfg.vocab_size;
    if (vocab == 0) {
        HALO_CHECK(tokenizer != nullptr, ErrorCode::Api, "sampling: vocab_size or a tokenizer is required");
        vocab = tokenizer->vocab_size();
    }
    HALO_CHECK(vocab > 0 && vocab < (std::size_t{1} << 31), ErrorCode::Api, "sampling: invalid vocab_size {}", vocab);
    // The logits row may be longer than the tokenizer vocabulary: the GGUF head is padded to
    // 248,320 rows while tokenizer.json has 248,077 entries. Ids past the tokenizer are never
    // allowed under structured output.
    HALO_CHECK(tokenizer == nullptr || tokenizer->vocab_size() <= vocab, ErrorCode::Api,
               "sampling: vocab_size {} is smaller than the tokenizer's {}", vocab, tokenizer ? tokenizer->vocab_size() : 0);

    std::unique_ptr<TokenMatcher> matcher;
    if (is_structured(params)) {
        const Grammar g = params.json_schema ? Grammar::from_json_schema(*params.json_schema, cfg.schema)
                                             : Grammar::any_json_object(cfg.schema);
        std::shared_ptr<const TokenVocab> tv = std::move(cfg.vocab);
        if (!tv) {
            HALO_CHECK(tokenizer != nullptr, ErrorCode::Api,
                       "sampling: structured output needs a tokenizer or a prebuilt TokenVocab");
            tv = TokenVocab::build(*tokenizer);
        }
        HALO_CHECK(tv->vocab_size() <= vocab, ErrorCode::Api, "sampling: TokenVocab has {} tokens, more than {}",
                   tv->vocab_size(), vocab);
        std::vector<std::int32_t> eos = std::move(cfg.eos_ids);
        if (eos.empty() && tokenizer != nullptr && tokenizer->eos()) eos.push_back(*tokenizer->eos());
        HALO_CHECK(!eos.empty(), ErrorCode::Api, "sampling: structured output needs at least one EOS id");
        matcher = std::make_unique<TokenMatcher>(g, std::move(tv), std::move(eos));
    }

    std::uint64_t seed = 0;
    if (params.seed) {
        seed = *params.seed;
    } else {
        std::random_device rd;
        seed = (std::uint64_t{rd()} << 32U) ^ std::uint64_t{rd()};
    }
    // M2: unstructured sampling must not emit padded LM-head rows past the tokenizer's real
    // vocabulary (structured output already excludes them via the TokenVocab mask).
    std::size_t emit_vocab = cfg.emit_vocab_size;
    if (emit_vocab == 0) emit_vocab = tokenizer != nullptr ? tokenizer->vocab_size() : vocab;
    HALO_CHECK(emit_vocab <= vocab, ErrorCode::Api, "sampling: emit_vocab_size {} exceeds vocab_size {}", emit_vocab,
               vocab);
    return Sampler(params, vocab, emit_vocab, seed, std::move(matcher));
}

std::int32_t Sampler::run_chain(std::span<const std::int32_t> history) {
    // work_ holds the (masked) candidates sorted by id.
    for (const auto t : history) {
        HALO_CHECK(t >= 0 && static_cast<std::size_t>(t) < vocab_size_, ErrorCode::Api,
                   "sampling: history token {} outside the vocabulary", t);
    }
    HALO_CHECK(!work_.empty(), ErrorCode::Api,
               "sampling: no token is possible (every logit is -inf or disallowed by the grammar)");
    const auto counts = chain::window_counts(history, params_.penalty_last_n);
    chain::apply_penalties(work_, params_, counts);  // finite in, finite out (double)
    if (params_.greedy()) return chain::greedy(work_);
    chain::apply_temperature(work_, params_.temperature);
    chain::apply_top_k(work_, params_.top_k);
    chain::apply_typical(work_, params_.typical_p);
    chain::apply_top_p(work_, params_.top_p);
    chain::apply_min_p(work_, params_.min_p);
    return chain::draw(work_, chain::uniform01(rng_));
}

std::int32_t Sampler::sample(std::span<const float> logits, std::span<const std::int32_t> history) {
    HALO_CHECK(logits.size() == vocab_size_, ErrorCode::Api, "sampling: logits row has {} entries, expected {}",
               logits.size(), vocab_size_);
    HALO_CHECK(matcher_ == nullptr || !matcher_->is_finished(), ErrorCode::Api,
               "sampling: structured output already finished (EOS accepted)");
    std::span<const std::uint64_t> mask;
    if (matcher_) mask = matcher_->mask();
    work_.clear();
    work_.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float l = logits[i];
        HALO_CHECK(!std::isnan(l) && l != std::numeric_limits<float>::infinity(), ErrorCode::Kernel,
                   "sampling: logit {} is {} (NaN/+Inf logits are rejected)", i, l);
        if (l == -std::numeric_limits<float>::infinity()) continue;
        const auto id = static_cast<std::int32_t>(i);
        if (matcher_ && !mask_has(mask, id)) continue;
        if (!matcher_ && i >= emit_vocab_size_) continue;  // M2: exclude padded LM-head rows
        work_.push_back(chain::Cand{id, static_cast<double>(l)});
    }
    return run_chain(history);
}

std::optional<std::int32_t> Sampler::try_sample(const CandidateSet& c, std::span<const std::int32_t> history) {
    HALO_CHECK(c.ids.size() == c.logits.size(), ErrorCode::Api, "sampling: {} candidate ids but {} logits",
               c.ids.size(), c.logits.size());
    HALO_CHECK(!c.ids.empty(), ErrorCode::Api, "sampling: empty candidate set");
    HALO_CHECK(matcher_ == nullptr || !matcher_->is_finished(), ErrorCode::Api,
               "sampling: structured output already finished (EOS accepted)");
    {
        std::vector<std::int32_t> ids(c.ids.begin(), c.ids.end());
        std::sort(ids.begin(), ids.end());
        const auto dup = std::adjacent_find(ids.begin(), ids.end());
        HALO_CHECK(dup == ids.end(), ErrorCode::Api, "sampling: duplicate candidate id {}", *dup);
    }
    std::span<const std::uint64_t> mask;
    if (matcher_) mask = matcher_->mask();
    work_.clear();
    bool any_finite = false;
    for (std::size_t i = 0; i < c.ids.size(); ++i) {
        const std::int32_t id = c.ids[i];
        const float l = c.logits[i];
        HALO_CHECK(id >= 0 && static_cast<std::size_t>(id) < vocab_size_, ErrorCode::Api,
                   "sampling: candidate id {} outside the vocabulary", id);
        HALO_CHECK(!std::isnan(l) && l != std::numeric_limits<float>::infinity(), ErrorCode::Kernel,
                   "sampling: candidate {} logit is {} (NaN/+Inf logits are rejected)", id, l);
        if (l == -std::numeric_limits<float>::infinity()) continue;
        any_finite = true;
        if (matcher_ && !mask_has(mask, id)) continue;
        work_.push_back(chain::Cand{id, static_cast<double>(l)});
    }
    std::sort(work_.begin(), work_.end(), by_id);
    if (matcher_ && work_.empty() && any_finite) return std::nullopt;
    return run_chain(history);
}

std::int32_t Sampler::sample(const CandidateSet& c, std::span<const std::int32_t> history) {
    const auto t = try_sample(c, history);
    HALO_CHECK(t.has_value(), ErrorCode::Api,
               "sampling: structured output allows none of the {} prefiltered candidates (sample the full row)",
               c.ids.size());
    return *t;
}

void Sampler::accept(std::int32_t token) {
    if (matcher_) matcher_->accept_token(token);
}

void Sampler::reset() {
    rng_.seed(seed_);
    if (matcher_) matcher_->reset();
}

bool Sampler::grammar_complete() const { return matcher_ != nullptr && matcher_->is_complete(); }
bool Sampler::grammar_finished() const noexcept { return matcher_ != nullptr && matcher_->is_finished(); }

}  // namespace halo::sampling
