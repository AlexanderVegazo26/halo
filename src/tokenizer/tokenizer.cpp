#include "halo/tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <deque>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "halo/core/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace halo::tokenizer {
namespace {

using unicode::CpOff;

constexpr std::size_t kMaxVocab = std::size_t{1} << 24;       // sanity bound for untrusted input
constexpr std::size_t kMaxHfFileBytes = std::size_t{256} << 20;  // 256 MiB
constexpr std::size_t kMaxCachedWordBytes = 64;
constexpr std::size_t kMaxTextBytes = std::numeric_limits<std::uint32_t>::max() - 1;

// The only pre-tokenizer regex this implementation reproduces (HF tokenizer.json of the
// Qwen3.x family, == llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 modulo the (?i) spelling).
constexpr std::string_view kQwen35Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// ---- GPT-2 byte <-> unicode mapping ------------------------------------------------------

struct ByteMap {
    std::array<char32_t, 256> byte_to_cp{};
    std::unordered_map<char32_t, std::uint8_t> cp_to_byte;
};

const ByteMap& byte_map() {
    static const ByteMap m = [] {
        ByteMap bm;
        std::array<bool, 256> direct{};
        for (int b = '!'; b <= '~'; ++b) direct[static_cast<std::size_t>(b)] = true;
        for (int b = 0xA1; b <= 0xAC; ++b) direct[static_cast<std::size_t>(b)] = true;
        for (int b = 0xAE; b <= 0xFF; ++b) direct[static_cast<std::size_t>(b)] = true;
        char32_t next = 256;
        for (std::size_t b = 0; b < 256; ++b) {
            bm.byte_to_cp[b] = direct[b] ? static_cast<char32_t>(b) : next++;
            bm.cp_to_byte.emplace(bm.byte_to_cp[b], static_cast<std::uint8_t>(b));
        }
        return bm;
    }();
    return m;
}

std::string byte_to_mapped(std::uint8_t b) {
    std::string s;
    unicode::append_utf8(s, byte_map().byte_to_cp[b]);
    return s;
}

// HF ByteLevel decoder rule: map every char back to its byte; if any char is outside the
// mapping, the token's own UTF-8 bytes are used unchanged.
std::string mapped_to_bytes(std::string_view mapped) {
    const auto& m = byte_map();
    std::string out;
    out.reserve(mapped.size());
    std::size_t i = 0;
    while (i < mapped.size()) {
        const auto st = unicode::decode_step(mapped, i);
        if (st.status != unicode::Utf8Step::Status::Ok) return std::string(mapped);
        const auto it = m.cp_to_byte.find(st.cp);
        if (it == m.cp_to_byte.end()) return std::string(mapped);
        out.push_back(static_cast<char>(it->second));
        i += st.len;
    }
    return out;
}

// ---- merge table: open addressing keyed by (left id, right id) ---------------------------

class MergeMap {
public:
    struct Value {
        std::uint32_t rank;
        std::int32_t new_id;
    };

    void reserve(std::size_t n) {
        std::size_t cap = 16;
        while (cap < n * 2) cap <<= 1;
        keys_.assign(cap, kEmpty);
        vals_.assign(cap, Value{0, 0});
        mask_ = cap - 1;
        shift_ = 64 - static_cast<unsigned>(std::countr_zero(cap));
    }

    // Later entries overwrite earlier ones, as HF's HashMap collect does for duplicates.
    void insert(std::int32_t a, std::int32_t b, Value v) {
        const std::uint64_t k = key(a, b);
        for (std::size_t i = slot(k);; i = (i + 1) & mask_) {
            if (keys_[i] == kEmpty || keys_[i] == k) {
                keys_[i] = k;
                vals_[i] = v;
                return;
            }
        }
    }

    [[nodiscard]] const Value* find(std::int32_t a, std::int32_t b) const noexcept {
        const std::uint64_t k = key(a, b);
        for (std::size_t i = slot(k);; i = (i + 1) & mask_) {
            if (keys_[i] == k) return &vals_[i];
            if (keys_[i] == kEmpty) return nullptr;
        }
    }

private:
    static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
    static std::uint64_t key(std::int32_t a, std::int32_t b) noexcept {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
    }
    [[nodiscard]] std::size_t slot(std::uint64_t k) const noexcept {
        return static_cast<std::size_t>((k * 0x9E3779B97F4A7C15ull) >> shift_) & mask_;
    }
    std::vector<std::uint64_t> keys_;
    std::vector<Value> vals_;
    std::size_t mask_ = 0;
    unsigned shift_ = 0;
};

struct AddedToken {
    std::string content;
    std::int32_t id;
    bool special;  // Control
};

// Per-call scratch so encode() is reentrant and allocation-light.
struct Scratch {
    struct Sym {
        std::int32_t id;
        std::int32_t prev;
        std::int32_t next;
        std::uint32_t len;
    };
    struct Merge {
        std::uint32_t rank;
        std::uint32_t pos;
        std::int32_t new_id;
    };
    std::vector<Sym> syms;
    std::vector<Merge> heap;
    std::vector<char32_t> cps;
    std::vector<std::uint32_t> starts;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pieces;
    std::unordered_map<std::string_view, std::pair<std::uint32_t, std::uint32_t>> cache;
    std::vector<std::int32_t> arena;
    std::deque<std::string> owned;  // normalized segments referenced by `cache` keys
};

}  // namespace

struct Tokenizer::Impl {
    std::vector<std::string> pieces;  // raw bytes per id
    std::vector<TokenType> types;
    std::unordered_map<std::string, std::int32_t> by_piece;  // raw bytes -> id
    std::array<std::int32_t, 256> byte_to_id{};
    MergeMap merges;
    std::vector<AddedToken> added;
    std::array<std::vector<std::uint32_t>, 256> added_by_byte;  // indices, longest first
    std::optional<std::int32_t> bos, eos, pad;
    Normalizer normalizer = Normalizer::Nfc;

    static std::unique_ptr<Impl> build(VocabSpec spec);

    void check_id(std::int32_t id) const {
        HALO_CHECK(id >= 0 && static_cast<std::size_t>(id) < pieces.size(), ErrorCode::Api,
                   "token id {} outside vocabulary of {}", id, pieces.size());
    }

    // Leftmost-longest added-token match at or after `from`; returns {pos, index} or
    // {npos, 0}.
    [[nodiscard]] std::pair<std::size_t, std::size_t> next_added(std::string_view s, std::size_t from,
                                                                 bool parse_special) const {
        for (std::size_t p = from; p < s.size(); ++p) {
            const auto& cands = added_by_byte[static_cast<unsigned char>(s[p])];
            for (std::uint32_t idx : cands) {
                const AddedToken& t = added[idx];
                if (t.special && !parse_special) continue;
                if (s.compare(p, t.content.size(), t.content) == 0) return {p, idx};
            }
        }
        return {std::string_view::npos, 0};
    }

    void bpe(std::string_view word, Scratch& sc, std::vector<std::int32_t>& out) const;

    template <typename OffsetMap>
    void encode_normalized(std::string_view s, Scratch& sc, std::vector<std::int32_t>& ids,
                           std::vector<std::size_t>* offs, const OffsetMap& to_input) const;

    void encode(std::string_view text, bool parse_special, std::vector<std::int32_t>& ids,
                std::vector<std::size_t>* offs) const;
};

std::unique_ptr<Tokenizer::Impl> Tokenizer::Impl::build(VocabSpec spec) {
    auto impl = std::make_unique<Impl>();
    const std::size_t n = spec.tokens.size();
    HALO_CHECK(n > 0 && n <= kMaxVocab, ErrorCode::Model, "vocabulary size {} out of range", n);
    HALO_CHECK(spec.types.empty() || spec.types.size() == n, ErrorCode::Model,
               "token_type count {} != token count {}", spec.types.size(), n);
    if (spec.types.empty()) spec.types.assign(n, TokenType::Normal);
    HALO_CHECK(spec.merges.size() <= kMaxVocab, ErrorCode::Model, "merge count {} out of range",
               spec.merges.size());

    impl->normalizer = spec.normalizer;
    impl->types = std::move(spec.types);
    impl->pieces.resize(n);
    std::unordered_map<std::string_view, std::int32_t> by_mapped;  // Normal tokens only
    by_mapped.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto id = static_cast<std::int32_t>(i);
        const TokenType t = impl->types[i];
        switch (t) {
            case TokenType::Normal:
                if (!by_mapped.emplace(spec.tokens[i], id).second) {
                    throw_error(ErrorCode::Model, "duplicate token string at id {}", id);
                }
                impl->pieces[i] = mapped_to_bytes(spec.tokens[i]);
                break;
            case TokenType::Control:
            case TokenType::UserDefined:
                HALO_CHECK(!spec.tokens[i].empty(), ErrorCode::Model, "empty added token at id {}", id);
                impl->pieces[i] = mapped_to_bytes(spec.tokens[i]);
                impl->added.push_back({spec.tokens[i], id, t == TokenType::Control});
                break;
            case TokenType::Unused:
                break;
            case TokenType::Unknown:
            case TokenType::Byte:
                throw_error(ErrorCode::Unsupported, "token type {} (id {}) is not supported by byte-level BPE",
                            static_cast<int>(t), id);
            default:
                throw_error(ErrorCode::Model, "invalid token type {} at id {}", static_cast<int>(t), id);
        }
    }
    for (std::size_t b = 0; b < 256; ++b) {
        const auto it = by_mapped.find(byte_to_mapped(static_cast<std::uint8_t>(b)));
        HALO_CHECK(it != by_mapped.end(), ErrorCode::Model, "vocabulary lacks the byte token for 0x{:02X}", b);
        impl->byte_to_id[b] = it->second;
    }
    impl->merges.reserve(spec.merges.size());
    std::string joined;
    for (std::size_t r = 0; r < spec.merges.size(); ++r) {
        const std::string_view m = spec.merges[r];
        const std::size_t sp = m.find(' ');
        HALO_CHECK(sp != std::string_view::npos && sp > 0 && sp + 1 < m.size() &&
                       m.find(' ', sp + 1) == std::string_view::npos,
                   ErrorCode::Model, "merge {} is not of the form 'a b'", r);
        const auto a = by_mapped.find(m.substr(0, sp));
        const auto b = by_mapped.find(m.substr(sp + 1));
        joined.assign(m.substr(0, sp)).append(m.substr(sp + 1));
        const auto c = by_mapped.find(joined);
        HALO_CHECK(a != by_mapped.end() && b != by_mapped.end() && c != by_mapped.end(), ErrorCode::Model,
                   "merge {} references a token missing from the vocabulary", r);
        impl->merges.insert(a->second, b->second, {static_cast<std::uint32_t>(r), c->second});
    }
    for (std::size_t i = 0; i < impl->added.size(); ++i) {
        const auto& t = impl->added[i];
        for (std::size_t j = 0; j < i; ++j) {
            HALO_CHECK(impl->added[j].content != t.content, ErrorCode::Model,
                       "added token '{}' appears twice (ids {} and {})", t.content, impl->added[j].id, t.id);
        }
        impl->added_by_byte[static_cast<unsigned char>(t.content[0])].push_back(static_cast<std::uint32_t>(i));
    }
    for (auto& v : impl->added_by_byte) {
        std::stable_sort(v.begin(), v.end(), [&](std::uint32_t x, std::uint32_t y) {
            return impl->added[x].content.size() > impl->added[y].content.size();
        });
    }
    impl->by_piece.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (impl->types[i] != TokenType::Unused) impl->by_piece.emplace(impl->pieces[i], static_cast<std::int32_t>(i));
    }
    const auto check_special = [&](const std::optional<std::int32_t>& id, std::string_view what) {
        HALO_CHECK(!id || (*id >= 0 && static_cast<std::size_t>(*id) < n), ErrorCode::Model,
                   "{} token id {} outside vocabulary of {}", what, id.value_or(-1), n);
    };
    check_special(spec.bos, "bos");
    check_special(spec.eos, "eos");
    check_special(spec.pad, "pad");
    impl->bos = spec.bos;
    impl->eos = spec.eos;
    impl->pad = spec.pad;
    return impl;
}

void Tokenizer::Impl::bpe(std::string_view word, Scratch& sc, std::vector<std::int32_t>& out) const {
    const std::size_t n = word.size();
    if (n == 1) {
        out.push_back(byte_to_id[static_cast<unsigned char>(word[0])]);
        return;
    }
    auto& syms = sc.syms;
    auto& heap = sc.heap;
    syms.clear();
    heap.clear();
    for (std::size_t i = 0; i < n; ++i) {
        syms.push_back({byte_to_id[static_cast<unsigned char>(word[i])], static_cast<std::int32_t>(i) - 1,
                        i + 1 < n ? static_cast<std::int32_t>(i + 1) : -1, 1});
    }
    // Min-heap on (rank, pos): HF's Merge ordering.
    const auto later = [](const Scratch::Merge& a, const Scratch::Merge& b) {
        return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
    };
    const auto push = [&](std::size_t pos, std::int32_t left, std::int32_t right) {
        if (const MergeMap::Value* v = merges.find(left, right)) {
            heap.push_back({v->rank, static_cast<std::uint32_t>(pos), v->new_id});
            std::push_heap(heap.begin(), heap.end(), later);
        }
    };
    for (std::size_t i = 0; i + 1 < n; ++i) push(i, syms[i].id, syms[i + 1].id);
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), later);
        const Scratch::Merge top = heap.back();
        heap.pop_back();
        Scratch::Sym& cur = syms[top.pos];
        if (cur.len == 0 || cur.next < 0) continue;
        const auto next_pos = static_cast<std::size_t>(cur.next);
        const Scratch::Sym right = syms[next_pos];
        const MergeMap::Value* v = merges.find(cur.id, right.id);
        if (v == nullptr || v->new_id != top.new_id) continue;  // stale entry
        cur.id = top.new_id;
        cur.len += right.len;
        cur.next = right.next;
        syms[next_pos].len = 0;
        if (right.next >= 0) syms[static_cast<std::size_t>(right.next)].prev = static_cast<std::int32_t>(top.pos);
        if (cur.prev >= 0) push(static_cast<std::size_t>(cur.prev), syms[static_cast<std::size_t>(cur.prev)].id, cur.id);
        if (cur.next >= 0) push(top.pos, cur.id, syms[static_cast<std::size_t>(cur.next)].id);
    }
    for (std::int32_t i = 0; i >= 0; i = syms[static_cast<std::size_t>(i)].next) {
        out.push_back(syms[static_cast<std::size_t>(i)].id);
    }
}

template <typename OffsetMap>
void Tokenizer::Impl::encode_normalized(std::string_view s, Scratch& sc, std::vector<std::int32_t>& ids,
                                        std::vector<std::size_t>* offs, const OffsetMap& to_input) const {
    auto& cps = sc.cps;
    auto& starts = sc.starts;
    cps.clear();
    starts.clear();
    for (std::size_t i = 0; i < s.size();) {
        const auto st = unicode::decode_step(s, i);  // s is valid UTF-8 here
        cps.push_back(st.cp);
        starts.push_back(static_cast<std::uint32_t>(i));
        i += st.len;
    }
    starts.push_back(static_cast<std::uint32_t>(s.size()));
    sc.pieces.clear();
    split_qwen35(cps, sc.pieces);
    for (const auto& [a, b] : sc.pieces) {
        const std::size_t begin = starts[a];
        const std::string_view word = s.substr(begin, starts[b] - begin);
        const std::size_t first = ids.size();
        if (word.size() > 1 && word.size() <= kMaxCachedWordBytes) {
            const auto it = sc.cache.find(word);
            if (it != sc.cache.end()) {
                const auto [off, len] = it->second;
                ids.insert(ids.end(), sc.arena.begin() + off, sc.arena.begin() + off + len);
            } else {
                bpe(word, sc, ids);
                const auto off = static_cast<std::uint32_t>(sc.arena.size());
                sc.arena.insert(sc.arena.end(), ids.begin() + static_cast<std::ptrdiff_t>(first), ids.end());
                sc.cache.emplace(word, std::pair{off, static_cast<std::uint32_t>(ids.size() - first)});
            }
        } else {
            bpe(word, sc, ids);
        }
        if (offs != nullptr) {
            std::size_t local = begin;
            for (std::size_t k = first; k < ids.size(); ++k) {
                offs->push_back(to_input(local));
                local += pieces[static_cast<std::size_t>(ids[k])].size();
            }
        }
    }
}

void Tokenizer::Impl::encode(std::string_view text, bool parse_special, std::vector<std::int32_t>& ids,
                             std::vector<std::size_t>* offs) const {
    HALO_CHECK(text.size() <= kMaxTextBytes, ErrorCode::Api, "text of {} bytes is too large to tokenize",
               text.size());
    // 1. Invalid UTF-8 -> U+FFFD, remembering where each output byte came from.
    std::string sanitized;
    std::vector<std::uint32_t> smap;
    std::string_view work = text;
    if (!unicode::is_valid_utf8(text)) {
        sanitized.reserve(text.size() + 8);
        for (std::size_t i = 0; i < text.size();) {
            const auto st = unicode::decode_step(text, i);
            const std::size_t before = sanitized.size();
            if (st.status == unicode::Utf8Step::Status::Ok) {
                sanitized.append(text.substr(i, st.len));
            } else {
                unicode::append_utf8(sanitized, unicode::kReplacement);
            }
            smap.insert(smap.end(), sanitized.size() - before, static_cast<std::uint32_t>(i));
            i += st.len;
        }
        smap.push_back(static_cast<std::uint32_t>(text.size()));
        work = sanitized;
    }
    const auto input_off = [&](std::size_t w) -> std::size_t { return smap.empty() ? w : smap[w]; };

    Scratch sc;
    const auto do_segment = [&](std::size_t seg_begin, std::size_t seg_end) {
        if (seg_begin == seg_end) return;
        const std::string_view seg = work.substr(seg_begin, seg_end - seg_begin);
        if (normalizer == Normalizer::None || unicode::nfc_quick_check(seg)) {
            encode_normalized(seg, sc, ids, offs, [&](std::size_t local) { return input_off(seg_begin + local); });
            return;
        }
        std::vector<CpOff> cps;
        cps.reserve(seg.size());
        for (std::size_t i = 0; i < seg.size();) {
            const auto st = unicode::decode_step(seg, i);
            cps.push_back({st.cp, static_cast<std::uint32_t>(i)});
            i += st.len;
        }
        unicode::nfc(cps);
        std::string& norm = sc.owned.emplace_back();
        std::vector<std::uint32_t> nmap;  // normalized byte -> seg byte
        norm.reserve(seg.size());
        for (const CpOff& c : cps) {
            const std::size_t before = norm.size();
            unicode::append_utf8(norm, c.cp);
            nmap.insert(nmap.end(), norm.size() - before, c.off);
        }
        nmap.push_back(static_cast<std::uint32_t>(seg.size()));
        if (norm == seg) {  // already NFC (e.g. ordered marks): keep exact offsets
            sc.owned.pop_back();
            encode_normalized(seg, sc, ids, offs, [&](std::size_t local) { return input_off(seg_begin + local); });
            return;
        }
        encode_normalized(norm, sc, ids, offs,
                          [&](std::size_t local) { return input_off(seg_begin + nmap[local]); });
    };

    // 2. Added tokens first; BPE only runs between them.
    std::size_t pos = 0;
    while (pos < work.size()) {
        const auto [m, idx] = next_added(work, pos, parse_special);
        if (m == std::string_view::npos) {
            do_segment(pos, work.size());
            break;
        }
        do_segment(pos, m);
        ids.push_back(added[idx].id);
        if (offs != nullptr) offs->push_back(input_off(m));
        pos = m + added[idx].content.size();
    }
}

// ---- Tokenizer ---------------------------------------------------------------------------

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Tokenizer Tokenizer::from_spec(VocabSpec spec) { return Tokenizer(Impl::build(std::move(spec))); }

namespace {

using Json = nlohmann::json;

void require(bool cond, std::string_view what) {
    if (!cond) throw_error(ErrorCode::Unsupported, "tokenizer.json: {}", what);
}

bool is_null_or(const Json& j, const char* key, const Json& expected) {
    const auto it = j.find(key);
    return it == j.end() || it->is_null() || *it == expected;
}

void check_pre_tokenizer(const Json& pt) {
    require(pt.is_object() && pt.value("type", "") == "Sequence", "pre_tokenizer must be a Sequence");
    const auto& seq = pt.at("pretokenizers");
    require(seq.is_array() && seq.size() == 2, "pre_tokenizer Sequence must be [Split, ByteLevel]");
    const auto& split = seq[0];
    require(split.value("type", "") == "Split", "first pre_tokenizer must be Split");
    require(split.contains("pattern") && split["pattern"].is_object() && split["pattern"].contains("Regex") &&
                split["pattern"]["Regex"].is_string() &&
                split["pattern"]["Regex"].get<std::string>() == kQwen35Regex,
            "Split pattern is not the qwen35 regex");
    require(split.value("behavior", "") == "Isolated", "Split behavior must be Isolated");
    require(!split.value("invert", false), "Split invert must be false");
    const auto& bl = seq[1];
    require(bl.value("type", "") == "ByteLevel", "second pre_tokenizer must be ByteLevel");
    require(!bl.value("add_prefix_space", false), "ByteLevel add_prefix_space must be false");
    require(!bl.value("use_regex", true), "ByteLevel use_regex must be false");
}

std::optional<std::string> special_token_content(const Json& cfg, const char* key) {
    const auto it = cfg.find(key);
    if (it == cfg.end() || it->is_null()) return std::nullopt;
    if (it->is_string()) return it->get<std::string>();
    if (it->is_object() && it->contains("content") && (*it)["content"].is_string()) {
        return (*it)["content"].get<std::string>();
    }
    throw_error(ErrorCode::Model, "tokenizer_config.json: {} has an unexpected type", key);
}

std::string read_file(const std::filesystem::path& p) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(p, ec);
    HALO_CHECK(!ec, ErrorCode::Io, "cannot stat {}: {}", p.string(), ec.message());
    HALO_CHECK(size <= kMaxHfFileBytes, ErrorCode::Io, "{} is {} bytes (limit {})", p.string(), size, kMaxHfFileBytes);
    std::ifstream f(p, std::ios::binary);
    HALO_CHECK(f.good(), ErrorCode::Io, "cannot open {}", p.string());
    std::string s(static_cast<std::size_t>(size), '\0');
    f.read(s.data(), static_cast<std::streamsize>(size));
    HALO_CHECK(f.gcount() == static_cast<std::streamsize>(size), ErrorCode::Io, "short read on {}", p.string());
    return s;
}

}  // namespace

Tokenizer Tokenizer::from_hf_json(std::string_view tokenizer_json, std::string_view tokenizer_config_json) {
    Json j;
    try {
        j = Json::parse(tokenizer_json);
    } catch (const Json::exception& e) {
        throw_error(ErrorCode::Model, "tokenizer.json is not valid JSON: {}", e.what());
    }
    VocabSpec spec;
    try {
        require(j.is_object(), "top level must be an object");
        require(is_null_or(j, "truncation", Json()) && is_null_or(j, "padding", Json()),
                "truncation/padding are not supported");
        // normalizer
        const auto nit = j.find("normalizer");
        if (nit == j.end() || nit->is_null()) {
            spec.normalizer = Normalizer::None;
        } else {
            require(nit->is_object() && nit->value("type", "") == "NFC" && nit->size() == 1,
                    "only the NFC normalizer is supported");
            spec.normalizer = Normalizer::Nfc;
        }
        check_pre_tokenizer(j.at("pre_tokenizer"));
        const auto& dec = j.at("decoder");
        require(dec.is_object() && dec.value("type", "") == "ByteLevel", "decoder must be ByteLevel");
        const auto pit = j.find("post_processor");
        require(pit == j.end() || pit->is_null() || (pit->is_object() && pit->value("type", "") == "ByteLevel"),
                "post_processor must be null or ByteLevel");
        const auto& model = j.at("model");
        require(model.value("type", "") == "BPE", "model must be BPE");
        require(is_null_or(model, "dropout", Json()), "BPE dropout is not supported");
        require(is_null_or(model, "unk_token", Json()), "BPE unk_token is not supported");
        require(!model.value("byte_fallback", false), "BPE byte_fallback is not supported");
        require(!model.value("ignore_merges", false), "BPE ignore_merges is not supported");
        require(is_null_or(model, "continuing_subword_prefix", Json("")), "continuing_subword_prefix unsupported");
        require(is_null_or(model, "end_of_word_suffix", Json("")), "end_of_word_suffix unsupported");

        const auto& vocab = model.at("vocab");
        HALO_CHECK(vocab.is_object(), ErrorCode::Model, "tokenizer.json: model.vocab must be an object");
        const auto& added = j.contains("added_tokens") ? j.at("added_tokens") : Json::array();
        HALO_CHECK(added.is_array(), ErrorCode::Model, "tokenizer.json: added_tokens must be an array");
        std::size_t n = 0;
        const auto take_id = [&](const Json& v) {
            HALO_CHECK(v.is_number_integer(), ErrorCode::Model, "tokenizer.json: token id must be an integer");
            const auto id = v.get<std::int64_t>();
            HALO_CHECK(id >= 0 && static_cast<std::uint64_t>(id) < kMaxVocab, ErrorCode::Model,
                       "tokenizer.json: token id {} out of range", id);
            n = std::max(n, static_cast<std::size_t>(id) + 1);
            return static_cast<std::size_t>(id);
        };
        for (const auto& [tok, id] : vocab.items()) take_id(id);
        for (const auto& a : added) take_id(a.at("id"));
        spec.tokens.assign(n, std::string());
        spec.types.assign(n, TokenType::Unused);
        for (const auto& [tok, idv] : vocab.items()) {
            const std::size_t id = idv.get<std::size_t>();
            HALO_CHECK(spec.types[id] == TokenType::Unused, ErrorCode::Model, "tokenizer.json: id {} used twice", id);
            spec.tokens[id] = tok;
            spec.types[id] = TokenType::Normal;
        }
        for (const auto& a : added) {
            const std::size_t id = a.at("id").get<std::size_t>();
            const auto content = a.at("content").get<std::string>();
            require(!a.value("lstrip", false) && !a.value("rstrip", false) && !a.value("single_word", false),
                    "added tokens with lstrip/rstrip/single_word are not supported");
            require(!a.value("normalized", false), "added tokens with normalized=true are not supported");
            HALO_CHECK(spec.types[id] == TokenType::Unused || spec.tokens[id] == content, ErrorCode::Model,
                       "tokenizer.json: added token {} conflicts with vocab entry", id);
            spec.tokens[id] = content;
            spec.types[id] = a.value("special", false) ? TokenType::Control : TokenType::UserDefined;
        }
        const auto& merges = model.at("merges");
        HALO_CHECK(merges.is_array(), ErrorCode::Model, "tokenizer.json: merges must be an array");
        spec.merges.reserve(merges.size());
        for (const auto& m : merges) {
            if (m.is_string()) {
                spec.merges.push_back(m.get<std::string>());
            } else {
                HALO_CHECK(m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string(), ErrorCode::Model,
                           "tokenizer.json: merge entry must be a string or a pair of strings");
                spec.merges.push_back(m[0].get<std::string>() + " " + m[1].get<std::string>());
            }
        }
        if (!tokenizer_config_json.empty()) {
            const Json cfg = Json::parse(tokenizer_config_json);
            HALO_CHECK(cfg.is_object(), ErrorCode::Model, "tokenizer_config.json must be an object");
            const auto lookup = [&](const char* key) -> std::optional<std::int32_t> {
                const auto content = special_token_content(cfg, key);
                if (!content) return std::nullopt;
                for (std::size_t i = 0; i < n; ++i) {
                    if (spec.types[i] != TokenType::Unused && spec.tokens[i] == *content) {
                        return static_cast<std::int32_t>(i);
                    }
                }
                throw_error(ErrorCode::Model, "tokenizer_config.json: {} '{}' is not in the vocabulary", key, *content);
            };
            spec.bos = lookup("bos_token");
            spec.eos = lookup("eos_token");
            spec.pad = lookup("pad_token");
        }
    } catch (const Json::exception& e) {
        throw_error(ErrorCode::Model, "tokenizer.json: malformed field: {}", e.what());
    }
    return from_spec(std::move(spec));
}

Tokenizer Tokenizer::from_hf_dir(const std::filesystem::path& dir) {
    const std::string tj = read_file(dir / "tokenizer.json");
    std::string cfg;
    if (std::filesystem::exists(dir / "tokenizer_config.json")) cfg = read_file(dir / "tokenizer_config.json");
    return from_hf_json(tj, cfg);
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view utf8, bool parse_special) const {
    std::vector<std::int32_t> ids;
    ids.reserve(utf8.size() / 3 + 4);
    impl_->encode(utf8, parse_special, ids, nullptr);
    return ids;
}

Encoding Tokenizer::encode_with_offsets(std::string_view utf8, bool parse_special) const {
    Encoding e;
    e.ids.reserve(utf8.size() / 3 + 4);
    e.offsets.reserve(utf8.size() / 3 + 4);
    impl_->encode(utf8, parse_special, e.ids, &e.offsets);
    return e;
}

std::size_t Encoding::token_index_at(std::size_t byte_offset) const noexcept {
    return static_cast<std::size_t>(std::lower_bound(offsets.begin(), offsets.end(), byte_offset) - offsets.begin());
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids, bool skip_special) const {
    std::string raw;
    for (const std::int32_t id : ids) {
        impl_->check_id(id);
        if (skip_special && impl_->types[static_cast<std::size_t>(id)] == TokenType::Control) continue;
        raw += impl_->pieces[static_cast<std::size_t>(id)];
    }
    return unicode::is_valid_utf8(raw) ? raw : unicode::to_valid_utf8(raw);
}

const std::string& Tokenizer::token_to_piece(std::int32_t id) const {
    impl_->check_id(id);
    return impl_->pieces[static_cast<std::size_t>(id)];
}

std::optional<std::int32_t> Tokenizer::piece_to_id(std::string_view piece) const {
    const auto it = impl_->by_piece.find(std::string(piece));
    if (it == impl_->by_piece.end()) return std::nullopt;
    return it->second;
}

std::size_t Tokenizer::vocab_size() const noexcept { return impl_->pieces.size(); }

TokenType Tokenizer::token_type(std::int32_t id) const {
    impl_->check_id(id);
    return impl_->types[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_special(std::int32_t id) const { return token_type(id) == TokenType::Control; }
std::optional<std::int32_t> Tokenizer::bos() const noexcept { return impl_->bos; }
std::optional<std::int32_t> Tokenizer::eos() const noexcept { return impl_->eos; }
std::optional<std::int32_t> Tokenizer::pad() const noexcept { return impl_->pad; }
Normalizer Tokenizer::normalizer() const noexcept { return impl_->normalizer; }

// ---- StreamDecoder -----------------------------------------------------------------------

StreamDecoder::StreamDecoder(const Tokenizer& tokenizer, bool skip_special)
    : tok_(&tokenizer), skip_special_(skip_special) {}

std::string StreamDecoder::push(std::int32_t id) {
    if (skip_special_ && tok_->is_special(id)) return {};
    pending_ += tok_->token_to_piece(id);
    std::string out;
    std::size_t i = 0;
    while (i < pending_.size()) {
        const auto st = unicode::decode_step(pending_, i);
        if (st.status == unicode::Utf8Step::Status::Incomplete) break;
        if (st.status == unicode::Utf8Step::Status::Ok) {
            out.append(pending_, i, st.len);
        } else {
            unicode::append_utf8(out, unicode::kReplacement);
        }
        i += st.len;
    }
    pending_.erase(0, i);
    return out;
}

std::string StreamDecoder::flush() {
    std::string out = unicode::to_valid_utf8(pending_);
    pending_.clear();
    return out;
}

}  // namespace halo::tokenizer
