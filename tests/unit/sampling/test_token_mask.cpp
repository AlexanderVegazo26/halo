// Token-mask engine and structured sampling tests (WS-H M3, TRD §25).
//
// References are independent of the code under test:
//   * the allowed-token mask is recomputed by a test-side scan: a separate matcher runtime
//     replays the accepted bytes, then every included token's bytes are stepped through it
//     (no trie, no mask cache, no TokenMatcher state);
//   * "never a dead state" is checked by breadth-first search for a completion from the
//     state after an allowed token (walk.h);
//   * generated documents are validated by the schema validator in json_check.h.
// Real-vocabulary tests use /root/halo-ref/tokenizer.json (248,077 entries; logits rows
// padded to 248,320) and skip with a reason when it is missing.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "halo/core/error.h"
#include "halo/sampling/sampler.h"
#include "halo/sampling/structured.h"
#include "halo/tokenizer/tokenizer.h"
#include "json_check.h"
#include "sampling/grammar_impl.h"
#include "walk.h"

namespace {

using halo::Error;
using halo::ErrorCode;
using halo::SamplingParams;
using halo::sampling::ByteMatcher;
using halo::sampling::CandidateSet;
using halo::sampling::Grammar;
using halo::sampling::Sampler;
using halo::sampling::SamplerConfig;
using halo::sampling::TokenMatcher;
using halo::sampling::TokenVocab;
using halo::sampling::detail::kDead;
using halo::sampling::detail::Runtime;
using halo::sampling::test::OJson;
using halo::sampling::test::shortest_completion;
using halo::sampling::test::validate_text;

constexpr std::size_t kPaddedVocab = 248320;  // GGUF output rows (logits row length)

template <typename F>
ErrorCode code_of(F&& f) {
    try {
        f();
    } catch (const Error& e) {
        return e.code();
    }
    ADD_FAILURE() << "expected halo::Error";
    return ErrorCode::Cancelled;
}

bool bit(std::span<const std::uint64_t> m, std::int32_t id) {
    const auto u = static_cast<std::size_t>(id);
    return (u >> 6U) < m.size() && ((m[u >> 6U] >> (u & 63U)) & 1U) != 0;
}

std::vector<std::int32_t> ids_of(std::span<const std::uint64_t> m) {
    std::vector<std::int32_t> out;
    for (std::size_t w = 0; w < m.size(); ++w) {
        for (unsigned b = 0; b < 64; ++b) {
            if ((m[w] >> b) & 1U) out.push_back(static_cast<std::int32_t>(w * 64 + b));
        }
    }
    return out;
}

/// Test-side reference: follows the same document with its own runtime.
class Replica {
public:
    explicit Replica(const Grammar& g) : rt_(g.data()), s_(rt_.initial()) {}

    /// The allowed-token mask by stepping every included token through the replica.
    std::vector<std::uint64_t> scan(const TokenVocab& v, std::span<const std::int32_t> eos) {
        std::vector<std::uint64_t> out((v.vocab_size() + 63) / 64, 0);
        for (std::size_t id = 0; id < v.vocab_size(); ++id) {
            if (!v.included(static_cast<std::int32_t>(id))) continue;
            if (after(v.piece(static_cast<std::int32_t>(id))) != kDead) out[id >> 6U] |= std::uint64_t{1} << (id & 63U);
        }
        if (rt_.accepting(s_)) {
            for (const auto e : eos) out[static_cast<std::size_t>(e) >> 6U] |= std::uint64_t{1} << (static_cast<unsigned>(e) & 63U);
        }
        return out;
    }
    std::int32_t after(const std::string& bytes) {
        std::int32_t s = s_;
        for (const char c : bytes) {
            s = rt_.step(s, static_cast<std::uint8_t>(c));
            if (s == kDead) return kDead;
        }
        return s;
    }
    bool completable_after(const std::string& bytes) {
        const std::int32_t s = after(bytes);
        return s != kDead && shortest_completion(rt_, s).has_value();
    }
    void advance(const std::string& bytes) {
        s_ = after(bytes);
        ASSERT_NE(s_, kDead);
    }
    bool accepting() const { return rt_.accepting(s_); }

private:
    Runtime rt_;
    std::int32_t s_;
};

struct WalkStats {
    std::size_t steps = 0;
    std::size_t tokens_checked = 0;
    double trie_us = 0;
    double scan_us = 0;
};

/// Random walk over allowed tokens; at every step the matcher's mask (cached and uncached)
/// must equal the replica scan, and up to `probe` random allowed tokens must lead to a
/// completable state. Returns the document bytes.
std::string checked_walk(const Grammar& g, const std::shared_ptr<const TokenVocab>& v, std::int32_t eos,
                         std::mt19937_64& rng, std::size_t max_steps, std::size_t probe, WalkStats& st) {
    TokenMatcher m(g, v, {eos});
    Replica ref(g);
    std::string doc;
    for (std::size_t step = 0; step < max_steps; ++step) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto uspan = m.mask_uncached();
        const std::vector<std::uint64_t> uncached(uspan.begin(), uspan.end());
        const auto t1 = std::chrono::steady_clock::now();
        const auto want = ref.scan(*v, std::vector<std::int32_t>{eos});
        const auto t2 = std::chrono::steady_clock::now();
        st.trie_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        st.scan_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        const auto cspan = m.mask();
        const std::vector<std::uint64_t> cached(cspan.begin(), cspan.end());
        EXPECT_EQ(uncached, want) << "trie mask differs from the reference scan after '" << doc << "'";
        EXPECT_EQ(cached, want) << "cached mask differs after '" << doc << "'";
        EXPECT_EQ(bit(cached, eos), ref.accepting()) << "EOS bit vs completeness after '" << doc << "'";
        ++st.steps;
        const auto allowed = ids_of(want);
        if (allowed.empty()) {
            ADD_FAILURE() << "no token allowed after '" << doc << "'";
            break;
        }
        for (std::size_t k = 0; k < probe && k < allowed.size(); ++k) {
            const auto id = allowed[rng() % allowed.size()];
            if (id == eos) continue;
            EXPECT_TRUE(ref.completable_after(v->piece(id))) << "token " << id << " leads to a dead end after '" << doc << "'";
            ++st.tokens_checked;
        }
        const auto id = allowed[rng() % allowed.size()];
        m.accept_token(id);
        if (id == eos) {
            EXPECT_TRUE(m.is_finished());
            break;
        }
        ref.advance(v->piece(id));
        doc += v->piece(id);
    }
    return doc;
}

// ---- synthetic vocabulary (no reference data) -----------------------------------------

struct Synthetic {
    std::shared_ptr<const TokenVocab> vocab;
    std::map<std::string, std::int32_t> id;
    std::int32_t eos = 0;
};

const Synthetic& synthetic() {
    static const Synthetic s = [] {
        Synthetic r;
        std::vector<std::string> pieces{"<eos>"};
        std::vector<bool> inc{false};
        const auto add = [&](const std::string& p) {
            r.id.emplace(p, static_cast<std::int32_t>(pieces.size()));
            pieces.push_back(p);
            inc.push_back(true);
        };
        for (int c = 0x20; c < 0x7F; ++c) add(std::string(1, static_cast<char>(c)));
        for (const char* p : {"\n", "\t", "\xC3", "\xA9", "\xC3\xA9", "\xC3\xA9\"", "\xE6", "\x97", "\xA5", "\xE6\x97",
                              "\xE6\x97\xA5", "\xF0\x9F", "\x98\x80", "{\"", "\":", "\",", "\"}", "true", "tr", "ue}",
                              "null", "  ", "},{", "[1", "12", "e+", "\\u", "d8", "00", "\\n", "\xFF"}) {
            add(p);
        }
        r.id.emplace("<eos>", 0);
        pieces.push_back("");  // an Unused-like entry: excluded
        inc.push_back(false);
        r.vocab = TokenVocab::from_pieces(std::move(pieces), std::move(inc));
        return r;
    }();
    return s;
}

TEST(TokenMask, SyntheticVocabMatchesReferenceScanOnRandomWalks) {
    const auto& sv = synthetic();
    const std::vector<Grammar> grammars{
        Grammar::any_json_value(), Grammar::any_json_object(),
        Grammar::from_json_schema(R"({"type":"object","properties":{"s":{"type":"string","maxLength":3},"b":{"type":"boolean"}},"required":["s"],"additionalProperties":false})"),
        Grammar::from_json_schema(R"({"type":"string","pattern":"^(é|日)+$"})"),
        Grammar::from_json_schema(R"({"enum":["é日","tr","true",12]})"),
        // An unsatisfiable optional property: its key must never be offered, because no
        // value could follow it (exercises the productive-rule pruning).
        Grammar::from_json_schema(R"({"type":"object","properties":{"a":{"type":"array","minItems":3,"maxItems":1},"b":{"type":"integer"}},"additionalProperties":false})")};
    std::mt19937_64 rng(31);
    WalkStats st;
    for (const auto& g : grammars) {
        for (int run = 0; run < 40; ++run) (void)checked_walk(g, sv.vocab, sv.eos, rng, 40, 8, st);
    }
    EXPECT_GT(st.steps, 1000U);
    EXPECT_GT(st.tokens_checked, 1000U);
}

TEST(TokenMask, TokensEndingMidUtf8AreByteLevel) {
    const auto& sv = synthetic();
    const auto g = Grammar::from_json_schema(R"({"type":"string","pattern":"^é+$"})");
    TokenMatcher m(g, sv.vocab, {sv.eos});
    const auto id = [&](const char* p) { return sv.id.at(p); };
    m.accept_token(id("\""));
    EXPECT_TRUE(m.is_allowed(id("\xC3")));        // first half of é
    EXPECT_TRUE(m.is_allowed(id("\xC3\xA9")));
    EXPECT_FALSE(m.is_allowed(id("\xA9")));       // continuation byte without a lead byte
    EXPECT_FALSE(m.is_allowed(id("\"")));         // needs at least one é
    EXPECT_FALSE(m.is_allowed(id("\xE6")));       // lead byte of a different character
    EXPECT_FALSE(m.is_allowed(sv.eos));
    m.accept_token(id("\xC3"));
    EXPECT_TRUE(m.is_allowed(id("\xA9")));
    EXPECT_FALSE(m.is_allowed(id("\xC3")));
    EXPECT_FALSE(m.is_allowed(id("\xC3\xA9")));
    EXPECT_FALSE(m.is_allowed(id("\"")));
    m.accept_token(id("\xA9"));
    EXPECT_TRUE(m.is_allowed(id("\"")));
    EXPECT_TRUE(m.is_allowed(id("\xC3\xA9\"")));
    EXPECT_FALSE(m.is_complete());
    m.accept_token(id("\xC3\xA9\""));
    EXPECT_TRUE(m.is_complete());
    EXPECT_TRUE(bit(m.mask(), sv.eos));
    // Invalid UTF-8 bytes are never allowed in any JSON string.
    TokenMatcher any(Grammar::any_json_value(), sv.vocab, {sv.eos});
    any.accept_token(id("\""));
    EXPECT_FALSE(any.is_allowed(id("\xFF")));
    EXPECT_TRUE(any.is_allowed(id("\xF0\x9F")));
    any.accept_token(id("\xF0\x9F"));
    EXPECT_TRUE(any.is_allowed(id("\x98\x80")));
    EXPECT_FALSE(any.is_allowed(id("\"")));
}

TEST(TokenMask, EosAcceptSnapshotAndErrors) {
    const auto& sv = synthetic();
    const auto g = Grammar::from_json_schema(R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})");
    TokenMatcher m(g, sv.vocab, {sv.eos});
    const auto id = [&](const char* p) { return sv.id.at(p); };
    EXPECT_EQ(code_of([&] { m.accept_token(sv.eos); }), ErrorCode::Api);  // EOS before completion
    EXPECT_EQ(code_of([&] { m.accept_token(id("true")); }), ErrorCode::Api);
    EXPECT_EQ(code_of([&] { m.accept_token(static_cast<std::int32_t>(sv.vocab->vocab_size() - 1)); }), ErrorCode::Api);
    m.accept_token(id("{\""));
    const auto snap = m.snapshot();
    const auto snap_span = m.mask();
    const std::vector<std::uint64_t> at_snap(snap_span.begin(), snap_span.end());
    m.accept_token(id("a"));
    m.accept_token(id("\":"));
    m.accept_token(id("12"));
    EXPECT_FALSE(m.is_complete());
    m.restore(snap);
    const auto restored = m.mask();
    EXPECT_EQ(std::vector<std::uint64_t>(restored.begin(), restored.end()), at_snap);
    for (const char* p : {"a", "\":", "12", "}"}) m.accept_token(id(p));
    EXPECT_TRUE(m.is_complete());
    m.accept_token(sv.eos);
    EXPECT_TRUE(m.is_finished());
    EXPECT_TRUE(ids_of(m.mask()).empty());
    EXPECT_EQ(code_of([&] { m.accept_token(id("}")); }), ErrorCode::Api);
    m.reset();
    EXPECT_FALSE(m.is_finished());
    EXPECT_TRUE(m.is_allowed(id("{\"")));
    // EOS must be a non-text token inside the vocabulary.
    EXPECT_EQ(code_of([&] { TokenMatcher bad(g, sv.vocab, {id("}")}); }), ErrorCode::Api);
    EXPECT_EQ(code_of([&] { TokenMatcher bad(g, sv.vocab, {}); }), ErrorCode::Api);
}

TEST(StructuredSampler, SyntheticIntegrationAndErrors) {
    const auto& sv = synthetic();
    const auto n = sv.vocab->vocab_size();
    SamplingParams p;
    p.json_schema = R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})";
    p.seed = 5;
    // A structured sampler needs a vocabulary source and EOS ids.
    EXPECT_EQ(code_of([&] {
                  SamplerConfig c;
                  c.vocab_size = n;
                  (void)Sampler::create(p, nullptr, c);
              }),
              ErrorCode::Api);
    SamplerConfig cfg;
    cfg.vocab_size = n + 7;  // padded logits row
    cfg.vocab = sv.vocab;
    EXPECT_EQ(code_of([&] { (void)Sampler::create(p, nullptr, cfg); }), ErrorCode::Api);  // no EOS
    cfg.eos_ids = {sv.eos};
    SamplingParams bad = p;
    bad.json_schema = R"({"type":"string","format":"email"})";
    EXPECT_EQ(code_of([&] { (void)Sampler::create(bad, nullptr, cfg); }), ErrorCode::Unsupported);
    auto s = Sampler::create(p, nullptr, cfg);
    ASSERT_TRUE(s.structured());
    // Every candidate disallowed: try_sample returns nullopt without consuming randomness.
    const std::vector<std::int32_t> ids{sv.id.at("true"), sv.id.at("12")};
    const std::vector<float> lg{3.0f, 2.0f};
    EXPECT_FALSE(s.try_sample(CandidateSet{ids, lg}, {}).has_value());
    EXPECT_EQ(code_of([&] { (void)s.sample(CandidateSet{ids, lg}, {}); }), ErrorCode::Api);
    std::mt19937 g(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> row(n + 7);
    for (auto& x : row) x = nd(g);
    auto fresh = Sampler::create(p, nullptr, cfg);
    EXPECT_EQ(s.sample(row, {}), fresh.sample(row, {}));
    // Padded ids (>= TokenVocab size) are never produced even with a huge logit.
    row[n + 3] = 100.0f;
    std::vector<std::int32_t> hist;
    std::string doc;
    for (int step = 0; step < 200 && !s.grammar_finished(); ++step) {
        const auto t = s.sample(row, hist);
        ASSERT_LT(static_cast<std::size_t>(t), n);
        const auto bad_id = sv.id.at("true");
        if (t != bad_id && !s.matcher()->is_allowed(bad_id)) {
            EXPECT_EQ(code_of([&] { s.accept(bad_id); }), ErrorCode::Api);  // state unchanged
        }
        s.accept(t);
        hist.push_back(t);
        if (t != sv.eos) doc += sv.vocab->piece(t);
        for (auto& x : row) x = nd(g);
        row[static_cast<std::size_t>(sv.eos)] += 6.0f;
        row[n + 3] = 100.0f;
    }
    ASSERT_TRUE(s.grammar_finished()) << doc;
    EXPECT_EQ(validate_text(OJson::parse(*p.json_schema), doc), "") << doc;
    EXPECT_EQ(code_of([&] { (void)s.sample(row, hist); }), ErrorCode::Api);  // after EOS
    s.reset();
    EXPECT_FALSE(s.grammar_finished());
}

// ---- real 248K vocabulary -------------------------------------------------------------

struct Real {
    std::optional<halo::tokenizer::Tokenizer> tok;
    std::shared_ptr<const TokenVocab> vocab;
    std::int32_t eos = -1;
    std::string why;
};

const Real& real() {
    static const Real r = [] {
        Real x;
        const std::filesystem::path ref = HALO_REF_DIR;
        if (!std::filesystem::exists(ref / "tokenizer.json")) {
            x.why = "no " + (ref / "tokenizer.json").string();
            return x;
        }
        x.tok.emplace(halo::tokenizer::Tokenizer::from_hf_dir(ref));
        x.vocab = TokenVocab::build(*x.tok);
        x.eos = x.tok->eos().value_or(-1);
        return x;
    }();
    return r;
}

#define REQUIRE_REAL_VOCAB()                                       \
    do {                                                           \
        if (!real().tok) GTEST_SKIP() << "real vocab: " << real().why; \
    } while (0)

TEST(TokenMaskReal, VocabularyFacts) {
    REQUIRE_REAL_VOCAB();
    const auto& r = real();
    EXPECT_EQ(r.tok->vocab_size(), 248077U);
    EXPECT_EQ(r.vocab->vocab_size(), 248077U);
    EXPECT_EQ(r.tok->token_to_piece(r.eos), "<|im_end|>");
    EXPECT_FALSE(r.vocab->included(r.eos));
    // Control / UserDefined tokens (<think>, <tool_call>, ...) are never text in JSON.
    std::size_t included = 0;
    for (std::size_t i = 0; i < r.vocab->vocab_size(); ++i) {
        const auto id = static_cast<std::int32_t>(i);
        const bool normal = r.tok->token_type(id) == halo::tokenizer::TokenType::Normal;
        EXPECT_EQ(r.vocab->included(id), normal) << id;
        included += r.vocab->included(id) ? 1 : 0;
    }
    EXPECT_EQ(included, 248044U);
    const auto think = r.tok->piece_to_id("<think>");
    ASSERT_TRUE(think.has_value());
    EXPECT_FALSE(r.vocab->included(*think));
}

TEST(TokenMaskReal, MaskEqualsReferenceScanAndNeverAllowsDeadEnds) {
    REQUIRE_REAL_VOCAB();
    const auto& r = real();
    const std::vector<std::string> schemas{
        R"({"type":"object","properties":{"name":{"type":"string","maxLength":10},"age":{"type":"integer"},"tags":{"type":"array","items":{"enum":["a","bé","日本"]},"maxItems":3}},"required":["name"],"additionalProperties":false})",
        R"({"type":"object","properties":{"id":{"type":"string","pattern":"^[A-Z]{2}-\\d{3}$"},"ok":{"type":"boolean"}},"required":["id","ok"]})",
        R"({"anyOf":[{"type":"null"},{"type":"array","items":{"type":"number"},"minItems":1,"maxItems":4}]})"};
    std::mt19937_64 rng(99);
    WalkStats st;
    for (const auto& text : schemas) {
        const auto g = Grammar::from_json_schema(text);
        for (int run = 0; run < 3; ++run) (void)checked_walk(g, r.vocab, r.eos, rng, 12, 16, st);
    }
    // json_object mode too.
    for (int run = 0; run < 3; ++run) (void)checked_walk(Grammar::any_json_object(), r.vocab, r.eos, rng, 12, 16, st);
    EXPECT_GE(st.steps, 100U);
    // Dev-host smoke numbers only (D-001): not a performance claim.
    std::printf("[dev-host smoke] %zu steps: trie mask (uncached) %.0f us/step, reference scan %.0f us/step\n",
                st.steps, st.trie_us / static_cast<double>(st.steps), st.scan_us / static_cast<double>(st.steps));
}

TEST(TokenMaskReal, EosAllowedExactlyWhenCompleteAlongTokenizedDocuments) {
    REQUIRE_REAL_VOCAB();
    const auto& r = real();
    struct Case {
        Grammar g;
        std::string doc;
    };
    const std::vector<Case> cases{
        {Grammar::any_json_object(), R"({"name": "Zoë 日本 😀", "n": [1, 2.5e-3, true, null], "o": {}})"},
        {Grammar::from_json_schema(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]})"),
         R"({"name":"Ann","age":31})"},
        {Grammar::from_json_schema(R"({"type":"array","items":{"type":"integer"}})"), "[10, 200, -3]"},
        {Grammar::from_json_schema(R"({"type":"integer"})"), "12345"}};
    for (const auto& c : cases) {
        const auto toks = r.tok->encode(c.doc, false);
        ASSERT_FALSE(toks.empty());
        TokenMatcher m(c.g, r.vocab, {r.eos});
        std::string prefix;
        for (const auto t : toks) {
            const auto mask = m.mask();
            ByteMatcher bm(c.g);
            ASSERT_TRUE(bm.accept_bytes(prefix));
            EXPECT_EQ(bit(mask, r.eos), bm.is_accepting()) << c.doc << " at '" << prefix << "'";
            ASSERT_TRUE(bit(mask, t)) << "token " << t << " of a valid document is masked: " << c.doc << " at '" << prefix << "'";
            m.accept_token(t);
            prefix += r.vocab->piece(t);
        }
        ASSERT_EQ(prefix, c.doc);
        EXPECT_TRUE(m.is_complete());
        EXPECT_TRUE(bit(m.mask(), r.eos));
        m.accept_token(r.eos);
        EXPECT_TRUE(m.is_finished());
    }
}

TEST(StructuredSamplerReal, RandomSchemaRunsAlwaysProduceValidJson) {
    REQUIRE_REAL_VOCAB();
    const auto& r = real();
    // Bounded schemas (every string has a length bound, enum or pattern), so random logits
    // terminate; EOS gets a boost so it wins once it becomes allowed.
    const std::vector<std::string> schemas{
        R"({"type":"object","properties":{"name":{"type":"string","maxLength":12},"age":{"type":"integer"},"tags":{"type":"array","items":{"enum":["a","b","c"]},"maxItems":3}},"required":["name"],"additionalProperties":false})",
        R"({"type":"object","properties":{"id":{"type":"string","pattern":"^[A-Z]{2}-\\d{3}$"},"ok":{"type":"boolean"}},"required":["id","ok"],"additionalProperties":false})",
        R"({"$defs":{"n":{"type":"object","properties":{"v":{"type":"number"},"kids":{"type":"array","items":{"$ref":"#/$defs/n"},"maxItems":1}},"required":["v"],"additionalProperties":false}},"$ref":"#/$defs/n"})",
        R"({"anyOf":[{"type":"null"},{"type":"array","items":{"type":"string","minLength":1,"maxLength":3},"minItems":1,"maxItems":4}]})",
        R"({"oneOf":[{"type":"object","properties":{"t":{"const":"x"},"n":{"type":"integer"}},"required":["t","n"],"additionalProperties":false},{"type":"object","properties":{"t":{"const":"y"},"s":{"type":"string","maxLength":4}},"required":["t"],"additionalProperties":false}]})",
        R"({"enum":["日本語","é","naïve","😀x",1.5,null,{"k":[true]}]})",
        R"({"type":"string","minLength":1,"maxLength":6})"};
    constexpr int kRunsPerSchema = 30;
    std::mt19937 g(2026);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<std::vector<float>> base(4, std::vector<float>(kPaddedVocab));
    for (auto& b : base) {
        for (auto& x : b) x = nd(g);
    }
    SamplerConfig cfg;
    cfg.vocab_size = kPaddedVocab;
    cfg.vocab = r.vocab;
    std::vector<float> row(kPaddedVocab);
    std::size_t runs = 0, tokens = 0;
    for (std::size_t si = 0; si < schemas.size(); ++si) {
        const auto schema = OJson::parse(schemas[si]);
        for (int run = 0; run < kRunsPerSchema; ++run) {
            SamplingParams p;
            p.json_schema = schemas[si];
            p.top_k = 40;
            p.seed = static_cast<std::uint64_t>(si * 1000 + static_cast<std::size_t>(run));
            auto s = Sampler::create(p, &*r.tok, cfg);
            std::mt19937_64 pick(*p.seed);
            std::vector<std::int32_t> hist;
            std::string doc;
            for (int step = 0; step < 400 && !s.grammar_finished(); ++step) {
                const auto& b = base[pick() % base.size()];
                const std::size_t off = pick() % kPaddedVocab;
                for (std::size_t i = 0; i < kPaddedVocab; ++i) row[i] = b[(i + off) % kPaddedVocab];
                row[static_cast<std::size_t>(r.eos)] += 8.0f;
                const auto t = s.sample(row, hist);
                ASSERT_LT(static_cast<std::size_t>(t), r.vocab->vocab_size());
                s.accept(t);
                hist.push_back(t);
                if (t != r.eos) doc += r.vocab->piece(t);
            }
            ASSERT_TRUE(s.grammar_finished()) << "run did not finish: " << schemas[si] << "\n" << doc;
            ASSERT_EQ(validate_text(schema, doc), "") << schemas[si] << "\n" << doc;
            ++runs;
            tokens += hist.size();
        }
    }
    EXPECT_EQ(runs, schemas.size() * kRunsPerSchema);
    EXPECT_GE(runs, 200U);
    std::printf("[structured] %zu runs, %zu tokens\n", runs, tokens);
}

}  // namespace
