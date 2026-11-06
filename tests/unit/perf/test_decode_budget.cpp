// End-to-end performance budget for the CPU Engine (ISO 25010 performance efficiency): the
// whole generate() path (scheduler tick, qwen35 forward, KV + GDN state, LM head, sampling,
// stream decoding) on a synthetic qwen35 model built in memory, so it runs in every lane
// without HALO_REF_DIR (the tiny-model engine tests in tests/unit/runtime need it).
//
// The model is tests/unit/model/gguf_builder.h's tiny_qwen35 (4 layers: 3 GDN + 1 attention,
// E=64, all-zero weights) with a byte-level vocabulary (256 byte tokens + 3 specials) so the
// tokenizer loads. Zero weights make greedy decoding deterministic (every logit is 0, argmax
// picks token 0) and the timings independent of the token values.
//
// Checks, single-threaded (threads = 1) so they don't depend on the runner's core count:
//   * Budget: decode ms per token after a short prompt, and the time to first token of a
//     1024-token prompt, best of N, must stay under ~10x the dev-host baseline. Catches gross
//     regressions anywhere on the path (a debug-only path left on, a per-token copy of the
//     weights or the KV pool, a lost batched forward), never runner noise.
//   * Scaling: per-token decode cost after a 3072-token prompt vs a 768-token prompt (4x the
//     context) may grow at most linearly. Only the attention layer depends on the context,
//     so an honest engine measures well under 4x; re-reading the history quadratically per
//     token blows far past it.
//   * Prefix cache (D-013, a second engine with the cache on): an exact re-submit of a
//     1024-token prompt restores all but the last prompt token from the cache, a follow-up
//     turn that extends the conversation reuses the same prefix, and in timing builds the
//     re-submit's time to first token is under a quarter of the cold prefill. These run in
//     every lane: the other prefix-cache engine tests (tests/unit/runtime) need HALO_REF_DIR.
//   * Prefix cache eviction (a third engine, prefix_cache_entries = 2): once a third distinct
//     prompt retires, the least recently used entry is the one evicted and the two newer ones
//     still hit.
//
// Dev-host baseline (2026-11-05, RelWithDebInfo clang-21, no -march, one thread): decode
// ~0.033 ms/token at 16 tokens of context, TTFT 1024-token prompt ~28 ms, context scaling
// 768 -> 3072 ~2.1x. Timings are NOT a HALO performance claim (DECISIONS.md D-001).
// Debug and sanitizer builds run one short generation (the functional checks) and skip the
// timing checks with a reason. ctest runs this RUN_SERIAL (label `perf`).
// Rerun: ctest --test-dir build -R test_decode_budget --output-on-failure

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "../model/gguf_builder.h"
#include "halo/runtime/cpu_engine.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define HALO_PERF_SANITIZED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define HALO_PERF_SANITIZED 1
#endif

namespace {

using halo::runtime::EngineConfig;
using halo::runtime::FinishReason;
using halo::runtime::GenerateRequest;
using halo::runtime::GenerateResult;
using halo::runtime::TokenEvent;

constexpr std::size_t kMaxContext = 4096;
constexpr std::int32_t kPromptToken = 'a';  // byte tokens are indexed by their byte value

// Why timings are not checked in this build, or nullptr when they are.
const char* skip_reason() {
#if defined(HALO_PERF_SANITIZED)
    return "sanitizer build: timings are not meaningful (perf budgets run in the RelWithDebInfo lane)";
#elif !defined(NDEBUG)
    return "Debug build (no NDEBUG): timings are not meaningful (perf budgets run in the RelWithDebInfo lane)";
#else
    return nullptr;
#endif
}

// GPT-2 byte-to-unicode token strings: printable bytes map to themselves, the rest to
// U+0100 + n in byte order (what the tokenizer requires of a byte-level vocabulary).
std::vector<std::string> byte_tokens() {
    std::vector<std::string> out(256);
    std::uint32_t n = 0;
    for (std::uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const std::uint32_t cp = printable ? b : 256 + n++;
        std::string& s = out[b];
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// tiny_qwen35 without MTP (no drafts: one token per tick, so ticks == tokens) and with the
// byte-level vocabulary; token_embd / output grow to the new vocabulary size.
std::vector<std::byte> synthetic_model() {
    using namespace halo::test;
    GgufSpec s = tiny_qwen35({.mtp = false});
    std::vector<std::string> tokens = byte_tokens();
    std::vector<std::int32_t> types(256, 1);  // 1 = normal
    for (const char* t : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) {
        tokens.emplace_back(t);
        types.push_back(3);  // 3 = control
    }
    const auto vocab = static_cast<std::int64_t>(tokens.size());
    s.kv("tokenizer.ggml.tokens", v_arr_str(tokens))
        .kv("tokenizer.ggml.token_type", v_arr_i32(types))
        .kv("tokenizer.ggml.merges", v_arr_str({}))
        .kv("tokenizer.ggml.bos_token_id", v_u32(256))
        .kv("tokenizer.ggml.eos_token_id", v_u32(258));
    const std::int64_t embd = s.find("token_embd.weight")->ne.at(0);
    s.find("token_embd.weight")->ne = {embd, vocab};
    s.find("output.weight")->ne = {embd, vocab};
    return s.build(true);
}

// Writes the synthetic model to a per-process temp file named after `tag`.
std::filesystem::path write_synthetic_model(const std::string& tag) {
    const auto path =
        std::filesystem::temp_directory_path() / ("halo-" + tag + "-" + std::to_string(::getpid()) + ".gguf");
    const auto bytes = synthetic_model();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    EXPECT_TRUE(f.good()) << "could not write the synthetic model to " << path;
    return path;
}

// Single-threaded, one sequence, no MTP.
std::unique_ptr<halo::runtime::Engine> make_engine(const std::filesystem::path& path, bool prefix_cache,
                                                   const halo::runtime::CpuEngineOptions& opts = {}) {
    EngineConfig c;
    c.model_path = path.string();
    c.max_context = kMaxContext;
    c.max_context_explicit = true;
    c.max_sequences = 1;
    c.threads = 1;
    c.mtp_enabled = false;
    c.prefix_cache = prefix_cache;
    return halo::runtime::create_cpu_engine(c, opts);
}

class DecodeBudget : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        path_ = write_synthetic_model("decode-budget");
        engine_ = make_engine(path_, /*prefix_cache=*/false);  // every repetition recomputes its prompt
    }
    static void TearDownTestSuite() {
        engine_.reset();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    void SetUp() override { ASSERT_NE(engine_, nullptr) << "engine creation on the synthetic model failed"; }

    // One greedy generation with the functional checks every measurement relies on.
    static GenerateResult generate(std::size_t prompt_len, std::size_t max_tokens) {
        GenerateRequest r;
        r.prompt.assign(prompt_len, kPromptToken);
        r.sampling.temperature = 0.0f;
        r.max_tokens = max_tokens;
        std::size_t callbacks = 0;
        GenerateResult res = engine_->generate(r, [&](const TokenEvent&) {
            ++callbacks;
            return true;
        });
        const std::string ctx = "prompt " + std::to_string(prompt_len) + ", max_tokens " + std::to_string(max_tokens);
        EXPECT_EQ(res.finish, FinishReason::Length) << ctx << ": error='" << res.error << "'";
        EXPECT_EQ(res.tokens.size(), max_tokens) << ctx;
        EXPECT_EQ(callbacks, max_tokens) << ctx << ": one callback per generated token";
        EXPECT_EQ(res.prompt_tokens, prompt_len) << ctx;
        EXPECT_EQ(res.cached_prompt_tokens, 0u) << ctx << ": prefix cache is off";
        EXPECT_GT(res.ttft_ms, 0.0) << ctx;
        if (max_tokens > 1) EXPECT_GT(res.decode_tps, 0.0) << ctx;
        return res;
    }

    // Best (lowest) decode ms per token and TTFT over `reps` runs after one warm-up run.
    struct Best {
        double decode_ms_per_token = 1e300;
        double ttft_ms = 1e300;
    };
    static Best best_of(int reps, std::size_t prompt_len, std::size_t max_tokens) {
        (void)generate(prompt_len, max_tokens);
        Best b;
        for (int i = 0; i < reps; ++i) {
            const GenerateResult res = generate(prompt_len, max_tokens);
            if (res.decode_tps > 0.0) b.decode_ms_per_token = std::min(b.decode_ms_per_token, 1000.0 / res.decode_tps);
            b.ttft_ms = std::min(b.ttft_ms, res.ttft_ms);
        }
        return b;
    }

    static inline std::filesystem::path path_;
    static inline std::unique_ptr<halo::runtime::Engine> engine_;
};

void expect_within_budget(const std::string& what, double ms, double budget_ms) {
    std::printf("[perf] %-46s best %9.4f ms  budget %8.2f ms  (%5.1f%% of budget)\n", what.c_str(), ms, budget_ms,
                100.0 * ms / budget_ms);
    ::testing::Test::RecordProperty(what + " ms", std::to_string(ms));
    EXPECT_LE(ms, budget_ms) << what << ": best of N took " << ms << " ms, budget is " << budget_ms
                             << " ms (~10x the 2026-11-05 dev-host baseline): a gross slowdown of the engine's "
                                "generate path, e.g. a debug-only path, a per-token copy, or a lost batched forward";
}

}  // namespace

TEST_F(DecodeBudget, GeneratesOnTheSyntheticModel) {
    // The functional part runs in every build, including Debug and the sanitizer lanes.
    const GenerateResult res = generate(16, 8);
    EXPECT_EQ(engine_->stats().threads, 1u) << "the budgets assume a single-threaded engine";
    EXPECT_EQ(engine_->model().vocab_size, 259u);
    for (std::size_t i = 0; i < res.tokens.size(); ++i) {
        EXPECT_EQ(res.tokens[i], 0) << "token " << i << ": all-zero weights give all-zero logits, greedy picks id 0";
    }
}

TEST_F(DecodeBudget, DecodePerTokenAfterAShortPrompt) {
    if (const char* why = skip_reason()) GTEST_SKIP() << why;
    const Best b = best_of(3, 16, 256);
    expect_within_budget("decode ms/token, context 16, 256 tokens", b.decode_ms_per_token, 0.35);  // 0.033
}

TEST_F(DecodeBudget, TimeToFirstTokenOfA1024TokenPrompt) {
    if (const char* why = skip_reason()) GTEST_SKIP() << why;
    const Best b = best_of(3, 1024, 2);
    expect_within_budget("TTFT, 1024-token prompt", b.ttft_ms, 300.0);  // 28
}

TEST_F(DecodeBudget, DecodeCostIsAtMostLinearInContext) {
    if (const char* why = skip_reason()) GTEST_SKIP() << why;
    const double small = best_of(3, 768, 128).decode_ms_per_token;
    const double large = best_of(3, 3072, 128).decode_ms_per_token;
    const double ratio = large / small;
    std::printf("[perf] %-46s %9.4f -> %9.4f ms/token  ratio %5.2f (limit 4.0)\n", "decode context 768 -> 3072", small,
                large, ratio);
    ::testing::Test::RecordProperty("decode context 768 -> 3072 ratio", std::to_string(ratio));
    EXPECT_LE(ratio, 4.0) << "decode ms/token after a 3072-token prompt (" << large << ") vs a 768-token prompt ("
                          << small << ") grew " << ratio
                          << "x for 4x the context; only attention depends on the context and is O(context) per "
                             "token, so more than 4x means per-token work grew faster than linear in the history";
}

namespace {

using Prompt = decltype(GenerateRequest{}.prompt);

// A second engine with the prefix cache on (D-013). Each test starts its prompts with its own
// token, so no test can hit an entry another test left behind (any order, --gtest_shuffle).
class PrefixCache : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        path_ = write_synthetic_model("prefix-cache");
        engine_ = make_engine(path_, /*prefix_cache=*/true);
    }
    static void TearDownTestSuite() {
        engine_.reset();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    void SetUp() override { ASSERT_NE(engine_, nullptr) << "engine creation on the synthetic model failed"; }

    static Prompt prompt_of(std::size_t len, std::int32_t token) { return Prompt(len, token); }

    // One greedy generation with the functional checks every case relies on.
    static GenerateResult run(const Prompt& prompt, std::size_t max_tokens, const std::string& ctx) {
        GenerateRequest r;
        r.prompt = prompt;
        r.sampling.temperature = 0.0f;
        r.max_tokens = max_tokens;
        GenerateResult res = engine_->generate(r, [](const TokenEvent&) { return true; });
        EXPECT_EQ(res.finish, FinishReason::Length) << ctx << ": error='" << res.error << "'";
        EXPECT_EQ(res.tokens.size(), max_tokens) << ctx;
        EXPECT_EQ(res.prompt_tokens, prompt.size()) << ctx;
        return res;
    }

    static inline std::filesystem::path path_;
    static inline std::unique_ptr<halo::runtime::Engine> engine_;
};

}  // namespace

TEST_F(PrefixCache, ExactResubmitRestoresAllButTheLastPromptToken) {
    const Prompt prompt = prompt_of(1024, 'p');
    const auto hits_before = engine_->stats().prefix_cache_hits;
    const GenerateResult cold = run(prompt, 4, "first submit of a 1024-token prompt");
    EXPECT_EQ(cold.cached_prompt_tokens, 0u) << "first submit: nothing cached starts with this token yet";
    const GenerateResult hot = run(prompt, 4, "exact re-submit of the same 1024-token prompt");
    EXPECT_EQ(hot.cached_prompt_tokens, prompt.size() - 1)
        << "exact re-submit should restore the KV rows and GDN state of every prompt token but the last (one row "
           "is always recomputed for the logits; the checkpoint at N - tail, D-013); 0 means the cache missed";
    EXPECT_EQ(engine_->stats().prefix_cache_hits, hits_before + 1) << "one hit (the re-submit), one miss";
    EXPECT_EQ(hot.tokens, cold.tokens) << "a restored prefix must generate what the cold prefill generated";
}

TEST_F(PrefixCache, FollowUpTurnReusesTheConversationPrefix) {
    // A chat client re-sends the whole conversation each turn: previous prompt + the reply + a
    // new user message. The previous prompt's state must come from the cache.
    const Prompt turn1 = prompt_of(512, 'q');
    const GenerateResult r1 = run(turn1, 8, "turn 1, 512-token prompt");
    EXPECT_EQ(r1.cached_prompt_tokens, 0u) << "turn 1: nothing cached starts with this token yet";
    Prompt turn2 = turn1;
    turn2.insert(turn2.end(), r1.tokens.begin(), r1.tokens.end());
    turn2.insert(turn2.end(), 64, 'r');
    const GenerateResult r2 = run(turn2, 4, "turn 2 = turn 1 prompt + reply + 64 new tokens");
    EXPECT_GE(r2.cached_prompt_tokens, turn1.size() - 1)
        << "turn 2 (" << turn2.size() << " tokens) should reuse at least turn 1's prompt minus its last token ("
        << turn1.size() - 1 << ") from the cache; it reused " << r2.cached_prompt_tokens;
    EXPECT_LT(r2.cached_prompt_tokens, turn2.size()) << "at least one prompt row is recomputed for the logits";
}

TEST_F(PrefixCache, ResubmitTimeToFirstTokenIsUnderAQuarterOfAColdPrefill) {
    if (const char* why = skip_reason()) GTEST_SKIP() << why;
    // Cold: a different first token each time, so every prompt misses. Hot: the same prompt again.
    (void)run(prompt_of(1024, 'A'), 2, "warm-up");
    double cold_ms = 1e300, hot_ms = 1e300;
    for (std::int32_t i = 0; i < 3; ++i) {
        const Prompt prompt = prompt_of(1024, 'B' + i);
        const GenerateResult cold = run(prompt, 2, "cold 1024-token prompt #" + std::to_string(i));
        ASSERT_EQ(cold.cached_prompt_tokens, 0u) << "cold run #" << i << " must miss the cache";
        const GenerateResult hot = run(prompt, 2, "re-submit #" + std::to_string(i));
        ASSERT_EQ(hot.cached_prompt_tokens, prompt.size() - 1) << "re-submit #" << i << " must hit the cache";
        cold_ms = std::min(cold_ms, cold.ttft_ms);
        hot_ms = std::min(hot_ms, hot.ttft_ms);
    }
    const double ratio = hot_ms / cold_ms;
    std::printf("[perf] %-46s cold %9.4f ms  hit %9.4f ms  ratio %5.3f (limit 0.25)\n", "TTFT 1024-token prompt, prefix cache",
                cold_ms, hot_ms, ratio);
    ::testing::Test::RecordProperty("prefix cache TTFT hit/cold ratio", std::to_string(ratio));
    EXPECT_LE(ratio, 0.25) << "TTFT of an exact re-submit (" << hot_ms << " ms, 1023 of 1024 tokens from the cache) vs "
                           << "the cold prefill (" << cold_ms << " ms): a hit recomputes one row, so it should be a "
                              "small fraction; near 1 means the restored prefix is being recomputed anyway";
}

namespace {

// A third engine whose cache holds two finished sequences (prefix_cache_entries = 2, with
// one slot), to pin which entry goes when a third distinct prompt retires (D-013: least
// recently used). The cache is shared, so each test uses its own first tokens (no leftover
// entry can serve its prompts) and submits two new prompts before it checks an eviction, which
// pushes out whatever an earlier test left behind (any order, --gtest_shuffle).
class PrefixCacheEviction : public ::testing::Test {
protected:
    static constexpr std::size_t kEntries = 2;
    static constexpr std::size_t kLen = 128;

    static void SetUpTestSuite() {
        path_ = write_synthetic_model("prefix-cache-eviction");
        halo::runtime::CpuEngineOptions o;
        o.prefix_cache_entries = kEntries;
        engine_ = make_engine(path_, /*prefix_cache=*/true, o);
    }
    static void TearDownTestSuite() {
        engine_.reset();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    void SetUp() override { ASSERT_NE(engine_, nullptr) << "engine creation on the synthetic model failed"; }

    // One greedy 2-token generation of a kLen-token prompt of `token`; returns the cached prompt tokens.
    static std::size_t cached(std::int32_t token, const std::string& ctx) {
        GenerateRequest r;
        r.prompt = Prompt(kLen, token);
        r.sampling.temperature = 0.0f;
        r.max_tokens = 2;
        const GenerateResult res = engine_->generate(r, [](const TokenEvent&) { return true; });
        EXPECT_EQ(res.finish, FinishReason::Length) << ctx << ": error='" << res.error << "'";
        EXPECT_EQ(res.prompt_tokens, kLen) << ctx;
        return res.cached_prompt_tokens;
    }

    static inline std::filesystem::path path_;
    static inline std::unique_ptr<halo::runtime::Engine> engine_;
};

}  // namespace

TEST_F(PrefixCacheEviction, ThirdDistinctPromptEvictsTheLeastRecentlyUsed) {
    // Prompts start with different tokens, so no entry can serve another prompt.
    EXPECT_EQ(cached('a', "prompt a, first submit"), 0u) << "nothing cached starts with 'a' yet";
    EXPECT_EQ(cached('b', "prompt b, first submit"), 0u) << "nothing cached starts with 'b' yet";
    EXPECT_EQ(cached('c', "prompt c, first submit"), 0u) << "nothing cached starts with 'c' yet";
    // Cache now holds c (most recent) and b; a, the oldest, was evicted when c retired.
    EXPECT_EQ(cached('a', "prompt a, re-submit after b and c"), 0u)
        << "with " << kEntries << " entries, a (the least recently used of a, b, c) must have been evicted; a hit "
        << "means the cache kept more than prefix_cache_entries or evicted a newer entry instead";
    // a's retirement evicted b (now the oldest); c is still cached.
    EXPECT_EQ(cached('c', "prompt c, re-submit"), kLen - 1)
        << "c was the most recent entry before a retired, so it must still hit (all but the last prompt token "
        << "restored); 0 means the newest entry was evicted instead of the oldest";
}

TEST_F(PrefixCacheEviction, CacheHoldsExactlyPrefixCacheEntriesDistinctPrompts) {
    EXPECT_EQ(cached('x', "prompt x, first submit"), 0u) << "nothing cached starts with 'x' yet";
    EXPECT_EQ(cached('y', "prompt y, first submit"), 0u) << "nothing cached starts with 'y' yet";
    // Only the older one is checked: a hit retires as a new entry too, so re-submitting y first
    // would evict x before x is checked.
    EXPECT_EQ(cached('x', "prompt x, re-submit after y"), kLen - 1)
        << "x and y are the only " << kEntries << " entries, so x must still hit; 0 means the cache holds fewer "
        << "than prefix_cache_entries finished sequences";
}

