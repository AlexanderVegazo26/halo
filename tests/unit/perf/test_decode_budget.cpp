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
//   * Batched decode (a fourth engine, max_sequences = 4): four concurrent requests share one
//     trunk forward per tick, tick by tick. Structural, so it runs in every lane; no timing:
//     on this tiny F32 model per-sequence work dominates (4 batched requests measured only
//     ~1.14x faster than 4 sequential ones).
//   * Batched throughput (a fifth engine, a wider model with Q6_K matmul weights, timing builds
//     only): 4 concurrent requests finish >= 1.5x faster than the same 4 in a row (dev host
//     2.32x), because a batched tick dequantizes each weight block once for every row. Catches
//     a forward per sequence that still reports one trunk pass, which BatchedDecode passes.
//   * Speculative decode (a sixth engine, the model keeps its embedded MTP block, gate Always):
//     greedy decoding drafts with the MTP, verifies, commits and keeps the MTP KV in step with
//     the trunk (Speculator checks that invariant every tick), and an MTP KV pool too small
//     for the request falls back to plain decoding without changing the tokens. Structural,
//     every lane; the Spec.* engine tests in tests/unit/runtime need HALO_REF_DIR.
//
// Dev-host baseline (2026-11-05, RelWithDebInfo clang-21, no -march, one thread): decode
// ~0.033 ms/token at 16 tokens of context, TTFT 1024-token prompt ~28 ms, context scaling
// 768 -> 3072 ~2.1x; batched throughput (2026-11-07, same build) 4 x 32 tokens 286 ms in a
// row vs 123 ms concurrent = 2.32x. Timings are NOT a HALO performance claim (DECISIONS.md D-001).
// Debug and sanitizer builds run one short generation (the functional checks) and skip the
// timing checks with a reason. ctest runs this RUN_SERIAL (label `perf`).
// Rerun: ctest --test-dir build -R test_decode_budget --output-on-failure

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
using halo::SamplingParams;

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
// byte-level vocabulary; token_embd / output grow to the new vocabulary size. `width` / `ffn`
// widen the model (BatchedThroughput); the defaults are tiny_qwen35's E=64, FF=128.
// `mtp` keeps tiny_qwen35's embedded MTP block (SpeculativeDecode).
// `echo` (F32 only) makes the trunk and the MTP repeat the token they are given, for 'A'..'Z'
// (kEchoFirst + i): those tokens get a one-hot embedding (dimension i) that output.weight reads
// back, and eh_proj passes the embedding half of the MTP input through. Every layer stays zero,
// so the residual stream carries the embedding unchanged: a sequence prompted with 'B' emits 'B'
// and drafts 'B', and a draft routed to another sequence is rejected (BatchedSpeculativeDecode).
// Any other token has a zero embedding and still decodes to id 0.
constexpr std::int32_t kEchoFirst = 'A';
constexpr std::int32_t kEchoCount = 26;
std::vector<std::byte> synthetic_model(std::int64_t width = 64, std::int64_t ffn = 128,
                                       halo::DType matmul_type = halo::DType::F32, bool mtp = false,
                                       bool echo = false) {
    using namespace halo::test;
    GgufSpec s = tiny_qwen35({.mtp = mtp, .embd = width, .ffn = ffn});
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
    if (echo) {
        EXPECT_EQ(matmul_type, halo::DType::F32) << "echo weights are written as F32 values";
        EXPECT_GE(embd, kEchoCount) << "echo needs one embedding dimension per echo token";
        std::vector<float> rows(static_cast<std::size_t>(embd * vocab), 0.0f);
        for (std::int32_t i = 0; i < kEchoCount && i < embd; ++i) {
            rows[static_cast<std::size_t>((kEchoFirst + i) * embd + i)] = 1.0f;
        }
        s.find("token_embd.weight")->values = rows;
        s.find("output.weight")->values = std::move(rows);
        if (mtp) {
            // eh_proj is {2E, E}: output row r reads concat(enorm(embedding), hnorm(hidden)), so
            // the identity on the first E columns forwards the drafted-from token alone.
            std::vector<float> eh(static_cast<std::size_t>(2 * embd * embd), 0.0f);
            for (std::int64_t r = 0; r < embd; ++r) eh[static_cast<std::size_t>(r * 2 * embd + r)] = 1.0f;
            s.find("blk.4.nextn.eh_proj.weight")->values = std::move(eh);
        }
    }
    // Every matmul weight (2-D, rows a whole number of `matmul_type` blocks) except the embedding
    // table takes `matmul_type`; zero blocks dequantize to zero weights whatever the type.
    for (auto& t : s.tensors) {
        if (t.ne.size() == 2 && t.ne[0] % halo::traits(matmul_type).block_elems == 0 && t.name != "token_embd.weight") {
            t.type = static_cast<std::uint32_t>(matmul_type);
        }
    }
    return s.build(true);
}

// Writes the synthetic model to a per-process temp file named after `tag`.
std::filesystem::path write_synthetic_model(const std::string& tag, std::int64_t width = 64, std::int64_t ffn = 128,
                                            halo::DType matmul_type = halo::DType::F32, bool mtp = false,
                                            bool echo = false) {
    const auto path =
        std::filesystem::temp_directory_path() / ("halo-" + tag + "-" + std::to_string(::getpid()) + ".gguf");
    const auto bytes = synthetic_model(width, ffn, matmul_type, mtp, echo);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    EXPECT_TRUE(f.good()) << "could not write the synthetic model to " << path;
    return path;
}

// Single-threaded, one sequence unless `max_sequences` says otherwise, no MTP.
std::unique_ptr<halo::runtime::Engine> make_engine(const std::filesystem::path& path, bool prefix_cache,
                                                   const halo::runtime::CpuEngineOptions& opts = {},
                                                   std::size_t max_sequences = 1) {
    EngineConfig c;
    c.model_path = path.string();
    c.max_context = kMaxContext;
    c.max_context_explicit = true;
    c.max_sequences = max_sequences;
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

namespace {

// One request through the engine's sampler path (anything but plain greedy), every token checked.
GenerateResult generate_sampled(halo::runtime::Engine& engine, const SamplingParams& sp, std::size_t max_tokens,
                                const std::string& ctx) {
    GenerateRequest r;
    r.prompt.assign(16, kPromptToken);
    r.sampling = sp;
    r.max_tokens = max_tokens;
    GenerateResult res = engine.generate(r, [](const TokenEvent&) { return true; });
    EXPECT_EQ(res.finish, FinishReason::Length) << ctx << ": error='" << res.error << "'";
    EXPECT_EQ(res.tokens.size(), max_tokens) << ctx;
    for (std::size_t i = 0; i < res.tokens.size(); ++i) {
        EXPECT_TRUE(res.tokens[i] >= 0 && res.tokens[i] < 256) << ctx << ": token " << i << " = " << res.tokens[i]
                                                               << ", expected a byte token (uniform logits, no EOS)";
    }
    return res;
}

}  // namespace

TEST_F(DecodeBudget, PenalizedGreedyRequestSamplesWithTheGrowingHistory) {
    // A presence penalty takes the request off the fast greedy path onto its sampler. The
    // logits are all zero, so the argmax is the lowest id not yet in the history: 0, 1, 2, ...
    // only if every emitted token reaches the sampler's history before the next step.
    SamplingParams sp;
    sp.temperature = 0.0f;
    sp.presence_penalty = 1.0f;
    sp.penalty_last_n = -1;
    const GenerateResult res = generate_sampled(*engine_, sp, 8, "greedy + presence_penalty 1");
    const std::vector<std::int32_t> want{0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(res.tokens, want) << "greedy with a presence penalty over uniform logits must pick the lowest unseen id "
                                   "each step (all zeros = generated tokens never reach the sampler's history)";
}

TEST_F(DecodeBudget, SeededSampledRequestIsReproducible) {
    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.seed = 42;
    const GenerateResult a = generate_sampled(*engine_, sp, 8, "temperature 1, seed 42, run 1");
    const GenerateResult b = generate_sampled(*engine_, sp, 8, "temperature 1, seed 42, run 2");
    EXPECT_EQ(a.tokens, b.tokens) << "the same seed must sample the same tokens";
    EXPECT_NE(a.tokens, std::vector<std::int32_t>(8, 0))
        << "temperature 1 over 259 equal logits drew id 0 eight times: the request took the greedy path";
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
    // The ratio is the median of 5 back-to-back (768, 3072) pairs, not best-of-N of all small
    // runs vs best-of-N of all large ones: a pair runs within ~30 ms, so a burst of host load
    // or a move to a slower core (hybrid CPU, powersave governor) usually hits both sides of a
    // pair, and the median drops a pair it splits. Measured as best of 3 each, back to back,
    // the ratio ranged 0.7-6.9 over 20 runs on the loaded dev host (2 over the limit) although
    // per-token work is linear.
    (void)generate(768, 128);
    (void)generate(3072, 128);
    constexpr int kPairs = 5;
    std::vector<double> ratios;
    double small = 1e300, large = 1e300;
    for (int i = 0; i < kPairs; ++i) {
        const double s_ms = 1000.0 / generate(768, 128).decode_tps;
        const double l_ms = 1000.0 / generate(3072, 128).decode_tps;
        ratios.push_back(l_ms / s_ms);
        small = std::min(small, s_ms);
        large = std::min(large, l_ms);
    }
    std::sort(ratios.begin(), ratios.end());
    const double ratio = ratios[kPairs / 2];
    std::printf("[perf] %-46s %9.4f -> %9.4f ms/token  median pair ratio %5.2f (limit 4.0)\n",
                "decode context 768 -> 3072", small, large, ratio);
    ::testing::Test::RecordProperty("decode context 768 -> 3072 ratio", std::to_string(ratio));
    EXPECT_LE(ratio, 4.0) << "decode ms/token after a 3072-token prompt (best " << large << ") vs a 768-token prompt "
                          << "(best " << small << ") grew " << ratio << "x (median of " << kPairs << " pairs)"
                          << " for 4x the context; only attention depends on the context and is O(context) per "
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

TEST_F(PrefixCache, FollowUpTurnReusesThePreviousPromptAndReply) {
    // A chat client re-sends the whole conversation each turn: previous prompt + the reply + a
    // new user message. The previous prompt's state and the reply's must come from the cache:
    // retirement checkpoints everything turn 1 fed (its prompt + every reply token but the last,
    // which was sampled and never fed), so only that last reply token and the new message are
    // computed again.
    const Prompt turn1 = prompt_of(512, 'q');
    const GenerateResult r1 = run(turn1, 8, "turn 1, 512-token prompt");
    EXPECT_EQ(r1.cached_prompt_tokens, 0u) << "turn 1: nothing cached starts with this token yet";
    Prompt turn2 = turn1;
    turn2.insert(turn2.end(), r1.tokens.begin(), r1.tokens.end());
    turn2.insert(turn2.end(), 64, 'r');
    const GenerateResult r2 = run(turn2, 4, "turn 2 = turn 1 prompt + reply + 64 new tokens");
    const std::size_t turn1_fed = turn1.size() + r1.tokens.size() - 1;
    EXPECT_GE(r2.cached_prompt_tokens, turn1_fed)
        << "turn 2 (" << turn2.size() << " tokens) should reuse turn 1's prompt (" << turn1.size()
        << ") and its reply minus the last, never-fed token (" << r1.tokens.size() - 1 << "), " << turn1_fed
        << " tokens, from the checkpoint taken at retirement; it reused " << r2.cached_prompt_tokens
        << (r2.cached_prompt_tokens + 1 >= turn1.size() && r2.cached_prompt_tokens < turn1_fed
                ? " (only the prompt: the retirement checkpoint is missing or cut short, so every turn recomputes the "
                  "previous reply)"
                : "");
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


namespace {

// A fourth engine with four sequence slots (the API server's default max_sequences), to pin
// that concurrent requests share one batched forward per tick (cpu_engine.h "Scheduling").
// Every other test here runs one sequence at a time, so a scheduler that served sequences
// round-robin (one forward per sequence per tick, or one sequence per tick) would pass them.
//
// The arrival of concurrent requests is made deterministic with the before_step fault hook:
// the first request's first step waits until the other three are queued, so they are all
// admitted together at the next tick. Then, with prompts of kPrompt tokens (one prefill
// chunk, which also samples the first token) and kTokens tokens each:
//   tick 1:              A prefills                       (1 sequence)
//   tick 2:              A decodes, B C D prefill          (4 sequences, one trunk pass)
//   ticks 3 .. kTokens:  A B C D decode                   (4 decode rows, one trunk pass)
//   tick kTokens + 1:    B C D decode their last token    (A finished at tick kTokens)
class BatchedDecode : public ::testing::Test {
protected:
    static constexpr std::size_t kSlots = 4;
    static constexpr std::size_t kPrompt = 16;
    static constexpr std::size_t kTokens = 64;
    static constexpr auto kWaitLimit = std::chrono::seconds(60);

    static void SetUpTestSuite() {
        path_ = write_synthetic_model("batched-decode");
        halo::runtime::CpuEngineOptions o;
        o.faults.before_step = [](std::span<const halo::speculative::StepRequest>) {
            if (!gate_armed_.exchange(false)) return;
            worker_waiting_ = true;
            const auto deadline = std::chrono::steady_clock::now() + kWaitLimit;
            while (!gate_open_ && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        };
        o.on_tick = [](const halo::runtime::TickInfo& t) {
            const std::lock_guard lk(ticks_mu_);
            ticks_.push_back(t);
        };
        engine_ = make_engine(path_, /*prefix_cache=*/false, o, kSlots);
    }
    static void TearDownTestSuite() {
        engine_.reset();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    void SetUp() override { ASSERT_NE(engine_, nullptr) << "engine creation on the synthetic model failed"; }

    // Polls `done` (no sleeps) until it holds or kWaitLimit passes; false on timeout.
    template <class F>
    static bool wait_for(F done) {
        const auto deadline = std::chrono::steady_clock::now() + kWaitLimit;
        while (!done()) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        return true;
    }

    static GenerateResult run_one(std::int32_t token) {
        GenerateRequest r;
        r.prompt.assign(kPrompt, token);
        r.sampling.temperature = 0.0f;
        r.max_tokens = kTokens;
        return engine_->generate(r, [](const TokenEvent&) { return true; });
    }

    struct Batch {
        std::vector<GenerateResult> results;
        std::vector<halo::runtime::TickInfo> ticks;  // the ticks of this batch only
        double wall_ms = 0;
    };

    // kSlots requests that arrive together (see the class comment). Fatal failures leave
    // `results` short; the caller checks its size.
    static Batch run_batch() {
        Batch b;
        const std::uint64_t ticks_before = engine_->stats().ticks;
        std::size_t recorded_before = 0;
        {
            const std::lock_guard lk(ticks_mu_);
            recorded_before = ticks_.size();
        }
        worker_waiting_ = false;
        gate_open_ = false;
        gate_armed_ = true;
        std::vector<GenerateResult> results(kSlots);
        std::vector<std::thread> threads;
        const auto t0 = std::chrono::steady_clock::now();
        threads.emplace_back([&] { results[0] = run_one('A'); });
        const bool first_waiting = wait_for([] { return worker_waiting_.load(); });
        for (std::size_t i = 1; i < kSlots; ++i) {
            threads.emplace_back([&, i] { results[i] = run_one(static_cast<std::int32_t>('A' + i)); });
        }
        const bool all_queued = wait_for([] { return engine_->stats().queued_requests == kSlots - 1; });
        gate_open_ = true;
        for (auto& t : threads) t.join();
        b.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        EXPECT_TRUE(first_waiting) << "the first request's first step never reached the before_step gate";
        EXPECT_TRUE(all_queued) << "the other " << kSlots - 1 << " requests were not all queued while the first "
                                << "step waited (queued_requests=" << engine_->stats().queued_requests << ")";
        // on_tick runs after the tick that delivered the last token, possibly after generate() returned.
        const std::uint64_t ticks_after = engine_->stats().ticks;
        const bool all_recorded = wait_for([&] {
            const std::lock_guard lk(ticks_mu_);
            return ticks_.size() - recorded_before >= ticks_after - ticks_before;
        });
        EXPECT_TRUE(all_recorded) << "on_tick did not report every tick the engine counted";
        const std::lock_guard lk(ticks_mu_);
        b.ticks.assign(ticks_.begin() + static_cast<std::ptrdiff_t>(recorded_before), ticks_.end());
        b.results = std::move(results);
        return b;
    }

    static inline std::filesystem::path path_;
    static inline std::unique_ptr<halo::runtime::Engine> engine_;
    static inline std::atomic<bool> gate_armed_{false};
    static inline std::atomic<bool> gate_open_{false};
    static inline std::atomic<bool> worker_waiting_{false};
    static inline std::mutex ticks_mu_;
    static inline std::vector<halo::runtime::TickInfo> ticks_;
};

}  // namespace

TEST_F(BatchedDecode, ConcurrentRequestsShareOneTrunkPassPerTick) {
    const Batch b = run_batch();
    ASSERT_EQ(b.results.size(), kSlots);
    for (std::size_t i = 0; i < kSlots; ++i) {
        const GenerateResult& r = b.results[i];
        EXPECT_EQ(r.finish, FinishReason::Length) << "request " << i << ": error='" << r.error << "'";
        EXPECT_EQ(r.tokens.size(), kTokens) << "request " << i;
        EXPECT_EQ(r.tokens, b.results[0].tokens) << "request " << i << ": all-zero weights must generate the same "
                                                 << "tokens whether or not a sequence shares its tick with others";
    }
    std::string trace;  // the first 8 and the last 3 ticks
    for (std::size_t i = 0; i < b.ticks.size(); ++i) {
        if (i == 8 && b.ticks.size() > 11) trace += " ...";
        if (i >= 8 && i + 3 < b.ticks.size()) continue;
        const auto& t = b.ticks[i];
        trace += " [" + std::to_string(i + 1) + ": seq " + std::to_string(t.sequences) + ", prefill " +
                 std::to_string(t.prefill_rows) + ", decode " + std::to_string(t.decode_rows) + ", trunk " +
                 std::to_string(t.trunk_passes) + "]";
    }
    ASSERT_EQ(b.ticks.size(), kTokens + 1)
        << kSlots << " requests of " << kTokens << " tokens, the first one tick ahead of the rest, should take "
        << kTokens + 1
        << " ticks when every tick serves every active sequence; " << kSlots * kTokens
        << " means one sequence per tick (no batching). Ticks:" << trace;
    std::size_t full = 0;
    for (std::size_t i = 0; i < b.ticks.size(); ++i) {
        const auto& t = b.ticks[i];
        const std::string at = "tick " + std::to_string(i + 1) + " of " + std::to_string(b.ticks.size());
        EXPECT_EQ(t.trunk_passes, 1u) << at << ": one batched trunk forward per tick, whatever the number of "
                                      << "sequences (" << t.sequences << "); more means a forward per sequence";
        const std::size_t expect_seq = i == 0 ? 1 : i == kTokens ? kSlots - 1 : kSlots;
        EXPECT_EQ(t.sequences, expect_seq) << at;
        const std::size_t expect_prefill = i == 0 ? kPrompt : i == 1 ? (kSlots - 1) * kPrompt : 0;
        EXPECT_EQ(t.prefill_rows, expect_prefill) << at << ": prompts of " << kPrompt << " tokens are one chunk";
        const std::size_t expect_decode = i == 0 ? 0 : i == 1 ? 1 : expect_seq;
        EXPECT_EQ(t.decode_rows, expect_decode) << at << ": one decode row per decoding sequence (no MTP)";
        full += t.sequences == kSlots ? 1 : 0;
    }
    EXPECT_EQ(full, kTokens - 1) << "ticks serving all " << kSlots << " sequences at once";
}

namespace {

// A fifth engine on a wider synthetic model (E=512, FF=1536: ~44 MB of F32 weights, so a
// decode step is dominated by streaming the weights once) with four sequence slots. Four
// requests that run together must finish clearly faster than the same four one after another:
// a batched tick reads every weight once for all decode rows. BatchedDecode pins the schedule
// (one trunk pass per tick, as the engine reports it); this pins that the shared pass is real
// in wall time, so a forward per sequence that still reported one pass would fail here. On
// tiny_qwen35 (E=64) per-sequence work dominates and batching measured only ~1.14x.
//
// Why Q6_K, E=256, FF=1024: with F32 weights a single-threaded decode is compute-bound (the
// dev host's 64 MiB L3 holds them), so 4 rows cost 4x and batching measured 1.00x even at
// E=512. Quantized weights are dequantized once per block for every row (cpu::matmul), so
// that cost is shared: measured Q8_0 1.36x, Q4_K 1.45x, Q6_K 2.32-2.35x (E=256 and E=512).
// Mutant check (2026-11-07): Speculative::step calling model_->forward once per sequence and
// reporting one pass measured 1.00x here while BatchedDecode stayed green.
//
// The ratio is the median of kPairs back-to-back (sequential, concurrent) pairs, the pattern
// that keeps DecodeCostIsAtMostLinearInContext stable on the loaded dev host. Timing builds
// only: the functional side of batching is BatchedDecode, which runs in every lane.
class BatchedThroughput : public ::testing::Test {
protected:
    static constexpr std::size_t kSlots = 4;
    static constexpr std::size_t kPrompt = 16;
    static constexpr std::size_t kTokens = 32;
    static constexpr int kPairs = 5;
    static constexpr double kMinSpeedup = 1.5;

    void SetUp() override {
        if (const char* why = skip_reason()) GTEST_SKIP() << why;
        path_ = write_synthetic_model("batched-throughput", 256, 1024, halo::DType::Q6_K);
        engine_ = make_engine(path_, /*prefix_cache=*/false, {}, kSlots);
        ASSERT_NE(engine_, nullptr) << "engine creation on the wide synthetic model failed";
    }
    void TearDown() override {
        engine_.reset();
        std::error_code ec;
        if (!path_.empty()) std::filesystem::remove(path_, ec);
    }

    GenerateResult run_one(std::int32_t token) {
        GenerateRequest r;
        r.prompt.assign(kPrompt, token);
        r.sampling.temperature = 0.0f;
        r.max_tokens = kTokens;
        return engine_->generate(r, [](const TokenEvent&) { return true; });
    }

    void expect_complete(const GenerateResult& r, const std::string& ctx) {
        EXPECT_EQ(r.finish, FinishReason::Length) << ctx << ": error='" << r.error << "'";
        EXPECT_EQ(r.tokens.size(), kTokens) << ctx;
    }

    // Wall time of kSlots requests, one after another.
    double sequential_ms() {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kSlots; ++i) {
            expect_complete(run_one(static_cast<std::int32_t>('a' + i)), "sequential request " + std::to_string(i));
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // Wall time of kSlots requests submitted together from their own threads.
    double concurrent_ms() {
        std::vector<GenerateResult> results(kSlots);
        std::vector<std::thread> threads;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kSlots; ++i) {
            threads.emplace_back([&, i] { results[i] = run_one(static_cast<std::int32_t>('a' + i)); });
        }
        for (auto& t : threads) t.join();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        for (std::size_t i = 0; i < kSlots; ++i) expect_complete(results[i], "concurrent request " + std::to_string(i));
        return ms;
    }

    std::filesystem::path path_;
    std::unique_ptr<halo::runtime::Engine> engine_;
};

}  // namespace

TEST_F(BatchedThroughput, FourConcurrentRequestsFinishFasterThanFourInARow) {
    (void)sequential_ms();  // warm-up: pages in the weights, sizes the KV pool
    (void)concurrent_ms();
    std::vector<double> speedups;
    double best_seq = 1e300, best_conc = 1e300;
    for (int i = 0; i < kPairs; ++i) {
        const double seq = sequential_ms();
        const double conc = concurrent_ms();
        speedups.push_back(seq / conc);
        best_seq = std::min(best_seq, seq);
        best_conc = std::min(best_conc, conc);
    }
    std::sort(speedups.begin(), speedups.end());
    const double speedup = speedups[kPairs / 2];
    std::printf("[perf] %-46s %9.2f -> %9.2f ms  median pair speedup %5.2f (min %.1f)\n",
                "4 x 32 tokens, in a row -> concurrent", best_seq, best_conc, speedup, kMinSpeedup);
    ::testing::Test::RecordProperty("batched speedup 4 sequences", std::to_string(speedup));
    EXPECT_GE(speedup, kMinSpeedup)
        << kSlots << " concurrent requests of " << kTokens << " tokens (best " << best_conc << " ms) were only "
        << speedup << "x faster than the same " << kSlots << " in a row (best " << best_seq << " ms), median of "
        << kPairs << " pairs; a batched tick streams the weights once for every decode row, so less than "
        << kMinSpeedup << "x means each sequence pays its own weight pass (a forward per sequence) or the "
        << "scheduler serialises the requests";
}

namespace {

// The synthetic model with its embedded MTP block (all-zero weights: the trunk and the MTP both
// predict token 0, so every draft is accepted and the output equals plain greedy decoding).
class SpeculativeDecode : public ::testing::Test {
protected:
    static constexpr std::size_t kPrompt = 16;
    static constexpr int kMaxDraft = 2;

    static void SetUpTestSuite() { path_ = write_synthetic_model("speculative", 64, 128, halo::DType::F32, /*mtp=*/true); }
    static void TearDownTestSuite() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    struct Run {
        GenerateResult res;
        halo::runtime::EngineStats stats;
        std::size_t ticks_drafting = 0;
    };

    // One greedy request on a fresh MTP engine (gate Always: no profit measurement in the way).
    static Run run(std::size_t max_tokens, std::optional<std::size_t> mtp_kv_blocks = std::nullopt) {
        EngineConfig c;
        c.model_path = path_.string();
        c.max_context = kMaxContext;
        c.max_context_explicit = true;
        c.max_sequences = 1;
        c.threads = 1;
        c.mtp_enabled = true;
        c.mtp_max_draft = kMaxDraft;
        halo::runtime::CpuEngineOptions o;
        o.gate_mode = halo::speculative::GateMode::Always;
        o.mtp_kv_blocks = mtp_kv_blocks;
        Run out;
        std::atomic<std::size_t> drafting{0};
        o.on_tick = [&](const halo::runtime::TickInfo& t) {
            if (t.drafted > 0) drafting.fetch_add(1);
        };
        auto engine = halo::runtime::create_cpu_engine(c, o);
        EXPECT_NE(engine, nullptr) << "MTP engine creation on the synthetic model failed";
        if (engine == nullptr) return out;
        EXPECT_TRUE(engine->model().has_mtp) << "tiny_qwen35({.mtp = true}) embeds an MTP block";
        GenerateRequest r;
        r.prompt.assign(kPrompt, kPromptToken);
        r.sampling.temperature = 0.0f;
        r.max_tokens = max_tokens;
        out.res = engine->generate(r, [](const TokenEvent&) { return true; });
        out.stats = engine->stats();
        engine.reset();  // joins the worker: every on_tick has run
        out.ticks_drafting = drafting.load();
        return out;
    }

    static void expect_plain_greedy_output(const GenerateResult& res, std::size_t max_tokens, const std::string& ctx) {
        EXPECT_EQ(res.finish, FinishReason::Length) << ctx << ": error='" << res.error << "'";
        ASSERT_EQ(res.tokens.size(), max_tokens) << ctx;
        for (std::size_t i = 0; i < res.tokens.size(); ++i) {
            EXPECT_EQ(res.tokens[i], 0) << ctx << ", token " << i
                                        << ": speculative decoding must emit what plain greedy does (id 0)";
        }
    }

    static inline std::filesystem::path path_;
};

}  // namespace

TEST_F(SpeculativeDecode, MtpDraftsAreAcceptedAndTheOutputEqualsPlainGreedy) {
    constexpr std::size_t kTokens = 24;
    const Run r = run(kTokens);
    expect_plain_greedy_output(r.res, kTokens, "MTP on, gate Always");
    EXPECT_GT(r.res.draft_tokens, 0u) << "the MTP never drafted: speculative decoding did not run";
    EXPECT_EQ(r.res.accepted_draft_tokens, r.res.draft_tokens)
        << "all-zero weights: the MTP draft (0) always equals the trunk's token, so every draft is accepted";
    EXPECT_GT(r.ticks_drafting, 1u) << "drafts in more than one tick: the MTP state carried over between ticks";
    // Accepted drafts emit several tokens per tick: 1 prefill tick + fewer decode ticks than tokens.
    EXPECT_LT(r.stats.ticks, 1 + kTokens) << "ticks " << r.stats.ticks << " for " << kTokens
                                          << " tokens: accepted drafts should save decode ticks";
    EXPECT_EQ(r.stats.mtp_fallbacks, 0u) << "the default MTP KV pool holds this request";
}

TEST_F(SpeculativeDecode, ExhaustedMtpKvFallsBackToPlainDecodingWithTheSameTokens) {
    // 2 blocks x 16 tokens of MTP KV: the 16-token prompt fits, the drafts run out of room.
    constexpr std::size_t kTokens = 40;
    const Run r = run(kTokens, /*mtp_kv_blocks=*/2);
    expect_plain_greedy_output(r.res, kTokens, "MTP KV of 2 blocks");
    EXPECT_GE(r.stats.mtp_fallbacks, 1u) << "a 32-token MTP KV cannot hold 16 + " << kTokens
                                         << " tokens: the tick should retry without MTP";
    EXPECT_GT(r.res.draft_tokens, 0u) << "the MTP drafted before its KV ran out";
}

namespace {

// Batched MTP: kSlots concurrent requests on the echo MTP synthetic model, admitted together
// (the first request's first step waits on before_step until the others are queued, as in
// BatchedDecode). Request i is prompted with 'A' + i, so its trunk emits and its MTP drafts
// 'A' + i only: a draft handed to the wrong sequence is a rejected draft. The single-sequence
// SpeculativeDecode tests can't see a speculator that drafts for one sequence only, mixes up
// draft rows between sequences, or runs one MTP forward per sequence instead of one batched
// call per draft depth.
class BatchedSpeculativeDecode : public ::testing::Test {
protected:
    static constexpr std::size_t kSlots = 3;
    static constexpr std::size_t kPrompt = 16;
    static constexpr std::size_t kTokens = 24;
    static constexpr int kMaxDraft = 2;
    static constexpr auto kWaitLimit = std::chrono::seconds(60);

    static void SetUpTestSuite() {
        path_ = write_synthetic_model("batched-speculative", 64, 128, halo::DType::F32, /*mtp=*/true, /*echo=*/true);
    }
    static void TearDownTestSuite() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    template <class F>
    static bool wait_for(F done) {
        const auto deadline = std::chrono::steady_clock::now() + kWaitLimit;
        while (!done()) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        return true;
    }

    struct Batch {
        std::vector<GenerateResult> results;
        std::vector<halo::runtime::TickInfo> ticks;
    };

    static Batch run_batch() {
        std::atomic<bool> gate_armed{true}, gate_open{false}, worker_waiting{false};
        std::mutex ticks_mu;
        Batch b;
        EngineConfig c;
        c.model_path = path_.string();
        c.max_context = kMaxContext;
        c.max_context_explicit = true;
        c.max_sequences = kSlots;
        c.threads = 1;
        c.mtp_enabled = true;
        c.mtp_max_draft = kMaxDraft;
        halo::runtime::CpuEngineOptions o;
        o.gate_mode = halo::speculative::GateMode::Always;
        o.faults.before_step = [&](std::span<const halo::speculative::StepRequest>) {
            if (!gate_armed.exchange(false)) return;
            worker_waiting = true;
            const auto deadline = std::chrono::steady_clock::now() + kWaitLimit;
            while (!gate_open && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        };
        o.on_tick = [&](const halo::runtime::TickInfo& t) {
            const std::lock_guard lk(ticks_mu);
            b.ticks.push_back(t);
        };
        auto engine = halo::runtime::create_cpu_engine(c, o);
        EXPECT_NE(engine, nullptr) << "MTP engine creation on the synthetic model failed";
        if (engine == nullptr) return b;
        auto run_one = [&](std::int32_t token) {
            GenerateRequest r;
            r.prompt.assign(kPrompt, token);
            r.sampling.temperature = 0.0f;
            r.max_tokens = kTokens;
            return engine->generate(r, [](const TokenEvent&) { return true; });
        };
        std::vector<GenerateResult> results(kSlots);
        std::vector<std::thread> threads;
        threads.emplace_back([&] { results[0] = run_one(kEchoFirst); });
        const bool first_waiting = wait_for([&] { return worker_waiting.load(); });
        for (std::size_t i = 1; i < kSlots; ++i) {
            threads.emplace_back([&, i] { results[i] = run_one(kEchoFirst + static_cast<std::int32_t>(i)); });
        }
        const bool all_queued = wait_for([&] { return engine->stats().queued_requests == kSlots - 1; });
        gate_open = true;
        for (auto& t : threads) t.join();
        engine.reset();  // joins the worker: every on_tick has run
        EXPECT_TRUE(first_waiting) << "the first request's first step never reached the before_step gate";
        EXPECT_TRUE(all_queued) << "the other " << kSlots - 1 << " requests were not all queued while the first "
                                << "step waited";
        b.results = std::move(results);
        return b;
    }

    static inline std::filesystem::path path_;
};

}  // namespace

TEST_F(BatchedSpeculativeDecode, ConcurrentMtpSequencesDraftInOneMtpForwardPerDepth) {
    const Batch b = run_batch();
    ASSERT_EQ(b.results.size(), kSlots);
    for (std::size_t i = 0; i < kSlots; ++i) {
        const GenerateResult& r = b.results[i];
        const std::int32_t echo = kEchoFirst + static_cast<std::int32_t>(i);
        const std::string ctx = "request " + std::to_string(i) + " (prompt token " + std::to_string(echo) + ")";
        EXPECT_EQ(r.finish, FinishReason::Length) << ctx << ": error='" << r.error << "'";
        ASSERT_EQ(r.tokens.size(), kTokens) << ctx;
        for (std::size_t t = 0; t < r.tokens.size(); ++t) {
            EXPECT_EQ(r.tokens[t], echo) << ctx << ", token " << t
                                         << ": the echo model's plain greedy output repeats the prompt token";
        }
        EXPECT_GT(r.draft_tokens, 0u) << ctx << ": the MTP never drafted for this sequence";
        EXPECT_EQ(r.accepted_draft_tokens, r.draft_tokens)
            << ctx << ": the echo MTP drafts this sequence's own token, which the trunk always accepts; a "
            << "rejected draft was routed from (or computed with) another sequence's token or hidden state";
    }
    std::size_t shared_draft_ticks = 0;
    std::string trace;
    for (std::size_t i = 0; i < b.ticks.size(); ++i) {
        const auto& t = b.ticks[i];
        trace += " [" + std::to_string(i + 1) + ": seq " + std::to_string(t.sequences) + ", decode " +
                 std::to_string(t.decode_rows) + ", drafted " + std::to_string(t.drafted) + ", max_draft " +
                 std::to_string(t.max_draft) + ", passes " + std::to_string(t.weight_passes) + "/trunk " +
                 std::to_string(t.trunk_passes) + "]";
        EXPECT_EQ(t.trunk_passes, 1u) << "tick " << i + 1 << ": one batched trunk forward per tick";
        EXPECT_FALSE(t.mtp_fallback) << "tick " << i + 1 << ": the default MTP KV pool holds " << kSlots
                                     << " sequences of " << kPrompt + kTokens << " tokens";
        EXPECT_EQ(t.drafts.size(), t.drafted) << "tick " << i + 1;
        if (t.drafted == 0) continue;
        EXPECT_EQ(t.accepted, t.drafted) << "tick " << i + 1 << " (" << t.sequences
                                         << " sequences): every echo draft is accepted unless it reached the "
                                            "wrong sequence";
        EXPECT_LE(t.weight_passes, 1u + t.max_draft)
            << "tick " << i + 1 << " (" << t.sequences << " sequences, drafted " << t.drafted
            << "): one trunk forward + at most one MTP forward per draft depth; more means the speculator ran "
               "a separate MTP forward per sequence";
        if (t.sequences == kSlots && t.drafted >= kSlots) ++shared_draft_ticks;
    }
    EXPECT_GE(shared_draft_ticks, 2u) << "the " << kSlots << " sequences should draft together in the same ticks; ticks:"
                                      << trace;
}
