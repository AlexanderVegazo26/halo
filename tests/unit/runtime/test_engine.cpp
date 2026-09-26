// Milestone 3: the CPU Engine on the tiny model — golden greedy output, the caller-thread
// callback guarantee, 4 concurrent == 4 sequential, stop / cancel semantics, prefix cache,
// MTP-KV fallback, the per-tick cost model, request validation, sampled + structured output.
//
// Kept small for ASan (each tick is a full 248K-vocab head pass on the tiny model).

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <future>
#include <mutex>
#include <thread>

#include "../models/tiny_golden.h"
#include "halo/core/error.h"
#include "halo/runtime/cpu_engine.h"
#include "halo/sampling/structured.h"
#include "halo/template/chat_template.h"
#include "halo/tokenizer/tokenizer.h"

#if defined(HALO_TEST_HIP)
#include "halo/backend/hip_backend.h"
#endif

using halo::runtime::CpuEngineOptions;
using halo::runtime::Engine;
using halo::runtime::EngineConfig;
using halo::runtime::FinishReason;
using halo::runtime::GenerateRequest;
using halo::runtime::GenerateResult;
using halo::runtime::TickInfo;
using halo::runtime::TokenEvent;
using halo::speculative::GateMode;
using halo::test::Golden;

namespace {

using Toks = std::vector<std::int32_t>;

Toks prefix(const Toks& t, std::size_t n) { return Toks(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(std::min(n, t.size()))); }

std::filesystem::path tiny_f32() { return halo::test::tiny_dir() / "tiny-f32.gguf"; }

EngineConfig base_cfg() {
    EngineConfig c;
    c.model_path = tiny_f32().string();
    c.max_context = 512;
    c.max_sequences = 4;
    c.mtp_max_draft = 2;
    c.prefix_cache = false;
    return c;
}

std::unique_ptr<Engine> make(CpuEngineOptions o = {}, EngineConfig c = base_cfg()) {
    return halo::runtime::create_cpu_engine(c, o);
}

GenerateRequest greedy(Toks prompt, std::size_t n) {
    GenerateRequest r;
    r.prompt = std::move(prompt);
    r.sampling.temperature = 0.0f;
    r.max_tokens = n;
    return r;
}

struct Collected {
    std::vector<TokenEvent> events;
    std::vector<std::thread::id> threads;
};

GenerateResult run(Engine& e, const GenerateRequest& r, Collected* c = nullptr) {
    return e.generate(r, [c](const TokenEvent& ev) {
        if (c != nullptr) {
            c->events.push_back(ev);
            c->threads.push_back(std::this_thread::get_id());
        }
        return true;
    });
}

class EngineTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        golden_ = Golden::load();
        if (!golden_ || !std::filesystem::exists(tiny_f32())) return;
        CpuEngineOptions o;
        o.gate_mode = GateMode::Always;  // exercise the MTP path in every greedy test
        engine_ = make(o);
    }
    static void TearDownTestSuite() { engine_.reset(); }
    void SetUp() override {
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        if (!engine_) GTEST_SKIP() << "tiny-f32.gguf missing under " << halo::test::tiny_dir();
    }
    static Toks prompt(const char* p) { return golden_->i32(std::string(p) + ".tokens"); }
    static Toks gold(const char* p) { return golden_->i32(std::string(p) + ".decode_tokens"); }

    static inline std::optional<Golden> golden_;
    static inline std::unique_ptr<Engine> engine_;
};

}  // namespace

// ---- creation ----------------------------------------------------------------------------

TEST_F(EngineTest, CreationErrorsAreTyped) {
    EngineConfig c = base_cfg();
    c.backend = "vulkan";
    // BI-6: "vulkan" is a real backend -- the engine constructs when a Vulkan device is
    // available (RADV on the EVO-X2, lavapipe on the dev host). Any failure must still be a
    // typed halo::Error (Device when no ICD/device, Unsupported when the adapter is not
    // compiled in), never an untyped exception.
    try {
        (void)make({}, c);
    } catch (const halo::Error& e) {
        EXPECT_TRUE(e.code() == halo::ErrorCode::Device || e.code() == halo::ErrorCode::Unsupported) << e.what();
    }
    c = base_cfg();
    c.backend = "not-a-backend";
    try {
        (void)make({}, c);
        ADD_FAILURE() << "expected Unsupported";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
    }
    c = base_cfg();
    c.model_path = (halo::test::tiny_dir() / "does-not-exist.gguf").string();
    EXPECT_THROW((void)make({}, c), halo::Error);
    c = base_cfg();
    c.max_memory_bytes = 1u << 20;  // the planner must refuse: weights alone are ~540 MB
    try {
        (void)make({}, c);
        ADD_FAILURE() << "expected MEMORY_ERROR";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Memory) << e.what();
    }
}

TEST_F(EngineTest, ModelInfoTokenizerAndTemplateComeFromTheGguf) {
    const auto& m = engine_->model();
    EXPECT_EQ(m.architecture, "qwen35");
    EXPECT_EQ(m.vocab_size, 248320u);
    EXPECT_TRUE(m.has_mtp);
    EXPECT_EQ(m.context_length, 512u);
    EXPECT_FALSE(m.id.empty());
    EXPECT_EQ(engine_->tokenizer().eos(), 248046);
    const halo::chat::OrderedJson msgs = halo::chat::OrderedJson::array({{{"role", "user"}, {"content", "hello there"}}});
    const std::string p = engine_->chat_template().apply(msgs, nullptr, {});
    EXPECT_NE(p.find("hello there"), std::string::npos) << p;
    EXPECT_NE(p.find("<|im_start|>assistant"), std::string::npos) << p;
}

// ---- greedy correctness -------------------------------------------------------------------

TEST_F(EngineTest, GreedyOutputMatchesTransformersGoldenWithMtp) {
    Collected c;
    const Toks g = gold("p0");
    const GenerateResult r = run(*engine_, greedy(prompt("p0"), g.size()), &c);
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_EQ(r.tokens, g);
    EXPECT_EQ(r.prompt_tokens, 150u);
    EXPECT_GT(r.draft_tokens, 0u) << "the MTP path must have run";
    ASSERT_EQ(c.events.size(), g.size());
    std::string text;
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_EQ(c.events[i].token, g[i]);
        EXPECT_FALSE(c.events[i].is_eos);
        text += c.events[i].piece;
    }
    EXPECT_EQ(text, engine_->tokenizer().decode(g, false)) << "pieces concatenate to the decoded text";
    EXPECT_GT(engine_->stats().speculative_attempts, 0u);
}

TEST_F(EngineTest, GreedyWithMtpDisabledIsIdentical) {
    EngineConfig c = base_cfg();
    c.mtp_enabled = false;
    CpuEngineOptions o;
    std::vector<TickInfo> ticks;
    std::mutex mu;
    o.on_tick = [&](const TickInfo& t) {
        const std::lock_guard lk(mu);
        ticks.push_back(t);
    };
    auto e = make(o, c);
    const GenerateResult r = run(*e, greedy(prompt("p1"), 10));
    const GenerateResult m = run(*engine_, greedy(prompt("p1"), 10));
    EXPECT_EQ(r.draft_tokens, 0u);
    EXPECT_GT(m.draft_tokens, 0u);
    EXPECT_EQ(r.tokens, m.tokens);
    EXPECT_EQ(r.tokens, prefix(gold("p1"), 10));
    const GenerateResult r0 = run(*e, greedy(prompt("p0"), 4));  // 149 catch-up pairs would flush if MTP ran
    EXPECT_EQ(r0.tokens, prefix(gold("p0"), 4));
    ASSERT_FALSE(ticks.empty());
    for (const TickInfo& t : ticks) {
        EXPECT_EQ(t.weight_passes, t.trunk_passes) << "MTP disabled: no MTP weight pass (catch-up or draft) may run";
    }
}

// ---- GPU backends (BI-6) --------------------------------------------------------------
// The engine over a GPU backend (Vulkan device; HIP host emulation with the bitwise
// profile) must produce the same greedy tokens as the CPU engine / transformers golden.
// KV/GDN state is still host memory imported per forward call (WS-BI-2), so this exercises
// dispatch and correctness, not performance.

TEST_F(EngineTest, VulkanBackendProducesTheSameGreedyOutput) {
#if defined(HALO_TEST_VULKAN)
    EngineConfig c = base_cfg();
    c.backend = "vulkan";
    c.mtp_enabled = false;  // dispatch + correctness only; MTP over GPU is exercised later
    std::unique_ptr<Engine> e;
    try {
        e = make({}, c);
    } catch (const halo::Error& x) {
        GTEST_SKIP() << "no Vulkan device in this environment: " << x.what();
    }
    const GenerateResult r = run(*e, greedy(prompt("p0"), 8));
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_EQ(r.tokens, prefix(gold("p0"), 8));
    EXPECT_EQ(e->stats().threads, 0u) << "no thread pool on a GPU backend";
#else
    GTEST_SKIP() << "built without the Vulkan backend";
#endif
}

TEST_F(EngineTest, HipEmulationBackendProducesTheSameGreedyOutput) {
#if defined(HALO_TEST_HIP)
    CpuEngineOptions o;
    o.backend_factory = [] {
        halo::backend::HipBackendOptions ho;
        ho.mode = halo::backend::HipMode::Emulation;
        ho.defaults = halo::backend::hip_bitwise_defaults();  // bit-identical to halo::cpu
        return halo::backend::make_hip_backend(ho);
    };
    EngineConfig c = base_cfg();
    c.backend = "hip";
    c.mtp_enabled = false;
    auto e = make(o, c);
    const GenerateResult r = run(*e, greedy(prompt("p0"), 8));
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_EQ(r.tokens, prefix(gold("p0"), 8));  // bitwise emulation: exact golden match
#else
    GTEST_SKIP() << "built without the HIP backend";
#endif
}

// ---- threading ---------------------------------------------------------------------------

TEST_F(EngineTest, CallbacksRunOnTheCallerThreadAndASlowReaderDoesNotStallOthers) {
    std::atomic<bool> b_done{false};
    std::atomic<bool> a_saw_b_done{false};
    Collected ca, cb;
    std::thread::id a_thread, b_thread;
    auto fa = std::async(std::launch::async, [&] {
        a_thread = std::this_thread::get_id();
        bool first = true;
        return engine_->generate(greedy(prompt("p0"), 6), [&](const TokenEvent& ev) {
            ca.events.push_back(ev);
            ca.threads.push_back(std::this_thread::get_id());
            if (first) {
                first = false;
                // A "slow SSE reader": block until B has finished (bounded wait).
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(300);
                while (!b_done.load() && std::chrono::steady_clock::now() < until) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                a_saw_b_done = b_done.load();
            }
            return true;
        });
    });
    // B starts once A has its first token (A is blocked in its callback from then on).
    while (ca.events.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto fb = std::async(std::launch::async, [&] {
        b_thread = std::this_thread::get_id();
        GenerateResult r = run(*engine_, greedy(prompt("p1"), 8), &cb);
        b_done = true;
        return r;
    });
    const GenerateResult rb = fb.get();
    const GenerateResult ra = fa.get();
    EXPECT_TRUE(a_saw_b_done) << "B could not finish while A's callback blocked: callbacks stall the batch";
    EXPECT_EQ(ra.tokens, prefix(gold("p0"), 6));
    EXPECT_EQ(rb.tokens, prefix(gold("p1"), 8));
    for (const auto& t : ca.threads) EXPECT_EQ(t, a_thread);
    for (const auto& t : cb.threads) EXPECT_EQ(t, b_thread);
}

TEST_F(EngineTest, FourConcurrentRequestsEqualFourSequential) {
    const Toks p0 = prompt("p0");
    std::vector<GenerateRequest> reqs = {
        greedy(Toks(p0.begin(), p0.begin() + 40), 8),
        greedy(prompt("p1"), 8),
        greedy(prompt("p2"), 8),
        greedy(Toks(p0.end() - 30, p0.end()), 8),
    };
    reqs[3].sampling.temperature = 0.9f;  // one seeded sampled request in the mix
    reqs[3].sampling.top_k = 40;
    reqs[3].sampling.seed = 11;
    std::vector<GenerateResult> seq;
    const auto t0 = engine_->stats().ticks;
    for (const auto& r : reqs) seq.push_back(run(*engine_, r));
    const auto seq_ticks = engine_->stats().ticks - t0;
    std::vector<GenerateResult> con(4);
    std::barrier start(4);
    std::vector<std::thread> th;
    const auto t1 = engine_->stats().ticks;
    for (std::size_t i = 0; i < 4; ++i) {
        th.emplace_back([&, i] {
            start.arrive_and_wait();
            con[i] = run(*engine_, reqs[i]);
        });
    }
    for (auto& t : th) t.join();
    const auto con_ticks = engine_->stats().ticks - t1;
    for (std::size_t i = 0; i < 4; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(con[i].tokens, seq[i].tokens);
        EXPECT_EQ(con[i].finish, FinishReason::Length);
        EXPECT_EQ(con[i].tokens.size(), 8u);
    }
    EXPECT_LT(con_ticks, seq_ticks) << "concurrent requests must share ticks (one batched forward per tick)";
    std::printf("ticks: sequential %llu, concurrent %llu\n", static_cast<unsigned long long>(seq_ticks),
                static_cast<unsigned long long>(con_ticks));
}

// ---- stop conditions ---------------------------------------------------------------------

TEST_F(EngineTest, StopConditionsEosStopSequencesStopStringsAndLength) {
    const Toks g = run(*engine_, greedy(prompt("p1"), 8)).tokens;
    ASSERT_EQ(g.size(), 8u);
    // EOS (test seam: g[4] plays EOS).
    CpuEngineOptions o;
    o.eos_token = g[4];
    auto e = make(o);
    Collected c;
    GenerateResult r = run(*e, greedy(prompt("p1"), 8), &c);
    const std::size_t eos_at = static_cast<std::size_t>(std::find(g.begin(), g.end(), g[4]) - g.begin());
    EXPECT_EQ(r.finish, FinishReason::Stop);
    EXPECT_EQ(r.tokens, Toks(g.begin(), g.begin() + static_cast<std::ptrdiff_t>(eos_at + 1)));
    ASSERT_FALSE(c.events.empty());
    EXPECT_TRUE(c.events.back().is_eos);
    EXPECT_TRUE(c.events.back().piece.empty());
    // Stop token sequence.
    GenerateRequest q = greedy(prompt("p1"), 8);
    q.stop_token_seqs = {{g[2], g[3]}};
    r = run(*engine_, q);
    EXPECT_EQ(r.finish, FinishReason::Stop);
    EXPECT_EQ(r.tokens, Toks(g.begin(), g.begin() + 4));
    // Stop string: the decoded text of g[2]; generation ends at the first token completing it.
    const std::string s = engine_->tokenizer().decode(Toks{g[2]}, false);
    if (!s.empty()) {
        std::size_t end = 0;
        while (engine_->tokenizer().decode(Toks(g.begin(), g.begin() + static_cast<std::ptrdiff_t>(end + 1)), false).find(s) ==
               std::string::npos) {
            ++end;
        }
        GenerateRequest q2 = greedy(prompt("p1"), 8);
        q2.stop_strings = {s};
        r = run(*engine_, q2);
        EXPECT_EQ(r.finish, FinishReason::Stop);
        EXPECT_EQ(r.tokens, Toks(g.begin(), g.begin() + static_cast<std::ptrdiff_t>(end + 1)));
    }
    // Length, including max_tokens capped by the context.
    r = run(*engine_, greedy(prompt("p1"), 3));
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_EQ(r.tokens.size(), 3u);
    Toks long_prompt(509, g[0]);
    r = run(*engine_, greedy(long_prompt, 50));
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_EQ(r.tokens.size(), 3u) << "509 + 3 == max_context 512";
}

// ---- cancellation ------------------------------------------------------------------------

TEST_F(EngineTest, CancellationDeliversExactlyTheAcceptedTokensAndLeavesTheEngineUsable) {
    const Toks g = gold("p1");
    std::size_t calls = 0;
    const auto gen0 = engine_->stats().tokens_generated;
    GenerateResult r = engine_->generate(greedy(prompt("p1"), 200), [&](const TokenEvent&) { return ++calls < 3; });
    EXPECT_EQ(r.finish, FinishReason::Cancelled);
    EXPECT_EQ(calls, 3u) << "no callback after it returned false";
    EXPECT_EQ(r.tokens, Toks(g.begin(), g.begin() + 3));
    // The worker retired the sequence at its next tick instead of decoding all 200 tokens
    // (it can be at most a few ticks ahead of the caller: each tick is a full head pass).
    EXPECT_LT(engine_->stats().tokens_generated - gen0, 40u) << "cancellation did not stop the worker";
    calls = 0;
    r = engine_->generate(greedy(prompt("p0"), 10), [&](const TokenEvent&) { return ++calls < 1; });
    EXPECT_EQ(r.tokens.size(), 1u);
    EXPECT_THROW((void)engine_->generate(greedy(prompt("p2"), 5),
                                         [](const TokenEvent&) -> bool { throw std::runtime_error("client gone"); }),
                 std::runtime_error);
    // The engine is unaffected.
    EXPECT_EQ(run(*engine_, greedy(prompt("p1"), 10)).tokens, Toks(g.begin(), g.begin() + 10));
    EXPECT_EQ(engine_->stats().active_sequences, 0u);
}

// ---- prefix cache (D-013) ------------------------------------------------------------------

TEST_F(EngineTest, PrefixCacheReusesCheckpointsAndMatchesRecompute) {
    EngineConfig c = base_cfg();
    c.prefix_cache = true;
    CpuEngineOptions o;
    o.gate_mode = GateMode::Always;
    o.prefill_chunk = 64;
    auto e = make(o, c);
    auto plain = make(o);  // prefix cache off
    const Toks p0 = prompt("p0");
    const GenerateResult r1 = run(*e, greedy(p0, 6));
    EXPECT_EQ(r1.cached_prompt_tokens, 0u);
    EXPECT_EQ(r1.tokens, prefix(gold("p0"), 6));
    // Exact re-submit: the N-1 checkpoint of r1's prefill; bit-identical path.
    const GenerateResult r2 = run(*e, greedy(p0, 6));
    EXPECT_EQ(r2.cached_prompt_tokens, 149u);
    EXPECT_EQ(r2.tokens, r1.tokens);
    // Multi-turn: prompt = p0 + r2's output + more. r2 retired with p0 + 5 fed tokens.
    Toks p3 = p0;
    p3.insert(p3.end(), r2.tokens.begin(), r2.tokens.end());
    const Toks p1 = prompt("p1");
    p3.insert(p3.end(), p1.begin(), p1.end());
    const GenerateResult r3 = run(*e, greedy(p3, 6));
    EXPECT_EQ(r3.cached_prompt_tokens, 155u) << "retirement checkpoint at all fed tokens";
    // Restored rows equal a fresh computation within fp32 tolerance (different chunking),
    // not bitwise; the greedy tokens must still agree on this input.
    EXPECT_EQ(r3.tokens, run(*plain, greedy(p3, 6)).tokens);
    const auto st = e->stats();
    EXPECT_EQ(st.prefix_cache_hits, 2u);
    EXPECT_EQ(st.prefix_cache_misses, 1u);
    EXPECT_EQ(st.prefix_cache_reused_tokens, 149u + 155u);
    // A prompt sharing only a short prefix below any checkpoint: miss, recompute.
    const GenerateResult r4 = run(*e, greedy(Toks(p0.begin(), p0.begin() + 20), 3));
    EXPECT_EQ(r4.cached_prompt_tokens, 0u);
    EXPECT_EQ(r4.tokens, run(*plain, greedy(Toks(p0.begin(), p0.begin() + 20), 3)).tokens);
}

TEST_F(EngineTest, PrefixCacheMatchesAcrossMultipleDifferentFirstTokens) {
    // L5: admit()'s LCP scan is bucketed by the query's first token. Retire entries under two
    // different first tokens, then confirm a query matches the right one and a query sharing
    // neither entry's first token is a clean miss -- not a crash and not a wrong match.
    EngineConfig c = base_cfg();
    c.prefix_cache = true;
    auto e = make({}, c);
    const Toks p0 = prompt("p0");
    const Toks p1 = prompt("p1");
    ASSERT_NE(p0[0], p1[0]) << "the two conversations must land in different first-token buckets";
    (void)run(*e, greedy(p0, 4));
    (void)run(*e, greedy(p1, 4));
    // Re-submitting p1 exactly must hit p1's checkpoint, not p0's (different bucket).
    const GenerateResult r = run(*e, greedy(p1, 4));
    EXPECT_GT(r.cached_prompt_tokens, 0u);
    // A query whose first token matches neither cached entry is a clean miss.
    Toks other(p0.begin() + 1, p0.end());
    ASSERT_NE(other[0], p0[0]);
    ASSERT_NE(other[0], p1[0]);
    const GenerateResult miss = run(*e, greedy(other, 2));
    EXPECT_EQ(miss.cached_prompt_tokens, 0u);
}

TEST_F(EngineTest, CheckpointHintsAreTakenAndReused) {
    EngineConfig c = base_cfg();
    c.prefix_cache = true;
    c.max_sequences = 1;
    auto e = make({}, c);
    const Toks p0 = prompt("p0");
    GenerateRequest a = greedy(Toks(p0.begin(), p0.begin() + 120), 2);
    a.checkpoint_hints = {100, 5000 /* out of range: ignored */};
    (void)run(*e, a);
    // A different continuation after token 100: the hint checkpoint serves it.
    Toks b = Toks(p0.begin(), p0.begin() + 100);
    b.insert(b.end(), p0.end() - 10, p0.end());
    const GenerateResult rb = run(*e, greedy(b, 2));
    EXPECT_EQ(rb.cached_prompt_tokens, 100u);
}

TEST_F(EngineTest, KvPoolPressureEvictsTheCacheInsteadOfFailing) {
    EngineConfig c = base_cfg();
    c.prefix_cache = true;
    c.max_sequences = 1;
    CpuEngineOptions o;
    o.kv_blocks = 14;  // 224 rows: one 150-token sequence, cached or active, not two
    o.prefix_cache_entries = 4;
    auto e = make(o, c);
    const Toks p0 = prompt("p0");
    const GenerateResult r1 = run(*e, greedy(p0, 4));
    const Toks other(p0.begin() + 1, p0.end());  // LCP 0 with p0: a miss that needs the blocks
    const GenerateResult r2 = run(*e, greedy(other, 4));
    EXPECT_EQ(r2.finish, FinishReason::Length);
    auto plain = make();
    EXPECT_EQ(r2.tokens, run(*plain, greedy(other, 4)).tokens);
    EXPECT_EQ(r1.tokens.size(), 4u);
}

// ---- MTP-KV exhaustion falls back to k = 0 (M3 constraint 2) -------------------------------------

TEST_F(EngineTest, MtpKvExhaustionFallsBackToPlainDecodeWithIdenticalOutput) {
    CpuEngineOptions o;
    o.gate_mode = GateMode::Always;
    o.mtp_kv_blocks = 10;  // 160 MTP rows: p0's catch-up (149) + a few drafts, then exhausted
    std::vector<TickInfo> ticks;
    std::mutex mu;
    o.on_tick = [&](const TickInfo& t) {
        const std::lock_guard lk(mu);
        ticks.push_back(t);
    };
    auto e = make(o);
    const GenerateResult r = run(*e, greedy(prompt("p0"), 16));
    auto plain_cfg = base_cfg();
    plain_cfg.mtp_enabled = false;
    auto plain = make({}, plain_cfg);
    EXPECT_EQ(r.tokens, run(*plain, greedy(prompt("p0"), 16)).tokens);
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_GE(e->stats().mtp_fallbacks, 1u);
    EXPECT_TRUE(std::any_of(ticks.begin(), ticks.end(), [](const TickInfo& t) { return t.mtp_fallback; }));
    EXPECT_GT(r.draft_tokens, 0u) << "MTP ran before the pool ran out";
    // A new request re-attaches MTP for its sequence.
    const GenerateResult r2 = run(*e, greedy(prompt("p1"), 6));
    EXPECT_GT(r2.draft_tokens, 0u);
}

// ---- cost model: one weight pass per tick; rollback costs no extra pass (D-012 / D-014) ------------

TEST_F(EngineTest, EveryTickIsOneTrunkPassAndRollbackNeverAddsOne) {
    CpuEngineOptions o;
    o.gate_mode = GateMode::Always;
    std::vector<TickInfo> ticks;
    std::mutex mu;
    o.on_tick = [&](const TickInfo& t) {
        const std::lock_guard lk(mu);
        ticks.push_back(t);
    };
    auto e = make(o);
    std::thread other([&] { (void)run(*e, greedy(prompt("p2"), 6)); });
    (void)run(*e, greedy(prompt("p1"), 8));
    other.join();
    ASSERT_FALSE(ticks.empty());
    EXPECT_TRUE(std::any_of(ticks.begin(), ticks.end(), [](const TickInfo& t) { return t.sequences >= 2; }))
        << "the two requests never shared a tick: the one-pass-per-tick check would be vacuous";
    std::size_t rollback_ticks = 0;
    for (const TickInfo& t : ticks) {
        EXPECT_EQ(t.trunk_passes, 1u) << "one batched trunk forward per tick";
        if (t.prefill_rows == 0) {
            EXPECT_EQ(t.weight_passes, 1u + t.max_draft) << "decode tick: 1 trunk + 1 MTP pass per draft depth";
            // Review N-4 / R-5(c): the pass counters are call counts; the byte figure is the
            // measured one. A decode tick reads every matrix once per call plus the heads,
            // i.e. the cost model minus the (tiny) norm / conv vectors.
            EXPECT_LE(t.measured_weight_bytes, t.predicted_bytes);
            EXPECT_GE(static_cast<double>(t.measured_weight_bytes), 0.99 * static_cast<double>(t.predicted_bytes))
                << "measured " << t.measured_weight_bytes << " vs predicted " << t.predicted_bytes;
        }
        if (t.accepted < t.drafted) ++rollback_ticks;
    }
    EXPECT_GT(rollback_ticks, 0u) << "rejected drafts were rolled back";
    const auto s = e->stats();
    EXPECT_GT(s.ticks, 0u);
    EXPECT_GT(s.last_tick_predicted_bytes, 0u);
}

// ---- validation, sampling, structured output ----------------------------------------------

TEST_F(EngineTest, InvalidRequestsThrowApiErrorsBeforeScheduling) {
    const auto api = [&](const GenerateRequest& r) {
        try {
            (void)engine_->generate(r, {});
            ADD_FAILURE() << "expected Error(Api)";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Api) << e.what();
        }
    };
    api(greedy({}, 4));
    api(greedy({1, 248320}, 4));
    api(greedy({1, -1}, 4));
    api(greedy(Toks(512, 1), 4));
    api(greedy({1, 2}, 0));
    GenerateRequest r = greedy({1, 2}, 4);
    r.sampling.top_p = 2.0f;
    api(r);
    r = greedy({1, 2}, 4);
    r.stop_token_seqs = {{}};
    api(r);
    EXPECT_EQ(engine_->stats().active_sequences, 0u);
}

TEST_F(EngineTest, SeededSamplingIsReproducibleAndJsonObjectIsConstrained) {
    GenerateRequest r = greedy(prompt("p1"), 6);
    r.sampling.temperature = 0.8f;
    r.sampling.seed = 1234;
    const GenerateResult a = run(*engine_, r);
    const GenerateResult b = run(*engine_, r);
    EXPECT_EQ(a.tokens, b.tokens);
    EXPECT_EQ(a.tokens.size(), 6u);
    EXPECT_EQ(a.draft_tokens, 0u) << "sampled requests decode without MTP (no stochastic acceptance)";
    GenerateRequest j = greedy(prompt("p1"), 12);
    j.sampling.json_object = true;
    const GenerateResult rj = run(*engine_, j);
    Toks body;
    for (const std::int32_t t : rj.tokens) {
        if (t != engine_->tokenizer().eos()) body.push_back(t);
    }
    const std::string text = engine_->tokenizer().decode(body, false);
    ASSERT_FALSE(text.empty()) << "no output";
    // Every byte so far must be a valid prefix of a JSON object (the grammar advanced with
    // each accepted token), not merely start with '{'.
    halo::sampling::ByteMatcher bm(halo::sampling::Grammar::any_json_object());
    EXPECT_TRUE(bm.accept_bytes(text)) << text;
}

// ---- accepted drafts: engine bookkeeping + prefix cache (forced oracle drafts) -------------
// Real MTP drafts are never accepted on the tiny model; the draft hook feeds the plain greedy
// continuation with one wrong token at a rotating depth so every acceptance count occurs.

TEST_F(EngineTest, AcceptedDraftsKeepTokensKvAndPrefixCacheConsistent) {
    const Toks p1 = prompt("p1");
    auto plain = make();  // cache off; real MTP (never accepted): reference tokens
    const Toks g1 = run(*plain, greedy(p1, 30)).tokens;
    Toks p3 = p1;  // multi-turn prompt, built after the first request below
    Toks g3;
    std::size_t step = 0;
    const auto oracle = [&](std::span<const std::int32_t> prompt_, std::span<const std::int32_t> gen) -> Toks {
        const Toks& g = prompt_.size() == p1.size() ? g1 : g3;
        const std::size_t m = gen.size() - 1;  // pending = g[m]
        const std::size_t wrong = step++ % 3;  // k = 2: wrong at depth 0, 1, or none
        Toks d;
        for (std::size_t i = 0; i < 2 && m + 1 + i < g.size(); ++i) d.push_back(g[m + 1 + i]);
        if (wrong < d.size()) d[wrong] = (d[wrong] + 1) % 248320;
        return d;
    };
    EngineConfig c = base_cfg();
    c.prefix_cache = true;
    CpuEngineOptions o;
    o.gate_mode = GateMode::Always;
    o.draft_hook = oracle;
    std::vector<TickInfo> ticks;
    std::mutex mu;
    o.on_tick = [&](const TickInfo& t) {
        const std::lock_guard lk(mu);
        ticks.push_back(t);
    };
    auto e = make(o, c);
    const GenerateResult r1 = run(*e, greedy(p1, 12));
    EXPECT_EQ(r1.tokens, prefix(g1, 12));
    std::vector<std::size_t> seen(3, 0);
    for (const TickInfo& t : ticks) {
        if (t.drafted > 0) ++seen[t.accepted];
    }
    for (std::size_t a = 0; a < 3; ++a) EXPECT_GT(seen[a], 0u) << "acceptance " << a << " never happened";
    EXPECT_GT(r1.accepted_draft_tokens, 0u);
    // Multi-turn: the retirement checkpoint covers every fed token (all but the last emitted).
    p3.insert(p3.end(), r1.tokens.begin(), r1.tokens.end());
    const Toks p2 = prompt("p2");
    p3.insert(p3.end(), p2.begin(), p2.end());
    g3 = run(*plain, greedy(p3, 10)).tokens;
    step = 0;
    const GenerateResult r3 = run(*e, greedy(p3, 10));
    EXPECT_EQ(r3.cached_prompt_tokens, p1.size() + r1.tokens.size() - 1);
    EXPECT_EQ(r3.tokens, g3) << "restored rows (from a sequence with accepted drafts) vs recompute";
    EXPECT_GT(r3.accepted_draft_tokens, 0u);
}
