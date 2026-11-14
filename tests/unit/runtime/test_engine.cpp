// Milestone 3: the CPU Engine on the tiny model — golden greedy output, the caller-thread
// callback guarantee, 4 concurrent == 4 sequential, stop / cancel semantics, prefix cache,
// MTP-KV fallback, the per-tick cost model, request validation, sampled + structured output.
//
// Kept small for ASan (each tick is a full 248K-vocab head pass on the tiny model).

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <future>
#include <mutex>
#include <span>
#include <thread>
#include <utility>

#include "../models/tiny_golden.h"
#include "halo/core/error.h"
#include "halo/runtime/cpu_engine.h"
#include "halo/sampling/structured.h"
#include "halo/speculative/ngram.h"
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

// ---- KV pool limits: fail loudly, split per layer ----------------------------------------------

TEST_F(EngineTest, ExplicitContextThatCannotBeHonouredIsATypedConfigError) {
    const auto expect_config = [](EngineConfig c, CpuEngineOptions o, const char* contains) {
        try {
            (void)make(o, c);
            ADD_FAILURE() << "expected Error(Config) (" << contains << ")";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Config) << e.what();
            EXPECT_NE(std::string(e.what()).find(contains), std::string::npos) << e.what();
        }
    };
    // Larger than the model's trained context: refused, not silently reduced (a defaulted value is
    // reduced -- the same request without the explicit flag is not an error on this check).
    EngineConfig c = base_cfg();
    c.max_context = std::size_t{1} << 33;  // > the 2^31 ceiling model.cpp accepts for a trained context
    c.max_context_explicit = true;
    expect_config(c, {}, "trained context");
    // The KV pool cannot hold the requested sequences x context under the backend's per-buffer
    // cap even split per layer (test seam: a 1-byte cap): an explicit --ctx is an error ...
    c = base_cfg();
    c.max_context_explicit = true;
    CpuEngineOptions o;
    o.kv_buffer_cap_bytes = 1;
    expect_config(c, o, "--ctx");
    // ... and so is a defaulted one that cannot be reduced any further (here already <= 1024).
    c.max_context_explicit = false;
    expect_config(c, o, "KV blocks");
}

TEST_F(EngineTest, PerLayerKvPoolGivesTheSameTokensAsTheContiguousPool) {
    EngineConfig c = base_cfg();
    c.max_sequences = 1;
    c.mtp_enabled = false;  // one pool, so its byte size is exactly the contiguous image's
    CpuEngineOptions o;
    o.kv_blocks = 64;
    const auto plain = make(o, c);
    ASSERT_EQ(plain->model().kv_segments, 1u);
    const std::uint64_t pool_bytes = plain->model().kv_pool_bytes;
    ASSERT_GT(pool_bytes, 0u);
    // One byte below the contiguous image: it must be split per layer to be allowed at all.
    o.kv_buffer_cap_bytes = pool_bytes - 1;
    std::unique_ptr<Engine> split;
    try {
        split = make(o, c);
    } catch (const halo::Error& e) {
        // A model with a single attention layer cannot be split: still a typed, explicit refusal.
        EXPECT_EQ(e.code(), halo::ErrorCode::Config) << e.what();
        GTEST_SKIP() << "the tiny model has one attention layer, so a per-layer split does not shrink the buffer: " << e.what();
    }
    ASSERT_GT(split->model().kv_segments, 1u) << "the cap forced a split, so the pool must have several segments";
    EXPECT_EQ(split->model().kv_pool_bytes, pool_bytes) << "splitting changes the buffers, not the pool size";
    const Toks g = gold("p1");
    const Toks expect(g.begin(), g.begin() + 10);
    EXPECT_EQ(run(*plain, greedy(prompt("p1"), 10)).tokens, expect);
    EXPECT_EQ(run(*split, greedy(prompt("p1"), 10)).tokens, expect);
    // A prefix-cache re-submit exercises copy-on-write on the split pool too.
    EngineConfig cc = c;
    cc.prefix_cache = true;
    const auto cached = make(o, cc);
    ASSERT_GT(cached->model().kv_segments, 1u);
    EXPECT_EQ(run(*cached, greedy(prompt("p1"), 10)).tokens, expect);
    const GenerateResult again = run(*cached, greedy(prompt("p1"), 10));
    EXPECT_EQ(again.tokens, expect);
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
    EXPECT_EQ(r.mtp_fallbacks, static_cast<std::size_t>(std::ranges::count_if(ticks, [](const TickInfo& t) { return t.mtp_fallback; })))
        << "the request alone: its fallbacks are the ticks that fell back";
    EXPECT_EQ(r.mtp_fallbacks, e->stats().mtp_fallbacks);
    EXPECT_GT(r.draft_tokens, 0u) << "MTP ran before the pool ran out";
    // A new request re-attaches MTP for its sequence and counts only its own fallbacks.
    const std::uint64_t fb0 = e->stats().mtp_fallbacks;
    const GenerateResult r2 = run(*e, greedy(prompt("p1"), 6));
    EXPECT_GT(r2.draft_tokens, 0u);
    EXPECT_EQ(r2.mtp_fallbacks, e->stats().mtp_fallbacks - fb0);
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

// ---- forced and prompt-lookup (n-gram) drafts on the GPU (next.md step 11) -----------------
// CpuEngineOptions::ngram_draft (HALO_NGRAM_DRAFT=1) drafts the tokens that followed an earlier
// occurrence of the context suffix (speculative::ngram_draft, unit-tested in test_ngram.cpp)
// instead of running the MTP head; the step then verifies them as forced drafts, the same path
// as draft_hook. The tiny model never repeats a 3-gram (no lookup hit in 300 greedy tokens from
// p0/p1/p2), so the GPU checks are: forced drafts (an oracle hook, wrong at a rotating depth) are
// verified and accepted over Vulkan exactly as greedy decoding needs, alone and batched with
// MTP-drafted and sampled sequences; and n-gram drafting on, with no match, is bitwise
// MTP-only drafting (same drafts, tokens and ticks) while it does search every greedy step.

namespace {

class VulkanEngine : public EngineTest {
protected:
    struct Built {
        std::unique_ptr<Engine> engine;
        std::vector<TickInfo> ticks;
        std::mutex mu;
    };
    /// Vulkan engine, MTP drafting k = 3 with the gate forced on. Null = no Vulkan device.
    static std::unique_ptr<Built> build(std::optional<bool> ngram,
                                        CpuEngineOptions::DraftHook hook = {}) {
#if defined(HALO_TEST_VULKAN)
        auto b = std::make_unique<Built>();
        EngineConfig c = base_cfg();
        c.backend = "vulkan";
        c.mtp_max_draft = 3;
        CpuEngineOptions o;
        o.gate_mode = GateMode::Always;
        o.ngram_draft = ngram;
        o.draft_hook = std::move(hook);
        Built* raw = b.get();
        o.on_tick = [raw](const TickInfo& t) {
            const std::lock_guard lk(raw->mu);
            raw->ticks.push_back(t);
        };
        try {
            b->engine = make(o, c);
        } catch (const halo::Error&) {
            return nullptr;
        }
        return b;
#else
        (void)ngram;
        (void)hook;
        return nullptr;
#endif
    }
    static std::size_t decode_ticks(const Built& b) {
        std::size_t n = 0;
        for (const TickInfo& t : b.ticks) n += t.prefill_rows == 0 ? 1 : 0;
        return n;
    }
};

}  // namespace

TEST_F(VulkanEngine, ForcedDraftsAreVerifiedAndAcceptedBitwise) {
    auto plain = build(false);
    if (!plain) GTEST_SKIP() << "no Vulkan device (or built without the Vulkan backend)";
    const Toks p1 = prompt("p1");
    const GenerateResult ref = run(*plain->engine, greedy(p1, 40));
    ASSERT_EQ(ref.tokens.size(), 40u);
    ASSERT_EQ(ref.accepted_draft_tokens, 0u) << "control: tiny-model MTP drafts are never accepted";
    // Oracle: the plain greedy continuation, one token wrong at depth 0, 1, 2 or none (rotating),
    // so every acceptance count 0..3 occurs. Other prompts get no forced drafts (MTP drafts).
    std::size_t step = 0;
    auto hooked = build(false, [&](std::span<const std::int32_t> pr, std::span<const std::int32_t> gen) -> Toks {
        if (!std::ranges::equal(pr, p1)) return {};
        const std::size_t m = gen.size() - 1;  // pending = ref[m]
        const std::size_t wrong = step++ % 4;
        Toks d;
        for (std::size_t i = 0; i < 3 && m + 1 + i < ref.tokens.size(); ++i) d.push_back(ref.tokens[m + 1 + i]);
        if (wrong < d.size()) d[wrong] = (d[wrong] + 1) % 248320;
        return d;
    });
    ASSERT_TRUE(hooked);
    const GenerateResult got = run(*hooked->engine, greedy(p1, 40));
    EXPECT_EQ(got.tokens, ref.tokens) << "forced drafts changed greedy output on Vulkan";
    EXPECT_GT(got.accepted_draft_tokens, 0u);
    std::vector<std::size_t> seen(4, 0);
    for (const TickInfo& t : hooked->ticks) {
        if (t.prefill_rows != 0) continue;
        EXPECT_EQ(t.trunk_passes, 1u) << "one trunk forward per tick";
        if (t.drafted > 0) ++seen[t.accepted];
    }
    for (std::size_t a = 0; a < seen.size(); ++a) EXPECT_GT(seen[a], 0u) << "acceptance " << a << " never happened";
    EXPECT_LT(decode_ticks(*hooked), decode_ticks(*plain)) << "accepted drafts must save decode ticks";

    // Batched: the forced-draft sequence shares ticks with an MTP-drafted greedy sequence and a
    // sampled one; each must equal its result on the plain engine alone.
    const Toks p0 = prompt("p0");
    std::vector<GenerateRequest> reqs = {greedy(p1, 40), greedy(prompt("p2"), 16),
                                         greedy(Toks(p0.end() - 30, p0.end()), 8)};
    reqs[2].sampling.temperature = 0.9f;
    reqs[2].sampling.top_k = 40;
    reqs[2].sampling.seed = 7;
    std::vector<GenerateResult> alone;
    for (const auto& r : reqs) alone.push_back(run(*plain->engine, r));
    step = 0;
    hooked->ticks.clear();
    std::vector<GenerateResult> con(reqs.size());
    std::barrier start(static_cast<std::ptrdiff_t>(reqs.size()));
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        th.emplace_back([&, i] {
            start.arrive_and_wait();
            con[i] = run(*hooked->engine, reqs[i]);
        });
    }
    for (auto& t : th) t.join();
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        EXPECT_EQ(con[i].tokens, alone[i].tokens) << "request " << i << " batched with forced drafts vs alone";
    }
    EXPECT_GT(con[0].accepted_draft_tokens, 0u);
    std::size_t shared = 0;
    for (const TickInfo& t : hooked->ticks) shared += (t.prefill_rows == 0 && t.sequences > 1 && t.drafted > 3) ? 1 : 0;
    EXPECT_GT(shared, 0u) << "no decode tick verified forced and MTP drafts of two sequences together";
}

TEST_F(VulkanEngine, NgramDraftingWithoutAMatchIsBitwiseMtpDrafting) {
    auto off = build(false);
    auto on = build(true);
    if (!off || !on) GTEST_SKIP() << "no Vulkan device (or built without the Vulkan backend)";
    for (const char* name : {"p0", "p1", "p2"}) {
        SCOPED_TRACE(name);
        const Toks p = prompt(name);
        off->ticks.clear();
        on->ticks.clear();
        const auto s0 = on->engine->stats();
        const GenerateResult a = run(*off->engine, greedy(p, 24));
        const GenerateResult b = run(*on->engine, greedy(p, 24));
        const auto s1 = on->engine->stats();
        // Premise: no context suffix recurs, at any step of this run.
        for (std::size_t j = 1; j <= a.tokens.size(); ++j) {
            const std::span<const std::int32_t> gen(a.tokens.data(), j);
            ASSERT_TRUE(halo::speculative::ngram_draft(p, gen, 3).empty()) << "lookup hit at generated token " << j;
        }
        EXPECT_EQ(b.tokens, a.tokens);
        ASSERT_EQ(on->ticks.size(), off->ticks.size()) << "n-gram on changed the tick count";
        for (std::size_t i = 0; i < on->ticks.size(); ++i) {
            EXPECT_EQ(on->ticks[i].drafts, off->ticks[i].drafts) << "tick " << i << ": drafts are not the MTP drafts";
            EXPECT_EQ(on->ticks[i].accepted, off->ticks[i].accepted) << "tick " << i;
        }
        EXPECT_EQ(s1.ngram_lookups - s0.ngram_lookups, decode_ticks(*on)) << "one lookup per greedy decode step";
        EXPECT_EQ(s1.ngram_drafts - s0.ngram_drafts, 0u);
    }
    EXPECT_EQ(off->engine->stats().ngram_lookups, 0u) << "ngram_draft = false must not search (env ignored)";
}


// ---- per-request draft accounting (GenerateResult n-gram / fallback counters) ----------------
// GenerateResult splits the request's drafts by source: ngram_* is the prompt-lookup share of
// draft_tokens / accepted_draft_tokens (the rest is MTP) plus this request's lookups, hits and
// MTP fallbacks, so `halo run` / serve can report each source per request instead of reading
// engine-wide EngineStats deltas that mix every concurrent request.

namespace {

/// p0, the first 12 tokens of its greedy continuation, p0 again, then that continuation's
/// first 2 tokens: the tiny model never repeats a 3-gram on its own (no lookup hit in 300
/// greedy tokens from p0-p2), but this context recurs, and its greedy decode hits the n-gram
/// lookup (1 hit, 3 drafted, 1 accepted on tiny-f32 when written).
Toks recurring_prompt(const Toks& p0, const Toks& ref) {
    Toks q = p0;
    q.insert(q.end(), ref.begin(), ref.begin() + 12);
    q.insert(q.end(), p0.begin(), p0.end());
    q.insert(q.end(), ref.begin(), ref.begin() + 2);
    return q;
}

struct DraftSplit {
    std::size_t lookups = 0, hits = 0, drafted = 0, accepted = 0;
};

/// Replays a request that ran alone (one sequence per tick) through its decode ticks: each one
/// searched the context (prompt + tokens emitted so far, the last one pending) and, on a hit,
/// verified exactly the n-gram draft.
DraftSplit replay_ngram(const Toks& prompt, const Toks& out, const std::vector<TickInfo>& ticks, std::size_t k) {
    DraftSplit d;
    std::size_t j = 1;  // the prefill's last row emitted the first token
    for (const TickInfo& t : ticks) {
        if (t.prefill_rows != 0) continue;
        EXPECT_EQ(t.sequences, 1u) << "replay needs the request alone";
        const std::vector<std::int32_t> draft =
            halo::speculative::ngram_draft(prompt, std::span(out.data(), std::min(j, out.size())), k);
        ++d.lookups;
        if (!draft.empty()) {
            ++d.hits;
            EXPECT_EQ(t.drafts, draft) << "decode tick after " << j << " tokens: a hit verifies the n-gram draft";
            d.drafted += t.drafted;
            d.accepted += t.accepted;
        }
        j += t.accepted + 1;
    }
    return d;
}

}  // namespace

TEST_F(EngineTest, ResultSplitsTheRequestsNgramDraftsFromItsMtpDrafts) {
    std::vector<TickInfo> ticks;
    std::mutex mu;
    CpuEngineOptions o;
    o.gate_mode = GateMode::Always;
    o.ngram_draft = true;
    o.on_tick = [&](const TickInfo& t) {
        const std::lock_guard lk(mu);
        ticks.push_back(t);
    };
    EngineConfig c = base_cfg();
    c.mtp_max_draft = 3;
    auto e = make(o, c);
    const Toks p0 = prompt("p0");
    const Toks q = recurring_prompt(p0, run(*engine_, greedy(p0, 24)).tokens);

    const halo::runtime::EngineStats s0 = e->stats();
    const GenerateResult a = run(*e, greedy(q, 24));
    const halo::runtime::EngineStats s1 = e->stats();
    ASSERT_EQ(a.tokens.size(), 24u);
    const DraftSplit want = replay_ngram(q, a.tokens, ticks, 3);
    ASSERT_GE(want.hits, 1u) << "premise: recurring_prompt hits the n-gram lookup (pick another context if the tiny "
                                "model changed)";
    EXPECT_EQ(a.ngram_lookups, want.lookups) << "one lookup per greedy decode step";
    EXPECT_EQ(a.ngram_hits, want.hits);
    EXPECT_EQ(a.ngram_draft_tokens, want.drafted) << "n-gram drafts = the drafts of the ticks that hit";
    EXPECT_EQ(a.ngram_accepted_draft_tokens, want.accepted);
    EXPECT_GT(a.draft_tokens, a.ngram_draft_tokens) << "the steps without a hit drafted with MTP";
    EXPECT_EQ(a.mtp_fallbacks, 0u);
    // Alone on the engine, the request's counters are exactly the engine's.
    EXPECT_EQ(a.ngram_lookups, s1.ngram_lookups - s0.ngram_lookups);
    EXPECT_EQ(a.ngram_hits, s1.ngram_drafts - s0.ngram_drafts);
    EXPECT_EQ(a.draft_tokens, s1.speculative_attempts - s0.speculative_attempts);

    // Batched with a request whose context never recurs: each keeps its own counts.
    const Toks p1 = prompt("p1");
    const GenerateResult b_alone = run(*e, greedy(p1, 12));
    ASSERT_GT(b_alone.ngram_lookups, 0u);
    ASSERT_EQ(b_alone.ngram_hits, 0u);
    {
        const std::lock_guard lk(mu);
        ticks.clear();
    }
    std::vector<GenerateResult> con(2);
    std::barrier start(2);
    std::thread tb([&] {
        start.arrive_and_wait();
        con[1] = run(*e, greedy(p1, 12));
    });
    start.arrive_and_wait();
    con[0] = run(*e, greedy(q, 24));
    tb.join();
    EXPECT_TRUE(std::ranges::any_of(ticks, [](const TickInfo& t) { return t.prefill_rows == 0 && t.sequences == 2; }))
        << "the two requests never shared a decode tick";
    EXPECT_EQ(con[0].tokens, a.tokens);
    EXPECT_EQ(con[1].tokens, b_alone.tokens);
    for (const auto& [i, alone] : {std::pair<std::size_t, const GenerateResult*>{0, &a}, {1, &b_alone}}) {
        SCOPED_TRACE(i == 0 ? "recurring request batched" : "plain request batched");
        EXPECT_EQ(con[i].ngram_lookups, alone->ngram_lookups);
        EXPECT_EQ(con[i].ngram_hits, alone->ngram_hits);
        EXPECT_EQ(con[i].ngram_draft_tokens, alone->ngram_draft_tokens);
        EXPECT_EQ(con[i].ngram_accepted_draft_tokens, alone->ngram_accepted_draft_tokens);
    }

    // Off: nothing searched, nothing attributed to n-gram.
    EngineConfig c_off = base_cfg();
    c_off.mtp_max_draft = 3;
    CpuEngineOptions o_off;
    o_off.gate_mode = GateMode::Always;
    o_off.ngram_draft = false;
    const GenerateResult off = run(*make(o_off, c_off), greedy(q, 24));
    EXPECT_EQ(off.tokens, a.tokens);
    EXPECT_EQ(off.ngram_lookups + off.ngram_hits + off.ngram_draft_tokens + off.ngram_accepted_draft_tokens, 0u);
    EXPECT_GT(off.draft_tokens, 0u);
}
