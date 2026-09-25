// Milestone 5: engine failure paths (code review 2026-09-25 R-1, R-2, R-3, R-5(a), N-2),
// driven through CpuEngineOptions::faults on the tiny model.
//
//   R-1  a tick-wide failure fails every sequence of the tick; a per-sequence failure in the
//        output phase fails only that sequence; prefix checkpoints are best-effort
//   R-2  a throwing retirement (cache part) never escapes the noexcept worker
//   R-3  stop token / deadline end a request that is queued, prefilling or decoding
//   R-5  destroying the Engine with requests queued and active releases every caller
//   N-2  with more cache entries than the old retry budget, the MTP fallback still runs

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <future>
#include <mutex>
#include <new>
#include <stop_token>
#include <thread>

#include "../models/tiny_golden.h"
#include "halo/core/error.h"
#include "halo/runtime/cpu_engine.h"

using halo::runtime::CpuEngineOptions;
using halo::runtime::Engine;
using halo::runtime::EngineConfig;
using halo::runtime::FinishReason;
using halo::runtime::GenerateRequest;
using halo::runtime::GenerateResult;
using halo::runtime::TickInfo;
using halo::runtime::TokenEvent;
using halo::test::Golden;
using Toks = std::vector<std::int32_t>;
using Clock = std::chrono::steady_clock;

namespace {

std::filesystem::path tiny() { return halo::test::tiny_dir() / "tiny-f32.gguf"; }

EngineConfig cfg(std::size_t seqs = 4) {
    EngineConfig c;
    c.model_path = tiny().string();
    c.max_context = 1024;
    c.max_sequences = seqs;
    c.mtp_max_draft = 2;
    c.prefix_cache = false;
    return c;
}

GenerateRequest greedy(Toks prompt, std::size_t n) {
    GenerateRequest r;
    r.prompt = std::move(prompt);
    r.sampling.temperature = 0.0f;
    r.max_tokens = n;
    return r;
}

Toks prefix(const Toks& t, std::size_t n) { return Toks(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(std::min(n, t.size()))); }

class Faults : public ::testing::Test {
protected:
    static void SetUpTestSuite() { golden_ = Golden::load(); }
    void SetUp() override {
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        if (!std::filesystem::exists(tiny())) GTEST_SKIP() << "tiny-f32.gguf missing under " << halo::test::tiny_dir();
    }
    static Toks prompt(const char* p) { return golden_->i32(std::string(p) + ".tokens"); }
    static Toks gold(const char* p) { return golden_->i32(std::string(p) + ".decode_tokens"); }
    static inline std::optional<Golden> golden_;
};

/// Runs `reqs` concurrently (one thread each, released together); results in order.
std::vector<GenerateResult> concurrently(Engine& e, const std::vector<GenerateRequest>& reqs) {
    std::vector<GenerateResult> out(reqs.size());
    std::barrier start(static_cast<std::ptrdiff_t>(reqs.size()));
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        th.emplace_back([&, i] {
            start.arrive_and_wait();
            out[i] = e.generate(reqs[i], {});
        });
    }
    for (auto& t : th) t.join();
    return out;
}

}  // namespace

// ---- R-1 ------------------------------------------------------------------------------------

TEST_F(Faults, TickWideFailureWithTwoSequencesFailsBothAndTheEngineRecovers) {
    std::atomic<bool> fired{false};
    std::atomic<std::size_t> seqs_at_failure{0};
    CpuEngineOptions o;
    o.faults.before_step = [&](std::span<const halo::speculative::StepRequest> reqs) {
        if (reqs.size() >= 2 && !fired.exchange(true)) {
            seqs_at_failure = reqs.size();
            halo::throw_error(halo::ErrorCode::Kernel, "injected tick failure");
        }
    };
    auto e = halo::runtime::create_cpu_engine(cfg(), o);
    // Long enough that the two requests certainly share a tick before either finishes.
    const auto r = concurrently(*e, {greedy(prompt("p1"), 40), greedy(prompt("p2"), 40)});
    ASSERT_TRUE(fired) << "the two requests never shared a tick";
    EXPECT_EQ(seqs_at_failure.load(), 2u);
    for (const auto& x : r) {
        EXPECT_EQ(x.finish, FinishReason::Error);
        EXPECT_NE(x.error.find("injected tick failure"), std::string::npos) << x.error;
    }
    EXPECT_EQ(e->stats().active_sequences, 0u);
    EXPECT_EQ(e->generate(greedy(prompt("p1"), 6), {}).tokens, prefix(gold("p1"), 6)) << "engine unusable after the failure";
}

TEST_F(Faults, OutputPhaseFailureOfOneSequenceSparesTheOthersInTheTick) {
    const Toks bad = prompt("p2");
    std::atomic<std::size_t> max_seqs{0};
    std::atomic<bool> fired{false};
    CpuEngineOptions o;
    o.faults.before_step = [&](std::span<const halo::speculative::StepRequest> reqs) {
        if (!fired && reqs.size() > max_seqs) max_seqs = reqs.size();
    };
    o.faults.output = [&](std::span<const std::int32_t> p) {
        // p2 fails in its first output phase that is shared with another sequence.
        if (max_seqs >= 2 && std::equal(p.begin(), p.end(), bad.begin(), bad.end()) && !fired.exchange(true)) {
            throw std::runtime_error("injected sampler failure");
        }
    };
    auto e = halo::runtime::create_cpu_engine(cfg(), o);
    const auto r = concurrently(*e, {greedy(prompt("p1"), 12), greedy(bad, 40)});
    ASSERT_TRUE(fired) << "the failure was never injected in a shared tick";
    EXPECT_EQ(r[0].finish, FinishReason::Length) << r[0].error;
    EXPECT_EQ(r[0].tokens, prefix(gold("p1"), 12)) << "the co-batched request must be unaffected";
    EXPECT_EQ(r[1].finish, FinishReason::Error);
    EXPECT_NE(r[1].error.find("injected sampler failure"), std::string::npos) << r[1].error;
    EXPECT_EQ(e->stats().active_sequences, 0u);
    EXPECT_EQ(e->generate(greedy(bad, 4), {}).tokens, prefix(gold("p2"), 4));
}

TEST_F(Faults, PrefixCheckpointFailureIsBestEffort) {
    EngineConfig c = cfg(1);
    c.prefix_cache = true;
    std::atomic<int> attempts{0};
    CpuEngineOptions o;
    o.faults.checkpoint = [&] {
        ++attempts;
        halo::throw_error(halo::ErrorCode::Memory, "injected checkpoint allocation failure");
    };
    auto e = halo::runtime::create_cpu_engine(c, o);
    const GenerateResult a = e->generate(greedy(prompt("p0"), 6), {});
    EXPECT_EQ(a.finish, FinishReason::Length) << a.error;
    EXPECT_EQ(a.tokens, prefix(gold("p0"), 6));
    EXPECT_GT(attempts.load(), 0) << "no checkpoint was attempted: the fault never fired";
    // The prompt-end checkpoint was lost, so an exact re-submit recomputes (always-recompute).
    const GenerateResult b = e->generate(greedy(prompt("p0"), 6), {});
    EXPECT_EQ(b.cached_prompt_tokens, 0u);
    EXPECT_EQ(b.tokens, a.tokens);
}

// ---- R-2 ------------------------------------------------------------------------------------

TEST_F(Faults, RetirementFailureNeverEscapesTheWorker) {
    EngineConfig c = cfg(2);
    c.prefix_cache = true;
    std::atomic<int> fired{0};
    CpuEngineOptions o;
    o.faults.retire_cache = [&] {
        ++fired;
        throw std::bad_alloc();
    };
    auto e = halo::runtime::create_cpu_engine(c, o);
    // Finished, cancelled (callback) and cancelled (stop token) retirements all hit the fault.
    const GenerateResult a = e->generate(greedy(prompt("p1"), 5), {});
    std::size_t calls = 0;
    const GenerateResult b = e->generate(greedy(prompt("p2"), 50), [&](const TokenEvent&) { return ++calls < 2; });
    std::stop_source src;
    GenerateRequest q = greedy(prompt("p1"), 50);
    q.cancel = src.get_token();
    const GenerateResult d = e->generate(q, [&](const TokenEvent&) {
        src.request_stop();
        return true;
    });
    EXPECT_EQ(a.finish, FinishReason::Length);
    EXPECT_EQ(a.tokens, prefix(gold("p1"), 5));
    EXPECT_EQ(b.finish, FinishReason::Cancelled);
    EXPECT_EQ(d.finish, FinishReason::Cancelled);
    EXPECT_GE(fired.load(), 3);
    EXPECT_EQ(e->stats().active_sequences, 0u);
    // Nothing was cached (the cache part failed), and the engine still serves exactly.
    const GenerateResult again = e->generate(greedy(prompt("p1"), 5), {});
    EXPECT_EQ(again.cached_prompt_tokens, 0u);
    EXPECT_EQ(again.tokens, a.tokens);
}

// ---- R-3 ------------------------------------------------------------------------------------

TEST_F(Faults, StopTokenEndsAQueuedRequestWithoutRunningIt) {
    auto e = halo::runtime::create_cpu_engine(cfg(1));
    std::stop_source long_src;
    GenerateRequest a = greedy(prompt("p1"), 400);  // occupies the only slot
    a.cancel = long_src.get_token();
    std::atomic<bool> a_started{false};
    auto fa = std::async(std::launch::async, [&] {
        return e->generate(a, [&](const TokenEvent&) {
            a_started = true;
            return true;
        });
    });
    while (!a_started) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::stop_source src;
    GenerateRequest b = greedy(prompt("p2"), 10);
    b.cancel = src.get_token();
    bool b_called = false;
    auto fb = std::async(std::launch::async, [&] { return e->generate(b, [&](const TokenEvent&) { return b_called = true; }); });
    const auto until = Clock::now() + std::chrono::seconds(300);  // b is queued behind a
    while (e->stats().queued_requests < 1 && Clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ASSERT_EQ(e->stats().queued_requests, 1u);
    src.request_stop();
    const GenerateResult rb = fb.get();
    EXPECT_EQ(rb.finish, FinishReason::Cancelled);
    EXPECT_TRUE(rb.tokens.empty());
    EXPECT_FALSE(b_called);
    EXPECT_EQ(fa.wait_for(std::chrono::seconds(0)), std::future_status::timeout) << "a must still be running";
    long_src.request_stop();  // and a decoding request ends at its next tick
    const GenerateResult ra = fa.get();
    EXPECT_EQ(ra.finish, FinishReason::Cancelled);
    EXPECT_LT(ra.tokens.size(), 400u);
    const std::size_t ka = std::min<std::size_t>(ra.tokens.size(), gold("p1").size());
    EXPECT_EQ(prefix(ra.tokens, ka), prefix(gold("p1"), ka));
    EXPECT_EQ(e->stats().active_sequences, 0u);
    EXPECT_EQ(e->stats().queued_requests, 0u);
}

TEST_F(Faults, StopTokenEndsAPrefillingRequestBeforeItsFirstToken) {
    CpuEngineOptions o;
    o.prefill_chunk = 64;
    std::stop_source src;
    std::atomic<int> prefill_ticks{0};
    o.faults.before_step = [&](std::span<const halo::speculative::StepRequest> reqs) {
        for (const auto& q : reqs) {
            if (!q.prefill.empty() && ++prefill_ticks == 2) src.request_stop();  // mid-prefill
        }
    };
    auto e = halo::runtime::create_cpu_engine(cfg(1), o);
    Toks long_prompt;
    for (int i = 0; i < 3; ++i) {
        const Toks p0 = prompt("p0");
        long_prompt.insert(long_prompt.end(), p0.begin(), p0.end());  // 450 tokens: 8 chunks of 64
    }
    GenerateRequest q = greedy(long_prompt, 10);
    q.cancel = src.get_token();
    bool called = false;
    const GenerateResult r = e->generate(q, [&](const TokenEvent&) { return called = true; });
    EXPECT_EQ(r.finish, FinishReason::Cancelled);
    EXPECT_TRUE(r.tokens.empty());
    EXPECT_FALSE(called);
    EXPECT_LE(prefill_ticks.load(), 3) << "the prefill kept running after the stop request";
    // A stop requested before generate() ends the request at its first tick.
    std::stop_source early;
    early.request_stop();
    GenerateRequest q2 = greedy(prompt("p1"), 10);
    q2.cancel = early.get_token();
    prefill_ticks = 100;  // the hook no longer matters
    EXPECT_EQ(e->generate(q2, {}).finish, FinishReason::Cancelled);
}

TEST_F(Faults, DeadlineEndsQueuedAndDecodingRequestsAsLength) {
    auto e = halo::runtime::create_cpu_engine(cfg(1));
    GenerateRequest past = greedy(prompt("p1"), 10);
    past.deadline = Clock::now() - std::chrono::seconds(1);
    bool called = false;
    const GenerateResult r = e->generate(past, [&](const TokenEvent&) { return called = true; });
    EXPECT_EQ(r.finish, FinishReason::Length);
    EXPECT_TRUE(r.deadline_expired);
    EXPECT_TRUE(r.tokens.empty());
    EXPECT_FALSE(called);
    const GenerateResult plain = e->generate(greedy(prompt("p1"), 3), {});
    EXPECT_EQ(plain.finish, FinishReason::Length);
    EXPECT_FALSE(plain.deadline_expired) << "max_tokens must not look like a deadline";
    // A deadline that expires while decoding: the request ends as Length and the tokens
    // delivered so far are a golden prefix.
    GenerateRequest q3 = greedy(prompt("p1"), 400);
    q3.deadline = Clock::now() + std::chrono::milliseconds(300);
    const GenerateResult r3 = e->generate(q3, {});
    EXPECT_EQ(r3.finish, FinishReason::Length);
    EXPECT_TRUE(r3.deadline_expired);
    EXPECT_LT(r3.tokens.size(), 400u) << "the deadline did not end decoding";
    std::printf("decode deadline: %zu tokens before the 300 ms deadline\n", r3.tokens.size());
    const std::size_t k = std::min<std::size_t>(r3.tokens.size(), gold("p1").size());
    EXPECT_EQ(prefix(r3.tokens, k), prefix(gold("p1"), k));
}

TEST_F(Faults, StopRequestedWhileTokensAreBufferedEndsDeliveryAtOnce) {
    // A slow reader: the first callback waits until the worker is >= 5 tokens ahead, then
    // requests a stop and returns true. None of the buffered tokens may be delivered.
    auto e = halo::runtime::create_cpu_engine(cfg(1));
    std::stop_source src;
    GenerateRequest q = greedy(prompt("p1"), 400);
    q.cancel = src.get_token();
    std::size_t calls = 0;
    const auto gen0 = e->stats().tokens_generated;
    const GenerateResult r = e->generate(q, [&](const TokenEvent&) {
        if (++calls == 1) {
            const auto until = Clock::now() + std::chrono::seconds(600);
            while (e->stats().tokens_generated < gen0 + 6 && Clock::now() < until) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            src.request_stop();
        }
        return true;
    });
    EXPECT_GE(e->stats().tokens_generated, gen0 + 6) << "the worker never got ahead: the test proves nothing";
    EXPECT_EQ(calls, 1u) << "buffered tokens were delivered after the stop request";
    EXPECT_EQ(r.finish, FinishReason::Cancelled);
    EXPECT_EQ(r.tokens, prefix(gold("p1"), 1));
    EXPECT_FALSE(r.deadline_expired);
}

// ---- R-5(a): destruction with requests in flight ----------------------------------------------

TEST_F(Faults, DestroyingTheEngineReleasesQueuedAndActiveCallers) {
    auto e = halo::runtime::create_cpu_engine(cfg(1));
    std::atomic<bool> started{false};
    std::vector<std::future<GenerateResult>> fs;
    fs.push_back(std::async(std::launch::async, [&] {
        return e->generate(greedy(prompt("p1"), 400), [&](const TokenEvent&) {
            started = true;
            return true;
        });
    }));
    while (!started) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    for (int i = 0; i < 2; ++i) {  // queued behind the active one
        fs.push_back(std::async(std::launch::async, [&] { return e->generate(greedy(prompt("p2"), 400), {}); }));
    }
    while (e->stats().queued_requests < 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    e.reset();  // joins the worker; every caller must be released
    for (auto& f : fs) {
        ASSERT_EQ(f.wait_for(std::chrono::seconds(300)), std::future_status::ready) << "a caller was never released";
        const GenerateResult r = f.get();
        EXPECT_EQ(r.finish, FinishReason::Error);
        EXPECT_NE(r.error.find("shut down"), std::string::npos) << r.error;
    }
}

// ---- N-2 --------------------------------------------------------------------------------------

TEST_F(Faults, MtpFallbackRunsEvenWithManyCacheEntries) {
    EngineConfig c = cfg(1);
    c.prefix_cache = true;
    CpuEngineOptions o;
    o.gate_mode = halo::speculative::GateMode::Always;
    o.prefix_cache_entries = 16;
    std::atomic<bool> arm{false};
    std::atomic<int> injected{0};
    o.faults.before_step = [&](std::span<const halo::speculative::StepRequest> reqs) {
        // Simulated MTP-KV exhaustion: any drafting step fails with Error(Memory), however
        // much cache is evicted; only the k = 0 fallback can make progress.
        if (!arm) return;
        for (const auto& q : reqs) {
            if (q.max_draft > 0 && q.seq->mtp_kv) {
                ++injected;
                halo::throw_error(halo::ErrorCode::Memory, "injected MTP KV exhaustion");
            }
        }
    };
    std::vector<TickInfo> ticks;
    std::mutex mu;
    o.on_tick = [&](const TickInfo& t) {
        const std::lock_guard lk(mu);
        ticks.push_back(t);
    };
    auto e = halo::runtime::create_cpu_engine(c, o);
    const Toks p0 = prompt("p0");
    for (std::size_t i = 0; i < 12; ++i) {  // 12 cached entries: more than the old 9-attempt budget
        (void)e->generate(greedy(Toks(p0.begin() + static_cast<std::ptrdiff_t>(i), p0.begin() + static_cast<std::ptrdiff_t>(i + 8)), 2), {});
    }
    arm = true;
    const GenerateResult r = e->generate(greedy(prompt("p1"), 6), {});
    EXPECT_EQ(r.finish, FinishReason::Length) << r.error;
    EXPECT_EQ(r.tokens, prefix(gold("p1"), 6));
    EXPECT_GE(e->stats().mtp_fallbacks, 1u);
    EXPECT_GE(injected.load(), 13) << "every cache entry should be evicted before the fallback";
}
