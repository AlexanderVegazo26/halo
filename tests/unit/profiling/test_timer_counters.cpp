#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/profiling/counters.h"
#include "halo/profiling/timer.h"
#include "halo/runtime/engine.h"

using namespace halo::profiling;
using halo::hardware::MemoryTier;

TEST(Timer, ManualClockNeverDecreases) {
    ManualClock c(100);
    c.advance(50);
    c.advance(-1000);  // ignored
    EXPECT_EQ(c.now_ns(), 150);
    Stopwatch sw(c);
    c.advance(2'500'000);
    EXPECT_EQ(sw.elapsed_ns(), 2'500'000);
    EXPECT_DOUBLE_EQ(sw.elapsed_ms(), 2.5);
    sw.restart();
    EXPECT_EQ(sw.elapsed_ns(), 0);
}

TEST(Timer, SteadyClockIsMonotonic) {
    const auto& c = SteadyClock::instance();
    std::int64_t prev = c.now_ns();
    for (int i = 0; i < 1000; ++i) {
        const std::int64_t now = c.now_ns();
        ASSERT_GE(now, prev);
        prev = now;
    }
}

TEST(Timer, ScopedOpTimerRecordsScopeDuration) {
    ManualClock clock;
    OpTimingRecorder rec(2);
    {
        ScopedOpTimer t(rec, "MATMUL", 1000, clock);
        clock.advance(30);
    }
    {
        ScopedOpTimer t(rec, "MATMUL", 1000, clock);
        clock.advance(10);
    }
    {
        ScopedOpTimer t(rec, "MATMUL", 0, clock);
        clock.advance(20);
    }
    {
        ScopedOpTimer t(rec, "ADD", 8, clock);
        clock.advance(5);
    }
    const auto snap = rec.snapshot();
    ASSERT_EQ(snap.size(), 2u);
    const OpTiming& m = snap.at("MATMUL");
    EXPECT_EQ(m.calls, 3u);
    EXPECT_EQ(m.total_ns, 60);
    EXPECT_EQ(m.min_ns, 10);
    EXPECT_EQ(m.max_ns, 30);
    EXPECT_EQ(m.bytes, 2000u);
    EXPECT_EQ(m.samples_ns.size(), 2u);  // capped
    EXPECT_EQ(snap.begin()->first, "ADD");  // sorted
    const nlohmann::json j = m;
    EXPECT_DOUBLE_EQ(j.at("mean_ns").get<double>(), 20.0);
    rec.clear();
    EXPECT_TRUE(rec.snapshot().empty());
}

TEST(Timer, RecorderIsThreadSafe) {
    OpTimingRecorder rec;
    std::vector<std::jthread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&] {
            for (int i = 0; i < 1000; ++i) rec.record("OP", 1, 2);
        });
    }
    ts.clear();  // join
    const auto snap = rec.snapshot();
    EXPECT_EQ(snap.at("OP").calls, 8000u);
    EXPECT_EQ(snap.at("OP").total_ns, 8000);
    EXPECT_EQ(snap.at("OP").bytes, 16000u);
}

TEST(Counters, DerivedValuesComeFromTotals) {
    PerfCounters c;
    EXPECT_FALSE(c.mtp_acceptance_rate());
    EXPECT_FALSE(c.effective_bandwidth_gbps(MemoryTier::Vram));
    c.speculative_attempts = 8;
    c.speculative_accepts = 6;
    EXPECT_DOUBLE_EQ(*c.mtp_acceptance_rate(), 0.75);
    c.tier_bytes_read[MemoryTier::Vram] = 3'000'000'000;
    c.tier_bytes_written[MemoryTier::Vram] = 1'000'000'000;
    EXPECT_FALSE(c.effective_bandwidth_gbps(MemoryTier::Vram));  // window 0
    c.window_ns = 2'000'000'000;                                  // 2 s
    EXPECT_DOUBLE_EQ(*c.effective_bandwidth_gbps(MemoryTier::Vram), 2.0);
    EXPECT_FALSE(c.effective_bandwidth_gbps(MemoryTier::Gtt));

    PerfCounters d = c;
    d += c;
    EXPECT_EQ(d.speculative_attempts, 16u);
    EXPECT_EQ(d.tier_bytes_read.at(MemoryTier::Vram), 6'000'000'000u);
    EXPECT_EQ(d.window_ns, 4'000'000'000u);
    EXPECT_DOUBLE_EQ(*d.effective_bandwidth_gbps(MemoryTier::Vram), 2.0);
}

TEST(Counters, FromEngineStats) {
    halo::runtime::EngineStats s;
    s.tokens_generated = 10;
    s.prefix_cache_hits = 3;
    s.prefix_cache_misses = 1;
    s.speculative_attempts = 4;
    s.speculative_accepts = 2;
    const PerfCounters c = from_engine_stats(s);
    EXPECT_EQ(c.tokens_generated, 10u);
    EXPECT_EQ(c.prefix_cache_hits, 3u);
    EXPECT_EQ(c.prefix_cache_misses, 1u);
    EXPECT_DOUBLE_EQ(*c.mtp_acceptance_rate(), 0.5);
}

TEST(Counters, PrometheusText) {
    PerfCounters c;
    c.tokens_generated = 42;
    c.kernel_time_ns = 1'500'000'000;
    c.tier_bytes_read[MemoryTier::Gtt] = 500;
    c.window_ns = 1000;
    c.memory_allocations_by_tier[MemoryTier::Vram] = 7;
    const std::string t = to_prometheus(c);
    EXPECT_NE(t.find("# TYPE halo_tokens_generated_total counter\nhalo_tokens_generated_total 42\n"), std::string::npos);
    EXPECT_NE(t.find("halo_kernel_time_seconds_total 1.500000000\n"), std::string::npos);
    EXPECT_NE(t.find("halo_memory_allocations_total{tier=\"VRAM\"} 7\n"), std::string::npos);
    EXPECT_NE(t.find("halo_memory_allocations_total{tier=\"HOST\"} 0\n"), std::string::npos);
    EXPECT_NE(t.find("halo_effective_bandwidth_gbps{tier=\"GTT\"} 0.500000\n"), std::string::npos);
    EXPECT_NE(t.find("halo_effective_bandwidth_gbps{tier=\"VRAM\"} NaN\n"), std::string::npos);
    EXPECT_NE(t.find("halo_mtp_acceptance_rate NaN\n"), std::string::npos);
    // Every sample line belongs to a metric that has HELP and TYPE lines.
    std::size_t pos = 0;
    while (pos < t.size()) {
        const std::size_t eol = t.find('\n', pos);
        ASSERT_NE(eol, std::string::npos) << "exposition must end with a newline";
        const std::string line = t.substr(pos, eol - pos);
        if (!line.starts_with('#')) {
            const std::string name = line.substr(0, line.find_first_of("{ "));
            EXPECT_NE(t.find("# TYPE " + name + " "), std::string::npos) << name;
            EXPECT_NE(t.find("# HELP " + name + " "), std::string::npos) << name;
        }
        pos = eol + 1;
    }
}

TEST(Counters, JsonUsesTrdNames) {
    PerfCounters c;
    c.kv_hits = 5;
    c.gpu_time_ns = 9;
    c.tier_bytes_written[MemoryTier::Host] = 4;
    c.window_ns = 2;
    const nlohmann::json j = c;
    for (const char* k : {"tokens_generated", "kernel_calls", "kernel_time_ns", "GPU_time_ns", "CPU_time_ns",
                          "memory_allocations_by_tier", "KV_hits", "KV_misses", "prefix_cache_hits",
                          "prefix_cache_misses", "deltanet_state_checkpoints", "state_evictions",
                          "speculative_attempts", "speculative_accepts", "mtp_acceptance_rate",
                          "effective_bandwidth_per_tier_gbps"}) {
        EXPECT_TRUE(j.contains(k)) << k;
    }
    EXPECT_EQ(j.at("KV_hits"), 5);
    EXPECT_TRUE(j.at("mtp_acceptance_rate").is_null());
    EXPECT_DOUBLE_EQ(j.at("effective_bandwidth_per_tier_gbps").at("HOST").get<double>(), 2.0);
    EXPECT_FALSE(j.at("effective_bandwidth_per_tier_gbps").contains("VRAM"));
}
