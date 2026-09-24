#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string_view>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"
#include "halo/hardware/bandwidth.h"

using namespace halo::hardware;

TEST(BandwidthStats, HandComputedOddSample) {
    const std::vector<double> s{4, 100, 1, 3, 2};
    const auto st = compute_stats(s);
    EXPECT_EQ(st.n, 5u);
    EXPECT_DOUBLE_EQ(st.min, 1);
    EXPECT_DOUBLE_EQ(st.max, 100);
    EXPECT_DOUBLE_EQ(st.mean, 22);
    EXPECT_DOUBLE_EQ(st.median, 3);
    // deviations -21,-20,-19,-18,78 -> sum of squares 7610 -> /(n-1) = 1902.5
    EXPECT_DOUBLE_EQ(st.stddev, std::sqrt(1902.5));
    EXPECT_DOUBLE_EQ(st.p95, 100);  // nearest rank ceil(4.75) = 5
}

TEST(BandwidthStats, EvenSampleAndNearestRankP95) {
    const auto st4 = compute_stats(std::vector<double>{4, 1, 3, 2});
    EXPECT_DOUBLE_EQ(st4.median, 2.5);
    EXPECT_DOUBLE_EQ(st4.p95, 4);
    std::vector<double> s20(20);
    std::iota(s20.begin(), s20.end(), 1.0);
    // Nearest rank: ceil(0.95 * 20) = 19 -> 19 (linear interpolation would give 19.05).
    EXPECT_DOUBLE_EQ(compute_stats(s20).p95, 19);
    const auto one = compute_stats(std::vector<double>{7});
    EXPECT_DOUBLE_EQ(one.stddev, 0);
    EXPECT_DOUBLE_EQ(one.p95, 7);
}

TEST(BandwidthStats, RejectsEmptyAndNonFinite) {
    EXPECT_THROW((void)compute_stats(std::vector<double>{}), halo::Error);
    EXPECT_THROW((void)compute_stats(std::vector<double>{1, std::numeric_limits<double>::infinity()}), halo::Error);
    EXPECT_THROW((void)compute_stats(std::vector<double>{std::nan("")}), halo::Error);
}

TEST(Bandwidth, OptionsValidated) {
    BandwidthOptions o;
    o.buffer_bytes = 1 << 20;
    o.threads = 2;
    EXPECT_THROW((void)measure_host_bandwidth(o), halo::Error);  // no label
    o.label = "dev-host";
    o.warmup = 1;
    o.iterations = 3;
    EXPECT_THROW((void)measure_host_bandwidth(o), halo::Error);  // below TRD §50 minimums
    o.allow_nonconformant = true;
    o.buffer_bytes = 8;  // one word for two threads
    EXPECT_THROW((void)measure_host_bandwidth(o), halo::Error);
}

// The TRD §50-conformant defaults (256 MiB buffers, 5 warm-up + 20 measured). Takes seconds
// and is meaningless under ASan, so it only runs when HALO_BANDWIDTH_FULL=1.
TEST(Bandwidth, ConformantDefaultsDevHost) {
    const char* flag = std::getenv("HALO_BANDWIDTH_FULL");
    if (flag == nullptr || std::string_view(flag) != "1")
        GTEST_SKIP() << "set HALO_BANDWIDTH_FULL=1 to run the full-size TRD §50 bandwidth benchmark";
    BandwidthOptions o;
    o.label = "dev-host";
    const auto r = measure_host_bandwidth(o);
    EXPECT_EQ(r.read_gbps.n, 20u);
    EXPECT_EQ(r.warmup, 5u);
    EXPECT_GT(r.read_gbps.min, 0.0);
    const nlohmann::json j = r;
    std::cout << "[dev-host, 256 MiB, NOT a HALO performance claim] " << j.dump() << "\n";
}

// Tiny, non-conformant run: exercises the threading/barrier path and the result record.
// The numbers are dev-host numbers and are only printed, never asserted against a target.
TEST(Bandwidth, SmallDevHostRunProducesPositiveFiniteStats) {
    BandwidthOptions o;
    o.buffer_bytes = 4 << 20;
    o.threads = 3;  // deliberately not a divisor of the word count
    o.warmup = 1;
    o.iterations = 4;
    o.label = "dev-host";
    o.allow_nonconformant = true;
    const auto r = measure_host_bandwidth(o);
    EXPECT_EQ(r.tier, MemoryTier::Host);
    EXPECT_EQ(r.processor, Processor::Cpu);
    EXPECT_EQ(r.label, "dev-host");
    EXPECT_EQ(r.threads, 3u);
    for (const auto* s : {&r.read_gbps, &r.write_gbps, &r.copy_gbps}) {
        EXPECT_EQ(s->n, 4u);
        EXPECT_GT(s->min, 0.0);
        EXPECT_TRUE(std::isfinite(s->max));
        EXPECT_LE(s->min, s->median);
        EXPECT_LE(s->median, s->max);
    }
    const nlohmann::json j = r;
    EXPECT_EQ(j["label"], "dev-host");
    EXPECT_EQ(j["processor"], "cpu");
    std::cout << "[dev-host, 4 MiB, NOT a HALO performance claim] " << j.dump() << "\n";
}
