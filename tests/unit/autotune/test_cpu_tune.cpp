// Real tuning on the dev host over the CPU reference kernels (TRD §64 chain):
//   hardware state -> key -> candidates -> measure (TRD §50) -> winner -> persist ->
//   close -> read-only reopen ("next launch") -> runtime lookup returns that winner.
// Dev-host timings are harness smoke tests only (D-001); nothing here asserts a speed.

#include <gtest/gtest.h>

#include <thread>

#include <nlohmann/json.hpp>

#include "autotune_test_util.h"
#include "halo/autotune/cpu_ops.h"
#include "halo/autotune/lookup.h"
#include "halo/hardware/bandwidth.h"

using namespace halo::autotune;
using test::TempDb;

namespace {

ProfileKey dev_host_key() {
    halo::profiling::HardwareState hw;  // CPU-only key: the WSL dev host has no amdgpu sysfs
    hw.power_mode = halo::profiling::make_power_mode("dev-host", std::nullopt, std::nullopt);
    hw.kernel = "test-kernel";
    return make_profile_key(hw, "tiny-model-sha", "tiny-pack", "x86-64");
}

std::vector<unsigned> thread_counts() {
    const unsigned hw = std::max(1U, std::min(4U, std::thread::hardware_concurrency()));
    std::vector<unsigned> t{1};
    if (hw >= 2) t.push_back(2);
    if (hw >= 4) t.push_back(4);
    return t;
}

}  // namespace

TEST(CpuTune, MatmulTunePersistReopenLookup) {
    TempDb t("cpu_matmul");
    const ProfileKey key = dev_host_key();
    CpuMatmulTunable op(1, 256, 512, thread_counts());
    TuneOptions o;
    o.strategy = Strategy::Exhaustive;
    // This test proves the persistence chain, not stability: under `ctest -j` the dev host
    // is loaded, so the 5 % cv policy would reject at random. Instability rejection is
    // tested deterministically with a ManualClock in test_tuner.cpp.
    o.stability.max_cv = 1e9;
    OpTuneResult result;
    {
        ProfileDb db = ProfileDb::open(t.path());
        TunableOp* ops[] = {&op};
        result = tune(db, key, ops, o).ops.at(0);
    }
    ASSERT_TRUE(result.winner) << nlohmann::json(result).dump();
    EXPECT_TRUE(result.persisted);
    EXPECT_EQ(result.candidates.size(), thread_counts().size());
    for (const auto& c : result.candidates) {
        EXPECT_TRUE(c.measured);
        EXPECT_TRUE(c.rejected.empty()) << c.rejected;  // every thread count is bit-identical
        EXPECT_EQ(c.stats.n, 20u);
        EXPECT_GT(c.stats.median, 0.0);
    }
    // Next launch: the runtime opens read-only and gets exactly the persisted winner.
    const auto lookup = ProfileLookup::open(t.path());
    const auto r = select_kernel(lookup, dev_host_key(), op.key(), "cpu", op.candidates(), nullptr, {});
    EXPECT_EQ(r.source, SelectionSource::Exact);
    ASSERT_TRUE(r.candidate);
    EXPECT_EQ(*r.candidate, *result.winner);
    EXPECT_DOUBLE_EQ(r.value, result.winner_median_ns);
    EXPECT_EQ(op.key().shape, "T=1,K=256,N=512,w=f32");
}

TEST(CpuTune, GdnChunkedCandidatesValidateAgainstRecurrent) {
    halo::cpu::GdnDims dims;
    dims.n_k_heads = 2;
    dims.n_v_heads = 4;
    dims.d_k = 16;
    dims.d_v = 16;
    CpuGdnChunkedTunable op(dims, 48, {8, 16, 64}, {1, 2});
    EXPECT_EQ(op.candidates().size(), 6u);
    for (const auto& c : op.candidates()) {
        std::string why;
        EXPECT_TRUE(op.validate(c, why)) << c.to_string() << ": " << why;
        const auto in = op.cost(c);
        ASSERT_TRUE(in);
        EXPECT_GT(in->flops, 0);
        EXPECT_LE(in->parallelism, 4u);
    }
    // Chunk size changes the predicted intra-chunk work.
    EXPECT_LT(op.cost(Candidate{{{"chunk", 8}, {"threads", 1}}})->flops,
              op.cost(Candidate{{{"chunk", 64}, {"threads", 1}}})->flops);
    // A tolerance of 0 must reject: the forms differ by fp32 rounding (proves the gate works).
    CpuGdnChunkedTunable strict(dims, 48, {16}, {1}, 2, 0.0f);
    std::string why;
    EXPECT_FALSE(strict.validate(Candidate{{{"chunk", 16}, {"threads", 1}}}, why));
    EXPECT_NE(why.find("differs from recurrent"), std::string::npos);
}

TEST(CpuTune, HeuristicWithMeasuredBandwidth) {
    halo::hardware::BandwidthOptions bo;
    bo.buffer_bytes = 4ULL << 20;
    bo.threads = 2;
    bo.warmup = 1;
    bo.iterations = 3;
    bo.label = "dev-host";
    bo.allow_nonconformant = true;  // a smoke measurement for the cost model, not §50 data
    const auto bw = halo::hardware::measure_host_bandwidth(bo);
    const std::vector<halo::hardware::TierBandwidth> measured{bw};
    const CostModel model = CostModel::from_measurements(measured, /*gflops_per_thread=*/1.0);

    TempDb t("cpu_heur");
    ProfileDb db = ProfileDb::open(t.path());
    db.record_tier_bandwidth(dev_host_key(), bw);
    EXPECT_EQ(db.tier_bandwidth(dev_host_key()).size(), 1u);

    halo::cpu::GdnDims dims{.n_k_heads = 2, .n_v_heads = 4, .d_k = 16, .d_v = 16};
    CpuGdnChunkedTunable op(dims, 32, {8, 16, 32}, {1, 2});
    TuneOptions o;
    o.strategy = Strategy::Heuristic;
    o.heuristic_keep = 2;
    o.cost_model = model;
    o.stability.max_cv = 1e9;  // see MatmulTunePersistReopenLookup
    TunableOp* ops[] = {&op};
    const auto r = tune(db, dev_host_key(), ops, o).ops.at(0);
    EXPECT_EQ(r.candidates.size(), 2u);
    ASSERT_TRUE(r.winner);
    for (const auto& c : r.candidates) ASSERT_TRUE(c.predicted);
}
