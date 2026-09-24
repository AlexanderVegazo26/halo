#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <thread>

#include <nlohmann/json.hpp>

#include "autotune_test_util.h"
#include "halo/autotune/lookup.h"
#include "halo/autotune/tuner.h"

using namespace halo::autotune;
using halo::ErrorCode;
using halo::profiling::ManualClock;
using test::FakeOp;
using test::TempDb;

namespace {

std::vector<Candidate> grid(std::initializer_list<std::int64_t> a, std::initializer_list<std::int64_t> b) {
    std::vector<Candidate> out;
    for (const auto x : a) {
        for (const auto y : b) out.push_back(Candidate{{{"chunk", x}, {"threads", y}}});
    }
    return out;
}

std::vector<std::string> texts(const std::vector<Candidate>& cs) {
    std::vector<std::string> out;
    for (const auto& c : cs) out.push_back(c.to_string());
    return out;
}

OpTuneResult only(const TuneReport& r) {
    EXPECT_EQ(r.ops.size(), 1u);
    return r.ops.at(0);
}

}  // namespace

// ---- strategies -----------------------------------------------------------------------------

TEST(Strategies, ExhaustiveAndGrid) {
    const auto all = grid({16, 32, 64, 128}, {1, 2, 4, 8});
    auto o = test::fast_options(Strategy::Exhaustive);
    EXPECT_EQ(plan_candidates(all, o, {}), all);

    o.strategy = Strategy::Grid;
    o.grid_stride = 2;  // per dimension keep indices 0, 2 and the last (3)
    const auto g = plan_candidates(all, o, {});
    EXPECT_EQ(texts(g), (std::vector<std::string>{"chunk=16;threads=1", "chunk=16;threads=4", "chunk=16;threads=8",
                                                  "chunk=64;threads=1", "chunk=64;threads=4", "chunk=64;threads=8",
                                                  "chunk=128;threads=1", "chunk=128;threads=4",
                                                  "chunk=128;threads=8"}));
    EXPECT_LT(g.size(), all.size());  // a strict subset
    o.grid_stride = 1;
    EXPECT_EQ(plan_candidates(all, o, {}), all);
    o.grid_stride = 0;
    test::expect_error(ErrorCode::Config, [&] { (void)plan_candidates(all, o, {}); });
}

TEST(Strategies, RandomIsSeededAndPinned) {
    const auto all = grid({16, 32, 64, 128}, {1, 2, 4, 8});
    auto o = test::fast_options(Strategy::Random);
    o.random_budget = 5;
    o.seed = 42;
    const auto a = plan_candidates(all, o, {});
    EXPECT_EQ(plan_candidates(all, o, {}), a);  // same seed, same picks
    ASSERT_EQ(a.size(), 5u);
    std::set<std::string> uniq;
    for (const auto& c : a) uniq.insert(c.to_string());
    EXPECT_EQ(uniq.size(), 5u);  // without replacement
    // Pinned: SplitMix64 partial Fisher-Yates (portable; not std::shuffle).
    // Expected values computed independently with a Python re-implementation (arbitrary
    // precision (r * (n - i)) >> 64), not copied from this build's output.
    EXPECT_EQ(texts(a), (std::vector<std::string>{"chunk=64;threads=8", "chunk=16;threads=8", "chunk=32;threads=2",
                                                  "chunk=32;threads=8", "chunk=32;threads=1"}));
    o.seed = 43;
    EXPECT_NE(plan_candidates(all, o, {}), a);
    o.random_budget = 100;
    EXPECT_EQ(plan_candidates(all, o, {}).size(), all.size());
}

TEST(Strategies, HeuristicKeepsBestPredictedAndBayesianIsUnsupported) {
    ManualClock clock;
    FakeOp op(clock, {{1, {.predicted_ns = 500}}, {2, {.predicted_ns = 100}}, {3, {.predicted_ns = 300}},
                      {4, {.predicted_ns = 100}}, {5, {.predicted_ns = 900}}});
    auto o = test::fast_options(Strategy::Heuristic);
    o.heuristic_keep = 3;
    const auto cost = [&](const Candidate& c) { return op.cost(c); };
    EXPECT_EQ(texts(plan_candidates(op.candidates(), o, cost)), (std::vector<std::string>{"v=2", "v=4", "v=3"}));
    o.cost_model.reset();
    test::expect_error(ErrorCode::Config, [&] { (void)plan_candidates(op.candidates(), o, cost); });
    o = test::fast_options(Strategy::Heuristic);
    test::expect_error(ErrorCode::Config, [&] { (void)plan_candidates(op.candidates(), o, {}); });

    o.strategy = Strategy::Bayesian;
    test::expect_error(ErrorCode::Unsupported, [&] { (void)plan_candidates(op.candidates(), o, cost); });
    TempDb t("bayes");
    ProfileDb db = ProfileDb::open(t.path());
    TunableOp* ops[] = {&op};
    test::expect_error(ErrorCode::Unsupported, [&] { (void)tune(db, test::sample_key(), ops, o, clock); });
    EXPECT_EQ(op.total_runs, 0);
    // Refused up front, even for an op that offers no candidates (no plan is ever made).
    FakeOp empty(clock, {});
    TunableOp* none[] = {&empty};
    test::expect_error(ErrorCode::Unsupported, [&] { (void)tune(db, test::sample_key(), none, o, clock); });
    EXPECT_EQ(db.count("operator_profile"), 0);
}

// ---- measured selection ---------------------------------------------------------------------

TEST(Tuner, PicksSmallestMedianAndPersists) {
    ManualClock clock;
    FakeOp op(clock, {{1, {.base_ns = 3000}}, {2, {.base_ns = 1000}}, {3, {.base_ns = 2000}}});
    TempDb t("pick");
    ProfileDb db = ProfileDb::open(t.path());
    TunableOp* ops[] = {&op};
    const auto r = only(tune(db, test::sample_key(), ops, test::fast_options(Strategy::Exhaustive), clock));
    ASSERT_TRUE(r.winner);
    EXPECT_EQ(r.winner->to_string(), "v=2");
    EXPECT_DOUBLE_EQ(r.winner_median_ns, 1000.0);
    EXPECT_TRUE(r.persisted);
    EXPECT_EQ(op.total_runs, 3 * (5 + 20));  // TRD §50 warm-up + measured per candidate
    for (const auto& c : r.candidates) {
        EXPECT_TRUE(c.measured);
        EXPECT_EQ(c.stats.n, 20u);
        EXPECT_EQ(c.stability, halo::profiling::Stability::Stable);
    }
    EXPECT_EQ(db.count("benchmark_run"), 3);
    const nlohmann::json j = r;
    EXPECT_EQ(j.at("winner"), "v=2");
    EXPECT_EQ(j.at("winner_metric"), "median_ns");
}

TEST(Tuner, TiesGoToTheSmallestCanonicalString) {
    ManualClock clock;
    FakeOp op(clock, {{7, {.base_ns = 1000}}, {3, {.base_ns = 1000}}});
    TempDb t("tie");
    ProfileDb db = ProfileDb::open(t.path());
    TunableOp* ops[] = {&op};
    const auto r = only(tune(db, test::sample_key(), ops, test::fast_options(Strategy::Exhaustive), clock));
    EXPECT_EQ(r.winner->to_string(), "v=3");
}

TEST(Tuner, UnstableCandidateWithBestMedianIsRejected) {
    ManualClock clock;
    // v=1 has the best median (500 ns) but alternates +-30 %: cv ~ 0.31 > 0.05.
    FakeOp op(clock, {{1, {.base_ns = 500, .jitter = 0.3}}, {2, {.base_ns = 1000}}});
    TempDb t("unstable");
    ProfileDb db = ProfileDb::open(t.path());
    TunableOp* ops[] = {&op};
    const auto r = only(tune(db, test::sample_key(), ops, test::fast_options(Strategy::Exhaustive), clock));
    ASSERT_TRUE(r.winner);
    EXPECT_EQ(r.winner->to_string(), "v=2");
    const auto& unstable = r.candidates.at(0);
    EXPECT_EQ(unstable.candidate.to_string(), "v=1");
    EXPECT_DOUBLE_EQ(unstable.stats.median, 500.0);
    EXPECT_EQ(unstable.stability, halo::profiling::Stability::Unstable);
    EXPECT_NE(unstable.rejected.find("unstable"), std::string::npos) << unstable.rejected;
}

TEST(Tuner, InvalidCandidateCannotWin) {
    ManualClock clock;
    FakeOp op(clock, {{1, {.base_ns = 100, .valid = false}}, {2, {.base_ns = 1000}}});
    TempDb t("invalid");
    ProfileDb db = ProfileDb::open(t.path());
    TunableOp* ops[] = {&op};
    const auto r = only(tune(db, test::sample_key(), ops, test::fast_options(Strategy::Exhaustive), clock));
    EXPECT_EQ(r.winner->to_string(), "v=2");
    EXPECT_FALSE(r.candidates.at(0).measured);
    EXPECT_EQ(r.candidates.at(0).rejected, "invalid: fake mismatch");
    EXPECT_EQ(op.total_runs, 25);  // the invalid candidate is never timed
}

TEST(Tuner, AllUnstableKeepsTheExistingWinner) {
    ManualClock clock;
    TempDb t("keep");
    ProfileDb db = ProfileDb::open(t.path());
    const ProfileKey key = test::sample_key();
    {
        FakeOp good(clock, {{1, {.base_ns = 1000}}, {2, {.base_ns = 2000}}});
        TunableOp* ops[] = {&good};
        ASSERT_TRUE(only(tune(db, key, ops, test::fast_options(Strategy::Exhaustive), clock)).persisted);
    }
    FakeOp noisy(clock, {{1, {.base_ns = 1000, .jitter = 0.5}}, {2, {.base_ns = 2000, .jitter = 0.5}}});
    TunableOp* ops[] = {&noisy};
    const auto r = only(tune(db, key, ops, test::fast_options(Strategy::Exhaustive), clock));
    EXPECT_FALSE(r.winner);
    EXPECT_FALSE(r.persisted);
    EXPECT_NE(r.note.find("existing winner left unchanged"), std::string::npos);
    EXPECT_EQ(db.count("benchmark_run"), 4);  // the noisy runs are still recorded
    const auto lk = ProfileLookup::open(t.path());
    const auto found = lk.find(key, {"FAKE", "S"}, "cpu");
    ASSERT_TRUE(found.candidate);
    EXPECT_EQ(found.candidate->to_string(), "v=1");
}

TEST(Tuner, HeuristicMeasuresOnlyThePrefilter) {
    ManualClock clock;
    // Prediction and reality disagree: the prefilter keeps v=1,v=2; v=3 (really fastest)
    // is never measured. That is the documented HEURISTIC trade-off.
    FakeOp op(clock, {{1, {.base_ns = 900, .predicted_ns = 10}},
                      {2, {.base_ns = 800, .predicted_ns = 20}},
                      {3, {.base_ns = 100, .predicted_ns = 30}}});
    TempDb t("heur");
    ProfileDb db = ProfileDb::open(t.path());
    auto o = test::fast_options(Strategy::Heuristic);
    o.heuristic_keep = 2;
    TunableOp* ops[] = {&op};
    const auto r = only(tune(db, test::sample_key(), ops, o, clock));
    EXPECT_EQ(r.candidates.size(), 2u);
    EXPECT_EQ(r.winner->to_string(), "v=2");
    ASSERT_TRUE(r.candidates[0].predicted);
    EXPECT_DOUBLE_EQ(r.candidates[0].predicted->total_ns, 10.0);
}

TEST(Tuner, WinnersAreKeptPerBackend) {
    ManualClock clock;
    TempDb t("backends");
    const ProfileKey key = test::sample_key();
    {
        ProfileDb db = ProfileDb::open(t.path());
        FakeOp cpu(clock, {{1, {.base_ns = 100}}, {2, {.base_ns = 200}}}, "S", "cpu");
        FakeOp vk(clock, {{1, {.base_ns = 200}}, {2, {.base_ns = 100}}}, "S", "vulkan");
        TunableOp* ops[] = {&cpu, &vk};
        (void)tune(db, key, ops, test::fast_options(Strategy::Exhaustive), clock);
        EXPECT_EQ(db.count("winning_configuration"), 2);
        // Re-tuning cpu (new winner) must not touch the vulkan winner.
        FakeOp cpu2(clock, {{1, {.base_ns = 300}}, {2, {.base_ns = 50}}}, "S", "cpu");
        TunableOp* again[] = {&cpu2};
        (void)tune(db, key, again, test::fast_options(Strategy::Exhaustive), clock);
        EXPECT_EQ(db.count("winning_configuration"), 2);
    }
    const auto lk = ProfileLookup::open(t.path());
    const auto c = lk.find(key, {"FAKE", "S"}, "cpu");
    const auto v = lk.find(key, {"FAKE", "S"}, "vulkan");
    EXPECT_EQ(c.source, SelectionSource::Exact);
    EXPECT_EQ(v.source, SelectionSource::Exact);
    ASSERT_TRUE(c.candidate && v.candidate);
    EXPECT_EQ(c.candidate->to_string(), "v=2");
    EXPECT_DOUBLE_EQ(c.value, 50.0);
    EXPECT_EQ(v.candidate->to_string(), "v=2");
    EXPECT_DOUBLE_EQ(v.value, 100.0);
    EXPECT_EQ(lk.find(key, {"FAKE", "S"}, "hip").source, SelectionSource::None);
}

TEST(Tuner, RefusesBelowMethodologyMinimum) {
    ManualClock clock;
    FakeOp op(clock, {{1, {}}});
    TempDb t("min");
    ProfileDb db = ProfileDb::open(t.path());
    TunableOp* ops[] = {&op};
    auto o = test::fast_options(Strategy::Exhaustive);
    o.iterations = 19;
    test::expect_error(ErrorCode::Config, [&] { (void)tune(db, test::sample_key(), ops, o, clock); });
    o.iterations = 20;
    o.warmup = 4;
    test::expect_error(ErrorCode::Config, [&] { (void)tune(db, test::sample_key(), ops, o, clock); });
    EXPECT_EQ(op.total_runs, 0);
}

// ---- runtime lookup (§56 steps 1-3) -----------------------------------------------------------

class LookupFixture : public ::testing::Test {
protected:
    void SetUp() override {
        ProfileDb db = ProfileDb::open(t_.path());
        ManualClock clock;
        FakeOp op(clock, {{1, {.base_ns = 2000}}, {2, {.base_ns = 1000}}});
        TunableOp* ops[] = {&op};
        (void)tune(db, test::sample_key(), ops, test::fast_options(Strategy::Exhaustive), clock);
    }
    TempDb t_{"lookup"};
    const OpKey op_{"FAKE", "S"};
};

TEST_F(LookupFixture, ExactCompatibleAndRejected) {
    const auto lk = ProfileLookup::open(t_.path());
    EXPECT_EQ(lk.size(), 1u);
    ProfileKey k = test::sample_key();
    auto r = lk.find(k, op_, "cpu");
    EXPECT_EQ(r.source, SelectionSource::Exact);
    EXPECT_EQ(r.candidate->to_string(), "v=2");
    EXPECT_TRUE(r.differing.empty());

    k.driver_version = "radv 25.3";
    k.kernel_version = "7.1.0";
    r = lk.find(k, op_, "cpu");
    EXPECT_EQ(r.source, SelectionSource::Compatible);
    EXPECT_EQ(r.differing, (std::vector<std::string>{"DRIVER_VERSION", "KERNEL_VERSION"}));
    EXPECT_EQ(r.candidate->to_string(), "v=2");

    k = test::sample_key();
    k.power_mode = "quiet|manual/COMPUTE";  // a different power mode is rejected
    r = lk.find(k, op_, "cpu");
    EXPECT_EQ(r.source, SelectionSource::None);
    EXPECT_FALSE(r.candidate);
    ASSERT_EQ(r.rejections.size(), 1u);
    EXPECT_NE(r.rejections[0].find("POWER_MODE"), std::string::npos);

    EXPECT_EQ(lk.find(test::sample_key(), {"FAKE", "other"}, "cpu").source, SelectionSource::None);
    EXPECT_EQ(lk.find(test::sample_key(), op_, "vulkan").source, SelectionSource::None);
    EXPECT_EQ(ProfileLookup::empty().find(test::sample_key(), op_, "cpu").source, SelectionSource::None);
}

TEST_F(LookupFixture, PrefersExactThenFewestDifferences) {
    ProfileKey near = test::sample_key();
    near.os = "Ubuntu 26.10";
    ProfileKey far = test::sample_key();
    far.os = "Ubuntu 27.04";
    far.halo_version = "0.3.0";
    {
        ProfileDb db = ProfileDb::open(t_.path());
        ManualClock clock;
        FakeOp a(clock, {{5, {.base_ns = 10}}});
        FakeOp b(clock, {{6, {.base_ns = 10}}});
        TunableOp* oa[] = {&a};
        TunableOp* ob[] = {&b};
        (void)tune(db, near, oa, test::fast_options(Strategy::Exhaustive), clock);
        (void)tune(db, far, ob, test::fast_options(Strategy::Exhaustive), clock);
    }
    const auto lk = ProfileLookup::open(t_.path());
    EXPECT_EQ(lk.find(test::sample_key(), op_, "cpu").candidate->to_string(), "v=2");  // exact beats compatible
    ProfileKey cur = test::sample_key();
    cur.os = "Ubuntu 28.04";  // differs from all three in OS; `far` also differs in HALO_VERSION
    const auto r = lk.find(cur, op_, "cpu");
    EXPECT_EQ(r.source, SelectionSource::Compatible);
    EXPECT_EQ(r.differing, std::vector<std::string>{"OS"});
    // Two entries (original and `near`) differ only in OS; the more recent (higher id) wins.
    EXPECT_EQ(r.candidate->to_string(), "v=5");
}

TEST_F(LookupFixture, SelectKernelWalksSteps1To3) {
    const auto lk = ProfileLookup::open(t_.path());
    ManualClock clock;
    FakeOp op(clock, {{1, {.predicted_ns = 50}}, {2, {.predicted_ns = 70}}, {3, {.predicted_ns = 20}}});
    const auto cost = [&](const Candidate& c) { return op.cost(c); };
    const CostModel model = test::unit_cost_model();

    auto r = select_kernel(lk, test::sample_key(), op_, "cpu", op.candidates(), &model, cost);
    EXPECT_EQ(r.source, SelectionSource::Exact);
    EXPECT_EQ(r.candidate->to_string(), "v=2");

    // Stored winner no longer offered (e.g. fewer threads now): falls to the heuristic.
    const std::vector<Candidate> fewer{Candidate{{{"v", 1}}}, Candidate{{{"v", 3}}}};
    r = select_kernel(lk, test::sample_key(), op_, "cpu", fewer, &model, cost);
    EXPECT_EQ(r.source, SelectionSource::Heuristic);
    EXPECT_EQ(r.candidate->to_string(), "v=3");
    EXPECT_DOUBLE_EQ(r.value, 20.0);
    EXPECT_FALSE(r.rejections.empty());

    ProfileKey other = test::sample_key();
    other.model_hash = "different-model";
    r = select_kernel(lk, other, op_, "cpu", op.candidates(), &model, cost);
    EXPECT_EQ(r.source, SelectionSource::Heuristic);
    EXPECT_EQ(r.candidate->to_string(), "v=3");
    r = select_kernel(lk, other, op_, "cpu", op.candidates(), nullptr, cost);
    EXPECT_EQ(r.source, SelectionSource::None);  // no silent default
    EXPECT_FALSE(r.candidate);
}

TEST_F(LookupFixture, SnapshotIsConcurrentReadSafe) {
    const auto lk = ProfileLookup::open(t_.path());
    std::atomic<int> exact{0};
    {
        std::vector<std::jthread> ts;
        for (int i = 0; i < 8; ++i) {
            ts.emplace_back([&] {
                for (int n = 0; n < 200; ++n) {
                    if (lk.find(test::sample_key(), op_, "cpu").source == SelectionSource::Exact) ++exact;
                }
            });
        }
    }
    EXPECT_EQ(exact.load(), 1600);
}

// ---- cost model -----------------------------------------------------------------------------

TEST(CostModel, Trd55Formula) {
    CostModel m;
    m.bandwidth_gbps[halo::hardware::MemoryTier::Host] = 50.0;
    m.bandwidth_threads = 4;
    m.gflops_per_thread = 2.0;
    m.launch_ns = 100;
    m.sync_ns = 10;
    CostInputs in;
    in.flops = 8000;       // /(2*2) = 2000 ns with 2 threads
    in.bytes = 50000;      // /(50 * 2/4) = 2000 ns... bandwidth scales with 2/4 threads
    in.parallelism = 2;
    in.launches = 1;
    in.syncs = 3;
    auto e = m.estimate(in);
    EXPECT_DOUBLE_EQ(e.t_compute_ns, 2000.0);
    EXPECT_DOUBLE_EQ(e.t_memory_ns, 2000.0);
    EXPECT_DOUBLE_EQ(e.total_ns, 2000.0 + 30.0 + 100.0);
    EXPECT_TRUE(e.memory_bound);
    in.parallelism = 8;  // memory saturates at 4 threads, compute keeps scaling
    e = m.estimate(in);
    EXPECT_DOUBLE_EQ(e.t_compute_ns, 500.0);
    EXPECT_DOUBLE_EQ(e.t_memory_ns, 1000.0);
    in.tier = halo::hardware::MemoryTier::Vram;
    test::expect_error(ErrorCode::Config, [&] { (void)m.estimate(in); });
    m.gflops_per_thread = 0;
    in.tier = halo::hardware::MemoryTier::Host;
    test::expect_error(ErrorCode::Config, [&] { (void)m.estimate(in); });

    EXPECT_DOUBLE_EQ(calibrate_gflops_per_thread(4e6, 1e6, 2), 2.0);
    test::expect_error(ErrorCode::Config, [] { (void)calibrate_gflops_per_thread(1, 0, 1); });
}

TEST(CostModel, FromMeasuredTierBandwidthUsesMedian) {
    halo::hardware::TierBandwidth host;
    host.tier = halo::hardware::MemoryTier::Host;
    host.threads = 8;
    host.read_gbps.median = 60.0;
    host.read_gbps.mean = 99.0;  // must not be used
    halo::hardware::TierBandwidth gtt = host;
    gtt.tier = halo::hardware::MemoryTier::Gtt;
    gtt.read_gbps.median = 200.0;
    const std::vector<halo::hardware::TierBandwidth> v{host, gtt};
    const CostModel m = CostModel::from_measurements(v, 1.5);
    EXPECT_DOUBLE_EQ(m.bandwidth_gbps.at(halo::hardware::MemoryTier::Host), 60.0);
    EXPECT_DOUBLE_EQ(m.bandwidth_gbps.at(halo::hardware::MemoryTier::Gtt), 200.0);
    EXPECT_EQ(m.bandwidth_threads, 8u);
    auto bad = v;
    bad[1].threads = 4;
    test::expect_error(ErrorCode::Config, [&] { (void)CostModel::from_measurements(bad, 1.0); });
    bad = v;
    bad[0].read_gbps.median = 0;
    test::expect_error(ErrorCode::Config, [&] { (void)CostModel::from_measurements(bad, 1.0); });
    test::expect_error(ErrorCode::Config, [] { (void)CostModel::from_measurements({}, 1.0); });
}
