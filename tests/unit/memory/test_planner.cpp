// Planner tests on hand-built tier descriptions (no hardware discovery involved).
#include <gtest/gtest.h>

#include <algorithm>
#include <format>
#include <iostream>
#include <limits>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"
#include "halo/memory/planner.h"
#include "qwen_constants.h"

using namespace halo::memory;
using namespace halo_test;
using halo::hardware::MemoryTiers;
using halo::hardware::MemoryTopology;

namespace {

/// The "32-GiB-only" topology: no GPU tiers, 32 GiB of OS RAM (30 GiB available).
MemoryTiers host_only_32g() {
    MemoryTiers t;
    t.topology = MemoryTopology::NoGpu;
    t.host_total = 32 * GiB;
    t.host_available = 30 * GiB;
    return t;
}

PlanRequest mtp2(std::uint64_t ctx, std::uint32_t seqs, KvLayout layout) {
    PlanRequest r;
    r.max_context = ctx;
    r.max_sequences = seqs;
    r.kv_layout = layout;
    r.mtp_enabled = true;
    r.mtp_draft_depth = 2;
    return r;
}

// Sum of every class the planner should place for 262144 × 8, MTP n=2, on a host-only
// topology — written out from the D-003/TRD constants, not from the planner's code.
std::uint64_t expected_host_only_demand(KvLayout layout) {
    const std::uint64_t cells = layout == KvLayout::Unified ? 262144ULL : 262144ULL * 8;
    const std::uint64_t kv = cells * kKvF16PerToken;
    const std::uint64_t mtp_kv = cells * kMtpKvF16PerToken;
    const std::uint64_t live_state = 8 * kGdnStatePerCopy;
    const std::uint64_t rollback = 3 * 8 * kGdnStatePerCopy;             // n + 1 = 3 copies
    const std::uint64_t logits = 8ULL * 3 * 248320 * 4;                  // seqs × (1 + n) rows, fp32
    const std::uint64_t activations = 256ULL * 4 * (2 * 5120 + 2 * 17408);  // ubatch × widest (FFN)
    const std::uint64_t workspace = 256ULL * 4 * 48 * (64 + 128 + 128);
    const std::uint64_t host_overhead = 1 * GiB;  // no GPU: GPU overhead not planned
    return kUdWeightsTotal + kv + mtp_kv + live_state + rollback + logits + activations + workspace + host_overhead;
}

}  // namespace

TEST(PlannerSizes, MatchDecisionsFacts) {
    const auto s = compute_sizes(qwen(), mtp2(131072, 8, KvLayout::Unified));
    EXPECT_EQ(s.kv_bytes_per_token, 64 * KiB);
    EXPECT_EQ(s.kv_cells, 131072u);
    // 131072 × 64 KiB is exactly 8 GiB (the engine report's "~8.4 GiB" mixes GB and GiB).
    EXPECT_EQ(s.kv, 8 * GiB);
    EXPECT_EQ(s.mtp_kv, 131072 * kMtpKvF16PerToken);  // one attention layer: 512 MiB
    EXPECT_EQ(s.mtp_kv, 512 * MiB);
    EXPECT_EQ(s.gdn_recurrent, 8 * 144 * MiB);
    EXPECT_EQ(s.gdn_conv, 8 * kGdnConvPerSeq);
    EXPECT_EQ(s.gdn_state_per_copy, kGdnStatePerCopy);  // 149.625 MiB
    EXPECT_EQ(s.gdn_rollback_copies, 3u);             // draft depth 2 -> K = n + 1
    EXPECT_EQ(s.gdn_rollback, 3 * 8 * kGdnStatePerCopy);
    EXPECT_EQ(s.prefix_checkpoints, 0u);  // opt-in
    EXPECT_EQ(s.weights.total(), kUdWeightsTotal);
}

TEST(PlannerSizes, PerSequenceLayoutAndQuantizedKv) {
    auto r = mtp2(131072, 8, KvLayout::PerSequence);
    EXPECT_EQ(compute_sizes(qwen(), r).kv, 64 * GiB);
    r.kv_dtype = DtypeSize::q8_0();
    // q8_0: 256 elems = 8 blocks × 34 B = 272 B per head row (not the report's "~32 KiB/token").
    EXPECT_EQ(compute_sizes(qwen(), r).kv_bytes_per_token, 16ULL * 4 * (272 + 272));
    r.mtp_enabled = false;
    const auto s = compute_sizes(qwen(), r);
    EXPECT_EQ(s.mtp_kv, 0u);
    EXPECT_EQ(s.gdn_rollback, 0u);
    EXPECT_EQ(s.weights.mtp, 0u);  // blk.64 not resident without MTP
    EXPECT_EQ(s.weights.total(), kUdWeightsTotal - ud_q4_k_xl_weights().mtp);
}

TEST(PlannerSizes, PrefixCheckpointsDerivedCount) {
    auto r = mtp2(131072, 8, KvLayout::Unified);
    r.prefix_checkpoints.enabled = true;
    const auto s = compute_sizes(qwen(), r);
    EXPECT_EQ(s.prefix_checkpoints_per_slot, 17u);  // 131072 / 8192 + 1
    EXPECT_EQ(s.prefix_checkpoints, 17ULL * 8 * kGdnStatePerCopy);
    r.prefix_checkpoints.max_per_slot = 4;
    EXPECT_EQ(compute_sizes(qwen(), r).prefix_checkpoints, 4ULL * 8 * kGdnStatePerCopy);
}

TEST(PlannerSizes, RejectsMisalignedQuantizedRows) {
    auto m = qwen();
    m.key_dim = 100;  // not a multiple of the q8_0 block (32)
    PlanRequest r;
    r.kv_dtype = DtypeSize::q8_0();
    EXPECT_THROW((void)compute_sizes(m, r), halo::Error);
}

TEST(PlannerOverflow, CheckedArithmetic) {
    EXPECT_EQ(checked_mul(1ULL << 31, 1ULL << 32), 1ULL << 63);
    EXPECT_FALSE(checked_mul(1ULL << 32, 1ULL << 32));
    EXPECT_FALSE(checked_add(std::numeric_limits<std::uint64_t>::max(), 1));
    EXPECT_EQ(checked_add(1, 2), 3u);
    EXPECT_THROW((void)mul_or_throw(1ULL << 40, 1ULL << 40, "x"), halo::Error);
}

TEST(PlannerOverflow, HugeContextIsTypedConfigError) {
    PlanRequest r;
    r.max_context = 1ULL << 62;
    r.max_sequences = 8;
    try {
        (void)plan_memory(qwen(), r, host_only_32g());
        FAIL() << "expected overflow error";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Config);
        EXPECT_NE(std::string(e.what()).find("overflow"), std::string::npos) << e.what();
    }
    // Weight bytes that overflow when summed.
    auto m = qwen();
    m.weights.ffn = std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW((void)compute_sizes(m, PlanRequest{}), halo::Error);
}

TEST(PlannerRefusal, Context262kTimes8On32GiBOnlyIsRefused) {
    const auto tiers = host_only_32g();
    const std::uint64_t budget = 30 * GiB * 9 / 10;  // floor(available × 0.9), exact here
    for (const auto layout : {KvLayout::Unified, KvLayout::PerSequence}) {
        const auto plan = plan_memory(qwen(), mtp2(262144, 8, layout), tiers);
        std::cout << plan.table();
        ASSERT_FALSE(plan.ok);
        ASSERT_FALSE(plan.overflows.empty());
        const auto& o = plan.overflows.front();
        EXPECT_EQ(o.constraint, "HOST");
        EXPECT_EQ(o.budget, budget);
        EXPECT_EQ(o.demand, expected_host_only_demand(layout));
        EXPECT_EQ(o.excess, expected_host_only_demand(layout) - budget);
        try {
            plan.require_ok();
            FAIL() << "expected MEMORY_ERROR";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Memory);
            const std::string what = e.what();
            EXPECT_NE(what.find("MEMORY_ERROR"), std::string::npos);
            EXPECT_NE(what.find("HOST"), std::string::npos);
            EXPECT_NE(what.find(std::format("short by {:.2f} GiB", static_cast<double>(o.excess) / GiB)), std::string::npos)
                << what;
        }
    }
}

// GTT pages come out of OS RAM: a plan whose GTT placements fit the GTT budget but not
// the OS pool (together with host-side allocations) must be refused on HOST.
TEST(PlannerPolicy, GttCountsAgainstTheOsPool) {
    MemoryTiers t;
    t.topology = MemoryTopology::GttPrimary;
    t.vram_total = 1 * GiB;
    t.vram_used = 0;
    t.gtt_total = 60 * GiB;  // budget 54 GiB
    t.gtt_used = 0;
    t.host_total = 64 * GiB;
    t.host_available = 40 * GiB;  // budget 36 GiB
    ModelShape m = qwen();
    m.weights = {};
    m.weights.ffn = 35 * GiB + 512 * MiB;  // fits GTT alone
    PlanRequest r;
    r.max_context = 1024;
    const auto plan = plan_memory(m, r, t);
    std::cout << plan.table();
    EXPECT_EQ(plan.tier_of(Component::FfnWeights), halo::hardware::MemoryTier::Gtt);
    ASSERT_FALSE(plan.ok);
    EXPECT_EQ(plan.overflows.front().constraint, "HOST");
    EXPECT_TRUE(std::ranges::none_of(plan.overflows, [](const Overflow& o) { return o.constraint == "GTT"; }));
}

TEST(PlannerPolicy, OverridePinsTierAndCannotTargetMissingTier) {
    MemoryTiers t;
    t.topology = MemoryTopology::CarveoutPrimary;
    t.vram_total = 96 * GiB;
    t.gtt_total = 16 * GiB;
    t.host_total = 32 * GiB;
    t.host_available = 28 * GiB;
    PlanRequest r;
    r.tier_overrides[Component::KvCache] = halo::hardware::MemoryTier::Gtt;
    const auto plan = plan_memory(qwen(), r, t);
    ASSERT_TRUE(plan.ok) << plan.table();
    EXPECT_EQ(plan.tier_of(Component::KvCache), halo::hardware::MemoryTier::Gtt);
    EXPECT_EQ(plan.tier_of(Component::FfnWeights), halo::hardware::MemoryTier::Vram);

    // An override never bypasses the budget: 64 GiB KV forced into a 14.4 GiB GTT budget.
    r.max_context = 131072;
    r.max_sequences = 8;
    const auto over = plan_memory(qwen(), r, t);
    EXPECT_FALSE(over.ok);
    EXPECT_EQ(over.tier_of(Component::KvCache), halo::hardware::MemoryTier::Gtt);  // no silent fallback to VRAM
    // Largest excess first: GTT (64 GiB vs 14.4 GiB) ahead of HOST (65 GiB vs 25.2 GiB).
    ASSERT_EQ(over.overflows.size(), 2u);
    EXPECT_EQ(over.overflows[0].constraint, "GTT");
    EXPECT_EQ(over.overflows[1].constraint, "HOST");

    PlanRequest bad;
    bad.tier_overrides[Component::KvCache] = halo::hardware::MemoryTier::Vram;
    EXPECT_THROW((void)plan_memory(qwen(), bad, host_only_32g()), halo::Error);
}

TEST(PlannerPolicy, SafetyFactorAndUserCap) {
    PlanRequest r;
    r.safety_factor = 1.0;
    EXPECT_THROW((void)plan_memory(qwen(), r, host_only_32g()), halo::Error);
    r.safety_factor = 0.0;
    EXPECT_THROW((void)plan_memory(qwen(), r, host_only_32g()), halo::Error);
    r.safety_factor = 0.9;
    const auto base = plan_memory(qwen(), r, host_only_32g());
    ASSERT_TRUE(base.ok) << base.table();
    for (const auto& t : base.tiers) EXPECT_EQ(t.reserve, t.available - t.budget);
    r.max_memory = 16 * GiB;  // below the weights alone
    const auto capped = plan_memory(qwen(), r, host_only_32g());
    ASSERT_FALSE(capped.ok);
    EXPECT_EQ(capped.overflows.front().constraint, "MAX_MEMORY");
    EXPECT_EQ(capped.overflows.front().excess, base.planned_total - 16 * GiB);
}

TEST(PlannerCheckpoints, NeverInHost) {
    PlanRequest r;
    r.prefix_checkpoints.enabled = true;
    EXPECT_THROW((void)plan_memory(qwen(), r, host_only_32g()), halo::Error);  // no GPU tier
    MemoryTiers t = host_only_32g();
    t.topology = MemoryTopology::CarveoutPrimary;
    t.vram_total = 96 * GiB;
    r.tier_overrides[Component::PrefixCheckpoints] = halo::hardware::MemoryTier::Host;
    EXPECT_THROW((void)plan_memory(qwen(), r, t), halo::Error);
}

// Checkpoints that fit GTT but not VRAM: allowed into GTT in the gtt-primary layout, but
// never in the carveout-primary layout, where GTT pages are the OS RAM pool.
TEST(PlannerCheckpoints, CarveoutPrimaryKeepsThemOutOfGtt) {
    PlanRequest r;
    r.max_context = 32768;
    r.prefix_checkpoints.enabled = true;
    r.prefix_checkpoints.max_per_slot = 20;  // 20 × 149.6 MiB ≈ 2.9 GiB, explicit
    MemoryTiers carve;
    carve.topology = MemoryTopology::CarveoutPrimary;
    carve.vram_total = 20 * GiB;  // budget 18 GiB: weights + state leave ~1.1 GiB
    carve.gtt_total = 16 * GiB;   // budget 14.4 GiB: would hold the checkpoints
    carve.host_total = 64 * GiB;
    carve.host_available = 60 * GiB;
    const auto refused = plan_memory(qwen(), r, carve);
    std::cout << refused.table();
    EXPECT_FALSE(refused.ok);
    EXPECT_EQ(refused.tier_of(Component::PrefixCheckpoints), halo::hardware::MemoryTier::Vram);  // overcommitted
    ASSERT_FALSE(refused.overflows.empty());
    EXPECT_EQ(refused.overflows.front().constraint, "VRAM");

    // Derived count on the same unit (32768 / 1024 + 1 = 33 per slot): shrinks inside VRAM
    // rather than spilling into GTT.
    r.prefix_checkpoints.max_per_slot.reset();
    r.prefix_checkpoints.spacing_tokens = 1024;
    const auto shrunk = plan_memory(qwen(), r, carve);
    ASSERT_TRUE(shrunk.ok) << shrunk.table();
    EXPECT_EQ(shrunk.prefix_checkpoints_requested, 33u);
    EXPECT_LT(shrunk.sizes.prefix_checkpoints_per_slot, 33u);
    EXPECT_NE(shrunk.tier_of(Component::PrefixCheckpoints), halo::hardware::MemoryTier::Gtt);

    // Same request in the gtt-primary layout: GTT is a GPU pool there, checkpoints may use it.
    MemoryTiers gttp = carve;
    gttp.topology = MemoryTopology::GttPrimary;
    gttp.vram_total = 1 * GiB;
    gttp.gtt_total = 60 * GiB;
    r.prefix_checkpoints.max_per_slot = 20;
    const auto ok = plan_memory(qwen(), r, gttp);
    ASSERT_TRUE(ok.ok) << ok.table();
    EXPECT_EQ(ok.tier_of(Component::PrefixCheckpoints), halo::hardware::MemoryTier::Gtt);
}

TEST(PlannerSearch, MaxContextBoundaryIsExact) {
    PlanRequest r;
    r.kv_layout = KvLayout::Unified;
    const auto tiers = host_only_32g();
    const auto n = max_context_for_budget(qwen(), r, tiers);
    ASSERT_GT(n, 0u);
    ASSERT_LT(n, 262144u);  // the cap is not what stopped the search
    r.max_context = n;
    EXPECT_TRUE(plan_memory(qwen(), r, tiers).ok);
    r.max_context = n + 1;
    EXPECT_FALSE(plan_memory(qwen(), r, tiers).ok);
    std::cout << "max_context on 32 GiB host-only (unified, no MTP): " << n << "\n";

    // An upper bound so large that the KV arithmetic overflows still yields the same answer.
    PlanRequest r2;
    r2.kv_layout = KvLayout::PerSequence;
    r2.max_sequences = 8;
    const auto n2 = max_context_for_budget(qwen(), r2, tiers, 1ULL << 62);
    EXPECT_EQ(n2, max_context_for_budget(qwen(), r2, tiers));
}

TEST(PlannerSearch, MaxSequencesBoundaryAndCap) {
    PlanRequest r;
    r.max_context = 32768;
    const auto tiers = host_only_32g();
    const auto n = max_sequences_for_budget(qwen(), r, tiers);
    ASSERT_GT(n, 0u);
    ASSERT_LT(n, 256u);
    r.max_sequences = n;
    EXPECT_TRUE(plan_memory(qwen(), r, tiers).ok);
    r.max_sequences = n + 1;
    EXPECT_FALSE(plan_memory(qwen(), r, tiers).ok);

    // A model that fits at any size returns the cap.
    ModelShape tiny = qwen();
    tiny.weights = {};
    tiny.weights.ffn = 1 * MiB;
    PlanRequest small;
    small.max_context = 16;
    small.host_runtime_overhead = 0;
    EXPECT_EQ(max_sequences_for_budget(tiny, small, tiers, 4), 4u);

    // Nothing fits -> 0.
    MemoryTiers none = tiers;
    none.host_available = 1 * GiB;
    EXPECT_EQ(max_sequences_for_budget(qwen(), r, none), 0u);
    EXPECT_EQ(max_context_for_budget(qwen(), r, none), 0u);
}

TEST(PlannerOutput, JsonAndTable) {
    const auto plan = plan_memory(qwen(), PlanRequest{}, host_only_32g());
    const nlohmann::json j = plan;
    EXPECT_EQ(j["status"], "ok");
    EXPECT_EQ(j["sizes"]["weights_total"], kUdWeightsTotal - ud_q4_k_xl_weights().mtp);  // MTP off
    EXPECT_TRUE(j["allocations"].is_array());
    EXPECT_EQ(j["tiers"][0]["tier"], "HOST");
    EXPECT_NE(plan.table().find("HOST"), std::string::npos);
    EXPECT_NE(plan.table().find("OK: fits"), std::string::npos);
}
