// Planner on tiers discovered from the sysfs fixture trees (fixture-tested, not measured on
// the EVO-X2). Needs halo_hardware for discover().
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "halo/core/error.h"
#include "halo/hardware/hardware.h"
#include "halo/memory/planner.h"
#include "qwen_constants.h"

using namespace halo::memory;
using namespace halo_test;
using halo::hardware::MemoryTier;

namespace {

halo::hardware::MemoryTiers tiers_of(const char* name) {
    halo::hardware::DiscoveryOptions o;
    o.root = std::filesystem::path(HALO_SOURCE_DIR) / "tests" / "fixtures" / "sysfs" / name;
    o.probe_runtime_cpuid = false;
    return halo::hardware::discover(o).tiers;
}

PlanRequest engine_report_config(KvLayout layout) {
    // llama-server config of the engine report: --ctx-size 131072 --parallel 8 --kv-unified
    // f16 KV, --spec-draft-n-max 2, -b 512 -ub 256.
    PlanRequest r;
    r.max_context = 131072;
    r.max_sequences = 8;
    r.kv_layout = layout;
    r.mtp_enabled = true;
    r.mtp_draft_depth = 2;
    r.batch = 512;
    r.ubatch = 256;
    return r;
}

void show(const char* title, const MemoryPlan& p) { std::cout << "=== " << title << " ===\n" << p.table(); }

}  // namespace

// The engine report's configuration (unified 131k-cell KV pool shared by 8 slots) with the
// architecture-review additions: n=2 rollback copies and prefix-cache checkpoints.
TEST(PlannerEvoX2, Unified131kTimes8WithRollbackAndCheckpointsFits) {
    const auto tiers = tiers_of("evo_x2");
    auto r = engine_report_config(KvLayout::Unified);
    r.prefix_checkpoints.enabled = true;
    const auto plan = plan_memory(qwen(), r, tiers);
    show("evo_x2: 131072 unified x 8, MTP n=2, checkpoints", plan);
    ASSERT_TRUE(plan.ok) << plan.message;
    EXPECT_EQ(plan.sizes.kv, 8 * GiB);
    EXPECT_EQ(plan.sizes.gdn_rollback, 3 * 8 * kGdnStatePerCopy);
    EXPECT_EQ(plan.sizes.prefix_checkpoints_per_slot, 17u);
    EXPECT_EQ(plan.prefix_checkpoints_requested, 17u);
    EXPECT_EQ(plan.sizes.prefix_checkpoints, 17 * 8 * kGdnStatePerCopy);
    // Everything fits the carveout: nothing spills into GTT or the 32 GiB OS pool.
    for (const auto& a : plan.allocations) {
        if (a.component == Component::HostRuntimeOverhead) EXPECT_EQ(a.tier, MemoryTier::Host);
        else EXPECT_EQ(a.tier, MemoryTier::Vram) << to_string(a.component);
    }
    EXPECT_EQ(plan.tier_of(Component::PrefixCheckpoints), MemoryTier::Vram);
    EXPECT_EQ(plan.placed_on(MemoryTier::Gtt), 0u);
}

// Per-sequence 131k (64 GiB KV): the load-bearing checks are the component bytes; the
// verdict is whatever the policy yields and is printed for the report.
TEST(PlannerEvoX2, PerSequence131kTimes8Sizes) {
    const auto tiers = tiers_of("evo_x2");
    const auto plan = plan_memory(qwen(), engine_report_config(KvLayout::PerSequence), tiers);
    show("evo_x2: 131072 per-sequence x 8, MTP n=2", plan);
    EXPECT_EQ(plan.sizes.kv, 64 * GiB);
    EXPECT_EQ(plan.sizes.mtp_kv, 4 * GiB);
    EXPECT_EQ(plan.sizes.gdn_recurrent + plan.sizes.gdn_conv, 8 * kGdnStatePerCopy);
    const auto& vram = plan.tiers.front();
    EXPECT_EQ(vram.tier, MemoryTier::Vram);
    EXPECT_EQ(vram.available, (98304ULL - 512) * MiB);
    // Observed policy result: fits, but only by spilling weight classes to GTT.
    EXPECT_TRUE(plan.ok) << plan.message;
    EXPECT_GT(plan.placed_on(MemoryTier::Gtt), 0u);
}

TEST(PlannerEvoX2, TightMemoryShrinksDerivedCheckpointsOrRefusesExplicitOnes) {
    const auto tiers = tiers_of("evo_x2");
    auto r = engine_report_config(KvLayout::PerSequence);
    r.prefix_checkpoints.enabled = true;  // derived count: 17 per slot = 19.9 GiB more
    const auto shrunk = plan_memory(qwen(), r, tiers);
    show("evo_x2: per-sequence 131k x 8 + derived checkpoints (shrink-to-fit)", shrunk);
    ASSERT_TRUE(shrunk.ok) << shrunk.message;
    EXPECT_EQ(shrunk.prefix_checkpoints_requested, 17u);
    const auto kept = shrunk.sizes.prefix_checkpoints_per_slot;
    EXPECT_LT(kept, 17u);
    EXPECT_TRUE(std::ranges::any_of(shrunk.notes, [](const std::string& n) { return n.find("shrunk") != std::string::npos; }));
    // The shrink is maximal: one more checkpoint per slot does not fit.
    auto one_more = r;
    one_more.prefix_checkpoints.max_per_slot = kept + 1;
    EXPECT_FALSE(plan_memory(qwen(), one_more, tiers).ok);
    // carveout-primary: checkpoints stay in VRAM (GTT here is the 32 GiB OS pool).
    if (kept > 0) EXPECT_EQ(shrunk.tier_of(Component::PrefixCheckpoints), MemoryTier::Vram);

    // Explicit configuration is honoured or refused, never shrunk.
    auto explicit_cfg = r;
    explicit_cfg.prefix_checkpoints.max_per_slot = 17;
    const auto refused = plan_memory(qwen(), explicit_cfg, tiers);
    show("evo_x2: per-sequence 131k x 8 + explicit 17 checkpoints", refused);
    EXPECT_FALSE(refused.ok);
    EXPECT_EQ(refused.sizes.prefix_checkpoints_per_slot, 17u);
    EXPECT_THROW(refused.require_ok(), halo::Error);
}

// The budget search must not buy slots by silently shrinking requested checkpoints.
TEST(PlannerEvoX2, MaxSequencesKeepsRequestedCheckpoints) {
    const auto tiers = tiers_of("evo_x2");
    auto with = engine_report_config(KvLayout::PerSequence);
    with.prefix_checkpoints.enabled = true;
    auto without = engine_report_config(KvLayout::PerSequence);
    const auto n_with = max_sequences_for_budget(qwen(), with, tiers);
    const auto n_without = max_sequences_for_budget(qwen(), without, tiers);
    std::cout << "evo_x2 per-sequence 131k, MTP n=2: max sequences " << n_without << " without checkpoints, "
              << n_with << " with 17/slot\n";
    ASSERT_GT(n_with, 0u);
    EXPECT_LT(n_with, n_without);
    with.max_sequences = n_with;
    const auto plan = plan_memory(qwen(), with, tiers);
    ASSERT_TRUE(plan.ok);
    EXPECT_EQ(plan.sizes.prefix_checkpoints_per_slot, 17u);  // not shrunk
}

// PRD topology (1 GiB carveout, 120 GiB GTT): with 8 sequences the GDN state (1.2 GiB live
// + 3.5 GiB rollback) cannot fit the ~0.68 GiB carveout budget, so state and KV go to GTT.
TEST(PlannerPrd, StateAndKvLandInGtt) {
    const auto tiers = tiers_of("prd_gtt_config");
    const auto plan = plan_memory(qwen(), engine_report_config(KvLayout::Unified), tiers);
    show("prd_gtt_config: 131072 unified x 8, MTP n=2", plan);
    ASSERT_TRUE(plan.ok) << plan.message;
    EXPECT_EQ(plan.tier_of(Component::GdnRecurrentState), MemoryTier::Gtt);
    EXPECT_EQ(plan.tier_of(Component::GdnRollbackState), MemoryTier::Gtt);
    EXPECT_EQ(plan.tier_of(Component::KvCache), MemoryTier::Gtt);
    EXPECT_EQ(plan.tier_of(Component::FfnWeights), MemoryTier::Gtt);
}

// Fastest tier that fits: one sequence's 144 MiB recurrent state does fit the carveout.
TEST(PlannerPrd, SingleSequenceStatePrefersCarveout) {
    const auto tiers = tiers_of("prd_gtt_config");
    PlanRequest r;
    r.max_context = 32768;
    r.max_sequences = 1;
    const auto plan = plan_memory(qwen(), r, tiers);
    ASSERT_TRUE(plan.ok) << plan.message;
    EXPECT_EQ(plan.tier_of(Component::GdnRecurrentState), MemoryTier::Vram);
    EXPECT_EQ(plan.tier_of(Component::KvCache), MemoryTier::Gtt);  // 2 GiB > carveout budget
}

// Priority, not size, decides who gets the carveout: at 10240 tokens the KV (640 MiB) and
// the recurrent state (144 MiB) each fit the ~0.68 GiB VRAM budget alone, but not together.
// The per-token GDN state must win.
TEST(PlannerPrd, StateBeatsKvForTheCarveout) {
    const auto tiers = tiers_of("prd_gtt_config");
    PlanRequest r;
    r.max_context = 10240;
    r.max_sequences = 1;
    const auto plan = plan_memory(qwen(), r, tiers);
    ASSERT_TRUE(plan.ok) << plan.message;
    const auto vram_budget = plan.tiers.front().budget;
    ASSERT_LE(plan.sizes.kv, vram_budget);  // KV alone would fit ...
    ASSERT_GT(plan.sizes.kv + plan.sizes.gdn_recurrent, vram_budget);  // ... but not with the state
    EXPECT_EQ(plan.tier_of(Component::GdnRecurrentState), MemoryTier::Vram);
    EXPECT_EQ(plan.tier_of(Component::KvCache), MemoryTier::Gtt);
}

TEST(PlannerNoGpu, EverythingInHost) {
    const auto tiers = tiers_of("no_gpu");
    PlanRequest r;
    r.max_context = 32768;
    const auto plan = plan_memory(qwen(), r, tiers);
    show("no_gpu: 32768 x 1", plan);
    ASSERT_TRUE(plan.ok) << plan.message;
    for (const auto& a : plan.allocations) EXPECT_EQ(a.tier, MemoryTier::Host);
    EXPECT_FALSE(plan.tier_of(Component::GpuRuntimeOverhead));
    ASSERT_EQ(plan.tiers.size(), 1u);
}
