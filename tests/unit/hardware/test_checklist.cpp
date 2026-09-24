#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>

#include <nlohmann/json.hpp>

#include "halo/hardware/checklist.h"

using namespace halo::hardware;

namespace {

HardwareInfo load(const char* name) {
    DiscoveryOptions o;
    o.root = std::filesystem::path(HALO_SOURCE_DIR) / "tests" / "fixtures" / "sysfs" / name;
    return discover(o);
}

CheckStatus status_of(const std::vector<CheckItem>& items, std::string_view check) {
    for (const auto& i : items)
        if (i.check == check) return i.status;
    ADD_FAILURE() << "check " << check << " missing";
    return CheckStatus::Unknown;
}

void print(const char* name, const std::vector<CheckItem>& items) {
    std::cout << "[fixture " << name << "]\n";
    for (const auto& i : items) std::cout << "  " << to_string(i.status) << "  " << i.check << ": " << i.detail << "\n";
}

constexpr std::uint64_t GiB = 1ULL << 30;

}  // namespace

TEST(Checklist, EvoX2LargeCarveoutIsValid) {
    const auto items = deployment_checklist(load("evo_x2"));
    print("evo_x2", items);
    ASSERT_EQ(items.size(), 9u);
    EXPECT_EQ(status_of(items, "amd_gpu"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "gfx_target"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "carveout"), CheckStatus::Ok);  // D-002: 96 GiB is not a failure
    EXPECT_EQ(status_of(items, "gtt"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "iommu"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "vulkan_driver"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "kernel"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "rocm"), CheckStatus::Ok);
    EXPECT_EQ(status_of(items, "performance_mode"), CheckStatus::Warn);  // fixture: "auto"
    EXPECT_EQ(overall_status(items), CheckStatus::Warn);
}

TEST(Checklist, PrdConfigFlagsAmdvlkIommuKernelRocm) {
    const auto items = deployment_checklist(load("prd_gtt_config"));
    print("prd_gtt_config", items);
    EXPECT_EQ(status_of(items, "carveout"), CheckStatus::Ok);  // 1 GiB: TRD small-carveout layout
    EXPECT_EQ(status_of(items, "gtt"), CheckStatus::Ok);       // 120 GiB
    EXPECT_EQ(status_of(items, "iommu"), CheckStatus::Warn);
    EXPECT_EQ(status_of(items, "vulkan_driver"), CheckStatus::Warn);
    EXPECT_EQ(status_of(items, "kernel"), CheckStatus::Warn);
    EXPECT_EQ(status_of(items, "rocm"), CheckStatus::Warn);
    EXPECT_EQ(status_of(items, "performance_mode"), CheckStatus::Ok);  // "high"
}

TEST(Checklist, NoGpuFails) {
    const auto items = deployment_checklist(load("no_gpu"));
    EXPECT_EQ(status_of(items, "amd_gpu"), CheckStatus::Fail);
    EXPECT_EQ(status_of(items, "carveout"), CheckStatus::Unknown);
    EXPECT_EQ(status_of(items, "performance_mode"), CheckStatus::Unknown);
    EXPECT_EQ(overall_status(items), CheckStatus::Fail);
}

TEST(Checklist, IntermediateCarveoutAndSmallGttWarn) {
    auto hw = load("prd_gtt_config");
    hw.tiers.vram_total = 16 * GiB;  // neither layout
    EXPECT_EQ(status_of(deployment_checklist(hw), "carveout"), CheckStatus::Warn);
    hw = load("prd_gtt_config");
    hw.tiers.gtt_total = 16 * GiB;  // small carveout but default-sized GTT
    EXPECT_EQ(status_of(deployment_checklist(hw), "gtt"), CheckStatus::Warn);
}

TEST(Checklist, OverallStatusOrdering) {
    EXPECT_EQ(overall_status({}), CheckStatus::Ok);
    EXPECT_EQ(overall_status({{"a", CheckStatus::Ok, ""}, {"b", CheckStatus::Unknown, ""}}), CheckStatus::Unknown);
    EXPECT_EQ(overall_status({{"a", CheckStatus::Warn, ""}, {"b", CheckStatus::Unknown, ""}}), CheckStatus::Warn);
    EXPECT_EQ(overall_status({{"a", CheckStatus::Fail, ""}, {"b", CheckStatus::Warn, ""}}), CheckStatus::Fail);
}

TEST(Checklist, Json) {
    const nlohmann::json j = deployment_checklist(load("evo_x2"));
    ASSERT_TRUE(j.is_array());
    EXPECT_EQ(j[0]["check"], "amd_gpu");
    EXPECT_EQ(j[0]["status"], "ok");
    EXPECT_TRUE(j[0]["detail"].is_string());
}
