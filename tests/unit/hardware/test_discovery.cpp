#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"
#include "halo/hardware/hardware.h"

using namespace halo::hardware;

namespace {

std::filesystem::path fixture(const char* name) {
    return std::filesystem::path(HALO_SOURCE_DIR) / "tests" / "fixtures" / "sysfs" / name;
}

HardwareInfo discover_fixture(const char* name, std::map<std::string, std::string> env = {}) {
    DiscoveryOptions o;
    o.root = fixture(name);
    o.env = std::move(env);
    o.use_process_env = false;  // fixtures must not see the dev host's VK_* variables
    return discover(o);
}

constexpr std::uint64_t GiB = 1ULL << 30;

}  // namespace

TEST(Discovery, EvoX2Cpu) {
    const auto hw = discover_fixture("evo_x2");
    EXPECT_EQ(hw.cpu.model_name, "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S");
    EXPECT_EQ(hw.cpu.logical_threads, 32u);
    EXPECT_EQ(hw.cpu.physical_cores, 16u);
    EXPECT_EQ(hw.cpu.sockets, 1u);
    EXPECT_TRUE(hw.cpu.simd.avx512f && hw.cpu.simd.avx512bw && hw.cpu.simd.avx512vl && hw.cpu.simd.avx512_vnni &&
                hw.cpu.simd.avx512_bf16 && hw.cpu.simd.avx2 && hw.cpu.simd.fma);
    // Two 32 MiB L3 instances; cpu16 repeats cpu0's shared_cpu_list and must not be re-added.
    EXPECT_EQ(hw.cpu.l3_bytes, 64ULL << 20);
    EXPECT_EQ(hw.cpu.l3_instances, 2u);
}

TEST(Discovery, EvoX2GpuAndTiers) {
    const auto hw = discover_fixture("evo_x2");
    ASSERT_EQ(hw.gpus.size(), 1u);  // card1-DP-1 connector is not a card
    const auto& g = hw.gpus.front();
    EXPECT_EQ(g.drm_card, "card1");
    EXPECT_EQ(g.pci_slot, "0000:c5:00.0");
    EXPECT_EQ(g.device_id, 0x1586u);
    EXPECT_EQ(g.vram_total, 98304ULL << 20);
    EXPECT_EQ(g.vram_used, 512ULL << 20);
    EXPECT_EQ(g.gtt_total, 16670208000ULL);
    EXPECT_EQ(g.kfd_node, 1u);
    EXPECT_EQ(g.kfd_association, "pci");
    EXPECT_EQ(g.gfx_target, "gfx1151");
    EXPECT_EQ(g.compute_units, 40u);
    EXPECT_EQ(g.wavefront_size, 32u);
    EXPECT_EQ(g.max_engine_clock_mhz, 2900u);
    ASSERT_TRUE(g.telemetry.temperature_c);
    EXPECT_DOUBLE_EQ(*g.telemetry.temperature_c, 45.0);
    ASSERT_TRUE(g.telemetry.power_w);
    EXPECT_DOUBLE_EQ(*g.telemetry.power_w, 12.0);  // power1_input fallback
    EXPECT_EQ(g.telemetry.sclk_mhz, 1000u);
    EXPECT_EQ(g.telemetry.mclk_mhz, 4000u);
    EXPECT_EQ(g.telemetry.performance_level, "auto");
    EXPECT_EQ(g.telemetry.power_profile, "BOOTUP_DEFAULT");

    const auto& t = hw.tiers;
    EXPECT_EQ(t.topology, MemoryTopology::CarveoutPrimary);
    EXPECT_EQ(t.host_total, 32559000ULL * 1024);
    EXPECT_EQ(t.host_available, 28 * GiB);
    // Carveout is invisible to the OS (additive); GTT is capped by OS RAM.
    EXPECT_EQ(t.gpu_accessible_bytes, (98304ULL << 20) + 16670208000ULL);
}

TEST(Discovery, EvoX2OsDriversAndNoWarnings) {
    const auto hw = discover_fixture("evo_x2");
    EXPECT_EQ(hw.os.kernel_release, "7.0.0-31-generic");
    EXPECT_EQ(hw.os.kernel_version, (KernelVersion{7, 0, 0}));
    EXPECT_EQ(hw.os.distro_pretty_name, "Ubuntu 26.04.1 LTS");
    EXPECT_EQ(hw.os.distro_version_id, "26.04");
    EXPECT_TRUE(hw.rocm.installed);
    EXPECT_EQ(hw.rocm.version, "7.15.26333");
    ASSERT_EQ(hw.vulkan.icds.size(), 1u);
    EXPECT_EQ(hw.vulkan.icds[0].kind, VulkanDriverKind::Radv);
    EXPECT_EQ(hw.vulkan.icds[0].api_version, "1.4.318");
    EXPECT_FALSE(hw.vulkan.override_active);
    EXPECT_EQ(hw.iommu.state, IommuState::Off);
    EXPECT_FALSE(hw.cmdline.amdgpu_gttsize_mib);
    EXPECT_TRUE(hw.warnings.empty()) << hw.warnings.front();
}

TEST(Discovery, PrdGttConfig) {
    const auto hw = discover_fixture("prd_gtt_config");
    ASSERT_EQ(hw.gpus.size(), 1u);
    const auto& g = hw.gpus.front();
    EXPECT_FALSE(g.pci_slot);
    EXPECT_EQ(g.kfd_association, "single-device");
    EXPECT_EQ(g.gfx_target, "gfx1151");
    EXPECT_EQ(g.telemetry.power_profile, "COMPUTE");
    EXPECT_EQ(g.telemetry.performance_level, "high");

    EXPECT_EQ(hw.tiers.topology, MemoryTopology::GttPrimary);
    EXPECT_EQ(hw.tiers.vram_total, 1 * GiB);
    EXPECT_EQ(hw.tiers.gtt_total, 120 * GiB);
    EXPECT_EQ(hw.tiers.gpu_accessible_bytes, 121 * GiB);

    EXPECT_EQ(hw.cmdline.amdgpu_gttsize_mib, 122880);
    EXPECT_EQ(hw.cmdline.ttm_pages_limit, 31457280u);
    EXPECT_EQ(hw.cmdline.ttm_pages_limit_bytes_4k, 120 * GiB);
    EXPECT_EQ(hw.iommu.state, IommuState::On);
    EXPECT_FALSE(hw.rocm.installed);
    EXPECT_TRUE(hw.vulkan.has(VulkanDriverKind::Radv));
    EXPECT_TRUE(hw.vulkan.has(VulkanDriverKind::Amdvlk));

    auto has_warning = [&](std::string_view needle) {
        return std::ranges::any_of(hw.warnings, [&](const std::string& w) { return w.find(needle) != std::string::npos; });
    };
    EXPECT_TRUE(has_warning("AMDVLK"));
    EXPECT_TRUE(has_warning("older than 6.19"));
    EXPECT_TRUE(has_warning("IOMMU is on"));
}

TEST(Discovery, GpuAccessibleNeverCountsMoreGttThanOsRam) {
    GpuInfo g;
    g.vram_total = 1 * GiB;
    g.gtt_total = 200 * GiB;  // driver limit above physical RAM
    HostMemory host{64 * GiB, 60 * GiB};
    const auto t = make_memory_tiers({g}, host);
    EXPECT_EQ(t.gpu_accessible_bytes, 65 * GiB);
}

TEST(Discovery, NoGpuIgnoresNonAmdCardsAndReportsBrokenIcd) {
    const auto hw = discover_fixture("no_gpu");
    EXPECT_TRUE(hw.gpus.empty());
    EXPECT_EQ(hw.tiers.topology, MemoryTopology::NoGpu);
    EXPECT_EQ(hw.tiers.gpu_accessible_bytes, 0u);
    EXPECT_EQ(hw.tiers.host_available, 30 * GiB);
    EXPECT_EQ(hw.cpu.logical_threads, 16u);
    EXPECT_FALSE(hw.cpu.simd.avx512f);
    EXPECT_FALSE(hw.cpu.l3_bytes);
    EXPECT_FALSE(hw.rocm.installed);
    ASSERT_EQ(hw.vulkan.icds.size(), 2u);
    EXPECT_EQ(hw.vulkan.icds[0].manifest_path, "/usr/share/vulkan/icd.d/broken_icd.json");
    EXPECT_EQ(hw.vulkan.icds[0].error, "malformed JSON");
    EXPECT_EQ(hw.vulkan.icds[1].kind, VulkanDriverKind::Lavapipe);
    EXPECT_EQ(hw.iommu.state, IommuState::Unknown);
    ASSERT_FALSE(hw.warnings.empty());
    EXPECT_NE(hw.warnings.front().find("no AMD GPU"), std::string::npos);
}

TEST(Discovery, VulkanEnvOverrideReplacesSearchDirs) {
    const auto hw = discover_fixture("prd_gtt_config", {{"VK_ICD_FILENAMES", "/etc/vulkan/icd.d/amd_icd64.json"}});
    EXPECT_TRUE(hw.vulkan.override_active);
    ASSERT_EQ(hw.vulkan.icds.size(), 1u);
    EXPECT_EQ(hw.vulkan.icds[0].kind, VulkanDriverKind::Amdvlk);
    EXPECT_EQ(hw.vulkan.icds[0].source, "VK_ICD_FILENAMES");

    // VK_DRIVER_FILES takes precedence over the legacy variable and accepts directories.
    const auto hw2 = discover_fixture("prd_gtt_config", {{"VK_ICD_FILENAMES", "/etc/vulkan/icd.d/amd_icd64.json"},
                                                         {"VK_DRIVER_FILES", "/usr/share/vulkan/icd.d"}});
    ASSERT_EQ(hw2.vulkan.icds.size(), 1u);
    EXPECT_EQ(hw2.vulkan.icds[0].kind, VulkanDriverKind::Radv);
    EXPECT_EQ(hw2.vulkan.icds[0].source, "VK_DRIVER_FILES");

    // Missing override target is reported, not silently dropped.
    const auto hw3 = discover_fixture("evo_x2", {{"VK_DRIVER_FILES", "/nope/x.json"}});
    ASSERT_EQ(hw3.vulkan.icds.size(), 1u);
    EXPECT_TRUE(hw3.vulkan.icds[0].error);
}

TEST(Discovery, MissingRootIsDeviceError) {
    DiscoveryOptions o;
    o.root = fixture("does_not_exist");
    try {
        (void)discover(o);
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Device);
    }
}

TEST(Discovery, JsonHasRawAndNormalizedTierFields) {
    const auto hw = discover_fixture("evo_x2");
    const nlohmann::json j = hw;
    EXPECT_EQ(j["memory_tiers"]["topology"], "carveout_primary");
    EXPECT_EQ(j["memory_tiers"]["vram_total"], 98304ULL << 20);
    EXPECT_EQ(j["memory_tiers"]["gpu_accessible_bytes"], hw.tiers.gpu_accessible_bytes);
    EXPECT_EQ(j["gpus"][0]["gfx_target"], "gfx1151");
    EXPECT_EQ(j["gpus"][0]["vendor_id"], "0x1002");
    EXPECT_EQ(j["vulkan"]["icds"][0]["driver"], "radv");
    EXPECT_EQ(j["iommu"]["state"], "off");
    // Round-trips through text.
    EXPECT_EQ(nlohmann::json::parse(j.dump()), j);
}

// Dev host (WSL2, no AMD GPU) — the real "/" root. Only checks that discovery of a real
// system does not throw and serializes; it asserts nothing about what the host has.
TEST(Discovery, DevHostRealRootSmoke) {
    DiscoveryOptions o;
    o.use_process_env = true;
    const auto hw = discover(o);
    const nlohmann::json j = hw;
    EXPECT_NO_THROW((void)nlohmann::json::parse(j.dump()));
    EXPECT_GT(hw.cpu.logical_threads, 0u);
    std::cout << "[dev-host] cpu=" << hw.cpu.model_name << " threads=" << hw.cpu.logical_threads
              << " amd_gpus=" << hw.gpus.size() << " topology=" << to_string(hw.tiers.topology)
              << " kernel=" << hw.os.kernel_release.value_or("?") << " icds=" << hw.vulkan.icds.size()
              << " runtime_avx2=" << (hw.cpu.runtime_simd ? hw.cpu.runtime_simd->avx2 : false) << "\n";
}
