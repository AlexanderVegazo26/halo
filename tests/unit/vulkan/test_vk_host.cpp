// Host-only tests of the Vulkan backend: no Vulkan device required (they also run under
// ASan/UBSan without a driver in the loop).

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/device.h"
#include "halo/backends/vulkan/kernel.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/backends/vulkan/shaders.h"
#include "halo/core/error.h"

namespace hv = halo::vulkan;

namespace {

std::string sha(std::string_view s) { return hv::sha256_hex(std::as_bytes(std::span(s.data(), s.size()))); }

}  // namespace

// ---------------------------------------------------------------- hashing / shaders

TEST(VkSha256, Fips180KnownAnswers) {
    EXPECT_EQ(sha(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // 56 bytes: padding spills into a second block.
    EXPECT_EQ(sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 112 bytes (two full blocks + padding block).
    EXPECT_EQ(sha("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrs"
                  "mnopqrstnopqrstu"),
              "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    EXPECT_EQ(sha(std::string(1000, 'a')), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST(VkShaders, AllExpectedShadersEmbeddedWithValidHashes) {
    const std::set<std::string> expected{"rms_norm",        "matvec_f32",           "matvec_q8_0",
                                         "matvec_q4_k",     "matvec_q5_k",          "matvec_q6_k",
                                         "matvec_iq4_xs",   "gated_delta_rule_decode",
                                         "argmax_partial", "argmax_final",   "conv1d_silu",
                                         "gated_rms_norm", "add_rms_norm",   "eltwise",
                                         "gdn_gates",      "rope_neox",      "get_rows_f32",
                                         "get_rows_q8_0",  "get_rows_q4_k",  "get_rows_q5_k",
                                         "get_rows_q6_k",  "get_rows_iq4_xs", "kv_write",
                                         "attention"};
    std::set<std::string> seen;
    for (const hv::EmbeddedShader* s : hv::embedded_shaders()) {
        ASSERT_NE(s, nullptr);
        EXPECT_TRUE(seen.insert(std::string(s->name)).second) << "duplicate shader " << s->name;
        ASSERT_FALSE(s->spirv.empty()) << s->name;
        EXPECT_EQ(s->spirv[0], 0x07230203u) << s->name << ": not SPIR-V (magic)";
        EXPECT_EQ(s->sha256.size(), 64u) << s->name;
        EXPECT_TRUE(hv::verify_shader(*s)) << s->name << ": embedded hash does not match the bytes";
        EXPECT_FALSE(s->source_path.starts_with("/")) << "absolute path embedded: " << s->source_path;
        std::cout << "[vk-shader] " << s->name << " (" << s->source_path << ") " << s->spirv.size_bytes()
                  << " B sha256=" << s->sha256 << "\n";
    }
    EXPECT_EQ(seen, expected);
    EXPECT_EQ(hv::shader_set_hash().size(), 64u);
    std::cout << "[vk-shader] set sha256=" << hv::shader_set_hash() << "\n";
}

TEST(VkShaders, VerifyDetectsTampering) {
    const hv::EmbeddedShader& s = hv::find_shader("rms_norm");
    std::vector<std::uint32_t> copy(s.spirv.begin(), s.spirv.end());
    copy[copy.size() / 2] ^= 1u;
    const hv::EmbeddedShader tampered{s.name, s.source_path, copy, s.sha256};
    EXPECT_FALSE(hv::verify_shader(tampered));
}

TEST(VkShaders, UnknownNameThrowsKernelError) {
    try {
        (void)hv::find_shader("no_such_shader");
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Kernel);
    }
}

// ---------------------------------------------------------------- ICD policy

namespace {

hv::DeviceInfo fake(std::uint32_t index, const char* name, VkDriverId driver, VkPhysicalDeviceType type,
                    std::uint32_t api = VK_API_VERSION_1_3) {
    hv::DeviceInfo d;
    d.index = index;
    d.name = name;
    d.driver_id = driver;
    d.type = type;
    d.api_version = api;
    d.compute_queue_family = 0;
    d.heaps = {{1ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}};
    d.memory_types = {{0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, hv::MemoryTier::Vram}};
    hv::finalize_device_info(d);
    return d;
}

bool any_contains(const std::vector<std::string>& v, std::string_view needle) {
    for (const auto& s : v) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

TEST(VkPolicy, ClassifiesDriverIds) {
    EXPECT_EQ(hv::classify_driver(VK_DRIVER_ID_MESA_RADV), hv::DriverKind::Radv);
    EXPECT_EQ(hv::classify_driver(VK_DRIVER_ID_AMD_OPEN_SOURCE), hv::DriverKind::AmdvlkOpen);
    EXPECT_EQ(hv::classify_driver(VK_DRIVER_ID_AMD_PROPRIETARY), hv::DriverKind::AmdProprietary);
    EXPECT_EQ(hv::classify_driver(VK_DRIVER_ID_MESA_LLVMPIPE), hv::DriverKind::Lavapipe);
    EXPECT_EQ(hv::classify_driver(VK_DRIVER_ID_NVIDIA_PROPRIETARY), hv::DriverKind::Other);
    EXPECT_TRUE(hv::is_amdvlk(hv::DriverKind::AmdvlkOpen));
    EXPECT_TRUE(hv::is_amdvlk(hv::DriverKind::AmdProprietary));
    EXPECT_FALSE(hv::is_amdvlk(hv::DriverKind::Radv));
}

TEST(VkPolicy, PrefersRadvOverAmdvlkAndLavapipeAndWarnsAboutAmdvlk) {
    // The "AMDVLK hijack" topology: both ICDs expose the same GPU, plus lavapipe.
    const std::vector<hv::DeviceInfo> devs{
        fake(0, "llvmpipe", VK_DRIVER_ID_MESA_LLVMPIPE, VK_PHYSICAL_DEVICE_TYPE_CPU),
        fake(1, "AMD Radeon 8060S (AMDVLK)", VK_DRIVER_ID_AMD_OPEN_SOURCE, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
        fake(2, "AMD Radeon 8060S (RADV GFX1151)", VK_DRIVER_ID_MESA_RADV, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
    };
    const hv::DeviceSelection sel = hv::select_device(devs, std::nullopt);
    EXPECT_EQ(sel.index, 2u);
    EXPECT_NE(sel.reason.find("RADV"), std::string::npos);
    EXPECT_TRUE(any_contains(sel.warnings, "AMDVLK ICD is also installed"));
    EXPECT_FALSE(any_contains(sel.warnings, "correctness-only"));
}

TEST(VkPolicy, AmdvlkAloneIsSelectedWithProminentWarning) {
    const std::vector<hv::DeviceInfo> devs{
        fake(0, "llvmpipe", VK_DRIVER_ID_MESA_LLVMPIPE, VK_PHYSICAL_DEVICE_TYPE_CPU),
        fake(1, "AMD Radeon 8060S", VK_DRIVER_ID_AMD_PROPRIETARY, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
    };
    const hv::DeviceSelection sel = hv::select_device(devs, std::nullopt);
    EXPECT_EQ(sel.index, 1u);
    EXPECT_TRUE(any_contains(sel.warnings, "AMDVLK selected"));
}

TEST(VkPolicy, LavapipeOnlyIsSelectedAndFlaggedCorrectnessOnly) {
    const std::vector<hv::DeviceInfo> devs{
        fake(0, "llvmpipe (LLVM 20.1.2, 256 bits)", VK_DRIVER_ID_MESA_LLVMPIPE, VK_PHYSICAL_DEVICE_TYPE_CPU)};
    EXPECT_TRUE(devs[0].correctness_only);
    EXPECT_FALSE(devs[0].tier_labels_meaningful);
    const hv::DeviceSelection sel = hv::select_device(devs, std::nullopt);
    EXPECT_EQ(sel.index, 0u);
    EXPECT_TRUE(any_contains(sel.warnings, "correctness-only"));
}

TEST(VkPolicy, NonRadvGpuBeatsLavapipeAndDiscreteBeatsIntegrated) {
    const std::vector<hv::DeviceInfo> devs{
        fake(0, "llvmpipe", VK_DRIVER_ID_MESA_LLVMPIPE, VK_PHYSICAL_DEVICE_TYPE_CPU),
        fake(1, "Intel iGPU", VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
        fake(2, "NVIDIA GeForce RTX 3060", VK_DRIVER_ID_NVIDIA_PROPRIETARY, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
    };
    const hv::DeviceSelection sel = hv::select_device(devs, std::nullopt);
    EXPECT_EQ(sel.index, 2u);
    EXPECT_TRUE(sel.warnings.empty());
}

TEST(VkPolicy, UnsuitableDevicesAreSkippedAndNoneSuitableThrows) {
    std::vector<hv::DeviceInfo> devs{
        fake(0, "old RADV", VK_DRIVER_ID_MESA_RADV, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_API_VERSION_1_2),
        fake(1, "llvmpipe", VK_DRIVER_ID_MESA_LLVMPIPE, VK_PHYSICAL_DEVICE_TYPE_CPU),
    };
    EXPECT_EQ(hv::select_device(devs, std::nullopt).index, 1u);
    devs[1].compute_queue_family.reset();
    try {
        (void)hv::select_device(devs, std::nullopt);
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Device);
    }
    EXPECT_THROW((void)hv::select_device({}, std::nullopt), halo::Error);
}

TEST(VkPolicy, OverrideByIndexOrNameAndNeverSilentlyReplaced) {
    const std::vector<hv::DeviceInfo> devs{
        fake(0, "llvmpipe", VK_DRIVER_ID_MESA_LLVMPIPE, VK_PHYSICAL_DEVICE_TYPE_CPU),
        fake(1, "AMD Radeon 8060S (RADV)", VK_DRIVER_ID_MESA_RADV, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
        fake(2, "old", VK_DRIVER_ID_MESA_RADV, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_API_VERSION_1_1),
    };
    EXPECT_EQ(hv::select_device(devs, "0").index, 0u);
    EXPECT_EQ(hv::select_device(devs, "LLVMPIPE").index, 0u);  // case-insensitive substring
    EXPECT_EQ(hv::select_device(devs, "8060s").index, 1u);
    EXPECT_THROW((void)hv::select_device(devs, "7"), halo::Error);        // no such index
    EXPECT_THROW((void)hv::select_device(devs, "nvidia"), halo::Error);   // no such name
    EXPECT_THROW((void)hv::select_device(devs, "2"), halo::Error);        // unsuitable
    EXPECT_EQ(hv::select_device(devs, "").index, 1u);                      // empty = policy
}

// ---------------------------------------------------------------- memory types

namespace {

// RADV on a Strix-Halo-like APU: heap 0 = carveout (device local), heap 1 = GTT.
std::vector<hv::MemoryTypeInfo> radv_apu_types() {
    constexpr auto dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr auto hv_ = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    constexpr auto hc = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr auto ca = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    return {{0, dl, hv::MemoryTier::Vram},
            {1, hv_ | hc, hv::MemoryTier::Gtt},
            {0, dl | hv_ | hc, hv::MemoryTier::Vram},
            {1, hv_ | hc | ca, hv::MemoryTier::Gtt},
            {0, dl | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD, hv::MemoryTier::Vram}};
}

}  // namespace

TEST(VkMemory, ChoosesTypesPerUsageOnApuLayout) {
    const auto t = radv_apu_types();
    EXPECT_EQ(hv::choose_memory_type(t, 0xFFu, hv::MemoryUsage::DeviceLocal), 0u);
    EXPECT_EQ(hv::choose_memory_type(t, 0xFFu, hv::MemoryUsage::HostVisible), 1u);
    EXPECT_EQ(hv::choose_memory_type(t, 0xFFu, hv::MemoryUsage::HostCached), 3u);
    // Restricted by memoryTypeBits.
    EXPECT_EQ(hv::choose_memory_type(t, 0b00100u, hv::MemoryUsage::DeviceLocal), 2u);
    EXPECT_EQ(hv::choose_memory_type(t, 0b00100u, hv::MemoryUsage::HostVisible), 2u);
    EXPECT_EQ(hv::choose_memory_type(t, 0b00100u, hv::MemoryUsage::HostCached), 2u);  // falls back
    EXPECT_EQ(hv::choose_memory_type(t, 0b00001u, hv::MemoryUsage::HostVisible), std::nullopt);
    // The AMD uncached debug type is never chosen.
    EXPECT_EQ(hv::choose_memory_type(t, 0b10000u, hv::MemoryUsage::DeviceLocal), std::nullopt);
    EXPECT_EQ(hv::choose_memory_type(t, 0u, hv::MemoryUsage::DeviceLocal), std::nullopt);
}

TEST(VkMemory, TierLabelsFollowHeapsAndAreFlaggedMeaninglessOnCpu) {
    hv::DeviceInfo d = fake(0, "RADV", VK_DRIVER_ID_MESA_RADV, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    d.heaps = {{96ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}, {32ull << 30, 0}};
    d.memory_types = radv_apu_types();
    for (auto& mt : d.memory_types) mt.tier = hv::MemoryTier::Host;  // finalize must overwrite
    hv::finalize_device_info(d);
    EXPECT_TRUE(d.tier_labels_meaningful);
    EXPECT_EQ(d.memory_types[0].tier, hv::MemoryTier::Vram);
    EXPECT_EQ(d.memory_types[1].tier, hv::MemoryTier::Gtt);
    EXPECT_EQ(d.memory_types[3].tier, hv::MemoryTier::Gtt);
    const nlohmann::json j = hv::to_json(d);
    EXPECT_EQ(j["memory_types"][1]["tier_label"], "GTT");
    EXPECT_EQ(j["memory_heaps"][0]["device_local"], true);
    EXPECT_EQ(j["driver"], "radv");
    EXPECT_EQ(j["correctness_only"], false);
}

// ---------------------------------------------------------------- shapes

TEST(VkShapes, Grid1dSplitsBeyondTheXLimit) {
    EXPECT_EQ(hv::grid_1d(1, 65535, 65535), (hv::GroupCount{1, 1, 1}));
    EXPECT_EQ(hv::grid_1d(65535, 65535, 65535), (hv::GroupCount{65535, 1, 1}));
    EXPECT_EQ(hv::grid_1d(248320, 65535, 65535), (hv::GroupCount{65535, 4, 1}));
    EXPECT_THROW((void)hv::grid_1d(0, 65535, 65535), halo::Error);
    EXPECT_THROW((void)hv::grid_1d(65535ull * 65535 + 1, 65535, 65535), halo::Error);
}

TEST(VkShapes, MatvecRowBytesFollowGgmlBlockSizes) {
    EXPECT_EQ(hv::matvec_row_bytes(halo::DType::F32, 5120), 5120u * 4);
    EXPECT_EQ(hv::matvec_row_bytes(halo::DType::Q8_0, 5120), 5120u / 32 * 34);
    EXPECT_EQ(hv::matvec_row_bytes(halo::DType::Q4_K, 5120), 5120u / 256 * 144);
    EXPECT_EQ(hv::matvec_row_bytes(halo::DType::Q5_K, 5120), 5120u / 256 * 176);
    EXPECT_EQ(hv::matvec_row_bytes(halo::DType::Q6_K, 5120), 5120u / 256 * 210);
    EXPECT_EQ(hv::matvec_row_bytes(halo::DType::IQ4_XS, 5120), 5120u / 256 * 136);
    EXPECT_THROW((void)hv::matvec_row_bytes(halo::DType::Q8_0, 33), halo::Error);
    EXPECT_THROW((void)hv::matvec_row_bytes(halo::DType::Q4_K, 128), halo::Error);
    try {
        (void)hv::matvec_row_bytes(halo::DType::Q3_K, 256);
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
    }
}
