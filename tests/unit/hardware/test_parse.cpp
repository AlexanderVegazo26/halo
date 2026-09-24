#include <gtest/gtest.h>

#include <limits>

#include "halo/hardware/parse.h"

using namespace halo::hardware;

TEST(Parse, SplitLinesStripsCarriageReturns) {
    const auto lines = split_lines("a\r\nb\r\n\r\nc");
    ASSERT_EQ(lines.size(), 4u);
    EXPECT_EQ(lines[0], "a");
    EXPECT_EQ(lines[1], "b");
    EXPECT_EQ(lines[2], "");
    EXPECT_EQ(lines[3], "c");
    EXPECT_TRUE(split_lines("").empty());
}

TEST(Parse, NumbersRejectGarbage) {
    EXPECT_EQ(parse_u64(" 42\r\n"), 42u);
    EXPECT_FALSE(parse_u64("42x"));
    EXPECT_FALSE(parse_u64("-1"));
    EXPECT_FALSE(parse_u64("99999999999999999999999"));  // > 2^64
    EXPECT_EQ(parse_i64("-1"), -1);
    EXPECT_EQ(parse_hex_u32("0x1002\n"), 0x1002u);
    EXPECT_EQ(parse_hex_u32("1586"), 0x1586u);
    EXPECT_FALSE(parse_hex_u32("0x"));
    EXPECT_FALSE(parse_hex_u32("0xZZ"));
}

TEST(Parse, MeminfoKilobytesToBytesWithCrlf) {
    const auto m = parse_meminfo("MemTotal:       32559000 kB\r\nMemFree: 1 kB\r\nMemAvailable:   29360128 kB\r\n");
    EXPECT_EQ(m.total_bytes, 32559000ULL * 1024);
    EXPECT_EQ(m.available_bytes, 29360128ULL * 1024);
}

TEST(Parse, MeminfoOverflowAndUnknownUnitAreRejected) {
    // 2^64 / 1024 rounded up: multiplying by 1024 overflows.
    const auto m = parse_meminfo("MemTotal: 18014398509481984 kB\nMemAvailable: 5 MB\n");
    EXPECT_FALSE(m.total_bytes);
    EXPECT_FALSE(m.available_bytes);
}

TEST(Parse, CpuinfoCountsThreadsCoresAndFlags) {
    const std::string block0 =
        "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: X\nphysical id\t: 0\ncpu cores\t: 2\n"
        "flags\t\t: fpu avx2 fma avx512f avx512bw avx512vl avx512_vnni avx512_bf16 avx_vnni\n\n";
    const std::string block1 = "processor\t: 1\nphysical id\t: 0\ncpu cores\t: 2\n\n";
    const std::string block2 = "processor\t: 2\nphysical id\t: 1\ncpu cores\t: 4\n\n";
    const auto c = parse_cpuinfo(block0 + block1 + block2);
    EXPECT_EQ(c.logical_threads, 3u);
    EXPECT_EQ(c.sockets, 2u);
    EXPECT_EQ(c.physical_cores, 6u);  // 2 (package 0, counted once) + 4 (package 1)
    EXPECT_EQ(c.model_name, "X");
    EXPECT_TRUE(c.simd.avx2 && c.simd.fma && c.simd.avx512f && c.simd.avx512bw && c.simd.avx512vl);
    EXPECT_TRUE(c.simd.avx512_vnni && c.simd.avx512_bf16 && c.simd.avx_vnni);
    EXPECT_FALSE(c.simd.avx512dq);
}

TEST(Parse, SimdFlagsMatchWholeTokensOnly) {
    // "avx512f" must not be inferred from "avx512fp16" or "avx2" from "avx".
    const auto f = simd_from_flags("avx avx512fp16 avx512_vnni_x");
    EXPECT_FALSE(f.avx2);
    EXPECT_FALSE(f.avx512f);
    EXPECT_FALSE(f.avx512_vnni);
}

TEST(Parse, CacheSize) {
    EXPECT_EQ(parse_cache_size("32768K\n"), 32768ULL * 1024);
    EXPECT_EQ(parse_cache_size("64M"), 64ULL << 20);
    EXPECT_EQ(parse_cache_size("512"), 512u);
    EXPECT_FALSE(parse_cache_size("K"));
    EXPECT_FALSE(parse_cache_size("99999999999999999999G"));
}

TEST(Parse, KernelVersionComparesNumerically) {
    const auto v68 = parse_kernel_version("6.8.0-45-generic");
    const auto v619 = parse_kernel_version("6.19-rc1");
    const auto v7 = parse_kernel_version("7.0.0-31-generic\n");
    ASSERT_TRUE(v68 && v619 && v7);
    EXPECT_EQ(*v68, (KernelVersion{6, 8, 0}));
    EXPECT_EQ(*v619, (KernelVersion{6, 19, 0}));
    EXPECT_EQ(*v7, (KernelVersion{7, 0, 0}));
    // A string compare would put "6.8" after "6.19".
    EXPECT_LT(*v68, *v619);
    EXPECT_LT(*v619, *v7);
    EXPECT_FALSE(parse_kernel_version("linux"));
    EXPECT_FALSE(parse_kernel_version("6"));
    EXPECT_FALSE(parse_kernel_version("99999999999.1"));
}

TEST(Parse, OsReleaseStripsQuotes) {
    const auto kv = parse_os_release("# c\nPRETTY_NAME=\"Ubuntu 26.04.1 LTS\"\r\nID=ubuntu\nVERSION_ID='26.04'\nBAD\n");
    EXPECT_EQ(kv.at("PRETTY_NAME"), "Ubuntu 26.04.1 LTS");
    EXPECT_EQ(kv.at("ID"), "ubuntu");
    EXPECT_EQ(kv.at("VERSION_ID"), "26.04");
    EXPECT_FALSE(kv.contains("BAD"));
}

TEST(Parse, Cmdline) {
    const auto c = parse_cmdline("ro quiet amdgpu.gttsize=122880 a=b=c\n");
    ASSERT_EQ(c.size(), 4u);
    EXPECT_EQ(c[0], (std::pair<std::string, std::string>{"ro", ""}));
    EXPECT_EQ(c[2], (std::pair<std::string, std::string>{"amdgpu.gttsize", "122880"}));
    EXPECT_EQ(c[3], (std::pair<std::string, std::string>{"a", "b=c"}));
}

TEST(Parse, GfxTargetVersionDecodesHexDigits) {
    EXPECT_EQ(decode_gfx_target_version(110501), "gfx1151");
    EXPECT_EQ(decode_gfx_target_version(90010), "gfx90a");  // stepping 10 -> 'a'
    EXPECT_EQ(decode_gfx_target_version(100300), "gfx1030");
    EXPECT_EQ(decode_gfx_target_version(90402), "gfx942");
    EXPECT_EQ(decode_gfx_target_version(110000), "gfx1100");
    EXPECT_FALSE(decode_gfx_target_version(0));       // CPU node
    EXPECT_FALSE(decode_gfx_target_version(111601));  // minor 16 is not a hex digit
    EXPECT_FALSE(decode_gfx_target_version(110516));  // stepping 16
    EXPECT_FALSE(decode_gfx_target_version(501));     // major 0
}

TEST(Parse, KfdPropertiesSkipsMalformedLines) {
    const auto p = parse_kfd_properties("simd_count 80\r\ngfx_target_version 110501\nbogus\nname x y\nneg -1\n");
    EXPECT_EQ(p.at("simd_count"), 80u);
    EXPECT_EQ(p.at("gfx_target_version"), 110501u);
    EXPECT_EQ(p.size(), 2u);
}

TEST(Parse, DpmLevels) {
    const auto l = parse_dpm_levels("S: 19Mhz\n0: 600Mhz \n1: 2900MHz *\r\nx: fast\n");
    ASSERT_EQ(l.size(), 3u);
    EXPECT_EQ(l[0].index, "S");
    EXPECT_EQ(l[1].mhz, 600u);
    EXPECT_FALSE(l[1].active);
    EXPECT_EQ(l[2].mhz, 2900u);
    EXPECT_TRUE(l[2].active);
}

TEST(Parse, ActivePowerProfileAcrossLayouts) {
    EXPECT_EQ(parse_active_power_profile(" 0 BOOTUP_DEFAULT*\n 1 3D_FULL_SCREEN\n"), "BOOTUP_DEFAULT");
    EXPECT_EQ(parse_active_power_profile("PROFILE_INDEX(NAME) CLOCK\n 0 BOOTUP_DEFAULT :\n 5 COMPUTE*:\n"), "COMPUTE");
    EXPECT_EQ(parse_active_power_profile(" 1 3D_FULL_SCREEN *:\n"), "3D_FULL_SCREEN");
    EXPECT_EQ(parse_active_power_profile(" 4 VR*(4)\n"), "VR");
    EXPECT_FALSE(parse_active_power_profile(" 0 BOOTUP_DEFAULT\n 1 VIDEO\n"));
}

TEST(Parse, PciSlotToKfdLocationId) {
    const auto a = parse_pci_slot_name("0000:c5:00.0\n");
    ASSERT_TRUE(a);
    EXPECT_EQ(a->domain, 0u);
    EXPECT_EQ(a->location_id, 0xc500u);  // 50432, as KFD reports it
    const auto b = parse_pci_slot_name("0001:03:1f.7");
    ASSERT_TRUE(b);
    EXPECT_EQ(b->domain, 1u);
    EXPECT_EQ(b->location_id, (0x03u << 8) | (0x1fu << 3) | 7u);
    EXPECT_FALSE(parse_pci_slot_name("0000:c5:20.0"));  // device > 0x1f
    EXPECT_FALSE(parse_pci_slot_name("garbage"));
}

TEST(Parse, ClassifyVulkanIcd) {
    EXPECT_EQ(classify_vulkan_icd("radeon_icd.x86_64.json", "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so"),
              VulkanDriverKind::Radv);
    EXPECT_EQ(classify_vulkan_icd("amd_icd64.json", "/usr/lib/x86_64-linux-gnu/amdvlk64.so"), VulkanDriverKind::Amdvlk);
    EXPECT_EQ(classify_vulkan_icd("lvp_icd.x86_64.json", "libvulkan_lvp.so"), VulkanDriverKind::Lavapipe);
    EXPECT_EQ(classify_vulkan_icd("intel_icd.json", "libvulkan_intel.so"), VulkanDriverKind::Other);
    // Library path wins over a misleading file name.
    EXPECT_EQ(classify_vulkan_icd("radeon_icd.json", "/opt/amdvlk/amdvlk64.so"), VulkanDriverKind::Amdvlk);
    // File-name fallback when the manifest had no library_path.
    EXPECT_EQ(classify_vulkan_icd("amd_icd64.json", ""), VulkanDriverKind::Amdvlk);
    EXPECT_EQ(classify_vulkan_icd("radeon_icd.i686.json", ""), VulkanDriverKind::Radv);
}
