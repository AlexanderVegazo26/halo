#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "autotune_test_util.h"
#include "halo/autotune/profile_key.h"

using namespace halo::autotune;
using halo::ErrorCode;

namespace {

halo::profiling::HardwareState evo_x2_state(const std::string& platform) {
    halo::hardware::DiscoveryOptions o;
    o.root = std::filesystem::path(HALO_SOURCE_DIR) / "tests" / "fixtures" / "sysfs" / "evo_x2";
    o.use_process_env = false;
    o.probe_runtime_cpuid = false;
    return halo::profiling::capture_hardware_state(o, platform, "2026-09-24T00:00:00Z");
}

}  // namespace

TEST(ProfileKey, BuiltFromEvoX2HardwareState) {
    const ProfileKey k = make_profile_key(evo_x2_state("performance"), "sha-model", "pack-x", "gfx1151");
    EXPECT_EQ(k.halo_version, std::string(halo_version()));
    EXPECT_FALSE(k.halo_version.empty());
    EXPECT_EQ(k.gpu_device, "1002:1586");
    EXPECT_EQ(k.gpu_arch, "gfx1151");
    EXPECT_EQ(k.rocm_version, "7.15.26333");
    EXPECT_EQ(k.vulkan_version, "1.4.318");
    EXPECT_EQ(k.kernel_version, "7.0.0-31-generic");
    EXPECT_EQ(k.os, "Ubuntu 26.04.1 LTS");
    EXPECT_EQ(k.power_mode, "performance|auto/BOOTUP_DEFAULT");
    EXPECT_EQ(k.driver_version, "");  // not discovered today: empty, never invented
    const nlohmann::json j = k;
    EXPECT_EQ(j.size(), 12u);
    EXPECT_EQ(j.at("POWER_MODE"), "performance|auto/BOOTUP_DEFAULT");
}

TEST(ProfileKey, PowerModeRule) {
    // GPU key without the platform (BIOS/EC) label: refused.
    test::expect_error(ErrorCode::Config, [] { (void)make_profile_key(evo_x2_state(""), "m", "p", "gfx1151"); });
    // CPU-only key (no GPU in sysfs): GPU parts may be unknown, the platform part may not.
    halo::profiling::HardwareState cpu;
    cpu.power_mode = halo::profiling::make_power_mode("dev-host", std::nullopt, std::nullopt);
    const ProfileKey k = make_profile_key(cpu, "m", "p", "x86-64");
    EXPECT_EQ(k.gpu_device, "cpu");
    EXPECT_EQ(k.gpu_arch, "");
    cpu.power_mode = halo::profiling::make_power_mode("", std::nullopt, std::nullopt);
    test::expect_error(ErrorCode::Config, [&] { (void)make_profile_key(cpu, "m", "p", "x86-64"); });
    // GPU key with a GPU part unknown: refused.
    auto gpu = evo_x2_state("performance");
    gpu.power_mode = halo::profiling::make_power_mode("performance", std::nullopt, std::string("COMPUTE"));
    test::expect_error(ErrorCode::Config, [&] { (void)make_profile_key(gpu, "m", "p", "gfx1151"); });
    test::expect_error(ErrorCode::Config, [&] { (void)make_profile_key(evo_x2_state("performance"), "", "p", "g"); });
    test::expect_error(ErrorCode::Config, [&] { (void)make_profile_key(evo_x2_state("performance"), "m", "", "g"); });
    test::expect_error(ErrorCode::Config, [&] { (void)make_profile_key(evo_x2_state("performance"), "m", "p", ""); });
}

TEST(ProfileKey, ExactMatch) {
    const ProfileKey k = test::sample_key();
    const KeyMatch m = match_keys(k, k);
    EXPECT_EQ(m.kind, MatchKind::Exact);
    EXPECT_TRUE(m.differing.empty() && m.mismatched.empty());
}

// One case per TRD §57 field: must-match fields reject, version fields yield a flagged
// compatible match naming exactly that field.
class ProfileKeyField : public ::testing::TestWithParam<std::pair<const char*, bool>> {};

TEST_P(ProfileKeyField, ChangingOneFieldClassifiesCorrectly) {
    const auto [name, must_match] = GetParam();
    const ProfileKey stored = test::sample_key();
    ProfileKey current = stored;
    const KeyField* field = nullptr;
    for (const auto& f : key_fields()) {
        if (f.name == name) field = &f;
    }
    ASSERT_NE(field, nullptr) << name;
    current.*(field->member) += "-changed";
    const KeyMatch m = match_keys(stored, current);
    if (must_match) {
        EXPECT_EQ(m.kind, MatchKind::Mismatch) << name;
        EXPECT_EQ(m.mismatched, std::vector<std::string>{name});
        EXPECT_TRUE(m.differing.empty());
    } else {
        EXPECT_EQ(m.kind, MatchKind::Compatible) << name;
        EXPECT_EQ(m.differing, std::vector<std::string>{name});
        EXPECT_TRUE(m.mismatched.empty());
    }
}

INSTANTIATE_TEST_SUITE_P(
    Trd57, ProfileKeyField,
    ::testing::Values(std::pair{"HALO_VERSION", false}, std::pair{"MODEL_HASH", true}, std::pair{"PACK_ID", true},
                      std::pair{"GPU_DEVICE", true}, std::pair{"GPU_ARCH", true}, std::pair{"DRIVER_VERSION", false},
                      std::pair{"ROCM_VERSION", false}, std::pair{"VULKAN_VERSION", false},
                      std::pair{"KERNEL_VERSION", false}, std::pair{"OS", false}, std::pair{"POWER_MODE", true},
                      std::pair{"ISA_TARGET", true}),
    [](const auto& info) { return std::string(info.param.first); });

TEST(ProfileKey, AllTwelveFieldsInTrdOrder) {
    const std::vector<std::string> expect{"HALO_VERSION",   "MODEL_HASH",     "PACK_ID",        "GPU_DEVICE",
                                          "GPU_ARCH",       "DRIVER_VERSION", "ROCM_VERSION",   "VULKAN_VERSION",
                                          "KERNEL_VERSION", "OS",             "POWER_MODE",     "ISA_TARGET"};
    std::vector<std::string> got;
    for (const auto& f : key_fields()) got.emplace_back(f.name);
    EXPECT_EQ(got, expect);
}

TEST(Candidate, CanonicalTextRoundTrip) {
    Candidate c{{{"threads", 8}, {"chunk", 64}}};
    EXPECT_EQ(c.to_string(), "chunk=64;threads=8");
    EXPECT_EQ(Candidate::parse("chunk=64;threads=8"), c);
    EXPECT_EQ(Candidate::parse("threads=8;chunk=64"), c);  // order-insensitive, canonical out
    EXPECT_EQ(Candidate::parse("x=-3").get("x"), -3);
    EXPECT_TRUE(Candidate::parse("").params.empty());
    for (const char* bad : {"a", "a=", "=1", "a=1;", ";a=1", "a=1;a=2", "A=1", "a=1x", "a=+1", "a=1;;b=2",
                            "a=99999999999999999999", "a b=1"}) {
        test::expect_error(ErrorCode::Config, [&] { (void)Candidate::parse(bad); });
    }
    test::expect_error(ErrorCode::Config, [&] { (void)c.get("missing"); });
}
