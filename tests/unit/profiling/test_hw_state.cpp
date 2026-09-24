#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "halo/profiling/hw_state.h"
#include "test_util.h"

using namespace halo::profiling;

namespace {

halo::hardware::DiscoveryOptions evo_x2() {
    halo::hardware::DiscoveryOptions o;
    o.root = test::sysfs_fixture("evo_x2");
    o.use_process_env = false;
    o.probe_runtime_cpuid = false;
    return o;
}

HardwareState pinned(double temp) {
    HardwareState s;
    s.power_mode = make_power_mode("performance", std::string("manual"), std::string("COMPUTE"));
    s.temperature_c = temp;
    return s;
}

}  // namespace

TEST(HwState, CapturedFromEvoX2Fixture) {
    const HardwareState s = capture_hardware_state(evo_x2(), "", "2026-09-24T00:00:00Z");
    EXPECT_EQ(s.captured_at, "2026-09-24T00:00:00Z");
    EXPECT_EQ(s.gpu_arch, "gfx1151");
    EXPECT_EQ(s.gpu_device, "1002:1586");
    EXPECT_EQ(s.gpu_performance_level, "auto");
    EXPECT_EQ(s.gpu_power_profile, "BOOTUP_DEFAULT");
    ASSERT_TRUE(s.temperature_c);
    EXPECT_DOUBLE_EQ(*s.temperature_c, 45.0);
    EXPECT_DOUBLE_EQ(*s.power_w, 12.0);
    EXPECT_EQ(s.clocks_mhz.at("sclk"), 1000u);
    EXPECT_EQ(s.clocks_mhz.at("mclk"), 4000u);
    EXPECT_EQ(s.kernel, "7.0.0-31-generic");
    EXPECT_EQ(s.os, "Ubuntu 26.04.1 LTS");
    EXPECT_EQ(s.rocm_version, "7.15.26333");
    EXPECT_EQ(s.vulkan_api_version, "1.4.318");
    EXPECT_FALSE(s.vulkan_drivers.empty());
    EXPECT_FALSE(s.mesa_version);  // not discovered: null, never invented
    EXPECT_FALSE(s.driver_version);
    // No platform (BIOS/EC) mode supplied -> power mode is not fully known.
    EXPECT_EQ(s.power_mode, "unknown|auto/BOOTUP_DEFAULT");
    EXPECT_FALSE(power_mode_known(s.power_mode));

    const HardwareState labeled = capture_hardware_state(evo_x2(), "performance");
    EXPECT_EQ(labeled.power_mode, "performance|auto/BOOTUP_DEFAULT");
    EXPECT_TRUE(power_mode_known(labeled.power_mode));
    EXPECT_EQ(labeled.captured_at.size(), 20u);  // "YYYY-MM-DDTHH:MM:SSZ"
    EXPECT_EQ(labeled.captured_at.back(), 'Z');
}

TEST(HwState, PowerModeString) {
    EXPECT_EQ(make_power_mode("", std::nullopt, std::nullopt), "unknown|unknown/unknown");
    EXPECT_EQ(make_power_mode("quiet", std::string(""), std::string("COMPUTE")), "quiet|unknown/COMPUTE");
    EXPECT_FALSE(power_mode_known(""));
    EXPECT_FALSE(power_mode_known("a||b"));
    EXPECT_FALSE(power_mode_known("a|b/"));
    EXPECT_TRUE(power_mode_known("a|b/c"));
}

TEST(HwState, CrossPowerModeComparisonRejected) {
    EXPECT_TRUE(check_comparable(pinned(40), pinned(41)).comparable);

    HardwareState other = pinned(40);
    other.power_mode = make_power_mode("quiet", std::string("manual"), std::string("COMPUTE"));
    const auto c = check_comparable(pinned(40), other);
    EXPECT_FALSE(c.comparable);
    EXPECT_NE(c.reason.find("differ"), std::string::npos) << c.reason;

    HardwareState unknown = pinned(40);
    unknown.power_mode = make_power_mode("", std::string("auto"), std::string("COMPUTE"));
    // Unknown vs unknown is not comparable either: §49 requires a pinned mode.
    EXPECT_FALSE(check_comparable(unknown, unknown).comparable);
    EXPECT_FALSE(check_comparable(pinned(40), unknown).comparable);
}

TEST(HwState, ThermalDriftInvalidatesRun) {
    const auto ok = check_thermal_drift(pinned(50), pinned(55));  // exactly at threshold: valid
    EXPECT_TRUE(ok.valid);
    EXPECT_DOUBLE_EQ(*ok.drift_c, 5.0);

    const auto hot = check_thermal_drift(pinned(50), pinned(55.5));
    EXPECT_FALSE(hot.valid);
    EXPECT_DOUBLE_EQ(*hot.drift_c, 5.5);
    EXPECT_NE(hot.reason.find("exceeds"), std::string::npos);

    EXPECT_FALSE(check_thermal_drift(pinned(60), pinned(50)).valid);  // cooling also counts
    EXPECT_TRUE(check_thermal_drift(pinned(50), pinned(58), 10.0).valid);

    HardwareState missing = pinned(50);
    missing.temperature_c.reset();
    const auto m = check_thermal_drift(pinned(50), missing);
    EXPECT_FALSE(m.valid);
    EXPECT_FALSE(m.drift_c);
}

TEST(HwState, JsonRoundTripKeepsNulls) {
    const HardwareState s = capture_hardware_state(evo_x2(), "performance", "2026-09-24T00:00:00Z");
    const nlohmann::json j = s;
    EXPECT_TRUE(j.at("mesa_version").is_null());
    EXPECT_EQ(j.at("gpu_arch"), "gfx1151");
    const HardwareState back = j.get<HardwareState>();
    EXPECT_EQ(nlohmann::json(back), j);
}
