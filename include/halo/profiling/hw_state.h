#pragma once
// Hardware-state header carried by every benchmark artifact (TRD §27, §49), captured from
// halo::hardware discovery before and after each suite.
//
// Field provenance (nothing is invented: a field discovery cannot see is null/unknown):
//   gpu_performance_level  amdgpu power_dpm_force_performance_level ("auto", "high", ...)
//   gpu_power_profile      active entry of pp_power_profile_mode ("COMPUTE", ...)
//   platform_power_mode    operator-supplied label for the EVO-X2 BIOS/EC performance
//                          (cTDP) mode, which sysfs does not expose; empty = not supplied
//   power_mode             "<platform_power_mode>|<gpu_performance_level>/<gpu_power_profile>"
//                          with "unknown" for each missing part. This is the string the
//                          harness and the profile key (TRD §57 POWER_MODE) compare.
//   temperature_c, power_w hwmon of the primary AMD GPU
//   clocks_mhz             {"sclk": active pp_dpm_sclk, "mclk": active pp_dpm_mclk}
//   gpu_arch               KFD gfx_target ("gfx1151"); gpu_device = PCI "vendor:device"
//   kernel, os             /proc/sys/kernel/osrelease, os-release PRETTY_NAME
//   rocm_version           /opt/rocm/.info/version
//   vulkan_drivers         ICD kinds found ("radv", "amdvlk", "lavapipe", ...)
//   vulkan_api_version     api_version of the first RADV ICD manifest (else first ICD)
//   mesa_version, driver_version  NOT discovered by halo_hardware today -> always null
//                          unless the caller fills them (e.g. from vulkaninfo).
//
// Policy (TRD §49):
//   * Two states are comparable only if both power modes are fully known (no "unknown"
//     part) and equal. Unknown-vs-unknown is NOT comparable: §49 requires a pinned mode.
//   * A run is invalidated by thermal drift when |T_after - T_before| exceeds the threshold,
//     or when either temperature is missing (drift cannot be ruled out).

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/hardware.h"

namespace halo::profiling {

struct HardwareState {
    std::string captured_at;  ///< ISO-8601 UTC, e.g. "2026-09-24T14:31:11Z"
    std::string root;         ///< discovery root ("/" or a fixture)
    std::string power_mode;
    std::string platform_power_mode;
    std::optional<std::string> gpu_performance_level;
    std::optional<std::string> gpu_power_profile;
    std::optional<double> temperature_c;
    std::optional<double> power_w;
    std::map<std::string, std::uint32_t> clocks_mhz;
    std::optional<std::string> gpu_arch;
    std::optional<std::string> gpu_device;
    std::optional<std::string> gpu_name;  ///< CPU model string on an APU (no GPU marketing name in sysfs)
    std::optional<std::string> kernel;
    std::optional<std::string> os;
    std::optional<std::string> rocm_version;
    std::vector<std::string> vulkan_drivers;
    std::optional<std::string> vulkan_api_version;
    std::optional<std::string> mesa_version;
    std::optional<std::string> driver_version;
};

/// Build the power-mode string (see header comment).
[[nodiscard]] std::string make_power_mode(const std::string& platform_power_mode,
                                          const std::optional<std::string>& performance_level,
                                          const std::optional<std::string>& power_profile);

/// True when no part of the power-mode string is "unknown".
[[nodiscard]] bool power_mode_known(const std::string& power_mode);

/// Project discovery results into a state header. `captured_at` defaults (empty) to the
/// system clock (UTC, second resolution).
[[nodiscard]] HardwareState make_hardware_state(const hardware::HardwareInfo& info,
                                                const std::string& platform_power_mode = {},
                                                const std::string& captured_at = {});

/// discover(options) + make_hardware_state.
[[nodiscard]] HardwareState capture_hardware_state(const hardware::DiscoveryOptions& options = {},
                                                   const std::string& platform_power_mode = {},
                                                   const std::string& captured_at = {});

struct Comparability {
    bool comparable = false;
    std::string reason;  ///< empty when comparable
};

/// TRD §49: comparisons across power modes (or with an unknown mode) are rejected.
[[nodiscard]] Comparability check_comparable(const HardwareState& a, const HardwareState& b);

/// HALO policy default: 5 °C of drift between the before/after snapshots invalidates a run.
inline constexpr double kDefaultThermalDriftC = 5.0;

struct ThermalCheck {
    bool valid = false;
    std::optional<double> drift_c;  ///< after - before
    std::string reason;
};

/// Drift between the before- and after-suite snapshots (TRD §49).
[[nodiscard]] ThermalCheck check_thermal_drift(const HardwareState& before, const HardwareState& after,
                                               double threshold_c = kDefaultThermalDriftC);

void to_json(nlohmann::json& j, const HardwareState& v);
void from_json(const nlohmann::json& j, HardwareState& v);

}  // namespace halo::profiling
