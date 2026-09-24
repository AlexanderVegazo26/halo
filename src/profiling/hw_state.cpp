#include "halo/profiling/hw_state.h"

#include <chrono>
#include <cmath>
#include <format>

#include <nlohmann/json.hpp>

namespace halo::profiling {

namespace {

constexpr std::string_view kUnknown = "unknown";

std::string utc_now_iso() {
    const auto now = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now());
    return std::format("{:%FT%TZ}", now);
}

template <class T>
nlohmann::json opt(const std::optional<T>& v) {
    return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
}

template <class T>
void read_opt(const nlohmann::json& j, const char* key, std::optional<T>& out) {
    if (j.contains(key) && !j.at(key).is_null()) {
        out = j.at(key).get<T>();
    } else {
        out.reset();
    }
}

}  // namespace

std::string make_power_mode(const std::string& platform_power_mode,
                            const std::optional<std::string>& performance_level,
                            const std::optional<std::string>& power_profile) {
    const auto part = [](const std::optional<std::string>& s) -> std::string_view {
        return s && !s->empty() ? std::string_view(*s) : kUnknown;
    };
    return std::format("{}|{}/{}", platform_power_mode.empty() ? kUnknown : std::string_view(platform_power_mode),
                       part(performance_level), part(power_profile));
}

bool power_mode_known(const std::string& power_mode) {
    if (power_mode.empty()) return false;
    std::size_t start = 0;
    while (start <= power_mode.size()) {
        const std::size_t end = power_mode.find_first_of("|/", start);
        const std::string_view part =
            std::string_view(power_mode).substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (part.empty() || part == kUnknown) return false;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return true;
}

HardwareState make_hardware_state(const hardware::HardwareInfo& info, const std::string& platform_power_mode,
                                  const std::string& captured_at) {
    HardwareState s;
    s.captured_at = captured_at.empty() ? utc_now_iso() : captured_at;
    s.root = info.root;
    s.platform_power_mode = platform_power_mode;
    if (!info.gpus.empty()) {
        const auto& g = info.gpus.front();
        s.gpu_performance_level = g.telemetry.performance_level;
        s.gpu_power_profile = g.telemetry.power_profile;
        s.temperature_c = g.telemetry.temperature_c;
        s.power_w = g.telemetry.power_w;
        if (g.telemetry.sclk_mhz) s.clocks_mhz["sclk"] = *g.telemetry.sclk_mhz;
        if (g.telemetry.mclk_mhz) s.clocks_mhz["mclk"] = *g.telemetry.mclk_mhz;
        s.gpu_arch = g.gfx_target;
        s.gpu_device = std::format("{:04x}:{:04x}", g.vendor_id, g.device_id);
        if (!info.cpu.model_name.empty()) s.gpu_name = info.cpu.model_name;
    }
    s.power_mode = make_power_mode(platform_power_mode, s.gpu_performance_level, s.gpu_power_profile);
    s.kernel = info.os.kernel_release;
    s.os = info.os.distro_pretty_name;
    s.rocm_version = info.rocm.version;
    const hardware::VulkanIcd* api_icd = nullptr;
    for (const auto& icd : info.vulkan.icds) {
        s.vulkan_drivers.emplace_back(hardware::to_string(icd.kind));
        const bool have_radv = api_icd != nullptr && api_icd->kind == hardware::VulkanDriverKind::Radv;
        if (api_icd == nullptr || (!have_radv && icd.kind == hardware::VulkanDriverKind::Radv)) api_icd = &icd;
    }
    if (api_icd != nullptr) s.vulkan_api_version = api_icd->api_version;
    return s;
}

HardwareState capture_hardware_state(const hardware::DiscoveryOptions& options, const std::string& platform_power_mode,
                                     const std::string& captured_at) {
    return make_hardware_state(hardware::discover(options), platform_power_mode, captured_at);
}

Comparability check_comparable(const HardwareState& a, const HardwareState& b) {
    if (!power_mode_known(a.power_mode) || !power_mode_known(b.power_mode)) {
        return {false, std::format("power mode not pinned/known ('{}' vs '{}'); TRD §49 requires a pinned mode",
                                   a.power_mode, b.power_mode)};
    }
    if (a.power_mode != b.power_mode) {
        return {false, std::format("power modes differ ('{}' vs '{}'); cross-mode comparisons are rejected (TRD §49)",
                                   a.power_mode, b.power_mode)};
    }
    return {true, {}};
}

ThermalCheck check_thermal_drift(const HardwareState& before, const HardwareState& after, double threshold_c) {
    ThermalCheck r;
    if (!before.temperature_c || !after.temperature_c) {
        r.reason = "temperature missing in a before/after snapshot; thermal drift cannot be ruled out";
        return r;
    }
    r.drift_c = *after.temperature_c - *before.temperature_c;
    if (std::fabs(*r.drift_c) > threshold_c) {
        r.reason = std::format("thermal drift {:.1f} C exceeds threshold {:.1f} C", *r.drift_c, threshold_c);
        return r;
    }
    r.valid = true;
    return r;
}

void to_json(nlohmann::json& j, const HardwareState& v) {
    j = {{"captured_at", v.captured_at},
         {"root", v.root},
         {"power_mode", v.power_mode},
         {"platform_power_mode", v.platform_power_mode},
         {"gpu_performance_level", opt(v.gpu_performance_level)},
         {"gpu_power_profile", opt(v.gpu_power_profile)},
         {"temperature_c", opt(v.temperature_c)},
         {"power_w", opt(v.power_w)},
         {"clocks_mhz", v.clocks_mhz},
         {"gpu_arch", opt(v.gpu_arch)},
         {"gpu_device", opt(v.gpu_device)},
         {"gpu_name", opt(v.gpu_name)},
         {"kernel", opt(v.kernel)},
         {"os", opt(v.os)},
         {"rocm_version", opt(v.rocm_version)},
         {"vulkan_drivers", v.vulkan_drivers},
         {"vulkan_api_version", opt(v.vulkan_api_version)},
         {"mesa_version", opt(v.mesa_version)},
         {"driver_version", opt(v.driver_version)}};
}

void from_json(const nlohmann::json& j, HardwareState& v) {
    v.captured_at = j.value("captured_at", "");
    v.root = j.value("root", "");
    v.power_mode = j.value("power_mode", "");
    v.platform_power_mode = j.value("platform_power_mode", "");
    read_opt(j, "gpu_performance_level", v.gpu_performance_level);
    read_opt(j, "gpu_power_profile", v.gpu_power_profile);
    read_opt(j, "temperature_c", v.temperature_c);
    read_opt(j, "power_w", v.power_w);
    v.clocks_mhz = j.value("clocks_mhz", std::map<std::string, std::uint32_t>{});
    read_opt(j, "gpu_arch", v.gpu_arch);
    read_opt(j, "gpu_device", v.gpu_device);
    read_opt(j, "gpu_name", v.gpu_name);
    read_opt(j, "kernel", v.kernel);
    read_opt(j, "os", v.os);
    read_opt(j, "rocm_version", v.rocm_version);
    v.vulkan_drivers = j.value("vulkan_drivers", std::vector<std::string>{});
    read_opt(j, "vulkan_api_version", v.vulkan_api_version);
    read_opt(j, "mesa_version", v.mesa_version);
    read_opt(j, "driver_version", v.driver_version);
}

}  // namespace halo::profiling
