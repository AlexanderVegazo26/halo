#include "halo/autotune/profile_key.h"

#include <nlohmann/json.hpp>

#include "halo/core/error.h"

#ifndef HALO_VERSION_STRING
#error "HALO_VERSION_STRING must be defined by the build (CMake PROJECT_VERSION)"
#endif

namespace halo::autotune {

std::string_view halo_version() noexcept { return HALO_VERSION_STRING; }

const std::array<KeyField, 12>& key_fields() noexcept {
    using F = FieldClass;
    static const std::array<KeyField, 12> fields{{
        {"HALO_VERSION", &ProfileKey::halo_version, F::VersionTolerant},
        {"MODEL_HASH", &ProfileKey::model_hash, F::MustMatch},
        {"PACK_ID", &ProfileKey::pack_id, F::MustMatch},
        {"GPU_DEVICE", &ProfileKey::gpu_device, F::MustMatch},
        {"GPU_ARCH", &ProfileKey::gpu_arch, F::MustMatch},
        {"DRIVER_VERSION", &ProfileKey::driver_version, F::VersionTolerant},
        {"ROCM_VERSION", &ProfileKey::rocm_version, F::VersionTolerant},
        {"VULKAN_VERSION", &ProfileKey::vulkan_version, F::VersionTolerant},
        {"KERNEL_VERSION", &ProfileKey::kernel_version, F::VersionTolerant},
        {"OS", &ProfileKey::os, F::VersionTolerant},
        {"POWER_MODE", &ProfileKey::power_mode, F::MustMatch},
        {"ISA_TARGET", &ProfileKey::isa_target, F::MustMatch},
    }};
    return fields;
}

namespace {

/// power_mode = "<platform>|<perf_level>/<power_profile>" (profiling::make_power_mode).
void check_power_mode(const std::string& pm, bool has_gpu) {
    const auto bar = pm.find('|');
    HALO_CHECK(bar != std::string::npos, ErrorCode::Config, "profile key: malformed power mode '{}'", pm);
    const std::string_view platform = std::string_view(pm).substr(0, bar);
    HALO_CHECK(!platform.empty() && platform != "unknown", ErrorCode::Config,
               "profile key: platform power mode unknown in '{}' (pass the EVO-X2 BIOS/EC mode label)", pm);
    if (has_gpu) {
        HALO_CHECK(profiling::power_mode_known(pm), ErrorCode::Config,
                   "profile key: GPU power state unknown in '{}' for a GPU key", pm);
    }
}

}  // namespace

ProfileKey make_profile_key(const profiling::HardwareState& hw, std::string model_hash, std::string pack_id,
                            std::string isa_target) {
    HALO_CHECK(!model_hash.empty(), ErrorCode::Config, "profile key: MODEL_HASH is empty");
    HALO_CHECK(!pack_id.empty(), ErrorCode::Config, "profile key: PACK_ID is empty");
    HALO_CHECK(!isa_target.empty(), ErrorCode::Config, "profile key: ISA_TARGET is empty");
    ProfileKey k;
    k.halo_version = std::string(halo_version());
    k.model_hash = std::move(model_hash);
    k.pack_id = std::move(pack_id);
    k.gpu_arch = hw.gpu_arch.value_or("");
    k.gpu_device = hw.gpu_device.value_or(k.gpu_arch.empty() ? "cpu" : "");
    HALO_CHECK(!k.gpu_device.empty(), ErrorCode::Config, "profile key: GPU arch known but GPU device unknown");
    k.driver_version = hw.driver_version.value_or(hw.mesa_version.value_or(""));
    k.rocm_version = hw.rocm_version.value_or("");
    k.vulkan_version = hw.vulkan_api_version.value_or("");
    k.kernel_version = hw.kernel.value_or("");
    k.os = hw.os.value_or("");
    k.power_mode = hw.power_mode;
    k.isa_target = std::move(isa_target);
    check_power_mode(k.power_mode, !k.gpu_arch.empty());
    return k;
}

std::string_view to_string(MatchKind k) noexcept {
    switch (k) {
        case MatchKind::Exact: return "exact";
        case MatchKind::Compatible: return "compatible";
        case MatchKind::Mismatch: return "mismatch";
    }
    return "?";
}

KeyMatch match_keys(const ProfileKey& stored, const ProfileKey& current) {
    KeyMatch m;
    for (const auto& f : key_fields()) {
        if (stored.*f.member == current.*f.member) continue;
        (f.cls == FieldClass::MustMatch ? m.mismatched : m.differing).emplace_back(f.name);
    }
    m.kind = !m.mismatched.empty() ? MatchKind::Mismatch
             : m.differing.empty() ? MatchKind::Exact
                                   : MatchKind::Compatible;
    return m;
}

void to_json(nlohmann::json& j, const ProfileKey& k) {
    j = nlohmann::json::object();
    for (const auto& f : key_fields()) j[std::string(f.name)] = k.*f.member;
}

}  // namespace halo::autotune
