#pragma once
// Profile key (TRD §57) and the exact / version-tolerant match of TRD §56 steps 1-2.
//
// Field classes (fixed; docs/benchmarks.md repeats this table):
//   must match (a difference rejects the profile):
//     MODEL_HASH, PACK_ID, GPU_DEVICE, GPU_ARCH, POWER_MODE, ISA_TARGET
//   version-tolerant (a difference yields a *compatible, flagged* match):
//     HALO_VERSION, DRIVER_VERSION, ROCM_VERSION, VULKAN_VERSION, KERNEL_VERSION, OS
//
// POWER_MODE rule (consistent with profiling::check_comparable, TRD §49): make_profile_key
// refuses a power mode whose platform (BIOS/EC) part is unknown, and one whose GPU parts
// are unknown when the key names a GPU. A CPU-only key (empty GPU_ARCH, e.g. the WSL dev
// host, which exposes no amdgpu sysfs) may carry unknown GPU parts. Matching compares the
// full string for equality.
//
// An absent value (no ROCm installed, no Vulkan ICD) is the empty string.

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/profiling/hw_state.h"

namespace halo::autotune {

/// HALO's version string (CMake PROJECT_VERSION), the HALO_VERSION key field.
[[nodiscard]] std::string_view halo_version() noexcept;

struct ProfileKey {
    std::string halo_version;
    std::string model_hash;   ///< SHA-256 hex of the trunk GGUF
    std::string pack_id;      ///< hash over trunk + MTP file hashes
    std::string gpu_device;   ///< PCI "vendor:device", or "cpu" for a CPU-only key
    std::string gpu_arch;     ///< "gfx1151"; empty for a CPU-only key
    std::string driver_version;
    std::string rocm_version;
    std::string vulkan_version;
    std::string kernel_version;
    std::string os;
    std::string power_mode;
    std::string isa_target;   ///< "gfx1151" / "gfx11-generic" / CPU ISA label

    friend bool operator==(const ProfileKey&, const ProfileKey&) = default;
};

enum class FieldClass : std::uint8_t { MustMatch, VersionTolerant };

struct KeyField {
    std::string_view name;  ///< TRD §57 name, e.g. "POWER_MODE"
    std::string ProfileKey::*member;
    FieldClass cls;
};

/// The 12 TRD §57 fields in TRD order.
[[nodiscard]] const std::array<KeyField, 12>& key_fields() noexcept;

/// Builds the key from an M1 hardware-state snapshot. Throws Error(Config) when
/// model_hash / pack_id / isa_target is empty or the power mode violates the rule above.
/// The tuner and the runtime lookup must both build their keys with this function.
[[nodiscard]] ProfileKey make_profile_key(const profiling::HardwareState& hw, std::string model_hash,
                                          std::string pack_id, std::string isa_target);

enum class MatchKind : std::uint8_t { Exact, Compatible, Mismatch };
[[nodiscard]] std::string_view to_string(MatchKind k) noexcept;

struct KeyMatch {
    MatchKind kind = MatchKind::Mismatch;
    std::vector<std::string> differing;   ///< tolerant fields that differ (Compatible)
    std::vector<std::string> mismatched;  ///< must-match fields that differ (Mismatch)
};

/// Compares a stored profile's key against the current one.
[[nodiscard]] KeyMatch match_keys(const ProfileKey& stored, const ProfileKey& current);

void to_json(nlohmann::json& j, const ProfileKey& k);

}  // namespace halo::autotune
