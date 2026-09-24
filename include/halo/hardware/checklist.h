#pragma once
// Deployment checklist for `halo devices --verify` (TRD §59, PRD §22), topology-aware per
// D-002: a 96 GiB carveout is a valid configuration, not a checklist failure.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/hardware.h"

namespace halo::hardware {

enum class CheckStatus : std::uint8_t { Ok, Warn, Fail, Unknown };
[[nodiscard]] std::string_view to_string(CheckStatus s) noexcept;

struct CheckItem {
    std::string check;   ///< stable id: "amd_gpu", "carveout", "gtt", "iommu", ...
    CheckStatus status = CheckStatus::Unknown;
    std::string detail;  ///< human-readable evidence and remedy
};

// Policy thresholds (HALO policy, not hardware facts).
/// Carveout at or above this is the D-002 "large carveout" layout.
inline constexpr std::uint64_t kLargeCarveoutBytes = 64ULL << 30;
/// Carveout at or below this is the TRD §59 "UMA frame buffer at minimum" layout.
inline constexpr std::uint64_t kSmallCarveoutBytes = 2ULL << 30;
/// In the small-carveout layout, GTT below this is flagged (TRD §59: "~120 GB class").
inline constexpr std::uint64_t kGttWorkhorseBytes = 64ULL << 30;

/// Checks, in order: amd_gpu, gfx_target, carveout, gtt, iommu, vulkan_driver, kernel,
/// rocm, performance_mode.
[[nodiscard]] std::vector<CheckItem> deployment_checklist(const HardwareInfo& info);

/// Worst status across items (Fail > Warn > Unknown > Ok).
[[nodiscard]] CheckStatus overall_status(const std::vector<CheckItem>& items) noexcept;

void to_json(nlohmann::json& j, const CheckItem& v);

}  // namespace halo::hardware
