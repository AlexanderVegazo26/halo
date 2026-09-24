#pragma once
// Pure parsers for the Linux text interfaces HALO reads during hardware discovery.
// All inputs are untrusted text (sysfs/procfs can be spoofed by fixtures, containers or a
// broken driver): parsers never throw on malformed input, they return std::nullopt or skip
// the offending line. CRLF line endings are tolerated everywhere.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace halo::hardware {

/// Strip leading/trailing ASCII whitespace (including '\r').
[[nodiscard]] std::string_view trim(std::string_view s) noexcept;

/// Split into lines on '\n', dropping a trailing '\r' from each line.
[[nodiscard]] std::vector<std::string_view> split_lines(std::string_view text);

/// Parse a full-string unsigned decimal (surrounding whitespace allowed).
[[nodiscard]] std::optional<std::uint64_t> parse_u64(std::string_view s) noexcept;
/// Parse a full-string signed decimal (surrounding whitespace allowed).
[[nodiscard]] std::optional<std::int64_t> parse_i64(std::string_view s) noexcept;
/// Parse "0x1002" / "1002" hexadecimal.
[[nodiscard]] std::optional<std::uint32_t> parse_hex_u32(std::string_view s) noexcept;

// ---- /proc/meminfo -------------------------------------------------------------------

struct MemInfo {
    std::optional<std::uint64_t> total_bytes;      ///< MemTotal
    std::optional<std::uint64_t> available_bytes;  ///< MemAvailable
};
/// Values are in kB (kibibytes, per proc(5)); converted to bytes with overflow checks.
[[nodiscard]] MemInfo parse_meminfo(std::string_view text);

// ---- /proc/cpuinfo -------------------------------------------------------------------

struct SimdFlags {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512dq = false;
    bool avx512_vnni = false;
    bool avx512_bf16 = false;
    bool avx_vnni = false;
};

struct CpuInfoText {
    std::string model_name;
    std::string vendor_id;
    std::uint32_t logical_threads = 0;  ///< number of "processor" entries
    std::uint32_t physical_cores = 0;   ///< sum of "cpu cores" over distinct "physical id"
    std::uint32_t sockets = 0;          ///< distinct "physical id" values
    SimdFlags simd;                     ///< from the first processor's "flags"
};
[[nodiscard]] CpuInfoText parse_cpuinfo(std::string_view text);
/// Map a space-separated /proc/cpuinfo "flags" value onto SimdFlags.
[[nodiscard]] SimdFlags simd_from_flags(std::string_view flags);

/// Linux cache "size" files, e.g. "32768K" or "64M". Returns bytes.
[[nodiscard]] std::optional<std::uint64_t> parse_cache_size(std::string_view s) noexcept;

// ---- kernel / OS ---------------------------------------------------------------------

struct KernelVersion {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
    friend constexpr auto operator<=>(const KernelVersion&, const KernelVersion&) = default;
};
/// "7.0.0-31-generic" -> {7,0,0}; "6.8.12" -> {6,8,12}; "6.19-rc1" -> {6,19,0}.
[[nodiscard]] std::optional<KernelVersion> parse_kernel_version(std::string_view release) noexcept;

/// /etc/os-release KEY=VALUE pairs; double/single quotes stripped, backslash escapes kept.
[[nodiscard]] std::map<std::string, std::string> parse_os_release(std::string_view text);

/// /proc/cmdline -> ordered key/value tokens ("quiet" -> {"quiet", ""}).
[[nodiscard]] std::vector<std::pair<std::string, std::string>> parse_cmdline(std::string_view text);

// ---- AMD GPU ---------------------------------------------------------------------------

/// KFD `gfx_target_version` is major*10000 + minor*100 + stepping (decimal). The gfx name
/// renders minor and stepping as single hex digits: 110501 -> "gfx1151", 90010 -> "gfx90a",
/// 90402 -> "gfx942". Returns nullopt for 0 (CPU node) and for minor/stepping >= 16.
[[nodiscard]] std::optional<std::string> decode_gfx_target_version(std::uint64_t v);

/// KFD topology `properties` file: "key value" lines of unsigned decimals.
[[nodiscard]] std::map<std::string, std::uint64_t> parse_kfd_properties(std::string_view text);

struct DpmLevel {
    std::string index;        ///< "0", "1", ... or "S" (APU deep-sleep level)
    std::uint32_t mhz = 0;
    bool active = false;
};
/// pp_dpm_sclk / pp_dpm_mclk: "0: 600Mhz\n1: 2900Mhz *\n".
[[nodiscard]] std::vector<DpmLevel> parse_dpm_levels(std::string_view text);

/// pp_power_profile_mode: returns the profile name marked with '*' (e.g. "COMPUTE"),
/// handling "1 3D_FULL_SCREEN*:", "5 COMPUTE *" and "  4 VR*" layouts.
[[nodiscard]] std::optional<std::string> parse_active_power_profile(std::string_view text);

/// PCI_SLOT_NAME "0000:c5:00.0" -> (domain, KFD-style location_id = bus<<8 | dev<<3 | fn).
struct PciAddress {
    std::uint32_t domain = 0;
    std::uint32_t location_id = 0;
};
[[nodiscard]] std::optional<PciAddress> parse_pci_slot_name(std::string_view s) noexcept;

// ---- Vulkan ------------------------------------------------------------------------------

enum class VulkanDriverKind : std::uint8_t { Radv, Amdvlk, Lavapipe, Other, Unknown };
[[nodiscard]] std::string_view to_string(VulkanDriverKind k) noexcept;

/// Classify an ICD by its manifest `library_path` (preferred) and file name (fallback):
/// libvulkan_radeon.so / radeon_icd* -> RADV; amdvlk*.so / amd_icd* -> AMDVLK;
/// libvulkan_lvp.so / lvp_icd* -> lavapipe.
[[nodiscard]] VulkanDriverKind classify_vulkan_icd(std::string_view manifest_filename,
                                                   std::string_view library_path);

}  // namespace halo::hardware
