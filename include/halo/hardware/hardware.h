#pragma once
// Hardware discovery (PRD FR-001, TRD §13.2, §59).
//
// Every path is resolved under a configurable filesystem root (default "/"), so the whole
// discovery can be exercised against fixture trees (tests/fixtures/sysfs/*). Discovery is
// best-effort: a missing or malformed file yields an empty optional plus, where it matters,
// a warning — never an exception. The one exception is a root that does not exist
// (ErrorCode::Device), since that is a caller error rather than a hardware property.
//
// Thread-safety: discover() is reentrant; it has no global state. The returned structs are
// plain values.

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/parse.h"
#include "halo/hardware/tier.h"

namespace halo::hardware {

inline constexpr std::uint32_t kAmdPciVendor = 0x1002;
/// Minimum kernel for the ROCm/HIP path on gfx1151 (PRD FR-001, TRD §3.3).
inline constexpr KernelVersion kMinRocmKernel{6, 19, 0};

struct CpuInfo {
    std::string model_name;
    std::string vendor_id;
    std::uint32_t logical_threads = 0;
    std::uint32_t physical_cores = 0;
    std::uint32_t sockets = 0;
    std::optional<std::uint64_t> l3_bytes;   ///< summed over distinct L3 instances in sysfs
    std::uint32_t l3_instances = 0;
    SimdFlags simd;                          ///< from <root>/proc/cpuinfo
    /// From the CPUID instruction of the *running process* (x86-64 only). This describes
    /// the machine executing HALO, which differs from `simd` when root is a fixture.
    std::optional<SimdFlags> runtime_simd;
};

struct HostMemory {
    std::optional<std::uint64_t> total_bytes;      ///< MemTotal
    std::optional<std::uint64_t> available_bytes;  ///< MemAvailable
};

struct GpuTelemetry {
    std::optional<double> temperature_c;  ///< hwmon temp1_input (edge)
    std::optional<double> power_w;        ///< hwmon power1_average, else power1_input
    std::vector<DpmLevel> sclk_levels;    ///< pp_dpm_sclk
    std::vector<DpmLevel> mclk_levels;    ///< pp_dpm_mclk
    std::optional<std::uint32_t> sclk_mhz;  ///< active sclk level
    std::optional<std::uint32_t> mclk_mhz;  ///< active mclk level
    std::optional<std::string> performance_level;  ///< power_dpm_force_performance_level
    std::optional<std::string> power_profile;      ///< active entry of pp_power_profile_mode
};

struct GpuInfo {
    std::string drm_card;  ///< "card1"
    std::optional<std::string> pci_slot;  ///< "0000:c5:00.0"
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    // Raw amdgpu memory counters (bytes), exactly as sysfs reports them.
    std::optional<std::uint64_t> vram_total;
    std::optional<std::uint64_t> vram_used;
    std::optional<std::uint64_t> vis_vram_total;
    std::optional<std::uint64_t> gtt_total;
    std::optional<std::uint64_t> gtt_used;
    // KFD topology (present only when a KFD node could be associated with this card).
    std::optional<std::uint32_t> kfd_node;
    std::string kfd_association;  ///< "pci" | "single-device" | "" (none)
    std::optional<std::uint64_t> gfx_target_version;  ///< e.g. 110501
    std::optional<std::string> gfx_target;             ///< e.g. "gfx1151"
    std::optional<std::uint32_t> compute_units;        ///< simd_count / simd_per_cu
    std::optional<std::uint32_t> simd_count;
    std::optional<std::uint32_t> simd_per_cu;
    std::optional<std::uint32_t> wavefront_size;
    std::optional<std::uint32_t> max_engine_clock_mhz;  ///< max_engine_clk_fcompute
    GpuTelemetry telemetry;
};

/// Which memory layout the unit is configured with (D-002). Classification only; the
/// planner always works from the byte counts, never from this label.
enum class MemoryTopology : std::uint8_t {
    NoGpu,             ///< no AMD GPU found: host memory only
    CarveoutPrimary,   ///< VRAM carveout >= GTT (the real EVO-X2: 96 GiB VRAM)
    GttPrimary,        ///< small carveout, large GTT (PRD/TRD assumption)
};
[[nodiscard]] std::string_view to_string(MemoryTopology t) noexcept;

/// Normalized tier view of the primary AMD GPU plus host memory (TRD §13.1/§13.2).
///
/// GTT pages are allocated from the same OS RAM that /proc/meminfo reports, so GTT and HOST
/// are *not* additive: `gpu_accessible_bytes = vram_total + min(gtt_total, host_total)`.
/// The carveout is invisible to the OS and therefore additive.
struct MemoryTiers {
    MemoryTopology topology = MemoryTopology::NoGpu;
    std::optional<std::uint64_t> vram_total;
    std::optional<std::uint64_t> vram_used;
    std::optional<std::uint64_t> vis_vram_total;
    std::optional<std::uint64_t> gtt_total;
    std::optional<std::uint64_t> gtt_used;
    std::optional<std::uint64_t> host_total;
    std::optional<std::uint64_t> host_available;
    std::uint64_t gpu_accessible_bytes = 0;
};

struct OsInfo {
    std::optional<std::string> kernel_release;  ///< /proc/sys/kernel/osrelease
    std::optional<KernelVersion> kernel_version;
    std::optional<std::string> distro_pretty_name;
    std::optional<std::string> distro_id;
    std::optional<std::string> distro_version_id;
};

struct RocmInfo {
    bool installed = false;
    std::optional<std::string> version;  ///< contents of .info/version
    std::optional<std::string> path;     ///< e.g. "/opt/rocm"
};

struct VulkanIcd {
    std::string manifest_path;  ///< path as the loader would see it (not root-prefixed)
    std::string source;         ///< "search-dir" | "VK_DRIVER_FILES" | "VK_ICD_FILENAMES" | "VK_ADD_DRIVER_FILES"
    std::optional<std::string> library_path;
    std::optional<std::string> api_version;
    VulkanDriverKind kind = VulkanDriverKind::Unknown;
    std::optional<std::string> error;  ///< unreadable / malformed manifest
};

struct VulkanInfo {
    std::vector<VulkanIcd> icds;
    /// VK_DRIVER_FILES or VK_ICD_FILENAMES is set: the loader ignores the search dirs.
    bool override_active = false;
    [[nodiscard]] bool has(VulkanDriverKind k) const noexcept;
};

enum class IommuState : std::uint8_t { Off, On, Passthrough, Unknown };
[[nodiscard]] std::string_view to_string(IommuState s) noexcept;

struct IommuInfo {
    IommuState state = IommuState::Unknown;
    std::string evidence;  ///< human-readable reason for the classification
};

struct KernelCmdline {
    std::optional<std::string> raw;
    std::optional<std::int64_t> amdgpu_gttsize_mib;  ///< amdgpu.gttsize (MiB; -1 = driver default)
    std::optional<std::uint64_t> ttm_pages_limit;    ///< ttm.pages_limit (pages)
    /// ttm_pages_limit × 4096. Assumes 4 KiB pages (x86-64 default); labelled as such.
    std::optional<std::uint64_t> ttm_pages_limit_bytes_4k;
    std::optional<std::string> amd_iommu;  ///< value of amd_iommu=
    std::optional<std::string> iommu;      ///< value of iommu=
};

struct HardwareInfo {
    std::string root;  ///< filesystem root that was inspected
    CpuInfo cpu;
    HostMemory host_memory;
    std::vector<GpuInfo> gpus;  ///< AMD GPUs only (vendor 0x1002), sorted by card index
    MemoryTiers tiers;          ///< for gpus.front() (or host-only when empty)
    OsInfo os;
    RocmInfo rocm;
    VulkanInfo vulkan;
    IommuInfo iommu;
    KernelCmdline cmdline;
    std::vector<std::string> warnings;
};

struct DiscoveryOptions {
    std::filesystem::path root = "/";
    /// Environment consulted for VK_DRIVER_FILES / VK_ICD_FILENAMES / VK_ADD_DRIVER_FILES.
    /// Paths found there are resolved under `root`. When `use_process_env` is true, missing
    /// keys are read from the process environment.
    std::map<std::string, std::string> env;
    bool use_process_env = false;
    /// Query CPUID of the running process (x86-64 only).
    bool probe_runtime_cpuid = true;
};

/// Run discovery. Throws Error(Device) only if `options.root` is not a directory.
[[nodiscard]] HardwareInfo discover(const DiscoveryOptions& options = {});

/// Compute the normalized tier view (exposed for tests and for callers that patch fields).
[[nodiscard]] MemoryTiers make_memory_tiers(const std::vector<GpuInfo>& gpus, const HostMemory& host);

/// Classify the Vulkan/IOMMU/kernel state into human-readable warnings (FR-003 list).
[[nodiscard]] std::vector<std::string> collect_warnings(const HardwareInfo& info);

/// SIMD flags of the running process via CPUID (+XGETBV for OS AVX/AVX-512 state).
/// nullopt on non-x86-64 builds.
[[nodiscard]] std::optional<SimdFlags> runtime_simd_flags() noexcept;

void to_json(nlohmann::json& j, const SimdFlags& v);
void to_json(nlohmann::json& j, const MemoryTiers& v);
void to_json(nlohmann::json& j, const HardwareInfo& v);

}  // namespace halo::hardware
