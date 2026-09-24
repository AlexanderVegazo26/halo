#pragma once
/// \file
/// Vulkan physical-device description, ICD identification and device-selection policy
/// (TRD §3.2, §20, §62). Everything in this header is plain data plus pure functions, so
/// the policy is unit-testable without a Vulkan device.
///
/// ICD policy (TRD §3.2):
///  - RADV (`VK_DRIVER_ID_MESA_RADV`) is the default and reference driver.
///  - AMDVLK (`VK_DRIVER_ID_AMD_OPEN_SOURCE` / `VK_DRIVER_ID_AMD_PROPRIETARY`) is allowed but
///    selecting it (or merely having it installed next to RADV) emits a prominent warning.
///  - lavapipe (`VK_DRIVER_ID_MESA_LLVMPIPE`) and any CPU device are flagged
///    `correctness_only`: they validate shader results, never performance (DECISIONS D-001).

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

namespace halo::vulkan {

/// Minimum device API version HALO requires (TRD §3.2: Vulkan 1.3 compute).
inline constexpr std::uint32_t k_min_api_version = VK_API_VERSION_1_3;

enum class DriverKind { Radv, AmdvlkOpen, AmdProprietary, Lavapipe, Other };

[[nodiscard]] std::string_view to_string(DriverKind k) noexcept;
[[nodiscard]] DriverKind classify_driver(VkDriverId id) noexcept;
[[nodiscard]] constexpr bool is_amdvlk(DriverKind k) noexcept {
    return k == DriverKind::AmdvlkOpen || k == DriverKind::AmdProprietary;
}

/// HALO memory-tier label (TRD §13.1) attached to a Vulkan memory type. **Best-effort
/// only**: Vulkan exposes heaps and property flags, not carveout-vs-GTT. Rule:
///  - type on a DEVICE_LOCAL heap                  -> Vram (on the APU: BIOS carveout)
///  - type on a non-device-local heap (host memory
///    the GPU can access)                          -> Gtt
///  - Host: CPU-only memory; never produced from a Vulkan memory type, listed for
///    completeness of the tier vocabulary.
/// On a CPU device (lavapipe) every heap is "device local" host RAM, so the label is
/// meaningless there; `DeviceInfo::tier_labels_meaningful` is false in that case.
enum class MemoryTier { Vram, Gtt, Host };

[[nodiscard]] std::string_view to_string(MemoryTier t) noexcept;

struct MemoryHeapInfo {
    std::uint64_t size = 0;
    VkMemoryHeapFlags flags = 0;
    [[nodiscard]] bool device_local() const noexcept {
        return (flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    }
};

struct MemoryTypeInfo {
    std::uint32_t heap = 0;
    VkMemoryPropertyFlags flags = 0;
    MemoryTier tier = MemoryTier::Gtt;
    [[nodiscard]] bool has(VkMemoryPropertyFlags f) const noexcept { return (flags & f) == f; }
};

struct DeviceInfo {
    std::uint32_t index = 0;  ///< position in vkEnumeratePhysicalDevices order
    std::string name;
    VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
    std::uint32_t driver_version = 0;
    VkDriverId driver_id = VK_DRIVER_ID_MAX_ENUM;
    DriverKind driver = DriverKind::Other;
    std::string driver_name;
    std::string driver_info;
    std::array<std::uint8_t, VK_UUID_SIZE> device_uuid{};

    std::uint32_t subgroup_size = 0;
    VkSubgroupFeatureFlags subgroup_ops = 0;
    VkShaderStageFlags subgroup_stages = 0;

    float timestamp_period_ns = 0.0f;  ///< nanoseconds per timestamp tick
    bool timestamp_compute_and_graphics = false;

    std::array<std::uint32_t, 3> max_workgroup_count{};
    std::array<std::uint32_t, 3> max_workgroup_size{};
    std::uint32_t max_workgroup_invocations = 0;
    std::uint32_t max_shared_memory = 0;
    std::uint32_t max_push_constants = 0;
    std::uint32_t max_storage_buffers_per_stage = 0;
    std::uint64_t max_storage_buffer_range = 0;
    std::uint64_t min_storage_buffer_offset_alignment = 1;
    std::uint64_t non_coherent_atom_size = 1;
    std::uint64_t max_memory_allocation_size = 0;

    std::vector<MemoryHeapInfo> heaps;
    std::vector<MemoryTypeInfo> memory_types;

    /// Queue family HALO would use for compute (nullopt: device has no compute queue).
    std::optional<std::uint32_t> compute_queue_family;
    std::uint32_t compute_queue_timestamp_bits = 0;
    bool compute_queue_dedicated = false;  ///< family has COMPUTE but not GRAPHICS

    bool correctness_only = false;      ///< lavapipe / CPU device: results only, no perf
    bool tier_labels_meaningful = true;  ///< false on CPU devices (see MemoryTier)

    /// Meets HALO's minimum requirements (API >= 1.3, a compute queue).
    [[nodiscard]] bool suitable() const noexcept {
        return api_version >= k_min_api_version && compute_queue_family.has_value();
    }
    [[nodiscard]] bool timestamps_supported() const noexcept {
        return compute_queue_timestamp_bits > 0 && timestamp_period_ns > 0.0f;
    }
};

/// Fill tier labels / flags that are derived from the raw fields (`tier`,
/// `correctness_only`, `tier_labels_meaningful`). Called by enumeration; exposed so tests
/// can build synthetic devices.
void finalize_device_info(DeviceInfo& info);

struct DeviceSelection {
    std::uint32_t index = 0;            ///< into the span passed to select_device()
    std::string reason;                 ///< human-readable: why this device
    std::vector<std::string> warnings;  ///< ICD policy warnings to surface prominently
};

/// Device-selection policy.
///  1. `override` (config or `HALO_VK_DEVICE`): a decimal device index, or a
///     case-insensitive substring of the device name. Must match a *suitable* device, else
///     throws Error(Device) — an explicit request is never silently replaced.
///  2. Otherwise the best suitable device by rank: RADV < other GPU (discrete, then
///     integrated, then virtual; AMDVLK ranks here) < CPU (lavapipe). Ties: lower index.
/// Throws Error(Device) when no suitable device exists.
[[nodiscard]] DeviceSelection select_device(std::span<const DeviceInfo> devices,
                                            std::optional<std::string_view> override);

/// Intended use of a buffer's memory.
enum class MemoryUsage {
    DeviceLocal,  ///< GPU-only working memory (weights, activations, state)
    HostVisible,  ///< CPU-writable, coherent (uploads / small per-step inputs)
    HostCached,   ///< CPU-readable, cached (downloads / readback)
};

[[nodiscard]] std::string_view to_string(MemoryUsage u) noexcept;

/// Choose a memory type index for `usage` among those allowed by `type_bits`
/// (VkMemoryRequirements::memoryTypeBits). Returns nullopt if none fits.
///  - DeviceLocal: prefer DEVICE_LOCAL without HOST_VISIBLE, then any DEVICE_LOCAL, then any.
///  - HostVisible: require HOST_VISIBLE|HOST_COHERENT; prefer non-cached, then device-local.
///  - HostCached : prefer HOST_VISIBLE|HOST_CACHED (coherent first); fall back to HostVisible.
[[nodiscard]] std::optional<std::uint32_t> choose_memory_type(std::span<const MemoryTypeInfo> types,
                                                              std::uint32_t type_bits,
                                                              MemoryUsage usage) noexcept;

}  // namespace halo::vulkan
