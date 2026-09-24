// Device description, ICD classification, selection policy, memory-type choice.

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "device_query.h"
#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/device.h"
#include "halo/core/error.h"
#include "vk_util.h"

namespace halo::vulkan {

std::string_view to_string(DriverKind k) noexcept {
    switch (k) {
        case DriverKind::Radv: return "radv";
        case DriverKind::AmdvlkOpen: return "amdvlk";
        case DriverKind::AmdProprietary: return "amd-proprietary";
        case DriverKind::Lavapipe: return "lavapipe";
        case DriverKind::Other: return "other";
    }
    return "other";
}

DriverKind classify_driver(VkDriverId id) noexcept {
    switch (id) {
        case VK_DRIVER_ID_MESA_RADV: return DriverKind::Radv;
        case VK_DRIVER_ID_AMD_OPEN_SOURCE: return DriverKind::AmdvlkOpen;
        case VK_DRIVER_ID_AMD_PROPRIETARY: return DriverKind::AmdProprietary;
        case VK_DRIVER_ID_MESA_LLVMPIPE: return DriverKind::Lavapipe;
        default: return DriverKind::Other;
    }
}

std::string_view to_string(MemoryTier t) noexcept {
    switch (t) {
        case MemoryTier::Vram: return "VRAM";
        case MemoryTier::Gtt: return "GTT";
        case MemoryTier::Host: return "HOST";
    }
    return "HOST";
}

std::string_view to_string(MemoryUsage u) noexcept {
    switch (u) {
        case MemoryUsage::DeviceLocal: return "device-local";
        case MemoryUsage::HostVisible: return "host-visible";
        case MemoryUsage::HostCached: return "host-cached";
    }
    return "device-local";
}

void finalize_device_info(DeviceInfo& info) {
    info.driver = classify_driver(info.driver_id);
    info.correctness_only =
        info.driver == DriverKind::Lavapipe || info.type == VK_PHYSICAL_DEVICE_TYPE_CPU;
    info.tier_labels_meaningful = !info.correctness_only;
    for (MemoryTypeInfo& t : info.memory_types) {
        const bool dl = t.heap < info.heaps.size() && info.heaps[t.heap].device_local();
        t.tier = dl ? MemoryTier::Vram : MemoryTier::Gtt;
    }
}

namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    std::ranges::transform(out, out.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

int rank(const DeviceInfo& d) {
    if (d.driver == DriverKind::Radv) return 0;
    if (d.correctness_only) return 3;
    switch (d.type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        default: return 2;
    }
}

int gpu_subrank(const DeviceInfo& d) {
    return d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 0 : 1;
}

void policy_warnings(std::span<const DeviceInfo> devices, const DeviceInfo& chosen,
                     std::vector<std::string>& out) {
    if (is_amdvlk(chosen.driver)) {
        out.push_back(std::format(
            "AMDVLK selected ({}, {}): RADV is HALO's default/reference driver. AMDVLK is "
            "discontinued, can hijack the ICD, and has regressed prefill in field reports "
            "(TRD §3.2). Benchmark results on it are autotuner candidates, not the reference.",
            chosen.name, chosen.driver_info));
    } else {
        for (const DeviceInfo& d : devices) {
            if (is_amdvlk(d.driver)) {
                out.push_back(std::format(
                    "AMDVLK ICD is also installed (device {} '{}'); it can hijack the ICD for "
                    "other Vulkan applications (TRD §3.2).",
                    d.index, d.name));
                break;
            }
        }
    }
    if (chosen.correctness_only) {
        out.push_back(std::format(
            "'{}' is a CPU Vulkan implementation: correctness-only device. No timing from it "
            "is a GPU performance number (DECISIONS D-001).",
            chosen.name));
    }
}

}  // namespace

DeviceSelection select_device(std::span<const DeviceInfo> devices,
                              std::optional<std::string_view> override) {
    DeviceSelection sel;
    if (override.has_value() && !override->empty()) {
        const std::string_view ov = *override;
        std::uint32_t idx = 0;
        const auto [ptr, ec] = std::from_chars(ov.data(), ov.data() + ov.size(), idx);
        const bool numeric = ec == std::errc{} && ptr == ov.data() + ov.size();
        std::optional<std::uint32_t> match;
        for (std::uint32_t i = 0; i < devices.size(); ++i) {
            const bool hit = numeric ? devices[i].index == idx
                                     : lower(devices[i].name).find(lower(ov)) != std::string::npos;
            if (hit) {
                match = i;
                break;
            }
        }
        HALO_CHECK(match.has_value(), ErrorCode::Device,
                   "Vulkan device override '{}' matches no device ({} enumerated)", ov,
                   devices.size());
        const DeviceInfo& d = devices[*match];
        HALO_CHECK(d.suitable(), ErrorCode::Device,
                   "Vulkan device override '{}' selects '{}', which is unsuitable (API {} < 1.3 "
                   "or no compute queue)",
                   ov, d.name, detail::version_string(d.api_version));
        sel.index = *match;
        sel.reason = std::format("explicit override '{}'", ov);
        policy_warnings(devices, d, sel.warnings);
        return sel;
    }

    std::optional<std::uint32_t> best;
    for (std::uint32_t i = 0; i < devices.size(); ++i) {
        const DeviceInfo& d = devices[i];
        if (!d.suitable()) continue;
        if (!best) {
            best = i;
            continue;
        }
        const DeviceInfo& b = devices[*best];
        const int rd = rank(d);
        const int rb = rank(b);
        if (rd < rb || (rd == rb && rd == 1 && gpu_subrank(d) < gpu_subrank(b))) best = i;
    }
    HALO_CHECK(best.has_value(), ErrorCode::Device,
               "no suitable Vulkan device (need API >= 1.3 and a compute queue; {} enumerated)",
               devices.size());
    const DeviceInfo& d = devices[*best];
    sel.index = *best;
    switch (rank(d)) {
        case 0: sel.reason = "RADV (default/reference driver)"; break;
        case 1: sel.reason = "no RADV device; best GPU"; break;
        case 2: sel.reason = "no RADV device or physical GPU; other device type"; break;
        default: sel.reason = "no GPU; CPU implementation (tests / correctness only)"; break;
    }
    policy_warnings(devices, d, sel.warnings);
    return sel;
}

std::optional<std::uint32_t> choose_memory_type(std::span<const MemoryTypeInfo> types,
                                                std::uint32_t type_bits,
                                                MemoryUsage usage) noexcept {
    struct Pref {
        VkMemoryPropertyFlags required;
        VkMemoryPropertyFlags avoided;
    };
    constexpr VkMemoryPropertyFlags dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    constexpr VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr VkMemoryPropertyFlags ca = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    // Types HALO never wants (protected / lazily allocated / AMD uncached debug types).
    constexpr VkMemoryPropertyFlags never = VK_MEMORY_PROPERTY_PROTECTED_BIT |
                                            VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |
                                            VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    std::vector<Pref> prefs;
    switch (usage) {
        case MemoryUsage::DeviceLocal: prefs = {{dl, hv}, {dl, 0}, {0, 0}}; break;
        case MemoryUsage::HostVisible: prefs = {{hv | hc, ca}, {hv | hc | dl, 0}, {hv | hc, 0}}; break;
        case MemoryUsage::HostCached:
            prefs = {{hv | ca | hc, 0}, {hv | ca, 0}, {hv | hc, 0}};
            break;
    }
    for (const Pref& p : prefs) {
        for (std::uint32_t i = 0; i < types.size() && i < 32; ++i) {
            if ((type_bits & (1u << i)) == 0) continue;
            const VkMemoryPropertyFlags f = types[i].flags;
            if ((f & never) != 0) continue;
            if ((f & p.required) != p.required) continue;
            if ((f & p.avoided) != 0) continue;
            return i;
        }
    }
    return std::nullopt;
}

namespace detail {

DeviceInfo query_device(VkPhysicalDevice pd, std::uint32_t index) {
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup.pNext = &driver;
    VkPhysicalDeviceIDProperties ids{};
    ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    ids.pNext = &subgroup;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &ids;
    vkGetPhysicalDeviceProperties2(pd, &props);
    const VkPhysicalDeviceProperties& p = props.properties;
    const VkPhysicalDeviceLimits& l = p.limits;

    DeviceInfo info;
    info.index = index;
    info.name = p.deviceName;
    info.type = p.deviceType;
    info.vendor_id = p.vendorID;
    info.device_id = p.deviceID;
    info.api_version = p.apiVersion;
    info.driver_version = p.driverVersion;
    info.driver_id = driver.driverID;
    info.driver_name = driver.driverName;
    info.driver_info = driver.driverInfo;
    std::memcpy(info.device_uuid.data(), ids.deviceUUID, VK_UUID_SIZE);
    info.subgroup_size = subgroup.subgroupSize;
    info.subgroup_ops = subgroup.supportedOperations;
    info.subgroup_stages = subgroup.supportedStages;
    info.timestamp_period_ns = l.timestampPeriod;
    info.timestamp_compute_and_graphics = l.timestampComputeAndGraphics == VK_TRUE;
    for (std::size_t i = 0; i < 3; ++i) {
        info.max_workgroup_count[i] = l.maxComputeWorkGroupCount[i];
        info.max_workgroup_size[i] = l.maxComputeWorkGroupSize[i];
    }
    info.max_workgroup_invocations = l.maxComputeWorkGroupInvocations;
    info.max_shared_memory = l.maxComputeSharedMemorySize;
    info.max_push_constants = l.maxPushConstantsSize;
    info.max_storage_buffers_per_stage = l.maxPerStageDescriptorStorageBuffers;
    info.max_storage_buffer_range = l.maxStorageBufferRange;
    info.min_storage_buffer_offset_alignment = std::max<VkDeviceSize>(1, l.minStorageBufferOffsetAlignment);
    info.non_coherent_atom_size = std::max<VkDeviceSize>(1, l.nonCoherentAtomSize);

    VkPhysicalDeviceMaintenance3Properties m3{};
    m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    VkPhysicalDeviceProperties2 props_m3{};
    props_m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props_m3.pNext = &m3;
    vkGetPhysicalDeviceProperties2(pd, &props_m3);
    info.max_memory_allocation_size = m3.maxMemoryAllocationSize;

    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        info.heaps.push_back({mem.memoryHeaps[i].size, mem.memoryHeaps[i].flags});
    }
    for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        MemoryTypeInfo t;
        t.heap = mem.memoryTypes[i].heapIndex;
        t.flags = mem.memoryTypes[i].propertyFlags;
        info.memory_types.push_back(t);
    }

    std::uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    // Prefer a dedicated compute family (async compute on AMD), then any compute family;
    // within each, prefer one with timestamp support.
    auto pick = [&](bool dedicated, bool need_ts) -> std::optional<std::uint32_t> {
        for (std::uint32_t i = 0; i < nq; ++i) {
            const VkQueueFlags f = qf[i].queueFlags;
            if ((f & VK_QUEUE_COMPUTE_BIT) == 0 || qf[i].queueCount == 0) continue;
            if (dedicated && (f & VK_QUEUE_GRAPHICS_BIT) != 0) continue;
            if (need_ts && qf[i].timestampValidBits == 0) continue;
            return i;
        }
        return std::nullopt;
    };
    for (const auto& [ded, ts] : {std::pair{true, true}, std::pair{false, true},
                                  std::pair{true, false}, std::pair{false, false}}) {
        if (auto fam = pick(ded, ts)) {
            info.compute_queue_family = fam;
            info.compute_queue_timestamp_bits = qf[*fam].timestampValidBits;
            info.compute_queue_dedicated = (qf[*fam].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0;
            break;
        }
    }
    finalize_device_info(info);
    return info;
}

}  // namespace detail

namespace {

std::string_view type_string(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

nlohmann::json memory_flag_list(VkMemoryPropertyFlags f) {
    nlohmann::json j = nlohmann::json::array();
    if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) j.push_back("device-local");
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) j.push_back("host-visible");
    if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) j.push_back("host-coherent");
    if ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) j.push_back("host-cached");
    if ((f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0) j.push_back("lazily-allocated");
    if ((f & VK_MEMORY_PROPERTY_PROTECTED_BIT) != 0) j.push_back("protected");
    if ((f & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD) != 0) j.push_back("device-coherent-amd");
    if ((f & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD) != 0) j.push_back("device-uncached-amd");
    return j;
}

}  // namespace

nlohmann::json to_json(const DeviceInfo& d) {
    nlohmann::json j;
    j["index"] = d.index;
    j["name"] = d.name;
    j["type"] = type_string(d.type);
    j["vendor_id"] = d.vendor_id;
    j["device_id"] = d.device_id;
    j["api_version"] = detail::version_string(d.api_version);
    j["driver_version_raw"] = d.driver_version;
    j["driver_id"] = static_cast<int>(d.driver_id);
    j["driver"] = to_string(d.driver);
    j["driver_name"] = d.driver_name;
    j["driver_info"] = d.driver_info;
    std::string uuid;
    for (const std::uint8_t b : d.device_uuid) uuid += std::format("{:02x}", b);
    j["device_uuid"] = uuid;
    j["suitable"] = d.suitable();
    j["correctness_only"] = d.correctness_only;
    j["subgroup_size"] = d.subgroup_size;
    j["subgroup_ops_mask"] = d.subgroup_ops;
    j["subgroup_stages_mask"] = d.subgroup_stages;
    j["timestamp_period_ns"] = d.timestamp_period_ns;
    j["timestamps_supported"] = d.timestamps_supported();
    j["limits"] = {
        {"max_workgroup_count", d.max_workgroup_count},
        {"max_workgroup_size", d.max_workgroup_size},
        {"max_workgroup_invocations", d.max_workgroup_invocations},
        {"max_shared_memory", d.max_shared_memory},
        {"max_push_constants", d.max_push_constants},
        {"max_storage_buffers_per_stage", d.max_storage_buffers_per_stage},
        {"max_storage_buffer_range", d.max_storage_buffer_range},
        {"min_storage_buffer_offset_alignment", d.min_storage_buffer_offset_alignment},
        {"non_coherent_atom_size", d.non_coherent_atom_size},
        {"max_memory_allocation_size", d.max_memory_allocation_size},
    };
    nlohmann::json heaps = nlohmann::json::array();
    for (std::size_t i = 0; i < d.heaps.size(); ++i) {
        heaps.push_back({{"index", i},
                         {"size", d.heaps[i].size},
                         {"device_local", d.heaps[i].device_local()}});
    }
    j["memory_heaps"] = heaps;
    nlohmann::json types = nlohmann::json::array();
    for (std::size_t i = 0; i < d.memory_types.size(); ++i) {
        const MemoryTypeInfo& t = d.memory_types[i];
        types.push_back({{"index", i},
                         {"heap", t.heap},
                         {"flags", memory_flag_list(t.flags)},
                         {"tier_label", to_string(t.tier)}});
    }
    j["memory_types"] = types;
    j["tier_labels_meaningful"] = d.tier_labels_meaningful;
    if (d.compute_queue_family) {
        j["compute_queue"] = {{"family", *d.compute_queue_family},
                              {"dedicated", d.compute_queue_dedicated},
                              {"timestamp_valid_bits", d.compute_queue_timestamp_bits}};
    } else {
        j["compute_queue"] = nullptr;
    }
    return j;
}

}  // namespace halo::vulkan
