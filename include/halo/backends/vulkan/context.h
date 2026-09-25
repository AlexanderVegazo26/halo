#pragma once
/// \file
/// Vulkan instance + logical device ownership (TRD §20).
///
/// Ownership / lifetime: `Instance` and `Context` are always held by `std::shared_ptr`.
/// Every resource created from a Context (Buffer, Kernel, Stream) keeps a shared_ptr to
/// it, so the VkDevice outlives all of its child objects regardless of destruction order
/// (TRD §32: GPU resources are released on every path, including exceptions).
///
/// Thread safety: a Context may be used from several threads. Queue submission and the
/// internal transfer path are serialized by an internal mutex (Vulkan requires external
/// synchronization of VkQueue). Streams are single-threaded objects.
///
/// Environment:
///  - `HALO_VK_DEVICE=<index|name substring>` overrides device selection (config wins).
///  - `HALO_VK_VALIDATION=1` enables `VK_LAYER_KHRONOS_validation` when it is installed
///    (a warning is logged when requested but unavailable).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <vulkan/vulkan.h>

#include "halo/backends/vulkan/device.h"

namespace halo::vulkan {

class Instance {
public:
    struct Options {
        /// nullopt: read HALO_VK_VALIDATION from the environment.
        std::optional<bool> validation;
    };

    /// Throws Error(Device) if the Vulkan loader or an instance is unavailable.
    [[nodiscard]] static std::shared_ptr<Instance> create(const Options& options);
    [[nodiscard]] static std::shared_ptr<Instance> create() { return create(Options{}); }

    ~Instance();
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    [[nodiscard]] VkInstance handle() const noexcept { return instance_; }
    [[nodiscard]] std::uint32_t api_version() const noexcept { return api_version_; }
    [[nodiscard]] bool validation_enabled() const noexcept { return validation_; }
    /// Number of validation-layer errors reported so far (0 when validation is off).
    [[nodiscard]] std::uint64_t validation_errors() const noexcept;

    /// Physical devices, described. `physical_device(i)` matches `devices()[i]`.
    [[nodiscard]] const std::vector<DeviceInfo>& devices() const noexcept { return infos_; }
    [[nodiscard]] VkPhysicalDevice physical_device(std::uint32_t index) const;

    /// Construction goes through create(); the key keeps the constructor unusable elsewhere
    /// while still allowing std::make_shared.
    class Key {
        friend class Instance;
        Key() = default;
    };
    explicit Instance(Key /*key*/) {}

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    std::uint32_t api_version_ = 0;
    bool validation_ = false;
    // Heap-allocated so the debug callback's user pointer stays valid.
    std::unique_ptr<std::atomic<std::uint64_t>> validation_errors_;
    std::vector<VkPhysicalDevice> physical_;
    std::vector<DeviceInfo> infos_;
};

struct ContextOptions {
    /// Device override (index or name substring); nullopt: HALO_VK_DEVICE, then policy.
    std::optional<std::string> device;
    Instance::Options instance;
    /// Route Buffer::upload/download through a staging buffer + GPU copy even when the
    /// destination is host-visible. Exercises the staging path on UMA/CPU devices.
    bool force_staging = false;
    /// Capacity of the context's one reusable staging buffer (code review N-1). Staged
    /// transfers larger than this are split into chunks of at most this many bytes, so
    /// host-side staging memory is bounded no matter how large a weight upload is. Must be
    /// > 0; clamped to maxMemoryAllocationSize.
    std::uint64_t staging_bytes = 16ull << 20;
    /// Initial VkPipelineCache contents (e.g. from a previous run). Ignored with a log
    /// line when the driver rejects it; the driver validates its own header.
    std::vector<std::byte> pipeline_cache_data;
};

class Context : public std::enable_shared_from_this<Context> {
public:
    /// Creates an instance, selects a device per the ICD policy, creates the logical
    /// device + compute queue. Throws Error(Device) when no suitable device exists.
    [[nodiscard]] static std::shared_ptr<Context> create(const ContextOptions& options);
    [[nodiscard]] static std::shared_ptr<Context> create() { return create(ContextOptions{}); }

    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] const DeviceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const DeviceSelection& selection() const noexcept { return selection_; }
    [[nodiscard]] const Instance& instance() const noexcept { return *instance_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_; }
    [[nodiscard]] VkQueue queue() const noexcept { return queue_; }
    [[nodiscard]] std::uint32_t queue_family() const noexcept { return queue_family_; }
    [[nodiscard]] VkPipelineCache pipeline_cache() const noexcept { return pipeline_cache_; }
    [[nodiscard]] bool force_staging() const noexcept { return force_staging_; }

    /// Thread-safe vkQueueSubmit of one command buffer signalling `fence`.
    void submit(VkCommandBuffer cmd, VkFence fence);

    /// Synchronous GPU buffer-to-buffer copy on an internal command buffer (staging
    /// path). Waits on its own fence only — never on device idle.
    void copy_buffer_sync(VkBuffer src, VkBuffer dst, VkDeviceSize src_offset,
                          VkDeviceSize dst_offset, VkDeviceSize size);

    /// Staged transfers used by Buffer::upload/download (code review N-1): the data goes
    /// through the context's single reusable staging buffer (created on first use, never
    /// grown past ContextOptions::staging_bytes) in chunks of at most that size, each
    /// chunk one synchronous GPU copy. Serialized with other transfers on this context.
    void staged_upload(VkBuffer dst, VkDeviceSize dst_offset, std::span<const std::byte> data);
    void staged_download(VkBuffer src, VkDeviceSize src_offset, std::span<std::byte> out);

    struct StagingStats {
        VkDeviceSize capacity = 0;      ///< bytes of the staging buffer (0 = not created yet)
        std::uint64_t allocations = 0;  ///< staging buffers ever created (1 once used)
        std::uint64_t chunks = 0;       ///< GPU copies issued by staged transfers
    };
    [[nodiscard]] StagingStats staging_stats() const;

    /// Current VkPipelineCache contents (driver-validated header + blob). Pair it with
    /// `shader_set_hash()` when persisting.
    [[nodiscard]] std::vector<std::byte> pipeline_cache_data() const;

    /// JSON description: selected device, policy warnings, all enumerated devices.
    [[nodiscard]] nlohmann::json describe() const;

    class Key {
        friend class Context;
        Key() = default;
    };
    explicit Context(Key /*key*/) {}

private:
    std::shared_ptr<Instance> instance_;
    DeviceInfo info_;
    DeviceSelection selection_;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    bool force_staging_ = false;

    std::mutex queue_mutex_;
    mutable std::mutex transfer_mutex_;
    VkCommandPool transfer_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer transfer_cmd_ = VK_NULL_HANDLE;
    VkFence transfer_fence_ = VK_NULL_HANDLE;

    // Reusable staging buffer (guarded by transfer_mutex_). Raw handles rather than a
    // Buffer: a Buffer holds a shared_ptr to its Context, which would be a cycle here.
    void copy_locked(VkBuffer src, VkBuffer dst, VkDeviceSize src_offset, VkDeviceSize dst_offset,
                     VkDeviceSize size);
    void ensure_staging_locked();
    void flush_staging_locked(VkDeviceSize size) const;
    void invalidate_staging_locked(VkDeviceSize size) const;
    VkDeviceSize staging_capacity_ = 0;
    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory_ = VK_NULL_HANDLE;
    VkDeviceSize staging_alloc_size_ = 0;
    VkMemoryPropertyFlags staging_flags_ = 0;
    std::byte* staging_mapped_ = nullptr;
    std::uint64_t staging_allocations_ = 0;
    std::uint64_t staging_chunks_ = 0;
};

/// JSON for one device (all fields of DeviceInfo, flags decoded).
[[nodiscard]] nlohmann::json to_json(const DeviceInfo& info);

/// Enumerate + select without creating a logical device (for `halo devices`). Returns
/// `{"available": false, "error": ...}` when there is no loader / instance.
[[nodiscard]] nlohmann::json describe_devices(std::optional<std::string_view> override = std::nullopt);

}  // namespace halo::vulkan
