#pragma once
/// \file
/// RAII Vulkan storage buffers (TRD §20, §32). A Buffer owns one VkBuffer and one
/// VkDeviceMemory allocation (no suballocator yet); both are released in the destructor,
/// including during stack unwinding. Host-visible memory is persistently mapped.
///
/// Complete buffers are not destroyed on release: they go to the context's recycling
/// cache (Context::recycle_put) fully bound and still mapped, and Buffer::create adopts a
/// fitting cached allocation instead of calling vkAllocateMemory/vkMapMemory. The engine
/// re-creates identically-sized buffers every decode step, so this removes the per-token
/// driver allocation churn and the first-touch page-fault cost of fresh mappings. Recycled
/// content is unspecified — callers that need zeros fill explicitly.
///
/// Every buffer is created with STORAGE | TRANSFER_SRC | TRANSFER_DST usage.
///
/// Synchronization contract: upload()/download() are synchronous with respect to the
/// host, but the caller must ensure no pending GPU work (an un-waited Stream) is
/// reading or writing the buffer at the time of the call.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

#include <vulkan/vulkan.h>

#include "halo/backends/vulkan/device.h"

namespace halo::vulkan {

class Context;

class Buffer {
public:
    /// Allocates `size` bytes (> 0). Throws Error(Memory) when no memory type fits or the
    /// allocation fails, Error(Backend) on other Vulkan failures.
    [[nodiscard]] static Buffer create(std::shared_ptr<Context> ctx, VkDeviceSize size,
                                       MemoryUsage usage);

    Buffer() = default;  ///< empty (valid() == false)
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return buffer_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }
    [[nodiscard]] MemoryUsage usage() const noexcept { return usage_; }
    [[nodiscard]] std::uint32_t memory_type() const noexcept { return memory_type_; }
    [[nodiscard]] VkMemoryPropertyFlags memory_flags() const noexcept { return memory_flags_; }
    [[nodiscard]] MemoryTier tier() const noexcept { return tier_; }
    [[nodiscard]] bool host_visible() const noexcept { return mapped_ != nullptr; }

    /// Copy bytes into the buffer at `offset`. Direct memcpy when host-visible (plus a
    /// flush for non-coherent memory), otherwise through the context's bounded, reused
    /// staging buffer in chunks (ContextOptions::staging_bytes) + GPU copies.
    /// Throws Error(Memory) if [offset, offset+size) exceeds the buffer.
    void upload(std::span<const std::byte> data, VkDeviceSize offset = 0);
    /// Copy bytes out of the buffer at `offset` (invalidate for non-coherent memory).
    void download(std::span<std::byte> out, VkDeviceSize offset = 0) const;

    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void upload(std::span<const T> data, VkDeviceSize offset = 0) {
        upload(std::as_bytes(data), offset);
    }
    template <typename T>
        requires(std::is_trivially_copyable_v<T> && !std::is_const_v<T>)
    void download(std::span<T> out, VkDeviceSize offset = 0) const {
        download(std::as_writable_bytes(out), offset);
    }

private:
    void release() noexcept;
    void flush(VkDeviceSize offset, VkDeviceSize size) const;
    void invalidate(VkDeviceSize offset, VkDeviceSize size) const;

    std::shared_ptr<Context> ctx_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkDeviceSize alloc_size_ = 0;
    VkDeviceSize buffer_size_ = 0;  ///< size the VkBuffer was created with (>= size_ when recycled)
    MemoryUsage usage_ = MemoryUsage::DeviceLocal;
    std::uint32_t memory_type_ = 0;
    VkMemoryPropertyFlags memory_flags_ = 0;
    MemoryTier tier_ = MemoryTier::Vram;
    std::byte* mapped_ = nullptr;
    bool complete_ = false;  ///< fully constructed (bound + mapped): eligible for recycling
};

}  // namespace halo::vulkan
