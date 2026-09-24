#include "halo/backends/vulkan/buffer.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "halo/backends/vulkan/context.h"
#include "halo/core/error.h"
#include "vk_util.h"

namespace halo::vulkan {

Buffer Buffer::create(std::shared_ptr<Context> ctx, VkDeviceSize size, MemoryUsage usage) {
    HALO_CHECK(ctx != nullptr, ErrorCode::Backend, "Buffer::create: null context");
    HALO_CHECK(size > 0, ErrorCode::Memory, "Buffer::create: zero-sized buffer");
    const DeviceInfo& info = ctx->info();
    HALO_CHECK(info.max_memory_allocation_size == 0 || size <= info.max_memory_allocation_size,
               ErrorCode::Memory, "Buffer::create: {} bytes exceeds maxMemoryAllocationSize {}",
               size, info.max_memory_allocation_size);

    Buffer b;
    b.ctx_ = std::move(ctx);
    b.size_ = size;
    b.usage_ = usage;
    const VkDevice dev = b.ctx_->device();

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    detail::vk_check(vkCreateBuffer(dev, &bci, nullptr, &b.buffer_), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b.buffer_, &req);
    const auto type = choose_memory_type(info.memory_types, req.memoryTypeBits, usage);
    HALO_CHECK(type.has_value(), ErrorCode::Memory,
               "no Vulkan memory type for {} (type bits 0x{:x})", to_string(usage),
               req.memoryTypeBits);
    b.memory_type_ = *type;
    b.memory_flags_ = info.memory_types[*type].flags;
    b.tier_ = info.memory_types[*type].tier;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = *type;
    detail::vk_check(vkAllocateMemory(dev, &mai, nullptr, &b.memory_), "vkAllocateMemory");
    b.alloc_size_ = req.size;
    detail::vk_check(vkBindBufferMemory(dev, b.buffer_, b.memory_, 0), "vkBindBufferMemory");

    if ((b.memory_flags_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* p = nullptr;
        detail::vk_check(vkMapMemory(dev, b.memory_, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory");
        b.mapped_ = static_cast<std::byte*>(p);
    }
    return b;  // on any throw above, ~Buffer releases what was created
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& o) noexcept
    : ctx_(std::move(o.ctx_)),
      buffer_(std::exchange(o.buffer_, VK_NULL_HANDLE)),
      memory_(std::exchange(o.memory_, VK_NULL_HANDLE)),
      size_(std::exchange(o.size_, 0)),
      alloc_size_(std::exchange(o.alloc_size_, 0)),
      usage_(o.usage_),
      memory_type_(o.memory_type_),
      memory_flags_(o.memory_flags_),
      tier_(o.tier_),
      mapped_(std::exchange(o.mapped_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        release();
        ctx_ = std::move(o.ctx_);
        buffer_ = std::exchange(o.buffer_, VK_NULL_HANDLE);
        memory_ = std::exchange(o.memory_, VK_NULL_HANDLE);
        size_ = std::exchange(o.size_, 0);
        alloc_size_ = std::exchange(o.alloc_size_, 0);
        usage_ = o.usage_;
        memory_type_ = o.memory_type_;
        memory_flags_ = o.memory_flags_;
        tier_ = o.tier_;
        mapped_ = std::exchange(o.mapped_, nullptr);
    }
    return *this;
}

void Buffer::release() noexcept {
    if (!ctx_) return;
    const VkDevice dev = ctx_->device();
    if (mapped_ != nullptr) vkUnmapMemory(dev, memory_);
    if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(dev, buffer_, nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory(dev, memory_, nullptr);
    mapped_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    ctx_.reset();
}

namespace {

// [offset, offset+size) rounded out to nonCoherentAtomSize and clamped to the allocation.
VkMappedMemoryRange atom_range(VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize size,
                               VkDeviceSize atom, VkDeviceSize alloc_size) {
    const VkDeviceSize begin = offset / atom * atom;
    VkDeviceSize end = (offset + size + atom - 1) / atom * atom;
    end = std::min(end, alloc_size);
    VkMappedMemoryRange r{};
    r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    r.memory = mem;
    r.offset = begin;
    r.size = (end == alloc_size) ? VK_WHOLE_SIZE : end - begin;
    return r;
}

}  // namespace

void Buffer::flush(VkDeviceSize offset, VkDeviceSize size) const {
    if ((memory_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) return;
    const VkMappedMemoryRange r =
        atom_range(memory_, offset, size, ctx_->info().non_coherent_atom_size, alloc_size_);
    detail::vk_check(vkFlushMappedMemoryRanges(ctx_->device(), 1, &r), "vkFlushMappedMemoryRanges");
}

void Buffer::invalidate(VkDeviceSize offset, VkDeviceSize size) const {
    if ((memory_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) return;
    const VkMappedMemoryRange r =
        atom_range(memory_, offset, size, ctx_->info().non_coherent_atom_size, alloc_size_);
    detail::vk_check(vkInvalidateMappedMemoryRanges(ctx_->device(), 1, &r),
                     "vkInvalidateMappedMemoryRanges");
}

void Buffer::upload(std::span<const std::byte> data, VkDeviceSize offset) {
    HALO_CHECK(valid(), ErrorCode::Memory, "upload to an empty Buffer");
    HALO_CHECK(offset <= size_ && data.size() <= size_ - offset, ErrorCode::Memory,
               "upload of {} bytes at offset {} exceeds buffer size {}", data.size(), offset, size_);
    if (data.empty()) return;
    if (mapped_ != nullptr && !ctx_->force_staging()) {
        std::memcpy(mapped_ + offset, data.data(), data.size());
        flush(offset, data.size());
        return;
    }
    Buffer staging = Buffer::create(ctx_, data.size(), MemoryUsage::HostVisible);
    HALO_CHECK(staging.mapped_ != nullptr, ErrorCode::Memory, "staging buffer is not host-visible");
    std::memcpy(staging.mapped_, data.data(), data.size());
    staging.flush(0, data.size());
    ctx_->copy_buffer_sync(staging.buffer_, buffer_, 0, offset, data.size());
}

void Buffer::download(std::span<std::byte> out, VkDeviceSize offset) const {
    HALO_CHECK(valid(), ErrorCode::Memory, "download from an empty Buffer");
    HALO_CHECK(offset <= size_ && out.size() <= size_ - offset, ErrorCode::Memory,
               "download of {} bytes at offset {} exceeds buffer size {}", out.size(), offset, size_);
    if (out.empty()) return;
    if (mapped_ != nullptr && !ctx_->force_staging()) {
        invalidate(offset, out.size());
        std::memcpy(out.data(), mapped_ + offset, out.size());
        return;
    }
    Buffer staging = Buffer::create(ctx_, out.size(), MemoryUsage::HostCached);
    HALO_CHECK(staging.mapped_ != nullptr, ErrorCode::Memory, "staging buffer is not host-visible");
    ctx_->copy_buffer_sync(buffer_, staging.buffer_, offset, 0, out.size());
    staging.invalidate(0, out.size());
    std::memcpy(out.data(), staging.mapped_, out.size());
}

}  // namespace halo::vulkan
