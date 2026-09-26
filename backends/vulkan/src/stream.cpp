#include <algorithm>
#include <utility>
#include <vector>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/kernel.h"
#include "halo/core/error.h"
#include "halo/core/log.h"
#include "vk_util.h"

namespace halo::vulkan {

namespace {
constexpr std::uint32_t k_sets_per_pool = 256;
constexpr std::uint32_t k_descriptors_per_pool = 256 * 16;
}  // namespace

Stream::Stream(std::shared_ptr<Context> ctx, std::uint32_t max_timestamps) : ctx_(std::move(ctx)) {
    HALO_CHECK(ctx_ != nullptr, ErrorCode::Backend, "Stream: null context");
    const VkDevice dev = ctx_->device();
    try {
        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = ctx_->queue_family();
        detail::vk_check(vkCreateCommandPool(dev, &cpi, nullptr, &pool_), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        detail::vk_check(vkAllocateCommandBuffers(dev, &cai, &cmd_), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        detail::vk_check(vkCreateFence(dev, &fci, nullptr, &fence_), "vkCreateFence");
        if (ctx_->info().timestamps_supported() && max_timestamps > 0) {
            VkQueryPoolCreateInfo qpi{};
            qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = max_timestamps;
            detail::vk_check(vkCreateQueryPool(dev, &qpi, nullptr, &queries_), "vkCreateQueryPool");
            max_timestamps_ = max_timestamps;
        }
    } catch (...) {
        release();
        throw;
    }
}

Stream::~Stream() {
    if (state_ == State::Submitted && fence_ != VK_NULL_HANDLE) {
        const VkResult r = vkWaitForFences(ctx_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS) {
            HALO_ERROR("vulkan", "~Stream: fence wait failed: {}", detail::result_string(r));
        }
    }
    release();
}

void Stream::release() noexcept {
    if (!ctx_) return;
    const VkDevice dev = ctx_->device();
    for (VkDescriptorPool p : desc_pools_) vkDestroyDescriptorPool(dev, p, nullptr);
    desc_pools_.clear();
    if (queries_ != VK_NULL_HANDLE) vkDestroyQueryPool(dev, queries_, nullptr);
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(dev, fence_, nullptr);
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(dev, pool_, nullptr);  // frees cmd_
    queries_ = VK_NULL_HANDLE;
    fence_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    cmd_ = VK_NULL_HANDLE;
}

void Stream::ensure_recording() {
    if (state_ == State::Recording) return;
    if (state_ == State::Submitted) wait();
    const VkDevice dev = ctx_->device();
    detail::vk_check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    for (VkDescriptorPool p : desc_pools_) {
        detail::vk_check(vkResetDescriptorPool(dev, p, 0), "vkResetDescriptorPool");
    }
    current_pool_ = 0;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    detail::vk_check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
    if (queries_ != VK_NULL_HANDLE) vkCmdResetQueryPool(cmd_, queries_, 0, max_timestamps_);
    used_timestamps_ = 0;
    dispatches_ = 0;
    needs_barrier_ = false;
    state_ = State::Recording;
}

void Stream::barrier_if_needed() {
    if (!needs_barrier_) return;
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd_, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    needs_barrier_ = false;
}

VkDescriptorSet Stream::allocate_set(VkDescriptorSetLayout layout, std::uint32_t num_buffers) {
    HALO_CHECK(num_buffers <= k_descriptors_per_pool / k_sets_per_pool, ErrorCode::Kernel,
               "kernel uses {} buffers; stream pools support at most {}", num_buffers,
               k_descriptors_per_pool / k_sets_per_pool);
    const VkDevice dev = ctx_->device();
    while (true) {
        if (current_pool_ == desc_pools_.size()) {
            VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, k_descriptors_per_pool};
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = k_sets_per_pool;
            pci.poolSizeCount = 1;
            pci.pPoolSizes = &ps;
            VkDescriptorPool p = VK_NULL_HANDLE;
            detail::vk_check(vkCreateDescriptorPool(dev, &pci, nullptr, &p), "vkCreateDescriptorPool");
            desc_pools_.push_back(p);
        }
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = desc_pools_[current_pool_];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(dev, &ai, &set);
        if (r == VK_SUCCESS) return set;
        if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
            ++current_pool_;  // this pool is full; move to (or create) the next one
            continue;
        }
        detail::throw_vk(r, "vkAllocateDescriptorSets");
    }
}

void Stream::dispatch(const Kernel& kernel, std::span<const BufferBinding> buffers,
                      std::span<const std::byte> push, GroupCount groups) {
    const KernelDesc& d = kernel.desc();
    const DeviceInfo& info = ctx_->info();
    const std::string_view name = kernel.name();
    HALO_CHECK(kernel.pipeline() != VK_NULL_HANDLE, ErrorCode::Kernel, "dispatch of an empty kernel");
    HALO_CHECK(buffers.size() == d.num_buffers, ErrorCode::Kernel, "kernel {}: {} buffers bound, {} expected",
               name, buffers.size(), d.num_buffers);
    HALO_CHECK(push.size() == d.push_constant_bytes, ErrorCode::Kernel,
               "kernel {}: {} push-constant bytes, {} expected", name, push.size(), d.push_constant_bytes);
    for (std::size_t i = 0; i < 3; ++i) {
        HALO_CHECK(groups[i] >= 1 && groups[i] <= info.max_workgroup_count[i], ErrorCode::Kernel,
                   "kernel {}: group count[{}]={} outside [1, {}]", name, i, groups[i],
                   info.max_workgroup_count[i]);
    }
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const BufferBinding& b = buffers[i];
        HALO_CHECK(b.buffer != nullptr && b.buffer->valid(), ErrorCode::Kernel,
                   "kernel {}: binding {} is empty", name, i);
        const VkDeviceSize size = b.buffer->size();
        HALO_CHECK(b.offset < size, ErrorCode::Kernel, "kernel {}: binding {} offset {} >= size {}", name,
                   i, b.offset, size);
        HALO_CHECK(b.offset % info.min_storage_buffer_offset_alignment == 0, ErrorCode::Kernel,
                   "kernel {}: binding {} offset {} not aligned to {}", name, i, b.offset,
                   info.min_storage_buffer_offset_alignment);
        const VkDeviceSize range = b.range == 0 ? size - b.offset : b.range;
        HALO_CHECK(range <= size - b.offset, ErrorCode::Kernel,
                   "kernel {}: binding {} range [{}, +{}) exceeds size {}", name, i, b.offset, range, size);
        HALO_CHECK(range <= info.max_storage_buffer_range, ErrorCode::Kernel,
                   "kernel {}: binding {} range {} > maxStorageBufferRange {}", name, i, range,
                   info.max_storage_buffer_range);
        infos[i] = {b.buffer->handle(), b.offset, range};
    }

    ensure_recording();
    const VkDescriptorSet set = allocate_set(kernel.set_layout(), d.num_buffers);
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx_->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                           nullptr);

    barrier_if_needed();
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline());
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout(), 0, 1, &set, 0, nullptr);
    if (!push.empty()) {
        vkCmdPushConstants(cmd_, kernel.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(push.size()), push.data());
    }
    vkCmdDispatch(cmd_, groups[0], groups[1], groups[2]);
    needs_barrier_ = true;
    ++dispatches_;
}

void Stream::copy(const Buffer& src, VkDeviceSize src_offset, const Buffer& dst, VkDeviceSize dst_offset,
                  VkDeviceSize size) {
    HALO_CHECK(src.valid() && dst.valid(), ErrorCode::Kernel, "copy with an empty buffer");
    HALO_CHECK(size > 0, ErrorCode::Kernel, "copy of zero bytes");
    HALO_CHECK(src_offset <= src.size() && size <= src.size() - src_offset, ErrorCode::Kernel,
               "copy source range out of bounds");
    HALO_CHECK(dst_offset <= dst.size() && size <= dst.size() - dst_offset, ErrorCode::Kernel,
               "copy destination range out of bounds");
    ensure_recording();
    barrier_if_needed();
    const VkBufferCopy region{src_offset, dst_offset, size};
    vkCmdCopyBuffer(cmd_, src.handle(), dst.handle(), 1, &region);
    needs_barrier_ = true;
}

void Stream::fill(const Buffer& dst, VkDeviceSize offset, VkDeviceSize size, std::uint32_t value) {
    HALO_CHECK(dst.valid(), ErrorCode::Kernel, "fill of an empty buffer");
    HALO_CHECK(size > 0 && offset % 4 == 0 && size % 4 == 0, ErrorCode::Kernel,
               "fill offset {} / size {} must be non-zero multiples of 4", offset, size);
    HALO_CHECK(offset <= dst.size() && size <= dst.size() - offset, ErrorCode::Kernel,
               "fill range [{}, +{}) exceeds the buffer ({} bytes)", offset, size, dst.size());
    ensure_recording();
    barrier_if_needed();
    vkCmdFillBuffer(cmd_, dst.handle(), offset, size, value);
    needs_barrier_ = true;
}

std::optional<std::uint32_t> Stream::timestamp() {
    if (queries_ == VK_NULL_HANDLE) return std::nullopt;
    ensure_recording();
    if (used_timestamps_ >= max_timestamps_) return std::nullopt;
    const std::uint32_t slot = used_timestamps_++;
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, slot);
    return slot;
}

void Stream::submit() {
    if (state_ != State::Recording) return;
    // Device -> host visibility for mapped reads after wait().
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    detail::vk_check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    detail::vk_check(vkResetFences(ctx_->device(), 1, &fence_), "vkResetFences");
    ctx_->submit(cmd_, fence_);
    state_ = State::Submitted;
}

void Stream::wait(std::uint64_t timeout_ns) {
    if (state_ != State::Submitted) return;
    const VkResult r = vkWaitForFences(ctx_->device(), 1, &fence_, VK_TRUE, timeout_ns);
    if (r == VK_TIMEOUT) {
        throw_error(ErrorCode::Backend, "Stream::wait timed out after {} ns", timeout_ns);
    }
    detail::vk_check(r, "vkWaitForFences");
    state_ = State::Idle;
    timestamp_values_.assign(used_timestamps_, 0);
    if (used_timestamps_ > 0) {
        detail::vk_check(vkGetQueryPoolResults(ctx_->device(), queries_, 0, used_timestamps_,
                                               timestamp_values_.size() * sizeof(std::uint64_t),
                                               timestamp_values_.data(), sizeof(std::uint64_t),
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                         "vkGetQueryPoolResults");
    }
}

std::optional<double> Stream::elapsed_ns(std::uint32_t begin, std::uint32_t end) const {
    if (begin >= timestamp_values_.size() || end >= timestamp_values_.size()) return std::nullopt;
    const std::uint32_t bits = ctx_->info().compute_queue_timestamp_bits;
    const std::uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const std::uint64_t ticks = (timestamp_values_[end] - timestamp_values_[begin]) & mask;
    return static_cast<double>(ticks) * static_cast<double>(ctx_->info().timestamp_period_ns);
}

}  // namespace halo::vulkan
