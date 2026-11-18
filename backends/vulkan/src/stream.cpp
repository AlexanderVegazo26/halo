#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
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
// The pools hold Context::kMaxBindings (the most buffers one dispatch may bind) descriptors per set.
constexpr std::uint32_t k_descriptors_per_pool = k_sets_per_pool * Context::kMaxBindings;
// See Stream::dispatch: command buffers are capped at this many dispatches.
constexpr std::uint32_t kMaxDispatchesPerSubmit = 512;
}  // namespace

std::uint32_t parse_sync_every(std::string_view value) {
    if (value.empty()) return 0;
    std::uint32_t n = 0;
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), n);
    HALO_CHECK(ec == std::errc{} && ptr == value.data() + value.size(), ErrorCode::Config,
               "HALO_VK_SYNC_EVERY='{}' is not a dispatch count (0 = off, N >= 1 = sync every N dispatches)",
               value);
    return n;
}

Stream::Stream(std::shared_ptr<Context> ctx, std::uint32_t max_timestamps) : ctx_(std::move(ctx)) {
    HALO_CHECK(ctx_ != nullptr, ErrorCode::Backend, "Stream: null context");
    if (const char* p = std::getenv("HALO_VK_OP_TIMINGS"); p != nullptr && *p != '\0') {
        op_timings_ = true;
        op_timings_path_ = p;
        max_timestamps = std::max(max_timestamps, 8192u);
    }
    if (const auto p = detail::getenv_str("HALO_VK_SYNC_EVERY")) sync_every_ = parse_sync_every(*p);
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
        // One command buffer and fence: the second pair a submit split hands over to is created
        // by the first split (allocate_split_buffer), so streams that never pass 512 dispatches
        // (every decode forward) don't pay for it.
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
    if ((state_ == State::Submitted || other_pending_) && fence_ != VK_NULL_HANDLE) discard();
    if (state_ == State::Hung) {
        // The pending submission still uses the command buffer, fence and descriptor sets:
        // destroying them is invalid and may block. Leak them; Context teardown is skipped too.
        HALO_ERROR("vulkan", "~Stream: the last submission never completed; leaking the stream's Vulkan objects");
        return;
    }
    release();
}

void Stream::release() noexcept {
    if (!ctx_) return;
    const VkDevice dev = ctx_->device();
    for (VkDescriptorPool p : desc_pools_) vkDestroyDescriptorPool(dev, p, nullptr);
    desc_pools_.clear();
    for (VkDescriptorPool p : other_pools_) vkDestroyDescriptorPool(dev, p, nullptr);
    other_pools_.clear();
    if (queries_ != VK_NULL_HANDLE) vkDestroyQueryPool(dev, queries_, nullptr);
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(dev, fence_, nullptr);
    if (other_fence_ != VK_NULL_HANDLE) vkDestroyFence(dev, other_fence_, nullptr);
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(dev, pool_, nullptr);  // frees cmd_ and other_cmd_
    queries_ = VK_NULL_HANDLE;
    fence_ = VK_NULL_HANDLE;
    other_fence_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    cmd_ = VK_NULL_HANDLE;
    other_cmd_ = VK_NULL_HANDLE;
}

void Stream::ensure_recording() {
    if (state_ == State::Recording) return;
    // A hung device (this stream's or another's submission never completed) cannot run more
    // work: fail now instead of queueing behind it and waiting out another full timeout.
    HALO_CHECK(state_ != State::Hung && !ctx_->hung(), ErrorCode::Backend,
               "the Vulkan device is hung (a submission did not complete); restart the process");
    if (state_ == State::Submitted) wait();
    begin_command_buffer(cmd_, desc_pools_);
    current_pool_ = 0;
    if (queries_ != VK_NULL_HANDLE) vkCmdResetQueryPool(cmd_, queries_, 0, max_timestamps_);
    used_timestamps_ = 0;
    part_first_timestamp_ = 0;
    harvested_timestamps_ = 0;
    harvested_values_.clear();
    op_marks_.clear();
    split_waits_ = 0;
    dispatches_ = 0;
    n_touched_ = 0;
    state_ = State::Recording;
}

void Stream::allocate_split_buffer() {
    const VkDevice dev = ctx_->device();
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    detail::vk_check(vkAllocateCommandBuffers(dev, &cai, &cmd), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    const VkResult r = vkCreateFence(dev, &fci, nullptr, &fence);
    if (r != VK_SUCCESS) {
        vkFreeCommandBuffers(dev, pool_, 1, &cmd);
        detail::throw_vk(r, "vkCreateFence");
    }
    other_cmd_ = cmd;
    other_fence_ = fence;
}

void Stream::begin_command_buffer(VkCommandBuffer cmd, std::span<const VkDescriptorPool> pools) {
    const VkDevice dev = ctx_->device();
    detail::vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
    for (VkDescriptorPool p : pools) {
        detail::vk_check(vkResetDescriptorPool(dev, p, 0), "vkResetDescriptorPool");
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    detail::vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
}

void Stream::split_submit() {
    // With per-dispatch timestamps (HALO_VK_OP_TIMINGS) the next part needs room for 2 per
    // dispatch. After this split only the part being submitted holds slots that are not read
    // back (every earlier part is done once the other buffer is waited for below), so only when
    // that part alone leaves too little room wait here as before: wait() writes the profile and
    // the next recording starts at slot 0.
    if (op_timings_ && queries_ != VK_NULL_HANDLE &&
        (used_timestamps_ - part_first_timestamp_) + 2 * kMaxDispatchesPerSubmit > max_timestamps_) {
        submit();
        wait();
        return;
    }
    // Before submit(): if this throws, the recording is untouched and still Recording.
    if (other_cmd_ == VK_NULL_HANDLE) allocate_split_buffer();
    submit();
    if (other_pending_) {
        // The other command buffer still runs the part of this recording before the previous
        // split: wait for it before resetting it and its descriptor pools.
        const VkResult r = vkWaitForFences(ctx_->device(), 1, &other_fence_, VK_TRUE, kWaitNs);
        if (r == VK_TIMEOUT) {
            ctx_->mark_hung();
            throw_error(ErrorCode::Backend, "Stream: a split submission did not complete in {} ns", kWaitNs);
        }
        detail::vk_check(r, "vkWaitForFences");
        other_pending_ = false;
        ++split_waits_;
    }
    // Every part before the one just submitted is done: read its timestamps back so the next
    // part can reuse their slots (reset at its start, below).
    const std::uint32_t done = part_first_timestamp_;
    read_timestamps(harvested_timestamps_, done);
    // Begin the other buffer before swapping: if this throws, cmd_/fence_ are still the
    // submission state_ (Submitted) refers to, so wait()/discard() wait on the right fence.
    begin_command_buffer(other_cmd_, other_pools_);
    for_timestamp_slots(harvested_timestamps_, done, [&](std::uint32_t slot, std::uint32_t count, std::uint32_t) {
        vkCmdResetQueryPool(other_cmd_, queries_, slot, count);
    });
    harvested_timestamps_ = done;
    part_first_timestamp_ = used_timestamps_;
    std::swap(cmd_, other_cmd_);
    std::swap(fence_, other_fence_);
    std::swap(desc_pools_, other_pools_);
    other_pending_ = true;
    current_pool_ = 0;
    // The touch marks don't cover the submitted part, so order everything recorded from here
    // after it (a pipeline barrier's first scope includes earlier submissions to the queue).
    // The timestamp slots and op marks carry on: the query pool was reset at the recording's
    // start, and query commands on one queue run in submission order.
    barrier();
    dispatches_ = 0;
    state_ = State::Recording;
}

void Stream::barrier() {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd_, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    n_touched_ = 0;
}

void Stream::mark_touched(const Buffer* buf, bool write) {
    // Callers (barrier_for) guarantee room: n_touched_ + bindings <= kMaxTouched.
    for (std::size_t i = 0; i < n_touched_; ++i) {
        Touch& t = touched_[i];
        if (t.buf == buf) {
            t.read = true;
            t.written = t.written || write;
            return;
        }
    }
    touched_[n_touched_++] = Touch{buf, true, write};
}

void Stream::barrier_for(std::span<const BufferBinding> buffers, std::uint64_t write_mask) {
    // Keep room for every binding of this op as a new entry. A barrier is always sound (it only
    // orders more), so overflow is handled by emitting one early rather than by a fallback map.
    if (n_touched_ + buffers.size() > kMaxTouched) barrier();
    // A barrier is needed iff some binding was written since the last barrier (RAW if we
    // read it, WAW if we write it), or is written now and was read since (WAR).
    bool hazard = false;
    for (std::size_t i = 0; i < buffers.size() && !hazard; ++i) {
        for (std::size_t j = 0; j < n_touched_; ++j) {
            const Touch& t = touched_[j];
            if (t.buf != buffers[i].buffer) continue;
            hazard = t.written || (t.read && ((write_mask >> i) & 1));
            break;
        }
    }
    if (hazard) barrier();
    for (std::size_t i = 0; i < buffers.size(); ++i) mark_touched(buffers[i].buffer, (write_mask >> i) & 1);
}

VkDescriptorSet Stream::allocate_set(VkDescriptorSetLayout layout) {
    // dispatch() has checked the kernel's binding count against Context::kMaxBindings (the pools' sizing).
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
                      std::span<const std::byte> push, GroupCount groups, std::uint64_t write_mask) {
    // Very long single command buffers (thousands of compute dispatches, e.g. a large
    // prefill) wedge the compute ring on this platform (gfx1151 / RADV, observed as
    // VK_ERROR_DEVICE_LOST at prompts >= 24 tokens). Bound a command buffer to
    // kMaxDispatchesPerSubmit dispatches: submit it and continue in the stream's other
    // command buffer (split_submit), so the GPU starts on the first part while the rest is
    // recorded and the host waits for nothing until wait() (or a third buffer's worth).
    if (dispatches_ >= kMaxDispatchesPerSubmit && state_ == State::Recording) split_submit();
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
    // Fixed-size scratch instead of per-dispatch heap vectors (the bound is also what the
    // descriptor pools are sized for, see allocate_set).
    HALO_CHECK(buffers.size() <= Context::kMaxBindings, ErrorCode::Kernel,
               "kernel {}: {} buffers bound, stream supports at most {}", name, buffers.size(), Context::kMaxBindings);
    std::array<VkDescriptorBufferInfo, Context::kMaxBindings> infos;
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
    // Push descriptors (ContextOptions::push_descriptors): the writes go into the command buffer at
    // bind time (dstSet is ignored), so no set is allocated from the pools or written here.
    const PFN_vkCmdPushDescriptorSetKHR push_set = ctx_->push_descriptor_fn();
    const VkDescriptorSet set =
        push_set != nullptr ? VK_NULL_HANDLE : allocate_set(kernel.set_layout());
    std::array<VkWriteDescriptorSet, Context::kMaxBindings> writes{};  // zero-init: pNext/pImageInfo/... stay null
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    if (push_set == nullptr)
        vkUpdateDescriptorSets(ctx_->device(), static_cast<std::uint32_t>(buffers.size()), writes.data(), 0,
                               nullptr);

    barrier_for(buffers, write_mask);
    const std::optional<std::uint32_t> t0 = op_timings_ ? timestamp() : std::nullopt;
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline());
    if (push_set != nullptr) {
        // A buffer-less kernel has nothing to push (descriptorWriteCount must be > 0); its empty set needs no binding.
        if (!buffers.empty()) {
            push_set(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout(), 0,
                     static_cast<std::uint32_t>(buffers.size()), writes.data());
        }
    } else {
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout(), 0, 1, &set, 0, nullptr);
    }
    if (!push.empty()) {
        vkCmdPushConstants(cmd_, kernel.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(push.size()), push.data());
    }
    vkCmdDispatch(cmd_, groups[0], groups[1], groups[2]);
    if (op_timings_) {
        if (const std::optional<std::uint32_t> t1 = timestamp(); t0.has_value() && t1.has_value())
            op_marks_.push_back({std::string(name), *t0, *t1});
    }
    if (sync_every_ != 0) {
        std::fprintf(stderr, "[vk-sync] dispatch %u: %.*s\n", sync_logged_, static_cast<int>(name.size()),
                     name.data());
        std::fflush(stderr);
        if (sync_logged_ % sync_every_ == 0) {
            submit();
            wait();
        }
        ++sync_logged_;
    }
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
    const std::array bindings{BufferBinding{&src, src_offset, size}, BufferBinding{&dst, dst_offset, size}};
    barrier_for(bindings, 0b10);  // src read, dst written
    const VkBufferCopy region{src_offset, dst_offset, size};
    vkCmdCopyBuffer(cmd_, src.handle(), dst.handle(), 1, &region);
}

void Stream::discard() noexcept {
    if (state_ == State::Hung) return;
    // A submit split's earlier part may be in flight while this part is still recording.
    std::array<VkFence, 2> fences{};
    std::uint32_t n = 0;
    if (other_pending_) fences[n++] = other_fence_;
    if (state_ == State::Submitted) fences[n++] = fence_;
    if (n > 0) {
        // Bounded: with UINT64_MAX a hung GPU froze the caller (the engine's only worker, which
        // gets here from VkStream::wait()'s abort() after wait() already timed out) for good.
        const VkResult r = vkWaitForFences(ctx_->device(), n, fences.data(), VK_TRUE, kDiscardWaitNs);
        if (r != VK_SUCCESS) {
            HALO_ERROR("vulkan", "Stream::discard: fence wait failed: {}; marking the device hung", detail::result_string(r));
            ctx_->mark_hung();
            state_ = State::Hung;  // the command buffer is still pending: never reset or reuse it
            return;
        }
    }
    // A Recording command buffer is never submitted; ensure_recording() resets it (and the
    // descriptor pools) before the next recording begins.
    other_pending_ = false;
    split_waits_ = 0;
    state_ = State::Idle;
}

void Stream::fill(const Buffer& dst, VkDeviceSize offset, VkDeviceSize size, std::uint32_t value) {
    HALO_CHECK(dst.valid(), ErrorCode::Kernel, "fill of an empty buffer");
    HALO_CHECK(size > 0 && offset % 4 == 0 && size % 4 == 0, ErrorCode::Kernel,
               "fill offset {} / size {} must be non-zero multiples of 4", offset, size);
    HALO_CHECK(offset <= dst.size() && size <= dst.size() - offset, ErrorCode::Kernel,
               "fill range [{}, +{}) exceeds the buffer ({} bytes)", offset, size, dst.size());
    ensure_recording();
    const std::array bindings{BufferBinding{&dst, offset, size}};
    barrier_for(bindings, 0b1);  // dst written
    vkCmdFillBuffer(cmd_, dst.handle(), offset, size, value);
}

std::optional<std::uint32_t> Stream::timestamp() {
    if (queries_ == VK_NULL_HANDLE) return std::nullopt;
    ensure_recording();
    // Every slot holds a timestamp not yet read back (the ring is full).
    if (used_timestamps_ - harvested_timestamps_ >= max_timestamps_) return std::nullopt;
    const std::uint32_t id = used_timestamps_++;
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, id % max_timestamps_);
    return id;
}

template <typename Fn>
void Stream::for_timestamp_slots(std::uint32_t first, std::uint32_t last, Fn&& fn) const {
    while (first < last) {
        const std::uint32_t slot = first % max_timestamps_;
        const std::uint32_t count = std::min(last - first, max_timestamps_ - slot);
        fn(slot, count, first);
        first += count;
    }
}

void Stream::read_timestamps(std::uint32_t first, std::uint32_t last) {
    if (harvested_values_.size() < last) harvested_values_.resize(last, 0);
    for_timestamp_slots(first, last, [&](std::uint32_t slot, std::uint32_t count, std::uint32_t id) {
        detail::vk_check(vkGetQueryPoolResults(ctx_->device(), queries_, slot, count, count * sizeof(std::uint64_t),
                                               harvested_values_.data() + id, sizeof(std::uint64_t),
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                         "vkGetQueryPoolResults");
    });
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
    // The last submission and, if a split left it in flight, the earlier part of the recording.
    const std::array<VkFence, 2> fences{fence_, other_fence_};
    const VkResult r =
        vkWaitForFences(ctx_->device(), other_pending_ ? 2u : 1u, fences.data(), VK_TRUE, timeout_ns);
    if (r == VK_TIMEOUT) {
        ctx_->mark_hung();
        throw_error(ErrorCode::Backend, "Stream::wait timed out after {} ns", timeout_ns);
    }
    detail::vk_check(r, "vkWaitForFences");
    other_pending_ = false;
    state_ = State::Idle;
    // The ids split_submit() has not read back yet, after the ones it has.
    read_timestamps(harvested_timestamps_, used_timestamps_);
    timestamp_values_ = std::move(harvested_values_);
    harvested_values_.clear();
    harvested_timestamps_ = used_timestamps_;
    if (op_timings_) {
        // One "# wait" line per completed fence wait, after that submission's dispatches, so a
        // profile also counts fences (scripts/op_profile.py: fence waits, dispatches per wait).
        if (std::FILE* f = std::fopen(op_timings_path_.c_str(), "a")) {
            for (const OpMark& m : op_marks_)
                if (const std::optional<double> ns = elapsed_ns(m.begin, m.end); ns.has_value())
                    std::fprintf(f, "%s %.0f\n", m.name.c_str(), *ns);
            // split_submit()'s fence waits inside this recording, then this one.
            for (std::uint32_t i = 0; i <= split_waits_; ++i) std::fputs("# wait\n", f);
            std::fclose(f);
        }
    }
    split_waits_ = 0;
}

std::optional<double> Stream::elapsed_ns(std::uint32_t begin, std::uint32_t end) const {
    if (begin >= timestamp_values_.size() || end >= timestamp_values_.size()) return std::nullopt;
    const std::uint32_t bits = ctx_->info().compute_queue_timestamp_bits;
    const std::uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const std::uint64_t ticks = (timestamp_values_[end] - timestamp_values_[begin]) & mask;
    return static_cast<double>(ticks) * static_cast<double>(ctx_->info().timestamp_period_ns);
}

}  // namespace halo::vulkan
