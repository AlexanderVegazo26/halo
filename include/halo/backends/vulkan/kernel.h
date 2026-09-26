#pragma once
/// \file
/// Compute kernels and command recording (TRD §20).
///
/// `Kernel` = SPIR-V module + specialization constants + push-constant block + N storage
/// buffer bindings (binding i = buffer i, set 0), compiled once into a compute pipeline
/// through the Context's VkPipelineCache. Kernels are immutable after construction and
/// may be shared across threads/streams.
///
/// `Stream` = command pool + command buffer + fence + descriptor pools + timestamp query
/// pool. Record dispatches/copies, `submit()`, then `wait()`. Waiting uses the stream's
/// fence only — never vkDeviceWaitIdle / vkQueueWaitIdle.
///
/// Synchronization inside a recording is conservative and automatic: every dispatch or
/// copy after the first is preceded by a full compute+transfer memory barrier, and the
/// recording ends with a device->host barrier so results are visible to mapped reads.
/// (Dependency tracking to drop redundant barriers is future work.)
///
/// Lifetime: resources bound in a recording (buffers, kernels) must stay alive until the
/// stream has been waited on. A Stream is not thread-safe; use one per thread.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.h>

#include "halo/backends/vulkan/shaders.h"

namespace halo::vulkan {

class Buffer;
class Context;

struct SpecConstant {
    std::uint32_t id = 0;
    std::uint32_t value = 0;  ///< 32-bit payload (uint/int/float bits)
};

struct KernelDesc {
    const EmbeddedShader* shader = nullptr;
    std::uint32_t num_buffers = 0;          ///< storage buffers at bindings 0..n-1
    std::uint32_t push_constant_bytes = 0;  ///< multiple of 4, <= device limit
    std::vector<SpecConstant> spec_constants;
    /// Effective local size after specialization (validated against device limits).
    std::array<std::uint32_t, 3> local_size{1, 1, 1};
};

class Kernel {
public:
    /// Throws Error(Kernel) on invalid descriptions (limits, push size, missing shader),
    /// Error(Backend) on Vulkan failures.
    Kernel(std::shared_ptr<Context> ctx, KernelDesc desc);
    ~Kernel();
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    [[nodiscard]] const KernelDesc& desc() const noexcept { return desc_; }
    [[nodiscard]] std::string_view name() const noexcept { return desc_.shader->name; }
    [[nodiscard]] VkPipeline pipeline() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }
    [[nodiscard]] VkDescriptorSetLayout set_layout() const noexcept { return set_layout_; }

private:
    void release() noexcept;
    std::shared_ptr<Context> ctx_;
    KernelDesc desc_;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

/// A storage-buffer binding: [offset, offset+range) of `buffer`. `range == 0` means
/// "to the end of the buffer". Offsets must honour minStorageBufferOffsetAlignment.
struct BufferBinding {
    const Buffer* buffer = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize range = 0;
};

using GroupCount = std::array<std::uint32_t, 3>;

/// Split `n` workgroups into a 2-D grid within the device X limit:
/// x = min(n, max_x), y = ceil(n / x). Shaders recover the linear id as
/// `gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x` and must discard ids >= n.
/// Throws Error(Kernel) if n == 0 or the grid cannot fit.
[[nodiscard]] GroupCount grid_1d(std::uint64_t n, std::uint32_t max_x, std::uint32_t max_y);

class Stream {
public:
    explicit Stream(std::shared_ptr<Context> ctx, std::uint32_t max_timestamps = 64);
    /// Waits for in-flight work on this stream's fence (errors are logged, not thrown).
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    /// Record a dispatch. `push` must be exactly the kernel's push_constant_bytes.
    /// Validates binding count, ranges, alignment, maxStorageBufferRange and group limits.
    void dispatch(const Kernel& kernel, std::span<const BufferBinding> buffers,
                  std::span<const std::byte> push, GroupCount groups);

    template <typename Push>
        requires std::is_trivially_copyable_v<Push>
    void dispatch(const Kernel& kernel, std::span<const BufferBinding> buffers, const Push& push,
                  GroupCount groups) {
        dispatch(kernel, buffers, std::as_bytes(std::span(&push, 1)), groups);
    }

    void copy(const Buffer& src, VkDeviceSize src_offset, const Buffer& dst,
              VkDeviceSize dst_offset, VkDeviceSize size);

    /// Record a fill of `size` bytes at `offset` of `dst` with the 32-bit `value`
    /// (vkCmdFillBuffer; offset and size must be non-zero multiples of 4). Ordered after
    /// all previously recorded work, like a dispatch.
    void fill(const Buffer& dst, VkDeviceSize offset, VkDeviceSize size, std::uint32_t value);

    /// Record a timestamp after all previously recorded work; returns its slot, or nullopt
    /// when the queue has no timestamp support or the slots are exhausted.
    std::optional<std::uint32_t> timestamp();

    /// Submit the recording (no-op if nothing was recorded). Non-blocking.
    void submit();
    /// Wait for the last submission. Throws Error(Backend) on timeout / device loss.
    void wait(std::uint64_t timeout_ns = 120'000'000'000ull);
    void submit_and_wait() {
        submit();
        wait();
    }

    /// Nanoseconds between two timestamp slots of the last completed submission
    /// (ticks * timestampPeriod), or nullopt if unavailable.
    [[nodiscard]] std::optional<double> elapsed_ns(std::uint32_t begin, std::uint32_t end) const;

    [[nodiscard]] std::uint32_t dispatch_count() const noexcept { return dispatches_; }

private:
    enum class State { Idle, Recording, Submitted };
    void ensure_recording();
    void barrier_if_needed();
    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout, std::uint32_t num_buffers);
    void release() noexcept;

    std::shared_ptr<Context> ctx_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    std::uint32_t max_timestamps_ = 0;
    std::uint32_t used_timestamps_ = 0;
    std::vector<std::uint64_t> timestamp_values_;
    std::vector<VkDescriptorPool> desc_pools_;
    std::size_t current_pool_ = 0;
    State state_ = State::Idle;
    bool needs_barrier_ = false;
    std::uint32_t dispatches_ = 0;
};

}  // namespace halo::vulkan
