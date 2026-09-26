#pragma once
/// \file
/// HALO HIP backend — host runtime (TRD §17): device discovery and properties, tier-aware
/// memory, streams, events, and hipError_t → halo::Error translation. HALO's own kernels
/// only; no rocBLAS / hipBLASLt / MIOpen (TRD §3.3).
///
/// This header does not include any HIP header, so it can be used from any HALO module.
/// HIP handles are exposed as opaque `void*` (a `hipStream_t` / `hipEvent_t`).
///
/// Status (DECISIONS D-001): the dev host has no AMD GPU. Everything that needs a device
/// is compile-checked here and unverified until run on the EVO-X2 (gfx1151).
///
/// Thread safety: Context, Buffer, Stream and Event are not internally synchronized; a
/// Stream must be used by one thread at a time. probe() and translate functions are
/// thread-safe.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "halo/core/error.h"

namespace halo::hip {

/// Result of probing the HIP runtime. Never throws; `reason` explains a zero count.
struct Availability {
    int device_count = 0;
    int hip_error = 0;      ///< hipGetDeviceCount's return code (0 = hipSuccess)
    std::string reason;     ///< empty when device_count > 0
    [[nodiscard]] bool available() const noexcept { return device_count > 0; }
};

/// Calls hipGetDeviceCount. A failure (e.g. hipErrorNoDevice on the dev host) is reported
/// as device_count = 0 with the HIP error string, never as an exception.
[[nodiscard]] Availability probe() noexcept;

/// The halo::ErrorCode a hipError_t value maps to:
///   out of memory → Memory; no device / invalid device / init / driver → Device;
///   launch failure, illegal address, bad configuration, missing kernel binary → Kernel;
///   not supported → Unsupported; everything else → Backend.
[[nodiscard]] ErrorCode error_code_for(int hip_error) noexcept;

/// hipGetErrorName + hipGetErrorString of a hipError_t value.
[[nodiscard]] std::string describe_hip_error(int hip_error);

/// Throws halo::Error(error_code_for(e)) with `what` and the HIP error text when e != 0.
void check(int hip_error, std::string_view what);

/// The offload architectures the device code was compiled for (CMAKE_HIP_ARCHITECTURES,
/// e.g. "gfx1151"); recorded in profile keys (TRD §3.3).
[[nodiscard]] std::string_view compiled_offload_archs() noexcept;

/// True when the build targets the gfx11-generic ISA instead of gfx1151 (TRD §63: logged).
[[nodiscard]] bool compiled_for_generic_isa() noexcept;

struct DeviceInfo {
    int ordinal = 0;
    std::string name;
    std::string gcn_arch;              ///< e.g. "gfx1151" (gcnArchName up to the first ':')
    std::string gcn_arch_full;         ///< e.g. "gfx1151:xnack-" (features included)
    int compute_units = 0;
    int warp_size = 0;
    int clock_khz = 0;
    std::uint64_t total_global_mem = 0;
    std::uint64_t shared_mem_per_block = 0;
    int max_threads_per_block = 0;
    bool integrated = false;           ///< APU (Strix Halo: true)
    bool managed_memory = false;
    int driver_version = 0;            ///< hipDriverGetVersion
    int runtime_version = 0;           ///< hipRuntimeGetVersion
};

/// One selected device. Creating a Context makes the device current on the calling thread.
class Context {
public:
    /// Throws Error(Device) when the ordinal does not exist or HIP has no device. When the
    /// build targets gfx1151 and the device reports another architecture, throws
    /// Error(Device) (the device code would not load; TRD §3.3 "fail loudly").
    [[nodiscard]] static std::shared_ptr<Context> create(int ordinal = 0);

    [[nodiscard]] const DeviceInfo& info() const noexcept { return info_; }
    /// Makes this context's device current on the calling thread.
    void make_current() const;
    /// hipDeviceSynchronize.
    void synchronize() const;
    /// Free and total device memory (hipMemGetInfo).
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> memory_info() const;

    explicit Context(DeviceInfo info) : info_(std::move(info)) {}

private:
    DeviceInfo info_;
};

/// Memory tiers (TRD §13; DECISIONS D-002). On Strix Halo "Device" is the VRAM carveout
/// and "HostPinned" is GTT (system memory mapped into the GPU address space).
enum class MemoryTier {
    Device,      ///< hipMalloc
    HostPinned,  ///< hipHostMalloc(Mapped) — device-accessible pinned host memory (GTT)
    Managed,     ///< hipMallocManaged
    Host,        ///< plain host memory wrapped for the host emulation (never device-visible)
};

[[nodiscard]] std::string_view to_string(MemoryTier t) noexcept;

/// An owning (or, for MemoryTier::Host, non-owning) memory range. Move-only.
class Buffer {
public:
    Buffer() = default;
    /// Allocates `bytes` (> 0) in `tier` (not Host). Throws Error(Memory) on allocation
    /// failure and Error(Config) for a zero size or the Host tier.
    [[nodiscard]] static Buffer allocate(const Context& ctx, std::uint64_t bytes, MemoryTier tier);
    /// Wraps host memory for the host emulation target. Non-owning: `data` must outlive the
    /// Buffer and every op that uses it.
    [[nodiscard]] static Buffer wrap_host(void* data, std::uint64_t bytes);

    ~Buffer();
    Buffer(Buffer&& o) noexcept;
    Buffer& operator=(Buffer&& o) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /// Device address (Device/Managed/HostPinned: the device-visible pointer) or host
    /// address (Host).
    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] MemoryTier tier() const noexcept { return tier_; }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

    /// Synchronous copies (hipMemcpy on the null stream; memcpy for Host). Ordered with work on
    /// every Stream (Streams use the default, blocking flags). Range-checked (Error(Memory)).
    void upload(const void* src, std::uint64_t bytes, std::uint64_t offset = 0) const;
    void download(void* dst, std::uint64_t bytes, std::uint64_t offset = 0) const;

private:
    Buffer(void* data, void* host, std::uint64_t bytes, MemoryTier tier) noexcept
        : data_(data), host_(host), bytes_(bytes), tier_(tier) {}
    void release() noexcept;

    void* data_ = nullptr;
    void* host_ = nullptr;  // HostPinned: the host pointer (freed with hipHostFree)
    std::uint64_t bytes_ = 0;
    MemoryTier tier_ = MemoryTier::Host;
};

/// An in-order command queue (hipStream_t). Move-only.
class Stream {
public:
    explicit Stream(const Context& ctx);
    ~Stream();
    Stream(Stream&& o) noexcept;
    Stream& operator=(Stream&& o) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] void* handle() const noexcept { return stream_; }
    /// Waits for all work; throws the translated error of a failed kernel.
    void synchronize() const;

private:
    void* stream_ = nullptr;
};

/// Copies `bytes` bytes from `src` at `src_offset` to `dst` at `dst_offset`, ordered in
/// `stream` (hipMemcpyAsync, device-to-device; ADR-001 COPY). With `stream` == nullptr both
/// buffers must be MemoryTier::Host (the host emulation) and the copy is a synchronous memcpy.
/// Range-checked (Error(Memory), as upload/download); the two ranges must not overlap and
/// the buffers' kinds must match the target (Error(Kernel)).
void copy_async(const Stream* stream, const Buffer& dst, std::uint64_t dst_offset, const Buffer& src,
                std::uint64_t src_offset, std::uint64_t bytes);

/// A timing event (hipEvent_t). Move-only.
class Event {
public:
    explicit Event(const Context& ctx);
    ~Event();
    Event(Event&& o) noexcept;
    Event& operator=(Event&& o) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(const Stream& s);
    void synchronize() const;
    /// Milliseconds between two recorded, completed events (hipEventElapsedTime).
    [[nodiscard]] static float elapsed_ms(const Event& start, const Event& stop);

private:
    void* event_ = nullptr;
};

}  // namespace halo::hip
