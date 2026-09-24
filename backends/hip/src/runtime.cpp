// HIP host runtime (TRD §17): discovery, memory tiers, streams, events, error translation.

#include "halo/backends/hip/runtime.h"

#include <hip/hip_runtime_api.h>

#include <cstring>
#include <format>
#include <utility>

#include "halo/core/log.h"

#ifndef HALO_HIP_OFFLOAD_ARCHS
#define HALO_HIP_OFFLOAD_ARCHS "unknown"
#endif
#ifndef HALO_HIP_GENERIC_ISA
#define HALO_HIP_GENERIC_ISA 0
#endif

namespace halo::hip {

namespace {

hipError_t as_hip(int e) noexcept { return static_cast<hipError_t>(e); }

}  // namespace

ErrorCode error_code_for(int hip_error) noexcept {
    switch (as_hip(hip_error)) {
        case hipErrorOutOfMemory:
            return ErrorCode::Memory;
        case hipErrorNoDevice:
        case hipErrorInvalidDevice:
        case hipErrorInitializationError:
        case hipErrorInsufficientDriver:
        case hipErrorSharedObjectInitFailed:
            return ErrorCode::Device;
        case hipErrorLaunchFailure:
        case hipErrorLaunchOutOfResources:
        case hipErrorIllegalAddress:
        case hipErrorInvalidConfiguration:
        case hipErrorInvalidDeviceFunction:
        case hipErrorNoBinaryForGpu:
        case hipErrorInvalidKernelFile:
            return ErrorCode::Kernel;
        case hipErrorNotSupported:
            return ErrorCode::Unsupported;
        default:
            return ErrorCode::Backend;
    }
}

std::string describe_hip_error(int hip_error) {
    const hipError_t e = as_hip(hip_error);
    const char* name = hipGetErrorName(e);
    const char* text = hipGetErrorString(e);
    return std::format("{} ({}): {}", name != nullptr ? name : "?", hip_error, text != nullptr ? text : "?");
}

void check(int hip_error, std::string_view what) {
    if (hip_error == 0) return;
    throw_error(error_code_for(hip_error), "HIP {} failed: {}", what, describe_hip_error(hip_error));
}

std::string_view compiled_offload_archs() noexcept { return HALO_HIP_OFFLOAD_ARCHS; }
bool compiled_for_generic_isa() noexcept { return HALO_HIP_GENERIC_ISA != 0; }

Availability probe() noexcept {
    Availability a;
    int n = 0;
    const hipError_t e = hipGetDeviceCount(&n);
    a.hip_error = static_cast<int>(e);
    if (e != hipSuccess) {
        a.device_count = 0;
        try {
            a.reason = describe_hip_error(a.hip_error);
        } catch (...) {
            a.reason = "hipGetDeviceCount failed";
        }
        return a;
    }
    a.device_count = n;
    if (n <= 0) a.reason = "hipGetDeviceCount reported 0 devices";
    return a;
}

// ---- Context --------------------------------------------------------------------------

std::shared_ptr<Context> Context::create(int ordinal) {
    const Availability a = probe();
    HALO_CHECK(a.available(), ErrorCode::Device, "HIP: no device ({})", a.reason);
    HALO_CHECK(ordinal >= 0 && ordinal < a.device_count, ErrorCode::Device,
               "HIP: device ordinal {} out of range (count {})", ordinal, a.device_count);
    check(hipSetDevice(ordinal), "hipSetDevice");
    hipDeviceProp_t p{};
    check(hipGetDeviceProperties(&p, ordinal), "hipGetDeviceProperties");
    DeviceInfo info;
    info.ordinal = ordinal;
    info.name = p.name;
    info.gcn_arch_full = p.gcnArchName;
    info.gcn_arch = info.gcn_arch_full.substr(0, info.gcn_arch_full.find(':'));
    info.compute_units = p.multiProcessorCount;
    info.warp_size = p.warpSize;
    info.clock_khz = p.clockRate;
    info.total_global_mem = p.totalGlobalMem;
    info.shared_mem_per_block = p.sharedMemPerBlock;
    info.max_threads_per_block = p.maxThreadsPerBlock;
    info.integrated = p.integrated != 0;
    info.managed_memory = p.managedMemory != 0;
    check(hipDriverGetVersion(&info.driver_version), "hipDriverGetVersion");
    check(hipRuntimeGetVersion(&info.runtime_version), "hipRuntimeGetVersion");

    const std::string_view archs = compiled_offload_archs();
    if (compiled_for_generic_isa()) {
        HALO_INFO("hip", "device code built for generic ISA '{}' (TRD §63); device is {}", archs, info.gcn_arch);
    } else {
        HALO_CHECK(archs.find(info.gcn_arch) != std::string_view::npos, ErrorCode::Device,
                   "HIP: device {} is {}, but the kernels were compiled for '{}' (rebuild with "
                   "-DCMAKE_HIP_ARCHITECTURES={})",
                   info.name, info.gcn_arch, archs, info.gcn_arch);
    }
    HALO_INFO("hip", "device {}: {} ({}), {} CUs, wave{}, {} MiB, driver {}, runtime {}", ordinal, info.name,
              info.gcn_arch_full, info.compute_units, info.warp_size, info.total_global_mem >> 20,
              info.driver_version, info.runtime_version);
    return std::make_shared<Context>(std::move(info));
}

void Context::make_current() const { check(hipSetDevice(info_.ordinal), "hipSetDevice"); }

void Context::synchronize() const {
    make_current();
    check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

std::pair<std::uint64_t, std::uint64_t> Context::memory_info() const {
    make_current();
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    check(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");
    return {free_b, total_b};
}

// ---- Buffer ---------------------------------------------------------------------------

std::string_view to_string(MemoryTier t) noexcept {
    switch (t) {
        case MemoryTier::Device: return "device";
        case MemoryTier::HostPinned: return "host-pinned";
        case MemoryTier::Managed: return "managed";
        case MemoryTier::Host: return "host";
    }
    return "?";
}

Buffer Buffer::allocate(const Context& ctx, std::uint64_t bytes, MemoryTier tier) {
    HALO_CHECK(bytes > 0, ErrorCode::Config, "HIP: zero-size allocation");
    HALO_CHECK(tier != MemoryTier::Host, ErrorCode::Config,
               "HIP: MemoryTier::Host is only for Buffer::wrap_host");
    ctx.make_current();
    void* p = nullptr;
    hipError_t e = hipSuccess;
    void* host = nullptr;
    switch (tier) {
        case MemoryTier::Device:
            e = hipMalloc(&p, bytes);
            break;
        case MemoryTier::Managed:
            e = hipMallocManaged(&p, bytes, hipMemAttachGlobal);
            break;
        case MemoryTier::HostPinned:
            e = hipHostMalloc(&host, bytes, hipHostMallocMapped);
            if (e == hipSuccess) {
                e = hipHostGetDevicePointer(&p, host, 0);
                if (e != hipSuccess) static_cast<void>(hipHostFree(host));
            }
            break;
        case MemoryTier::Host:
            break;
    }
    if (e != hipSuccess) {
        throw_error(ErrorCode::Memory, "HIP: allocating {} bytes ({}) failed: {}", bytes, to_string(tier),
                    describe_hip_error(static_cast<int>(e)));
    }
    return Buffer(p, host, bytes, tier);
}

Buffer Buffer::wrap_host(void* data, std::uint64_t bytes) {
    HALO_CHECK(data != nullptr && bytes > 0, ErrorCode::Config, "HIP: wrap_host needs non-empty memory");
    return Buffer(data, data, bytes, MemoryTier::Host);
}

void Buffer::release() noexcept {
    if (data_ == nullptr) return;
    switch (tier_) {
        case MemoryTier::Device:
        case MemoryTier::Managed:
            static_cast<void>(hipFree(data_));
            break;
        case MemoryTier::HostPinned:
            static_cast<void>(hipHostFree(host_));
            break;
        case MemoryTier::Host:
            break;
    }
    data_ = nullptr;
    host_ = nullptr;
    bytes_ = 0;
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& o) noexcept
    : data_(std::exchange(o.data_, nullptr)), host_(std::exchange(o.host_, nullptr)),
      bytes_(std::exchange(o.bytes_, 0)), tier_(o.tier_) {}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        release();
        data_ = std::exchange(o.data_, nullptr);
        host_ = std::exchange(o.host_, nullptr);
        bytes_ = std::exchange(o.bytes_, 0);
        tier_ = o.tier_;
    }
    return *this;
}

void Buffer::upload(const void* src, std::uint64_t n, std::uint64_t offset) const {
    HALO_CHECK(offset <= bytes_ && n <= bytes_ - offset, ErrorCode::Memory,
               "HIP: upload of {} bytes at {} exceeds buffer of {}", n, offset, bytes_);
    if (n == 0) return;
    void* dst = static_cast<char*>(data_) + offset;
    if (tier_ == MemoryTier::Host) {
        std::memcpy(dst, src, n);
        return;
    }
    check(hipMemcpy(dst, src, n, hipMemcpyHostToDevice), "hipMemcpy(upload)");
}

void Buffer::download(void* dst, std::uint64_t n, std::uint64_t offset) const {
    HALO_CHECK(offset <= bytes_ && n <= bytes_ - offset, ErrorCode::Memory,
               "HIP: download of {} bytes at {} exceeds buffer of {}", n, offset, bytes_);
    if (n == 0) return;
    const void* src = static_cast<const char*>(data_) + offset;
    if (tier_ == MemoryTier::Host) {
        std::memcpy(dst, src, n);
        return;
    }
    check(hipMemcpy(dst, src, n, hipMemcpyDeviceToHost), "hipMemcpy(download)");
}

// ---- Stream / Event -------------------------------------------------------------------

Stream::Stream(const Context& ctx) {
    ctx.make_current();
    hipStream_t s = nullptr;
    // Default (blocking) flags: the stream synchronizes with the null stream, which is where
    // Buffer::upload/download (hipMemcpy) run, so a kernel never races a copy.
    check(hipStreamCreateWithFlags(&s, hipStreamDefault), "hipStreamCreate");
    stream_ = s;
}

Stream::~Stream() {
    if (stream_ != nullptr) static_cast<void>(hipStreamDestroy(static_cast<hipStream_t>(stream_)));
}

Stream::Stream(Stream&& o) noexcept : stream_(std::exchange(o.stream_, nullptr)) {}

Stream& Stream::operator=(Stream&& o) noexcept {
    if (this != &o) {
        if (stream_ != nullptr) static_cast<void>(hipStreamDestroy(static_cast<hipStream_t>(stream_)));
        stream_ = std::exchange(o.stream_, nullptr);
    }
    return *this;
}

void Stream::synchronize() const {
    check(hipStreamSynchronize(static_cast<hipStream_t>(stream_)), "hipStreamSynchronize");
}

Event::Event(const Context& ctx) {
    ctx.make_current();
    hipEvent_t e = nullptr;
    check(hipEventCreate(&e), "hipEventCreate");
    event_ = e;
}

Event::~Event() {
    if (event_ != nullptr) static_cast<void>(hipEventDestroy(static_cast<hipEvent_t>(event_)));
}

Event::Event(Event&& o) noexcept : event_(std::exchange(o.event_, nullptr)) {}

Event& Event::operator=(Event&& o) noexcept {
    if (this != &o) {
        if (event_ != nullptr) static_cast<void>(hipEventDestroy(static_cast<hipEvent_t>(event_)));
        event_ = std::exchange(o.event_, nullptr);
    }
    return *this;
}

void Event::record(const Stream& s) {
    check(hipEventRecord(static_cast<hipEvent_t>(event_), static_cast<hipStream_t>(s.handle())), "hipEventRecord");
}

void Event::synchronize() const { check(hipEventSynchronize(static_cast<hipEvent_t>(event_)), "hipEventSynchronize"); }

float Event::elapsed_ms(const Event& start, const Event& stop) {
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start.event_), static_cast<hipEvent_t>(stop.event_)),
          "hipEventElapsedTime");
    return ms;
}

}  // namespace halo::hip
