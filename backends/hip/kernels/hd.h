#pragma once
// HALO HIP backend — definitions shared by the device kernels (.hip, ROCm clang) and the
// host emulation (plain C++, clang 18). Not a public header.
//
// Execution model (the contract every kernel body in this directory is written against):
//
//   template <class Exec> HALO_HD void body(Exec& ex, Shared& sh, const Params& p,
//                                          unsigned bx, unsigned by);
//
//   * One call of `body` is one workgroup (block (bx, by) of the launch grid).
//   * `ex.phase(f)` runs `f(tid, regs)` for every thread tid < ex.block_dim() and then
//     acts as a workgroup barrier (__syncthreads on the device). `regs` is that thread's
//     private register struct; it persists across phases of the same block.
//   * Code outside phases must be block-uniform (depends only on params and block index),
//     so every thread reaches every barrier.
//   * `Shared` is the workgroup's LDS: a trivially constructible struct of arrays.
//   * Inside one phase a thread may read shared memory written by *other* threads only if
//     it was written in an earlier phase. The host emulator checks this by running
//     threads (and blocks) in forward and reverse order and poisoning LDS/regs with NaN.
//
// Device: DeviceExec (src/device_exec.h) maps phase() to f(threadIdx.x, regs) +
// __syncthreads(). Host: HostExec (src/host_exec.h) loops. The per-element math, the
// tiling and the reduction order are therefore the same source on both sides; what the
// host emulation does NOT model is device scheduling, the memory model, wave-level
// behaviour and the device libm (expf, cos) — see docs/hip.md.
//
// Floating point: every file including this header is compiled with -ffp-contract=off
// (device and host), so a*b+c is never fused and the emulation reproduces the device's
// rounding sequence. Transcendentals differ: the device uses ROCm's ocml (expf is not
// guaranteed correctly rounded), the host uses the C library, exactly like halo::cpu.

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__HIP__)
#define HALO_HD __host__ __device__
#define HALO_HIP_COMPILER 1
#else
#define HALO_HD
#define HALO_HIP_COMPILER 0
#endif

#if !HALO_HIP_COMPILER || !defined(__HIP_DEVICE_COMPILE__)
#include <cmath>
#endif

namespace halo::hip::kern {

// ---- math: device ocml vs host libm (the host side is what halo::cpu uses) -------------

HALO_HD inline float hexp(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return ::expf(x);
#else
    return std::exp(x);
#endif
}

HALO_HD inline float hsqrt(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return ::sqrtf(x);  // correctly rounded (HIP default -fhip-fp32-correctly-rounded-divide-sqrt)
#else
    return std::sqrt(x);
#endif
}

HALO_HD inline double hcos(double x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return ::cos(x);
#else
    return std::cos(x);
#endif
}

HALO_HD inline double hsin(double x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return ::sin(x);
#else
    return std::sin(x);
#endif
}

HALO_HD inline bool hisnan(float x) { return x != x; }

/// Same expressions as halo::cpu::detail (backends/cpu/kernel_common.h).
HALO_HD inline float sigmoidf(float x) { return 1.0f / (1.0f + hexp(-x)); }
HALO_HD inline float siluf(float x) { return x / (1.0f + hexp(-x)); }

// ---- the CPU reference's fixed-order dot product ---------------------------------------
// halo::cpu::detail::dot: 8 interleaved fp32 partial sums over full groups of 8, combined
// pairwise ((0+4)+(1+5))+((2+6)+(3+7)), plus a sequential tail. Reproducing it exactly is
// what lets the GDN / gated-norm kernels match the CPU reference bit for bit.

/// Lane `lane` (< 8) partial sum of a(i)*b(i) over i = lane, lane+8, ... < n - n % 8.
template <class A, class B>
HALO_HD float dot8_lane(const A& a, const B& b, unsigned n, unsigned lane) {
    float acc = 0.0f;
    const unsigned full = n - n % 8u;
    for (unsigned i = lane; i < full; i += 8u) acc += a(i) * b(i);
    return acc;
}

/// Combines 8 lane partials with the tail [n - n % 8, n) exactly as cpu::detail::dot.
template <class A, class B>
HALO_HD float dot8_combine(const float* p, const A& a, const B& b, unsigned n) {
    float tail = 0.0f;
    for (unsigned i = n - n % 8u; i < n; ++i) tail += a(i) * b(i);
    const float s = ((p[0] + p[4]) + (p[1] + p[5])) + ((p[2] + p[6]) + (p[3] + p[7]));
    return s + tail;
}

/// Serial evaluation by one thread (bit-identical to cpu::detail::dot).
template <class A, class B>
HALO_HD float dot8(const A& a, const B& b, unsigned n) {
    float p[8];
    for (unsigned j = 0; j < 8u; ++j) p[j] = dot8_lane(a, b, n, j);
    return dot8_combine(p, a, b, n);
}

/// Accessor over a strided float array.
struct At {
    const float* p;
    HALO_HD float operator()(unsigned i) const { return p[i]; }
};

/// Launch geometry of one kernel (grid.z is always 1).
struct Launch {
    unsigned grid_x = 1;
    unsigned grid_y = 1;
    unsigned block = 1;
};

/// Device status word bits (written with atomic OR on the device; see ops.h DeviceStatus).
inline constexpr std::uint32_t kStatusPositiveG = 1u;  // chunked GDN: some g > 0
inline constexpr std::uint32_t kStatusNaN = 2u;        // NaN seen by argmax/top-k

}  // namespace halo::hip::kern

namespace halo::hip::kern {

/// Compiler-only memory fence (no instruction). Placed every 16 steps of the unrolled
/// d_k loops so the scheduler cannot hoist all 128 LDS/global loads at once next to the
/// 128-register state column (measured: without it the recurrent kernel needed 256 VGPRs
/// plus scratch; with it, no scratch — see docs/hip.md). No effect on results.
HALO_HD inline void sched_fence() {
#if defined(__HIP_DEVICE_COMPILE__)
    __asm__ volatile("" ::: "memory");
#endif
}

}  // namespace halo::hip::kern
