#pragma once
// Internal helpers shared by the CPU reference kernels. Not a public header.

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>

#include "halo/backends/cpu/views.h"
#include "halo/core/error.h"

namespace halo::cpu::detail {

/// Fixed-order fp32 dot product: 8 interleaved partial sums (vectorizable without
/// -ffast-math), combined pairwise in a fixed order, then a sequential tail.
[[nodiscard]] inline float dot(const float* a, const float* b, std::size_t n) noexcept {
    constexpr std::size_t kLanes = 8;
    std::array<float, kLanes> acc{};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (std::size_t j = 0; j < kLanes; ++j) acc[j] += a[i + j] * b[i + j];
    }
    float tail = 0.0f;
    for (; i < n; ++i) tail += a[i] * b[i];
    const float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    return s + tail;
}

[[nodiscard]] inline float sum_squares(const float* a, std::size_t n) noexcept { return dot(a, a, n); }

[[nodiscard]] inline float sigmoid(float x) noexcept { return 1.0f / (1.0f + std::exp(-x)); }
[[nodiscard]] inline float silu(float x) noexcept { return x / (1.0f + std::exp(-x)); }
[[nodiscard]] inline float softplus(float x) noexcept { return x > 20.0f ? x : std::log1p(std::exp(x)); }

/// True when the element ranges touched by a and b intersect.
[[nodiscard]] inline bool overlaps(const float* a_begin, const float* a_end, const float* b_begin,
                                   const float* b_end) noexcept {
    if (a_begin == a_end || b_begin == b_end) return false;
    const std::less<const float*> lt;
    return lt(a_begin, b_end) && lt(b_begin, a_end);
}

template <class A, class B>
[[nodiscard]] bool overlaps(const A& a, const B& b) noexcept {
    return overlaps(a.data(), a.end_ptr(), b.data(), b.end_ptr());
}

/// Output `out` may alias `in` only exactly (same pointer and stride).
inline void check_alias_exact_or_disjoint(ConstRows in, ConstRows out, const char* op,
                                          const char* what) {
    if (!overlaps(in, out)) return;
    HALO_CHECK(in.data() == out.data() && in.stride() == out.stride(), ErrorCode::Kernel,
               "{}: output partially overlaps {} (only exact aliasing is allowed)", op, what);
}

/// Output `out` must not overlap `in` at all.
inline void check_disjoint(ConstRows in, ConstRows out, const char* op, const char* what) {
    HALO_CHECK(!overlaps(in, out), ErrorCode::Kernel, "{}: output overlaps {}", op, what);
}

inline void check_same_shape(ConstRows a, ConstRows b, const char* op, const char* what) {
    HALO_CHECK(a.rows() == b.rows() && a.cols() == b.cols(), ErrorCode::Kernel,
               "{}: {} is {}x{}, expected {}x{}", op, what, b.rows(), b.cols(), a.rows(), a.cols());
}

/// Overflow-checked a * b.
[[nodiscard]] inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* op) {
    HALO_CHECK(a == 0 || b <= static_cast<std::size_t>(-1) / a, ErrorCode::Kernel,
               "{}: size overflow {} * {}", op, a, b);
    return a * b;
}

}  // namespace halo::cpu::detail
