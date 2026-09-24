#pragma once
// Numerical tolerance models for the CPU reference kernel tests (WS-D).
//
// These were fixed *before* any kernel-vs-reference difference was measured; tests print
// the observed error as a fraction of the tolerance ("ratio") so the margin is visible.
//
// eps = 2^-23 ≈ 1.19e-7 (fp32 machine epsilon; unit roundoff u = eps/2).
//
// Model E (element-wise): a result of <= 4 fp32 operations, each correctly rounded or a
//   <= 1-ulp libm/SLEEF transcendental, on both sides of the comparison:
//   |a - b| <= 8 eps |ref|  (+ a denormal floor of 1e-37).
//
// Model R (reduction of length n): fixed-order fp32 sums of n terms differ from any other
//   summation order (torch pairwise/SIMD, fp64) by a random-walk rounding error of
//   ~ sqrt(n) u * sum|terms|. With safety factor 8 on eps (16 on u):
//   |a - b| <= 8 sqrt(n) eps * S, S = sum of |terms| (or a stated upper bound of it).
//   Worst-case (n eps) bounds are ~sqrt(n) looser and would hide real bugs.
//
// Model G (Gated DeltaNet state recursion): per token the state update is a product of
//   the non-expansive maps S -> S exp(g) (g <= 0) and S -> (I - beta k k^T) S
//   (||k||_2 = 1, beta in [0, 1] => spectral norm <= 1), plus length-d_k dot products
//   (Model R). Rounding errors are therefore not amplified and add as a random walk over
//   T tokens. The chunked form adds length-C (chunk) intra-chunk reductions and a
//   unit-triangular solve of size C. Combined:
//   |a - b| <= 8 eps sqrt(T (d_k + 64)) * scale. Examples: T=300, d_k=32 -> 1.6e-4 * scale;
//   T=70, d_k=128 -> 1.1e-4 * scale. Scale, per value head:
//     outputs: max over (t, c) of sum_i |S_t[i][c] q_t[i]| (the |terms| of the output dot
//              product, as in Model R), from the fp64 reference;
//     state:   max |S| over the head.
//   (Correction, recorded: the first version used max |out| as the output scale. That
//   fails for T = 1 on heads where k.q cancels -- observed err 7.1e-10 on |out| = 3.9e-5,
//   i.e. ordinary fp32 rounding of a cancelling dot product -- because |out| is not the
//   magnitude of the rounded terms. Model R's sum-of-|terms| is the consistent scale.)
//   Two fp32 implementations each within this of fp64 are within 2x of each other.

#include <cmath>
#include <cstddef>

namespace halo::cpu::test {

inline constexpr double kEps = 1.1920928955078125e-07;  // 2^-23
inline constexpr double kDenormFloor = 1e-37;

[[nodiscard]] inline double tol_elementwise(double ref) { return 8.0 * kEps * std::abs(ref) + kDenormFloor; }

[[nodiscard]] inline double tol_reduction(std::size_t n, double sum_abs_terms) {
    return 8.0 * std::sqrt(static_cast<double>(n)) * kEps * sum_abs_terms + kDenormFloor;
}

[[nodiscard]] inline double tol_gdn(std::size_t tokens, std::size_t d_k, double scale) {
    return 8.0 * kEps * std::sqrt(static_cast<double>(tokens) * static_cast<double>(d_k + 64)) * scale;
}

}  // namespace halo::cpu::test
