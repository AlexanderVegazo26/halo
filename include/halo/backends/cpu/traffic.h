#pragma once
// HALO CPU reference backend — bytes moved per operator call (WS-D).
//
// Compulsory-traffic model for cost-model checks (e.g. a CI roofline test): every
// operand element is counted once per call (inputs read, outputs written, in/out state
// both read and written), at its stored size. It is the traffic an ideal kernel must
// move to/from memory; it is NOT a measurement of cache traffic of the CPU reference.
// Activations and state are fp32 (4 B); weights are passed as their stored byte size so
// quantized formats are counted correctly.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "halo/backends/cpu/ops.h"

namespace halo::cpu {

struct OpTraffic {
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    [[nodiscard]] constexpr std::uint64_t total() const noexcept { return bytes_read + bytes_written; }
    friend constexpr bool operator==(const OpTraffic&, const OpTraffic&) = default;
};

namespace traffic_detail {
inline constexpr std::uint64_t kF32 = 4;
[[nodiscard]] constexpr std::uint64_t u(std::size_t v) noexcept { return static_cast<std::uint64_t>(v); }
}  // namespace traffic_detail

/// rms_norm over [rows, dim]: x + w read, out written.
[[nodiscard]] constexpr OpTraffic traffic_rms_norm(std::size_t rows, std::size_t dim) noexcept {
    using namespace traffic_detail;
    return {kF32 * (u(rows) * u(dim) + u(dim)), kF32 * u(rows) * u(dim)};
}

/// gated_rms_norm over [rows, dim]: x + z + w read, out written.
[[nodiscard]] constexpr OpTraffic traffic_gated_rms_norm(std::size_t rows, std::size_t dim) noexcept {
    using namespace traffic_detail;
    return {kF32 * (2 * u(rows) * u(dim) + u(dim)), kF32 * u(rows) * u(dim)};
}

/// l2_norm_heads / unary element-wise / softmax_rows over `elems` elements.
[[nodiscard]] constexpr OpTraffic traffic_unary(std::size_t elems) noexcept {
    using namespace traffic_detail;
    return {kF32 * u(elems), kF32 * u(elems)};
}

/// Binary element-wise (add, mul, swiglu, mul_sigmoid) over `elems` elements.
[[nodiscard]] constexpr OpTraffic traffic_binary(std::size_t elems) noexcept {
    using namespace traffic_detail;
    return {2 * kF32 * u(elems), kF32 * u(elems)};
}

/// partial_rope_neox in place: only the rotated dims are read and written, + positions.
[[nodiscard]] constexpr OpTraffic traffic_partial_rope(std::size_t tokens, std::size_t n_heads,
                                                       std::size_t rot_dims) noexcept {
    using namespace traffic_detail;
    const std::uint64_t rot = u(tokens) * u(n_heads) * u(rot_dims);
    return {kF32 * rot + 4 * u(tokens), kF32 * rot};
}

/// causal_conv1d_silu: x, weight, conv_state read; out, conv_state and the written slots.
[[nodiscard]] constexpr OpTraffic traffic_causal_conv1d(std::size_t tokens, std::size_t channels,
                                                        std::size_t kernel, std::size_t n_slots = 0) noexcept {
    using namespace traffic_detail;
    const std::uint64_t st = u(kernel - 1) * u(channels);
    const std::uint64_t slots = u(std::min(tokens, n_slots)) * st;
    return {kF32 * (u(tokens) * u(channels) + u(channels) * u(kernel) + st),
            kF32 * (u(tokens) * u(channels) + st + slots)};
}

/// gated_delta_rule_{recurrent,chunked}: q, k, v, g, beta and the state read; out, the
/// state and the written rollback slots (min(T, n_slots) of them).
[[nodiscard]] constexpr OpTraffic traffic_gated_delta_rule(const GdnDims& d, std::size_t tokens,
                                                           std::size_t n_slots = 0) noexcept {
    using namespace traffic_detail;
    const std::uint64_t per_tok = 2 * u(d.n_k_heads) * u(d.d_k) + u(d.n_v_heads) * u(d.d_v) + 2 * u(d.n_v_heads);
    const std::uint64_t state = u(d.n_v_heads) * u(d.d_k) * u(d.d_v);
    const std::uint64_t slots = u(std::min(tokens, n_slots)) * state;
    return {kF32 * (u(tokens) * per_tok + state),
            kF32 * (u(tokens) * u(d.n_v_heads) * u(d.d_v) + state + slots)};
}

/// attention_gqa: q read, the attended K/V history rows [0, q_offset + T) read, out written.
[[nodiscard]] constexpr OpTraffic traffic_attention(const AttentionDims& d, std::size_t tokens,
                                                    std::size_t q_offset) noexcept {
    using namespace traffic_detail;
    const std::uint64_t q = u(tokens) * u(d.n_head) * u(d.head_dim);
    const std::uint64_t kv = 2 * (u(q_offset) + u(tokens)) * u(d.n_kv_head) * u(d.head_dim);
    return {kF32 * (q + kv), kF32 * q};
}

/// matmul y[T, N] = x[T, K] W^T with W stored in `weight_bytes`.
[[nodiscard]] constexpr OpTraffic traffic_matmul(std::size_t tokens, std::size_t k, std::size_t n,
                                                 std::uint64_t weight_bytes) noexcept {
    using namespace traffic_detail;
    return {kF32 * u(tokens) * u(k) + weight_bytes, kF32 * u(tokens) * u(n)};
}

/// matmul_argmax over one row: x + W read, one (index, value) written.
[[nodiscard]] constexpr OpTraffic traffic_matmul_argmax(std::size_t k, std::uint64_t weight_bytes) noexcept {
    using namespace traffic_detail;
    return {kF32 * u(k) + weight_bytes, 8};
}

/// argmax (k = 1) / top_k over n logits: logits read, k (index, value) pairs written.
[[nodiscard]] constexpr OpTraffic traffic_top_k(std::size_t n, std::size_t k) noexcept {
    using namespace traffic_detail;
    return {kF32 * u(n), 8 * u(k)};
}

}  // namespace halo::cpu
