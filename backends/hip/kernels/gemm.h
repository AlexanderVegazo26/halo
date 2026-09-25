#pragma once
// QUANT_GEMM (prefill): Y[t][n] = sum_i X[t][i] * W[n][i] for T >= 2 token rows, tiled so a
// dequantized W tile is reused by 16 token rows (the GEMV re-dequantizes W per token).
// See hd.h for the execution model.
//
// Tile: 16 tokens x 16 weight rows per 256-thread workgroup; thread (tm, tn) owns one output.
// K is walked in chunks of 64: a phase dequantizes the W tile (wq_elem, bit-identical to
// tensor::dequantize_row) and copies the X tile into LDS, the next phase accumulates. Each
// thread keeps the CPU dot's 8 interleaved partial sums: element i goes to lane i % 8 in
// increasing i (chunks start at multiples of 64, so the local and global lane agree), then
// the tail [K - K%8, K) is summed sequentially and combined exactly as cpu::detail::dot. So
// the result is BIT-IDENTICAL to dequantize_row + cpu::matmul (tested), not merely bounded.

#include "gemv.h"
#include "hd.h"

namespace halo::hip::kern {

inline constexpr unsigned kGemmTile = 16;   // tokens and weight rows per workgroup
inline constexpr unsigned kGemmChunk = 64;  // K elements per LDS stage (multiple of 8)
inline constexpr unsigned kGemmBlock = kGemmTile * kGemmTile;

struct GemmShared {
    float w[kGemmTile][kGemmChunk];
    float x[kGemmTile][kGemmChunk];
};

struct GemmRegs {
    float p[8];
};

inline Launch gemm_launch(unsigned rows, unsigned n_tok) {
    return Launch{(rows + kGemmTile - 1) / kGemmTile, (n_tok + kGemmTile - 1) / kGemmTile, kGemmBlock};
}

template <class Exec>
HALO_HD void gemm_body(Exec& ex, GemmShared& sh, const GemvParams& p, unsigned bx, unsigned by) {
    const unsigned n0 = bx * kGemmTile;
    const unsigned t0 = by * kGemmTile;
    const unsigned full = p.cols - p.cols % 8u;
    const unsigned bd = ex.block_dim();
    ex.phase([&](unsigned, GemmRegs& r) {
        for (unsigned j = 0; j < 8; ++j) r.p[j] = 0.0f;
    });
    for (unsigned k0 = 0; k0 < full; k0 += kGemmChunk) {
        const unsigned len = full - k0 < kGemmChunk ? full - k0 : kGemmChunk;
        ex.phase([&](unsigned tid, GemmRegs&) {
            for (unsigned e = tid; e < kGemmTile * kGemmChunk; e += bd) {
                const unsigned rr = e / kGemmChunk;
                const unsigned k = e % kGemmChunk;
                const unsigned n = n0 + rr;
                const unsigned t = t0 + rr;
                const bool in_k = k < len;
                sh.w[rr][k] = (in_k && n < p.rows)
                                  ? wq_elem(p.type, p.w + static_cast<std::uint64_t>(n) * p.w_stride, k0 + k)
                                  : 0.0f;
                sh.x[rr][k] = (in_k && t < p.n_vec) ? p.x[static_cast<std::uint64_t>(t) * p.x_stride + k0 + k] : 0.0f;
            }
        });
        ex.phase([&](unsigned tid, GemmRegs& r) {
            const unsigned tm = tid / kGemmTile;
            const unsigned tn = tid % kGemmTile;
            for (unsigned k = 0; k < len; ++k) r.p[k % 8u] = r.p[k % 8u] + sh.x[tm][k] * sh.w[tn][k];
        });
    }
    ex.phase([&](unsigned tid, GemmRegs& r) {
        const unsigned t = t0 + tid / kGemmTile;
        const unsigned n = n0 + tid % kGemmTile;
        if (t >= p.n_vec || n >= p.rows) return;
        const float* x = p.x + static_cast<std::uint64_t>(t) * p.x_stride;
        const std::uint8_t* row = p.w + static_cast<std::uint64_t>(n) * p.w_stride;
        float tail = 0.0f;
        for (unsigned i = full; i < p.cols; ++i) tail = tail + x[i] * wq_elem(p.type, row, i);
        const float s = ((r.p[0] + r.p[4]) + (r.p[1] + r.p[5])) + ((r.p[2] + r.p[6]) + (r.p[3] + r.p[7]));
        p.y[static_cast<std::uint64_t>(t) * p.y_stride + n] = s + tail;
    });
}

}  // namespace halo::hip::kern
