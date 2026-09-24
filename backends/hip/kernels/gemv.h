#pragma once
// QUANT_GEMV / MATMUL (decode) kernel bodies: y[t][n] = sum_i x[t][i] * W[n][i], W in ggml
// block layout (DECISIONS D-007, D-014 type priority: Q4_K, Q5_K, Q6_K, Q8_0; plus F16, F32).
// See hd.h for the execution model.
//
// Dequantization (wq_elem) reproduces src/tensor/quant.cpp (the scalar port of ggml's
// reference dequantizers) operation by operation, so every dequantized weight is
// bit-identical to halo::tensor::dequantize_row. Two reduction orders:
//
//   generic (gemv_generic_body): 8 lanes per row, lane l sums x[i]*w[i] for i = l, l+8, ...
//     in increasing i, then lane 0 combines ((p0+p4)+(p1+p5))+((p2+p6)+(p3+p7)) + tail —
//     exactly halo::cpu::detail::dot, i.e. bit-identical to dequantize_row + cpu::matmul.
//   wave (gemv_wave_body): 32 lanes per row; lane l owns groups of 8 contiguous elements
//     g = l, l+32, ... (one scale lookup per group, contiguous loads), sums them in
//     increasing i, then a fixed-order LDS tree (16, 8, 4, 2, 1). Deterministic, but not
//     the CPU's order: accepted within a derived summation-error bound (tests).

#include "hd.h"

namespace halo::hip::kern {

enum class WType : unsigned { F32 = 0, F16 = 1, Q8_0 = 2, Q4_K = 3, Q5_K = 4, Q6_K = 5 };

/// Elements and bytes per block of each weight type (ggml geometry).
HALO_HD inline unsigned wq_block_elems(WType t) {
    switch (t) {
        case WType::F32:
        case WType::F16: return 1;
        case WType::Q8_0: return 32;
        default: return 256;
    }
}
HALO_HD inline unsigned wq_block_bytes(WType t) {
    switch (t) {
        case WType::F32: return 4;
        case WType::F16: return 2;
        case WType::Q8_0: return 34;
        case WType::Q4_K: return 144;
        case WType::Q5_K: return 176;
        case WType::Q6_K: return 210;
    }
    return 0;
}

/// halo::tensor::fp16_to_fp32 (ggml's portable reference), bit for bit.
HALO_HD inline float h2f(std::uint16_t h) {
    const std::uint32_t w = static_cast<std::uint32_t>(h) << 16;
    const std::uint32_t sign = w & 0x80000000u;
    const std::uint32_t two_w = w + w;
    constexpr std::uint32_t exp_offset = 0xE0u << 23;
    constexpr float exp_scale = 0x1.0p-112f;
    const float normalized_value = __builtin_bit_cast(float, (two_w >> 4) + exp_offset) * exp_scale;
    constexpr std::uint32_t magic_mask = 126u << 23;
    constexpr float magic_bias = 0.5f;
    const float denormalized_value = __builtin_bit_cast(float, (two_w >> 17) | magic_mask) - magic_bias;
    constexpr std::uint32_t denormalized_cutoff = 1u << 27;
    const std::uint32_t result =
        sign | (two_w < denormalized_cutoff ? __builtin_bit_cast(std::uint32_t, denormalized_value)
                                            : __builtin_bit_cast(std::uint32_t, normalized_value));
    return __builtin_bit_cast(float, result);
}

/// Little-endian u16 at any byte address (quant blocks are not 2-byte aligned in general).
HALO_HD inline std::uint16_t ld16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

HALO_HD inline float fli(int v) { return static_cast<float>(v); }

/// ggml get_scale_min_k4 (src/tensor/quant.cpp scale_min_k4).
HALO_HD inline void scale_min_k4(unsigned j, const std::uint8_t* q, unsigned& d, unsigned& m) {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/// Element i of a weight row starting at `row` — the value tensor::dequantize_row writes.
HALO_HD inline float wq_elem(WType t, const std::uint8_t* row, unsigned i) {
    switch (t) {
        case WType::F32: {
            float v;
            __builtin_memcpy(&v, row + 4u * i, 4);
            return v;
        }
        case WType::F16:
            return h2f(ld16(row + 2u * i));
        case WType::Q8_0: {
            const std::uint8_t* blk = row + 34u * (i / 32u);
            const float d = h2f(ld16(blk));
            const int q = static_cast<std::int8_t>(blk[2u + i % 32u]);
            return fli(q) * d;
        }
        case WType::Q4_K: {
            const std::uint8_t* blk = row + 144u * (i / 256u);
            const unsigned r = i % 256u;
            const unsigned j64 = r / 64u;
            const unsigned within = r % 64u;
            const unsigned hi = within / 32u;
            const unsigned l = within % 32u;
            const float d = h2f(ld16(blk));
            const float min = h2f(ld16(blk + 2));
            unsigned sc = 0, m = 0;
            scale_min_k4(2u * j64 + hi, blk + 4, sc, m);
            const float d1 = d * fli(static_cast<int>(sc));
            const float m1 = min * fli(static_cast<int>(m));
            const std::uint8_t qb = blk[16u + 32u * j64 + l];
            const unsigned nib = hi != 0u ? (qb >> 4) : (qb & 0xFu);
            return d1 * fli(static_cast<int>(nib)) - m1;
        }
        case WType::Q5_K: {
            const std::uint8_t* blk = row + 176u * (i / 256u);
            const unsigned r = i % 256u;
            const unsigned j64 = r / 64u;
            const unsigned within = r % 64u;
            const unsigned hi = within / 32u;
            const unsigned l = within % 32u;
            const float d = h2f(ld16(blk));
            const float min = h2f(ld16(blk + 2));
            unsigned sc = 0, m = 0;
            scale_min_k4(2u * j64 + hi, blk + 4, sc, m);
            const float d1 = d * fli(static_cast<int>(sc));
            const float m1 = min * fli(static_cast<int>(m));
            const std::uint8_t qh = blk[16u + l];
            const std::uint8_t ql = blk[48u + 32u * j64 + l];
            const unsigned u = (hi != 0u ? 2u : 1u) << (2u * j64);
            const unsigned q = (hi != 0u ? (ql >> 4) : (ql & 0xFu)) + ((qh & u) != 0u ? 16u : 0u);
            return d1 * fli(static_cast<int>(q)) - m1;
        }
        case WType::Q6_K: {
            const std::uint8_t* blk = row + 210u * (i / 256u);
            const unsigned r = i % 256u;
            const unsigned n = r / 128u;
            const unsigned rr = r % 128u;
            const unsigned qd = rr / 32u;
            const unsigned l = rr % 32u;
            const std::uint8_t* ql = blk + 64u * n;
            const std::uint8_t* qh = blk + 128u + 32u * n;
            const std::uint8_t* sc = blk + 192u + 8u * n;
            const float d = h2f(ld16(blk + 208));
            const unsigned is = l / 16u;
            const unsigned lo = (qd & 1u) != 0u ? ql[l + 32u] : ql[l];
            const unsigned nib = qd >= 2u ? (lo >> 4) : (lo & 0xFu);
            const int q = static_cast<int>(nib | (((qh[l] >> (2u * qd)) & 3u) << 4)) - 32;
            const float s = fli(static_cast<std::int8_t>(sc[is + 2u * qd]));
            return d * s * fli(q);
        }
    }
    return 0.0f;
}

struct GemvParams {
    WType type = WType::F32;
    const std::uint8_t* w = nullptr;
    std::uint64_t w_stride = 0;  // bytes between rows
    const float* x = nullptr;
    std::uint64_t x_stride = 0;  // elements between vectors
    float* y = nullptr;
    std::uint64_t y_stride = 0;
    unsigned rows = 0;           // N
    unsigned cols = 0;           // K
    unsigned n_vec = 1;          // T
};

struct GemvNoRegs {
    float unused;
};

// ---- generic: 8 lanes per row (the CPU dot order) -------------------------------------

inline constexpr unsigned kGemvGenericLanes = 8;
inline constexpr unsigned kGemvMaxBlock = 256;

struct GemvGenericShared {
    float part[kGemvMaxBlock];
};

inline Launch gemv_generic_launch(unsigned rows, unsigned n_vec, unsigned block) {
    const unsigned rpb = block / kGemvGenericLanes;
    return Launch{(rows + rpb - 1) / rpb, n_vec, block};
}

template <class Exec>
HALO_HD void gemv_generic_body(Exec& ex, GemvGenericShared& sh, const GemvParams& p, unsigned bx, unsigned by) {
    const unsigned rpb = ex.block_dim() / kGemvGenericLanes;
    const float* x = p.x + static_cast<std::uint64_t>(by) * p.x_stride;
    const unsigned full = p.cols - p.cols % kGemvGenericLanes;
    ex.phase([&](unsigned tid, GemvNoRegs&) {
        const unsigned n = bx * rpb + tid / kGemvGenericLanes;
        const unsigned lane = tid % kGemvGenericLanes;
        float acc = 0.0f;
        if (n < p.rows) {
            const std::uint8_t* row = p.w + static_cast<std::uint64_t>(n) * p.w_stride;
            for (unsigned i = lane; i < full; i += kGemvGenericLanes) acc = acc + x[i] * wq_elem(p.type, row, i);
        }
        sh.part[tid] = acc;
    });
    ex.phase([&](unsigned tid, GemvNoRegs&) {
        if (tid % kGemvGenericLanes != 0u) return;
        const unsigned n = bx * rpb + tid / kGemvGenericLanes;
        if (n >= p.rows) return;
        const std::uint8_t* row = p.w + static_cast<std::uint64_t>(n) * p.w_stride;
        const float* q = sh.part + tid;
        float tail = 0.0f;
        for (unsigned i = full; i < p.cols; ++i) tail = tail + x[i] * wq_elem(p.type, row, i);
        const float s = ((q[0] + q[4]) + (q[1] + q[5])) + ((q[2] + q[6]) + (q[3] + q[7]));
        p.y[static_cast<std::uint64_t>(by) * p.y_stride + n] = s + tail;
    });
}

// ---- wave: 32 lanes per row, 8-element groups, fixed LDS tree --------------------------

inline constexpr unsigned kGemvWaveLanes = 32;
inline constexpr unsigned kGemvGroup = 8;

struct GemvWaveShared {
    float red[kGemvMaxBlock];
};

inline Launch gemv_wave_launch(unsigned rows, unsigned n_vec, unsigned block) {
    const unsigned rpb = block / kGemvWaveLanes;
    return Launch{(rows + rpb - 1) / rpb, n_vec, block};
}

template <class Exec>
HALO_HD void gemv_wave_body(Exec& ex, GemvWaveShared& sh, const GemvParams& p, unsigned bx, unsigned by) {
    const unsigned rpb = ex.block_dim() / kGemvWaveLanes;
    const float* x = p.x + static_cast<std::uint64_t>(by) * p.x_stride;
    ex.phase([&](unsigned tid, GemvNoRegs&) {
        const unsigned n = bx * rpb + tid / kGemvWaveLanes;
        const unsigned lane = tid % kGemvWaveLanes;
        float acc = 0.0f;
        if (n < p.rows) {
            const std::uint8_t* row = p.w + static_cast<std::uint64_t>(n) * p.w_stride;
            for (unsigned g0 = lane * kGemvGroup; g0 < p.cols; g0 += kGemvWaveLanes * kGemvGroup) {
                const unsigned g1 = g0 + kGemvGroup < p.cols ? g0 + kGemvGroup : p.cols;
                for (unsigned i = g0; i < g1; ++i) acc = acc + x[i] * wq_elem(p.type, row, i);
            }
        }
        sh.red[tid] = acc;
    });
    // Tree step s: lanes < s write red[lane] += red[lane + s]. Writers touch lanes [0, s) and
    // read only their own slot plus lanes [s, 2s), which nobody writes in this step.
    for (unsigned s = kGemvWaveLanes / 2; s > 0; s /= 2) {
        ex.phase([&](unsigned tid, GemvNoRegs&) {
            if (tid % kGemvWaveLanes < s) sh.red[tid] = sh.red[tid] + sh.red[tid + s];
        });
    }
    ex.phase([&](unsigned tid, GemvNoRegs&) {
        if (tid % kGemvWaveLanes != 0u) return;
        const unsigned n = bx * rpb + tid / kGemvWaveLanes;
        if (n < p.rows) p.y[static_cast<std::uint64_t>(by) * p.y_stride + n] = sh.red[tid];
    });
}

/// Depth of the wave variant's summation tree for K columns (for the error bound): the
/// longest chain of additions any product passes through.
HALO_HD inline unsigned gemv_wave_depth(unsigned cols) {
    const unsigned per_lane = (cols + kGemvWaveLanes - 1) / kGemvWaveLanes;  // <= this many terms
    return per_lane + 5u;                                                   // + log2(32) tree levels
}

/// Depth of halo::cpu::detail::dot (= generic variant): K/8 sequential terms per lane, 3
/// pairwise levels, the tail sum (up to 7 terms) and the final add.
HALO_HD inline unsigned gemv_cpu_depth(unsigned cols) { return cols / 8u + 3u + (cols % 8u) + 1u; }

}  // namespace halo::hip::kern
