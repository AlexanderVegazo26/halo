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
#include "head.h"
// The IQ lookup tables are halo::tensor's own (src/tensor/ggml_tables.h, verbatim from ggml):
// one source for the CPU reference and the kernels, no private copy (review M-2).
#include "tensor/ggml_tables.h"

namespace halo::hip::kern {

enum class WType : unsigned {
    F32 = 0,
    F16 = 1,
    Q8_0 = 2,
    Q4_K = 3,
    Q5_K = 4,
    Q6_K = 5,
    IQ4_NL = 6,  // D-014 second tier (UD-Q4_K_XL)
    IQ4_XS = 7,
    Q3_K = 8,
    IQ3_S = 9,
};

/// Elements and bytes per block of each weight type (ggml geometry).
HALO_HD inline unsigned wq_block_elems(WType t) {
    switch (t) {
        case WType::F32:
        case WType::F16: return 1;
        case WType::Q8_0:
        case WType::IQ4_NL: return 32;
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
        case WType::IQ4_NL: return 18;
        case WType::IQ4_XS: return 136;
        case WType::Q3_K: return 110;
        case WType::IQ3_S: return 110;
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

/// Little-endian u32 at any byte address (quant.cpp memcpy on a little-endian host).
HALO_HD inline std::uint32_t ld32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

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
        case WType::IQ4_NL: {  // block_iq4_nl { f16 d; u8 qs[16]; } — quant.cpp deq_iq4_nl
            const std::uint8_t* blk = row + 18u * (i / 32u);
            const unsigned r = i % 32u;
            const float d = h2f(ld16(blk));
            const std::uint8_t qb = blk[2u + r % 16u];
            const unsigned nib = r < 16u ? (qb & 0xFu) : (qb >> 4);
            return d * fli(tensor::detail::kvalues_iq4nl[nib]);
        }
        case WType::IQ4_XS: {  // block_iq4_xs { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
            const std::uint8_t* blk = row + 136u * (i / 256u);
            const unsigned r = i % 256u;
            const unsigned ib = r / 32u;
            const unsigned w = r % 32u;
            const float d = h2f(ld16(blk));
            const unsigned scales_h = ld16(blk + 2);
            const int ls = static_cast<int>(((blk[4u + ib / 2u] >> (4u * (ib % 2u))) & 0xFu) |
                                            (((scales_h >> (2u * ib)) & 3u) << 4));
            const float dl = d * fli(ls - 32);
            const std::uint8_t qb = blk[8u + 16u * ib + w % 16u];
            const unsigned nib = w < 16u ? (qb & 0xFu) : (qb >> 4);
            return dl * fli(tensor::detail::kvalues_iq4nl[nib]);
        }
        case WType::Q3_K: {  // block_q3_K { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; }
            const std::uint8_t* blk = row + 110u * (i / 256u);
            const unsigned r = i % 256u;
            const unsigned nh = r / 128u;  // 128-element half
            const unsigned rr = r % 128u;
            const unsigned j = rr / 32u;   // shift group
            const unsigned w = rr % 32u;
            const unsigned l = w % 16u;
            const unsigned hi = w / 16u;
            const float d_all = h2f(ld16(blk + 108));
            // The 16 6-bit scales, unpacked exactly as quant.cpp (kmask1/kmask2 over 3 words).
            constexpr std::uint32_t kmask1 = 0x03030303u;
            constexpr std::uint32_t kmask2 = 0x0f0f0f0fu;
            const std::uint32_t a0 = ld32(blk + 96);
            const std::uint32_t a1 = ld32(blk + 100);
            const std::uint32_t a2 = ld32(blk + 104);  // quant.cpp tmp
            const unsigned is = r / 16u;
            std::uint32_t word = 0;
            switch (is / 4u) {
                case 0: word = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4); break;
                case 1: word = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4); break;
                case 2: word = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4); break;
                default: word = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4); break;
            }
            const int sc = static_cast<std::int8_t>(static_cast<std::uint8_t>((word >> (8u * (is % 4u))) & 0xFFu));
            const float dl = d_all * fli(sc - 32);
            const unsigned qi = 32u * nh + l + 16u * hi;
            const unsigned m = 1u << (j + 4u * nh);
            const int q = static_cast<int>((blk[32u + qi] >> (2u * j)) & 3u) - ((blk[l + 16u * hi] & m) != 0u ? 0 : 4);
            return dl * fli(q);
        }
        case WType::IQ3_S: {  // block_iq3_s { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; }
            const std::uint8_t* blk = row + 110u * (i / 256u);
            const unsigned r = i % 256u;
            const unsigned sb = r / 32u;    // 32-element sub-block (ib32 + half)
            const unsigned half = sb % 2u;
            const unsigned within = r % 32u;
            const unsigned l = within / 8u;
            const unsigned k = within % 8u;
            const unsigned j = k % 4u;
            const unsigned second = k / 4u;
            const float d = h2f(ld16(blk));
            const std::uint8_t* qs = blk + 2u + 8u * sb;
            const unsigned h = blk[66u + sb];
            const std::uint8_t* signs = blk + 74u + 4u * sb;
            const std::uint8_t scb = blk[106u + sb / 2u];
            const float db = d * fli(static_cast<int>(1u + 2u * (half == 0u ? (scb & 0xFu) : (scb >> 4))));
            const unsigned g = second == 0u ? (qs[2u * l] | ((h << (8u - 2u * l)) & 256u))
                                            : (qs[2u * l + 1u] | ((h << (7u - 2u * l)) & 256u));
            const float grid = fli(static_cast<int>((tensor::detail::iq3s_grid[g] >> (8u * j)) & 0xFFu));
            const float sign = (signs[l] & tensor::detail::kmask_iq2xs[j + 4u * second]) != 0u ? -1.f : 1.f;
            return db * grid * sign;
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
    ArgPart* part = nullptr;     // LM head: per-workgroup argmax partials, [n_vec][n_parts]
    unsigned n_parts = 0;        // = grid.x
};

struct GemvNoRegs {
    float unused;
};

/// ARGMAX_FUSED epilogue: thread 0 ranks this workgroup's rows (rows n >= p.rows excluded)
/// into one partial; NaN rows only set the flag. Stage 2 is argmax_reduce_body.
template <class Exec>
HALO_HD void gemv_argmax_epilogue(Exec& ex, const float* rowv, const GemvParams& p, unsigned rpb, unsigned bx,
                                  unsigned by) {
    ex.phase([&](unsigned tid, GemvNoRegs&) {
        if (tid != 0) return;
        float bv = 0.0f;
        std::uint32_t bi = kNoIndex;
        std::uint32_t nan = 0;
        for (unsigned r = 0; r < rpb; ++r) {
            const unsigned n = bx * rpb + r;
            if (n >= p.rows) break;
            const float v = rowv[r];
            if (hisnan(v)) {
                nan = 1;
                continue;
            }
            if (ranks_before(v, n, bv, bi)) {
                bv = v;
                bi = n;
            }
        }
        ArgPart& o = p.part[static_cast<std::uint64_t>(by) * p.n_parts + bx];
        o.value = bv;
        o.index = bi;
        o.nan = nan;
        o.pad = 0;
    });
}

// ---- generic: 8 lanes per row (the CPU dot order) -------------------------------------

inline constexpr unsigned kGemvGenericLanes = 8;
inline constexpr unsigned kGemvMaxBlock = 256;

struct GemvGenericShared {
    float part[kGemvMaxBlock];
    float rowv[kGemvMaxBlock / 8];
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
        const float v = s + tail;
        sh.rowv[tid / kGemvGenericLanes] = v;
        if (p.y != nullptr) p.y[static_cast<std::uint64_t>(by) * p.y_stride + n] = v;
    });
    if (p.part != nullptr) gemv_argmax_epilogue(ex, sh.rowv, p, rpb, bx, by);
}

// ---- wave: 32 lanes per row, 8-element groups, fixed LDS tree --------------------------

inline constexpr unsigned kGemvWaveLanes = 32;
inline constexpr unsigned kGemvGroup = 8;

struct GemvWaveShared {
    float red[kGemvMaxBlock];
    float rowv[kGemvMaxBlock / 8];
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
        if (n >= p.rows) return;
        sh.rowv[tid / kGemvWaveLanes] = sh.red[tid];
        if (p.y != nullptr) p.y[static_cast<std::uint64_t>(by) * p.y_stride + n] = sh.red[tid];
    });
    if (p.part != nullptr) gemv_argmax_epilogue(ex, sh.rowv, p, rpb, bx, by);
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
