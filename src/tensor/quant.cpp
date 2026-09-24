// Scalar reference dequantization. Ported from ggml/src/ggml-quants.c (dequantize_row_*)
// at llama.cpp commit bd4f514db. ggml is MIT licensed, Copyright (c) 2023-2026 The ggml
// authors; see src/tensor/ggml_tables.h for the full notice.
//
// Porting rules:
//  - Floating-point expressions keep ggml's evaluation order (e.g. `(d * sc) * q - m`), which
//    is also gguf-py's order, so results are bit-identical to `gguf.quants.dequantize`.
//  - FP contraction is disabled for this file (pragma below + -ffp-contract=off in CMake) as a
//    defensive measure. For the formats implemented today it is not strictly required: every
//    product that could be fused into a following +/- (e.g. `d1 * q` in `d1 * q - m1`) is
//    exact in float32 (fp16 scale: 11 significant bits x <= 6-bit sub-scale x <= 5-bit quant
//    fits in 24 bits), so an FMA rounds identically. Measured 2026-09-24: -march=native with
//    contraction on is still bit-exact vs gguf-py. The flag protects future formats (and
//    changed evaluation orders) where the product is not exact.
//  - Block fields are read with memcpy (blocks are packed and arbitrarily aligned; e.g. the
//    Q6_K scale sits at byte 208 of a 210-byte block), never through a cast pointer.

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

#include "halo/tensor/quant.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "ggml_tables.h"
#include "halo/core/error.h"
#include "halo/tensor/fp16.h"

namespace halo::tensor {
namespace {

using detail::iq3s_grid;
using detail::kmask_iq2xs;
using detail::kvalues_iq4nl;

constexpr int kQK = 256;  // QK_K

inline float f16_at(const std::uint8_t* p) noexcept {
    std::uint16_t h = 0;
    std::memcpy(&h, p, sizeof h);
    return fp16_to_fp32(h);
}

inline std::uint16_t u16_at(const std::uint8_t* p) noexcept {
    std::uint16_t v = 0;
    std::memcpy(&v, p, sizeof v);
    return v;
}

inline std::uint32_t u32_at(const std::uint8_t* p) noexcept {
    std::uint32_t v = 0;
    std::memcpy(&v, p, sizeof v);
    return v;
}

inline float fl(int v) noexcept { return static_cast<float>(v); }

// ---- plain types ------------------------------------------------------------------------

void deq_f32(const std::uint8_t* x, float* y, std::size_t nb) {
    std::memcpy(y, x, nb * sizeof(float));
}

void deq_f16(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i) y[i] = f16_at(x + 2 * i);
}

void deq_bf16(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i) y[i] = bf16_to_fp32(u16_at(x + 2 * i));
}

// ---- 32-element blocks ------------------------------------------------------------------

// block_q4_0 { f16 d; u8 qs[16]; }  (18 B)
void deq_q4_0(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 18, y += 32) {
        const float d = f16_at(x);
        const std::uint8_t* qs = x + 2;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >> 4) - 8;
            y[j + 0] = fl(x0) * d;
            y[j + 16] = fl(x1) * d;
        }
    }
}

// block_q4_1 { f16 d; f16 m; u8 qs[16]; }  (20 B)
void deq_q4_1(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 20, y += 32) {
        const float d = f16_at(x);
        const float m = f16_at(x + 2);
        const std::uint8_t* qs = x + 4;
        for (int j = 0; j < 16; ++j) {
            const int x0 = qs[j] & 0x0F;
            const int x1 = qs[j] >> 4;
            y[j + 0] = fl(x0) * d + m;
            y[j + 16] = fl(x1) * d + m;
        }
    }
}

// block_q5_0 { f16 d; u8 qh[4]; u8 qs[16]; }  (22 B)
void deq_q5_0(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 22, y += 32) {
        const float d = f16_at(x);
        const std::uint32_t qh = u32_at(x + 2);
        const std::uint8_t* qs = x + 6;
        for (int j = 0; j < 16; ++j) {
            const auto xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
            const auto xh_1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
            const int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            const int x1 = ((qs[j] >> 4) | xh_1) - 16;
            y[j + 0] = fl(x0) * d;
            y[j + 16] = fl(x1) * d;
        }
    }
}

// block_q5_1 { f16 d; f16 m; u8 qh[4]; u8 qs[16]; }  (24 B)
void deq_q5_1(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 24, y += 32) {
        const float d = f16_at(x);
        const float m = f16_at(x + 2);
        const std::uint32_t qh = u32_at(x + 4);
        const std::uint8_t* qs = x + 8;
        for (int j = 0; j < 16; ++j) {
            const auto xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
            const auto xh_1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
            const int x0 = (qs[j] & 0x0F) | xh_0;
            const int x1 = (qs[j] >> 4) | xh_1;
            y[j + 0] = fl(x0) * d + m;
            y[j + 16] = fl(x1) * d + m;
        }
    }
}

// block_q8_0 { f16 d; i8 qs[32]; }  (34 B)
void deq_q8_0(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 34, y += 32) {
        const float d = f16_at(x);
        for (int j = 0; j < 32; ++j) {
            const auto q = static_cast<std::int8_t>(x[2 + j]);
            y[j] = fl(q) * d;
        }
    }
}

// block_iq4_nl { f16 d; u8 qs[16]; }  (18 B)
void deq_iq4_nl(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 18, y += 32) {
        const float d = f16_at(x);
        const std::uint8_t* qs = x + 2;
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = d * fl(kvalues_iq4nl[qs[j] & 0xF]);
            y[j + 16] = d * fl(kvalues_iq4nl[qs[j] >> 4]);
        }
    }
}

// ---- 256-element super-blocks -----------------------------------------------------------

// block_q2_K { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin; }  (84 B)
void deq_q2_k(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 84) {
        const std::uint8_t* scales = x;
        const std::uint8_t* q = x + 16;
        const float d = f16_at(x + 80);
        const float min = f16_at(x + 82);
        int is = 0;
        for (int n = 0; n < kQK; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                std::uint8_t sc = scales[is++];
                float dl = d * fl(sc & 0xF);
                float ml = min * fl(sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * fl((q[l] >> shift) & 3) - ml;

                sc = scales[is++];
                dl = d * fl(sc & 0xF);
                ml = min * fl(sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * fl((q[l + 16] >> shift) & 3) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

// block_q3_K { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; }  (110 B)
void deq_q3_k(const std::uint8_t* x, float* y, std::size_t nb) {
    constexpr std::uint32_t kmask1 = 0x03030303;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    for (std::size_t i = 0; i < nb; ++i, x += 110) {
        const float d_all = f16_at(x + 108);
        const std::uint8_t* q = x + 32;
        const std::uint8_t* hm = x;
        std::uint8_t m = 1;

        std::array<std::uint32_t, 4> aux{};
        std::memcpy(aux.data(), x + 96, 12);
        const std::uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        std::array<std::int8_t, 16> scales{};
        std::memcpy(scales.data(), aux.data(), 16);

        int is = 0;
        for (int n = 0; n < kQK; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * fl(scales[static_cast<std::size_t>(is++)] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * fl(((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                dl = d_all * fl(scales[static_cast<std::size_t>(is++)] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * fl(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m = static_cast<std::uint8_t>(m << 1);
            }
            q += 32;
        }
    }
}

// ggml get_scale_min_k4
inline void scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) noexcept {
    if (j < 4) {
        d = static_cast<std::uint8_t>(q[j] & 63);
        m = static_cast<std::uint8_t>(q[j + 4] & 63);
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// block_q4_K { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; }  (144 B)
void deq_q4_k(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 144) {
        const float d = f16_at(x);
        const float min = f16_at(x + 2);
        const std::uint8_t* scales = x + 4;
        const std::uint8_t* q = x + 16;
        int is = 0;
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        for (int j = 0; j < kQK; j += 64) {
            scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * fl(sc);
            const float m1 = min * fl(m);
            scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * fl(sc);
            const float m2 = min * fl(m);
            for (int l = 0; l < 32; ++l) *y++ = d1 * fl(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * fl(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

// block_q5_K { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }  (176 B)
void deq_q5_k(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 176) {
        const float d = f16_at(x);
        const float min = f16_at(x + 2);
        const std::uint8_t* scales = x + 4;
        const std::uint8_t* qh = x + 16;
        const std::uint8_t* ql = x + 48;
        int is = 0;
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        std::uint8_t u1 = 1;
        std::uint8_t u2 = 2;
        for (int j = 0; j < kQK; j += 64) {
            scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * fl(sc);
            const float m1 = min * fl(m);
            scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * fl(sc);
            const float m2 = min * fl(m);
            for (int l = 0; l < 32; ++l) *y++ = d1 * fl((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * fl((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = static_cast<std::uint8_t>(u1 << 2);
            u2 = static_cast<std::uint8_t>(u2 << 2);
        }
    }
}

// block_q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }  (210 B)
void deq_q6_k(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 210) {
        const float d = f16_at(x + 208);
        const std::uint8_t* ql = x;
        const std::uint8_t* qh = x + 128;
        const std::uint8_t* sc = x + 192;
        auto s = [&](int k) { return fl(static_cast<std::int8_t>(sc[k])); };
        for (int n = 0; n < kQK; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * s(is + 0) * fl(q1);
                y[l + 32] = d * s(is + 2) * fl(q2);
                y[l + 64] = d * s(is + 4) * fl(q3);
                y[l + 96] = d * s(is + 6) * fl(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// block_iq4_xs { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }  (136 B)
void deq_iq4_xs(const std::uint8_t* x, float* y, std::size_t nb) {
    for (std::size_t i = 0; i < nb; ++i, x += 136) {
        const float d = f16_at(x);
        const std::uint16_t scales_h = u16_at(x + 2);
        const std::uint8_t* scales_l = x + 4;
        const std::uint8_t* qs = x + 8;
        for (int ib = 0; ib < kQK / 32; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * fl(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * fl(kvalues_iq4nl[qs[j] & 0xf]);
                y[j + 16] = dl * fl(kvalues_iq4nl[qs[j] >> 4]);
            }
            y += 32;
            qs += 16;
        }
    }
}

// block_iq3_s { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; }  (110 B)
void deq_iq3_s(const std::uint8_t* x, float* y, std::size_t nb) {
    auto grid_byte = [](unsigned idx, int j) {
        return fl(static_cast<int>((iq3s_grid[idx] >> (8 * j)) & 0xFFu));
    };
    auto sign = [](std::uint8_t s, int j) { return (s & kmask_iq2xs[static_cast<std::size_t>(j)]) ? -1.f : 1.f; };
    for (std::size_t i = 0; i < nb; ++i, x += 110) {
        const float d = f16_at(x);
        const std::uint8_t* qs = x + 2;
        const std::uint8_t* qh = x + 66;
        const std::uint8_t* signs = x + 74;
        const std::uint8_t* scales = x + 106;
        for (int ib32 = 0; ib32 < kQK / 32; ib32 += 2) {
            const float db1 = d * fl(1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2 = d * fl(1 + 2 * (scales[ib32 / 2] >> 4));
            for (int half = 0; half < 2; ++half) {
                const float db = half == 0 ? db1 : db2;
                const unsigned h = qh[half];
                for (int l = 0; l < 4; ++l) {
                    const unsigned g1 = qs[2 * l + 0] | ((h << (8 - 2 * l)) & 256u);
                    const unsigned g2 = qs[2 * l + 1] | ((h << (7 - 2 * l)) & 256u);
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * grid_byte(g1, j) * sign(signs[l], j + 0);
                        y[j + 4] = db * grid_byte(g2, j) * sign(signs[l], j + 4);
                    }
                    y += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
}

using BlockFn = void (*)(const std::uint8_t*, float*, std::size_t);

BlockFn block_fn(DType t) {
    switch (t) {
        case DType::F32: return deq_f32;
        case DType::F16: return deq_f16;
        case DType::BF16: return deq_bf16;
        case DType::Q4_0: return deq_q4_0;
        case DType::Q4_1: return deq_q4_1;
        case DType::Q5_0: return deq_q5_0;
        case DType::Q5_1: return deq_q5_1;
        case DType::Q8_0: return deq_q8_0;
        case DType::Q2_K: return deq_q2_k;
        case DType::Q3_K: return deq_q3_k;
        case DType::Q4_K: return deq_q4_k;
        case DType::Q5_K: return deq_q5_k;
        case DType::Q6_K: return deq_q6_k;
        case DType::IQ4_NL: return deq_iq4_nl;
        case DType::IQ4_XS: return deq_iq4_xs;
        case DType::IQ3_S: return deq_iq3_s;
        default: break;
    }
    throw_error(ErrorCode::Unsupported, "dequantization of {} is not implemented",
                ggml_type_name(static_cast<std::uint32_t>(t)));
}

// Validates n against the block size and returns the block count.
std::size_t checked_blocks(DType t, std::int64_t n) {
    const DTypeTraits& tr = traits(t);
    HALO_CHECK(n >= 0, ErrorCode::Model, "negative element count {} for {}", n, tr.name);
    const auto un = static_cast<std::uint64_t>(n);
    HALO_CHECK(un % tr.block_elems == 0, ErrorCode::Model,
               "{} elements is not a multiple of the {} block size {}", n, tr.name, tr.block_elems);
    return static_cast<std::size_t>(un / tr.block_elems);
}

}  // namespace

std::size_t row_bytes(DType t, std::int64_t n) {
    const DTypeTraits& tr = traits(t);
    const std::size_t nb = checked_blocks(t, n);
    constexpr auto kMax = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    HALO_CHECK(nb <= kMax / tr.block_bytes, ErrorCode::Model, "byte size of {} x {} overflows", n, tr.name);
    return nb * tr.block_bytes;
}

void dequantize_row(DType t, const std::byte* src, float* dst, std::int64_t n) {
    const BlockFn fn = block_fn(t);
    const std::size_t nb = checked_blocks(t, n);
    if (nb == 0) return;
    fn(reinterpret_cast<const std::uint8_t*>(src), dst, nb);
}

void dequantize_row(DType t, std::span<const std::byte> src, std::span<float> dst) {
    HALO_CHECK(dst.size() <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
               ErrorCode::Model, "row too long");
    const auto n = static_cast<std::int64_t>(dst.size());
    const std::size_t need = row_bytes(t, n);
    HALO_CHECK(src.size() == need, ErrorCode::Model, "{} row of {} elements needs {} bytes, got {}",
               traits(t).name, n, need, src.size());
    dequantize_row(t, src.data(), dst.data(), n);
}

float vec_dot_row(DType t, const std::byte* row, const float* x, std::int64_t n) {
    const BlockFn fn = block_fn(t);
    const DTypeTraits& tr = traits(t);
    const std::size_t nb = checked_blocks(t, n);
    // Dequantize up to 256 elements at a time (whole blocks), accumulate in double.
    const std::size_t blocks_per_chunk = 256 / tr.block_elems;
    std::array<float, 256> buf{};
    const auto* src = reinterpret_cast<const std::uint8_t*>(row);
    double acc = 0.0;
    for (std::size_t b = 0; b < nb; b += blocks_per_chunk) {
        const std::size_t cnt = std::min(blocks_per_chunk, nb - b);
        fn(src + b * tr.block_bytes, buf.data(), cnt);
        const float* xv = x + b * tr.block_elems;
        const std::size_t elems = cnt * tr.block_elems;
        for (std::size_t i = 0; i < elems; ++i) {
            acc += static_cast<double>(buf[i]) * static_cast<double>(xv[i]);
        }
    }
    return static_cast<float>(acc);
}

float vec_dot_row(DType t, std::span<const std::byte> row, std::span<const float> x) {
    HALO_CHECK(x.size() <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
               ErrorCode::Model, "row too long");
    const auto n = static_cast<std::int64_t>(x.size());
    const std::size_t need = row_bytes(t, n);
    HALO_CHECK(row.size() == need, ErrorCode::Model, "{} row of {} elements needs {} bytes, got {}",
               traits(t).name, n, need, row.size());
    return vec_dot_row(t, row.data(), x.data(), n);
}

}  // namespace halo::tensor
