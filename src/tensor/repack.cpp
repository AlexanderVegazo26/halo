#include "halo/tensor/repack.h"

#include <cstring>

#include "halo/core/error.h"

// Layout: see the header comment. Every output word is assembled byte by byte (little
// endian), so the result does not depend on the host byte order.

namespace halo::tensor {
namespace {

constexpr std::size_t kQK = 256;

struct Geometry {
    std::size_t block_bytes;
};

Geometry geometry(DType t) {
    switch (t) {
        case DType::Q5_K: return {176};
        case DType::Q6_K: return {210};
        case DType::IQ4_XS: return {136};
        default:
            throw_error(ErrorCode::Unsupported, "gemv repack: weight type id {} has no repacked layout (Q5_K, Q6_K, IQ4_XS)",
                        static_cast<std::uint32_t>(t));
    }
}

std::size_t round16(std::size_t n) { return (n + 15) & ~std::size_t{15}; }

const std::byte* at(const std::byte* p, std::size_t i) { return p + i; }

// Writes the 4 source bytes s[0..3] to d[0..3] (one little-endian word, byte-wise).
void put4(std::byte* d, const std::byte* s0, const std::byte* s1, const std::byte* s2, const std::byte* s3) {
    d[0] = *s0;
    d[1] = *s1;
    d[2] = *s2;
    d[3] = *s3;
}

void repack_q5_k(const std::byte* src, std::size_t nb, std::byte* dst) {
    std::byte* const qs_p = dst;
    std::byte* const qh_p = dst + 128 * nb;
    std::byte* const hdr_p = dst + 160 * nb;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = src + b * 176;
        const std::byte* qh = blk + 16;
        const std::byte* qs = blk + 48;
        std::memcpy(hdr_p + b * 16, blk, 16);
        for (std::size_t t = 0; t < 16; ++t) {
            const std::size_t ir = t & 3, v_in = (t >> 2) & 1, v_im = t >> 3;
            const std::size_t q_off = 32 * v_im + 4 * ir + 2 * v_in;
            std::byte* o = qs_p + b * 128 + t * 8;
            put4(o, at(qs, q_off), at(qs, q_off + 1), at(qs, q_off + 16), at(qs, q_off + 17));
            put4(o + 4, at(qs, q_off + 64), at(qs, q_off + 65), at(qs, q_off + 80), at(qs, q_off + 81));
        }
        for (std::size_t j = 0; j < 8; ++j) {
            const std::size_t l0 = 2 * j;
            put4(qh_p + b * 32 + j * 4, at(qh, l0), at(qh, l0 + 1), at(qh, l0 + 16), at(qh, l0 + 17));
        }
    }
}

void repack_q6_k(const std::byte* src, std::size_t nb, std::byte* dst) {
    std::byte* const ql_p = dst;
    std::byte* const qh_p = dst + 128 * nb;
    std::byte* const sc_p = dst + 192 * nb;
    std::byte* const d_p = dst + 208 * nb;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = src + b * 210;
        const std::byte* ql = blk;
        for (std::size_t t = 0; t < 16; ++t) {
            const std::size_t v_im = t >> 3, v_in = t & 7;
            const std::size_t o0 = 64 * v_im + 4 * v_in;
            std::byte* o = ql_p + b * 128 + t * 8;
            std::memcpy(o, ql + o0, 4);
            std::memcpy(o + 4, ql + o0 + 32, 4);
        }
        std::memcpy(qh_p + b * 64, blk + 128, 64);
        std::memcpy(sc_p + b * 16, blk + 192, 16);
        std::memcpy(d_p + b * 2, blk + 208, 2);
    }
}

void repack_iq4_xs(const std::byte* src, std::size_t nb, std::byte* dst) {
    std::byte* const qs_p = dst;
    std::byte* const hdr_p = dst + 128 * nb;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = src + b * 136;
        std::memcpy(hdr_p + b * 8, blk, 8);
        std::memcpy(qs_p + b * 128, blk + 8, 128);
    }
}

}  // namespace

bool gemv_repack_supported(DType t) noexcept { return t == DType::Q5_K || t == DType::Q6_K || t == DType::IQ4_XS; }

std::size_t gemv_repack_row_stride(DType t, std::size_t cols) {
    const Geometry g = geometry(t);
    HALO_CHECK(cols > 0 && cols % kQK == 0, ErrorCode::Model, "gemv repack: cols={} is not a multiple of {}", cols, kQK);
    return round16(cols / kQK * g.block_bytes);
}

void gemv_repack(DType t, std::span<const std::byte> src, std::size_t rows, std::size_t cols, std::span<std::byte> dst) {
    const std::size_t stride = gemv_repack_row_stride(t, cols);  // validates t and cols
    const std::size_t nb = cols / kQK;
    const std::size_t row_bytes = nb * geometry(t).block_bytes;
    HALO_CHECK(src.size() == rows * row_bytes, ErrorCode::Model, "gemv repack: source is {} bytes, expected {} ({} rows x {})",
               src.size(), rows * row_bytes, rows, row_bytes);
    HALO_CHECK(dst.size() == rows * stride, ErrorCode::Model, "gemv repack: destination is {} bytes, expected {} ({} rows x {})",
               dst.size(), rows * stride, rows, stride);
    std::memset(dst.data(), 0, dst.size());  // row padding is deterministic
    for (std::size_t r = 0; r < rows; ++r) {
        const std::byte* s = src.data() + r * row_bytes;
        std::byte* d = dst.data() + r * stride;
        switch (t) {
            case DType::Q5_K: repack_q5_k(s, nb, d); break;
            case DType::Q6_K: repack_q6_k(s, nb, d); break;
            default: repack_iq4_xs(s, nb, d); break;  // geometry() rejected everything else
        }
    }
}

}  // namespace halo::tensor
