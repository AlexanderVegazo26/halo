#pragma once
// Test-data generation and fp64 error-bound scales for the Vulkan differential tests.
//
// No kernel semantics live here (code review M-2): the values a Vulkan result is compared
// against come from the HALO CPU backend (halo::cpu) and HALO's own dequantization
// (halo::tensor::dequantize_row) run on the same bytes and arrays. fp64 is used only to
// compute the magnitudes that the a-priori error bounds are scaled by.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#include "halo/tensor/dtype.h"
#include "halo/tensor/quant.h"

namespace halo::vulkan::ref {

inline constexpr double k_u = 0x1p-24;  // fp32 unit roundoff

[[nodiscard]] constexpr double dbl(float f) { return static_cast<double>(f); }

// ---------------------------------------------------------------- fp16 fields

// Random fp16 bit pattern that is a finite *normal* number (exponent 1..30), any sign.
// Subnormal scales are excluded on purpose: they are valid ggml data, but some Vulkan
// implementations flush fp16 denormals in unpackHalf2x16, which would test the driver's
// denormal mode rather than HALO's dequantization.
inline std::uint16_t random_normal_fp16(std::mt19937& rng, std::uint32_t min_exp = 1,
                                        std::uint32_t max_exp = 30) {
    std::uniform_int_distribution<std::uint32_t> e(min_exp, max_exp), m(0, 0x3FF), s(0, 1);
    return static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
}

inline void store_u16(std::uint8_t* p, std::uint16_t v) { std::memcpy(p, &v, 2); }

// ---------------------------------------------------------------- block layouts (ggml)

inline constexpr std::size_t k_q8_0_bytes = 34;   // fp16 d; int8 qs[32]
inline constexpr std::size_t k_q4_k_bytes = 144;  // fp16 d, dmin; u8 scales[12]; u8 qs[128]
inline constexpr std::size_t k_q6_k_bytes = 210;  // u8 ql[128]; u8 qh[64]; i8 scales[16]; fp16 d
inline constexpr std::size_t k_q5_k_bytes = 176;  // fp16 d, dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]
inline constexpr std::size_t k_iq4_xs_bytes = 136;  // fp16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]
inline constexpr std::size_t k_iq4_nl_bytes = 18;   // fp16 d; u8 qs[16]
inline constexpr std::size_t k_q3_k_bytes = 110;    // u8 hmask[32]; u8 qs[64]; u8 scales[12]; fp16 d
inline constexpr std::size_t k_iq3_s_bytes = 110;   // fp16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]

enum class WType { F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S };

[[nodiscard]] inline halo::DType dtype(WType t) {
    switch (t) {
        case WType::F32: return halo::DType::F32;
        case WType::Q8_0: return halo::DType::Q8_0;
        case WType::Q4_K: return halo::DType::Q4_K;
        case WType::Q5_K: return halo::DType::Q5_K;
        case WType::Q6_K: return halo::DType::Q6_K;
        case WType::IQ4_XS: return halo::DType::IQ4_XS;
        case WType::IQ4_NL: return halo::DType::IQ4_NL;
        case WType::Q3_K: return halo::DType::Q3_K;
        case WType::IQ3_S: return halo::DType::IQ3_S;
    }
    return halo::DType::F32;
}

inline std::size_t block_elems(WType t) {
    return (t == WType::Q8_0 || t == WType::IQ4_NL) ? 32 : (t == WType::F32 ? 1 : 256);
}
inline std::size_t block_bytes(WType t) {
    switch (t) {
        case WType::F32: return 4;
        case WType::Q8_0: return k_q8_0_bytes;
        case WType::Q4_K: return k_q4_k_bytes;
        case WType::Q5_K: return k_q5_k_bytes;
        case WType::Q6_K: return k_q6_k_bytes;
        case WType::IQ4_XS: return k_iq4_xs_bytes;
        case WType::IQ4_NL: return k_iq4_nl_bytes;
        case WType::Q3_K: return k_q3_k_bytes;
        case WType::IQ3_S: return k_iq3_s_bytes;
    }
    return 0;
}

/// halo::tensor::dequantize_row over the first `n` elements stored in `bytes`.
inline std::vector<float> dequantize(WType t, std::span<const std::uint8_t> bytes, std::size_t n) {
    std::vector<float> y(n);
    halo::tensor::dequantize_row(dtype(t), std::as_bytes(bytes).data(), y.data(), static_cast<std::int64_t>(n));
    return y;
}

// ---------------------------------------------------------------- random weights

struct Weights {
    std::vector<std::uint8_t> bytes;  // device layout (rows back to back), padded to a multiple of 4
    std::vector<float> deq;           // halo::tensor::dequantize_row of `bytes` [rows*cols]
    std::vector<float> mag;           // per-element error-bound magnitude (see random_weights)
};

// Random valid weights: random payload bytes, fp16 scale fields sanitized to finite normal.
// `max_exp` bounds the scale exponent (fp16 exponent field, bias 15). `mag` is |w|, except
// for Q4_K / Q5_K where it is |d*sc*q| + |dmin*m| (d*sc*q - dmin*m can cancel, and the rounding
// error is relative to the parts): both parts come from halo::tensor, by dequantizing
// copies of the blocks with dmin = 0 and with d = 0 respectively.
inline Weights random_weights(WType t, std::uint32_t rows, std::uint32_t cols, std::mt19937& rng,
                              std::uint32_t max_exp = 30) {
    Weights w;
    const std::size_t n = std::size_t{rows} * cols;
    if (t == WType::F32) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        w.deq.resize(n);
        w.mag.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            w.deq[i] = nd(rng);
            w.mag[i] = std::fabs(w.deq[i]);
        }
        w.bytes.resize(n * 4);
        std::memcpy(w.bytes.data(), w.deq.data(), n * 4);
        return w;
    }
    const std::size_t nb = n / block_elems(t);
    const std::size_t bb = block_bytes(t);
    const std::size_t raw = nb * bb;
    w.bytes.assign((raw + 3) / 4 * 4, 0);
    std::uniform_int_distribution<int> byte(0, 255);
    for (std::size_t b = 0; b < nb; ++b) {
        std::uint8_t* blk = w.bytes.data() + b * bb;
        for (std::size_t i = 0; i < bb; ++i) blk[i] = static_cast<std::uint8_t>(byte(rng));
        switch (t) {
            case WType::Q8_0: store_u16(blk, random_normal_fp16(rng, 1, max_exp)); break;
            case WType::Q4_K:
            case WType::Q5_K:
                store_u16(blk, random_normal_fp16(rng, 1, max_exp));
                store_u16(blk + 2, random_normal_fp16(rng, 1, max_exp));
                break;
            case WType::Q6_K: store_u16(blk + 208, random_normal_fp16(rng, 1, max_exp)); break;
            case WType::IQ4_XS: store_u16(blk, random_normal_fp16(rng, 1, max_exp)); break;
            case WType::IQ4_NL: store_u16(blk, random_normal_fp16(rng, 1, max_exp)); break;
            case WType::Q3_K: store_u16(blk + 108, random_normal_fp16(rng, 1, max_exp)); break;
            case WType::IQ3_S: store_u16(blk, random_normal_fp16(rng, 1, max_exp)); break;
            case WType::F32: break;
        }
    }
    const std::span<const std::uint8_t> payload(w.bytes.data(), raw);
    w.deq = dequantize(t, payload, n);
    w.mag.resize(n);
    if (t == WType::Q4_K || t == WType::Q5_K) {
        std::vector<std::uint8_t> scaled(payload.begin(), payload.end());
        std::vector<std::uint8_t> offset = scaled;
        for (std::size_t b = 0; b < nb; ++b) {
            store_u16(scaled.data() + b * bb + 2, 0);  // dmin = 0 -> d*sc*q
            store_u16(offset.data() + b * bb, 0);      // d = 0    -> -dmin*m
        }
        const auto a = dequantize(t, scaled, n);
        const auto m = dequantize(t, offset, n);
        for (std::size_t i = 0; i < n; ++i) w.mag[i] = std::fabs(a[i]) + std::fabs(m[i]);
    } else {
        for (std::size_t i = 0; i < n; ++i) w.mag[i] = std::fabs(w.deq[i]);
    }
    return w;
}

inline std::vector<float> random_vec(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

// ---------------------------------------------------------------- fp64 bound scales

// Error-bound scale of y = W x: scale[r] = sum_c mag[r,c] * |x[c]|.
inline std::vector<double> matvec_scale(std::span<const float> mag, std::span<const float> x, std::uint32_t rows,
                                        std::uint32_t cols) {
    std::vector<double> s(rows, 0.0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::uint32_t c = 0; c < cols; ++c) acc += dbl(mag[std::size_t{r} * cols + c]) * std::fabs(dbl(x[c]));
        s[r] = acc;
    }
    return s;
}

// |rms_norm(x) * w| in fp64: the magnitude the relative rms_norm bound is scaled by.
inline std::vector<double> rms_norm_scale(std::span<const float> x, std::span<const float> w, std::uint32_t rows,
                                          std::uint32_t cols, float eps) {
    std::vector<double> y(std::size_t{rows} * cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (std::uint32_t c = 0; c < cols; ++c) {
            const double v = dbl(x[std::size_t{r} * cols + c]);
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / cols + static_cast<double>(eps));
        for (std::uint32_t c = 0; c < cols; ++c) {
            const std::size_t i = std::size_t{r} * cols + c;
            y[i] = std::fabs(dbl(x[i]) * inv * dbl(w[c]));
        }
    }
    return y;
}

// State update of one recurrent Gated DeltaNet step (DECISIONS D-004 item 6) in fp64, in
// place on `state` [n_v, d_k, d_v]; value head j uses key head j % n_k (GGUF tiled order);
// k is used as given. Only the state magnitude max|S| (which the GDN bounds are scaled by)
// is taken from it; the output is not needed for that (|o| <= max|S| * ||q||_1).
inline void gdn_state_step(std::span<const float> k, std::span<const float> v, std::span<const float> g,
                           std::span<const float> beta, std::vector<double>& state, std::uint32_t n_v,
                           std::uint32_t n_k, std::uint32_t d_k, std::uint32_t d_v) {
    for (std::uint32_t j = 0; j < n_v; ++j) {
        const std::uint32_t kh = j % n_k;
        const double decay = std::exp(static_cast<double>(g[j]));
        double* S = state.data() + std::size_t{j} * d_k * d_v;
        const float* kk = k.data() + std::size_t{kh} * d_k;
        for (std::size_t i = 0; i < std::size_t{d_k} * d_v; ++i) S[i] *= decay;
        for (std::uint32_t b = 0; b < d_v; ++b) {
            double kv = 0.0;
            for (std::uint32_t a = 0; a < d_k; ++a) kv += S[std::size_t{a} * d_v + b] * dbl(kk[a]);
            const double delta = (static_cast<double>(v[std::size_t{j} * d_v + b]) - kv) * dbl(beta[j]);
            for (std::uint32_t a = 0; a < d_k; ++a) S[std::size_t{a} * d_v + b] += dbl(kk[a]) * delta;
        }
    }
}

}  // namespace halo::vulkan::ref
