#pragma once
// CPU scalar references for the Vulkan differential tests.
//
// Dequantization is a faithful port of ggml-quants.c (llama.cpp commit bd4f514db, MIT
// License, Copyright (c) 2023-2024 The ggml authors): dequantize_row_q8_0,
// dequantize_row_q4_K + get_scale_min_k4, dequantize_row_q6_K. Block layouts are ggml's
// block_q8_0 / block_q4_K / block_q6_K from ggml-common.h. Fields are read with memcpy so
// the UBSan build stays clean on unaligned block starts.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace halo::vulkan::ref {

inline constexpr double k_u = 0x1p-24;  // fp32 unit roundoff

[[nodiscard]] constexpr double dbl(float f) { return static_cast<double>(f); }

// ---------------------------------------------------------------- fp16

// Exact IEEE binary16 -> binary32 (every fp16 value, incl. subnormals, is representable).
inline float fp16_to_fp32(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1u;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t man = h & 0x3FFu;
    float mag;
    if (exp == 0) {
        mag = std::ldexp(static_cast<float>(man), -24);
    } else if (exp == 31) {
        mag = man == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        mag = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
    }
    return sign != 0 ? -mag : mag;
}

// Random fp16 bit pattern that is a finite *normal* number (exponent 1..30), any sign.
// Subnormal scales are excluded on purpose: they are valid ggml data, but some Vulkan
// implementations flush fp16 denormals in unpackHalf2x16, which would test the driver's
// denormal mode rather than HALO's dequantization.
inline std::uint16_t random_normal_fp16(std::mt19937& rng, std::uint32_t min_exp = 1,
                                        std::uint32_t max_exp = 30) {
    std::uniform_int_distribution<std::uint32_t> e(min_exp, max_exp), m(0, 0x3FF), s(0, 1);
    return static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
}

inline std::uint16_t load_u16(const std::uint8_t* p) {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline void store_u16(std::uint8_t* p, std::uint16_t v) { std::memcpy(p, &v, 2); }

// ---------------------------------------------------------------- block layouts

inline constexpr std::size_t k_q8_0_bytes = 34;   // fp16 d; int8 qs[32]
inline constexpr std::size_t k_q4_k_bytes = 144;  // fp16 d, dmin; u8 scales[12]; u8 qs[128]
inline constexpr std::size_t k_q6_k_bytes = 210;  // u8 ql[128]; u8 qh[64]; i8 scales[16]; fp16 d

// ggml get_scale_min_k4
inline void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// ggml dequantize_row_q8_0 for one block.
inline void dequant_q8_0(const std::uint8_t* blk, float* y, float* mag = nullptr) {
    const float d = fp16_to_fp32(load_u16(blk));
    for (int j = 0; j < 32; ++j) {
        const auto q = static_cast<std::int8_t>(blk[2 + j]);
        y[j] = static_cast<float>(q) * d;
        if (mag) mag[j] = std::fabs(y[j]);
    }
}

// ggml dequantize_row_q4_K for one block. `mag` (optional) receives |d1*q| + |m1|, the
// magnitude the fp32 dequant rounding error is relative to (d1*q - m1 can cancel).
inline void dequant_q4_k(const std::uint8_t* blk, float* y, float* mag = nullptr) {
    const float d = fp16_to_fp32(load_u16(blk));
    const float min = fp16_to_fp32(load_u16(blk + 2));
    const std::uint8_t* scales = blk + 4;
    const std::uint8_t* q = blk + 16;
    int is = 0;
    std::uint8_t sc = 0, m = 0;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            const auto qv = static_cast<float>(q[l] & 0xF);
            y[j + l] = d1 * qv - m1;
            if (mag) mag[j + l] = std::fabs(d1 * qv) + std::fabs(m1);
        }
        for (int l = 0; l < 32; ++l) {
            const auto qv = static_cast<float>(q[l] >> 4);
            y[j + 32 + l] = d2 * qv - m2;
            if (mag) mag[j + 32 + l] = std::fabs(d2 * qv) + std::fabs(m2);
        }
        q += 32;
        is += 2;
    }
}

// ggml dequantize_row_q6_K for one block.
inline void dequant_q6_k(const std::uint8_t* blk, float* y, float* mag = nullptr) {
    const float d = fp16_to_fp32(load_u16(blk + 208));
    const std::uint8_t* ql = blk;
    const std::uint8_t* qh = blk + 128;
    const std::uint8_t* scb = blk + 192;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            const auto sc = [&](int i) { return static_cast<float>(static_cast<std::int8_t>(scb[i])); };
            y[n + l + 0] = d * sc(is + 0) * static_cast<float>(q1);
            y[n + l + 32] = d * sc(is + 2) * static_cast<float>(q2);
            y[n + l + 64] = d * sc(is + 4) * static_cast<float>(q3);
            y[n + l + 96] = d * sc(is + 6) * static_cast<float>(q4);
        }
        ql += 64;
        qh += 32;
        scb += 8;
    }
    if (mag) {
        for (int i = 0; i < 256; ++i) mag[i] = std::fabs(y[i]);
    }
}

// ---------------------------------------------------------------- random weights

enum class WType { F32, Q8_0, Q4_K, Q6_K };

inline std::size_t block_elems(WType t) { return t == WType::Q8_0 ? 32 : (t == WType::F32 ? 1 : 256); }
inline std::size_t block_bytes(WType t) {
    switch (t) {
        case WType::F32: return 4;
        case WType::Q8_0: return k_q8_0_bytes;
        case WType::Q4_K: return k_q4_k_bytes;
        case WType::Q6_K: return k_q6_k_bytes;
    }
    return 0;
}

struct Weights {
    std::vector<std::uint8_t> bytes;  // device layout, padded to a multiple of 4
    std::vector<float> deq;           // ggml-dequantized values [rows*cols]
    std::vector<float> mag;           // per-element rounding-error magnitude
};

// Random valid weights: random payload bytes, fp16 scale fields sanitized to finite normal.
// `max_exp` bounds the scale exponent (fp16 exponent field, bias 15).
inline Weights random_weights(WType t, std::uint32_t rows, std::uint32_t cols, std::mt19937& rng,
                              std::uint32_t max_exp = 30) {
    Weights w;
    const std::size_t n = std::size_t{rows} * cols;
    w.deq.resize(n);
    w.mag.resize(n);
    if (t == WType::F32) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
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
    w.bytes.assign((nb * bb + 3) / 4 * 4, 0);
    std::uniform_int_distribution<int> byte(0, 255);
    for (std::size_t b = 0; b < nb; ++b) {
        std::uint8_t* blk = w.bytes.data() + b * bb;
        for (std::size_t i = 0; i < bb; ++i) blk[i] = static_cast<std::uint8_t>(byte(rng));
        float* y = w.deq.data() + b * block_elems(t);
        float* m = w.mag.data() + b * block_elems(t);
        switch (t) {
            case WType::Q8_0:
                store_u16(blk, random_normal_fp16(rng, 1, max_exp));
                dequant_q8_0(blk, y, m);
                break;
            case WType::Q4_K:
                store_u16(blk, random_normal_fp16(rng, 1, max_exp));
                store_u16(blk + 2, random_normal_fp16(rng, 1, max_exp));
                dequant_q4_k(blk, y, m);
                break;
            case WType::Q6_K:
                store_u16(blk + 208, random_normal_fp16(rng, 1, max_exp));
                dequant_q6_k(blk, y, m);
                break;
            case WType::F32: break;
        }
    }
    return w;
}

inline std::vector<float> random_vec(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

// ---------------------------------------------------------------- ops

// y = W x in double; `bound_scale[r]` = Σ_c mag[r,c]·|x[c]| (error-bound scale).
inline void matvec(std::span<const float> w, std::span<const float> mag, std::span<const float> x,
                   std::uint32_t rows, std::uint32_t cols, std::vector<double>& y,
                   std::vector<double>& bound_scale) {
    y.assign(rows, 0.0);
    bound_scale.assign(rows, 0.0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0, s = 0.0;
        for (std::uint32_t c = 0; c < cols; ++c) {
            const std::size_t i = std::size_t{r} * cols + c;
            acc += dbl(w[i]) * dbl(x[c]);
            s += static_cast<double>(mag[i]) * std::fabs(static_cast<double>(x[c]));
        }
        y[r] = acc;
        bound_scale[r] = s;
    }
}

inline std::vector<double> rms_norm(std::span<const float> x, std::span<const float> w, std::uint32_t rows,
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
            y[i] = dbl(x[i]) * inv * dbl(w[c]);
        }
    }
    return y;
}

// One recurrent Gated DeltaNet step (DECISIONS D-004 item 6), double precision, in place
// on `state` [n_v, d_k, d_v]; value head j uses key head j % n_k (GGUF tiled order).
inline void gdn_step(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                     std::span<const float> g, std::span<const float> beta, std::vector<double>& state,
                     std::vector<double>& out, std::uint32_t n_v, std::uint32_t n_k, std::uint32_t d_k,
                     std::uint32_t d_v) {
    out.assign(std::size_t{n_v} * d_v, 0.0);
    for (std::uint32_t j = 0; j < n_v; ++j) {
        const std::uint32_t kh = j % n_k;
        const double decay = std::exp(static_cast<double>(g[j]));
        double* S = state.data() + std::size_t{j} * d_k * d_v;
        const float* qh = q.data() + std::size_t{kh} * d_k;
        const float* kk = k.data() + std::size_t{kh} * d_k;
        for (std::size_t i = 0; i < std::size_t{d_k} * d_v; ++i) S[i] *= decay;
        for (std::uint32_t b = 0; b < d_v; ++b) {
            double kv = 0.0;
            for (std::uint32_t a = 0; a < d_k; ++a) kv += S[std::size_t{a} * d_v + b] * dbl(kk[a]);
            const double delta = (static_cast<double>(v[std::size_t{j} * d_v + b]) - kv) * dbl(beta[j]);
            for (std::uint32_t a = 0; a < d_k; ++a) S[std::size_t{a} * d_v + b] += dbl(kk[a]) * delta;
            double o = 0.0;
            for (std::uint32_t a = 0; a < d_k; ++a) o += S[std::size_t{a} * d_v + b] * dbl(qh[a]);
            out[std::size_t{j} * d_v + b] = o;
        }
    }
}

// Lowest index of the maximum, NaN ignored; 0xFFFFFFFF if every element is NaN.
inline std::uint32_t argmax(std::span<const float> x) {
    std::uint32_t best = 0xFFFFFFFFu;
    for (std::uint32_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) continue;
        if (best == 0xFFFFFFFFu || x[i] > x[best]) best = i;
    }
    return best;
}

}  // namespace halo::vulkan::ref
