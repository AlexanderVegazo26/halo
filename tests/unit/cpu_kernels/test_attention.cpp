// attention_gqa vs naive fp64 O(T*S) attention, paged block tables, causal masking.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/core/error.h"
#include "reference.h"
#include "tolerance.h"

// This file compares against fp64 reference math and promotes float operands to double on
// purpose. -Wdouble-promotion exists to catch *accidental* promotion in float code; it stays
// enabled for the CPU kernels themselves (backends/cpu) and is disabled for this file only.
#pragma GCC diagnostic ignored "-Wdouble-promotion"

namespace {

using namespace halo::cpu;
using namespace halo::cpu::test;

struct Case {
    AttentionDims dims;
    std::size_t T, q_offset;
};

/// Naive fp64 causal GQA attention (HF eager_attention_forward semantics).
std::vector<double> attn_ref(const Case& c, const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, float scale, std::vector<double>* bound) {
    const auto [nh, nkv, hd] = c.dims;
    const std::size_t group = nh / nkv;
    std::vector<double> out(c.T * nh * hd, 0.0);
    bound->assign(out.size(), 0.0);
    for (std::size_t t = 0; t < c.T; ++t) {
        for (std::size_t h = 0; h < nh; ++h) {
            const std::size_t kvh = h / group, n = c.q_offset + t + 1;
            std::vector<double> p(n);
            double m = -1e300, qn = 0, a = 0, vmax = 0;
            for (std::size_t i = 0; i < hd; ++i) qn += double(q[t * nh * hd + h * hd + i]) * q[t * nh * hd + h * hd + i];
            for (std::size_t s = 0; s < n; ++s) {
                double dot = 0, kn = 0;
                for (std::size_t i = 0; i < hd; ++i) {
                    dot += double(q[t * nh * hd + h * hd + i]) * k[s * nkv * hd + kvh * hd + i];
                    kn += double(k[s * nkv * hd + kvh * hd + i]) * k[s * nkv * hd + kvh * hd + i];
                }
                p[s] = dot * scale;
                m = std::max(m, p[s]);
                a = std::max(a, std::sqrt(qn * kn) * scale);  // Cauchy-Schwarz bound on |score|
                for (std::size_t i = 0; i < hd; ++i) vmax = std::max(vmax, double(std::abs(v[s * nkv * hd + kvh * hd + i])));
            }
            double sum = 0;
            for (auto& e : p) sum += (e = std::exp(e - m));
            for (std::size_t s = 0; s < n; ++s)
                for (std::size_t i = 0; i < hd; ++i) out[t * nh * hd + h * hd + i] += p[s] / sum * v[s * nkv * hd + kvh * hd + i];
            // Tolerance (Model R): score rounding sqrt(hd) eps * |q||k| scale shifts each
            // softmax weight by that relative amount; the weighted sum over n keys adds
            // sqrt(n) eps; outputs are bounded by max|v|. Plus exp/divide (Model E).
            const double tol = 8.0 * kEps * (std::sqrt(double(hd)) * a + std::sqrt(double(n)) + 4.0) * vmax + kDenormFloor;
            for (std::size_t i = 0; i < hd; ++i) (*bound)[t * nh * hd + h * hd + i] = tol;
        }
    }
    return out;
}

TEST(CpuAttention, MatchesNaiveFp64) {
    Rng rng(20);
    double worst = 0;
    for (const Case& c : {Case{{6, 2, 64}, 1, 0}, Case{{6, 2, 64}, 5, 35}, Case{{6, 6, 32}, 7, 0},
                          Case{{24, 4, 256}, 3, 14}}) {
        const auto [nh, nkv, hd] = c.dims;
        const std::size_t S = c.q_offset + c.T;
        auto q = rng.normal(c.T * nh * hd), k = rng.normal(S * nkv * hd), v = rng.normal(S * nkv * hd);
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        std::vector<float> out(c.T * nh * hd);
        attention_gqa(c.dims, ConstRows(std::span<const float>(q), c.T, nh * hd),
                      PagedRows::contiguous(k.data(), S, nkv * hd, nkv * hd),
                      PagedRows::contiguous(v.data(), S, nkv * hd, nkv * hd), c.q_offset, scale,
                      Rows(std::span<float>(out), c.T, nh * hd));
        std::vector<double> bound;
        const auto ref = attn_ref(c, q, k, v, scale, &bound);
        for (std::size_t i = 0; i < out.size(); ++i) {
            worst = std::max(worst, std::abs(out[i] - ref[i]) / bound[i]);
            ASSERT_LE(std::abs(out[i] - ref[i]), bound[i]) << "nh=" << nh << " T=" << c.T << " i=" << i;
        }
    }
    std::printf("[tol] attention worst err/tol = %.3g\n", worst);
}

TEST(CpuAttention, PagedShuffledBlocksEqualContiguousBitwise) {
    Rng rng(21);
    const AttentionDims dims{24, 4, 64};
    const std::size_t T = 4, off = 33, S = off + T, cols = 4 * 64, block_rows = 5, pad = 8;
    auto q = rng.normal(T * 24 * 64), k = rng.normal(S * cols), v = rng.normal(S * cols);
    const float scale = 0.125f;
    std::vector<float> ref(T * 24 * 64), out(ref.size());
    attention_gqa(dims, ConstRows(std::span<const float>(q), T, 24 * 64), PagedRows::contiguous(k.data(), S, cols, cols),
                  PagedRows::contiguous(v.data(), S, cols, cols), off, scale, Rows(std::span<float>(ref), T, 24 * 64));
    // Block pool with shuffled physical order and padded rows (row_stride > cols).
    const std::size_t n_blocks = (S + block_rows - 1) / block_rows;
    std::vector<std::size_t> phys(n_blocks);
    std::iota(phys.begin(), phys.end(), 0);
    std::shuffle(phys.begin(), phys.end(), std::mt19937(3));
    const std::size_t stride = cols + pad;
    std::vector<float> kpool(n_blocks * block_rows * stride, -7.0f), vpool(kpool.size(), -7.0f);
    std::vector<const float*> kt(n_blocks), vt(n_blocks);
    for (std::size_t b = 0; b < n_blocks; ++b) {
        kt[b] = kpool.data() + phys[b] * block_rows * stride;
        vt[b] = vpool.data() + phys[b] * block_rows * stride;
        for (std::size_t r = 0; r < block_rows && b * block_rows + r < S; ++r) {
            std::copy_n(&k[(b * block_rows + r) * cols], cols, kpool.data() + phys[b] * block_rows * stride + r * stride);
            std::copy_n(&v[(b * block_rows + r) * cols], cols, vpool.data() + phys[b] * block_rows * stride + r * stride);
        }
    }
    attention_gqa(dims, ConstRows(std::span<const float>(q), T, 24 * 64), PagedRows::paged(kt, block_rows, S, cols, stride),
                  PagedRows::paged(vt, block_rows, S, cols, stride), off, scale, Rows(std::span<float>(out), T, 24 * 64));
    EXPECT_TRUE(bitwise_equal(out, ref));
    for (std::size_t threads : {1u, 2u, 3u, 7u, 16u, 200u}) {
        ThreadPool pool(threads);
        std::vector<float> o2(out.size());
        attention_gqa(dims, ConstRows(std::span<const float>(q), T, 24 * 64),
                      PagedRows::paged(kt, block_rows, S, cols, stride), PagedRows::paged(vt, block_rows, S, cols, stride),
                      off, scale, Rows(std::span<float>(o2), T, 24 * 64), &pool);
        EXPECT_TRUE(bitwise_equal(o2, ref)) << "threads=" << threads;
    }
    std::vector<const float*> short_table(n_blocks - 1);
    EXPECT_THROW((void)PagedRows::paged(short_table, block_rows, S, cols, stride), halo::Error);
}

TEST(CpuAttention, CausalMaskIgnoresFutureKeysAndOffsetBounds) {
    Rng rng(22);
    const AttentionDims dims{4, 2, 32};
    const std::size_t T = 6, off = 10, S = off + T + 5;  // 5 rows beyond the last query
    auto q = rng.normal(T * 128), k = rng.normal(S * 64), v = rng.normal(S * 64);
    std::vector<float> a(T * 128), b(T * 128);
    auto go = [&](std::vector<float>& o) {
        attention_gqa(dims, ConstRows(std::span<const float>(q), T, 128), PagedRows::contiguous(k.data(), S, 64, 64),
                      PagedRows::contiguous(v.data(), S, 64, 64), off, 0.2f, Rows(std::span<float>(o), T, 128));
    };
    go(a);
    // Query t may see keys <= off + t only: perturb key/value off+t+1 and check query t unchanged
    // while query t+1 changes.
    for (std::size_t t = 0; t + 1 < T; ++t) {
        auto k0 = k, v0 = v;
        for (std::size_t i = 0; i < 64; ++i) {
            k[(off + t + 1) * 64 + i] += 3.0f;
            v[(off + t + 1) * 64 + i] += 3.0f;
        }
        go(b);
        for (std::size_t r = 0; r <= t; ++r)
            ASSERT_TRUE(bitwise_equal(std::span(a).subspan(r * 128, 128), std::span(b).subspan(r * 128, 128)))
                << "query " << r << " saw key " << off + t + 1;
        EXPECT_FALSE(bitwise_equal(std::span(a).subspan((t + 1) * 128, 128), std::span(b).subspan((t + 1) * 128, 128)))
            << "query " << t + 1 << " did not see key " << off + t + 1;
        k = k0;
        v = v0;
    }
    EXPECT_THROW(attention_gqa(dims, ConstRows(std::span<const float>(q), T, 128), PagedRows::contiguous(k.data(), S, 64, 64),
                               PagedRows::contiguous(v.data(), S, 64, 64), S - T + 1, 0.2f, Rows(std::span<float>(a), T, 128)),
                 halo::Error);
    EXPECT_THROW(attention_gqa({4, 3, 32}, ConstRows(std::span<const float>(q), T, 128),
                               PagedRows::contiguous(k.data(), S, 64, 64), PagedRows::contiguous(v.data(), S, 64, 64), off,
                               0.2f, Rows(std::span<float>(a), T, 128)),
                 halo::Error);
}

}  // namespace
