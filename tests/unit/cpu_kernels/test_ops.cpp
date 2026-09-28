// Unit tests for the non-GDN CPU reference operators against independent fp64 math,
// plus edge cases, aliasing/shape validation and thread-count invariance.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
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

ConstRows cview(const std::vector<float>& v, std::size_t rows, std::size_t cols) {
    return {std::span<const float>(v), rows, cols};
}
Rows view(std::vector<float>& v, std::size_t rows, std::size_t cols) { return {std::span<float>(v), rows, cols}; }

// Tracks the worst observed |err| / tol across a test for reporting.
struct Worst {
    double ratio = 0.0;
    void add(double err, double tol) { ratio = std::max(ratio, err / tol); }
};

const std::vector<std::size_t> kThreadCounts{1, 2, 3, 7, 16, 64};

double silu64(double x) { return x / (1.0 + std::exp(-x)); }

// ------------------------------------------------------------------------------ rms_norm

TEST(CpuRmsNorm, MatchesFp64AndAliases) {
    Rng rng(1);
    for (std::size_t d : {1u, 7u, 256u, 5120u}) {
        const std::size_t rows = 5;
        auto x = rng.normal(rows * d, 3.0f);
        auto w = rng.normal(d, 1.0f);
        std::vector<float> y(rows * d);
        rms_norm(cview(x, rows, d), w, 1e-6f, view(y, rows, d));
        Worst worst;
        for (std::size_t r = 0; r < rows; ++r) {
            double ss = 0;
            for (std::size_t i = 0; i < d; ++i) ss += double(x[r * d + i]) * x[r * d + i];
            const double inv = 1.0 / std::sqrt(ss / double(d) + 1e-6);
            for (std::size_t i = 0; i < d; ++i) {
                const double ref = x[r * d + i] * inv * w[i];
                // Model R on the sum of squares (relative), then Model E for the products.
                const double tol = tol_reduction(d, std::abs(ref)) + tol_elementwise(ref);
                const double err = std::abs(y[r * d + i] - ref);
                worst.add(err, tol);
                ASSERT_LE(err, tol) << "d=" << d << " r=" << r << " i=" << i;
            }
        }
        std::vector<float> inplace = x;
        rms_norm(cview(inplace, rows, d), w, 1e-6f, view(inplace, rows, d));
        EXPECT_TRUE(bitwise_equal(inplace, y)) << "in-place differs, d=" << d;
        RecordProperty("worst_ratio_d" + std::to_string(d), std::to_string(worst.ratio));
        std::printf("[tol] rms_norm d=%zu worst err/tol = %.3g\n", d, worst.ratio);
    }
}

TEST(CpuRmsNorm, StridedRowsAndValidation) {
    Rng rng(2);
    const std::size_t rows = 4, d = 32, stride = 48;
    auto buf = rng.normal(rows * stride);
    auto w = rng.normal(d);
    std::vector<float> dense(rows * d), out_dense(rows * d), out_strided(rows * d);
    for (std::size_t r = 0; r < rows; ++r) std::copy_n(&buf[r * stride], d, &dense[r * d]);
    rms_norm(cview(dense, rows, d), w, 1e-6f, view(out_dense, rows, d));
    rms_norm(ConstRows(buf.data(), rows, d, stride), w, 1e-6f, view(out_strided, rows, d));
    EXPECT_TRUE(bitwise_equal(out_dense, out_strided));

    std::vector<float> w_bad(d + 1);
    EXPECT_THROW(rms_norm(cview(dense, rows, d), w_bad, 1e-6f, view(out_dense, rows, d)), halo::Error);
    // Partial overlap (shifted by one element) must be rejected.
    std::vector<float> big(rows * d + 1);
    EXPECT_THROW(rms_norm(ConstRows(big.data(), rows, d, d), w, 1e-6f, Rows(big.data() + 1, rows, d, d)),
                 halo::Error);
}

TEST(CpuGatedRmsNorm, MatchesFp64) {
    Rng rng(3);
    const std::size_t rows = 12, d = 128;
    auto x = rng.normal(rows * d, 2.0f), z = rng.normal(rows * d, 2.0f), w = rng.normal(d);
    std::vector<float> y(rows * d);
    gated_rms_norm(cview(x, rows, d), w, cview(z, rows, d), 1e-6f, view(y, rows, d));
    Worst worst;
    for (std::size_t r = 0; r < rows; ++r) {
        double ss = 0;
        for (std::size_t i = 0; i < d; ++i) ss += double(x[r * d + i]) * x[r * d + i];
        const double inv = 1.0 / std::sqrt(ss / double(d) + 1e-6);
        for (std::size_t i = 0; i < d; ++i) {
            const double ref = w[i] * (x[r * d + i] * inv) * silu64(z[r * d + i]);
            const double tol = tol_reduction(d, std::abs(ref)) + 2 * tol_elementwise(ref);
            const double err = std::abs(y[r * d + i] - ref);
            worst.add(err, tol);
            ASSERT_LE(err, tol);
        }
    }
    std::printf("[tol] gated_rms_norm worst err/tol = %.3g\n", worst.ratio);
    std::vector<float> zin = z;
    gated_rms_norm(cview(x, rows, d), w, cview(zin, rows, d), 1e-6f, view(zin, rows, d));
    EXPECT_TRUE(bitwise_equal(zin, y));
}

TEST(CpuL2Norm, MatchesFp64IncludingZeroHead) {
    Rng rng(4);
    const std::size_t t = 3, h = 4, d = 128;
    auto x = rng.normal(t * h * d, 2.0f);
    std::fill_n(x.begin() + d, d, 0.0f);  // token 0, head 1 all zero
    std::vector<float> y(x.size());
    l2_norm_heads(cview(x, t, h * d), h, d, view(y, t, h * d));
    Worst worst;
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t hh = 0; hh < h; ++hh) {
            double ss = 0;
            const std::size_t o = r * h * d + hh * d;
            for (std::size_t i = 0; i < d; ++i) ss += double(x[o + i]) * x[o + i];
            const double inv = 1.0 / std::sqrt(ss + 1e-6);
            for (std::size_t i = 0; i < d; ++i) {
                const double ref = x[o + i] * inv;
                const double tol = tol_reduction(d, std::abs(ref)) + tol_elementwise(ref);
                ASSERT_TRUE(std::isfinite(y[o + i]));
                const double err = std::abs(y[o + i] - ref);
                worst.add(err, tol);
                ASSERT_LE(err, tol);
            }
        }
    }
    std::printf("[tol] l2_norm_heads worst err/tol = %.3g\n", worst.ratio);
    EXPECT_THROW(l2_norm_heads(cview(x, t, h * d), h + 1, d, view(y, t, h * d)), halo::Error);
}

// ---------------------------------------------------------------------------------- RoPE

TEST(CpuRope, MatchesFp64OfContractAnglesAndPassesThroughTail) {
    Rng rng(5);
    const std::size_t t = 6, h = 3, hd = 256, rot = 64;
    const float theta = 1e7f;
    const std::vector<std::int32_t> pos{0, 1, 5, 1000, 65535, 262143};
    auto x = rng.normal(t * h * hd, 2.0f);
    auto y = x;
    partial_rope_neox(view(y, t, h * hd), h, hd, pos, rot, theta);
    const std::vector<float> inv = rope_inv_freq(rot, theta);
    Worst worst;
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t hh = 0; hh < h; ++hh) {
            const std::size_t o = r * h * hd + hh * hd;
            for (std::size_t i = 0; i < rot / 2; ++i) {
                const float angle = static_cast<float>(pos[r]) * inv[i];  // contract: fp32 angle
                const double c = std::cos(double(angle)), s = std::sin(double(angle));
                const double x1 = x[o + i], x2 = x[o + i + rot / 2];
                const double r1 = x1 * c - x2 * s, r2 = x2 * c + x1 * s;
                const double tol = tol_elementwise(std::abs(x1) + std::abs(x2));  // cancellation-aware scale
                worst.add(std::abs(y[o + i] - r1), tol);
                worst.add(std::abs(y[o + i + rot / 2] - r2), tol);
                ASSERT_LE(std::abs(y[o + i] - r1), tol);
                ASSERT_LE(std::abs(y[o + i + rot / 2] - r2), tol);
            }
            for (std::size_t i = rot; i < hd; ++i) ASSERT_EQ(y[o + i], x[o + i]) << "pass-through dim changed";
        }
    }
    std::printf("[tol] rope worst err/tol = %.3g\n", worst.ratio);
    // Position 0 is the identity.
    for (std::size_t i = 0; i < h * hd; ++i) ASSERT_EQ(y[i], x[i]);
    EXPECT_THROW(partial_rope_neox(view(y, t, h * hd), h, hd, pos, 63, theta), halo::Error);
    EXPECT_THROW(partial_rope_neox(view(y, t, h * hd), h, hd, std::span(pos).first(2), rot, theta), halo::Error);
}

// -------------------------------------------------------------------------------- conv1d

std::vector<double> conv_ref(const std::vector<float>& x, std::size_t T, std::size_t C, const std::vector<float>& w,
                             std::size_t K, const std::vector<float>& state0, std::vector<double>* pre_abs) {
    std::vector<double> out(T * C);
    if (pre_abs) pre_abs->assign(T * C, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t c = 0; c < C; ++c) {
            double acc = 0, aabs = 0;
            for (std::size_t j = 0; j < K; ++j) {
                const long idx = long(t) - long(K - 1) + long(j);  // input index; < 0 -> state
                const double xin = idx >= 0 ? x[std::size_t(idx) * C + c]
                                            : state0[std::size_t(long(K - 1) + idx) * C + c];
                acc += w[c * K + j] * xin;
                aabs += std::abs(w[c * K + j] * xin);
            }
            out[t * C + c] = silu64(acc);
            if (pre_abs) (*pre_abs)[t * C + c] = aabs;
        }
    }
    return out;
}

TEST(CpuConv1d, MatchesFp64WithState) {
    Rng rng(6);
    for (std::size_t K : {1u, 2u, 4u}) {
        const std::size_t T = 11, C = 130;  // 130: two full 64-channel blocks + a partial one
        auto x = rng.normal(T * C), w = rng.normal(C * K, 0.5f), st0 = rng.normal((K - 1) * C);
        auto st = st0;
        std::vector<float> y(T * C);
        causal_conv1d_silu(cview(x, T, C), cview(w, C, K), view(st, K - 1, C), view(y, T, C));
        std::vector<double> pre_abs;
        const auto ref = conv_ref(x, T, C, w, K, st0, &pre_abs);
        Worst worst;
        for (std::size_t i = 0; i < T * C; ++i) {
            // Model R over K taps (silu' <= 1.1), plus Model E for silu itself.
            const double tol = 1.1 * tol_reduction(K, pre_abs[i]) + tol_elementwise(ref[i]) + 8 * kEps * 1e-30;
            worst.add(std::abs(y[i] - ref[i]), tol);
            ASSERT_LE(std::abs(y[i] - ref[i]), tol) << "K=" << K << " i=" << i;
        }
        std::printf("[tol] conv1d K=%zu worst err/tol = %.3g\n", K, worst.ratio);
        // New state = last K-1 inputs, oldest first.
        for (std::size_t s = 0; s + 1 < K; ++s)
            for (std::size_t c = 0; c < C; ++c) ASSERT_EQ(st[s * C + c], x[(T - (K - 1) + s) * C + c]);
    }
}

TEST(CpuConv1d, SplitInvarianceIsBitwise) {
    Rng rng(7);
    const std::size_t T = 37, C = 100, K = 4;
    auto x = rng.normal(T * C), w = rng.normal(C * K, 0.5f), st0 = rng.normal((K - 1) * C);
    auto st_one = st0;
    std::vector<float> y_one(T * C);
    causal_conv1d_silu(cview(x, T, C), cview(w, C, K), view(st_one, K - 1, C), view(y_one, T, C));
    // Splits: all T=1 steps; and an irregular split including a 1-token and a 2-token call.
    for (const std::vector<std::size_t>& split :
         {std::vector<std::size_t>(T, 1), std::vector<std::size_t>{2, 1, 10, 1, 1, 22}}) {
        auto st = st0;
        std::vector<float> y(T * C);
        std::size_t t0 = 0;
        for (std::size_t n : split) {
            causal_conv1d_silu(ConstRows(x.data() + t0 * C, n, C, C), cview(w, C, K), view(st, K - 1, C),
                               Rows(y.data() + t0 * C, n, C, C));
            t0 += n;
        }
        ASSERT_EQ(t0, T);
        EXPECT_TRUE(bitwise_equal(y, y_one));
        EXPECT_TRUE(bitwise_equal(st, st_one));
    }
    // In place (out aliases x).
    auto st = st0;
    auto xin = x;
    causal_conv1d_silu(cview(xin, T, C), cview(w, C, K), view(st, K - 1, C), view(xin, T, C));
    EXPECT_TRUE(bitwise_equal(xin, y_one));
    EXPECT_TRUE(bitwise_equal(st, st_one));
    // Validation.
    std::vector<float> bad_state(K * C);
    EXPECT_THROW(causal_conv1d_silu(cview(x, T, C), cview(w, C, K), view(bad_state, K, C), view(y_one, T, C)),
                 halo::Error);
}

TEST(CpuConv1d, RollbackSlotsEqualTruncatedRunBitwise) {
    Rng rng(13);
    const std::size_t C = 70, K = 4, kSlots = 3;
    for (std::size_t T : {1u, 2u, 3u, 9u}) {
        auto x = rng.normal(T * C), w = rng.normal(C * K, 0.5f), st0 = rng.normal((K - 1) * C);
        auto st = st0;
        std::vector<float> y(T * C), slots(kSlots * (K - 1) * C, 12345.0f);
        causal_conv1d_silu(cview(x, T, C), cview(w, C, K), view(st, K - 1, C), view(y, T, C), nullptr, slots);
        std::vector<float> y_plain(T * C);
        auto st_plain = st0;
        causal_conv1d_silu(cview(x, T, C), cview(w, C, K), view(st_plain, K - 1, C), view(y_plain, T, C));
        EXPECT_TRUE(bitwise_equal(y, y_plain));
        EXPECT_TRUE(bitwise_equal(st, st_plain));
        for (std::size_t s = 0; s < kSlots; ++s) {
            const auto slot = std::span(slots).subspan(s * (K - 1) * C, (K - 1) * C);
            if (s >= T) {
                for (float e : slot) ASSERT_EQ(e, 12345.0f) << "slot " << s << " >= T=" << T << " was written";
                continue;
            }
            auto st_t = st0;
            std::vector<float> y_t((T - s) * C);
            causal_conv1d_silu(ConstRows(x.data(), T - s, C, C), cview(w, C, K), view(st_t, K - 1, C),
                               view(y_t, T - s, C));
            EXPECT_TRUE(bitwise_equal(slot, st_t)) << "T=" << T << " slot " << s;
        }
    }
    std::vector<float> bad(5);
    std::vector<float> x(2 * C), w(C * K), st((K - 1) * C), y(2 * C);
    EXPECT_THROW(causal_conv1d_silu(cview(x, 2, C), cview(w, C, K), view(st, K - 1, C), view(y, 2, C), nullptr, bad),
                 halo::Error);
}

TEST(CpuConv1d, RingFormMatchesSlotFormBitwise) {
    // ADR-001 §5.3: the ring form places the same values the conv_state/state_slots form
    // computes: final state at slab[(live+1) mod P], logical slot s at slab[(live+1+s) mod P].
    Rng rng(14);
    const std::size_t C = 70, K = 4;
    for (std::size_t T : {1u, 2u, 3u, 9u}) {
        auto x = rng.normal(T * C), w = rng.normal(C * K, 0.5f), st0 = rng.normal((K - 1) * C);
        for (std::size_t n_slots : {0u, 3u}) {
            for (std::size_t P : {2u, 5u}) {
                if (n_slots > P - 1) continue;
                for (std::size_t live : {std::size_t{0}, P - 1}) {  // covers the wrap
                    SCOPED_TRACE(testing::Message() << "T=" << T << " n_slots=" << n_slots << " P=" << P
                                                    << " live=" << live);
                    auto st = st0;
                    std::vector<float> y(T * C), slots(n_slots * (K - 1) * C, 12345.0f);
                    causal_conv1d_silu(cview(x, T, C), cview(w, C, K), view(st, K - 1, C), view(y, T, C), nullptr,
                                       slots);
                    const std::size_t sn = (K - 1) * C;
                    std::vector<float> slab(P * sn, 12345.0f);
                    std::copy_n(st0.begin(), sn, slab.begin() + static_cast<std::ptrdiff_t>(live * sn));
                    std::vector<float> yr(T * C);
                    causal_conv1d_silu(cview(x, T, C), cview(w, C, K),
                                       StateRingView{std::span<float>(slab), P, live, n_slots}, view(yr, T, C));
                    EXPECT_TRUE(bitwise_equal(y, yr)) << "out differs";
                    // The ring always writes the final state (logical slot 0), n_slots or
                    // not; rollback slots are s < min(T, n_slots) (ADR-001 §5.3).
                    const std::size_t used = std::max<std::size_t>(1, std::min(T, n_slots));
                    for (std::size_t p = 0; p < P; ++p) {
                        const auto phys = std::span(slab).subspan(p * sn, sn);
                        if (p == live) {
                            EXPECT_TRUE(bitwise_equal(phys, st0)) << "the input slot was written";
                            continue;
                        }
                        const std::size_t s = (p + P - ((live + 1) % P)) % P;
                        if (s >= used) {
                            for (float e : phys) EXPECT_EQ(e, 12345.0f) << "unwritten physical slot " << p;
                        } else if (s == 0) {
                            EXPECT_TRUE(bitwise_equal(phys, st)) << "final state";
                        } else {
                            EXPECT_TRUE(bitwise_equal(phys, std::span(slots).subspan(s * sn, sn))) << "slot " << s;
                        }
                    }
                }
            }
        }
    }
    // Validation.
    const std::size_t sn = (K - 1) * C;
    std::vector<float> x(2 * C), w(C * K), slab(5 * sn, 0.0f), y(2 * C);
    EXPECT_THROW(causal_conv1d_silu(cview(x, 2, C), cview(w, C, K), StateRingView{slab, 1, 0, 0}, view(y, 2, C)),
                 halo::Error)
        << "P < 2";
    EXPECT_THROW(causal_conv1d_silu(cview(x, 2, C), cview(w, C, K), StateRingView{slab, 5, 5, 0}, view(y, 2, C)),
                 halo::Error)
        << "live >= P";
    EXPECT_THROW(causal_conv1d_silu(cview(x, 2, C), cview(w, C, K), StateRingView{slab, 5, 0, 5}, view(y, 2, C)),
                 halo::Error)
        << "slots > P-1";
}

// ---------------------------------------------------------------------------- elementwise
TEST(CpuElementwise, MatchFp64AndTorchThresholds) {
    std::vector<float> x;
    for (int i = -3000; i <= 3000; ++i) x.push_back(static_cast<float>(i) * 0.01f);
    for (float v : {19.99f, 20.0f, 20.0001f, 20.01f, 25.0f, -100.0f, 100.0f, 0.0f, -0.0f, 1e-8f, -88.0f, 88.0f})
        x.push_back(v);
    Rng rng(8);
    auto b = rng.normal(x.size(), 3.0f);
    const std::size_t n = x.size();
    std::vector<float> y(n);
    Worst worst;
    auto check = [&](const char* name, auto ref_fn) {
        for (std::size_t i = 0; i < n; ++i) {
            const double ref = ref_fn(double(x[i]), double(b[i]));
            const double tol = tol_elementwise(ref);
            worst.add(std::abs(y[i] - ref), tol);
            ASSERT_LE(std::abs(y[i] - ref), tol) << name << " x=" << x[i];
        }
    };
    silu(ConstRows(std::span<const float>(x)), Rows(std::span<float>(y)));
    check("silu", [](double v, double) { return silu64(v); });
    sigmoid(ConstRows(std::span<const float>(x)), Rows(std::span<float>(y)));
    check("sigmoid", [](double v, double) { return 1.0 / (1.0 + std::exp(-v)); });
    softplus(ConstRows(std::span<const float>(x)), Rows(std::span<float>(y)));
    check("softplus", [](double v, double) { return v > 20.0 ? v : std::log1p(std::exp(v)); });
    swiglu(ConstRows(std::span<const float>(x)), ConstRows(std::span<const float>(b)), Rows(std::span<float>(y)));
    check("swiglu", [](double v, double u) { return silu64(v) * u; });
    mul_sigmoid(ConstRows(std::span<const float>(b)), ConstRows(std::span<const float>(x)), Rows(std::span<float>(y)));
    check("mul_sigmoid", [](double v, double u) { return u / (1.0 + std::exp(-v)); });
    std::printf("[tol] elementwise worst err/tol = %.3g\n", worst.ratio);

    // torch softplus threshold (x > 20 -> x). In fp32 log1p(exp(x)) already rounds to x for
    // x >= ~16 (e^-16 < ulp(16)/2), so the > vs >= choice at 20 is numerically invisible; the
    // threshold matters where exp overflows: softplus(100) must be 100, not inf.
    std::vector<float> th{20.0f, std::nextafter(20.0f, 30.0f), 100.0f, 1000.0f}, tho(4);
    softplus(ConstRows(std::span<const float>(th)), Rows(std::span<float>(tho)));
    EXPECT_EQ(tho[0], std::log1p(std::exp(20.0f)));
    EXPECT_EQ(tho[1], th[1]);
    EXPECT_EQ(tho[2], 100.0f);
    EXPECT_EQ(tho[3], 1000.0f);

    // add / mul are single correctly rounded operations: exact.
    add(ConstRows(std::span<const float>(x)), ConstRows(std::span<const float>(b)), Rows(std::span<float>(y)));
    for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(y[i], x[i] + b[i]);
    mul(ConstRows(std::span<const float>(x)), ConstRows(std::span<const float>(b)), Rows(std::span<float>(y)));
    for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(y[i], x[i] * b[i]);
    std::vector<float> short_b(n - 1);
    EXPECT_THROW(add(ConstRows(std::span<const float>(x)), ConstRows(std::span<const float>(short_b)),
                     Rows(std::span<float>(y))),
                 halo::Error);
}

TEST(CpuSoftmax, StableAndMatchesFp64) {
    Rng rng(9);
    const std::size_t rows = 3, d = 1000;
    auto x = rng.normal(rows * d, 5.0f);
    for (std::size_t i = 0; i < d; ++i) x[d + i] += 1000.0f;  // overflow without max-subtraction
    for (std::size_t i = 0; i < 10; ++i) x[2 * d + i] -= 1e4f;
    std::vector<float> y(rows * d);
    softmax_rows(cview(x, rows, d), view(y, rows, d));
    Worst worst;
    for (std::size_t r = 0; r < rows; ++r) {
        double m = -1e300, s = 0;
        for (std::size_t i = 0; i < d; ++i) m = std::max(m, double(x[r * d + i]));
        for (std::size_t i = 0; i < d; ++i) s += std::exp(double(x[r * d + i]) - m);
        double sum = 0;
        for (std::size_t i = 0; i < d; ++i) {
            const double ref = std::exp(double(x[r * d + i]) - m) / s;
            // exp argument (x - m) carries eps|x - m| absolute error -> relative error in p;
            // the length-d sum is Model R; plus Model E for exp and divide.
            const double tol = (kEps * std::abs(double(x[r * d + i]) - m)) * ref + tol_reduction(d, ref) +
                               tol_elementwise(ref);
            worst.add(std::abs(y[r * d + i] - ref), tol);
            ASSERT_LE(std::abs(y[r * d + i] - ref), tol) << "r=" << r << " i=" << i;
            sum += y[r * d + i];
        }
        EXPECT_NEAR(sum, 1.0, 1e-5);
    }
    std::printf("[tol] softmax worst err/tol = %.3g\n", worst.ratio);
}

// ------------------------------------------------------------------------ matmul / logits

WeightMatrix dequant_of(const std::vector<float>& w, std::size_t n, std::size_t k) {
    return WeightMatrix::dequantized(n, k, [&w, k](std::size_t r0, std::size_t nr, std::span<float> out) {
        std::copy_n(w.begin() + static_cast<std::ptrdiff_t>(r0 * k), nr * k, out.begin());
    });
}

TEST(CpuMatmul, MatchesFp64DenseAndDequantBitwiseEqual) {
    Rng rng(10);
    for (auto [t, k, n] : {std::tuple{1u, 5120u, 67u}, std::tuple{7u, 300u, 37u}, std::tuple{3u, 17u, 1u}}) {
        auto x = rng.normal(t * k), w = rng.normal(n * k, 0.1f);
        std::vector<float> y(t * n), y2(t * n);
        matmul(cview(x, t, k), WeightMatrix::dense(cview(w, n, k)), view(y, t, n));
        matmul(cview(x, t, k), dequant_of(w, n, k), view(y2, t, n));
        EXPECT_TRUE(bitwise_equal(y, y2));
        Worst worst;
        for (std::size_t i = 0; i < t; ++i) {
            for (std::size_t o = 0; o < n; ++o) {
                double ref = 0, sabs = 0;
                for (std::size_t j = 0; j < k; ++j) {
                    ref += double(x[i * k + j]) * w[o * k + j];
                    sabs += std::abs(double(x[i * k + j]) * w[o * k + j]);
                }
                const double tol = tol_reduction(k, sabs);
                worst.add(std::abs(y[i * n + o] - ref), tol);
                ASSERT_LE(std::abs(y[i * n + o] - ref), tol);
            }
        }
        std::printf("[tol] matmul K=%u worst err/tol = %.3g\n", k, worst.ratio);
    }
    std::vector<float> x(10), w(12), y(2);
    EXPECT_THROW(matmul(cview(x, 1, 10), WeightMatrix::dense(cview(w, 2, 6)), view(y, 1, 2)), halo::Error);
}

TEST(CpuLogits, ArgmaxTopKTiesAndNaN) {
    std::vector<float> l{1.0f, 5.0f, -2.0f, 5.0f, 3.0f, 5.0f, 0.0f};
    EXPECT_EQ(argmax(l), (TopKEntry{1, 5.0f}));
    const auto tk = top_k(l, 4);
    ASSERT_EQ(tk.size(), 4u);
    EXPECT_EQ(tk[0], (TopKEntry{1, 5.0f}));
    EXPECT_EQ(tk[1], (TopKEntry{3, 5.0f}));
    EXPECT_EQ(tk[2], (TopKEntry{5, 5.0f}));
    EXPECT_EQ(tk[3], (TopKEntry{4, 3.0f}));
    EXPECT_EQ(top_k(l, l.size()).back(), (TopKEntry{2, -2.0f}));
    EXPECT_THROW((void)top_k(l, 0), halo::Error);
    EXPECT_THROW((void)top_k(l, l.size() + 1), halo::Error);
    EXPECT_THROW((void)argmax(std::span<const float>()), halo::Error);
    l[4] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW((void)argmax(l), halo::Error);
    EXPECT_THROW((void)top_k(l, 2), halo::Error);
    std::vector<float> neg{-std::numeric_limits<float>::infinity(), -3.0f};
    EXPECT_EQ(argmax(neg).index, 1);
}

TEST(CpuLogits, FusedMatmulArgmaxEqualsUnfusedIncludingTies) {
    Rng rng(11);
    const std::size_t k = 64, n = 1000;
    auto x = rng.normal(k), w = rng.normal(n * k);
    // Duplicate the best row at a lower and a higher index to force exact ties.
    std::vector<float> y(n);
    matmul(cview(x, 1, k), WeightMatrix::dense(cview(w, n, k)), view(y, 1, n));
    const auto best = argmax(y);
    std::copy_n(w.begin() + best.index * static_cast<std::ptrdiff_t>(k), k, w.begin() + 700 * k);
    std::copy_n(w.begin() + best.index * static_cast<std::ptrdiff_t>(k), k, w.begin() + 3 * k);
    matmul(cview(x, 1, k), WeightMatrix::dense(cview(w, n, k)), view(y, 1, n));
    const auto unfused = argmax(y);
    EXPECT_EQ(unfused.index, std::min<std::int32_t>(3, best.index));
    for (std::size_t threads : kThreadCounts) {
        ThreadPool pool(threads);
        EXPECT_EQ(matmul_argmax(x, WeightMatrix::dense(cview(w, n, k)), &pool), unfused) << threads;
        EXPECT_EQ(matmul_argmax(x, dequant_of(w, n, k), &pool), unfused) << threads;
    }
    w[5 * k] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW((void)matmul_argmax(x, WeightMatrix::dense(cview(w, n, k))), halo::Error);
}

// ------------------------------------------------------------------ thread-count invariance

TEST(CpuThreadInvariance, AllNonGdnOpsBitIdentical) {
    Rng rng(12);
    const std::size_t T = 9, D = 256, H = 4, HD = 64, C = 200, K = 4, N = 150;
    auto x = rng.normal(T * D), z = rng.normal(T * D), w = rng.normal(D), wc = rng.normal(C * K);
    auto xc = rng.normal(T * C), st0 = rng.normal((K - 1) * C), wm = rng.normal(N * D, 0.1f);
    std::vector<std::int32_t> pos(T);
    std::iota(pos.begin(), pos.end(), 100);

    auto run_all = [&](ThreadPool* pool) {
        std::vector<float> res;
        auto append = [&](const std::vector<float>& v) { res.insert(res.end(), v.begin(), v.end()); };
        std::vector<float> y(T * D);
        rms_norm(cview(x, T, D), w, 1e-6f, view(y, T, D), pool);
        append(y);
        gated_rms_norm(cview(x, T * H, HD), std::span(w).first(HD), cview(z, T * H, HD), 1e-6f, view(y, T * H, HD),
                       pool);
        append(y);
        l2_norm_heads(cview(x, T, D), H, HD, view(y, T, D), 1e-6f, pool);
        append(y);
        y = x;
        partial_rope_neox(view(y, T, D), H, HD, pos, 16, 1e7f, pool);
        append(y);
        auto st = st0;
        std::vector<float> yc(T * C);
        causal_conv1d_silu(cview(xc, T, C), cview(wc, C, K), view(st, K - 1, C), view(yc, T, C), pool);
        append(yc);
        append(st);
        swiglu(cview(x, T, D), cview(z, T, D), view(y, T, D), pool);
        append(y);
        mul_sigmoid(cview(x, T, D), cview(z, T, D), view(y, T, D), pool);
        append(y);
        softplus(cview(x, T, D), view(y, T, D), pool);
        append(y);
        softmax_rows(cview(x, T, D), view(y, T, D), pool);
        append(y);
        std::vector<float> ym(T * N);
        matmul(cview(x, T, D), dequant_of(wm, N, D), view(ym, T, N), pool);
        append(ym);
        const auto am = matmul_argmax(std::span(x).first(D), WeightMatrix::dense(cview(wm, N, D)), pool);
        res.push_back(static_cast<float>(am.index));
        res.push_back(am.value);
        return res;
    };
    const auto base = run_all(nullptr);
    for (std::size_t threads : kThreadCounts) {
        ThreadPool pool(threads);
        EXPECT_TRUE(bitwise_equal(run_all(&pool), base)) << "threads=" << threads;
    }
}

TEST(CpuOpRegistry, StableUniqueIds) {
    for (std::size_t i = 0; i < kOpContracts.size(); ++i)
        for (std::size_t j = i + 1; j < kOpContracts.size(); ++j) EXPECT_NE(kOpContracts[i].id, kOpContracts[j].id);
    EXPECT_EQ(kOpContracts.front().id, "RMS_NORM");
}

}  // namespace
