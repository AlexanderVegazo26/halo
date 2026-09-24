// Differential acceptance tests of the Vulkan kernels against the HALO CPU reference
// (halo::cpu operators, halo::tensor dequantization) on the SAME host input arrays
// (TRD §30; DECISIONS D-016; code review M-2).
//
// Every assertion compares the Vulkan result with the CPU result:
//     |vk - cpu| <= bound_vk + bound_cpu
// where each bound is that backend's a-priori error bound against exact arithmetic. fp64
// (cpu_kernels/reference.h) is used only to compute the scale the bounds multiply.
//
// GATED_DELTANET bounds (per value head, u = 2^-24):
//   bound_cpu = Model G of tests/unit/cpu_kernels/tolerance.h (random walk over T tokens).
//   bound_vk  = T (2 d_k + 48) u — worst case, linear in T. Per token each state column
//               sees two length-d_k dot products (<= d_k u |terms| each), exp(g) (<= 3 + 2|g|
//               ulp, |g| <= 4 -> 11 ulp), and the in-kernel L2 normalization of q and k (sum
//               of squares via strided partials + a tree: <= (ceil(d_k/WG) + log2 WG) u,
//               then sqrt, divide and two products: <= 8 ulp). The update map is
//               non-expansive (exp(g) <= 1, beta in (0, 1), ||k||_2 = 1), so per-token
//               errors add at most linearly.
//   Scales: state = max |S| over the head and trajectory (fp64); out = that max times
//   max_t ||q_eff,t||_1 (q after the kernel's own normalization/scale), which bounds both
//   the propagated state error and the output dot product's |terms|.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "cpu_kernels/reference.h"
#include "cpu_kernels/tolerance.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "vk_test_util.h"

// fp64 scale computations promote float operands on purpose.
#pragma GCC diagnostic ignored "-Wdouble-promotion"

namespace hv = halo::vulkan;
namespace hc = halo::cpu;
namespace ct = halo::cpu::test;
using hv::test::download;
using hv::test::upload;

namespace {

constexpr double k_u = 0x1p-24;

// ---------------------------------------------------------------- GDN fixtures

struct GdnCase {
    std::uint32_t n_v, n_k, d_k, d_v;
    [[nodiscard]] std::size_t qk_cols() const { return std::size_t{n_k} * d_k; }
    [[nodiscard]] std::size_t v_cols() const { return std::size_t{n_v} * d_v; }
    [[nodiscard]] std::size_t state_n() const { return std::size_t{n_v} * d_k * d_v; }
    [[nodiscard]] hc::GdnDims cpu() const { return {n_k, n_v, d_k, d_v, hc::GdnHeadMapping::Tiled}; }
    [[nodiscard]] ct::GdnRefDims ref() const { return {n_k, n_v, d_k, d_v, true}; }
};

/// Token-major raw inputs shared by both backends.
struct GdnData {
    GdnCase c;
    std::size_t T = 0;
    std::vector<float> q, k, v, g, beta, s0;
};

enum class QkInput {
    Raw,            // unnormalized (conv + SiLU output); the kernel must L2-normalize
    Prenormalized,  // unit-norm per head on the host (for qk_l2norm = false)
};

void l2_normalize_heads(std::vector<float>& x, std::size_t T, std::size_t heads, std::size_t d) {
    for (std::size_t r = 0; r < T * heads; ++r) {
        double ss = 0;
        for (std::size_t i = 0; i < d; ++i) ss += double(x[r * d + i]) * double(x[r * d + i]);
        const double inv = 1.0 / std::sqrt(ss);
        for (std::size_t i = 0; i < d; ++i) x[r * d + i] = static_cast<float>(double(x[r * d + i]) * inv);
    }
}

// g in [-4, -0.01) (or the slow long-memory band), beta in (0.02, 0.98).
GdnData make_gdn(const GdnCase& c, std::size_t T, QkInput qk, std::uint64_t seed, bool slow_decay = false) {
    ct::Rng rng(seed);
    GdnData d;
    d.c = c;
    d.T = T;
    d.q = rng.normal(T * c.qk_cols(), 2.0f);  // raw: norms far from 1 on purpose
    d.k = rng.normal(T * c.qk_cols(), 2.0f);
    d.v = rng.normal(T * c.v_cols());
    d.g = slow_decay ? rng.uniform(T * c.n_v, -0.05f, -0.001f) : rng.uniform(T * c.n_v, -4.0f, -0.01f);
    d.beta = slow_decay ? rng.uniform(T * c.n_v, 0.9f, 0.999f) : rng.uniform(T * c.n_v, 0.02f, 0.98f);
    d.s0 = rng.normal(c.state_n(), 0.5f);
    if (qk == QkInput::Prenormalized) {
        l2_normalize_heads(d.q, T, c.n_k, c.d_k);
        l2_normalize_heads(d.k, T, c.n_k, c.d_k);
    }
    return d;
}

struct GdnOut {
    std::vector<float> out, state;
};

GdnOut run_cpu(const GdnData& d, const hc::GdnQkParams& qk) {
    GdnOut r{std::vector<float>(d.T * d.c.v_cols()), d.s0};
    const std::size_t qc = d.c.qk_cols(), vc = d.c.v_cols(), nv = d.c.n_v;
    const hc::GdnInputs in{hc::ConstRows(d.q.data(), d.T, qc, qc), hc::ConstRows(d.k.data(), d.T, qc, qc),
                           hc::ConstRows(d.v.data(), d.T, vc, vc), hc::ConstRows(d.g.data(), d.T, nv, nv),
                           hc::ConstRows(d.beta.data(), d.T, nv, nv)};
    hc::gated_delta_rule_recurrent(d.c.cpu(), in, r.state, hc::Rows(r.out.data(), d.T, vc, vc), qk);
    return r;
}

/// One Vulkan dispatch over all T rows, state in place, same host arrays as run_cpu.
GdnOut run_vk(const std::shared_ptr<hv::Context>& ctx, hv::Ops& ops, const GdnData& d, bool qk_l2norm,
              std::optional<float> q_scale) {
    hv::Buffer q = upload(ctx, std::span<const float>(d.q));
    hv::Buffer k = upload(ctx, std::span<const float>(d.k));
    hv::Buffer v = upload(ctx, std::span<const float>(d.v));
    hv::Buffer g = upload(ctx, std::span<const float>(d.g));
    hv::Buffer beta = upload(ctx, std::span<const float>(d.beta));
    hv::Buffer state = upload(ctx, std::span<const float>(d.s0), hv::MemoryUsage::HostVisible);
    hv::Buffer out = hv::Buffer::create(ctx, d.T * d.c.v_cols() * 4, hv::MemoryUsage::HostCached);
    hv::GdnDecodeArgs a;
    a.q = &q;
    a.k = &k;
    a.v = &v;
    a.g = &g;
    a.beta = &beta;
    a.state = &state;
    a.out = &out;
    a.n_v = d.c.n_v;
    a.n_k = d.c.n_k;
    a.d_k = d.c.d_k;
    a.d_v = d.c.d_v;
    a.n_tokens = static_cast<std::uint32_t>(d.T);
    a.qk_l2norm = qk_l2norm;
    a.q_scale = q_scale;
    hv::Stream s(ctx);
    ops.gated_delta_rule_decode(s, a);
    s.submit_and_wait();
    return {download<float>(out, d.T * d.c.v_cols()), download<float>(state, d.c.state_n())};
}

/// fp64 scales per value head (see file header).
struct GdnScales {
    std::vector<double> state, out;
};

GdnScales gdn_scales(const GdnData& d, bool l2, double q_scale) {
    const GdnCase& c = d.c;
    GdnScales s{std::vector<double>(c.n_v, 0.0), std::vector<double>(c.n_v, 0.0)};
    std::vector<double> st(d.s0.begin(), d.s0.end());
    std::vector<double> qn1(c.n_v, 0.0);
    const std::size_t sh = std::size_t{c.d_k} * c.d_v;
    auto track = [&] {
        for (std::size_t j = 0; j < c.n_v; ++j)
            for (std::size_t e = 0; e < sh; ++e) s.state[j] = std::max(s.state[j], std::abs(st[j * sh + e]));
    };
    track();
    for (std::size_t t = 0; t < d.T; ++t) {
        const auto sl = [&](const std::vector<float>& x, std::size_t cols) {
            return std::span<const float>(x.data() + t * cols, cols);
        };
        (void)ct::gdn_ref(c.ref(), 1, sl(d.q, c.qk_cols()), sl(d.k, c.qk_cols()), sl(d.v, c.v_cols()),
                          sl(d.g, c.n_v), sl(d.beta, c.n_v), st, l2, q_scale);
        track();
        for (std::size_t j = 0; j < c.n_v; ++j) {
            const float* qh = d.q.data() + t * c.qk_cols() + (j % c.n_k) * c.d_k;
            double ss = 0, l1 = 0;
            for (std::size_t i = 0; i < c.d_k; ++i) ss += double(qh[i]) * double(qh[i]);
            const double inv = l2 ? 1.0 / std::sqrt(ss + 1e-6) : 1.0;
            for (std::size_t i = 0; i < c.d_k; ++i) l1 += std::abs(double(qh[i]) * inv * q_scale);
            qn1[j] = std::max(qn1[j], l1);
        }
    }
    for (std::size_t j = 0; j < c.n_v; ++j) s.out[j] = s.state[j] * qn1[j];
    return s;
}

double vk_gdn_factor(std::size_t T, std::uint32_t d_k) { return double(T) * (2.0 * d_k + 48.0) * k_u; }

/// |vk - cpu| <= (bound_vk + bound_cpu) * scale, per head; returns the worst err/bound.
double check_gdn(const GdnData& d, const GdnOut& vk, const GdnOut& cpu, const GdnScales& sc, const std::string& what) {
    const GdnCase& c = d.c;
    const double f = vk_gdn_factor(d.T, c.d_k);
    const std::size_t sh = std::size_t{c.d_k} * c.d_v;
    double worst = 0;
    for (std::size_t j = 0; j < c.n_v; ++j) {
        const double to = f * sc.out[j] + ct::tol_gdn(d.T, c.d_k, sc.out[j]) + ct::kDenormFloor;
        const double ts = f * sc.state[j] + ct::tol_gdn(d.T, c.d_k, sc.state[j]) + ct::kDenormFloor;
        double eo = 0, es = 0;
        for (std::size_t t = 0; t < d.T; ++t)
            for (std::size_t b = 0; b < c.d_v; ++b) {
                const std::size_t i = t * c.v_cols() + j * c.d_v + b;
                const double e = std::abs(double(vk.out[i]) - double(cpu.out[i]));
                eo = std::isfinite(double(vk.out[i])) ? std::max(eo, e) : std::numeric_limits<double>::infinity();
            }
        for (std::size_t e = 0; e < sh; ++e) {
            const double x = double(vk.state[j * sh + e]);
            const double err = std::abs(x - double(cpu.state[j * sh + e]));
            es = std::isfinite(x) ? std::max(es, err) : std::numeric_limits<double>::infinity();
        }
        EXPECT_LE(eo, to) << what << " head " << j << " out |vk-cpu| " << eo << " scale " << sc.out[j];
        EXPECT_LE(es, ts) << what << " head " << j << " state |vk-cpu| " << es << " scale " << sc.state[j];
        worst = std::max({worst, eo / to, es / ts});
    }
    std::cout << "[vk-diff] " << what << ": worst |vk-cpu|/bound = " << worst << "\n";
    return worst;
}

void gdn_differential(const std::shared_ptr<hv::Context>& ctx, const GdnCase& c, std::size_t T, QkInput in,
                      bool l2, std::optional<float> q_scale, std::uint64_t seed, const std::string& what,
                      bool slow_decay = false, const hv::OpsOptions& opts = {}) {
    hv::Ops ops(ctx, opts);
    const GdnData d = make_gdn(c, T, in, seed, slow_decay);
    const hc::GdnQkParams qk{.qk_l2norm = l2, .q_scale = q_scale};
    const GdnOut cpu = run_cpu(d, qk);
    const GdnOut vk = run_vk(ctx, ops, d, l2, q_scale);
    check_gdn(d, vk, cpu, gdn_scales(d, l2, hc::gdn_q_scale(qk, c.d_k)), what);
}

constexpr GdnCase k_real{48, 16, 128, 128};
constexpr GdnCase k_small{6, 2, 16, 8};  // n_v / n_k = 3: tiled j % n_k is discriminated

}  // namespace

// ---------------------------------------------------------------- GATED_DELTANET (D-016)

TEST(VkDiffGdn, RawQkWithInKernelL2NormMatchesCpu) {
    // The qwen35 forward contract: raw q/k, qk_l2norm = true, q_scale = 1/sqrt(d_k).
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const float qs = 1.0f / std::sqrt(128.0f);
    gdn_differential(ctx, k_real, 1, QkInput::Raw, true, qs, 11, "real dims T=1 l2");
    gdn_differential(ctx, k_real, 4, QkInput::Raw, true, qs, 12, "real dims T=4 l2");
    gdn_differential(ctx, k_small, 20, QkInput::Raw, true, 1.0f / std::sqrt(16.0f), 13, "small dims T=20 l2");
    gdn_differential(ctx, k_small, 20, QkInput::Raw, true, std::nullopt, 14, "small dims T=20 l2 default scale");
    gdn_differential(ctx, k_small, 20, QkInput::Raw, true, 0.3f, 15, "small dims T=20 l2 slow decay q_scale=0.3",
                     true);
}

TEST(VkDiffGdn, NoL2NormAppliesExactlyTheGivenQScale) {
    // qk_l2norm = false with a non-default q_scale: nothing may be normalized or scaled
    // implicitly. Inputs are unit-norm on the host so the recursion stays non-expansive.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_differential(ctx, k_real, 3, QkInput::Prenormalized, false, 0.5f, 21, "real dims T=3 no-l2 q_scale=0.5");
    gdn_differential(ctx, k_small, 20, QkInput::Prenormalized, false, 0.5f, 22, "small dims T=20 no-l2 q_scale=0.5");
    gdn_differential(ctx, k_small, 5, QkInput::Prenormalized, false, 1.0f, 23, "small dims T=5 no-l2 q_scale=1");
}

TEST(VkDiffGdn, InKernelL2NormAtOtherWorkgroupSizes) {
    // The per-head sum of squares is strided partials + a tree over WG. The default WG=128
    // equals d_k and is a power of two; these cover WG < d_k (several partials per thread)
    // and a non-power-of-two WG (the tree's `tid + s < WG` tail).
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const float qs = 1.0f / std::sqrt(128.0f);
    for (const std::uint32_t wg : {32u, 96u, 100u}) {
        const std::string w = " wg=" + std::to_string(wg);
        gdn_differential(ctx, k_real, 2, QkInput::Raw, true, qs, 41 + wg, "real dims T=2 l2" + w, false,
                         hv::OpsOptions{.gdn_workgroup = wg});
        gdn_differential(ctx, k_small, 6, QkInput::Raw, true, 0.25f, 42 + wg, "small dims T=6 l2" + w, false,
                         hv::OpsOptions{.gdn_workgroup = wg});
    }
}

TEST(VkDiffGdn, NonFiniteQScaleIsRejectedLikeCpu) {
    // Same validation as cpu::gdn_q_scale (D-016): nothing is recorded.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const GdnData d = make_gdn(k_small, 1, QkInput::Raw, 31);
    for (const float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        EXPECT_THROW((void)hc::gdn_q_scale(hc::GdnQkParams{.qk_l2norm = true, .q_scale = bad}, k_small.d_k),
                     halo::Error);
        EXPECT_THROW((void)run_vk(ctx, ops, d, true, bad), halo::Error);
    }
}
