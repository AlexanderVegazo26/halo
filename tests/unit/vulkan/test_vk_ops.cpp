// Differential tests: every Vulkan kernel against a CPU scalar reference (TRD §29-§30).
//
// Tolerances are a priori bounds, not fitted to observed output. With u = 2^-24 (fp32
// unit roundoff), a GPU sum of n terms computed as WG strided partial sums of
// ceil(n/WG) terms followed by a log2(WG)-level tree has forward error
//     |ŷ - y| <= (ceil(n/WG) + log2(WG) + c) · u · Σ|termᵢ|        (Higham, §3.1)
// where c covers the per-term rounding (dequant multiply-adds and the product). Every
// matvec check uses that bound with c = 4 against the exact (double) sum over the
// ggml-dequantized weights, with Σ|termᵢ| taken over |w|·|x| (for Q4_K over
// (|d·sc·q| + |dmin·m|)·|x|, because d·sc·q - dmin·m can cancel). The reference is
// evaluated in double, whose own error is ~1e-16 relative and ignored.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/vulkan/ops.h"
#include "reference.h"
#include "vk_test_util.h"

namespace hv = halo::vulkan;
namespace ref = halo::vulkan::ref;
using hv::test::compare;
using hv::test::download;
using hv::test::report;
using hv::test::upload;

namespace {

double log2u(std::uint32_t wg) { return std::log2(static_cast<double>(wg)); }

double sum_bound_factor(std::uint32_t n, std::uint32_t wg, double c = 4.0) {
    return (std::ceil(static_cast<double>(n) / wg) + log2u(wg) + c) * ref::k_u;
}

halo::DType dtype_of(ref::WType t) {
    switch (t) {
        case ref::WType::F32: return halo::DType::F32;
        case ref::WType::Q8_0: return halo::DType::Q8_0;
        case ref::WType::Q4_K: return halo::DType::Q4_K;
        case ref::WType::Q6_K: return halo::DType::Q6_K;
    }
    return halo::DType::F32;
}

const char* name_of(ref::WType t) {
    switch (t) {
        case ref::WType::F32: return "f32";
        case ref::WType::Q8_0: return "q8_0";
        case ref::WType::Q4_K: return "q4_k";
        case ref::WType::Q6_K: return "q6_k";
    }
    return "?";
}

std::string timing_label(const hv::Context& ctx) {
    return ctx.info().correctness_only ? "lavapipe (CPU) timing — not a GPU performance number"
                                       : "GPU timing (unvalidated, not a benchmark)";
}

// Runs one matvec differential case; returns the error stats.
hv::test::ErrorStats run_matvec(const std::shared_ptr<hv::Context>& ctx, hv::Ops& ops, ref::WType t,
                                std::uint32_t rows, std::uint32_t cols, std::uint32_t seed,
                                std::uint32_t max_exp = 30, bool print_timing = false) {
    std::mt19937 rng(seed);
    const ref::Weights w = ref::random_weights(t, rows, cols, rng, max_exp);
    const std::vector<float> x = ref::random_vec(cols, rng);
    std::vector<double> y_ref, scale;
    ref::matvec(w.deq, w.mag, x, rows, cols, y_ref, scale);

    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
    hv::Buffer bx = upload(ctx, std::span<const float>(x));
    hv::Buffer by = hv::Buffer::create(ctx, std::uint64_t{rows} * 4, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    const auto t0 = s.timestamp();
    ops.matvec(s, dtype_of(t), bw, bx, by, rows, cols);
    const auto t1 = s.timestamp();
    s.submit_and_wait();
    if (print_timing && t0 && t1) {
        std::cout << "[vk-timing] matvec_" << name_of(t) << " " << rows << "x" << cols << ": "
                  << *s.elapsed_ns(*t0, *t1) / 1e6 << " ms — " << timing_label(*ctx) << "\n";
    }
    const std::vector<float> y = download<float>(by, rows);
    const double f = sum_bound_factor(cols, ops.options().reduce_workgroup);
    return compare(y, y_ref, [&](std::size_t i) { return f * scale[i]; });
}

}  // namespace

// ---------------------------------------------------------------- rms_norm

TEST(VkOps, RmsNormMatchesReference) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    struct Case {
        std::uint32_t rows, cols, wg;
    };
    // cols: below the workgroup, not a multiple of it, the real hidden size, head dims.
    const Case cases[] = {{1, 1, 256},  {3, 100, 256}, {5, 257, 256}, {7, 5120, 256},
                          {48, 128, 256}, {4, 256, 64}, {2, 5120, 64}, {70000, 4, 32}};
    for (const Case& c : cases) {
        hv::Ops ops(ctx, hv::OpsOptions{.reduce_workgroup = c.wg});
        std::mt19937 rng(1000 + c.cols + c.rows);
        const auto x = ref::random_vec(std::size_t{c.rows} * c.cols, rng, 3.0f);
        const auto w = ref::random_vec(c.cols, rng);
        const float eps = 1e-6f;
        const auto y_ref = ref::rms_norm(x, w, c.rows, c.cols, eps);

        hv::Buffer bx = upload(ctx, std::span<const float>(x));
        hv::Buffer bw = upload(ctx, std::span<const float>(w));
        hv::Buffer by = hv::Buffer::create(ctx, x.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.rms_norm(s, bx, bw, by, c.rows, c.cols, eps);
        s.submit_and_wait();
        const auto y = download<float>(by, x.size());
        // Sum of squares: positive terms -> relative bound on the sum itself; then mean,
        // +eps, sqrt (<= 3 ulp in Vulkan: 1/inversesqrt), 1/x (2.5 ulp), two products.
        const double rtol = (std::ceil(double(c.cols) / c.wg) + log2u(c.wg) + 16.0) * ref::k_u;
        const auto st = compare(y, y_ref, [&](std::size_t i) { return rtol * std::fabs(y_ref[i]) + 1e-30; });
        report("rms_norm " + std::to_string(c.rows) + "x" + std::to_string(c.cols) + " wg" + std::to_string(c.wg),
               st);
        EXPECT_LE(st.max_ratio, 1.0) << "rows=" << c.rows << " cols=" << c.cols << " worst=" << st.worst;
    }
}

// ---------------------------------------------------------------- matvec

class VkMatvec : public ::testing::TestWithParam<ref::WType> {};

TEST_P(VkMatvec, MatchesReferenceAcrossShapes) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const ref::WType t = GetParam();
    const std::uint32_t blk = static_cast<std::uint32_t>(ref::block_elems(t));
    const std::uint32_t kmin = std::max<std::uint32_t>(blk, 32);
    struct Case {
        std::uint32_t rows, cols, wg;
    };
    std::vector<Case> cases{{1, kmin, 256},
                            {3, kmin * 3, 256},     // odd block count: Q8_0/Q6_K rows not word-aligned
                            {17, 5120, 256},        // hidden size
                            {64, 17408, 256},       // FFN down K dim
                            {9, 5120, 64},          // non-default workgroup (spec constant)
                            {5, kmin * 5, 32}};
    if (t == ref::WType::F32) cases.push_back({4, 1000, 256});  // not a multiple of the WG
    for (const Case& c : cases) {
        hv::Ops ops_wg(ctx, hv::OpsOptions{.reduce_workgroup = c.wg});
        const auto st = run_matvec(ctx, ops_wg, t, c.rows, c.cols, 7 * c.rows + c.cols);
        report(std::string("matvec_") + name_of(t) + " " + std::to_string(c.rows) + "x" + std::to_string(c.cols) +
                   " wg" + std::to_string(c.wg),
               st);
        EXPECT_LE(st.max_ratio, 1.0) << name_of(t) << " rows=" << c.rows << " cols=" << c.cols;
    }
}

TEST_P(VkMatvec, RealisticScalesMatchWithSmallRelativeError) {
    // Scales in a realistic band (fp16 exponent 5..14: ~1e-4..1): the bound still applies
    // and the relative error should be small for all well-conditioned outputs.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const ref::WType t = GetParam();
    hv::Ops ops(ctx);
    const std::uint32_t rows = 128, cols = 5120;
    const auto st = run_matvec(ctx, ops, t, rows, cols, 99, 14, /*print_timing=*/true);
    report(std::string("matvec_") + name_of(t) + " realistic 128x5120", st);
    EXPECT_LE(st.max_ratio, 1.0);
}

INSTANTIATE_TEST_SUITE_P(Types, VkMatvec,
                         ::testing::Values(ref::WType::F32, ref::WType::Q8_0, ref::WType::Q4_K, ref::WType::Q6_K),
                         [](const auto& info) { return std::string(name_of(info.param)); });

TEST(VkOps, MatvecQ6KBeyondWorkgroupCountLimitUses2DGrid) {
    // LM-head-like row count (248,320 > 65,535) with a small K to keep the test fast.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t rows = 248320;
    ASSERT_GT(rows, ctx->info().max_workgroup_count[0]) << "device X limit too large to exercise the 2-D grid";
    const auto st = run_matvec(ctx, ops, ref::WType::Q6_K, rows, 256, 5, 14, /*print_timing=*/true);
    report("matvec_q6_k 248320x256 (2-D grid)", st);
    EXPECT_LE(st.max_ratio, 1.0);
}

TEST(VkOps, MatvecValidatesShapesAndBuffers) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer w = hv::Buffer::create(ctx, 34 * 2, hv::MemoryUsage::DeviceLocal);  // 1 row x 64 Q8_0
    hv::Buffer x = hv::Buffer::create(ctx, 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer y = hv::Buffer::create(ctx, 16, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    EXPECT_NO_THROW(ops.matvec(s, halo::DType::Q8_0, w, x, y, 1, 64));
    EXPECT_THROW(ops.matvec(s, halo::DType::Q8_0, w, x, y, 2, 64), halo::Error);   // W too small
    EXPECT_THROW(ops.matvec(s, halo::DType::Q8_0, w, x, y, 1, 48), halo::Error);   // not a block multiple
    EXPECT_THROW(ops.matvec(s, halo::DType::Q8_0, w, x, y, 5, 32), halo::Error);   // y too small
    EXPECT_THROW(ops.matvec(s, halo::DType::Q8_0, w, x, x, 1, 64), halo::Error);   // aliasing
    EXPECT_THROW(ops.matvec(s, halo::DType::Q4_0, w, x, y, 1, 64), halo::Error);   // unsupported type
    EXPECT_THROW(ops.matvec(s, halo::DType::F32, w, x, y, 0, 64), halo::Error);    // empty
    // Overflow-checked size arithmetic.
    EXPECT_THROW(ops.matvec(s, halo::DType::F32, w, x, y, 0xFFFFFFFFu, 0xFFFFFFFFu), halo::Error);
    hv::Buffer w_odd = hv::Buffer::create(ctx, 34, hv::MemoryUsage::DeviceLocal);  // 34 B: last word partial
    EXPECT_THROW(ops.matvec(s, halo::DType::Q8_0, w_odd, x, y, 1, 32), halo::Error);
}

// ---------------------------------------------------------------- dequant cross-check dump

TEST(VkRef, DequantReferenceDumpForGgufPyCrossCheck) {
    // When HALO_VK_DUMP_DIR is set, write random blocks + this file's CPU dequantization so
    // an external script can compare against gguf-py (python/…/gguf quants). Otherwise
    // this only checks the reference on hand-built blocks.
    // Hand-built Q8_0: d = 0.5, qs = -128..-97.
    std::uint8_t q8[34];
    ref::store_u16(q8, 0x3800);  // 0.5
    for (int i = 0; i < 32; ++i) q8[2 + i] = static_cast<std::uint8_t>(-128 + i);
    float y[256];
    ref::dequant_q8_0(q8, y);
    EXPECT_EQ(y[0], -64.0f);
    EXPECT_EQ(y[31], -48.5f);
    // Hand-built Q4_K: d = 1, dmin = 2, scales for sub-block 5 (j >= 4 branch):
    // sc = (s[9] & 0xF) | ((s[1] >> 6) << 4), m = (s[9] >> 4) | ((s[5] >> 6) << 4).
    std::uint8_t q4[144] = {};
    ref::store_u16(q4, 0x3C00);      // d = 1
    ref::store_u16(q4 + 2, 0x4000);  // dmin = 2
    q4[4 + 1] = 0xC0;                // s[1] top bits -> sc high = 3
    q4[4 + 5] = 0x40;                // s[5] top bits -> m high = 1
    q4[4 + 9] = 0x25;                // sc low = 5, m low = 2
    q4[16 + 64 + 3] = 0x70;          // chunk 2, l = 3, high nibble 7 -> element 128+32+3 (sub-block 5)
    ref::dequant_q4_k(q4, y);
    // sc = 5 | 3<<4 = 53, m = 2 | 1<<4 = 18 -> 1*53*7 - 2*18 = 335
    EXPECT_EQ(y[128 + 32 + 3], 335.0f);
    // Hand-built Q6_K: d = 1, scale[8+ (is=0) + 2*quarter=4] ... element n=1, quarter 2, l = 0.
    std::uint8_t q6[210] = {};
    ref::store_u16(q6 + 208, 0x3C00);
    q6[64 + 0] = 0xA0;                          // ql[64+0] high nibble = 10 (quarter >= 2)
    q6[128 + 32 + 0] = 0x30;                    // qh[32+0] bits 4..5 = 3
    q6[192 + 8 + 4] = static_cast<std::uint8_t>(-3);  // scales[8 + 0 + 4] = -3
    ref::dequant_q6_k(q6, y);
    // q = (10 | 3<<4) - 32 = 26 -> 1 * -3 * 26 = -78 at y[128 + 64 + 0]
    EXPECT_EQ(y[192], -78.0f);

    const auto dir = std::getenv("HALO_VK_DUMP_DIR");  // NOLINT(concurrency-mt-unsafe)
    if (dir == nullptr) return;
    for (const ref::WType t : {ref::WType::Q8_0, ref::WType::Q4_K, ref::WType::Q6_K}) {
        std::mt19937 rng(2024);
        const ref::Weights w = ref::random_weights(t, 16, 1024, rng);
        const std::size_t raw = 16 * 1024 / ref::block_elems(t) * ref::block_bytes(t);
        std::ofstream(std::string(dir) + "/" + name_of(t) + ".blocks", std::ios::binary)
            .write(reinterpret_cast<const char*>(w.bytes.data()), static_cast<std::streamsize>(raw));
        std::ofstream(std::string(dir) + "/" + name_of(t) + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(w.deq.data()), static_cast<std::streamsize>(w.deq.size() * 4));
    }
    std::cout << "[vk-ref] dumped dequant fixtures to " << dir << "\n";
}

// ---------------------------------------------------------------- gated delta rule

namespace {

struct GdnDims {
    std::uint32_t n_v, n_k, d_k, d_v;
    [[nodiscard]] std::size_t state_elems() const { return std::size_t{n_v} * d_k * d_v; }
};

struct GdnRow {
    std::vector<float> q, k, v, g, beta;
};

// Log-decay / beta ranges of a fixture. Fast: g in [-4, -0.01) (state forgets in a few
// steps). Slow: g in [-0.05, -0.001), beta near 1 — the long-memory heads, where the
// initial state and early rows still dominate after 20 steps.
struct GdnGates {
    float g_lo = -4.0f, g_hi = -0.01f, b_lo = 0.02f, b_hi = 0.98f;
};
constexpr GdnGates k_fast_decay{};
constexpr GdnGates k_slow_decay{-0.05f, -0.001f, 0.9f, 0.999f};

// One row of inputs: q, k L2-normalized per head, q scaled by 1/sqrt(d_k) on the host
// (run with qk_l2norm = false, q_scale = 1, see gdn_args); g (log-decay) and beta drawn
// from `gates`.
GdnRow gdn_row(const GdnDims& d, std::mt19937& rng, const GdnGates& gates = k_fast_decay) {
    GdnRow r;
    r.q = ref::random_vec(std::size_t{d.n_k} * d.d_k, rng);
    r.k = ref::random_vec(std::size_t{d.n_k} * d.d_k, rng);
    r.v = ref::random_vec(std::size_t{d.n_v} * d.d_v, rng);
    for (auto* vec : {&r.q, &r.k}) {
        for (std::uint32_t h = 0; h < d.n_k; ++h) {
            double nrm = 0.0;
            for (std::uint32_t a = 0; a < d.d_k; ++a) nrm += ref::dbl((*vec)[h * d.d_k + a]) * ref::dbl((*vec)[h * d.d_k + a]);
            const double inv = 1.0 / std::sqrt(nrm + 1e-12);
            for (std::uint32_t a = 0; a < d.d_k; ++a) {
                (*vec)[h * d.d_k + a] = static_cast<float>(ref::dbl((*vec)[h * d.d_k + a]) * inv);
            }
        }
    }
    const float qs = 1.0f / std::sqrt(static_cast<float>(d.d_k));
    for (float& x : r.q) x *= qs;
    std::uniform_real_distribution<float> gd(gates.g_lo, gates.g_hi), bd(gates.b_lo, gates.b_hi);
    r.g.resize(d.n_v);
    r.beta.resize(d.n_v);
    for (auto& x : r.g) x = gd(rng);
    for (auto& x : r.beta) x = bd(rng);
    return r;
}

// Concatenate T rows into token-major arrays.
GdnRow concat(const std::vector<GdnRow>& rows) {
    GdnRow c;
    for (const auto& r : rows) {
        c.q.insert(c.q.end(), r.q.begin(), r.q.end());
        c.k.insert(c.k.end(), r.k.begin(), r.k.end());
        c.v.insert(c.v.end(), r.v.begin(), r.v.end());
        c.g.insert(c.g.end(), r.g.begin(), r.g.end());
        c.beta.insert(c.beta.end(), r.beta.begin(), r.beta.end());
    }
    return c;
}

double max_abs(std::span<const double> v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// A priori per-step bound (see file header): each state column update is two length-d_k
// dot products plus exp (<= 3 + 2|g| ulp; |g| < 4 -> 11 ulp) and a few roundings; the
// update map is non-expansive (decay <= 1, beta in (0,1), ||k||2 = 1), so errors add at
// most linearly over T steps. Normalized by the tensor's largest magnitude.
double gdn_factor(std::uint32_t d_k, std::uint32_t steps) { return steps * (2.0 * d_k + 24.0) * ref::k_u; }

struct GdnBuffers {
    hv::Buffer q, k, v, g, beta, out;
};

GdnBuffers gdn_upload(const std::shared_ptr<hv::Context>& ctx, const GdnRow& r, std::size_t out_elems) {
    return {upload(ctx, std::span<const float>(r.q)),    upload(ctx, std::span<const float>(r.k)),
            upload(ctx, std::span<const float>(r.v)),    upload(ctx, std::span<const float>(r.g)),
            upload(ctx, std::span<const float>(r.beta)),
            hv::Buffer::create(ctx, out_elems * 4, hv::MemoryUsage::HostCached)};
}

hv::GdnDecodeArgs gdn_args(GdnBuffers& b, hv::Buffer& state, const GdnDims& d, std::uint32_t T) {
    hv::GdnDecodeArgs a;
    a.q = &b.q;
    a.k = &b.k;
    a.v = &b.v;
    a.g = &b.g;
    a.beta = &b.beta;
    a.state = &state;
    a.out = &b.out;
    a.n_v = d.n_v;
    a.n_k = d.n_k;
    a.d_k = d.d_k;
    a.d_v = d.d_v;
    a.n_tokens = T;
    // These tests feed pre-normalized, pre-scaled q/k (gdn_row), so they select the D-016
    // "apply nothing" contract explicitly; the raw-q/k in-kernel-L2 path is covered
    // differentially in test_vk_differential.cpp.
    a.qk_l2norm = false;
    a.q_scale = 1.0f;
    return a;
}

std::vector<float> random_state(const GdnDims& d, std::mt19937& rng) {
    return ref::random_vec(d.state_elems(), rng, 0.5f);
}

// Single-dispatch-per-step continuity: `steps` separate decode calls, state in place.
void gdn_continuity(const std::shared_ptr<hv::Context>& ctx, const GdnDims& d, std::uint32_t steps,
                    std::uint32_t wg, const std::string& label, const GdnGates& gates = k_fast_decay) {
    hv::Ops ops(ctx, hv::OpsOptions{.gdn_workgroup = wg});
    std::mt19937 rng(77 + d.n_v + d.d_k);
    const auto s0 = random_state(d, rng);
    std::vector<double> s_ref(s0.begin(), s0.end());
    hv::Buffer state = upload(ctx, std::span<const float>(s0), hv::MemoryUsage::HostVisible);
    double worst_out = 0.0, worst_state = 0.0;
    for (std::uint32_t t = 0; t < steps; ++t) {
        const GdnRow r = gdn_row(d, rng, gates);
        std::vector<double> o_ref;
        ref::gdn_step(r.q, r.k, r.v, r.g, r.beta, s_ref, o_ref, d.n_v, d.n_k, d.d_k, d.d_v);
        GdnBuffers b = gdn_upload(ctx, r, std::size_t{d.n_v} * d.d_v);
        hv::Stream s(ctx);
        ops.gated_delta_rule_decode(s, gdn_args(b, state, d, 1));
        s.submit_and_wait();
        const auto o = download<float>(b.out, o_ref.size());
        const double scale_o = max_abs(s_ref) * 1.0;  // |o_b| <= Σ_a |S_ab||q_a| <= max|S| * ||q||_1 <= max|S|
        const double f = gdn_factor(d.d_k, t + 1);
        const auto so = compare(o, o_ref, [&](std::size_t) { return f * scale_o; });
        EXPECT_LE(so.max_ratio, 1.0) << label << " step " << t << " out worst=" << so.worst;
        worst_out = std::max(worst_out, so.max_ratio);
        if (t + 1 == steps) report(label + " out (step " + std::to_string(t + 1) + ")", so);
    }
    const auto st = download<float>(state, d.state_elems());
    const double f = gdn_factor(d.d_k, steps);
    const double scale_s = max_abs(s_ref);
    const auto ss = compare(st, s_ref, [&](std::size_t) { return f * scale_s; });
    worst_state = ss.max_ratio;
    report(label + " final state", ss);
    EXPECT_LE(ss.max_ratio, 1.0) << label << " final state worst=" << ss.worst;
    std::cout << "[vk-err] " << label << " worst err/bound: out=" << worst_out << " state=" << worst_state << "\n";
}

}  // namespace

TEST(VkGdn, SingleStepRealDims) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_continuity(ctx, {48, 16, 128, 128}, 1, 128, "gdn real dims 1 step");
}

TEST(VkGdn, SingleStepSmallDimsTiledMapping) {
    // n_v/n_k = 3 > 1 so the j % n_k (tiled) vs j / 3 (grouped) mapping is discriminated.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_continuity(ctx, {6, 2, 16, 8}, 1, 128, "gdn small dims 1 step");
    gdn_continuity(ctx, {6, 2, 16, 40}, 1, 16, "gdn small dims 1 step wg16 (d_v > wg)");
}

TEST(VkGdn, TwentyStepContinuityRealDims) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_continuity(ctx, {48, 16, 128, 128}, 20, 128, "gdn real dims 20 steps");
}

TEST(VkGdn, TwentyStepContinuitySmallDims) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_continuity(ctx, {6, 2, 16, 8}, 20, 64, "gdn small dims 20 steps");
}

TEST(VkGdn, TwentyStepContinuitySlowDecay) {
    // Long-memory regime: exp(g) >= 0.95 per step, so the state carried across all 20
    // steps is not washed out and accumulated error is actually exercised.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_continuity(ctx, {48, 16, 128, 128}, 20, 128, "gdn real dims 20 steps slow decay", k_slow_decay);
    gdn_continuity(ctx, {6, 2, 16, 8}, 20, 64, "gdn small dims 20 steps slow decay", k_slow_decay);
}

namespace {

// T-row call with K slots vs the CPU reference run row by row.
void gdn_multi_row(const std::shared_ptr<hv::Context>& ctx, const GdnDims& d, std::uint32_t T, std::uint32_t K,
                   const std::string& label) {
    hv::Ops ops(ctx);
    std::mt19937 rng(9000 + T * 10 + K);
    const auto s0 = random_state(d, rng);
    std::vector<GdnRow> rows;
    for (std::uint32_t t = 0; t < T; ++t) rows.push_back(gdn_row(d, rng));
    const GdnRow all = concat(rows);

    // CPU: states after each row, outputs per row.
    std::vector<double> s_ref(s0.begin(), s0.end());
    std::vector<std::vector<double>> states_after;
    std::vector<double> out_ref;
    for (std::uint32_t t = 0; t < T; ++t) {
        std::vector<double> o;
        ref::gdn_step(rows[t].q, rows[t].k, rows[t].v, rows[t].g, rows[t].beta, s_ref, o, d.n_v, d.n_k, d.d_k,
                      d.d_v);
        states_after.push_back(s_ref);
        out_ref.insert(out_ref.end(), o.begin(), o.end());
    }

    const std::size_t S = d.state_elems();
    // Input state at element offset S (region 1 of 2), output in a separate buffer at offset 3.
    std::vector<float> in_host(2 * S, -7.0f);
    std::copy(s0.begin(), s0.end(), in_host.begin() + static_cast<std::ptrdiff_t>(S));
    hv::Buffer state_in = upload(ctx, std::span<const float>(in_host), hv::MemoryUsage::HostVisible);
    hv::Buffer state_out = hv::Buffer::create(ctx, (S + 3) * 4, hv::MemoryUsage::HostCached);
    // Slots buffer with a sentinel so untouched slots (s >= T) can be checked.
    const std::uint32_t slot_cap = K;
    std::vector<float> slot_init(std::size_t{slot_cap} * S + 5, 42.0f);
    hv::Buffer slots = upload(ctx, std::span<const float>(slot_init), hv::MemoryUsage::HostVisible);

    GdnBuffers b = gdn_upload(ctx, all, std::size_t{T} * d.n_v * d.d_v);
    hv::GdnDecodeArgs a = gdn_args(b, state_in, d, T);
    a.state_offset = S;
    a.state_out = &state_out;
    a.state_out_offset = 3;
    a.state_slots = K > 0 ? &slots : nullptr;
    a.slots_offset = 5;
    a.n_slots = K;
    hv::Stream s(ctx);
    ops.gated_delta_rule_decode(s, a);
    s.submit_and_wait();

    const double scale = max_abs(states_after.back());
    const double f = gdn_factor(d.d_k, T);
    const auto o = download<float>(b.out, out_ref.size());
    const auto so = compare(o, out_ref, [&](std::size_t) { return f * scale; });
    report(label + " out", so);
    EXPECT_LE(so.max_ratio, 1.0) << label;

    std::vector<float> fin(S);
    state_out.download(std::span<float>(fin), 3 * 4);
    const auto sf = compare(fin, states_after.back(), [&](std::size_t) { return f * scale; });
    report(label + " final state", sf);
    EXPECT_LE(sf.max_ratio, 1.0) << label;

    // Input region untouched (not in place), including the guard region before it.
    EXPECT_EQ(download<float>(state_in, 2 * S), in_host) << label << ": input state was modified";

    const auto slot_data = download<float>(slots, slot_init.size());
    for (std::uint32_t sl = 0; sl < K; ++sl) {
        const std::span<const float> got(slot_data.data() + 5 + std::size_t{sl} * S, S);
        if (sl < T) {
            const auto& expect = states_after[T - 1 - sl];
            const auto st = compare(got, expect, [&](std::size_t) { return gdn_factor(d.d_k, T - sl) * scale; });
            report(label + " slot " + std::to_string(sl), st);
            EXPECT_LE(st.max_ratio, 1.0) << label << " slot " << sl;
        } else {
            EXPECT_TRUE(std::all_of(got.begin(), got.end(), [](float x) { return x == 42.0f; }))
                << label << " slot " << sl << " (>= T) must be untouched";
        }
    }
    for (int i = 0; i < 5; ++i) EXPECT_EQ(slot_data[static_cast<std::size_t>(i)], 42.0f) << "slot guard";
}

}  // namespace

TEST(VkGdn, MultiRowWithStateSlotsRealDims) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_multi_row(ctx, {48, 16, 128, 128}, 4, 3, "gdn real T=4 K=3");
}

TEST(VkGdn, MultiRowWithStateSlotsSmallDims) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    gdn_multi_row(ctx, {6, 2, 16, 8}, 5, 5, "gdn small T=5 K=5");
    gdn_multi_row(ctx, {6, 2, 16, 8}, 3, 6, "gdn small T=3 K=6 (slots beyond T untouched)");
    gdn_multi_row(ctx, {6, 2, 16, 8}, 7, 0, "gdn small T=7 no slots");
}

TEST(VkGdn, SameBufferDisjointRegionsAndInPlaceAgree) {
    // Input region [0,S) -> output region [S,2S) of one buffer, vs the in-place default.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const GdnDims d{6, 2, 16, 8};
    const std::size_t S = d.state_elems();
    hv::Ops ops(ctx);
    std::mt19937 rng(5);
    const auto s0 = random_state(d, rng);
    const GdnRow r = gdn_row(d, rng);
    std::vector<double> s_ref(s0.begin(), s0.end()), o_ref;
    ref::gdn_step(r.q, r.k, r.v, r.g, r.beta, s_ref, o_ref, d.n_v, d.n_k, d.d_k, d.d_v);

    std::vector<float> two(2 * S, 0.0f);
    std::copy(s0.begin(), s0.end(), two.begin());
    hv::Buffer shared = upload(ctx, std::span<const float>(two), hv::MemoryUsage::HostVisible);
    hv::Buffer inplace = upload(ctx, std::span<const float>(s0), hv::MemoryUsage::HostVisible);
    GdnBuffers b1 = gdn_upload(ctx, r, std::size_t{d.n_v} * d.d_v);
    GdnBuffers b2 = gdn_upload(ctx, r, std::size_t{d.n_v} * d.d_v);
    hv::GdnDecodeArgs a1 = gdn_args(b1, shared, d, 1);
    a1.state_out_offset = S;
    hv::GdnDecodeArgs a2 = gdn_args(b2, inplace, d, 1);
    hv::Stream s(ctx);
    ops.gated_delta_rule_decode(s, a1);
    ops.gated_delta_rule_decode(s, a2);
    s.submit_and_wait();
    const auto got = download<float>(shared, 2 * S);
    EXPECT_TRUE(std::equal(s0.begin(), s0.end(), got.begin())) << "input region modified";
    const auto ip = download<float>(inplace, S);
    EXPECT_TRUE(std::equal(ip.begin(), ip.end(), got.begin() + static_cast<std::ptrdiff_t>(S)))
        << "disjoint-region result differs bitwise from the in-place result";
    const double f = gdn_factor(d.d_k, 1), scale = max_abs(s_ref);
    const auto st = compare(ip, s_ref, [&](std::size_t) { return f * scale; });
    EXPECT_LE(st.max_ratio, 1.0);
}

TEST(VkGdn, StateRegionsBeyondMaxStorageBufferRange) {
    // C-1 arenas: a state region can sit deeper in a buffer than maxStorageBufferRange
    // (128 MiB on lavapipe; one sequence's 48-layer state is 144 MiB, D-003). Regions are
    // bound at their own (aligned) offset; the odd element offset exercises the
    // sub-alignment remainder.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const GdnDims d{6, 2, 16, 8};
    const std::uint64_t S = d.state_elems();
    const std::uint64_t base = (ctx->info().max_storage_buffer_range / 4) + 5;  // element offset
    const std::uint32_t T = 2, K = 2;
    const std::uint64_t total = base + 4 * S;  // in, out, 2 slots
    if (total * 4 > ctx->info().max_memory_allocation_size) {
        GTEST_SKIP() << "maxMemoryAllocationSize " << ctx->info().max_memory_allocation_size
                     << " too small for a buffer past maxStorageBufferRange";
    }
    std::cout << "[vk-gdn] arena " << total * 4 << " B, maxStorageBufferRange "
              << ctx->info().max_storage_buffer_range << " B, input region at byte " << base * 4 << "\n";
    hv::Ops ops(ctx);
    std::mt19937 rng(8);
    const auto s0 = random_state(d, rng);
    const std::vector<GdnRow> rows{gdn_row(d, rng), gdn_row(d, rng)};
    std::vector<double> s_ref(s0.begin(), s0.end()), o;
    std::vector<std::vector<double>> after;
    for (const auto& r : rows) {
        ref::gdn_step(r.q, r.k, r.v, r.g, r.beta, s_ref, o, d.n_v, d.n_k, d.d_k, d.d_v);
        after.push_back(s_ref);
    }
    hv::Buffer arena = hv::Buffer::create(ctx, total * 4, hv::MemoryUsage::HostVisible);
    arena.upload(std::span<const float>(s0), base * 4);
    GdnBuffers b = gdn_upload(ctx, concat(rows), std::size_t{T} * d.n_v * d.d_v);
    hv::GdnDecodeArgs a = gdn_args(b, arena, d, T);
    a.state_offset = base;
    a.state_out_offset = base + S;
    a.state_slots = &arena;
    a.slots_offset = base + 2 * S;
    a.n_slots = K;
    hv::Stream s(ctx);
    ops.gated_delta_rule_decode(s, a);
    s.submit_and_wait();
    const double scale = max_abs(after.back());
    auto check = [&](std::uint64_t off, const std::vector<double>& expect, const char* what) {
        std::vector<float> got(S);
        arena.download(std::span<float>(got), off * 4);
        const auto st = compare(got, expect, [&](std::size_t) { return gdn_factor(d.d_k, T) * scale; });
        report(std::string("gdn deep arena ") + what, st);
        EXPECT_LE(st.max_ratio, 1.0) << what;
    };
    check(base + S, after[1], "final state");
    check(base + 2 * S, after[1], "slot 0");
    check(base + 3 * S, after[0], "slot 1");
    std::vector<float> in_after(S);
    arena.download(std::span<float>(in_after), base * 4);
    EXPECT_EQ(in_after, s0) << "input region modified";
}

TEST(VkGdn, ValidatesRegionsAndSizes) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const GdnDims d{6, 2, 16, 8};
    const std::size_t S = d.state_elems();
    hv::Ops ops(ctx);
    std::mt19937 rng(1);
    const GdnRow r = concat({gdn_row(d, rng), gdn_row(d, rng)});
    GdnBuffers b = gdn_upload(ctx, r, 2 * std::size_t{d.n_v} * d.d_v);
    hv::Buffer state = hv::Buffer::create(ctx, 3 * S * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer slots = hv::Buffer::create(ctx, 2 * S * 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    auto base = [&] { return gdn_args(b, state, d, 2); };
    EXPECT_NO_THROW(ops.gated_delta_rule_decode(s, base()));
    auto a = base();
    a.state_out_offset = S / 2;  // partial overlap with [0, S)
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.state_offset = 2 * S + 1;  // input region past the end
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.n_slots = 1;  // slots without a buffer
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.state_slots = &state;  // slots overlapping the state region
    a.n_slots = 1;
    a.slots_offset = S / 2;
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a.slots_offset = S;  // disjoint in the same buffer: allowed
    EXPECT_NO_THROW(ops.gated_delta_rule_decode(s, a));
    a = base();
    a.state_slots = &slots;
    a.n_slots = 3;  // only min(T, n_slots) = 2 slots are written -> fits in 2*S
    EXPECT_NO_THROW(ops.gated_delta_rule_decode(s, a));
    a = base();
    a.n_tokens = 3;  // inputs sized for T = 2
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.out = &state;  // output aliasing state
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.d_k = 512;  // > gdn_max_dk
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    s.submit_and_wait();
}

// ---------------------------------------------------------------- argmax

namespace {

std::uint32_t gpu_argmax(const std::shared_ptr<hv::Context>& ctx, hv::Ops& ops, std::span<const float> x,
                         float* value = nullptr, bool print_timing = false) {
    hv::Buffer bx = upload(ctx, x);
    hv::Buffer scratch = hv::Buffer::create(ctx, ops.argmax_scratch_bytes(static_cast<std::uint32_t>(x.size())),
                                            hv::MemoryUsage::DeviceLocal);
    hv::Buffer result = hv::Buffer::create(ctx, hv::k_argmax_result_bytes, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    const auto t0 = s.timestamp();
    ops.argmax(s, bx, static_cast<std::uint32_t>(x.size()), scratch, result);
    const auto t1 = s.timestamp();
    s.submit_and_wait();
    if (print_timing && t0 && t1) {
        std::cout << "[vk-timing] argmax n=" << x.size() << ": " << *s.elapsed_ns(*t0, *t1) / 1e6 << " ms — "
                  << timing_label(*ctx) << "\n";
    }
    const auto r = download<std::uint32_t>(result, 2);
    if (value) *value = std::bit_cast<float>(r[1]);
    return r[0];
}

}  // namespace

TEST(VkArgmax, VocabSizedRowMatchesReferenceExactly) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint32_t n = 248320;
    for (const std::uint32_t wg : {256u, 64u}) {
        hv::Ops ops(ctx, hv::OpsOptions{.reduce_workgroup = wg});
        std::cout << "[vk-argmax] wg=" << wg << " partials=" << ops.argmax_partials(n) << "\n";
        EXPECT_GT(ops.argmax_partials(n), 1u);
        std::mt19937 rng(123 + wg);
        auto x = ref::random_vec(n, rng, 4.0f);
        float v = 0;
        EXPECT_EQ(gpu_argmax(ctx, ops, x, &v, wg == 256), ref::argmax(x));
        EXPECT_EQ(v, x[ref::argmax(x)]);

        const float big = 1000.0f;
        auto check_at = [&](std::vector<std::uint32_t> at, std::uint32_t expect, const char* what) {
            auto y = x;
            for (auto i : at) y[i] = big;
            EXPECT_EQ(ref::argmax(y), expect) << what << " (reference)";
            EXPECT_EQ(gpu_argmax(ctx, ops, y), expect) << what << " wg=" << wg;
        };
        check_at({0}, 0, "max at index 0");
        check_at({n - 1}, n - 1, "max at the last index (ragged last group)");
        check_at({200001, 5003}, 5003, "tie across different partials -> lowest index");
        check_at({wg * 16 * 3 + 7, wg * 16 * 3 + 5}, wg * 16 * 3 + 5, "tie inside one partial");
        check_at({wg * 16 - 1, wg * 16}, wg * 16 - 1, "tie straddling a partial boundary");
    }
}

TEST(VkArgmax, EdgeValuesInfNanAndAllEqual) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> all_eq(100000, 2.5f);
    EXPECT_EQ(gpu_argmax(ctx, ops, all_eq), 0u);
    std::vector<float> neg_inf(10000, -inf);
    EXPECT_EQ(gpu_argmax(ctx, ops, neg_inf), 0u) << "all -inf: first index";
    neg_inf[7777] = -1e30f;
    EXPECT_EQ(gpu_argmax(ctx, ops, neg_inf), 7777u);
    std::vector<float> with_nan(50000, 0.0f);
    with_nan[10] = nan;
    with_nan[20] = 1.0f;
    with_nan[30] = nan;
    EXPECT_EQ(ref::argmax(with_nan), 20u);
    EXPECT_EQ(gpu_argmax(ctx, ops, with_nan), 20u) << "NaN ignored";
    with_nan.assign(9000, nan);
    with_nan[8999] = -inf;
    EXPECT_EQ(gpu_argmax(ctx, ops, with_nan), 8999u) << "only non-NaN element is -inf";
    with_nan.assign(9000, nan);
    EXPECT_EQ(ref::argmax(with_nan), hv::k_argmax_none);
    EXPECT_EQ(gpu_argmax(ctx, ops, with_nan), hv::k_argmax_none) << "all NaN -> none";
    std::vector<float> pos_inf(300, 1.0f);
    pos_inf[299] = inf;
    pos_inf[150] = inf;
    EXPECT_EQ(gpu_argmax(ctx, ops, pos_inf), 150u);
    std::vector<float> one{-3.0f};
    EXPECT_EQ(gpu_argmax(ctx, ops, one), 0u);
}

TEST(VkArgmax, ValidatesBuffers) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops64(ctx, hv::OpsOptions{.reduce_workgroup = 64});  // chunk 1024 -> 4 partials for n=4000
    ASSERT_EQ(ops64.argmax_partials(4000), 4u);
    hv::Buffer x = hv::Buffer::create(ctx, 4000 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer small = hv::Buffer::create(ctx, 8, hv::MemoryUsage::DeviceLocal);
    hv::Buffer scratch = hv::Buffer::create(ctx, ops64.argmax_scratch_bytes(4000), hv::MemoryUsage::DeviceLocal);
    hv::Buffer result = hv::Buffer::create(ctx, 8, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    EXPECT_THROW(ops64.argmax(s, x, 0, scratch, result), halo::Error);        // empty
    EXPECT_THROW(ops64.argmax(s, x, 4001, scratch, result), halo::Error);     // logits too small
    EXPECT_THROW(ops64.argmax(s, x, 4000, small, result), halo::Error);       // scratch too small
    EXPECT_THROW(ops64.argmax(s, x, 4000, scratch, scratch), halo::Error);    // aliasing
    EXPECT_NO_THROW(ops64.argmax(s, x, 4000, scratch, result));
    s.submit_and_wait();
}

// ---------------------------------------------------------------- chained ops

TEST(VkOps, ChainedNormMatvecArgmaxInOneSubmission) {
    // rms_norm -> matvec_q8_0 -> argmax recorded into one stream: exercises the automatic
    // inter-dispatch barriers (a missing barrier would not necessarily fail on lavapipe,
    // which executes dispatches serially — see report).
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t cols = 1024, rows = 3000;
    std::mt19937 rng(31);
    const auto x = ref::random_vec(cols, rng);
    const auto nw = ref::random_vec(cols, rng);
    const ref::Weights w = ref::random_weights(ref::WType::Q8_0, rows, cols, rng, 14);
    const auto xn = ref::rms_norm(x, nw, 1, cols, 1e-6f);
    std::vector<float> xn_f(xn.begin(), xn.end());
    std::vector<double> logits_ref, scale;
    ref::matvec(w.deq, w.mag, xn_f, rows, cols, logits_ref, scale);

    hv::Buffer bx = upload(ctx, std::span<const float>(x));
    hv::Buffer bn = upload(ctx, std::span<const float>(nw));
    hv::Buffer bxn = hv::Buffer::create(ctx, cols * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
    hv::Buffer blog = hv::Buffer::create(ctx, rows * 4, hv::MemoryUsage::HostCached);
    hv::Buffer scratch = hv::Buffer::create(ctx, ops.argmax_scratch_bytes(rows), hv::MemoryUsage::DeviceLocal);
    hv::Buffer result = hv::Buffer::create(ctx, 8, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    ops.rms_norm(s, bx, bn, bxn, 1, cols, 1e-6f);
    ops.matvec(s, halo::DType::Q8_0, bw, bxn, blog, rows, cols);
    ops.argmax(s, blog, rows, scratch, result);
    s.submit_and_wait();
    EXPECT_EQ(s.dispatch_count(), 4u);
    const auto logits = download<float>(blog, rows);
    // Compare the GPU argmax with the argmax of the GPU logits (exact), and the logits
    // with the reference within the matvec bound widened by the rms_norm input error.
    EXPECT_EQ(download<std::uint32_t>(result, 1)[0], ref::argmax(logits));
    const double f = sum_bound_factor(cols, 256) + 64 * ref::k_u;
    const auto st = compare(logits, logits_ref, [&](std::size_t i) { return f * scale[i]; });
    report("chained norm->q8_0 matvec", st);
    EXPECT_LE(st.max_ratio, 1.0);
}
