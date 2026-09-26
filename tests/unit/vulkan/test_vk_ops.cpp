// Differential tests: every Vulkan kernel against the HALO CPU backend (halo::cpu) and
// HALO's dequantization (halo::tensor::dequantize_row), run on the SAME input arrays and
// weight bytes (TRD §29-§30; code review M-2). fp64 is used only for bound scales.
//
// Tolerances are a priori bounds, not fitted to observed output. Each assertion is
//     |vk - cpu| <= (bound_vk + bound_cpu) · scale
// where each bound is that backend's forward-error bound against exact arithmetic. With
// u = 2^-24 (fp32 unit roundoff), a sum of n terms computed as L strided partial sums of
// ceil(n/L) terms followed by a log2(L)-level tree has forward error
//     |ŷ - y| <= (ceil(n/L) + log2(L) + c) · u · Σ|termᵢ|        (Higham, §3.1)
// where c covers the per-term rounding (dequant multiply-adds and the product). The GPU
// kernels use L = WG (the reduction workgroup); the CPU dot product (backends/cpu
// kernel_common.h detail::dot) uses L = 8 lanes. Matvec checks use c = 4 on both sides
// with Σ|termᵢ| over |w|·|x| (for Q4_K over (|d·sc·q| + |dmin·m|)·|x|, because
// d·sc·q - dmin·m can cancel), computed in fp64.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

#include "cpu_kernels/tolerance.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/weight_matrix.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/tensor/quant.h"
#include "reference.h"
#include "vk_test_util.h"

namespace hv = halo::vulkan;
namespace hc = halo::cpu;
namespace ct = halo::cpu::test;
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

// CPU dot product (detail::dot): 8 strided lanes + a 3-level tree.
constexpr std::uint32_t k_cpu_lanes = 8;

halo::DType dtype_of(ref::WType t) { return ref::dtype(t); }

std::vector<double> to_dbl(std::span<const float> v) { return {v.begin(), v.end()}; }

// halo::cpu::matmul of one x row with the SAME weights the GPU reads: F32 as a dense view
// (w.deq holds exactly the uploaded bytes), quantized rows dequantized on demand from the
// uploaded bytes by halo::tensor::dequantize_row.
std::vector<float> cpu_matvec(const ref::Weights& w, ref::WType t, std::uint32_t rows, std::uint32_t cols,
                              std::span<const float> x) {
    const std::size_t rb = halo::tensor::row_bytes(ref::dtype(t), cols);
    const hc::WeightMatrix m =
        t == ref::WType::F32
            ? hc::WeightMatrix::dense(hc::ConstRows(w.deq.data(), rows, cols, cols))
            : hc::WeightMatrix::dequantized(rows, cols, [&w, t, rb, cols](std::size_t first, std::size_t n,
                                                                          std::span<float> out) {
                  for (std::size_t r = 0; r < n; ++r) {
                      halo::tensor::dequantize_row(ref::dtype(t), std::as_bytes(std::span(w.bytes)).data() + (first + r) * rb,
                                                   out.data() + r * cols, cols);
                  }
              });
    std::vector<float> y(rows);
    hc::matmul(hc::ConstRows(x.data(), 1, cols, cols), m, hc::Rows(y.data(), 1, rows, rows));
    return y;
}

const char* name_of(ref::WType t) {
    switch (t) {
        case ref::WType::F32: return "f32";
        case ref::WType::Q8_0: return "q8_0";
        case ref::WType::Q4_K: return "q4_k";
        case ref::WType::Q5_K: return "q5_k";
        case ref::WType::Q6_K: return "q6_k";
        case ref::WType::IQ4_XS: return "iq4_xs";
        case ref::WType::IQ4_NL: return "iq4_nl";
        case ref::WType::Q3_K: return "q3_k";
        case ref::WType::IQ3_S: return "iq3_s";
    }
    return "?";
}

std::string timing_label(const hv::Context& ctx) {
    return ctx.info().correctness_only ? "lavapipe (CPU) timing — not a GPU performance number"
                                       : "GPU timing (unvalidated, not a benchmark)";
}

// Runs one matvec differential case (GPU vs halo::cpu::matmul on the same weight bytes);
// returns the error stats.
hv::test::ErrorStats run_matvec(const std::shared_ptr<hv::Context>& ctx, hv::Ops& ops, ref::WType t,
                                std::uint32_t rows, std::uint32_t cols, std::uint32_t seed,
                                std::uint32_t max_exp = 30, bool print_timing = false) {
    std::mt19937 rng(seed);
    const ref::Weights w = ref::random_weights(t, rows, cols, rng, max_exp);
    const std::vector<float> x = ref::random_vec(cols, rng);
    const std::vector<double> y_cpu = to_dbl(cpu_matvec(w, t, rows, cols, x));
    const std::vector<double> scale = ref::matvec_scale(w.mag, x, rows, cols);

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
    const double f = sum_bound_factor(cols, ops.options().reduce_workgroup) + sum_bound_factor(cols, k_cpu_lanes);
    return compare(y, y_cpu, [&](std::size_t i) { return f * scale[i] + ct::kDenormFloor; });
}

}  // namespace

// ---------------------------------------------------------------- rms_norm

TEST(VkOps, RmsNormMatchesCpu) {
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
        std::vector<float> y_cpu(x.size());
        hc::rms_norm(hc::ConstRows(x.data(), c.rows, c.cols, c.cols), w, eps,
                     hc::Rows(y_cpu.data(), c.rows, c.cols, c.cols));
        const auto mag = ref::rms_norm_scale(x, w, c.rows, c.cols, eps);

        hv::Buffer bx = upload(ctx, std::span<const float>(x));
        hv::Buffer bw = upload(ctx, std::span<const float>(w));
        hv::Buffer by = hv::Buffer::create(ctx, x.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.rms_norm(s, bx, bw, by, c.rows, c.cols, eps);
        s.submit_and_wait();
        const auto y = download<float>(by, x.size());
        // Sum of squares: positive terms -> relative bound on the sum itself; then mean,
        // +eps, sqrt (<= 3 ulp in Vulkan: 1/inversesqrt), 1/x (2.5 ulp), two products. The
        // CPU op has the same shape with an 8-lane sum and correctly rounded sqrt/divide.
        const double rtol = (std::ceil(double(c.cols) / c.wg) + log2u(c.wg) + 16.0) * ref::k_u +
                            (std::ceil(double(c.cols) / k_cpu_lanes) + log2u(k_cpu_lanes) + 16.0) * ref::k_u;
        const auto st = compare(y, to_dbl(y_cpu), [&](std::size_t i) { return rtol * mag[i] + 1e-30; });
        report("rms_norm " + std::to_string(c.rows) + "x" + std::to_string(c.cols) + " wg" + std::to_string(c.wg),
               st);
        EXPECT_LE(st.max_ratio, 1.0) << "rows=" << c.rows << " cols=" << c.cols << " worst=" << st.worst;
    }
}

// ---------------------------------------------------------------- matvec

class VkMatvec : public ::testing::TestWithParam<ref::WType> {};

TEST_P(VkMatvec, MatchesCpuAcrossShapes) {
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
                         ::testing::Values(ref::WType::F32, ref::WType::Q8_0, ref::WType::Q4_K, ref::WType::Q5_K,
                                           ref::WType::Q6_K, ref::WType::IQ4_XS, ref::WType::IQ4_NL,
                                           ref::WType::Q3_K, ref::WType::IQ3_S),
                         [](const auto& info) { return std::string(name_of(info.param)); });

TEST(VkOps, MatvecQ6KBeyondWorkgroupCountLimitUses2DGrid) {
    // LM-head-like row count (248,320 > 65,535) with a small K to keep the test fast.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t rows = 248320;
    // RADV exposes a 2^32-1 X limit, so no allocatable row count exceeds it there; the 2-D
    // grid path is then only exercisable on devices with a smaller limit (e.g. lavapipe).
    if (rows <= ctx->info().max_workgroup_count[0])
        GTEST_SKIP() << "device X limit " << ctx->info().max_workgroup_count[0] << " too large to exercise the 2-D grid";
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

TEST(VkRef, TensorDequantMatchesHandBuiltGgmlBlocks) {
    // The Vulkan tests take their dequantized reference from halo::tensor::dequantize_row;
    // pin it on hand-built blocks. When HALO_VK_DUMP_DIR is set, also write random blocks +
    // that dequantization so an external script can compare against gguf-py.
    // Hand-built Q8_0: d = 0.5, qs = -128..-97.
    std::uint8_t q8[34];
    ref::store_u16(q8, 0x3800);  // 0.5
    for (int i = 0; i < 32; ++i) q8[2 + i] = static_cast<std::uint8_t>(-128 + i);
    std::vector<float> y = ref::dequantize(ref::WType::Q8_0, q8, 32);
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
    y = ref::dequantize(ref::WType::Q4_K, q4, 256);
    // sc = 5 | 3<<4 = 53, m = 2 | 1<<4 = 18 -> 1*53*7 - 2*18 = 335
    EXPECT_EQ(y[128 + 32 + 3], 335.0f);
    // Low-nibble path, sub-block 0 (j < 4 branch): sc = s[0] & 63 = 3, m = s[4] & 63 = 2,
    // element 0 = low nibble of qs[0] = 5 -> 1*3*5 - 2*2 = 11.
    q4[4 + 0] = 3;
    q4[4 + 4] = 2;
    q4[16 + 0] = 0x05;
    y = ref::dequantize(ref::WType::Q4_K, q4, 256);
    EXPECT_EQ(y[0], 11.0f);
    EXPECT_EQ(y[128 + 32 + 3], 335.0f);
    // Hand-built Q6_K: d = 1, scale[8+ (is=0) + 2*quarter=4] ... element n=1, quarter 2, l = 0.
    std::uint8_t q6[210] = {};
    ref::store_u16(q6 + 208, 0x3C00);
    q6[64 + 0] = 0xA0;                          // ql[64+0] high nibble = 10 (quarter >= 2)
    q6[128 + 32 + 0] = 0x30;                    // qh[32+0] bits 4..5 = 3
    q6[192 + 8 + 4] = static_cast<std::uint8_t>(-3);  // scales[8 + 0 + 4] = -3
    y = ref::dequantize(ref::WType::Q6_K, q6, 256);
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

// |vk - cpu| bound after `steps` tokens: the GPU a-priori bound above plus the CPU
// kernel's own bound vs exact arithmetic (Model G, tests/unit/cpu_kernels/tolerance.h).
double gdn_bound(std::uint32_t d_k, std::uint32_t steps, double scale) {
    return gdn_factor(d_k, steps) * scale + ct::tol_gdn(steps, d_k, scale) + ct::kDenormFloor;
}

// halo::cpu::gated_delta_rule_recurrent over one row, state in place, on the same host
// arrays the GPU receives; q/k used as given (D-016 qk_l2norm = false, q_scale = 1).
std::vector<float> cpu_gdn_row(const GdnDims& d, const GdnRow& r, std::vector<float>& state) {
    const std::size_t qc = std::size_t{d.n_k} * d.d_k, vc = std::size_t{d.n_v} * d.d_v, nv = d.n_v;
    std::vector<float> out(vc);
    const hc::GdnInputs in{hc::ConstRows(r.q.data(), 1, qc, qc), hc::ConstRows(r.k.data(), 1, qc, qc),
                           hc::ConstRows(r.v.data(), 1, vc, vc), hc::ConstRows(r.g.data(), 1, nv, nv),
                           hc::ConstRows(r.beta.data(), 1, nv, nv)};
    hc::gated_delta_rule_recurrent(hc::GdnDims{d.n_k, d.n_v, d.d_k, d.d_v, hc::GdnHeadMapping::Tiled}, in, state,
                                   hc::Rows(out.data(), 1, vc, vc),
                                   hc::GdnQkParams{.qk_l2norm = false, .q_scale = 1.0f});
    return out;
}

// fp64 state after one row (bound scale max|S| only).
void scale_step(const GdnDims& d, const GdnRow& r, std::vector<double>& s) {
    ref::gdn_state_step(r.k, r.v, r.g, r.beta, s, d.n_v, d.n_k, d.d_k, d.d_v);
}

struct GdnBuffers {
    hv::Buffer q, k, v, g, beta, out;
};

GdnBuffers gdn_upload(const std::shared_ptr<hv::Context>& ctx, const GdnRow& r, std::size_t out_elems) {
    return {upload(ctx, std::span<const float>(r.q)),    upload(ctx, std::span<const float>(r.k)),
            upload(ctx, std::span<const float>(r.v)),    upload(ctx, std::span<const float>(r.g)),
            upload(ctx, std::span<const float>(r.beta)),
            hv::Buffer::create(ctx, out_elems * 4, hv::MemoryUsage::HostCached)};
}

// View of `buf` starting at float element `elem` (to its end).
hv::BufferView f32_view(const hv::Buffer& buf, std::uint64_t elem) { return {buf, elem * 4}; }

hv::GdnDecodeArgs gdn_args(GdnBuffers& b, hv::Buffer& state, const GdnDims& d, std::uint32_t T) {
    hv::GdnDecodeArgs a;
    a.q = b.q;
    a.k = b.k;
    a.v = b.v;
    a.g = b.g;
    a.beta = b.beta;
    a.state = state;
    a.out = b.out;
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
    std::vector<double> s_ref(s0.begin(), s0.end());  // fp64: bound scale only
    std::vector<float> s_cpu = s0;
    hv::Buffer state = upload(ctx, std::span<const float>(s0), hv::MemoryUsage::HostVisible);
    double worst_out = 0.0, worst_state = 0.0;
    for (std::uint32_t t = 0; t < steps; ++t) {
        const GdnRow r = gdn_row(d, rng, gates);
        const std::vector<float> o_cpu = cpu_gdn_row(d, r, s_cpu);
        scale_step(d, r, s_ref);
        GdnBuffers b = gdn_upload(ctx, r, std::size_t{d.n_v} * d.d_v);
        hv::Stream s(ctx);
        ops.gated_delta_rule_decode(s, gdn_args(b, state, d, 1));
        s.submit_and_wait();
        const auto o = download<float>(b.out, o_cpu.size());
        const double scale_o = max_abs(s_ref) * 1.0;  // |o_b| <= Σ_a |S_ab||q_a| <= max|S| * ||q||_1 <= max|S|
        const double bound = gdn_bound(d.d_k, t + 1, scale_o);
        const auto so = compare(o, to_dbl(o_cpu), [&](std::size_t) { return bound; });
        EXPECT_LE(so.max_ratio, 1.0) << label << " step " << t << " out worst=" << so.worst;
        worst_out = std::max(worst_out, so.max_ratio);
        if (t + 1 == steps) report(label + " out (step " + std::to_string(t + 1) + ")", so);
    }
    const auto st = download<float>(state, d.state_elems());
    const double bound = gdn_bound(d.d_k, steps, max_abs(s_ref));
    const auto ss = compare(st, to_dbl(s_cpu), [&](std::size_t) { return bound; });
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

    // halo::cpu: states after each row, outputs per row (fp64 run for the bound scale only).
    std::vector<double> s_ref(s0.begin(), s0.end());
    std::vector<float> s_cpu = s0;
    std::vector<std::vector<double>> states_after;
    std::vector<double> out_ref;
    for (std::uint32_t t = 0; t < T; ++t) {
        const std::vector<float> o = cpu_gdn_row(d, rows[t], s_cpu);
        scale_step(d, rows[t], s_ref);
        states_after.push_back(to_dbl(s_cpu));
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
    a.state = f32_view(state_in, S);
    a.state_out = f32_view(state_out, 3);

    if (K > 0) a.state_slots = f32_view(slots, 5);

    a.n_slots = K;
    hv::Stream s(ctx);
    ops.gated_delta_rule_decode(s, a);
    s.submit_and_wait();

    const double scale = max_abs(s_ref);
    const double bound = gdn_bound(d.d_k, T, scale);
    const auto o = download<float>(b.out, out_ref.size());
    const auto so = compare(o, out_ref, [&](std::size_t) { return bound; });
    report(label + " out", so);
    EXPECT_LE(so.max_ratio, 1.0) << label;

    std::vector<float> fin(S);
    state_out.download(std::span<float>(fin), 3 * 4);
    const auto sf = compare(fin, states_after.back(), [&](std::size_t) { return bound; });
    report(label + " final state", sf);
    EXPECT_LE(sf.max_ratio, 1.0) << label;

    // Input region untouched (not in place), including the guard region before it.
    EXPECT_EQ(download<float>(state_in, 2 * S), in_host) << label << ": input state was modified";

    const auto slot_data = download<float>(slots, slot_init.size());
    for (std::uint32_t sl = 0; sl < K; ++sl) {
        const std::span<const float> got(slot_data.data() + 5 + std::size_t{sl} * S, S);
        if (sl < T) {
            const auto& expect = states_after[T - 1 - sl];
            const double bs = gdn_bound(d.d_k, T - sl, scale);
            const auto st = compare(got, expect, [&](std::size_t) { return bs; });
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
    std::vector<double> s_ref(s0.begin(), s0.end());
    scale_step(d, r, s_ref);
    std::vector<float> s_cpu = s0;
    (void)cpu_gdn_row(d, r, s_cpu);

    std::vector<float> two(2 * S, 0.0f);
    std::copy(s0.begin(), s0.end(), two.begin());
    hv::Buffer shared = upload(ctx, std::span<const float>(two), hv::MemoryUsage::HostVisible);
    hv::Buffer inplace = upload(ctx, std::span<const float>(s0), hv::MemoryUsage::HostVisible);
    GdnBuffers b1 = gdn_upload(ctx, r, std::size_t{d.n_v} * d.d_v);
    GdnBuffers b2 = gdn_upload(ctx, r, std::size_t{d.n_v} * d.d_v);
    hv::GdnDecodeArgs a1 = gdn_args(b1, shared, d, 1);
    a1.state_out = f32_view(shared, S);
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
    const double bound = gdn_bound(d.d_k, 1, max_abs(s_ref));
    const auto st = compare(ip, to_dbl(s_cpu), [&](std::size_t) { return bound; });
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
    std::vector<double> s_ref(s0.begin(), s0.end());
    std::vector<float> s_cpu = s0;
    std::vector<std::vector<double>> after;
    for (const auto& r : rows) {
        (void)cpu_gdn_row(d, r, s_cpu);
        scale_step(d, r, s_ref);
        after.push_back(to_dbl(s_cpu));
    }
    hv::Buffer arena = hv::Buffer::create(ctx, total * 4, hv::MemoryUsage::HostVisible);
    arena.upload(std::span<const float>(s0), base * 4);
    GdnBuffers b = gdn_upload(ctx, concat(rows), std::size_t{T} * d.n_v * d.d_v);
    hv::GdnDecodeArgs a = gdn_args(b, arena, d, T);
    a.state = f32_view(arena, base);
    a.state_out = f32_view(arena, base + S);
    a.state_slots = f32_view(arena, base + 2 * S);

    a.n_slots = K;
    hv::Stream s(ctx);
    ops.gated_delta_rule_decode(s, a);
    s.submit_and_wait();
    const double bound = gdn_bound(d.d_k, T, max_abs(s_ref));
    auto check = [&](std::uint64_t off, const std::vector<double>& expect, const char* what) {
        std::vector<float> got(S);
        arena.download(std::span<float>(got), off * 4);
        const auto st = compare(got, expect, [&](std::size_t) { return bound; });
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
    a.state_out = f32_view(state, S / 2);  // partial overlap with [0, S)
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.state = f32_view(state, 2 * S + 1);  // input region past the end
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.n_slots = 1;  // slots without a buffer
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.state_slots = f32_view(state, S / 2);  // slots overlapping the state region
    a.n_slots = 1;

    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a.state_slots = f32_view(state, S);  // disjoint in the same buffer: allowed
    EXPECT_NO_THROW(ops.gated_delta_rule_decode(s, a));
    a = base();
    a.state_slots = slots;
    a.n_slots = 3;  // only min(T, n_slots) = 2 slots are written -> fits in 2*S
    EXPECT_NO_THROW(ops.gated_delta_rule_decode(s, a));
    a = base();
    a.n_tokens = 3;  // inputs sized for T = 2
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.out = state;  // output aliasing state
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    a = base();
    a.d_k = 512;  // > gdn_max_dk
    EXPECT_THROW(ops.gated_delta_rule_decode(s, a), halo::Error);
    s.submit_and_wait();
}

// ---------------------------------------------------------------- argmax

namespace {

// One argmax dispatch; returns the three raw result words {index, value bits, nan}.
std::array<std::uint32_t, 3> gpu_argmax_words(const std::shared_ptr<hv::Context>& ctx, hv::Ops& ops,
                                              std::span<const float> x, bool print_timing = false) {
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
    std::array<std::uint32_t, 3> w{};
    result.download(std::span<std::uint32_t>(w));
    return w;
}

// The host API path: decode_argmax raises Error(Kernel) on the NaN word (D-016).
std::uint32_t gpu_argmax(const std::shared_ptr<hv::Context>& ctx, hv::Ops& ops, std::span<const float> x,
                         float* value = nullptr, bool print_timing = false) {
    const hv::ArgmaxResult r = hv::decode_argmax(gpu_argmax_words(ctx, ops, x, print_timing));
    if (value) *value = r.value;
    return r.index;
}

}  // namespace

namespace {
// cpu::argmax index as the GPU's uint32 index word.
std::uint32_t cpu_argmax(std::span<const float> x) { return static_cast<std::uint32_t>(hc::argmax(x).index); }
}  // namespace

TEST(VkArgmax, VocabSizedRowMatchesCpuExactly) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint32_t n = 248320;
    for (const std::uint32_t wg : {256u, 64u}) {
        hv::Ops ops(ctx, hv::OpsOptions{.reduce_workgroup = wg});
        std::cout << "[vk-argmax] wg=" << wg << " partials=" << ops.argmax_partials(n) << "\n";
        EXPECT_GT(ops.argmax_partials(n), 1u);
        std::mt19937 rng(123 + wg);
        auto x = ref::random_vec(n, rng, 4.0f);
        float v = 0;
        EXPECT_EQ(gpu_argmax(ctx, ops, x, &v, wg == 256), cpu_argmax(x));
        EXPECT_EQ(v, hc::argmax(x).value);

        const float big = 1000.0f;
        auto check_at = [&](std::vector<std::uint32_t> at, std::uint32_t expect, const char* what) {
            auto y = x;
            for (auto i : at) y[i] = big;
            EXPECT_EQ(cpu_argmax(y), expect) << what << " (halo::cpu)";
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
    // Every non-NaN case: the GPU index equals halo::cpu::argmax on the same array.
    auto same = [&](const std::vector<float>& x, std::uint32_t expect, const char* what) {
        EXPECT_EQ(cpu_argmax(x), expect) << what << " (halo::cpu)";
        EXPECT_EQ(gpu_argmax(ctx, ops, x), expect) << what;
    };
    same(std::vector<float>(100000, 2.5f), 0u, "all equal: first index");
    std::vector<float> neg_inf(10000, -inf);
    same(neg_inf, 0u, "all -inf: first index");
    neg_inf[7777] = -1e30f;
    same(neg_inf, 7777u, "one finite among -inf");
    std::vector<float> pos_inf(300, 1.0f);
    pos_inf[299] = inf;
    pos_inf[150] = inf;
    same(pos_inf, 150u, "+inf tie -> lowest index");
    same(std::vector<float>{-3.0f}, 0u, "single element");

    // NaN: both backends raise Error(Kernel) (D-016, code review S-3). Differential: for
    // every array, the GPU path throws exactly when halo::cpu::argmax does.
    auto nan_case = [&](const std::vector<float>& x, const std::string& what) {
        EXPECT_THROW((void)hc::argmax(x), halo::Error) << what << " (halo::cpu)";
        EXPECT_THROW((void)gpu_argmax(ctx, ops, x), halo::Error) << what;
        EXPECT_EQ(gpu_argmax_words(ctx, ops, x)[2], 1u) << what << ": NaN word";
    };
    std::vector<float> with_nan(50000, 0.0f);
    with_nan[10] = nan;
    with_nan[20] = 1.0f;
    with_nan[30] = nan;
    nan_case(with_nan, "NaN next to the max, same partial");
    with_nan.assign(9000, nan);
    with_nan[8999] = -inf;
    nan_case(with_nan, "only non-NaN element is -inf");
    nan_case(std::vector<float>(9000, nan), "all NaN");
    nan_case(std::vector<float>{nan}, "single NaN");
    for (const std::size_t at : {std::size_t{0}, std::size_t{4095}, std::size_t{4096}, std::size_t{49999}}) {
        std::vector<float> y(50000, 0.5f);
        y[123] = 9.0f;  // the max sits in partial 0; the NaN in partial 0, 0, 1 and the last one
        y[at] = nan;
        nan_case(y, "one NaN at " + std::to_string(at));
    }
    std::vector<float> neg_nan(5000, 1.0f);
    neg_nan[777] = -nan;  // sign bit set
    nan_case(neg_nan, "negative NaN");
}

TEST(VkArgmax, NanFlagAcrossWorkgroupSizesAndResultReuse) {
    // The NaN word is OR-reduced through both passes at every workgroup size, and every
    // dispatch rewrites it: a result buffer that held a NaN result reads clean afterwards.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::uint32_t n = 248320;
    for (const std::uint32_t wg : {32u, 64u, 256u}) {
        hv::Ops ops(ctx, hv::OpsOptions{.reduce_workgroup = wg});
        std::mt19937 rng(700 + wg);
        const auto x = ref::random_vec(n, rng, 4.0f);
        auto bad = x;
        bad[n - 1 - wg] = nan;  // late partial, not the max's
        hv::Buffer bx = upload(ctx, std::span<const float>(x));
        hv::Buffer bbad = upload(ctx, std::span<const float>(bad));
        hv::Buffer scratch = hv::Buffer::create(ctx, ops.argmax_scratch_bytes(n), hv::MemoryUsage::DeviceLocal);
        hv::Buffer result = hv::Buffer::create(ctx, hv::k_argmax_result_bytes, hv::MemoryUsage::HostCached);
        for (int round = 0; round < 2; ++round) {
            hv::Stream s1(ctx);
            ops.argmax(s1, bbad, n, scratch, result);
            s1.submit_and_wait();
            EXPECT_THROW((void)hc::argmax(bad), halo::Error);
            EXPECT_THROW((void)hv::read_argmax(result), halo::Error) << "wg=" << wg << " round " << round;
            hv::Stream s2(ctx);
            ops.argmax(s2, bx, n, scratch, result);
            s2.submit_and_wait();
            const hv::ArgmaxResult r = hv::read_argmax(result);
            EXPECT_EQ(r.index, cpu_argmax(x)) << "wg=" << wg << " round " << round;
            EXPECT_EQ(r.value, hc::argmax(x).value);
        }
    }
}

TEST(VkArgmax, DecodeArgmaxRaisesOnNanWordAndMissingIndex) {
    // Host decoding alone (no device needed).
    const std::array<std::uint32_t, 3> ok{7u, std::bit_cast<std::uint32_t>(2.5f), 0u};
    const hv::ArgmaxResult r = hv::decode_argmax(ok);
    EXPECT_EQ(r.index, 7u);
    EXPECT_EQ(r.value, 2.5f);
    EXPECT_THROW((void)hv::decode_argmax(std::array<std::uint32_t, 3>{7u, 0u, 1u}), halo::Error);
    EXPECT_THROW((void)hv::decode_argmax(std::array<std::uint32_t, 3>{hv::k_argmax_none, 0u, 0u}), halo::Error);
    try {
        (void)hv::decode_argmax(std::array<std::uint32_t, 3>{7u, 0u, 1u});
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Kernel);
    }
}

TEST(VkArgmax, ValidatesBuffers) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops64(ctx, hv::OpsOptions{.reduce_workgroup = 64});  // chunk 1024 -> 4 partials for n=4000
    ASSERT_EQ(ops64.argmax_partials(4000), 4u);
    hv::Buffer x = hv::Buffer::create(ctx, 4000 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer small = hv::Buffer::create(ctx, 8, hv::MemoryUsage::DeviceLocal);
    hv::Buffer scratch = hv::Buffer::create(ctx, ops64.argmax_scratch_bytes(4000), hv::MemoryUsage::DeviceLocal);
    hv::Buffer result = hv::Buffer::create(ctx, hv::k_argmax_result_bytes, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    EXPECT_THROW(ops64.argmax(s, x, 4000, scratch, small), halo::Error);      // result < 12 bytes (no NaN word)
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
    // halo::cpu on the same arrays: rms_norm -> matmul (tensor-dequantized weights).
    std::vector<float> xn_cpu(cols);
    hc::rms_norm(hc::ConstRows(x.data(), 1, cols, cols), nw, 1e-6f, hc::Rows(xn_cpu.data(), 1, cols, cols));
    const std::vector<double> logits_cpu = to_dbl(cpu_matvec(w, ref::WType::Q8_0, rows, cols, xn_cpu));
    const std::vector<double> scale = ref::matvec_scale(w.mag, xn_cpu, rows, cols);

    hv::Buffer bx = upload(ctx, std::span<const float>(x));
    hv::Buffer bn = upload(ctx, std::span<const float>(nw));
    hv::Buffer bxn = hv::Buffer::create(ctx, cols * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
    hv::Buffer blog = hv::Buffer::create(ctx, rows * 4, hv::MemoryUsage::HostCached);
    hv::Buffer scratch = hv::Buffer::create(ctx, ops.argmax_scratch_bytes(rows), hv::MemoryUsage::DeviceLocal);
    hv::Buffer result = hv::Buffer::create(ctx, hv::k_argmax_result_bytes, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    ops.rms_norm(s, bx, bn, bxn, 1, cols, 1e-6f);
    ops.matvec(s, halo::DType::Q8_0, bw, bxn, blog, rows, cols);
    ops.argmax(s, blog, rows, scratch, result);
    s.submit_and_wait();
    EXPECT_EQ(s.dispatch_count(), 4u);
    const auto logits = download<float>(blog, rows);
    // Compare the GPU argmax with halo::cpu::argmax of the GPU logits (exact), and the
    // logits with the CPU logits within both matvec bounds widened by both rms_norm
    // relative input errors (see RmsNormMatchesCpu; the normalized input enters linearly).
    EXPECT_EQ(hv::read_argmax(result).index, cpu_argmax(logits));
    const double rms_rel = (std::ceil(double(cols) / 256) + log2u(256) + 16.0) * ref::k_u +
                           (std::ceil(double(cols) / k_cpu_lanes) + log2u(k_cpu_lanes) + 16.0) * ref::k_u;
    const double f = sum_bound_factor(cols, 256) + sum_bound_factor(cols, k_cpu_lanes) + rms_rel;
    const auto st = compare(logits, logits_cpu, [&](std::size_t i) { return f * scale[i] + ct::kDenormFloor; });
    report("chained norm->q8_0 matvec", st);
    EXPECT_LE(st.max_ratio, 1.0);
}
