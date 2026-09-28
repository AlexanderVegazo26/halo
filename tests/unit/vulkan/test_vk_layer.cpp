// Differential tests of the Vulkan layer operators (WS-F2 V1) against halo::cpu on the SAME
// host arrays (TRD §30; code review M-2). Verified on lavapipe only (DECISIONS D-001).
//
// Where the Vulkan kernel does only IEEE +, *, sqrt-free arithmetic in the CPU op's order
// (uncontracted, `precise`), the result must be BIT-IDENTICAL:
//   ADD, the h of ADD+RMS_NORM, PARTIAL_ROPE (host cos/sin table), the conv1d state and
//   slots (copies of inputs).
// Where a transcendental is involved, the bound is per element, relative to the CPU value:
//   |vk - cpu| <= f * u * |cpu| + FLT_MIN   (u = 2^-24; FLT_MIN absorbs a device that flushes
//                                            denormal results)
// with f from Vulkan's precision rules (exp: 3 + 2|x| ULP; division: 2.5 ULP; + - *
// correctly rounded) plus the CPU's (libm exp / log1p <= 1 ULP), term by term:
//   sigmoid(x) = 1/(1+exp(-x)):  vk 3+2|x| (exp) + 1 (add) + 2.5 (div)  + cpu 3        -> 10 + 2|x|  (12 used)
//   silu(x)    = x/(1+exp(-x)):  same                                                   -> 12 + 2|x|
//   swiglu     = silu(g)*up, mul_sigmoid = a*sigmoid(g): + 1 (vk mul) + 1 (cpu mul)     -> 14 + 2|g|
//   softplus(s) = log1p(exp(s)): exp 3+2|s| (the condition of log1p(e) in e is <= 1), plus
//     halo_log1p (series: 24 Horner steps ~ 6 ULP; log branch: 2^-21 absolute on a result
//     >= log 1.5 ~ 20 ULP) + cpu 2                                                      -> 26 + 2|s|
//   g = ssm_a * softplus(s): + 2 (one mul each side)  (s = a + dt is the same IEEE add)  -> 28 + 2|s| (32 used)
//   gated norm: the rms_norm bound of test_vk_ops (reduction of cols squares on both
//     sides + sqrt/div) + silu(z) (12 + 2|z|) + 3 muls per side                        -> rtol + (18 + 2|z|)
//   conv1d out = silu(acc), acc bit-identical                                           -> 12 + 2|acc|
// Every bound is fixed a priori; observed ratios are printed.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cpu_kernels/reference.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "vk_test_util.h"

#pragma GCC diagnostic ignored "-Wdouble-promotion"

namespace hv = halo::vulkan;
namespace hc = halo::cpu;
namespace ct = halo::cpu::test;
using hv::test::download;
using hv::test::upload;

namespace {

constexpr double k_u = 0x1p-24;
constexpr double k_floor = 1.1754943508222875e-38;  // FLT_MIN
constexpr std::uint32_t k_cpu_lanes = 8;

bool bitwise(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

/// max over i of |vk - cpu| / (f(i) * u * |cpu| + floor); EXPECTs <= 1; prints the ratio.
template <class F>
double check_rel(std::span<const float> vk, std::span<const float> cpu, F f, const std::string& what) {
    EXPECT_EQ(vk.size(), cpu.size()) << what;
    double worst = 0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < std::min(vk.size(), cpu.size()); ++i) {
        const double d = std::abs(double(vk[i]) - double(cpu[i]));
        const double tol = f(i) * k_u * std::abs(double(cpu[i])) + k_floor;
        const double r = std::isfinite(double(vk[i])) ? d / tol : std::numeric_limits<double>::infinity();
        if (r > worst) {
            worst = r;
            at = i;
        }
    }
    EXPECT_LE(worst, 1.0) << what << ": worst at " << at << " vk " << vk[at] << " cpu " << cpu[at];
    std::cout << "[vk-layer] " << what << ": worst |vk-cpu|/bound = " << worst << "\n";
    return worst;
}

double rms_rtol(std::uint32_t cols, std::uint32_t wg) {
    return (std::ceil(double(cols) / wg) + std::log2(double(wg)) + 16.0) +
           (std::ceil(double(cols) / k_cpu_lanes) + std::log2(double(k_cpu_lanes)) + 16.0);
}

hc::ConstRows crows(const std::vector<float>& v, std::size_t rows, std::size_t cols, std::size_t stride = 0) {
    return {v.data(), rows, cols, stride == 0 ? cols : stride};
}
hc::Rows mrows(std::vector<float>& v, std::size_t rows, std::size_t cols, std::size_t stride = 0) {
    return {v.data(), rows, cols, stride == 0 ? cols : stride};
}

}  // namespace

// ---------------------------------------------------------------- CONV1D_SHORT

namespace {

void conv_case(const std::shared_ptr<hv::Context>& ctx, std::uint32_t T, std::uint32_t C, std::uint32_t K,
               std::uint32_t n_slots, bool in_place, std::uint64_t seed) {
    const std::string what = "conv1d T=" + std::to_string(T) + " C=" + std::to_string(C) + " K=" + std::to_string(K) +
                             " slots=" + std::to_string(n_slots) + (in_place ? " in-place" : "");
    hv::Ops ops(ctx);
    ct::Rng rng(seed);
    const std::vector<float> x = rng.normal(std::size_t{T} * C, 1.5f);
    const std::vector<float> w = rng.normal(std::size_t{C} * K, 0.6f);
    const std::size_t hist = K - 1;
    const std::vector<float> s0 = rng.normal(hist * C, 1.0f);
    const float sentinel = 99.0f;
    // CPU on the same arrays.
    std::vector<float> st_cpu = s0, out_cpu(x.size()), sl_cpu(std::size_t{n_slots} * hist * C, sentinel);
    hc::causal_conv1d_silu(crows(x, T, C), crows(w, C, K), mrows(st_cpu, hist, C), mrows(out_cpu, T, C), nullptr,
                           sl_cpu);
    // Vulkan.
    hv::Buffer bx = upload(ctx, std::span<const float>(x), hv::MemoryUsage::HostCached);
    hv::Buffer bw = upload(ctx, std::span<const float>(w));
    hv::Buffer bs = upload(ctx, std::span<const float>(s0.empty() ? std::vector<float>{0.0f} : s0),
                           hv::MemoryUsage::HostCached);
    hv::Buffer bo = hv::Buffer::create(ctx, x.size() * 4, hv::MemoryUsage::HostCached);
    const std::vector<float> sl_init(std::max<std::size_t>(sl_cpu.size(), 1), sentinel);
    hv::Buffer bsl = upload(ctx, std::span<const float>(sl_init), hv::MemoryUsage::HostCached);
    hv::Conv1dArgs a;
    a.x = bx;
    a.weight = bw;
    if (hist > 0) a.conv_state = bs;
    a.out = in_place ? hv::BufferView(bx) : hv::BufferView(bo);
    if (n_slots > 0) a.state_slots = bsl;
    a.n_tokens = T;
    a.channels = C;
    a.kernel = K;
    a.n_slots = n_slots;
    hv::Stream s(ctx);
    ops.causal_conv1d_silu(s, a);
    s.submit_and_wait();
    const auto out = download<float>(in_place ? bx : bo, x.size());
    // |acc| (fp64, error-bound scale only).
    std::vector<double> acc(x.size());
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t c = 0; c < C; ++c) {
            double v = 0;
            for (std::size_t j = 0; j < K; ++j) {
                const std::ptrdiff_t src = std::ptrdiff_t(t) - std::ptrdiff_t(hist) + std::ptrdiff_t(j);
                const double in = src >= 0 ? x[std::size_t(src) * C + c] : s0[std::size_t(std::ptrdiff_t(hist) + src) * C + c];
                v += double(w[c * K + j]) * in;
            }
            acc[t * C + c] = std::abs(v);
        }
    check_rel(out, out_cpu, [&](std::size_t i) { return 12.0 + 2.0 * acc[i] * 1.001 + 1.0; }, what + " out");
    if (hist > 0) {
        EXPECT_TRUE(bitwise(download<float>(bs, s0.size()), st_cpu)) << what << ": conv_state not bit-identical";
    }
    if (n_slots > 0) {
        EXPECT_TRUE(bitwise(download<float>(bsl, sl_cpu.size()), sl_cpu))
            << what << ": slots not bit-identical (incl. untouched slots >= T)";
    }
}

}  // namespace

TEST(VkLayer, Conv1dSiluMatchesCpuStateAndSlotsBitwise) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    conv_case(ctx, 1, 10240, 4, 0, false, 1);   // decode, real channels
    conv_case(ctx, 5, 10240, 4, 4, false, 2);   // MTP verify with slots
    conv_case(ctx, 3, 300, 4, 6, false, 3);     // slots beyond T untouched
    conv_case(ctx, 70, 257, 4, 0, true, 4);     // prefill-like, out aliases x
    conv_case(ctx, 4, 64, 8, 2, false, 5);      // max kernel
    conv_case(ctx, 3, 50, 1, 0, false, 6);      // kernel 1 (no state)
}

namespace {

/// Ring form (ADR-001 §5.3) vs the legacy conv_state + slots form: bit-identical values at
/// modular slab positions; the input slot and unwritten physical slots stay untouched.
void conv_ring_case(const std::shared_ptr<hv::Context>& ctx, std::uint32_t T, std::uint32_t C, std::uint32_t K,
                    std::uint32_t n_slots, std::uint32_t P, std::uint32_t live, std::uint64_t seed) {
    const std::string what = "conv1d ring T=" + std::to_string(T) + " C=" + std::to_string(C) + " K=" +
                             std::to_string(K) + " slots=" + std::to_string(n_slots) + " P=" + std::to_string(P) +
                             " live=" + std::to_string(live);
    hv::Ops ops(ctx);
    ct::Rng rng(seed);
    const std::vector<float> x = rng.normal(std::size_t{T} * C, 1.5f);
    const std::vector<float> w = rng.normal(std::size_t{C} * K, 0.6f);
    const std::size_t hist = K - 1;
    const std::size_t sn = hist * C;
    const std::vector<float> s0 = rng.normal(sn, 1.0f);
    const float sentinel = 99.0f;
    // CPU reference: the ring overload over a host slab (bit-identical to the slot form,
    // tests/unit/cpu_kernels) — the Vulkan kernel must match it bit for bit (state/slots are
    // pure copies; the conv accumulation order is shared).
    std::vector<float> slab_cpu(std::size_t{P} * sn, sentinel);
    std::copy(s0.begin(), s0.end(), slab_cpu.begin() + static_cast<std::ptrdiff_t>(live * sn));
    std::vector<float> out_cpu(x.size());
    hc::causal_conv1d_silu(crows(x, T, C), crows(w, C, K),
                           hc::StateRingView{std::span<float>(slab_cpu), P, live, n_slots}, mrows(out_cpu, T, C));
    // Vulkan.
    hv::Buffer bx = upload(ctx, std::span<const float>(x), hv::MemoryUsage::HostCached);
    hv::Buffer bw = upload(ctx, std::span<const float>(w));
    hv::Buffer bs = upload(ctx, std::span<const float>(slab_cpu), hv::MemoryUsage::HostCached);
    hv::Buffer bo = hv::Buffer::create(ctx, x.size() * 4, hv::MemoryUsage::HostCached);
    hv::Conv1dArgs a;
    a.x = bx;
    a.weight = bw;
    if (hist > 0) a.conv_state = bs;
    a.out = bo;
    a.n_tokens = T;
    a.channels = C;
    a.kernel = K;
    a.n_slots = n_slots;
    a.ring = hv::GdnRing{P, live};
    hv::Stream s(ctx);
    ops.causal_conv1d_silu(s, a);
    s.submit_and_wait();
    // out uses the device exp in SiLU (same bound as the legacy form's test); the state
    // placement is pure copies and must be bit-identical to the CPU ring overload's slab.
    const auto out = download<float>(bo, x.size());
    std::vector<double> acc(x.size());
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t c = 0; c < C; ++c) {
            double v = 0;
            for (std::size_t j = 0; j < K; ++j) {
                const std::ptrdiff_t src = std::ptrdiff_t(t) - std::ptrdiff_t(hist) + std::ptrdiff_t(j);
                const double in = src >= 0 ? x[std::size_t(src) * C + c] : s0[std::size_t(std::ptrdiff_t(hist) + src) * C + c];
                v += double(w[c * K + j]) * in;
            }
            acc[t * C + c] = std::abs(v);
        }
    check_rel(out, out_cpu, [&](std::size_t i) { return 12.0 + 2.0 * acc[i] * 1.001 + 1.0; }, what + " out");
    if (hist > 0) {
        EXPECT_TRUE(bitwise(download<float>(bs, slab_cpu.size()), slab_cpu)) << what << ": slab not bit-identical";
    }
}

}  // namespace

TEST(VkLayer, Conv1dRingMatchesCpuBitwise) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    conv_ring_case(ctx, 1, 10240, 4, 0, 2, 0, 11);     // decode, real channels
    conv_ring_case(ctx, 1, 300, 4, 0, 2, 1, 12);       // decode at live=1 (wrap)
    conv_ring_case(ctx, 5, 10240, 4, 4, 5, 3, 13);     // verify with slots
    conv_ring_case(ctx, 3, 300, 4, 5, 6, 5, 14);       // slots past T untouched, wrap
    conv_ring_case(ctx, 70, 257, 4, 0, 3, 2, 15);      // prefill-like
}

TEST(VkLayer, Conv1dRingValidation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer x = hv::Buffer::create(ctx, 4 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer w = hv::Buffer::create(ctx, 64 * 4 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer slab = hv::Buffer::create(ctx, 3 * 3 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer o = hv::Buffer::create(ctx, 4 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    auto base = [&] {
        hv::Conv1dArgs a;
        a.x = x;
        a.weight = w;
        a.conv_state = slab;
        a.out = o;
        a.n_tokens = 4;
        a.channels = 64;
        a.kernel = 4;
        a.ring = hv::GdnRing{3, 0};
        return a;
    };
    EXPECT_NO_THROW(ops.causal_conv1d_silu(s, base()));
    auto a = base();
    a.ring = hv::GdnRing{1, 0};  // P < 2
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.ring = hv::GdnRing{3, 3};  // live >= P
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.n_slots = 3;  // slots > P-1
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.state_slots = slab;  // slots view must be unset with a ring
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    s.submit_and_wait();
}

TEST(VkLayer, Conv1dValidation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer x = hv::Buffer::create(ctx, 4 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer w = hv::Buffer::create(ctx, 64 * 4 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer st = hv::Buffer::create(ctx, 3 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer o = hv::Buffer::create(ctx, 4 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    auto base = [&] {
        hv::Conv1dArgs a;
        a.x = x;
        a.weight = w;
        a.conv_state = st;
        a.out = o;
        a.n_tokens = 4;
        a.channels = 64;
        a.kernel = 4;
        return a;
    };
    EXPECT_NO_THROW(ops.causal_conv1d_silu(s, base()));
    auto a = base();
    a.kernel = 9;
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.out = hv::BufferView(x, 4, 4 * 64 * 4 - 4);  // partial overlap with x
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.conv_state = hv::BufferView(o, 0, 3 * 64 * 4);  // state overlaps out
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.n_slots = 2;  // no slots view
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    a = base();
    a.n_tokens = 5;  // x too small
    EXPECT_THROW(ops.causal_conv1d_silu(s, a), halo::Error);
    s.submit_and_wait();
}

// ---------------------------------------------------------------- GATED_NORM, per-head RMS_NORM

TEST(VkLayer, GatedRmsNormMatchesCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const bool in_place : {false, true}) {
        hv::Ops ops(ctx);
        const std::uint32_t rows = 3 * 48, cols = 128;  // T = 3 tokens x 48 value heads, d_v 128
        ct::Rng rng(20 + in_place);
        auto x = rng.normal(std::size_t{rows} * cols, 2.0f);
        // Rows 0..7 are tiny (mean x^2 ~ 1e-6, the size of eps), so eps changes them by
        // tens of percent: a kernel that drops or misplaces eps is visible (it is not on
        // O(1) rows, where eps moves the result by ~2 ULP).
        for (std::size_t i = 0; i < 8 * std::size_t{cols}; ++i) x[i] *= 5e-4f;
        const auto z = rng.normal(std::size_t{rows} * cols, 3.0f);
        const auto w = rng.normal(cols, 1.0f);
        std::vector<float> cpu(x.size());
        hc::gated_rms_norm(crows(x, rows, cols), w, crows(z, rows, cols), 1e-6f, mrows(cpu, rows, cols));
        hv::Buffer bx = upload(ctx, std::span<const float>(x), hv::MemoryUsage::HostCached);
        hv::Buffer bz = upload(ctx, std::span<const float>(z));
        hv::Buffer bw = upload(ctx, std::span<const float>(w));
        hv::Buffer bo = hv::Buffer::create(ctx, x.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.gated_rms_norm(s, hv::GatedNormArgs{bx, bz, bw, in_place ? hv::BufferView(bx) : hv::BufferView(bo), rows,
                                                cols, 1e-6f});
        s.submit_and_wait();
        const auto got = download<float>(in_place ? bx : bo, x.size());
        const double rt = rms_rtol(cols, ops.options().reduce_workgroup);
        check_rel(got, cpu, [&](std::size_t i) { return rt + 18.0 + 2.0 * std::abs(double(z[i])); },
                  std::string("gated_rms_norm 144x128") + (in_place ? " in place (out = x)" : ""));
    }
}

TEST(VkLayer, PerHeadQkRmsNormOnInterleavedQGateRows) {
    // attn_q output rows are [n_head x (256 Q | 256 gate)] (D-004). Per-head q-norm reads the
    // Q halves as rows = T * n_head of 256 with row stride 512 (no de-interleave copy); the
    // k-norm reads dense [n_kv x 256] rows. Output into distinct buffers (ADR-001 §5.2).
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t T = 3, nh = 24, nkv = 4, hd = 256;
    ct::Rng rng(30);
    const auto qg = rng.normal(std::size_t{T} * nh * 2 * hd, 2.0f);
    const auto k = rng.normal(std::size_t{T} * nkv * hd, 2.0f);
    const auto wq = rng.normal(hd), wk = rng.normal(hd);
    std::vector<float> q_cpu(std::size_t{T} * nh * hd), k_cpu(k.size());
    hc::rms_norm(hc::ConstRows(qg.data(), T * nh, hd, 2 * hd), wq, 1e-6f, mrows(q_cpu, T * nh, hd));
    hc::rms_norm(crows(k, T * nkv, hd), wk, 1e-6f, mrows(k_cpu, T * nkv, hd));
    hv::Buffer bqg = upload(ctx, std::span<const float>(qg));
    hv::Buffer bk = upload(ctx, std::span<const float>(k));
    hv::Buffer bwq = upload(ctx, std::span<const float>(wq)), bwk = upload(ctx, std::span<const float>(wk));
    hv::Buffer bq_out = hv::Buffer::create(ctx, q_cpu.size() * 4, hv::MemoryUsage::HostCached);
    hv::Buffer bk_out = hv::Buffer::create(ctx, k_cpu.size() * 4, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    ops.rms_norm(s, hv::BufferView(bqg, 0, 0, 2 * hd * 4), bwq, bq_out, T * nh, hd, 1e-6f);
    ops.rms_norm(s, bk, bwk, bk_out, T * nkv, hd, 1e-6f);
    s.submit_and_wait();
    const double rt = rms_rtol(hd, ops.options().reduce_workgroup);
    check_rel(download<float>(bq_out, q_cpu.size()), q_cpu, [&](std::size_t) { return rt; },
              "q-norm on interleaved [Q|gate] (stride 512)");
    check_rel(download<float>(bk_out, k_cpu.size()), k_cpu, [&](std::size_t) { return rt; }, "k-norm dense");
}

// ---------------------------------------------------------------- PARTIAL_ROPE

namespace {

void rope_case(const std::shared_ptr<hv::Context>& ctx, std::uint32_t T, std::uint32_t nh, std::uint32_t hd,
               std::uint32_t rot, std::uint32_t head_stride, float theta, const std::vector<std::int32_t>& pos,
               const std::string& what) {
    hv::Ops ops(ctx);
    ct::Rng rng(40 + T + nh + head_stride);
    const std::uint32_t hs = head_stride == 0 ? hd : head_stride;
    const std::size_t row = std::size_t{nh} * hs;
    const auto x0 = rng.normal(T * row, 2.0f);
    // CPU: its views are dense per head, so apply it head by head on a [T, head_dim] view
    // with row stride = the interleaved row (one head at a time, n_heads = 1). This is the
    // same per-element operation (the CPU op is element-wise per (token, head, pair)).
    std::vector<float> cpu = x0;
    for (std::uint32_t h = 0; h < nh; ++h) {
        hc::partial_rope_neox(hc::Rows(cpu.data() + h * hs, T, hd, row), 1, hd, pos, rot, theta);
    }
    hv::Buffer bx = upload(ctx, std::span<const float>(x0), hv::MemoryUsage::HostCached);
    const std::vector<float> table = hv::rope_cos_sin_table(pos, rot, theta);
    hv::Buffer bt = upload(ctx, std::span<const float>(table));
    hv::RopeArgs a;
    a.x = hv::BufferView(bx, 0, 0, row * 4);
    a.cos_sin = bt;
    a.n_tokens = T;
    a.n_heads = nh;
    a.head_dim = hd;
    a.rot_dims = rot;
    a.head_stride = head_stride;
    hv::Stream s(ctx);
    ops.partial_rope_neox(s, a);
    s.submit_and_wait();
    const auto got = download<float>(bx, x0.size());
    EXPECT_TRUE(bitwise(got, cpu)) << what << ": not bit-identical to cpu::partial_rope_neox";
    std::size_t diff = 0;
    for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != cpu[i];
    std::cout << "[vk-layer] " << what << ": " << diff << " elements differ from halo::cpu\n";
}

}  // namespace

TEST(VkLayer, PartialRopeWithHeadStrideIsBitIdenticalToCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::vector<std::int32_t> pos{0, 1, 7, 1000, 32767, 131071, 262143};
    // qwen35 Q in place inside the interleaved [Q | gate] row: head_dim 256, stride 512,
    // rot 64, theta 1e7 (D-004). The gate halves must be untouched (checked by bitwise
    // equality with the CPU run, which never touches them either).
    rope_case(ctx, 7, 24, 256, 64, 512, 1e7f, pos, "rope Q in [Q|gate] rows (24 heads, stride 512)");
    rope_case(ctx, 7, 4, 256, 64, 0, 1e7f, pos, "rope K dense (4 heads)");
    rope_case(ctx, 3, 2, 64, 64, 96, 10000.0f, {5, 6, 900}, "rope full rotation, stride 96");
}

TEST(VkLayer, RopeTableEqualsCpuInvFreqAngles) {
    // The host table is the CPU op's own fp32 angle / double cos-sin (checked against
    // cpu::rope_inv_freq directly).
    const std::vector<std::int32_t> pos{0, 3, 131071};
    const auto t = hv::rope_cos_sin_table(pos, 64, 1e7f);
    const auto inv = hc::rope_inv_freq(64, 1e7f);
    for (std::size_t p = 0; p < pos.size(); ++p)
        for (std::size_t i = 0; i < 32; ++i) {
            const float angle = static_cast<float>(pos[p]) * inv[i];
            EXPECT_EQ(t[p * 64 + i], static_cast<float>(std::cos(static_cast<double>(angle))));
            EXPECT_EQ(t[p * 64 + 32 + i], static_cast<float>(std::sin(static_cast<double>(angle))));
        }
    EXPECT_THROW((void)hv::rope_cos_sin_table(pos, 63, 1e7f), halo::Error);
}

// ---------------------------------------------------------------- SWIGLU, MUL_SIGMOID, ADD

TEST(VkLayer, SwigluMulSigmoidAddMatchCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t T = 3;
    ct::Rng rng(50);
    {  // SWIGLU over the FFN width, out distinct and out = gate in place.
        const std::uint32_t cols = 17408;
        const auto gate = rng.uniform(std::size_t{T} * cols, -12.0f, 12.0f);
        const auto up = rng.normal(std::size_t{T} * cols, 2.0f);
        std::vector<float> cpu(gate.size());
        hc::swiglu(crows(gate, T, cols), crows(up, T, cols), mrows(cpu, T, cols));
        for (const bool in_place : {false, true}) {
            hv::Buffer bg = upload(ctx, std::span<const float>(gate), hv::MemoryUsage::HostCached);
            hv::Buffer bu = upload(ctx, std::span<const float>(up));
            hv::Buffer bo = hv::Buffer::create(ctx, gate.size() * 4, hv::MemoryUsage::HostCached);
            hv::Stream s(ctx);
            ops.swiglu(s, hv::EltwiseArgs{bg, bu, in_place ? hv::BufferView(bg) : hv::BufferView(bo), T, cols});
            s.submit_and_wait();
            check_rel(download<float>(in_place ? bg : bo, gate.size()), cpu,
                      [&](std::size_t i) { return 14.0 + 2.0 * std::abs(double(gate[i])); },
                      std::string("swiglu 3x17408") + (in_place ? " in place" : ""));
        }
    }
    {  // MUL_SIGMOID with the gate read in place from the interleaved [Q | gate] rows.
        const std::uint32_t nh = 24, hd = 256;
        const auto att = rng.normal(std::size_t{T} * nh * hd, 1.0f);
        const auto qg = rng.uniform(std::size_t{T} * nh * 2 * hd, -10.0f, 10.0f);
        std::vector<float> cpu(att.size());
        hc::mul_sigmoid(crows(att, T * nh, hd), hc::ConstRows(qg.data() + hd, T * nh, hd, 2 * hd),
                        mrows(cpu, T * nh, hd));
        hv::Buffer ba = upload(ctx, std::span<const float>(att));
        hv::Buffer bq = upload(ctx, std::span<const float>(qg));
        hv::Buffer bo = hv::Buffer::create(ctx, att.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.mul_sigmoid(s, hv::EltwiseArgs{ba, hv::BufferView(bq, hd * 4, 0, 2 * hd * 4), bo, T * nh, hd});
        s.submit_and_wait();
        check_rel(download<float>(bo, att.size()), cpu,
                  [&](std::size_t i) {
                      const std::size_t r = i / hd, c = i % hd;
                      return 14.0 + 2.0 * std::abs(double(qg[r * 2 * hd + hd + c]));
                  },
                  "mul_sigmoid att x sigmoid(gate half of [Q|gate])");
    }
    {  // ADD: bit-identical; h = a in place.
        const std::uint32_t cols = 5120;
        const auto a = rng.normal(std::size_t{T} * cols, 3.0f), b = rng.normal(std::size_t{T} * cols, 3.0f);
        std::vector<float> cpu(a.size());
        hc::add(crows(a, T, cols), crows(b, T, cols), mrows(cpu, T, cols));
        for (const bool in_place : {false, true}) {
            hv::Buffer ba = upload(ctx, std::span<const float>(a), hv::MemoryUsage::HostCached);
            hv::Buffer bb = upload(ctx, std::span<const float>(b));
            hv::Buffer bo = hv::Buffer::create(ctx, a.size() * 4, hv::MemoryUsage::HostCached);
            hv::Stream s(ctx);
            ops.add(s, hv::EltwiseArgs{ba, bb, in_place ? hv::BufferView(ba) : hv::BufferView(bo), T, cols});
            s.submit_and_wait();
            EXPECT_TRUE(bitwise(download<float>(in_place ? ba : bo, a.size()), cpu))
                << "add" << (in_place ? " in place" : "") << " not bit-identical to cpu::add";
        }
    }
}

TEST(VkLayer, EltwiseAliasingRules) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer a = hv::Buffer::create(ctx, 2 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer b = hv::Buffer::create(ctx, 2 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    EXPECT_NO_THROW(ops.add(s, hv::EltwiseArgs{a, b, a, 2, 64}));                              // exact alias
    EXPECT_THROW(ops.add(s, hv::EltwiseArgs{a, b, hv::BufferView(a, 4, 2 * 64 * 4 - 4), 2, 63}), halo::Error);
    EXPECT_THROW(ops.swiglu(s, hv::EltwiseArgs{a, b, hv::BufferView(a, 0, 0, 64 * 4), 2, 32}), halo::Error);  // same start, other stride (hence other end)
    s.submit_and_wait();
}

// ---------------------------------------------------------------- ADD + RMS_NORM

TEST(VkLayer, AddRmsNormMatchesCpuCompositionAndSeparateVulkanOpsBitwise) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const bool in_place : {false, true}) {
        hv::Ops ops(ctx);
        const std::uint32_t rows = 3, cols = 5120;
        ct::Rng rng(60 + in_place);
        const auto a = rng.normal(std::size_t{rows} * cols, 3.0f), b = rng.normal(std::size_t{rows} * cols, 1.0f);
        const auto w = rng.normal(cols, 1.0f);
        // CPU composition (ADR-001 §5.4): cpu::add then cpu::rms_norm.
        std::vector<float> h_cpu(a.size()), y_cpu(a.size());
        hc::add(crows(a, rows, cols), crows(b, rows, cols), mrows(h_cpu, rows, cols));
        hc::rms_norm(crows(h_cpu, rows, cols), w, 1e-6f, mrows(y_cpu, rows, cols));
        hv::Buffer ba = upload(ctx, std::span<const float>(a), hv::MemoryUsage::HostCached);
        hv::Buffer bb = upload(ctx, std::span<const float>(b));
        hv::Buffer bw = upload(ctx, std::span<const float>(w));
        hv::Buffer bh = hv::Buffer::create(ctx, a.size() * 4, hv::MemoryUsage::HostCached);
        hv::Buffer by = hv::Buffer::create(ctx, a.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.add_rms_norm(s, hv::AddRmsNormArgs{ba, bb, in_place ? hv::BufferView(ba) : hv::BufferView(bh), bw, by, rows,
                                               cols, 1e-6f});
        s.submit_and_wait();
        const std::string what = std::string("add_rms_norm 3x5120") + (in_place ? " (h = a)" : "");
        const auto h = download<float>(in_place ? ba : bh, a.size());
        const auto y = download<float>(by, a.size());
        EXPECT_TRUE(bitwise(h, h_cpu)) << what << ": h not bit-identical to cpu::add";
        const double rt = rms_rtol(cols, ops.options().reduce_workgroup);
        check_rel(y, y_cpu, [&](std::size_t) { return rt; }, what + " y");
        // Fused == separate Vulkan rms_norm(h), bit for bit.
        hv::Buffer bh2 = upload(ctx, std::span<const float>(h));
        hv::Buffer by2 = hv::Buffer::create(ctx, a.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s2(ctx);
        ops.rms_norm(s2, bh2, bw, by2, rows, cols, 1e-6f);
        s2.submit_and_wait();
        EXPECT_TRUE(bitwise(y, download<float>(by2, a.size()))) << what << ": fused y != Ops::rms_norm(h)";
    }
    HALO_VK_CONTEXT_OR_SKIP(ctx2);
    hv::Ops ops(ctx2);
    hv::Buffer x = hv::Buffer::create(ctx2, 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer w = hv::Buffer::create(ctx2, 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx2);
    EXPECT_THROW(ops.add_rms_norm(s, hv::AddRmsNormArgs{x, w, x, w, x, 1, 64, 1e-6f}), halo::Error);  // y = h
    s.submit_and_wait();
}

// ---------------------------------------------------------------- GDN gates

TEST(VkLayer, GdnGatesMatchCpuForwardSequence) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const bool in_place : {false, true}) {
        hv::Ops ops(ctx);
        const std::uint32_t rows = 5, nv = 48;
        ct::Rng rng(70 + in_place);
        auto b = rng.normal(std::size_t{rows} * nv, 3.0f);
        auto a = rng.uniform(std::size_t{rows} * nv, -40.0f, 30.0f);  // spans the log1p series, log and > 20 branches
        const auto dt = rng.normal(nv, 1.0f);
        std::vector<float> ssm_a(nv);
        for (auto& v : ssm_a) v = -std::exp(rng.normal(1, 1.0f)[0]);
        b[0] = 0.0f;
        a[1] = 20.5f - dt[1];  // just above the softplus threshold
        a[2] = 100.0f;         // exp(s) overflows fp32: only the threshold keeps g finite
        // The CPU forward's exact sequence (src/models/qwen35.cpp).
        std::vector<float> beta_cpu(b.size()), alpha = a;
        hc::sigmoid(crows(b, rows, nv), mrows(beta_cpu, rows, nv));
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < nv; ++j) alpha[r * nv + j] += dt[j];
        const std::vector<float> s_host = alpha;  // a + dt (for the bound's |s|)
        hc::softplus(crows(alpha, rows, nv), mrows(alpha, rows, nv));
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < nv; ++j) alpha[r * nv + j] = ssm_a[j] * alpha[r * nv + j];
        hv::Buffer bb = upload(ctx, std::span<const float>(b), hv::MemoryUsage::HostCached);
        hv::Buffer ba = upload(ctx, std::span<const float>(a), hv::MemoryUsage::HostCached);
        hv::Buffer bdt = upload(ctx, std::span<const float>(dt));
        hv::Buffer bsa = upload(ctx, std::span<const float>(ssm_a));
        hv::Buffer bbeta = hv::Buffer::create(ctx, b.size() * 4, hv::MemoryUsage::HostCached);
        hv::Buffer bg = hv::Buffer::create(ctx, a.size() * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.gdn_gates(s, hv::GdnGateArgs{bb, ba, bdt, bsa, in_place ? hv::BufferView(bb) : hv::BufferView(bbeta),
                                         in_place ? hv::BufferView(ba) : hv::BufferView(bg), rows, nv});
        s.submit_and_wait();
        const std::string suffix = in_place ? " in place" : "";
        check_rel(download<float>(in_place ? bb : bbeta, b.size()), beta_cpu,
                  [&](std::size_t i) { return 12.0 + 2.0 * std::abs(double(b[i])); }, "gdn beta" + suffix);
        check_rel(download<float>(in_place ? ba : bg, a.size()), alpha,
                  [&](std::size_t i) { return 32.0 + 2.0 * std::abs(double(s_host[i])); }, "gdn g" + suffix);
    }
}
