// Batched GEMV, LM head (logits GEMV + argmax) and TOP_K of the Vulkan backend (WS-F2 V3),
// differentially tested against halo::cpu / halo::tensor on the same inputs (TRD §30; code
// review M-2). Verified on lavapipe only (DECISIONS D-001).
//
//  - GEMV: y[t] vs cpu::matmul (T rows) under the matvec bound of test_vk_ops.cpp
//      |vk - cpu| <= (s(cols, WG) + s(cols, 8)) * sum_c |W[r,c] x[t][c]|,
//      s(n, w) = (ceil(n / w) + log2 w + 4) u
//    and, bitwise, vector t of a batch == a separate n_vec = 1 matvec of x[t] (the batch does not
//    change the arithmetic).
//  - LM head: logits bitwise == gemv; the argmax index and value bitwise == cpu::argmax on those
//    same logits; and vs the CPU's own logits, the chosen row is a CPU maximum up to the GEMV
//    bound (a near-tie may legitimately pick another row). NaN -> read_argmax throws, as
//    cpu::matmul_argmax.
//  - TOP_K: only comparisons, so ids and values must be BIT-IDENTICAL to cpu::top_k (duplicates,
//    +-0, -inf included); NaN -> k_status_nan, as cpu::top_k throws.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/weight_matrix.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"
#include "reference.h"
#include "vk_test_util.h"

namespace hv = halo::vulkan;
namespace hc = halo::cpu;
namespace ref = halo::vulkan::ref;
using hv::test::download;
using hv::test::upload;

namespace {

constexpr std::uint32_t k_cpu_lanes = 8;
constexpr double k_floor = 1e-37;

double sum_factor(std::uint32_t n, std::uint32_t wg) {
    return (std::ceil(double(n) / wg) + std::log2(double(wg)) + 4.0) * ref::k_u;
}

const char* tname(ref::WType t) {
    switch (t) {
        case ref::WType::F32: return "F32";
        case ref::WType::Q8_0: return "Q8_0";
        case ref::WType::Q4_K: return "Q4_K";
        case ref::WType::Q5_K: return "Q5_K";
        case ref::WType::Q6_K: return "Q6_K";
        case ref::WType::IQ4_XS: return "IQ4_XS";
        case ref::WType::IQ4_NL: return "IQ4_NL";
        case ref::WType::Q3_K: return "Q3_K";
        case ref::WType::IQ3_S: return "IQ3_S";
    }
    return "?";
}

hc::WeightMatrix weight_matrix(const ref::Weights& w, ref::WType t, std::uint32_t rows, std::uint32_t cols) {
    if (t == ref::WType::F32) return hc::WeightMatrix::dense(hc::ConstRows(w.deq.data(), rows, cols, cols));
    const std::size_t rb = halo::tensor::row_bytes(ref::dtype(t), cols);
    return hc::WeightMatrix::dequantized(rows, cols, [&w, t, rb, cols](std::size_t first, std::size_t n,
                                                                       std::span<float> out) {
        for (std::size_t r = 0; r < n; ++r) {
            halo::tensor::dequantize_row(ref::dtype(t), std::as_bytes(std::span(w.bytes)).data() + (first + r) * rb,
                                         out.data() + r * cols, cols);
        }
    });
}

bool bitwise(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

}  // namespace

// ---------------------------------------------------------------- batched GEMV

TEST(VkHead, GemvBatchedMatchesCpuMatmulAndEqualsPerVectorMatvecBitwise) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t wg = ops.options().reduce_workgroup;
    for (const ref::WType t : {ref::WType::F32, ref::WType::Q8_0, ref::WType::Q4_K, ref::WType::Q5_K, ref::WType::Q6_K,
                               ref::WType::IQ4_XS, ref::WType::IQ4_NL, ref::WType::Q3_K, ref::WType::IQ3_S}) {
        for (const std::uint32_t n_vec : {1u, 3u, 8u, 11u, 17u}) {
            SCOPED_TRACE(std::string(tname(t)) + " n_vec " + std::to_string(n_vec));
            const std::uint32_t rows = 37, cols = t == ref::WType::F32 ? 300 : 512;
            std::mt19937 rng(static_cast<unsigned>(t) * 100 + n_vec);
            const ref::Weights w = ref::random_weights(t, rows, cols, rng, 14);
            // x: n_vec vectors with a row stride > cols, at a lead of 3 floats; y likewise.
            const std::uint32_t xs = cols + 5, ys = rows + 3, lead = 3;
            std::vector<float> xbuf(lead + std::size_t{n_vec} * xs, 1e30f);
            std::vector<float> xd(std::size_t{n_vec} * cols);  // dense copy for the CPU
            for (std::uint32_t v = 0; v < n_vec; ++v) {
                const auto xv = ref::random_vec(cols, rng);
                std::copy(xv.begin(), xv.end(), xbuf.begin() + lead + std::size_t{v} * xs);
                std::copy(xv.begin(), xv.end(), xd.begin() + std::size_t{v} * cols);
            }
            std::vector<float> ycpu(std::size_t{n_vec} * rows);
            hc::matmul(hc::ConstRows(xd.data(), n_vec, cols, cols), weight_matrix(w, t, rows, cols),
                       hc::Rows(ycpu.data(), n_vec, rows, rows));

            hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
            hv::Buffer bx = upload(ctx, std::span<const float>(xbuf));
            const std::vector<float> yinit(lead + std::size_t{n_vec} * ys, -7.0f);
            hv::Buffer by = upload(ctx, std::span<const float>(yinit), hv::MemoryUsage::HostCached);
            hv::Buffer by1 = hv::Buffer::create(ctx, std::uint64_t{n_vec} * rows * 4, hv::MemoryUsage::HostCached);
            hv::Stream s(ctx);
            hv::GemvArgs g;
            g.wtype = ref::dtype(t);
            g.w = bw;
            g.x = hv::BufferView(bx, lead * 4, 0, xs * 4);
            g.y = hv::BufferView(by, lead * 4, 0, ys * 4);
            g.rows = rows;
            g.cols = cols;
            g.n_vec = n_vec;
            ops.gemv(s, g);
            for (std::uint32_t v = 0; v < n_vec; ++v) {  // the same vectors one at a time
                ops.matvec(s, ref::dtype(t), bw, hv::BufferView(bx, (lead + std::uint64_t{v} * xs) * 4, cols * 4),
                           hv::BufferView(by1, std::uint64_t{v} * rows * 4, rows * 4), rows, cols);
            }
            s.submit_and_wait();
            const auto yb = download<float>(by, yinit.size());
            const auto y1 = download<float>(by1, std::size_t{n_vec} * rows);
            const double f = sum_factor(cols, wg) + sum_factor(cols, k_cpu_lanes);
            double worst = 0;
            for (std::uint32_t v = 0; v < n_vec; ++v) {
                const float* got = yb.data() + lead + std::size_t{v} * ys;
                EXPECT_TRUE(bitwise(std::span(got, rows), std::span(y1.data() + std::size_t{v} * rows, rows)))
                    << "vector " << v << " of the batch != its own matvec";
                for (std::uint32_t r = 0; r < ys - rows; ++r) EXPECT_EQ(got[rows + r], -7.0f) << "gap written";
                const float* xv = xd.data() + std::size_t{v} * cols;
                for (std::uint32_t r = 0; r < rows; ++r) {
                    double scale = 0;
                    for (std::uint32_t c = 0; c < cols; ++c)
                        scale += double(w.mag[std::size_t{r} * cols + c]) * std::fabs(double(xv[c]));
                    const double d = std::fabs(double(got[r]) - double(ycpu[std::size_t{v} * rows + r]));
                    const double b = f * scale + k_floor;
                    worst = std::max(worst, d / b);
                    EXPECT_LE(d, b) << "v " << v << " r " << r;
                }
            }
            std::cout << "[vk-head] gemv " << tname(t) << " n_vec " << n_vec << ": worst |vk-cpu|/bound = " << worst
                      << "\n";
        }
    }
}

TEST(VkHead, GemvValidation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer w = hv::Buffer::create(ctx, 4 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer x = hv::Buffer::create(ctx, 3 * 64 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer y = hv::Buffer::create(ctx, 3 * 4 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    hv::GemvArgs g{halo::DType::F32, w, x, y, 4, 64, 3};
    EXPECT_NO_THROW(ops.gemv(s, g));
    g.n_vec = 4;  // x and y too small
    EXPECT_THROW(ops.gemv(s, g), halo::Error);
    g.n_vec = 3;
    g.y = hv::BufferView(x, 0, 3 * 4 * 4);  // y overlaps x
    EXPECT_THROW(ops.gemv(s, g), halo::Error);
    g = hv::GemvArgs{halo::DType::Q4_0, w, x, y, 4, 64, 3};
    EXPECT_THROW(ops.gemv(s, g), halo::Error);
    g = hv::GemvArgs{halo::DType::F32, w, x, y, 4, 64, 0};
    EXPECT_THROW(ops.gemv(s, g), halo::Error);
    s.submit_and_wait();
}

// ---------------------------------------------------------------- LM head

TEST(VkHead, LmHeadArgmaxMatchesCpuOnTheSameLogitsAndCpuMatmulArgmax) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t wg = ops.options().reduce_workgroup;
    for (const ref::WType t : {ref::WType::Q6_K, ref::WType::Q4_K, ref::WType::F32}) {
        for (const bool logits_out : {true, false}) {
            SCOPED_TRACE(std::string(tname(t)) + (logits_out ? " logits in y" : " logits in workspace"));
            const std::uint32_t rows = 5000, cols = 512, n_vec = 3;
            std::mt19937 rng(static_cast<unsigned>(t) + (logits_out ? 50u : 60u));
            const ref::Weights w = ref::random_weights(t, rows, cols, rng, 14);
            std::vector<float> x;
            for (std::uint32_t v = 0; v < n_vec; ++v) {
                const auto xv = ref::random_vec(cols, rng);
                x.insert(x.end(), xv.begin(), xv.end());
            }
            const hc::WeightMatrix wm = weight_matrix(w, t, rows, cols);
            std::vector<float> ycpu(std::size_t{n_vec} * rows);
            hc::matmul(hc::ConstRows(x.data(), n_vec, cols, cols), wm, hc::Rows(ycpu.data(), n_vec, rows, rows));

            hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
            hv::Buffer bx = upload(ctx, std::span<const float>(x));
            hv::Buffer by = hv::Buffer::create(ctx, std::uint64_t{n_vec} * rows * 4, hv::MemoryUsage::HostCached);
            hv::Buffer bref = hv::Buffer::create(ctx, std::uint64_t{n_vec} * rows * 4, hv::MemoryUsage::HostCached);
            const std::uint64_t ws_bytes = ops.lm_head_workspace_bytes(rows, n_vec, !logits_out);
            hv::Buffer bws = hv::Buffer::create(ctx, ws_bytes + 8, hv::MemoryUsage::DeviceLocal);
            hv::Buffer bres = hv::Buffer::create(ctx, 4 + n_vec * hv::k_argmax_result_bytes, hv::MemoryUsage::HostCached);
            hv::LmHeadArgs a;
            a.gemv = hv::GemvArgs{ref::dtype(t), bw, bx, logits_out ? hv::BufferView(by) : hv::BufferView(), rows, cols,
                                  n_vec};
            a.workspace = hv::BufferView(bws, 8, ws_bytes);
            a.result = hv::BufferView(bres, 4, n_vec * hv::k_argmax_result_bytes);
            hv::Stream s(ctx);
            ops.lm_head(s, a);
            ops.gemv(s, hv::GemvArgs{ref::dtype(t), bw, bx, bref, rows, cols, n_vec});
            s.submit_and_wait();
            const auto logits = download<float>(bref, std::size_t{n_vec} * rows);
            if (logits_out) EXPECT_TRUE(bitwise(download<float>(by, logits.size()), logits)) << "logits != gemv";
            const double f = sum_factor(cols, wg) + sum_factor(cols, k_cpu_lanes);
            for (std::uint32_t v = 0; v < n_vec; ++v) {
                const hv::ArgmaxResult r = hv::read_argmax(bres, 4 + std::uint64_t{v} * hv::k_argmax_result_bytes);
                const std::span<const float> lv(logits.data() + std::size_t{v} * rows, rows);
                const hc::TopKEntry c = hc::argmax(lv);
                EXPECT_EQ(r.index, static_cast<std::uint32_t>(c.index)) << "v " << v << " vs cpu::argmax on vk logits";
                EXPECT_EQ(std::memcmp(&r.value, &c.value, 4), 0) << "v " << v;
                // vs the CPU's own logits: the chosen row is a CPU maximum up to the GEMV bound.
                const std::span<const float> xv(x.data() + std::size_t{v} * cols, cols);
                const hc::TopKEntry cm = hc::matmul_argmax(xv, wm);
                auto bound = [&](std::uint32_t row) {
                    double sc = 0;
                    for (std::uint32_t cc = 0; cc < cols; ++cc)
                        sc += double(w.mag[std::size_t{row} * cols + cc]) * std::fabs(double(xv[cc]));
                    return f * sc + k_floor;
                };
                const double gap = double(cm.value) - double(ycpu[std::size_t{v} * rows + r.index]);
                EXPECT_LE(gap, bound(r.index) + bound(static_cast<std::uint32_t>(cm.index)))
                    << "v " << v << ": vk picked " << r.index << ", cpu " << cm.index;
                std::cout << "[vk-head] lm_head " << tname(t) << " v" << v << ": vk " << r.index << " cpu "
                          << cm.index << (r.index == static_cast<std::uint32_t>(cm.index) ? " (same)" : " (near-tie)")
                          << "\n";
            }
        }
    }
}

TEST(VkHead, LmHeadNanRaisesLikeCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t rows = 300, cols = 64, n_vec = 2;
    std::mt19937 rng(3);
    std::vector<float> wv = ref::random_vec(std::size_t{rows} * cols, rng);
    wv[std::size_t{123} * cols + 5] = std::numeric_limits<float>::quiet_NaN();  // row 123
    std::vector<float> x = ref::random_vec(std::size_t{n_vec} * cols, rng);
    EXPECT_THROW((void)hc::matmul_argmax(std::span(x).first(cols),
                                         hc::WeightMatrix::dense(hc::ConstRows(wv.data(), rows, cols, cols))),
                 halo::Error);
    hv::Buffer bw = upload(ctx, std::span<const float>(wv));
    hv::Buffer bx = upload(ctx, std::span<const float>(x));
    const std::uint64_t ws_bytes = ops.lm_head_workspace_bytes(rows, n_vec, true);
    hv::Buffer bws = hv::Buffer::create(ctx, ws_bytes, hv::MemoryUsage::DeviceLocal);
    hv::Buffer bres = hv::Buffer::create(ctx, n_vec * hv::k_argmax_result_bytes, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    ops.lm_head(s, hv::LmHeadArgs{hv::GemvArgs{halo::DType::F32, bw, bx, {}, rows, cols, n_vec}, bws, bres});
    s.submit_and_wait();
    for (std::uint32_t v = 0; v < n_vec; ++v) {
        EXPECT_THROW((void)hv::read_argmax(bres, std::uint64_t{v} * hv::k_argmax_result_bytes), halo::Error)
            << "vector " << v;
    }
}

TEST(VkHead, LmHeadValidation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t rows = 300, cols = 64, n_vec = 2;
    hv::Buffer w = hv::Buffer::create(ctx, rows * cols * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer x = hv::Buffer::create(ctx, n_vec * cols * 4, hv::MemoryUsage::DeviceLocal);
    const std::uint64_t ws_bytes = ops.lm_head_workspace_bytes(rows, n_vec, true);
    hv::Buffer ws = hv::Buffer::create(ctx, ws_bytes, hv::MemoryUsage::DeviceLocal);
    hv::Buffer res = hv::Buffer::create(ctx, n_vec * 12, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    const hv::GemvArgs g{halo::DType::F32, w, x, {}, rows, cols, n_vec};
    EXPECT_NO_THROW(ops.lm_head(s, hv::LmHeadArgs{g, ws, res}));
    EXPECT_THROW(ops.lm_head(s, hv::LmHeadArgs{g, hv::BufferView(ws, 0, ws_bytes - 4), res}), halo::Error);
    EXPECT_THROW(ops.lm_head(s, hv::LmHeadArgs{g, ws, hv::BufferView(res, 0, 12)}), halo::Error);
    EXPECT_THROW(ops.lm_head(s, hv::LmHeadArgs{g, ws, hv::BufferView(ws, 0, n_vec * 12)}), halo::Error);
    const std::uint32_t before = s.dispatch_count();
    EXPECT_THROW(ops.lm_head(s, hv::LmHeadArgs{g, ws, hv::BufferView(ws, 0, n_vec * 12)}), halo::Error);
    EXPECT_EQ(s.dispatch_count(), before) << "a rejected lm_head recorded work";
    s.submit_and_wait();
}

// ---------------------------------------------------------------- TOP_K

namespace {

void topk_case(const std::shared_ptr<hv::Context>& ctx, std::uint32_t n, std::uint32_t k, std::uint32_t n_vec,
               std::uint64_t seed, bool with_nan = false) {
    const std::string what = "top_k n " + std::to_string(n) + " k " + std::to_string(k) + " n_vec " +
                             std::to_string(n_vec) + (with_nan ? " NaN" : "");
    SCOPED_TRACE(what);
    hv::Ops ops(ctx);
    std::mt19937 rng(static_cast<unsigned>(seed));
    // Few distinct levels (many exact ties), +-0, -inf and a wide range.
    std::uniform_int_distribution<int> lvl(-40, 40);
    const std::uint32_t stride = n + 7;
    std::vector<float> lg(std::size_t{n_vec} * stride, 1e30f);
    for (std::uint32_t v = 0; v < n_vec; ++v) {
        float* row = lg.data() + std::size_t{v} * stride;
        for (std::uint32_t i = 0; i < n; ++i) {
            const int l = lvl(rng);
            row[i] = l == 0 ? ((rng() & 1u) ? 0.0f : -0.0f)
                            : (l == -40 ? -std::numeric_limits<float>::infinity() : 0.37f * float(l));
        }
        if (with_nan && v == n_vec - 1) row[n / 2] = std::numeric_limits<float>::quiet_NaN();
    }
    hv::Buffer bl = upload(ctx, std::span<const float>(lg));
    const std::uint64_t ws = hv::topk_workspace_bytes(n, k, n_vec);
    hv::Buffer bws = hv::Buffer::create(ctx, ws + 16, hv::MemoryUsage::DeviceLocal);
    hv::Buffer bi = hv::Buffer::create(ctx, std::uint64_t{n_vec} * k * 4 + 8, hv::MemoryUsage::HostCached);
    hv::Buffer bv = hv::Buffer::create(ctx, std::uint64_t{n_vec} * k * 4 + 12, hv::MemoryUsage::HostCached);
    const std::vector<std::uint32_t> st{0xDEADBEEFu, 0xDEADBEEFu};
    hv::Buffer bs = upload(ctx, std::span<const std::uint32_t>(st), hv::MemoryUsage::HostCached);
    hv::TopKArgs a;
    a.logits = hv::BufferView(bl, 0, 0, std::uint64_t{stride} * 4);
    a.n = n;
    a.k = k;
    a.n_vec = n_vec;
    a.workspace = hv::BufferView(bws, 4, ws == 0 ? 4 : ws);
    a.ids = hv::BufferView(bi, 8, std::uint64_t{n_vec} * k * 4);
    a.values = hv::BufferView(bv, 12, std::uint64_t{n_vec} * k * 4);
    a.status = hv::BufferView(bs, 4, 4);
    hv::Stream s(ctx);
    ops.top_k(s, a);
    s.submit_and_wait();
    const std::uint32_t status = hv::read_status(bs, 4);
    if (with_nan) {
        EXPECT_EQ(status, hv::k_status_nan);
        EXPECT_THROW(hv::check_status(status, "top_k"), halo::Error);
        EXPECT_THROW((void)hc::top_k(std::span<const float>(lg.data() + std::size_t{n_vec - 1} * stride, n), k),
                     halo::Error);
        return;
    }
    EXPECT_EQ(status, 0u);
    std::vector<std::int32_t> ids(std::size_t{n_vec} * k);
    std::vector<float> vals(ids.size());
    bi.download(std::span<std::int32_t>(ids), 8);
    bv.download(std::span<float>(vals), 12);
    std::size_t bad = 0;
    for (std::uint32_t v = 0; v < n_vec; ++v) {
        const auto want = hc::top_k(std::span<const float>(lg.data() + std::size_t{v} * stride, n), k);
        for (std::uint32_t i = 0; i < k; ++i) {
            const std::size_t o = std::size_t{v} * k + i;
            const bool same = ids[o] == want[i].index && std::memcmp(&vals[o], &want[i].value, 4) == 0;
            if (!same && bad++ < 3) {
                ADD_FAILURE() << "v " << v << " rank " << i << ": vk (" << ids[o] << ", " << vals[o] << ") cpu ("
                              << want[i].index << ", " << want[i].value << ")";
            }
        }
    }
    EXPECT_EQ(bad, 0u);
    std::cout << "[vk-head] " << what << ": " << bad << " entries differ from cpu::top_k\n";
}

}  // namespace

TEST(VkHead, TopKIsBitIdenticalToCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    topk_case(ctx, 5, 5, 1, 1);         // k = n
    topk_case(ctx, 5, 1, 2, 2);         // k = 1
    topk_case(ctx, 2048, 40, 2, 3);     // exactly one chunk
    topk_case(ctx, 2049, 40, 2, 4);     // two chunks, the second holding one entry
    topk_case(ctx, 2049, 1024, 1, 5);   // max k: two passes
    topk_case(ctx, 248320, 40, 2, 6);   // the qwen35 vocabulary: three passes
    topk_case(ctx, 248320, 1024, 1, 7); // max k over the vocabulary: several passes
}

TEST(VkHead, TopKNanSetsStatusLikeCpuThrows) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    topk_case(ctx, 3000, 10, 2, 8, /*with_nan=*/true);
}

TEST(VkHead, TopKValidation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t n = 5000, k = 16;
    hv::Buffer lg = hv::Buffer::create(ctx, n * 4, hv::MemoryUsage::DeviceLocal);
    const std::uint64_t ws = hv::topk_workspace_bytes(n, k, 1);
    hv::Buffer bws = hv::Buffer::create(ctx, ws, hv::MemoryUsage::DeviceLocal);
    hv::Buffer out = hv::Buffer::create(ctx, 2 * k * 4 + 4, hv::MemoryUsage::DeviceLocal);
    auto base = [&] {
        hv::TopKArgs a;
        a.logits = lg;
        a.n = n;
        a.k = k;
        a.workspace = bws;
        a.ids = hv::BufferView(out, 0, k * 4);
        a.values = hv::BufferView(out, k * 4, k * 4);
        a.status = hv::BufferView(out, 2 * k * 4, 4);
        return a;
    };
    hv::Stream s(ctx);
    EXPECT_NO_THROW(ops.top_k(s, base()));
    auto a = base();
    a.k = 1025;
    try {
        ops.top_k(s, a);
        ADD_FAILURE() << "k 1025 accepted";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
    }
    a = base();
    a.k = 0;
    EXPECT_THROW(ops.top_k(s, a), halo::Error);
    a = base();
    a.n = 10;
    a.k = 11;
    EXPECT_THROW(ops.top_k(s, a), halo::Error);
    a = base();
    a.workspace = hv::BufferView(bws, 0, ws - 4);
    EXPECT_THROW(ops.top_k(s, a), halo::Error);
    a = base();
    a.values = hv::BufferView(out, 4, k * 4);  // overlaps ids
    EXPECT_THROW(ops.top_k(s, a), halo::Error);
    a = base();
    a.workspace = hv::BufferView(lg, 0, ws);  // workspace inside logits
    EXPECT_THROW(ops.top_k(s, a), halo::Error);
    EXPECT_EQ(hv::topk_workspace_bytes(2048, 40, 3), 0u);
    s.submit_and_wait();
}

TEST(VkHead, GemvSplitsWeightsLargerThanOneBindingIntoRowSlabs) {
    // A W larger than one binding (the 248320-row LM head) runs as row slabs; the arithmetic
    // per output row is unchanged, so the split result is bitwise the unsplit one.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint32_t rows = 53, cols = 512, n_vec = 3;
    std::mt19937 rng(21);
    const ref::Weights w = ref::random_weights(ref::WType::Q4_K, rows, cols, rng, 14);
    std::vector<float> x;
    for (std::uint32_t v = 0; v < n_vec; ++v) {
        const auto xv = ref::random_vec(cols, rng);
        x.insert(x.end(), xv.begin(), xv.end());
    }
    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
    hv::Buffer bx = upload(ctx, std::span<const float>(x));
    std::vector<std::vector<float>> ys;
    for (const std::uint64_t limit : {std::uint64_t{0}, std::uint64_t{3000}, std::uint64_t{400}}) {
        hv::OpsOptions o;
        o.max_binding_bytes = limit;
        hv::Ops ops(ctx, o);
        hv::Buffer by = hv::Buffer::create(ctx, std::uint64_t{n_vec} * (rows + 2) * 4, hv::MemoryUsage::HostCached);
        hv::Stream s(ctx);
        ops.gemv(s, hv::GemvArgs{halo::DType::Q4_K, bw, bx, hv::BufferView(by, 0, 0, (rows + 2) * 4), rows, cols, n_vec});
        std::cout << "[vk-head] gemv max_binding_bytes " << limit << ": " << s.dispatch_count() << " dispatch(es)\n";
        if (limit != 0) EXPECT_GT(s.dispatch_count(), 1u) << "W was not split";
        s.submit_and_wait();
        auto y = download<float>(by, std::size_t{n_vec} * (rows + 2));
        for (std::uint32_t v = 0; v < n_vec; ++v) y[std::size_t{v} * (rows + 2) + rows] = y[std::size_t{v} * (rows + 2) + rows + 1] = 0;
        ys.push_back(std::move(y));
    }
    for (std::size_t i = 1; i < ys.size(); ++i) EXPECT_TRUE(bitwise(ys[i], ys[0])) << "split run " << i;
}
