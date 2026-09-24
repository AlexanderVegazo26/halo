// BufferView operands (code review S-1): sub-ranges, arenas, row strides and byte-range
// aliasing checks, differential against halo::cpu / halo::tensor on the SAME bytes.
//
// Every numeric case is checked twice:
//   1. against halo::cpu within |vk - cpu| <= (bound_vk + bound_cpu) * scale (the same
//      a-priori bounds as test_vk_ops.cpp), and
//   2. bitwise against the same Vulkan kernel run on dense, standalone copies of the same
//      data. The kernels are deterministic and the reduction order does not depend on
//      where an operand lives, so any offset / stride / remainder mistake shows up as a
//      bit difference even when it stays inside the numeric bound.
// Bytes outside every view are filled with sentinels (and NaN for fp32 regions) and are
// checked to be unread (results unaffected) and, for outputs, unwritten.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
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

constexpr std::uint32_t k_cpu_lanes = 8;
constexpr float k_nan = std::numeric_limits<float>::quiet_NaN();

double sum_bound_factor(std::uint32_t n, std::uint32_t lanes, double c = 4.0) {
    return (std::ceil(static_cast<double>(n) / lanes) + std::log2(static_cast<double>(lanes)) + c) * ref::k_u;
}

std::vector<double> to_dbl(std::span<const float> v) { return {v.begin(), v.end()}; }

bool bitwise_equal(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// halo::cpu::matmul of one x row with W rows read from `bytes` at `offset + r * stride`
// (tensor::dequantize_row for quantized types, a strided float view for F32).
std::vector<float> cpu_matvec_at(ref::WType t, std::span<const std::uint8_t> bytes, std::size_t offset,
                                 std::size_t stride, std::uint32_t rows, std::uint32_t cols,
                                 std::span<const float> x) {
    const halo::DType dt = ref::dtype(t);
    std::vector<float> f32_copy;
    hc::WeightMatrix m = [&] {
        if (t == ref::WType::F32) {
            f32_copy.resize(bytes.size() / 4);
            std::memcpy(f32_copy.data(), bytes.data(), f32_copy.size() * 4);
            return hc::WeightMatrix::dense(hc::ConstRows(f32_copy.data() + offset / 4, rows, cols, stride / 4));
        }
        return hc::WeightMatrix::dequantized(rows, cols, [=](std::size_t first, std::size_t n, std::span<float> out) {
            for (std::size_t r = 0; r < n; ++r) {
                halo::tensor::dequantize_row(dt, std::as_bytes(bytes).data() + offset + (first + r) * stride,
                                             out.data() + r * cols, cols);
            }
        });
    }();
    std::vector<float> y(rows);
    hc::matmul(hc::ConstRows(x.data(), 1, cols, cols), m, hc::Rows(y.data(), 1, rows, rows));
    return y;
}

struct ArenaWeight {
    ArenaWeight(ref::WType type, std::uint32_t r, std::uint32_t c) : t(type), rows(r), cols(c) {}
    ref::WType t;
    std::uint32_t rows, cols;
    std::size_t offset = 0, stride = 0;  // bytes within the arena
    ref::Weights w;                      // dense copy (bytes, deq, mag)
};

}  // namespace

// ---------------------------------------------------------------- matvec in one arena

TEST(VkViews, ArenaWeightsAtOddByteOffsetsAndStridesMatchCpu) {
    // Several weights of every supported type share one arena buffer. Quantized weights
    // start at arbitrary byte offsets (odd, half-word and word aligned) and some rows are padded
    // (row_stride > row size); padding bytes are 0xA5 garbage. x lives inside a larger
    // buffer at an odd element offset; every y is a sub-range of one output buffer.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t wg = ops.options().reduce_workgroup;
    std::mt19937 rng(4242);
    std::vector<ArenaWeight> ws{{ref::WType::Q8_0, 5, 64},    {ref::WType::Q4_K, 3, 512},
                                {ref::WType::Q5_K, 3, 256},   {ref::WType::Q6_K, 4, 256},
                                {ref::WType::IQ4_XS, 2, 512}, {ref::WType::F32, 3, 40},
                                {ref::WType::Q6_K, 2, 512}};
    std::size_t cursor = 0;
    std::uint32_t max_cols = 0;
    for (std::size_t i = 0; i < ws.size(); ++i) {
        ArenaWeight& a = ws[i];
        a.w = ref::random_weights(a.t, a.rows, a.cols, rng, 14);
        const std::size_t rb = halo::tensor::row_bytes(ref::dtype(a.t), a.cols);
        const bool f32 = a.t == ref::WType::F32;
        // Quantized: 1 or 3 bytes past the previous weight, so starts land at mixed byte
        // alignments (observed: 1, 344, 1223, 1754, 2616, 3669 -- odd, 2-mod-4 and 0-mod-4).
        a.offset = f32 ? (cursor + 3) / 4 * 4 + 4 : cursor + 1 + 2 * (i % 2);
        a.stride = rb + (i % 2 == 0 ? 0 : (f32 ? 12 : 7));                        // some rows padded
        cursor = a.offset + (a.rows - 1) * a.stride + rb;
        max_cols = std::max(max_cols, a.cols);
    }
    std::vector<std::uint8_t> arena(cursor + 11, 0xA5);
    for (const ArenaWeight& a : ws) {
        const std::size_t rb = halo::tensor::row_bytes(ref::dtype(a.t), a.cols);
        for (std::uint32_t r = 0; r < a.rows; ++r) {
            std::memcpy(arena.data() + a.offset + r * a.stride, a.w.bytes.data() + r * rb, rb);
        }
    }
    hv::Buffer barena = upload(ctx, std::span<const std::uint8_t>(arena));

    const std::size_t x_off = 3;  // elements
    std::vector<float> xbuf(x_off + max_cols + 5, k_nan);
    const std::vector<float> x = ref::random_vec(max_cols, rng);
    std::copy(x.begin(), x.end(), xbuf.begin() + static_cast<std::ptrdiff_t>(x_off));
    hv::Buffer bx = upload(ctx, std::span<const float>(xbuf));

    std::size_t y_total = 1;
    std::vector<std::size_t> y_off;
    for (const ArenaWeight& a : ws) {
        y_off.push_back(y_total);
        y_total += a.rows + 2;  // 2 guard floats between outputs
    }
    const float guard = -12345.0f;
    hv::Buffer by = upload(ctx, std::span<const float>(std::vector<float>(y_total, guard)),
                           hv::MemoryUsage::HostCached);

    hv::Stream s(ctx);
    for (std::size_t i = 0; i < ws.size(); ++i) {
        const ArenaWeight& a = ws[i];
        const std::size_t rb = halo::tensor::row_bytes(ref::dtype(a.t), a.cols);
        const hv::BufferView wv(barena, a.offset, (a.rows - 1) * a.stride + rb, a.stride);
        ops.matvec(s, ref::dtype(a.t), wv, hv::BufferView(bx, x_off * 4, std::uint64_t{a.cols} * 4),
                   hv::BufferView(by, y_off[i] * 4, std::uint64_t{a.rows} * 4), a.rows, a.cols);
    }
    s.submit_and_wait();
    const std::vector<float> yall = download<float>(by, y_total);

    for (std::size_t i = 0; i < ws.size(); ++i) {
        const ArenaWeight& a = ws[i];
        const std::string what = "arena weight " + std::to_string(i) + " type " +
                                 std::to_string(static_cast<int>(a.t)) + " at byte " + std::to_string(a.offset) +
                                 " stride " + std::to_string(a.stride);
        const std::span<const float> xs(x.data(), a.cols);
        const std::span<const float> y(yall.data() + y_off[i], a.rows);
        // 1. vs halo::cpu on the same arena bytes.
        const std::vector<float> y_cpu = cpu_matvec_at(a.t, arena, a.offset, a.stride, a.rows, a.cols, xs);
        const std::vector<double> scale = ref::matvec_scale(a.w.mag, xs, a.rows, a.cols);
        const double f = sum_bound_factor(a.cols, wg) + sum_bound_factor(a.cols, k_cpu_lanes);
        const auto st = compare(y, to_dbl(y_cpu), [&](std::size_t r) { return f * scale[r] + ct::kDenormFloor; });
        report(what, st);
        EXPECT_LE(st.max_ratio, 1.0) << what;
        // 2. bitwise vs the same kernel on a dense standalone copy.
        hv::Buffer dw = upload(ctx, std::span<const std::uint8_t>(a.w.bytes));
        hv::Buffer dx = upload(ctx, xs);
        hv::Buffer dy = hv::Buffer::create(ctx, std::uint64_t{a.rows} * 4, hv::MemoryUsage::HostCached);
        hv::Stream s2(ctx);
        ops.matvec(s2, ref::dtype(a.t), dw, dx, dy, a.rows, a.cols);
        s2.submit_and_wait();
        EXPECT_TRUE(bitwise_equal(y, download<float>(dy, a.rows))) << what << ": view result != dense result";
    }
    // Guards between the outputs are untouched.
    std::vector<bool> inside(y_total, false);
    for (std::size_t i = 0; i < ws.size(); ++i) {
        for (std::uint32_t r = 0; r < ws[i].rows; ++r) inside[y_off[i] + r] = true;
    }
    for (std::size_t i = 0; i < y_total; ++i) {
        if (!inside[i]) EXPECT_EQ(yall[i], guard) << "output guard " << i << " written";
    }
}

TEST(VkViews, MatvecWeightBeyondMaxStorageBufferRange) {
    // A weight view deep inside an arena (past maxStorageBufferRange, at an odd byte) is
    // bound at its own aligned offset with only its own extent.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint64_t base = ctx->info().max_storage_buffer_range + 3;
    const std::uint32_t rows = 4, cols = 256;
    std::mt19937 rng(77);
    const ref::Weights w = ref::random_weights(ref::WType::Q6_K, rows, cols, rng, 14);
    const std::uint64_t total = base + w.bytes.size() + 8;
    if (total > ctx->info().max_memory_allocation_size) {
        GTEST_SKIP() << "maxMemoryAllocationSize " << ctx->info().max_memory_allocation_size
                     << " too small for a buffer past maxStorageBufferRange";
    }
    hv::Ops ops(ctx);
    hv::Buffer arena = hv::Buffer::create(ctx, total, hv::MemoryUsage::HostVisible);
    arena.upload(std::span<const std::uint8_t>(w.bytes), base);
    const std::vector<float> x = ref::random_vec(cols, rng);
    hv::Buffer bx = upload(ctx, std::span<const float>(x));
    hv::Buffer by = hv::Buffer::create(ctx, rows * 4, hv::MemoryUsage::HostCached);
    const std::uint64_t rb = hv::matvec_row_bytes(halo::DType::Q6_K, cols);
    hv::Stream s(ctx);
    ops.matvec(s, halo::DType::Q6_K, hv::BufferView(arena, base, rows * rb), bx, by, rows, cols);
    s.submit_and_wait();
    const std::vector<float> y = download<float>(by, rows);
    const std::vector<float> y_cpu = cpu_matvec_at(ref::WType::Q6_K, w.bytes, 0, rb, rows, cols, x);
    const std::vector<double> scale = ref::matvec_scale(w.mag, x, rows, cols);
    const double f = sum_bound_factor(cols, ops.options().reduce_workgroup) + sum_bound_factor(cols, k_cpu_lanes);
    const auto st = compare(y, to_dbl(y_cpu), [&](std::size_t r) { return f * scale[r] + ct::kDenormFloor; });
    report("matvec_q6_k weight past maxStorageBufferRange", st);
    EXPECT_LE(st.max_ratio, 1.0);
}

// ---------------------------------------------------------------- rms_norm strided

TEST(VkViews, RmsNormStridedViewsMatchCpuAndDense) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx, hv::OpsOptions{.reduce_workgroup = 64});
    const std::uint32_t rows = 7, cols = 300;
    const std::size_t xs = cols + 3, ys = cols + 1;  // row strides (elements)
    const std::size_t x_off = 5, w_off = 7, y_off = 2;
    std::mt19937 rng(9);
    const std::vector<float> xd = ref::random_vec(std::size_t{rows} * cols, rng, 3.0f);
    const std::vector<float> w = ref::random_vec(cols, rng);
    std::vector<float> xbuf(x_off + rows * xs, k_nan), wbuf(w_off + cols + 4, k_nan);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::copy_n(xd.begin() + r * cols, cols, xbuf.begin() + static_cast<std::ptrdiff_t>(x_off + r * xs));
    }
    std::copy(w.begin(), w.end(), wbuf.begin() + static_cast<std::ptrdiff_t>(w_off));
    const float guard = 777.0f;
    hv::Buffer bx = upload(ctx, std::span<const float>(xbuf));
    hv::Buffer bw = upload(ctx, std::span<const float>(wbuf));
    const std::size_t y_n = y_off + rows * ys + 3;
    hv::Buffer by = upload(ctx, std::span<const float>(std::vector<float>(y_n, guard)), hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    ops.rms_norm(s, hv::BufferView(bx, x_off * 4, 0, xs * 4), hv::BufferView(bw, w_off * 4, cols * 4),
                 hv::BufferView(by, y_off * 4, 0, ys * 4), rows, cols, 1e-6f);
    s.submit_and_wait();
    const std::vector<float> yall = download<float>(by, y_n);

    // CPU on strided views of the same host arrays.
    std::vector<float> ycpu(rows * ys, 0.0f);
    hc::rms_norm(hc::ConstRows(xbuf.data() + x_off, rows, cols, xs), std::span<const float>(w), 1e-6f,
                 hc::Rows(ycpu.data(), rows, cols, ys));
    std::vector<float> y(std::size_t{rows} * cols), yc(std::size_t{rows} * cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::copy_n(yall.begin() + static_cast<std::ptrdiff_t>(y_off + r * ys), cols, y.begin() + r * cols);
        std::copy_n(ycpu.begin() + static_cast<std::ptrdiff_t>(r * ys), cols, yc.begin() + r * cols);
    }
    const auto mag = ref::rms_norm_scale(xd, w, rows, cols, 1e-6f);
    const double rtol = (std::ceil(double(cols) / 64) + 6.0 + 16.0) * ref::k_u +
                        (std::ceil(double(cols) / k_cpu_lanes) + 3.0 + 16.0) * ref::k_u;
    const auto st = compare(y, to_dbl(yc), [&](std::size_t i) { return rtol * mag[i] + 1e-30; });
    report("rms_norm strided views", st);
    EXPECT_LE(st.max_ratio, 1.0);
    for (std::size_t i = 0; i < y_n; ++i) {
        const bool inside = i >= y_off && (i - y_off) / ys < rows && (i - y_off) % ys < cols;
        if (!inside) EXPECT_EQ(yall[i], guard) << "y guard " << i << " written";
    }
    // Bitwise vs dense.
    hv::Buffer dx = upload(ctx, std::span<const float>(xd));
    hv::Buffer dw = upload(ctx, std::span<const float>(w));
    hv::Buffer dy = hv::Buffer::create(ctx, xd.size() * 4, hv::MemoryUsage::HostCached);
    hv::Stream s2(ctx);
    ops.rms_norm(s2, dx, dw, dy, rows, cols, 1e-6f);
    s2.submit_and_wait();
    EXPECT_TRUE(bitwise_equal(y, download<float>(dy, xd.size()))) << "strided result != dense result";
}

// ---------------------------------------------------------------- argmax sub-views

TEST(VkViews, ArgmaxOnSubViewsIgnoresBytesOutsideTheView) {
    // Logits are a sub-range of a larger buffer whose other elements are +huge and NaN;
    // scratch and result share one buffer at odd word offsets.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx, hv::OpsOptions{.reduce_workgroup = 64});
    const std::uint32_t n = 5000;
    const std::size_t off = 13;
    std::mt19937 rng(31);
    const std::vector<float> x = ref::random_vec(n, rng, 2.0f);
    std::vector<float> buf(off + n + 9, 1e30f);
    buf[off - 1] = k_nan;
    buf[off + n] = k_nan;
    std::copy(x.begin(), x.end(), buf.begin() + static_cast<std::ptrdiff_t>(off));
    hv::Buffer bl = upload(ctx, std::span<const float>(buf));
    const std::uint64_t sb = ops.argmax_scratch_bytes(n);
    const std::uint64_t s_off = 4, r_off = 4 + sb + 4;  // bytes; odd word offsets
    hv::Buffer work = hv::Buffer::create(ctx, r_off + hv::k_argmax_result_bytes + 8, hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    ops.argmax(s, hv::BufferView(bl, off * 4, std::uint64_t{n} * 4), n, hv::BufferView(work, s_off, sb),
               hv::BufferView(work, r_off, hv::k_argmax_result_bytes));
    s.submit_and_wait();
    const hv::ArgmaxResult r = hv::read_argmax(work, r_off);
    const hc::TopKEntry c = hc::argmax(x);
    EXPECT_EQ(r.index, static_cast<std::uint32_t>(c.index));
    EXPECT_EQ(r.value, c.value);
}

// ---------------------------------------------------------------- aliasing and validation

TEST(VkViews, ByteRangeOverlapChecksAcceptDisjointAndRejectOverlap) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t cols = 64, rows = 4;
    std::mt19937 rng(5);
    const ref::Weights w = ref::random_weights(ref::WType::Q8_0, rows, cols, rng, 14);
    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(w.bytes));
    // x = elements [0, 64), y = elements [64, 68) of ONE buffer: disjoint -> accepted.
    const std::vector<float> x = ref::random_vec(cols, rng);
    std::vector<float> xy(cols + rows + 4, 0.0f);
    std::copy(x.begin(), x.end(), xy.begin());
    hv::Buffer bxy = upload(ctx, std::span<const float>(xy), hv::MemoryUsage::HostCached);
    const hv::BufferView xv(bxy, 0, cols * 4);
    hv::Stream s(ctx);
    ops.matvec(s, halo::DType::Q8_0, bw, xv, hv::BufferView(bxy, cols * 4, rows * 4), rows, cols);
    s.submit_and_wait();
    const auto got = download<float>(bxy, xy.size());
    const std::vector<float> y_cpu = cpu_matvec_at(ref::WType::Q8_0, w.bytes, 0, hv::matvec_row_bytes(halo::DType::Q8_0, cols),
                                                   rows, cols, x);
    const std::vector<double> scale = ref::matvec_scale(w.mag, x, rows, cols);
    const double f = sum_bound_factor(cols, 256) + sum_bound_factor(cols, k_cpu_lanes);
    const auto st = compare(std::span<const float>(got.data() + cols, rows), to_dbl(y_cpu),
                            [&](std::size_t r) { return f * scale[r] + ct::kDenormFloor; });
    EXPECT_LE(st.max_ratio, 1.0) << "same-buffer disjoint x/y";

    hv::Stream v(ctx);  // validation only; nothing valid is submitted below
    // y overlapping x by one element (4 bytes).
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, bw, xv, hv::BufferView(bxy, (cols - 1) * 4, rows * 4), rows, cols),
                 halo::Error);
    // y overlapping the weights (same buffer, quantized W read as bytes).
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, bw, xv, hv::BufferView(bw, 8, rows * 4), rows, cols), halo::Error);
    // fp32 view offset / stride not a multiple of 4.
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, bw, hv::BufferView(bxy, 2, cols * 4), hv::BufferView(bxy, cols * 4 + 16),
                            rows, cols),
                 halo::Error);
    // view `bytes` smaller than the operand, and past the buffer end.
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, bw, hv::BufferView(bxy, 0, cols * 4 - 4),
                            hv::BufferView(bxy, cols * 4, rows * 4), rows, cols),
                 halo::Error);
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, bw, xv, hv::BufferView(bxy, bxy.size() - 8, rows * 4), rows, cols),
                 halo::Error);
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, bw, xv, hv::BufferView(bxy, bxy.size() + 4), rows, cols),
                 halo::Error);
    // W row_stride smaller than the row.
    EXPECT_THROW(ops.matvec(v, halo::DType::Q8_0, hv::BufferView(bw, 0, 0, 60), xv, hv::BufferView(bxy, cols * 4), rows,
                            cols),
                 halo::Error);
    // rms_norm in place (y == x) is rejected; y after x in the same buffer is accepted.
    hv::Buffer nw = upload(ctx, std::span<const float>(std::vector<float>(16, 1.0f)));
    EXPECT_THROW(ops.rms_norm(v, hv::BufferView(bxy, 0, 64), nw, hv::BufferView(bxy, 0, 64), 1, 16, 1e-6f), halo::Error);
    EXPECT_THROW(ops.rms_norm(v, hv::BufferView(bxy, 0, 64), nw, hv::BufferView(bxy, 60, 64), 1, 16, 1e-6f), halo::Error);
    EXPECT_NO_THROW(ops.rms_norm(v, hv::BufferView(bxy, 0, 64), nw, hv::BufferView(bxy, 64, 64), 1, 16, 1e-6f));
    // argmax result overlapping scratch or logits.
    const std::uint64_t sb = ops.argmax_scratch_bytes(16);
    hv::Buffer work = hv::Buffer::create(ctx, 256, hv::MemoryUsage::DeviceLocal);
    EXPECT_THROW(ops.argmax(v, hv::BufferView(bxy, 0, 64), 16, hv::BufferView(work, 0, sb),
                            hv::BufferView(work, sb - 4, hv::k_argmax_result_bytes)),
                 halo::Error);
    EXPECT_THROW(ops.argmax(v, hv::BufferView(bxy, 0, 64), 16, hv::BufferView(work, 0, sb),
                            hv::BufferView(bxy, 56, hv::k_argmax_result_bytes)),
                 halo::Error);
    EXPECT_NO_THROW(ops.argmax(v, hv::BufferView(bxy, 0, 64), 16, hv::BufferView(work, 0, sb),
                               hv::BufferView(work, sb, hv::k_argmax_result_bytes)));
    v.submit_and_wait();
}
