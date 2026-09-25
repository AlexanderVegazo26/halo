// QUANT_GEMV of the HIP backend (F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S, Q4_0),
// differentially tested
// against halo::tensor::dequantize_row + halo::cpu::matmul on the same input bytes
// (TRD §30, review M-2).
//
//   gemv_generic_b64 : reproduces cpu::detail::dot's order  -> must be BIT-IDENTICAL.
//   gemv_wave32_r4/8 : different fixed order                -> within a derived bound.
//
// The bound. Every product x_i * w_i is formed identically on both sides (the dequantized
// weights are bit-identical, products are single roundings of the same operands), so the
// two results differ only by summation rounding. A sum whose longest addition chain has
// depth d satisfies |s_hat - s| <= d * u * sum|t_i| (u = 2^-24, first order), hence
//   |y_hip - y_cpu| <= (d_cpu + d_hip) * u * sum_i |x_i w_i| * (1 + 1e-3)
// with d_cpu = K/8 + 3 + K%8 + 1 and d_hip = ceil(K/32) + 5 (kernels/gemv.h). sum|x_i w_i| is
// computed in double from the dequantized weights. This is a worst-case bound, not a guess.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "halo/backends/cpu/ops.h"
#include "halo/tensor/fp16.h"
#include "halo/tensor/quant.h"
#include "hip_test_util.h"
#include "kernels/gemv.h"

namespace halo::hip::test {
namespace {

struct GemvCase {
    DType type = DType::Q4_K;
    std::uint32_t rows = 37;
    std::uint32_t cols = 512;
    std::uint32_t n_vec = 1;
    std::uint64_t stride_pad = 0;  // extra bytes between weight rows
    std::uint64_t offset = 0;      // byte offset of the first row in the buffer
    std::uint32_t x_pad = 0;       // extra floats between x vectors (strided x view)
    std::uint32_t y_pad = 0;       // extra floats between y vectors (strided y view)
    std::uint32_t seed = 1;
};

/// Random but valid weight bytes: random quants, finite f16 scales (one subnormal scale per
/// row to exercise the fp16 subnormal path).
std::vector<std::byte> make_weights(const GemvCase& c, std::uint64_t row_bytes, std::uint64_t stride) {
    std::mt19937 rng(c.seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(-0.05f, 0.05f);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    std::vector<std::byte> buf(c.offset + (c.rows - 1) * stride + row_bytes + 7);
    for (auto& b : buf) b = static_cast<std::byte>(byte(rng));
    auto put16 = [&](std::uint64_t at, float f) {
        const std::uint16_t h = tensor::fp32_to_fp16(f);
        buf[at] = static_cast<std::byte>(h & 0xFF);
        buf[at + 1] = static_cast<std::byte>(h >> 8);
    };
    for (std::uint32_t n = 0; n < c.rows; ++n) {
        const std::uint64_t row = c.offset + n * stride;
        switch (c.type) {
            case DType::F32:
                for (std::uint32_t i = 0; i < c.cols; ++i) {
                    const float f = (i == 3) ? 1e-40f : val(rng);  // one fp32 subnormal
                    std::memcpy(&buf[row + 4 * i], &f, 4);
                }
                break;
            case DType::F16:
                for (std::uint32_t i = 0; i < c.cols; ++i) put16(row + 2 * i, i == 5 ? 3e-6f : val(rng));
                break;
            case DType::Q8_0:
                for (std::uint32_t b = 0; b < c.cols / 32; ++b) put16(row + 34 * b, b == 0 ? 2e-6f : scale(rng));
                break;
            case DType::Q4_K:
            case DType::Q5_K: {
                const std::uint64_t bb = c.type == DType::Q4_K ? 144 : 176;
                for (std::uint32_t b = 0; b < c.cols / 256; ++b) {
                    put16(row + bb * b, b == 0 ? 4e-6f : scale(rng));
                    put16(row + bb * b + 2, std::fabs(scale(rng)));
                }
                break;
            }
            case DType::Q6_K:
                for (std::uint32_t b = 0; b < c.cols / 256; ++b) put16(row + 210 * b + 208, b == 0 ? 1e-6f : scale(rng));
                break;
            case DType::IQ4_NL:  // f16 d at the start of each 18-byte block of 32
                for (std::uint32_t b = 0; b < c.cols / 32; ++b) put16(row + 18 * b, b == 0 ? 2e-6f : scale(rng));
                break;
            case DType::IQ4_XS:  // f16 d at the start of each 136-byte block of 256
                for (std::uint32_t b = 0; b < c.cols / 256; ++b) put16(row + 136 * b, b == 0 ? 3e-6f : scale(rng));
                break;
            case DType::Q3_K:  // f16 d at byte 108 of each 110-byte block
                for (std::uint32_t b = 0; b < c.cols / 256; ++b) put16(row + 110 * b + 108, b == 0 ? 1e-6f : scale(rng));
                break;
            case DType::IQ3_S:  // f16 d at the start of each 110-byte block
                for (std::uint32_t b = 0; b < c.cols / 256; ++b) put16(row + 110 * b, b == 0 ? 2e-6f : scale(rng));
                break;
            case DType::Q4_0:  // f16 d at the start of each 18-byte block of 32
                for (std::uint32_t b = 0; b < c.cols / 32; ++b) put16(row + 18 * b, b == 0 ? 2e-6f : scale(rng));
                break;
            default:
                break;
        }
    }
    return buf;
}

struct GemvRef {
    std::vector<float> y;      // cpu::matmul on tensor::dequantize_row rows
    std::vector<double> mag;   // sum_i |x_i w_i| per output
};

GemvRef reference(const GemvCase& c, const std::vector<std::byte>& wb, std::uint64_t row_bytes, std::uint64_t stride,
                  const std::vector<float>& x) {
    std::vector<float> w(static_cast<std::size_t>(c.rows) * c.cols);
    for (std::uint32_t n = 0; n < c.rows; ++n) {
        tensor::dequantize_row(c.type, std::span<const std::byte>(&wb[c.offset + n * stride], row_bytes),
                               std::span<float>(&w[static_cast<std::size_t>(n) * c.cols], c.cols));
    }
    GemvRef r;
    r.y.assign(static_cast<std::size_t>(c.n_vec) * c.rows, 0.0f);
    cpu::matmul(cpu::ConstRows(x.data(), c.n_vec, c.cols, c.cols),
                cpu::WeightMatrix::dense(cpu::ConstRows(w.data(), c.rows, c.cols, c.cols)),
                cpu::Rows(r.y.data(), c.n_vec, c.rows, c.rows));
    r.mag.resize(r.y.size());
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        for (std::uint32_t n = 0; n < c.rows; ++n) {
            double m = 0.0;
            for (std::uint32_t i = 0; i < c.cols; ++i) {
                m += std::fabs(static_cast<double>(x[static_cast<std::size_t>(t) * c.cols + i]) *
                               static_cast<double>(w[static_cast<std::size_t>(n) * c.cols + i]));
            }
            r.mag[static_cast<std::size_t>(t) * c.rows + n] = m;
        }
    }
    return r;
}

::testing::AssertionResult within_bound(const GemvCase& c, const GemvRef& ref, const std::vector<float>& got) {
    const double u = std::ldexp(1.0, -24);
    const double depth = kern::gemv_cpu_depth(c.cols) + kern::gemv_wave_depth(c.cols);
    std::size_t bad = 0;
    std::size_t first = 0;
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double bound = depth * u * ref.mag[i] * (1.0 + 1e-3) + 1e-37;
        const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref.y[i]));
        worst_ratio = std::max(worst_ratio, d / bound);
        if (!(d <= bound)) {
            if (bad++ == 0) first = i;
        }
    }
    if (bad == 0) {
        std::printf("    wave vs cpu: worst |d| / bound = %.4f\n", worst_ratio);
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << bad << " of " << got.size() << " outputs outside the summation bound; first "
                                         << first << ": cpu " << ref.y[first] << " hip " << got[first];
}

void check_gemv(const GemvCase& c, const std::string& variant, Runner& r) {
    SCOPED_TRACE(r.name() + " " + variant);
    SCOPED_TRACE(testing::Message() << "type " << traits(c.type).name << " rows " << c.rows << " cols " << c.cols
                                    << " n_vec " << c.n_vec << " pad " << c.stride_pad << " offset " << c.offset);
    const std::uint64_t row_bytes = tensor::row_bytes(c.type, c.cols);
    const std::uint64_t stride = row_bytes + c.stride_pad;
    std::vector<std::byte> wbytes = make_weights(c, row_bytes, stride);
    std::mt19937 rng(c.seed + 1000);
    std::vector<float> x = uniform(rng, static_cast<std::size_t>(c.n_vec) * c.cols, -1.0f, 1.0f);
    const GemvRef ref = reference(c, wbytes, row_bytes, stride, x);

    // Weight bytes go through a float vector so both runners can hold them.
    std::vector<float> wfl((wbytes.size() + 3) / 4, 0.0f);
    std::memcpy(wfl.data(), wbytes.data(), wbytes.size());
    // Strided x and y: vector t starts at t * (cols + x_pad) / t * (rows + y_pad); the
    // padding of y holds a sentinel that must survive.
    const std::size_t xs = c.cols + c.x_pad;
    const std::size_t ys = c.rows + c.y_pad;
    std::vector<float> xh(c.n_vec * xs, 12345.0f);
    for (std::uint32_t t = 0; t < c.n_vec; ++t) std::copy_n(&x[t * c.cols], c.cols, &xh[t * xs]);
    std::vector<float> y(c.n_vec * ys, 999.0f);
    const Buffer bw = r.make(wfl), bx = r.make(xh), by = r.make(y);
    GemvArgs a;
    a.wtype = c.type;
    a.w = BufferView(bw, c.offset, (c.rows - 1) * stride + row_bytes, stride);
    a.x = BufferView(bx, 0, 0, xs * sizeof(float));
    a.y = BufferView(by, 0, 0, ys * sizeof(float));
    a.rows = c.rows;
    a.cols = c.cols;
    a.n_vec = c.n_vec;
    OpsOptions o;
    o.gemv = variant;
    Ops(o).gemv(r.target(), a);
    r.finish();
    r.fetch(by, y);
    std::vector<float> yd(static_cast<std::size_t>(c.n_vec) * c.rows);
    std::size_t pad_bad = 0;
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        std::copy_n(&y[t * ys], c.rows, &yd[t * c.rows]);
        for (std::size_t i = c.rows; i < ys; ++i) pad_bad += y[t * ys + i] != 999.0f ? 1u : 0u;
    }
    EXPECT_EQ(pad_bad, 0u) << "y padding overwritten";
    if (variant == "gemv_generic_b64") {
        EXPECT_TRUE(matches(ref.y, yd, true, "y (generic, bitwise vs cpu::matmul)"));
    } else {
        EXPECT_TRUE(within_bound(c, ref, yd));
    }
}

std::vector<GemvCase> gemv_cases() {
    std::vector<GemvCase> v;
    std::uint32_t seed = 100;
    // D-014 order, then the second tier the UD-Q4_K_XL pack needs (IQ4_XS, IQ4_NL, Q3_K, IQ3_S).
    for (DType t : {DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::Q8_0, DType::F16, DType::F32, DType::IQ4_XS,
                    DType::IQ4_NL, DType::Q3_K, DType::IQ3_S, DType::Q4_0}) {  // Q4_0: ggml-org MTP pack (D-006)
        const bool plain = t == DType::F32 || t == DType::F16;
        GemvCase a{.type = t, .rows = 37, .cols = plain ? 100u : 512u, .seed = ++seed};  // plain: dot tail (100 % 8)
        v.push_back(a);
        // qwen35 hidden size K = 5120, MTP-verify shaped (3 vectors), rows not a multiple of
        // any rows-per-block, unaligned row stride and start (F32 keeps 4-byte alignment),
        // strided x and y views.
        GemvCase b{.type = t, .rows = 13, .cols = 5120, .n_vec = 3, .stride_pad = plain ? 4u : 3u,
                   .offset = plain ? 4u : 1u, .x_pad = 5, .y_pad = 3, .seed = ++seed};
        v.push_back(b);
    }
    return v;
}

const std::vector<std::string> kVariants{"gemv_generic_b64", "gemv_wave32_r4", "gemv_wave32_r8"};

TEST(HipGemvEmu, AllTypesVsTensorDequantAndCpuMatmul) {
    for (const GemvCase& c : gemv_cases()) {
        for (const std::string& v : kVariants) {
            for (auto& r : emulation_runners()) check_gemv(c, v, *r);
        }
    }
}

TEST(HipGemvDevice, AllTypesVsTensorDequantAndCpuMatmul) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    // Device expectation: the generic variant uses only IEEE fp32 multiply/add (no
    // transcendentals), so it is expected to stay bit-identical on the device too; a
    // mismatch there is a finding (e.g. denormal flushing), not a tolerance to widen.
    for (const GemvCase& c : gemv_cases()) {
        for (const std::string& v : kVariants) check_gemv(c, v, r);
    }
}

TEST(HipGemvEmu, RejectsUnsupportedTypesAndBadShapes) {
    std::vector<float> f(1u << 14, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * sizeof(float));
    const Ops ops;
    const Target t = Target::emulation();
    GemvArgs a;
    a.wtype = DType::Q8_0;
    a.w = BufferView(b, 0, 34 * 8);  // 2 rows x 4 blocks... resized below
    a.x = BufferView(b, 8192, 512);
    a.y = BufferView(b, 12288, 64);
    a.rows = 2;
    a.cols = 128;
    ASSERT_NO_THROW(ops.gemv(t, a));  // valid baseline
    auto expect = [&](ErrorCode code, const char* needle) {
        try {
            ops.gemv(t, a);
            ADD_FAILURE() << "no error; expected: " << needle;
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), code) << e.what();
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    a.wtype = DType::Q4_1;  // in D-007, not a HIP GEMV type
    expect(ErrorCode::Unsupported, "no HIP GEMV kernel");
    a.wtype = DType::Q8_0;
    a.cols = 100;
    expect(ErrorCode::Kernel, "not a multiple of the block size 32");
    a.cols = 128;
    a.y = BufferView(b, 16, 64);  // inside w
    expect(ErrorCode::Kernel, "y overlaps w");
    a.y = BufferView(b, 12288, 64);
    a.w = BufferView(b, 0, 34 * 7);  // one block short
    expect(ErrorCode::Kernel, "w needs 272 bytes");
    a.w = BufferView(b, 0, 34 * 8);
    a.wtype = DType::F32;
    a.cols = 64;
    a.w = BufferView(b, 2, 0);  // F32 weights must be 4-byte aligned
    expect(ErrorCode::Kernel, "w is not 4-byte aligned");
}

}  // namespace
}  // namespace halo::hip::test
