// Decode primitives of the HIP backend: GET_ROWS (embedding lookup), ADD and the fused
// ADD + RMS_NORM, differentially tested against halo::tensor / halo::cpu on the same inputs
// (TRD §30, review M-2). Emulation runs both thread/workgroup orders and must be bitwise
// equal; device tests skip on the dev host (D-001). KV write is in test_hip_kv.cpp.

#include <cmath>
#include <cstring>

#include "halo/backends/cpu/ops.h"
#include "halo/tensor/fp16.h"
#include "halo/tensor/quant.h"
#include "hip_test_util.h"

namespace halo::hip::test {
namespace {

// ---- GET_ROWS -----------------------------------------------------------------------------

/// Random valid rows of `t`: random quant bytes, finite f16 scales written at every block's
/// scale offset (the per-type offsets of ggml's block structs).
std::vector<std::byte> table_bytes(DType t, std::uint32_t rows, std::uint32_t cols, std::mt19937& rng) {
    const std::size_t rb = tensor::row_bytes(t, cols);
    std::vector<std::byte> buf(rows * rb + 4);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(-0.05f, 0.05f), val(-1.0f, 1.0f);
    for (auto& b : buf) b = static_cast<std::byte>(byte(rng));
    auto put16 = [&](std::size_t at, float f) {
        const std::uint16_t h = tensor::fp32_to_fp16(f);
        buf[at] = static_cast<std::byte>(h & 0xFF);
        buf[at + 1] = static_cast<std::byte>(h >> 8);
    };
    const std::size_t be = traits(t).block_elems, bb = traits(t).block_bytes;
    for (std::uint32_t n = 0; n < rows; ++n) {
        const std::size_t row = n * rb;
        for (std::size_t b = 0; b < cols / be; ++b) {
            const std::size_t blk = row + b * bb;
            switch (t) {
                case DType::F32: {
                    const float f = val(rng);
                    std::memcpy(&buf[blk], &f, 4);
                    break;
                }
                case DType::F16: put16(blk, val(rng)); break;
                case DType::Q4_0:
                case DType::Q8_0:
                case DType::IQ4_NL:
                case DType::IQ4_XS:
                case DType::IQ3_S: put16(blk, sc(rng)); break;
                case DType::Q4_K:
                case DType::Q5_K:
                    put16(blk, sc(rng));
                    put16(blk + 2, std::fabs(sc(rng)));
                    break;
                case DType::Q6_K: put16(blk + 208, sc(rng)); break;
                case DType::Q3_K: put16(blk + 108, sc(rng)); break;
                default: ADD_FAILURE() << "type not covered"; break;
            }
        }
    }
    return buf;
}

void check_get_rows(DType t, std::uint32_t cols, Runner& r) {
    SCOPED_TRACE(r.name() + " get_rows " + std::string(traits(t).name));
    constexpr std::uint32_t kRows = 97;
    std::mt19937 rng(1000 + static_cast<std::uint32_t>(t));
    std::vector<std::byte> wb = table_bytes(t, kRows, cols, rng);
    const std::size_t rb = tensor::row_bytes(t, cols);
    const std::vector<std::int32_t> ids{0, 96, 17, 17, 50, 3};  // first, last, repeated
    std::vector<float> ref(ids.size() * cols);
    for (std::size_t k = 0; k < ids.size(); ++k) {
        tensor::dequantize_row(t, std::span<const std::byte>(&wb[static_cast<std::size_t>(ids[k]) * rb], rb),
                               std::span<float>(&ref[k * cols], cols));
    }
    std::vector<float> wfl((wb.size() + 3) / 4);
    std::memcpy(wfl.data(), wb.data(), wb.size());
    std::vector<float> idf(ids.size());
    std::memcpy(idf.data(), ids.data(), ids.size() * 4);
    std::vector<float> out(ref.size(), 999.0f);
    std::vector<std::uint32_t> status{0xDEADBEEFu};
    const Buffer bw = r.make(wfl), bi = r.make(idf), bo = r.make(out), bs = r.make_words(status);
    GetRowsArgs a;
    a.wtype = t;
    a.w = BufferView(bw, 0, kRows * rb);
    a.n_rows = kRows;
    a.cols = cols;
    a.ids = bi;
    a.n_ids = static_cast<std::uint32_t>(ids.size());
    a.out = bo;
    a.status = bs;
    Ops().get_rows(r.target(), a);
    r.finish();
    EXPECT_NO_THROW(Ops::check_status(r.target(), bs));
    r.fetch(bo, out);
    // Pure dequantization, no arithmetic reordering: bitwise on the device too.
    EXPECT_TRUE(matches(ref, out, true, "get_rows vs tensor::dequantize_row"));
}

const std::vector<DType> kTypes{DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::Q8_0, DType::F16, DType::F32,
                                DType::IQ4_XS, DType::IQ4_NL, DType::Q3_K, DType::IQ3_S, DType::Q4_0};

TEST(HipDecodeEmu, GetRowsAllTypesBitExactVsDequantizeRow) {
    for (DType t : kTypes) {
        for (auto& r : emulation_runners()) check_get_rows(t, t == DType::F32 || t == DType::F16 ? 136 : 512, *r);
    }
    // The pack's token_embd is Q4_K at hidden 5120 (D-007 census).
    for (auto& r : emulation_runners()) check_get_rows(DType::Q4_K, 5120, *r);
}

TEST(HipDecodeDevice, GetRowsAllTypesVsDequantizeRow) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (DType t : kTypes) check_get_rows(t, t == DType::F32 || t == DType::F16 ? 136 : 512, r);
}

TEST(HipDecodeEmu, GetRowsBadIdRaises) {
    std::vector<float> f(4096, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * 4);
    const std::int32_t bad[2] = {1, 8};  // 8 rows: id 8 is out of range
    std::memcpy(&f[2000], bad, sizeof(bad));
    GetRowsArgs a;
    a.wtype = DType::F32;
    a.w = BufferView(b, 0, 8 * 16 * 4);
    a.n_rows = 8;
    a.cols = 16;
    a.ids = BufferView(b, 8000, 8);
    a.n_ids = 2;
    a.out = BufferView(b, 12000, 128);
    a.status = BufferView(b, 15000, 4);
    Ops().get_rows(Target::emulation(), a);
    try {
        Ops::check_status(Target::emulation(), a.status);
        ADD_FAILURE() << "id 8 of 8 accepted";
    } catch (const Error& e) {
        EXPECT_NE(std::string(e.what()).find("get_rows id outside the table"), std::string::npos) << e.what();
    }
}

// ---- ADD and ADD + RMS_NORM -------------------------------------------------------------------

void check_add_norm(std::uint32_t rows, std::uint32_t cols, int alias, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "add_rms_norm rows " << rows << " cols " << cols << " alias " << alias);
    std::mt19937 rng(1100 + cols);
    std::vector<float> a = uniform(rng, rows * cols, -3.0f, 3.0f), b = uniform(rng, rows * cols, -3.0f, 3.0f);
    std::vector<float> w = uniform(rng, cols, 0.5f, 1.5f);
    std::vector<float> h(rows * cols, 999.0f), y(rows * cols, 999.0f);
    constexpr float kEps = 1e-6f;
    // CPU composition: cpu::add then cpu::rms_norm.
    auto ra = a, rb = b, rh = h, ry = y;
    {
        cpu::Rows xa(ra.data(), rows, cols, cols), xb(rb.data(), rows, cols, cols), xh(rh.data(), rows, cols, cols);
        const cpu::Rows hh = alias == 1 ? xa : alias == 2 ? xb : xh;
        cpu::add(xa, xb, hh);
        cpu::rms_norm(hh, w, kEps, cpu::Rows(ry.data(), rows, cols, cols));
    }
    const Buffer ba = r.make(a), bb = r.make(b), bh = r.make(h), bw = r.make(w), by = r.make(y);
    AddRmsNormArgs args;
    args.a = ba;
    args.b = bb;
    args.h = alias == 1 ? BufferView(ba) : alias == 2 ? BufferView(bb) : BufferView(bh);
    args.w = bw;
    args.y = by;
    args.rows = rows;
    args.cols = cols;
    args.eps = kEps;
    Ops().add_rms_norm(r.target(), args);
    r.finish();
    r.fetch(ba, a);
    r.fetch(bb, b);
    r.fetch(bh, h);
    r.fetch(by, y);
    EXPECT_TRUE(matches(ra, a, r.exact(), "a"));
    EXPECT_TRUE(matches(rb, b, r.exact(), "b"));
    EXPECT_TRUE(matches(rh, h, r.exact(), "h (residual)"));
    EXPECT_TRUE(matches(ry, y, r.exact(), "y (normed)"));

    // Plain ADD on the same inputs (cpu::add).
    std::vector<float> a2 = uniform(rng, rows * cols, -3.0f, 3.0f), o2(rows * cols, 999.0f), r2(rows * cols);
    cpu::add(cpu::ConstRows(a2.data(), rows, cols, cols), cpu::ConstRows(ra.data(), rows, cols, cols),
             cpu::Rows(r2.data(), rows, cols, cols));
    std::vector<float> ra_copy = ra;
    const Buffer b2 = r.make(a2), bra = r.make(ra_copy), bo2 = r.make(o2);
    EltwiseArgs e;
    e.a = b2;
    e.b = bra;
    e.out = bo2;
    e.rows = rows;
    e.cols = cols;
    Ops().add(r.target(), e);
    r.finish();
    r.fetch(bo2, o2);
    EXPECT_TRUE(matches(r2, o2, r.exact(), "add"));
}

TEST(HipDecodeEmu, AddAndFusedAddRmsNormBitExactVsCpuComposition) {
    for (auto& r : emulation_runners()) {
        check_add_norm(1, 5120, 1, *r);  // decode residual in place (h aliases a), hidden 5120
        check_add_norm(3, 5120, 0, *r);
        check_add_norm(4, 37, 2, *r);    // 8-lane tail, h aliases b
    }
}

TEST(HipDecodeDevice, AddAndFusedAddRmsNormVsCpuComposition) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    check_add_norm(1, 5120, 1, r);
    check_add_norm(3, 5120, 0, r);
}

}  // namespace
}  // namespace halo::hip::test
