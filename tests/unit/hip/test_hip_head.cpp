// LM head (logits GEMV + fused argmax), ARGMAX, TOP_K, RMS_NORM, PARTIAL_ROPE, SWIGLU and
// MUL_SIGMOID of the HIP backend, differentially tested against halo::cpu / halo::tensor on
// the same inputs (TRD §19, §30; D-016; review M-2).
//
// Emulation cases run in forward and reverse thread/workgroup order and must equal the CPU
// bit for bit, except the LM head with a wave32 GEMV variant: its logits use a different
// (fixed) summation order, so its value is accepted within the M2 summation bound and its
// index must equal the CPU's (the test asserts the CPU's top-2 gap exceeds twice the bound,
// so the index is determined). Device tests skip on the dev host (D-001).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "halo/backends/cpu/ops.h"
#include "halo/tensor/fp16.h"
#include "halo/tensor/quant.h"
#include "hip_test_util.h"
#include "kernels/gemv.h"

namespace halo::hip::test {
namespace {

std::vector<std::uint32_t> words_zero(std::size_t n) { return std::vector<std::uint32_t>(n, 0xA5A5A5A5u); }

template <class F>
void expect_kernel_error(F&& f, const char* needle) {
    try {
        f();
        ADD_FAILURE() << "no error; expected: " << needle;
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Kernel) << e.what();
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

// ---------------------------------------------------------------------------------------
// LM head
// ---------------------------------------------------------------------------------------

struct HeadCase {
    DType type = DType::Q6_K;
    std::uint32_t rows = 1003;
    std::uint32_t cols = 512;
    std::uint32_t n_vec = 2;
    bool write_logits = true;
    int special = 0;  // 1 = duplicate the winning row at a higher index (tie), 2 = NaN weight
    std::uint32_t seed = 1;
};

/// Random, valid weight rows of `type` (finite f16 scales); dense.
std::vector<std::byte> random_rows(DType t, std::uint32_t rows, std::uint32_t cols, std::mt19937& rng) {
    const std::size_t rb = tensor::row_bytes(t, cols);
    std::vector<std::byte> buf(rows * rb + 4);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(-0.05f, 0.05f);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    for (auto& b : buf) b = static_cast<std::byte>(byte(rng));
    auto put16 = [&](std::size_t at, float f) {
        const std::uint16_t h = tensor::fp32_to_fp16(f);
        buf[at] = static_cast<std::byte>(h & 0xFF);
        buf[at + 1] = static_cast<std::byte>(h >> 8);
    };
    for (std::uint32_t n = 0; n < rows; ++n) {
        const std::size_t row = n * rb;
        switch (t) {
            case DType::F32:
                for (std::uint32_t i = 0; i < cols; ++i) {
                    const float f = val(rng);
                    std::memcpy(&buf[row + 4 * i], &f, 4);
                }
                break;
            case DType::F16:
                for (std::uint32_t i = 0; i < cols; ++i) put16(row + 2 * i, val(rng));
                break;
            case DType::Q4_K:
                for (std::uint32_t b = 0; b < cols / 256; ++b) {
                    put16(row + 144 * b, scale(rng));
                    put16(row + 144 * b + 2, std::fabs(scale(rng)));
                }
                break;
            case DType::Q6_K:
                for (std::uint32_t b = 0; b < cols / 256; ++b) put16(row + 210 * b + 208, scale(rng));
                break;
            default:
                ADD_FAILURE() << "unsupported test type";
        }
    }
    return buf;
}

void check_head(const HeadCase& c, const std::string& variant, Runner& r) {
    SCOPED_TRACE(r.name() + " " + variant);
    SCOPED_TRACE(testing::Message() << "type " << traits(c.type).name << " rows " << c.rows << " cols " << c.cols
                                    << " n_vec " << c.n_vec << " special " << c.special);
    std::mt19937 rng(c.seed);
    std::vector<std::byte> wb = random_rows(c.type, c.rows, c.cols, rng);
    const std::size_t rb = tensor::row_bytes(c.type, c.cols);
    std::vector<float> x = uniform(rng, static_cast<std::size_t>(c.n_vec) * c.cols, -1.0f, 1.0f);
    std::vector<float> w(static_cast<std::size_t>(c.rows) * c.cols);
    auto dequant_all = [&] {
        for (std::uint32_t n = 0; n < c.rows; ++n) {
            tensor::dequantize_row(c.type, std::span<const std::byte>(&wb[n * rb], rb),
                                   std::span<float>(&w[static_cast<std::size_t>(n) * c.cols], c.cols));
        }
    };
    dequant_all();
    const cpu::WeightMatrix wm = cpu::WeightMatrix::dense(cpu::ConstRows(w.data(), c.rows, c.cols, c.cols));
    if (c.special == 1) {
        // Make the winner of vector 0 appear twice: copy its row to a higher index.
        const cpu::TopKEntry best = cpu::matmul_argmax(std::span<const float>(x.data(), c.cols), wm);
        const std::uint32_t dup = c.rows - 2;
        ASSERT_LT(static_cast<std::uint32_t>(best.index), dup);
        std::memcpy(&wb[dup * rb], &wb[static_cast<std::size_t>(best.index) * rb], rb);
        dequant_all();
    }
    if (c.special == 2) {
        ASSERT_EQ(c.type, DType::F32);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(&wb[500 * rb + 4 * 3], &nan, 4);
        dequant_all();
    }
    // CPU reference per vector.
    std::vector<cpu::TopKEntry> ref(c.n_vec);
    bool cpu_nan = false;
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        try {
            ref[t] = cpu::matmul_argmax(std::span<const float>(&x[static_cast<std::size_t>(t) * c.cols], c.cols), wm);
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Kernel);
            cpu_nan = true;
        }
    }
    std::vector<float> wfl((wb.size() + 3) / 4);
    std::memcpy(wfl.data(), wb.data(), wb.size());
    std::vector<float> logits(static_cast<std::size_t>(c.n_vec) * c.rows, 999.0f);
    OpsOptions o;
    o.gemv = variant;
    const Ops ops(o);
    std::vector<float> ws(ops.lm_head_workspace_bytes(c.rows, c.n_vec) / 4, 0.0f);
    std::vector<std::uint32_t> res = words_zero(3 * c.n_vec);
    const Buffer bw = r.make(wfl), bx = r.make(x), by = r.make(logits), bws = r.make(ws), bres = r.make_words(res);
    LmHeadArgs a;
    a.gemv.wtype = c.type;
    a.gemv.w = BufferView(bw, 0, c.rows * rb);
    a.gemv.x = bx;
    if (c.write_logits) a.gemv.y = by;
    a.gemv.rows = c.rows;
    a.gemv.cols = c.cols;
    a.gemv.n_vec = c.n_vec;
    a.workspace = bws;
    a.result = bres;
    ops.lm_head_argmax(r.target(), a);
    r.finish();
    if (cpu_nan) {
        expect_kernel_error([&] { static_cast<void>(Ops::read_argmax(r.target(), bres, c.n_vec)); }, "NaN logit");
        return;
    }
    const std::vector<ArgmaxResult> got = Ops::read_argmax(r.target(), bres, c.n_vec);
    const bool exact = variant == "gemv_generic_b64";
    const double u = std::ldexp(1.0, -24);
    const double depth = kern::gemv_cpu_depth(c.cols) + kern::gemv_wave_depth(c.cols);
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        SCOPED_TRACE(testing::Message() << "vector " << t);
        EXPECT_EQ(got[t].index, static_cast<std::uint32_t>(ref[t].index));
        if (exact) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(got[t].value), std::bit_cast<std::uint32_t>(ref[t].value))
                << got[t].value << " vs " << ref[t].value;
            continue;
        }
        // Wave variant: the value within the summation bound, and the CPU's top-2 gap wider
        // than twice the bound (so the index is determined by the data, not by rounding).
        std::vector<float> row(c.rows);
        cpu::matmul(cpu::ConstRows(&x[static_cast<std::size_t>(t) * c.cols], 1, c.cols, c.cols), wm,
                    cpu::Rows(row.data(), 1, c.rows, c.rows));
        double mag = 0.0;
        double worst_bound = 0.0;
        for (std::uint32_t n = 0; n < c.rows; ++n) {
            double m = 0.0;
            for (std::uint32_t i = 0; i < c.cols; ++i) {
                m += std::fabs(static_cast<double>(x[static_cast<std::size_t>(t) * c.cols + i]) *
                               static_cast<double>(w[static_cast<std::size_t>(n) * c.cols + i]));
            }
            if (n == static_cast<std::uint32_t>(ref[t].index)) mag = m;
            worst_bound = std::max(worst_bound, depth * u * m * (1 + 1e-3));
        }
        std::vector<float> sorted = row;
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        if (c.special != 1) {
            ASSERT_GT(static_cast<double>(sorted[0]) - static_cast<double>(sorted[1]), 2.0 * worst_bound)
                << "test data has a near-tie; the index would be rounding-determined";
        }
        EXPECT_LE(std::fabs(static_cast<double>(got[t].value) - static_cast<double>(ref[t].value)),
                  depth * u * mag * (1 + 1e-3));
    }
    if (c.write_logits) {
        r.fetch(by, logits);
        std::vector<float> rl(logits.size());
        cpu::matmul(cpu::ConstRows(x.data(), c.n_vec, c.cols, c.cols), wm, cpu::Rows(rl.data(), c.n_vec, c.rows, c.rows));
        if (exact) EXPECT_TRUE(matches(rl, logits, true, "logits (generic)"));
    }
}

std::vector<HeadCase> head_cases() {
    return {
        {.type = DType::Q6_K, .rows = 1003, .cols = 512, .n_vec = 2, .seed = 201},  // D-007: the LM head is Q6_K
        {.type = DType::Q4_K, .rows = 517, .cols = 768, .n_vec = 1, .write_logits = false, .seed = 202},
        {.type = DType::F16, .rows = 300, .cols = 136, .n_vec = 3, .seed = 203},
        {.type = DType::F32, .rows = 777, .cols = 64, .n_vec = 1, .special = 1, .seed = 204},  // exact tie
        {.type = DType::F32, .rows = 777, .cols = 64, .n_vec = 2, .special = 2, .seed = 205},  // NaN -> error
        // Production head shape: 248,320 rows (D-008) -> ~62K stage-1 partials per vector.
        {.type = DType::F32, .rows = 248320, .cols = 16, .n_vec = 2, .write_logits = false, .seed = 206},
    };
}

const std::vector<std::string> kGemvVariants{"gemv_generic_b64", "gemv_wave32_r4", "gemv_wave32_r8"};

TEST(HipHeadEmu, LmHeadArgmaxVsCpuMatmulArgmax) {
    for (const HeadCase& c : head_cases()) {
        for (const std::string& v : kGemvVariants) {
            for (auto& r : emulation_runners()) check_head(c, v, *r);
        }
    }
}

TEST(HipHeadDevice, LmHeadArgmaxVsCpuMatmulArgmax) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const HeadCase& c : head_cases()) {
        for (const std::string& v : kGemvVariants) check_head(c, v, r);
    }
}

// M2: rows at/after valid_rows (GGUF LM-head padding) never win the fused argmax, but their
// raw values still land in the logits; mirrors the CPU backend test of the same name.
void check_valid_rows(const std::string& variant, Runner& r) {
    SCOPED_TRACE(r.name() + " " + variant);
    constexpr std::uint32_t kRows = 20, kCols = 4, kValid = 15;
    std::mt19937 rng(911);
    std::vector<float> w = uniform(rng, kRows * kCols, -1.0f, 1.0f);
    std::vector<float> x(kCols, 0.0f);
    x[0] = 1.0f;             // one-hot: every weight row's dot product is its w[row][0]
    w[10 * kCols] = 1.0f;    // best real row (< kValid)
    w[17 * kCols] = 100.0f;  // padded row (>= kValid): an unclamped argmax would pick it
    std::vector<float> logits(kRows, 999.0f);
    OpsOptions o;
    o.gemv = variant;
    const Ops ops(o);
    std::vector<float> ws(ops.lm_head_workspace_bytes(kRows, 1) / 4, 0.0f);
    std::vector<std::uint32_t> res = words_zero(3);
    const Buffer bw = r.make(w), bx = r.make(x), by = r.make(logits), bws = r.make(ws), bres = r.make_words(res);
    LmHeadArgs a;
    a.gemv.wtype = DType::F32;
    a.gemv.w = BufferView(bw, 0, kRows * kCols * sizeof(float));
    a.gemv.x = bx;
    a.gemv.y = by;
    a.gemv.rows = kRows;
    a.gemv.cols = kCols;
    a.gemv.n_vec = 1;
    a.workspace = bws;
    a.result = bres;
    a.valid_rows = kValid;
    ops.lm_head_argmax(r.target(), a);
    r.finish();
    const std::vector<ArgmaxResult> got = Ops::read_argmax(r.target(), bres, 1);
    EXPECT_EQ(got[0].index, 10u) << "the padded row (17) must never win, even though its raw logit is larger";
    EXPECT_EQ(std::bit_cast<std::uint32_t>(got[0].value), std::bit_cast<std::uint32_t>(1.0f));
    r.fetch(by, logits);
    EXPECT_EQ(logits[17], 100.0f) << "the padded row's raw logit is still reported";
    // valid_rows == 0 is "no clamp": the larger (padded) logit wins as before.
    a.valid_rows = 0;
    ops.lm_head_argmax(r.target(), a);
    r.finish();
    const std::vector<ArgmaxResult> unclamped = Ops::read_argmax(r.target(), bres, 1);
    EXPECT_EQ(unclamped[0].index, 17u);
    // The clamp does not hide poison: a NaN in a padded row still fails the vector.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(&w[18 * kCols], &nan, 4);
    const Buffer bn = r.make(w);
    a.valid_rows = kValid;
    a.gemv.w = BufferView(bn, 0, kRows * kCols * sizeof(float));
    ops.lm_head_argmax(r.target(), a);
    r.finish();
    expect_kernel_error([&] { static_cast<void>(Ops::read_argmax(r.target(), bres, 1)); }, "NaN logit");
}

TEST(HipHeadEmu, LmHeadValidRowsExcludesPaddedRowsFromArgmaxButNotFromLogits) {
    for (const std::string& v : kGemvVariants) {
        for (auto& r : emulation_runners()) check_valid_rows(v, *r);
    }
}

TEST(HipHeadDevice, LmHeadValidRowsExcludesPaddedRowsFromArgmaxButNotFromLogits) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const std::string& v : kGemvVariants) check_valid_rows(v, r);
}

// ---------------------------------------------------------------------------------------
// ARGMAX and TOP_K over logits vectors
// ---------------------------------------------------------------------------------------

struct LogitsCase {
    std::uint32_t n = 248320;  // qwen35 vocabulary (D-008)
    std::uint32_t n_vec = 2;
    std::uint32_t pad = 0;     // extra floats between vectors
    std::uint32_t k = 40;
    float step = 0.0f;         // > 0: quantize logits to multiples of step (many exact ties)
    int special = 0;           // 1 = all -inf, 2 = NaN at one position, 3 = max tied at two positions
    std::uint32_t seed = 1;
};

std::vector<float> make_logits(const LogitsCase& c) {
    std::mt19937 rng(c.seed);
    const std::size_t s = c.n + c.pad;
    std::vector<float> v = uniform(rng, c.n_vec * s, -20.0f, 20.0f);
    for (float& f : v) {
        if (c.step > 0.0f) f = std::round(f / c.step) * c.step;
        if (c.special == 1) f = -std::numeric_limits<float>::infinity();
    }
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        float* row = &v[t * s];
        if (c.special == 2) row[c.n / 3] = std::numeric_limits<float>::quiet_NaN();
        if (c.special == 3) {
            row[c.n - 5] = 50.0f;
            row[c.n / 2 + t] = 50.0f;  // the lower index must win
        }
    }
    return v;
}

void check_argmax(const LogitsCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "argmax n " << c.n << " n_vec " << c.n_vec << " special " << c.special);
    std::vector<float> lg = make_logits(c);
    const std::size_t s = c.n + c.pad;
    const Ops ops;
    std::vector<float> ws(ops.argmax_workspace_bytes(c.n, c.n_vec) / 4, 0.0f);
    std::vector<std::uint32_t> res = words_zero(3 * c.n_vec);
    const Buffer bl = r.make(lg), bws = r.make(ws), bres = r.make_words(res);
    ArgmaxArgs a;
    a.logits = BufferView(bl, 0, 0, s * sizeof(float));
    a.n = c.n;
    a.n_vec = c.n_vec;
    a.workspace = bws;
    a.result = bres;
    ops.argmax(r.target(), a);
    r.finish();
    if (c.special == 2) {
        expect_kernel_error([&] { static_cast<void>(Ops::read_argmax(r.target(), bres, c.n_vec)); }, "NaN logit");
        EXPECT_THROW(static_cast<void>(cpu::argmax(std::span<const float>(lg.data(), c.n))), Error);
        return;
    }
    const std::vector<ArgmaxResult> got = Ops::read_argmax(r.target(), bres, c.n_vec);
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        const cpu::TopKEntry ref = cpu::argmax(std::span<const float>(&lg[t * s], c.n));
        EXPECT_EQ(got[t].index, static_cast<std::uint32_t>(ref.index)) << "vector " << t;
        EXPECT_EQ(std::bit_cast<std::uint32_t>(got[t].value), std::bit_cast<std::uint32_t>(ref.value));
    }
}

void check_topk(const LogitsCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "top_k n " << c.n << " k " << c.k << " n_vec " << c.n_vec << " step " << c.step
                                    << " special " << c.special);
    std::vector<float> lg = make_logits(c);
    const std::size_t s = c.n + c.pad;
    std::vector<float> ws(topk_workspace_bytes(c.n, c.k, c.n_vec) / 4 + 1, 0.0f);
    std::vector<float> ids(c.n_vec * c.k, -7.0f), vals(c.n_vec * c.k, -7.0f);
    std::vector<std::uint32_t> status{0xDEADBEEFu};
    const Buffer bl = r.make(lg), bws = r.make(ws), bi = r.make(ids), bv = r.make(vals), bs = r.make_words(status);
    TopKArgs a;
    a.logits = BufferView(bl, 0, 0, s * sizeof(float));
    a.n = c.n;
    a.k = c.k;
    a.n_vec = c.n_vec;
    a.workspace = bws;
    a.ids = bi;
    a.values = bv;
    a.status = bs;
    Ops().top_k(r.target(), a);
    r.finish();
    if (c.special == 2) {
        expect_kernel_error([&] { Ops::check_status(r.target(), bs); }, "NaN");
        EXPECT_THROW(static_cast<void>(cpu::top_k(std::span<const float>(lg.data(), c.n), c.k)), Error);
        return;
    }
    EXPECT_NO_THROW(Ops::check_status(r.target(), bs));
    r.fetch(bi, ids);
    r.fetch(bv, vals);
    for (std::uint32_t t = 0; t < c.n_vec; ++t) {
        const std::vector<cpu::TopKEntry> ref = cpu::top_k(std::span<const float>(&lg[t * s], c.n), c.k);
        std::size_t bad = 0;
        for (std::uint32_t e = 0; e < c.k; ++e) {
            const std::int32_t gi = std::bit_cast<std::int32_t>(ids[t * c.k + e]);
            const bool ok = gi == ref[e].index &&
                            std::bit_cast<std::uint32_t>(vals[t * c.k + e]) == std::bit_cast<std::uint32_t>(ref[e].value);
            if (!ok && bad++ == 0) {
                ADD_FAILURE() << "vector " << t << " rank " << e << ": cpu (" << ref[e].index << ", " << ref[e].value
                              << ") hip (" << gi << ", " << vals[t * c.k + e] << ")";
            }
        }
        EXPECT_EQ(bad, 0u) << "vector " << t;
    }
}

std::vector<LogitsCase> argmax_cases() {
    return {
        {.n = 248320, .n_vec = 3, .pad = 7, .seed = 301},
        {.n = 4099, .n_vec = 2, .special = 3, .seed = 302},  // tied maximum -> lower index
        {.n = 1000, .n_vec = 1, .special = 1, .seed = 303},  // all -inf -> index 0
        {.n = 70000, .n_vec = 2, .special = 2, .seed = 304}, // NaN -> Error(Kernel)
        {.n = 1, .n_vec = 1, .seed = 305},
    };
}

std::vector<LogitsCase> topk_cases() {
    return {
        {.n = 248320, .n_vec = 2, .pad = 3, .k = 40, .seed = 401},
        {.n = 248320, .n_vec = 1, .k = 1024, .step = 0.25f, .seed = 402},  // 8 rounds, heavy exact ties
        {.n = 248320, .n_vec = 1, .k = 1, .seed = 403},
        {.n = 2049, .n_vec = 2, .k = 1024, .step = 1.0f, .seed = 404},  // 2 rounds, just over one chunk
        {.n = 100, .n_vec = 1, .k = 100, .seed = 405},                  // k = n, one round
        {.n = 5000, .n_vec = 1, .k = 7, .special = 1, .seed = 406},     // all -inf: indices 0..6
        {.n = 9000, .n_vec = 1, .k = 10, .special = 2, .seed = 407},    // NaN -> Error(Kernel)
    };
}

TEST(HipHeadEmu, ArgmaxBitExactVsCpu) {
    for (const LogitsCase& c : argmax_cases()) {
        for (auto& r : emulation_runners()) check_argmax(c, *r);
    }
}

TEST(HipHeadEmu, TopKBitExactVsCpu) {
    for (const LogitsCase& c : topk_cases()) {
        for (auto& r : emulation_runners()) check_topk(c, *r);
    }
}

TEST(HipHeadDevice, ArgmaxAndTopKVsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const LogitsCase& c : argmax_cases()) check_argmax(c, r);
    for (const LogitsCase& c : topk_cases()) check_topk(c, r);
}

TEST(HipHeadEmu, RejectsBadTopKAndReadsWithoutResult) {
    std::vector<float> f(1u << 14, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * 4);
    TopKArgs a;
    a.logits = BufferView(b, 0, 4000);
    a.n = 1000;
    a.k = 1025;
    a.ids = BufferView(b, 8192, 4100);
    a.values = BufferView(b, 16384, 4100);
    a.status = BufferView(b, 32768, 4);
    expect_kernel_error([&] { Ops().top_k(Target::emulation(), a); }, "k = 1025 for 1000 entries");
    a.k = 0;
    expect_kernel_error([&] { Ops().top_k(Target::emulation(), a); }, "k = 0");
    a.k = 10;
    a.values = BufferView(b, 8192 + 20, 40);  // overlaps ids
    expect_kernel_error([&] { Ops().top_k(Target::emulation(), a); }, "ids overlaps values");
}

// ---------------------------------------------------------------------------------------
// RMS_NORM, PARTIAL_ROPE, SWIGLU, MUL_SIGMOID
// ---------------------------------------------------------------------------------------

struct RmsCase {
    std::uint32_t rows, cols;
    bool alias;
    std::uint32_t pad;
    std::string variant;
    std::uint32_t seed;
};

void check_rms(const RmsCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "rms rows " << c.rows << " cols " << c.cols << " alias " << c.alias);
    std::mt19937 rng(c.seed);
    const std::size_t s = c.cols + c.pad;
    std::vector<float> x = uniform(rng, c.rows * s, -3.0f, 3.0f), w = uniform(rng, c.cols, 0.5f, 1.5f);
    std::vector<float> out(c.rows * s, 999.0f);
    auto rx = x, rout = out;
    {
        cpu::Rows xr(rx.data(), c.rows, c.cols, s);
        cpu::rms_norm(xr, w, 1e-6f, c.alias ? xr : cpu::Rows(rout.data(), c.rows, c.cols, s));
    }
    const Buffer bx = r.make(x), bw = r.make(w), bo = r.make(out);
    RmsNormArgs a;
    a.x = BufferView(bx, 0, 0, s * 4);
    a.w = bw;
    a.out = c.alias ? a.x : BufferView(bo, 0, 0, s * 4);
    a.rows = c.rows;
    a.cols = c.cols;
    a.eps = 1e-6f;
    OpsOptions o;
    if (!c.variant.empty()) o.rms_norm = c.variant;
    Ops(o).rms_norm(r.target(), a);
    r.finish();
    r.fetch(bx, x);
    r.fetch(bo, out);
    EXPECT_TRUE(matches(rx, x, r.exact(), "x"));
    EXPECT_TRUE(matches(rout, out, r.exact(), "out"));
}

struct RopeCase {
    std::uint32_t T, heads, head_dim, rot;
    float theta;
    std::uint32_t pad;
    std::uint32_t seed;
};

void check_rope(const RopeCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "rope T " << c.T << " heads " << c.heads << " head_dim " << c.head_dim << " rot "
                                    << c.rot);
    std::mt19937 rng(c.seed);
    const std::size_t cols = static_cast<std::size_t>(c.heads) * c.head_dim;
    const std::size_t s = cols + c.pad;
    std::vector<float> x = uniform(rng, c.T * s, -2.0f, 2.0f);
    std::vector<std::int32_t> pos(c.T);
    const std::int32_t special[] = {0, 1, 7, 4095, 131071, 262143};
    for (std::uint32_t t = 0; t < c.T; ++t) pos[t] = special[t % 6] + static_cast<std::int32_t>(t);
    auto rx = x;
    cpu::partial_rope_neox(cpu::Rows(rx.data(), c.T, cols, s), c.heads, c.head_dim, pos, c.rot, c.theta);
    std::vector<float> posf(c.T);
    std::memcpy(posf.data(), pos.data(), c.T * 4);
    const Buffer bx = r.make(x), bp = r.make(posf);
    RopeArgs a;
    a.x = BufferView(bx, 0, 0, s * 4);
    a.positions = bp;
    a.n_tokens = c.T;
    a.n_heads = c.heads;
    a.head_dim = c.head_dim;
    a.rot_dims = c.rot;
    a.theta = c.theta;
    Ops().partial_rope_neox(r.target(), a);
    r.finish();
    r.fetch(bx, x);
    EXPECT_TRUE(matches(rx, x, r.exact(), "x (in place)"));
}

void check_eltwise(bool swiglu, std::uint32_t rows, std::uint32_t cols, int alias, std::uint32_t seed, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << (swiglu ? "swiglu" : "mul_sigmoid") << " rows " << rows << " cols " << cols
                                    << " alias " << alias);
    std::mt19937 rng(seed);
    std::vector<float> a = uniform(rng, rows * cols, -8.0f, 8.0f), b = uniform(rng, rows * cols, -8.0f, 8.0f);
    a[0] = 90.0f;  // exp(-x) underflow side
    b[1] = -90.0f;
    std::vector<float> out(rows * cols, 999.0f);
    auto ra = a, rb = b, ro = out;
    {
        cpu::Rows xa(ra.data(), rows, cols, cols), xb(rb.data(), rows, cols, cols), xo(ro.data(), rows, cols, cols);
        const cpu::Rows o = alias == 1 ? xa : alias == 2 ? xb : xo;
        if (swiglu) {
            cpu::swiglu(xa, xb, o);
        } else {
            cpu::mul_sigmoid(xa, xb, o);
        }
    }
    const Buffer ba = r.make(a), bb = r.make(b), bo = r.make(out);
    EltwiseArgs e;
    e.a = ba;
    e.b = bb;
    e.out = alias == 1 ? BufferView(ba) : alias == 2 ? BufferView(bb) : BufferView(bo);
    e.rows = rows;
    e.cols = cols;
    const Ops ops;
    if (swiglu) {
        ops.swiglu(r.target(), e);
    } else {
        ops.mul_sigmoid(r.target(), e);
    }
    r.finish();
    r.fetch(ba, a);
    r.fetch(bb, b);
    r.fetch(bo, out);
    EXPECT_TRUE(matches(ra, a, r.exact(), "a"));
    EXPECT_TRUE(matches(rb, b, r.exact(), "b"));
    EXPECT_TRUE(matches(ro, out, r.exact(), "out"));
}

const std::vector<RmsCase> kRms{
    {48, 256, true, 0, "", 501},             // per-head q/k norm (D-004), in place
    {3, 5120, false, 9, "", 502},            // hidden-size norm, strided
    {5, 37, false, 0, "rms_norm_b32", 503},  // 8-lane tail
};
const std::vector<RopeCase> kRope{
    {6, 24, 256, 64, 1e7f, 0, 601},  // qwen35 q heads: 64 of 256 dims, theta 1e7 (D-004)
    {6, 4, 256, 64, 1e7f, 5, 602},   // kv heads, strided
    {3, 2, 128, 128, 10000.0f, 0, 603},
};

TEST(HipMiscEmu, RmsRopeEltwiseBitExactVsCpu) {
    for (auto& r : emulation_runners()) {
        for (const RmsCase& c : kRms) check_rms(c, *r);
        for (const RopeCase& c : kRope) check_rope(c, *r);
        check_eltwise(true, 3, 17408, 0, 701, *r);  // FFN width (D-004)
        check_eltwise(true, 2, 300, 1, 702, *r);
        check_eltwise(false, 2, 6144, 0, 703, *r);  // 24 heads x 256 attention output
        check_eltwise(false, 3, 77, 2, 704, *r);
    }
}

TEST(HipMiscDevice, RmsRopeEltwiseVsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const RmsCase& c : kRms) check_rms(c, r);
    for (const RopeCase& c : kRope) check_rope(c, r);
    check_eltwise(true, 3, 17408, 0, 701, r);
    check_eltwise(false, 2, 6144, 0, 703, r);
}

TEST(HipMiscEmu, RopeRejectsUnsupportedRotDims) {
    std::vector<float> f(4096, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * 4);
    RopeArgs a;
    a.x = BufferView(b, 0, 2048 * 4);
    a.positions = BufferView(b, 2048 * 4, 4);
    a.n_tokens = 1;
    a.n_heads = 1;
    a.head_dim = 256;
    a.rot_dims = 64;
    ASSERT_NO_THROW(Ops().partial_rope_neox(Target::emulation(), a));
    a.rot_dims = 130;
    expect_kernel_error([&] { Ops().partial_rope_neox(Target::emulation(), a); }, "rot_dims 130 exceeds");
    a.rot_dims = 63;
    expect_kernel_error([&] { Ops().partial_rope_neox(Target::emulation(), a); }, "must be even");
}

// ---------------------------------------------------------------------------------------
// PARTIAL_ROPE with a head stride (TD-9): in place on the interleaved [Q | gate] layout
// ---------------------------------------------------------------------------------------

void check_rope_interleaved(Runner& r) {
    SCOPED_TRACE(r.name());
    constexpr std::uint32_t kT = 5, kHeads = 24, kHd = 256, kRot = 64;
    constexpr std::size_t kRow = static_cast<std::size_t>(kHeads) * 2 * kHd;  // [Q_h | gate_h] per head
    std::mt19937 rng(901);
    std::vector<float> x = uniform(rng, kT * kRow, -2.0f, 2.0f);
    const std::vector<float> orig = x;
    std::vector<std::int32_t> pos{0, 3, 17, 4095, 131071};
    // CPU reference: de-interleave Q, rotate densely, re-interleave.
    std::vector<float> qd(kT * static_cast<std::size_t>(kHeads) * kHd);
    for (std::size_t t = 0; t < kT; ++t) {
        for (std::size_t h = 0; h < kHeads; ++h) std::copy_n(&x[t * kRow + h * 2 * kHd], kHd, &qd[(t * kHeads + h) * kHd]);
    }
    cpu::partial_rope_neox(cpu::Rows(qd.data(), kT, static_cast<std::size_t>(kHeads) * kHd,
                                     static_cast<std::size_t>(kHeads) * kHd),
                           kHeads, kHd, pos, kRot, 1e7f);
    std::vector<float> ref = orig;
    for (std::size_t t = 0; t < kT; ++t) {
        for (std::size_t h = 0; h < kHeads; ++h) std::copy_n(&qd[(t * kHeads + h) * kHd], kHd, &ref[t * kRow + h * 2 * kHd]);
    }
    std::vector<float> posf(kT);
    std::memcpy(posf.data(), pos.data(), kT * 4);
    const Buffer bx = r.make(x), bp = r.make(posf);
    RopeArgs a;
    a.x = BufferView(bx, 0, 0, kRow * 4);
    a.positions = bp;
    a.n_tokens = kT;
    a.n_heads = kHeads;
    a.head_dim = kHd;
    a.head_stride = 2 * kHd;
    a.rot_dims = kRot;
    a.theta = 1e7f;
    Ops().partial_rope_neox(r.target(), a);
    r.finish();
    r.fetch(bx, x);
    EXPECT_TRUE(matches(ref, x, r.exact(), "interleaved [Q | gate] (Q rotated, gate untouched)"));
}

TEST(HipRopeEmu, HeadStrideInterleavedQGate) {
    for (auto& r : emulation_runners()) check_rope_interleaved(*r);
    std::vector<float> f(4096, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * 4);
    RopeArgs a;
    a.x = BufferView(b, 0, 2048 * 4);
    a.positions = BufferView(b, 2048 * 4, 4);
    a.n_tokens = 1;
    a.n_heads = 2;
    a.head_dim = 256;
    a.rot_dims = 64;
    a.head_stride = 255;
    try {
        Ops().partial_rope_neox(Target::emulation(), a);
        ADD_FAILURE() << "head_stride < head_dim accepted";
    } catch (const Error& e) {
        EXPECT_NE(std::string(e.what()).find("head_stride 255 < head_dim 256"), std::string::npos) << e.what();
    }
}

TEST(HipRopeDevice, HeadStrideInterleavedQGate) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    check_rope_interleaved(r);
}

}  // namespace
}  // namespace halo::hip::test
