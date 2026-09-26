// WS-BI-1 (ADR-001 §5, §6.5): the CPU backend adapter.
//  - every op is bit-identical to the halo::cpu / halo::tensor composition its contract names,
//    on the same bytes (the adapter adds views, never arithmetic);
//  - head-strided RoPE and attention (the per-head decomposition, ADR §5.2 / A-3) are
//    bit-identical to the dense op on a de-interleaved copy;
//  - host validation: ownership, range, alignment, read-only outputs, the neutral aliasing
//    rule, unknown variant ids, status words, foreign streams, run_op dispatch;
//  - the registry pins (op, id) -> name.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <random>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/thread_pool.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"

using namespace halo::backend;  // NOLINT(google-build-using-namespace)
namespace cpu = halo::cpu;
using halo::DType;
using halo::ErrorCode;

namespace {

std::vector<float> rnd(std::size_t n, std::uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
}

bool bits_equal(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void expect_code(ErrorCode code, const std::function<void()>& fn, const char* what, const char* contains = nullptr) {
    try {
        fn();
        ADD_FAILURE() << what << ": no error thrown";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), code) << what << ": " << e.what();
        if (contains != nullptr) {
            EXPECT_NE(std::string(e.what()).find(contains), std::string::npos) << what << ": " << e.what();
        }
    }
}

class CpuBackendTest : public ::testing::Test {
protected:
    cpu::ThreadPool pool{3};
    std::unique_ptr<Backend> be = make_cpu_backend(&pool);
    std::unique_ptr<Stream> st = be->create_stream();
    std::vector<std::unique_ptr<Buffer>> keep;

    /// Imports `v` (zero-copy) and returns a dense ref over it.
    TensorRef imp(std::vector<float>& v) {
        keep.push_back(be->import_host(std::as_writable_bytes(std::span(v))));
        return TensorRef::of(*keep.back());
    }
    TensorRef imp(std::vector<std::int32_t>& v) {
        keep.push_back(be->import_host(std::as_writable_bytes(std::span(v))));
        return TensorRef::of(*keep.back());
    }
    TensorRef imp(std::vector<std::uint32_t>& v) {
        keep.push_back(be->import_host(std::as_writable_bytes(std::span(v))));
        return TensorRef::of(*keep.back());
    }
    TensorRef imp_bytes(std::vector<std::byte>& v) {
        keep.push_back(be->import_host(std::span(v)));
        return TensorRef::of(*keep.back());
    }
    /// ref at element offset `off`, element stride `stride` (0 = dense).
    static TensorRef at(TensorRef r, std::uint64_t off, std::uint64_t stride = 0) {
        r.offset += off * 4;
        r.row_stride = stride * 4;
        return r;
    }
};

}  // namespace

// ---------------------------------------------------------------------------------------
// Registry, limits, buffers
// ---------------------------------------------------------------------------------------

TEST_F(CpuBackendTest, RegistryPinsOneReferenceVariantPerOp) {
    EXPECT_EQ(be->kind(), Kind::Cpu);
    EXPECT_EQ(to_string(Kind::HipEmulation), "hip-emulation");
    for (std::size_t i = 0; i < kOpCount; ++i) {
        const auto op = static_cast<OpId>(i);
        const auto vs = be->variants(op);
        ASSERT_EQ(vs.size(), 1u) << op_name(op);
        EXPECT_EQ(vs[0].id, 0u) << op_name(op);
        EXPECT_EQ(vs[0].name, "reference") << op_name(op);
        EXPECT_EQ(vs[0].op, op);
        EXPECT_EQ(be->variants(op, "chunked").size(), 1u);
    }
    // Stable op ids (append only).
    EXPECT_EQ(static_cast<int>(OpId::GatedDeltaRule), 6);
    EXPECT_EQ(static_cast<int>(OpId::Copy), 17);
    EXPECT_EQ(op_name(OpId::GatedDeltaRule), "GATED_DELTANET");
    const Limits l = be->limits();
    EXPECT_TRUE(l.gdn_chunked);
    EXPECT_EQ(l.max_gdn_chunk, 1024u);  // cpu::gated_delta_rule_chunked's range
}

TEST_F(CpuBackendTest, BuffersAllocateZeroedAndImportZeroCopy) {
    auto b = be->allocate(64, Tier::Vram);
    EXPECT_EQ(b->bytes(), 64u);
    EXPECT_EQ(b->tier(), Tier::Host);
    EXPECT_EQ(b->backend(), be.get());
    EXPECT_TRUE(b->writable());
    ASSERT_NE(b->host_ptr(), nullptr);
    EXPECT_TRUE(std::all_of(b->host_data(), b->host_data() + 64, [](std::byte x) { return x == std::byte{0}; }));
    std::vector<std::byte> host(16);
    auto i = be->import_host(std::span(host));
    EXPECT_EQ(i->host_data(), host.data());
    const std::vector<std::byte> ro(16);
    auto r = be->import_host_readonly(std::span(ro));
    EXPECT_FALSE(r->writable());
    EXPECT_EQ(r->host_ptr(), nullptr);
    EXPECT_EQ(r->host_data(), ro.data());
    std::vector<std::byte> out(16);
    be->download(*st, TensorRef::of(*r), std::span(out));
    expect_code(ErrorCode::Kernel, [&] { be->upload(*st, TensorRef::of(*r), std::span<const std::byte>(ro)); }, "upload to read-only");
    const std::vector<std::byte> src(8, std::byte{7});
    be->upload(*st, TensorRef{b.get(), 8, 8, 0}, std::span<const std::byte>(src));
    EXPECT_EQ(b->host_data()[8], std::byte{7});
    EXPECT_EQ(b->host_data()[7], std::byte{0});
    expect_code(ErrorCode::Kernel, [&] { be->upload(*st, TensorRef{b.get(), 60, 0, 0}, std::span<const std::byte>(src)); },
                "upload past the end");
}

// ---------------------------------------------------------------------------------------
// Host validation (each check shown red by its own expectation)
// ---------------------------------------------------------------------------------------

TEST_F(CpuBackendTest, ValidationRejectsBadOperandsBeforeWriting) {
    auto x = rnd(65, 1);  // one spare float: a 2-byte offset stays in range (alignment alone fails)
    auto w = rnd(16, 2);
    std::vector<float> out(64, 5.0f);
    const TensorRef rx = imp(x), rw = imp(w), ro = imp(out);
    RmsNormArgs a{rx, rw, ro, 4, 16, 1e-6f, {}};
    be->rms_norm(*st, a);  // valid

    std::fill(out.begin(), out.end(), 5.0f);
    RmsNormArgs mis = a;
    mis.x.offset = 2;  // not 4-byte aligned
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, mis); }, "misaligned offset");
    RmsNormArgs range = a;
    range.rows = 5;  // 5 x 16 floats > 64
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, range); }, "out of range");
    RmsNormArgs limit = a;
    limit.x.bytes = 60;  // the view is shorter than 4 x 16 floats
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, limit); }, "view byte limit");
    RmsNormArgs stride = a;
    stride.x.row_stride = 8;  // < 16 floats
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, stride); }, "stride < cols", "row stride");
    RmsNormArgs inplace = a;
    inplace.out = inplace.x;  // the neutral rule: RMS_NORM may not run in place (Vulkan)
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, inplace); }, "in-place rms_norm");
    RmsNormArgs variant = a;
    variant.kernel.variant_id = 7;
    expect_code(ErrorCode::Config, [&] { be->rms_norm(*st, variant); }, "unknown variant");
    RmsNormArgs empty = a;
    empty.w = {};
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, empty); }, "empty operand");

    auto other = make_cpu_backend(nullptr);
    auto foreign = other->allocate(64 * 4, Tier::Host);
    RmsNormArgs f = a;
    f.x = TensorRef::of(*foreign);
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, f); }, "foreign buffer");
    auto ost = other->create_stream();
    expect_code(ErrorCode::Api, [&] { be->rms_norm(*ost, a); }, "foreign stream");

    const std::vector<float> cx(64, 1.0f);
    auto rob = be->import_host_readonly(std::as_bytes(std::span(cx)));
    RmsNormArgs readonly = a;
    readonly.out = TensorRef::of(*rob);
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, readonly); }, "read-only output");
    EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return v == 5.0f; })) << "a rejected op wrote its output";

    // ADD: out identical to a is the residual accumulate; any other overlap is rejected.
    EltwiseArgs add{rx, ro, rx, 4, 16, {}};
    be->add(*st, add);
    EltwiseArgs add_b = add;
    add_b.out = add_b.b;
    expect_code(ErrorCode::Kernel, [&] { be->add(*st, add_b); }, "add out == b");
    EltwiseArgs add_part = add;
    add_part.out = at(rx, 1);
    add_part.rows = 3;
    expect_code(ErrorCode::Kernel, [&] { be->add(*st, add_part); }, "add partial overlap");
    EltwiseArgs sw = add;
    expect_code(ErrorCode::Kernel, [&] { be->swiglu(*st, sw); }, "swiglu in place");

    GdnArgs g{};
    g.status.word = 0;
    expect_code(ErrorCode::Unsupported, [&] { be->gated_delta_rule(*st, g); }, "status word");

    OpInvocation inv{OpId::Gemv, a};  // RmsNormArgs under the GEMV id
    expect_code(ErrorCode::Api, [&] { be->run_op(*st, inv); }, "run_op mismatch");
}

// ---------------------------------------------------------------------------------------
// Ops == the halo::cpu composition, bitwise
// ---------------------------------------------------------------------------------------

TEST_F(CpuBackendTest, NormsAndEltwiseEqualCpuOps) {
    const std::size_t R = 6, D = 32;
    auto x = rnd(R * 2 * D, 3);  // rows interleaved with junk: stride 2D
    auto w = rnd(D, 4, 0.5f, 1.5f);
    auto b = rnd(R * D, 5);
    std::vector<float> out(R * D), ref(R * D);
    // RMS_NORM over a strided x (per-head norm reading every other head block).
    be->rms_norm(*st, {at(imp(x), 0, 2 * D), imp(w), imp(out), R, D, 1e-6f, {}});
    cpu::rms_norm(cpu::ConstRows(x.data(), R, D, 2 * D), w, 1e-6f, cpu::Rows(std::span(ref), R, D));
    EXPECT_TRUE(bits_equal(out, ref));

    // ADD_RMS_NORM == cpu::add then cpu::rms_norm, with h identical to a.
    auto a1 = rnd(R * D, 6), a2 = a1;
    std::vector<float> o1(R * D), o2(R * D);
    const TensorRef ra = imp(a1);
    be->add_rms_norm(*st, {ra, imp(b), ra, imp(w), imp(o1), R, D, 1e-6f, {}});
    cpu::add(cpu::ConstRows(std::span<const float>(a2), R, D), cpu::ConstRows(std::span<const float>(b), R, D),
             cpu::Rows(std::span(a2), R, D));
    cpu::rms_norm(cpu::ConstRows(std::span<const float>(a2), R, D), w, 1e-6f, cpu::Rows(std::span(o2), R, D));
    EXPECT_TRUE(bits_equal(a1, a2));
    EXPECT_TRUE(bits_equal(o1, o2));

    // GATED_NORM, SWIGLU, MUL_SIGMOID (gate read with a head stride from an interleaved row).
    auto z = rnd(R * D, 7);
    be->gated_rms_norm(*st, {imp(b), imp(z), imp(w), imp(out), R, D, 1e-6f, {}});
    cpu::gated_rms_norm(cpu::ConstRows(std::span<const float>(b), R, D), w, cpu::ConstRows(std::span<const float>(z), R, D), 1e-6f,
                        cpu::Rows(std::span(ref), R, D));
    EXPECT_TRUE(bits_equal(out, ref));
    be->swiglu(*st, {imp(b), imp(z), imp(out), R, D, {}});
    cpu::swiglu(cpu::ConstRows(std::span<const float>(b), R, D), cpu::ConstRows(std::span<const float>(z), R, D),
                cpu::Rows(std::span(ref), R, D));
    EXPECT_TRUE(bits_equal(out, ref));
    be->mul_sigmoid(*st, {imp(b), at(imp(x), D, 2 * D), imp(out), R, D, {}});
    cpu::mul_sigmoid(cpu::ConstRows(std::span<const float>(b), R, D), cpu::ConstRows(x.data() + D, R, D, 2 * D),
                     cpu::Rows(std::span(ref), R, D));
    EXPECT_TRUE(bits_equal(out, ref));
}

TEST_F(CpuBackendTest, GemvF32AndQuantizedEqualCpuMatmul) {
    const std::size_t N = 40, K = 64, T = 3;
    auto x = rnd(T * K, 8);
    // F32 weights: a dense in-place view.
    auto wf = rnd(N * K, 9);
    std::vector<float> y(T * N), ref(T * N);
    be->gemv(*st, {DType::F32, imp(wf), imp(x), imp(y), N, K, T, {}});
    cpu::matmul(cpu::ConstRows(std::span<const float>(x), T, K), cpu::WeightMatrix::dense(cpu::ConstRows(std::span<const float>(wf), N, K)),
                cpu::Rows(std::span(ref), T, N));
    EXPECT_TRUE(bits_equal(y, ref));
    // Q8_0 weights (2-byte f16 scale + 32 int8 per block): dequantized with tensor::dequantize_row.
    const std::size_t rb = halo::tensor::row_bytes(DType::Q8_0, K);
    std::vector<std::byte> wq(N * rb);
    std::mt19937 g(10);
    for (std::size_t i = 0; i < wq.size(); ++i) wq[i] = static_cast<std::byte>(g() & 0xFFu);
    for (std::size_t blk = 0; blk < wq.size(); blk += 34) {
        wq[blk] = std::byte{0x00};
        wq[blk + 1] = std::byte{0x2C};  // f16 1/16
    }
    be->gemv(*st, {DType::Q8_0, imp_bytes(wq), imp(x), imp(y), N, K, T, {}});
    const std::byte* base = wq.data();
    const auto wm = cpu::WeightMatrix::dequantized(N, K, [base, rb](std::size_t first, std::size_t n, std::span<float> out) {
        halo::tensor::dequantize_row(DType::Q8_0, base + first * rb, out.data(), static_cast<std::int64_t>(n * 64));
    });
    cpu::matmul(cpu::ConstRows(std::span<const float>(x), T, K), wm, cpu::Rows(std::span(ref), T, N));
    EXPECT_TRUE(bits_equal(y, ref));
    expect_code(ErrorCode::Kernel, [&] { be->gemv(*st, {DType::Q8_0, imp_bytes(wq), imp(x), imp(x), N, K, T, {}}); }, "y overlaps x");
}

TEST_F(CpuBackendTest, GdnGatesEqualTheQwen35Sequence) {
    const std::size_t T = 5, nv = 6;
    auto alpha = rnd(T * nv, 11, -4.0f, 4.0f), beta = rnd(T * nv, 12, -3.0f, 3.0f);
    auto dt = rnd(nv, 13), a = rnd(nv, 14, -2.0f, -0.1f);
    std::vector<float> g(T * nv), bo(T * nv);
    be->gdn_gates(*st, {imp(alpha), imp(beta), imp(dt), imp(a), imp(g), imp(bo), T, nv, {}});
    // Reference: the pre-backend qwen35.cpp sequence, in place.
    auto ra = alpha, rbt = beta;
    cpu::sigmoid(cpu::ConstRows(std::span<const float>(rbt), T, nv), cpu::Rows(std::span(rbt), T, nv));
    for (std::size_t r = 0; r < T; ++r)
        for (std::size_t j = 0; j < nv; ++j) ra[r * nv + j] += dt[j];
    cpu::softplus(cpu::ConstRows(std::span<const float>(ra), T, nv), cpu::Rows(std::span(ra), T, nv));
    for (std::size_t r = 0; r < T; ++r)
        for (std::size_t j = 0; j < nv; ++j) ra[r * nv + j] = a[j] * ra[r * nv + j];
    EXPECT_TRUE(bits_equal(g, ra));
    EXPECT_TRUE(bits_equal(bo, rbt));
}

TEST_F(CpuBackendTest, ConvAndGatedDeltaRuleWithSlotsEqualCpuOps) {
    const std::size_t T = 9, C = 2 * 16 + 32, K = 4;
    const cpu::GdnDims dims{2, 4, 8, 8, cpu::GdnHeadMapping::Tiled};
    auto x = rnd(T * C, 15), w = rnd(C * K, 16);
    auto st1 = rnd((K - 1) * C, 17), st2 = st1;
    std::vector<float> out1(T * C), out2(T * C), sl1(3 * (K - 1) * C), sl2(sl1.size());
    be->conv1d_silu(*st, {imp(x), imp(w), imp(st1), imp(out1), imp(sl1), T, C, K, 3, {}});
    cpu::causal_conv1d_silu(cpu::ConstRows(std::span<const float>(x), T, C), cpu::ConstRows(std::span<const float>(w), C, K),
                            cpu::Rows(std::span(st2), K - 1, C), cpu::Rows(std::span(out2), T, C), nullptr, sl2);
    EXPECT_TRUE(bits_equal(out1, out2));
    EXPECT_TRUE(bits_equal(st1, st2));
    EXPECT_TRUE(bits_equal(sl1, sl2));
    expect_code(ErrorCode::Kernel, [&] { be->conv1d_silu(*st, {imp(x), imp(w), imp(st1), imp(x), {}, T, C, K, 0, {}}); },
                "conv in place (neutral rule: distinct output)");

    // GDN over q/k/v views into the fused conv row, both forms, with slots.
    const std::size_t kd = 16, vd = 32, nv = 4, sf = nv * 8 * 8;
    auto g = rnd(T * nv, 18, -2.0f, -0.01f), beta = rnd(T * nv, 19, 0.0f, 1.0f);
    for (const GdnForm form : {GdnForm::Recurrent, GdnForm::Chunked}) {
        auto s1 = rnd(sf, 20), s2 = s1;
        std::vector<float> o1(T * vd), o2(T * vd), q1(4 * sf), q2(4 * sf);
        const TensorRef rc = imp(out1);
        GdnArgs a{};
        a.form = form;
        a.q = at(rc, 0, C);
        a.k = at(rc, kd, C);
        a.v = at(rc, 2 * kd, C);
        a.g = imp(g);
        a.beta = imp(beta);
        a.state = imp(s1);
        a.state_slots = imp(q1);
        a.out = imp(o1);
        a.n_k = 2;
        a.n_v = 4;
        a.d_k = 8;
        a.d_v = 8;
        a.n_tokens = T;
        a.n_slots = 4;
        a.qk_l2norm = true;
        a.q_scale = 1.0f / std::sqrt(8.0f);
        a.chunk_size = 4;
        be->gated_delta_rule(*st, a);
        const cpu::GdnInputs in{cpu::ConstRows(out1.data(), T, kd, C), cpu::ConstRows(out1.data() + kd, T, kd, C),
                                cpu::ConstRows(out1.data() + 2 * kd, T, vd, C), cpu::ConstRows(std::span<const float>(g), T, nv),
                                cpu::ConstRows(std::span<const float>(beta), T, nv)};
        const cpu::GdnQkParams qk{.qk_l2norm = true, .q_scale = 1.0f / std::sqrt(8.0f)};
        if (form == GdnForm::Chunked) {
            cpu::gated_delta_rule_chunked(dims, in, s2, cpu::Rows(std::span(o2), T, vd), qk, 4, nullptr, q2);
        } else {
            cpu::gated_delta_rule_recurrent(dims, in, s2, cpu::Rows(std::span(o2), T, vd), qk, nullptr, q2);
        }
        EXPECT_TRUE(bits_equal(o1, o2));
        EXPECT_TRUE(bits_equal(s1, s2));
        EXPECT_TRUE(bits_equal(q1, q2));
    }
}

TEST_F(CpuBackendTest, HeadStridedRopeEqualsDenseRopeOnDeinterleavedCopy) {
    // qwen35 attn_q rows: per head [Q (hd) | gate (hd)]; RoPE on Q in place, head_stride 2 hd.
    const std::size_t T = 7, nh = 4, hd = 16, rot = 8;
    auto qg = rnd(T * 2 * nh * hd, 21);
    std::vector<std::int32_t> pos(T);
    for (std::size_t i = 0; i < T; ++i) pos[i] = static_cast<std::int32_t>(3 + 5 * i);
    std::vector<float> dense(T * nh * hd);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t h = 0; h < nh; ++h)
            std::copy_n(&qg[t * 2 * nh * hd + h * 2 * hd], hd, &dense[t * nh * hd + h * hd]);
    const auto before = qg;
    // The row stride is explicit: the dense default would be the extent of the last head.
    RopeArgs a{at(imp(qg), 0, 2 * nh * hd), imp(pos), T, nh, hd, rot, 1e6f, 2 * hd, {}};
    be->partial_rope(*st, a);
    cpu::partial_rope_neox(cpu::Rows(std::span(dense), T, nh * hd), nh, hd, pos, rot, 1e6f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < nh; ++h) {
            EXPECT_TRUE(bits_equal(std::span(&qg[t * 2 * nh * hd + h * 2 * hd], hd), std::span(&dense[t * nh * hd + h * hd], hd)))
                << "t " << t << " h " << h;
            EXPECT_TRUE(bits_equal(std::span(&qg[t * 2 * nh * hd + h * 2 * hd + hd], hd),
                                   std::span(&before[t * 2 * nh * hd + h * 2 * hd + hd], hd)))
                << "gate half touched at t " << t << " h " << h;
        }
    }
    // Dense (head_stride 0) is the plain op.
    auto d2 = rnd(T * nh * hd, 22), d3 = d2;
    be->partial_rope(*st, {imp(d2), imp(pos), T, nh, hd, rot, 1e6f, 0, {}});
    cpu::partial_rope_neox(cpu::Rows(std::span(d3), T, nh * hd), nh, hd, pos, rot, 1e6f);
    EXPECT_TRUE(bits_equal(d2, d3));
}

namespace {

/// A KvPool storage image: n_blocks x block[layer][K|V][token][kv_dim].
struct PoolImage {
    std::size_t n_layers, bt, kvd, n_blocks;
    std::vector<float> data;
    PoolImage(std::size_t l, std::size_t b, std::size_t k, std::size_t n)
        : n_layers(l), bt(b), kvd(k), n_blocks(n), data(n * l * 2 * b * k, 0.0f) {}
    float* k_rows(std::size_t id, std::size_t layer) { return &data[id * n_layers * 2 * bt * kvd + layer * 2 * bt * kvd]; }
    float* v_rows(std::size_t id, std::size_t layer) { return k_rows(id, layer) + bt * kvd; }
};

}  // namespace

TEST_F(CpuBackendTest, KvWriteAndAttentionOverThePoolImageEqualCpuOps) {
    const std::size_t nh = 4, nkv = 2, hd = 16, kvd = nkv * hd, bt = 4, L = 2, layer = 1;
    PoolImage pool(L, bt, kvd, 6);
    std::vector<std::uint32_t> table{5, 2, 0, 3};  // 16 rows
    const std::size_t start = 6, T = 5;            // rows 6..10 cross a block boundary
    auto k = rnd(T * kvd, 23), v = rnd(T * kvd, 24);
    const TensorRef rp = imp(pool.data), rt = imp(table);
    // History rows 0..5 written by hand, rows 6..10 by KV_WRITE.
    auto hk = rnd(start * kvd, 25), hv = rnd(start * kvd, 26);
    for (std::size_t r = 0; r < start; ++r) {
        std::copy_n(&hk[r * kvd], kvd, pool.k_rows(table[r / bt], layer) + (r % bt) * kvd);
        std::copy_n(&hv[r * kvd], kvd, pool.v_rows(table[r / bt], layer) + (r % bt) * kvd);
    }
    KvWriteArgs w{rp, rt, imp(k), imp(v), 6, L, layer, bt, kvd, 4, start, T, {}, {}};
    be->kv_write(*st, w);
    for (std::size_t i = 0; i < T; ++i) {
        const std::size_t r = start + i;
        EXPECT_TRUE(bits_equal(std::span(pool.k_rows(table[r / bt], layer) + (r % bt) * kvd, kvd), std::span(&k[i * kvd], kvd)));
        EXPECT_TRUE(bits_equal(std::span(pool.v_rows(table[r / bt], layer) + (r % bt) * kvd, kvd), std::span(&v[i * kvd], kvd)));
    }
    EXPECT_TRUE(std::all_of(pool.k_rows(table[0], 0), pool.k_rows(table[0], 0) + bt * kvd, [](float f) { return f == 0.0f; }))
        << "layer 0 touched";
    KvWriteArgs bad = w;
    std::vector<std::uint32_t> bad_table{5, 2, 9, 3};
    bad.block_table = imp(bad_table);
    expect_code(ErrorCode::Kernel, [&] { be->kv_write(*st, bad); }, "block id outside the pool");
    KvWriteArgs past = w;
    past.start = 14;
    expect_code(ErrorCode::Kernel, [&] { be->kv_write(*st, past); }, "rows past the table");

    // ATTENTION: dense q, and q read with a head stride out of an interleaved [Q | gate] row.
    const std::size_t rows = start + T;
    std::vector<const float*> kt, vt;
    for (const std::uint32_t id : table) {
        kt.push_back(pool.k_rows(id, layer));
        vt.push_back(pool.v_rows(id, layer));
    }
    const auto keys = cpu::PagedRows::paged(kt, bt, rows, kvd, kvd);
    const auto vals = cpu::PagedRows::paged(vt, bt, rows, kvd, kvd);
    auto qg = rnd(T * 2 * nh * hd, 27);
    std::vector<float> q(T * nh * hd);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t h = 0; h < nh; ++h) std::copy_n(&qg[t * 2 * nh * hd + h * 2 * hd], hd, &q[t * nh * hd + h * hd]);
    std::vector<float> ref(T * nh * hd), o1(T * nh * hd), o2(T * nh * hd);
    const float scale = 0.25f;
    cpu::attention_gqa({nh, nkv, hd}, cpu::ConstRows(std::span<const float>(q), T, nh * hd), keys, vals, start, scale,
                       cpu::Rows(std::span(ref), T, nh * hd));
    AttentionArgs a{};
    a.q = imp(q);
    a.kv_pool = rp;
    a.block_table = rt;
    a.out = imp(o1);
    a.n_pool_blocks = 6;
    a.n_layers = L;
    a.layer = layer;
    a.block_tokens = bt;
    a.n_block_table = 4;
    a.n_head = nh;
    a.n_kv_head = nkv;
    a.head_dim = hd;
    a.n_tokens = T;
    a.q_offset = start;
    a.scale = scale;
    be->attention(*st, a);
    EXPECT_TRUE(bits_equal(o1, ref));
    AttentionArgs s = a;
    s.q = at(imp(qg), 0, 2 * nh * hd);
    s.q_head_stride = 2 * hd;
    s.out = imp(o2);
    be->attention(*st, s);
    EXPECT_TRUE(bits_equal(o2, ref)) << "head-strided attention differs from dense";
}

TEST_F(CpuBackendTest, LmHeadAcrossSlabsEqualsMatmulArgmaxAndNaNPoisonsOnlyItsRow) {
    const std::size_t V = 8192 + 900, E = 16, n = 3;
    auto w = rnd(V * E, 28), x = rnd(n * E, 29);
    std::vector<float> logits(n * V), ref(n * V);
    std::vector<std::byte> res(n * kArgmaxResultBytes);
    LmHeadArgs a{};
    a.gemv = {DType::F32, imp(w), imp(x), imp(logits), V, E, n, {}};
    a.result = imp_bytes(res);
    be->lm_head(*st, a);
    cpu::matmul(cpu::ConstRows(std::span<const float>(x), n, E), cpu::WeightMatrix::dense(cpu::ConstRows(std::span<const float>(w), V, E)),
                cpu::Rows(std::span(ref), n, V));
    EXPECT_TRUE(bits_equal(logits, ref));
    const auto best = decode_argmax(res);
    for (std::size_t i = 0; i < n; ++i) {
        const auto e = cpu::argmax(std::span<const float>(&ref[i * V], V));
        EXPECT_EQ(best[i].index, e.index);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(best[i].value), std::bit_cast<std::uint32_t>(e.value));
    }
    // Argmax only (no logits operand) gives the same words.
    std::vector<std::byte> res2(n * kArgmaxResultBytes);
    LmHeadArgs b = a;
    b.gemv.y = {};
    b.result = imp_bytes(res2);
    be->lm_head(*st, b);
    EXPECT_EQ(res, res2);
    // ARGMAX / TOP_K over the logits.
    std::vector<std::byte> res3(n * kArgmaxResultBytes);
    be->argmax(*st, {imp(logits), imp_bytes(res3), V, n, {}, {}});
    EXPECT_EQ(res, res3);
    std::vector<std::int32_t> ids(n * 5);
    std::vector<float> vals(n * 5);
    be->top_k(*st, {imp(logits), imp(ids), imp(vals), V, 5, n, {}, {}});
    for (std::size_t i = 0; i < n; ++i) {
        const auto tk = cpu::top_k(std::span<const float>(&ref[i * V], V), 5);
        for (std::size_t j = 0; j < 5; ++j) {
            EXPECT_EQ(ids[i * 5 + j], tk[j].index);
            EXPECT_EQ(vals[i * 5 + j], tk[j].value);
        }
    }
    // NaN (H1): poisons only its own row -- decode_argmax reports {-1, NaN} for that row and
    // leaves the others (a different row of the batch here) untouched, instead of throwing for
    // the whole batch. Poisoning x (not w) affects exactly one batch row's dot products.
    auto x2 = x;
    x2[1 * E] = std::nanf("");  // batch row 1's activation
    std::vector<std::byte> res4(n * kArgmaxResultBytes);
    LmHeadArgs c = a;
    c.gemv.x = imp(x2);
    c.gemv.y = {};
    c.result = imp_bytes(res4);
    be->lm_head(*st, c);
    const auto best4 = decode_argmax(res4);
    ASSERT_EQ(best4.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        if (i == 1) {
            EXPECT_EQ(best4[i].index, -1);
            EXPECT_TRUE(std::isnan(best4[i].value));
        } else {
            EXPECT_EQ(best4[i].index, best[i].index) << "row " << i << " must be unaffected by row 1's NaN";
        }
    }
}

TEST_F(CpuBackendTest, LmHeadValidRowsExcludesPaddedRowsFromArgmaxButNotFromLogits) {
    // M2: rows at/after valid_rows are GGUF LM-head padding; argmax must never pick one, but
    // the full logits row (when requested) must still report every row's raw value.
    const std::size_t V = 20, E = 4, n = 2;
    auto w = rnd(V * E, 41);
    std::vector<float> x(n * E, 0.0f);
    // Row 0's activation is a one-hot selecting w[.., 0], so each weight row's dot product is
    // exactly that row's w[.., 0] value. Make the best real (< valid_rows) row smaller than a
    // padded (>= valid_rows) row, so an unclamped argmax would pick the padded one.
    const std::size_t valid_rows = 15;
    w[10 * E] = 1.0f;    // real row 10: dot = 1
    w[17 * E] = 100.0f;  // padded row 17: dot = 100
    x[0] = 1.0f;
    std::vector<float> logits(n * V);
    std::vector<std::byte> res(n * kArgmaxResultBytes);
    LmHeadArgs a{};
    a.gemv = {DType::F32, imp(w), imp(x), imp(logits), V, E, n, {}};
    a.valid_rows = static_cast<std::uint32_t>(valid_rows);
    a.result = imp_bytes(res);
    be->lm_head(*st, a);
    const auto best = decode_argmax(res);
    ASSERT_EQ(best.size(), n);
    EXPECT_EQ(best[0].index, 10) << "the padded row (17) must never win, even though its raw logit is larger";
    EXPECT_EQ(logits[0 * V + 17], 100.0f) << "the padded row's raw logit is still reported";
    // valid_rows == 0 is "no clamp": unaffected by this change.
    LmHeadArgs b = a;
    b.valid_rows = 0;
    std::vector<std::byte> res2(n * kArgmaxResultBytes);
    b.result = imp_bytes(res2);
    be->lm_head(*st, b);
    EXPECT_EQ(decode_argmax(res2)[0].index, 17) << "with no clamp, the larger (padded) logit wins as before";
}

TEST_F(CpuBackendTest, GetRowsAndCopyEqualDequantizeRow) {
    const std::size_t rows = 10, cols = 32;
    auto t = rnd(rows * cols, 30);
    std::vector<std::int32_t> ids{3, 0, 9, 3};
    std::vector<float> out(ids.size() * cols);
    be->get_rows(*st, {DType::F32, imp(t), imp(ids), imp(out), rows, cols, 4, {}, {}});
    for (std::size_t i = 0; i < ids.size(); ++i) {
        EXPECT_TRUE(bits_equal(std::span(&out[i * cols], cols), std::span(&t[static_cast<std::size_t>(ids[i]) * cols], cols)));
    }
    std::vector<std::int32_t> bad{3, 10};
    expect_code(ErrorCode::Kernel, [&] { be->get_rows(*st, {DType::F32, imp(t), imp(bad), imp(out), rows, cols, 2, {}, {}}); },
                "id outside the table");
    std::vector<float> dst(cols);
    be->copy(*st, {at(imp(t), cols), imp(dst), cols * 4, {}});
    EXPECT_TRUE(bits_equal(dst, std::span(&t[cols], cols)));
    const TensorRef rt = imp(t);
    expect_code(ErrorCode::Kernel, [&] { be->copy(*st, {rt, at(rt, 4), cols * 4, {}}); }, "overlapping copy");
    // run_op dispatches to the typed method.
    std::vector<float> dst2(cols);
    be->run_op(*st, OpInvocation{OpId::Copy, CopyArgs{at(rt, cols), imp(dst2), cols * 4, {}}});
    EXPECT_EQ(dst, dst2);
}

// Shape / layout checks not covered above, one expectation per check.
TEST_F(CpuBackendTest, ValidationRejectsBadShapesAndLayouts) {
    // TensorRef::shifted may not turn a bounded view into an unbounded one.
    auto buf = be->allocate(64, Tier::Host);
    const TensorRef bounded{buf.get(), 0, 16, 0};
    EXPECT_EQ(bounded.shifted(4).bytes, 12u);
    EXPECT_EQ(bounded.shifted(4).offset, 4u);
    EXPECT_EQ(TensorRef::of(*buf).shifted(40).bytes, 0u);  // unbounded stays unbounded
    expect_code(ErrorCode::Kernel, [&] { (void)bounded.shifted(16); }, "shift to the end of a bounded view");

    // decode_argmax: the NaN word poisons only its own vector (H1), not the whole decode; the
    // size check still throws.
    std::vector<std::byte> words(2 * kArgmaxResultBytes);
    const std::uint32_t nan = 1;
    std::memcpy(words.data() + kArgmaxResultBytes + 8, &nan, 4);
    {
        const auto out = decode_argmax(words);
        ASSERT_EQ(out.size(), 2u);
        EXPECT_EQ(out[0].index, 0);  // untouched entry: zero-initialized words decode to id 0
        EXPECT_TRUE(std::isnan(out[1].value));
        EXPECT_EQ(out[1].index, -1);
    }
    EXPECT_EQ(decode_argmax(std::span(words).first(kArgmaxResultBytes)).size(), 1u);
    expect_code(ErrorCode::Kernel, [&] { (void)decode_argmax(std::span(words).first(8)); }, "argmax words size");

    // Offset past the buffer end, and a view longer than the buffer (range checks isolated
    // from the extent check: the operand itself is empty-sized in the first case).
    std::vector<float> x(16), w(4), o(16);
    RmsNormArgs a{imp(x), imp(w), imp(o), 4, 4, 1e-6f, {}};
    be->rms_norm(*st, a);
    RmsNormArgs past = a;
    past.x.offset = 68;
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, past); }, "offset past the buffer end");
    RmsNormArgs longer = a;
    longer.x.bytes = 68;  // 64-byte buffer
    expect_code(ErrorCode::Kernel, [&] { be->rms_norm(*st, longer); }, "view longer than the buffer");

    // PARTIAL_ROPE: head_stride < head_dim, and n_heads == 0.
    std::vector<float> q(4 * 32);
    std::vector<std::int32_t> pos(4);
    RopeArgs r{imp(q), imp(pos), 4, 2, 16, 8, 1e4f, 0, {}};
    be->partial_rope(*st, r);
    RopeArgs hs = r;
    hs.head_stride = 8;
    expect_code(ErrorCode::Kernel, [&] { be->partial_rope(*st, hs); }, "head_stride < head_dim");
    RopeArgs nh0 = r;
    nh0.n_heads = 0;
    expect_code(ErrorCode::Kernel, [&] { be->partial_rope(*st, nh0); }, "n_heads == 0", "n_heads is 0");

    // CONV1D_SHORT: conv_state must be dense.
    const std::size_t C = 8, K = 4, T = 2;
    std::vector<float> cx(T * C), cw(C * K), cst(2 * (K - 1) * C), cout(T * C);
    Conv1dArgs c{imp(cx), imp(cw), imp(cst), imp(cout), {}, T, C, K, 0, {}};
    be->conv1d_silu(*st, c);
    Conv1dArgs cs = c;
    cs.conv_state = at(cs.conv_state, 0, 2 * C);
    expect_code(ErrorCode::Kernel, [&] { be->conv1d_silu(*st, cs); }, "strided conv_state", "must be dense");

    // KV_WRITE / ATTENTION: layer range, dense pool, head grouping, q head stride.
    const std::size_t hd = 8, nkv = 1, kvd = nkv * hd, bt = 2;
    PoolImage pool(1, bt, kvd, 4);
    std::vector<std::uint32_t> table{0, 1};
    std::vector<float> k(kvd), v(kvd);
    const TensorRef rp = imp(pool.data), rt = imp(table);
    KvWriteArgs kw{rp, rt, imp(k), imp(v), 4, 1, 0, bt, kvd, 2, 0, 1, {}, {}};
    be->kv_write(*st, kw);
    KvWriteArgs layer = kw;
    layer.layer = 1;
    expect_code(ErrorCode::Kernel, [&] { be->kv_write(*st, layer); }, "layer >= n_layers");
    KvWriteArgs sparse = kw;
    sparse.n_pool_blocks = 2;
    sparse.kv_pool = at(rp, 0, 2 * 2 * bt * kvd);  // every other block
    expect_code(ErrorCode::Kernel, [&] { be->kv_write(*st, sparse); }, "strided kv_pool", "must be dense");
    std::vector<float> aq(3 * hd), ao(3 * hd);
    AttentionArgs at_{};
    at_.q = imp(aq);
    at_.kv_pool = rp;
    at_.block_table = rt;
    at_.out = imp(ao);
    at_.n_pool_blocks = 4;
    at_.block_tokens = bt;
    at_.n_block_table = 2;
    at_.n_head = 3;
    at_.n_kv_head = 1;
    at_.head_dim = hd;
    at_.n_tokens = 1;
    at_.q_offset = 0;
    be->attention(*st, at_);
    AttentionArgs grp = at_;
    grp.n_kv_head = 2;  // 3 heads are not a multiple of 2
    expect_code(ErrorCode::Kernel, [&] { be->attention(*st, grp); }, "n_head % n_kv_head", "not a multiple");
    AttentionArgs qs = at_;
    qs.q_head_stride = hd - 1;
    expect_code(ErrorCode::Kernel, [&] { be->attention(*st, qs); }, "q_head_stride < head_dim");
}
