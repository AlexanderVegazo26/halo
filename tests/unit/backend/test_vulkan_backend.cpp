// The Vulkan backend adapter (src/backend/vulkan_adapter.*, WS-F2 V5) against the CPU backend
// on the same host inputs, op by op, through the halo::backend interface. Verified on lavapipe
// only (DECISIONS D-001).
//
// What this file checks is the ADAPTER: argument mapping (a swapped operand or a wrong stride is
// a large error), the interface rules (ownership, the neutral aliasing rule, status / kernel
// choices), memory semantics (zero-fill, import write-back, abort, download ordering) and the
// device data errors surfacing at Stream::wait(). Kernel accuracy is accepted by the per-kernel
// differential tests in tests/unit/vulkan (a-priori bounds); here ops that are bit-identical
// there are compared bitwise, and the others with a wiring tolerance of 1e-4 relative to the
// largest CPU magnitude (orders of magnitude above their kernel bounds, orders below any
// mis-wiring).

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "backend/vulkan_adapter.h"
#include "halo/backend/backend.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/vulkan/context.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"
#include "vulkan/vk_test_util.h"

namespace hb = halo::backend;
namespace hv = halo::vulkan;

namespace {

constexpr double k_wiring = 1e-4;

std::vector<float> randn(std::size_t n, std::uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

/// One backend plus a stream and the buffers of one test.
struct Dev {
    hb::Backend& be;
    std::unique_ptr<hb::Stream> s;
    std::vector<std::unique_ptr<hb::Buffer>> bufs;

    explicit Dev(hb::Backend& b) : be(b), s(b.create_stream()) {}

    hb::TensorRef put_bytes(std::span<const std::byte> bytes) {
        bufs.push_back(be.allocate(bytes.size(), hb::Tier::Vram));
        const hb::TensorRef r = hb::TensorRef::of(*bufs.back());
        be.upload(*s, r, bytes);
        return r;
    }
    template <class T>
    hb::TensorRef put(const std::vector<T>& v) {
        return put_bytes(std::as_bytes(std::span(v)));
    }
    hb::TensorRef zeros(std::size_t bytes, hb::Tier tier = hb::Tier::Vram) {
        bufs.push_back(be.allocate(bytes, tier));
        return hb::TensorRef::of(*bufs.back());
    }
    hb::TensorRef import_ro(std::span<const std::byte> bytes) {
        bufs.push_back(be.import_host_readonly(bytes));
        return hb::TensorRef::of(*bufs.back());
    }
    hb::TensorRef import_rw(std::span<std::byte> bytes) {
        bufs.push_back(be.import_host(bytes));
        return hb::TensorRef::of(*bufs.back());
    }
    template <class T>
    std::vector<T> get(hb::TensorRef r, std::size_t n) {
        std::vector<T> out(n);
        be.download(*s, r, std::as_writable_bytes(std::span(out)));
        sync();
        return out;
    }
    void sync() {
        s->submit();
        s->wait();
    }
};

/// The CPU reference backend and the Vulkan backend under test.
struct Pair {
    std::unique_ptr<hb::Backend> cpu, vk;
};

Pair make_pair(const std::shared_ptr<hv::Context>& ctx, const hb::VulkanBackendOptions& o = {}) {
    return {hb::make_cpu_backend(nullptr), hb::make_vulkan_backend(ctx, o)};
}

::testing::AssertionResult close(std::span<const float> vk, std::span<const float> cpu, double rtol, const std::string& what) {
    if (vk.size() != cpu.size()) return ::testing::AssertionFailure() << what << ": size " << vk.size() << " vs " << cpu.size();
    double mag = 0, worst = 0;
    std::size_t at = 0;
    for (float c : cpu) mag = std::max(mag, std::abs(double(c)));
    for (std::size_t i = 0; i < vk.size(); ++i) {
        const double d = std::isfinite(double(vk[i])) ? std::abs(double(vk[i]) - double(cpu[i])) : 1e300;
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    const double tol = rtol * mag + 1e-30;
    std::cout << "[vk-backend] " << what << ": max|vk-cpu| / (rtol * max|cpu|) = " << worst / tol << "\n";
    if (worst <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << what << ": |vk-cpu| " << worst << " at " << at << " (vk " << vk[at] << ", cpu "
                                         << cpu[at] << ") > " << tol;
}

::testing::AssertionResult bitwise(std::span<const float> vk, std::span<const float> cpu, const std::string& what) {
    if (vk.size() == cpu.size() && std::memcmp(vk.data(), cpu.data(), vk.size_bytes()) == 0) {
        return ::testing::AssertionSuccess();
    }
    for (std::size_t i = 0; i < std::min(vk.size(), cpu.size()); ++i) {
        if (std::bit_cast<std::uint32_t>(vk[i]) != std::bit_cast<std::uint32_t>(cpu[i])) {
            return ::testing::AssertionFailure() << what << ": first difference at " << i << " (vk " << vk[i] << ", cpu "
                                                 << cpu[i] << ")";
        }
    }
    return ::testing::AssertionFailure() << what << ": sizes " << vk.size() << " vs " << cpu.size();
}

template <class F>
void expect_code(F&& f, halo::ErrorCode code, const std::string& what) {
    try {
        f();
        ADD_FAILURE() << what << ": accepted";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), code) << what << ": " << e.what();
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------
// Identity, limits, memory
// ---------------------------------------------------------------------------------------

TEST(VulkanBackend, IdentityLimitsVariantsAndZeroFill) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    auto be = hb::make_vulkan_backend(ctx);
    EXPECT_EQ(be->kind(), hb::Kind::Vulkan);
    EXPECT_NE(be->describe().find("vulkan"), std::string::npos);
    const hb::Limits l = be->limits();
    EXPECT_TRUE(l.gdn_chunked);
    EXPECT_EQ(l.max_gdn_chunk, 64u);
    EXPECT_EQ(l.max_head_dim, 256u);
    EXPECT_EQ(l.max_conv_k, 8u);
    EXPECT_EQ(l.max_top_k, 1024u);
    EXPECT_EQ(l.max_gdn_dk, 256u);
    for (std::size_t i = 0; i < hb::kOpCount; ++i) {
        const auto v = be->variants(static_cast<hb::OpId>(i));
        ASSERT_EQ(v.size(), 1u);
        EXPECT_EQ(v[0].id, 0u);
        EXPECT_EQ(v[0].op, static_cast<hb::OpId>(i));
    }
    Dev d(*be);
    for (const hb::Tier t : {hb::Tier::Vram, hb::Tier::Gtt, hb::Tier::Host}) {
        const hb::TensorRef z = d.zeros(1003, t);
        EXPECT_EQ(z.buffer->bytes(), 1003u);
        EXPECT_EQ(z.buffer->host_data(), nullptr);
        EXPECT_EQ(z.buffer->host_ptr(), nullptr);
        EXPECT_TRUE(z.buffer->writable());
        const auto bytes = d.get<std::uint8_t>(z, 1003);
        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b == 0; })) << "tier " << int(t);
    }
}

TEST(VulkanBackend, ImportsAreDeviceMirrorsWrittenBackAtWait) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    auto be = hb::make_vulkan_backend(ctx);
    Dev d(*be);
    std::vector<float> host(64, 1.5f), other(16, 0.0f);
    const hb::TensorRef rw = d.import_rw(std::as_writable_bytes(std::span(host)));
    const std::vector<float> src = randn(64, 1);
    const hb::TensorRef in = d.put(src);
    // Recorded, not yet waited: the host memory is unchanged.
    be->copy(*d.s, hb::CopyArgs{in, rw, 64 * 4, {}});
    EXPECT_EQ(host[0], 1.5f);
    d.sync();
    EXPECT_TRUE(bitwise(host, src, "import written back at wait"));
    // Overlapping writable imports are rejected; disjoint ones are fine.
    expect_code([&] { (void)be->import_host(std::as_writable_bytes(std::span(host).subspan(8, 8))); }, halo::ErrorCode::Kernel,
                "overlapping writable import");
    EXPECT_NO_THROW((void)d.import_rw(std::as_writable_bytes(std::span(other))));
    // A read-only import is not an output.
    const std::vector<float> ro(8, 2.0f);
    const hb::TensorRef r = d.import_ro(std::as_bytes(std::span(ro)));
    EXPECT_FALSE(r.buffer->writable());
    expect_code([&] { be->copy(*d.s, hb::CopyArgs{in, r, 32, {}}); }, halo::ErrorCode::Kernel, "write to read-only import");
}

TEST(VulkanBackend, AbortDiscardsTheRecordingAndNeverWritesBack) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    auto be = hb::make_vulkan_backend(ctx);
    Dev d(*be);
    std::vector<float> host(32, 3.0f);
    const hb::TensorRef rw = d.import_rw(std::as_writable_bytes(std::span(host)));
    const hb::TensorRef in = d.put(randn(32, 2));
    d.sync();
    be->copy(*d.s, hb::CopyArgs{in, rw, 32 * 4, {}});
    d.s->abort();
    EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](float x) { return x == 3.0f; }));
    // The discarded copy never ran: a later wait writes back the unchanged device copy.
    d.sync();
    EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](float x) { return x == 3.0f; })) << "aborted copy executed";
    // The stream is reusable.
    be->copy(*d.s, hb::CopyArgs{in, rw, 32 * 4, {}});
    d.sync();
    EXPECT_NE(host[0], 3.0f);
}

TEST(VulkanBackend, DownloadsSnapshotAtTheirStreamPosition) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    auto be = hb::make_vulkan_backend(ctx);
    Dev d(*be);
    const std::vector<float> a = randn(16, 3), b = randn(16, 4);
    const hb::TensorRef x = d.put(a);
    const hb::TensorRef y = d.put(b);
    std::vector<float> first(16), second(16);
    be->download(*d.s, x, std::as_writable_bytes(std::span(first)));
    be->copy(*d.s, hb::CopyArgs{y, x, 64, {}});
    be->download(*d.s, x, std::as_writable_bytes(std::span(second)));
    d.sync();
    EXPECT_TRUE(bitwise(first, a, "download before the copy"));
    EXPECT_TRUE(bitwise(second, b, "download after the copy"));
}

TEST(VulkanBackend, InterfaceRules) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    Pair p = make_pair(ctx);
    Dev d(*p.vk), c(*p.cpu);
    const hb::TensorRef a = d.put(randn(64, 5)), b = d.put(randn(64, 6)), out = d.zeros(256);
    const hb::TensorRef foreign = c.zeros(256);
    // Neutral aliasing rule (backend.h): only ADD's out == a may alias.
    EXPECT_NO_THROW(p.vk->add(*d.s, hb::EltwiseArgs{a, b, a, 4, 16, {}}));
    expect_code([&] { p.vk->add(*d.s, hb::EltwiseArgs{a, b, b, 4, 16, {}}); }, halo::ErrorCode::Kernel, "add out == b");
    expect_code([&] { p.vk->swiglu(*d.s, hb::EltwiseArgs{a, b, a, 4, 16, {}}); }, halo::ErrorCode::Kernel, "swiglu out == a");
    expect_code([&] { p.vk->mul_sigmoid(*d.s, hb::EltwiseArgs{a, b, b, 4, 16, {}}); }, halo::ErrorCode::Kernel,
                "mul_sigmoid out == b");
    expect_code([&] { p.vk->gated_rms_norm(*d.s, hb::GatedNormArgs{a, b, b.shifted(0), a, 4, 16, 1e-6f, {}}); },
                halo::ErrorCode::Kernel, "gated_norm out == x");
    expect_code([&] { p.vk->rms_norm(*d.s, hb::RmsNormArgs{a, b, a, 4, 16, 1e-6f, {}}); }, halo::ErrorCode::Kernel,
                "rms_norm out == x");
    expect_code(
        [&] {
            p.vk->conv1d_silu(*d.s, hb::Conv1dArgs{a, b, out, a, {}, 4, 16, 4, 0, {}});
        },
        halo::ErrorCode::Kernel, "conv out == x");
    expect_code([&] { p.vk->gdn_gates(*d.s, hb::GdnGateArgs{a, b, out, out.shifted(64), a, out.shifted(128), 1, 16, {}}); },
                halo::ErrorCode::Kernel, "gdn_gates g_out == alpha");
    // Ownership, streams, kernel choice, status words.
    expect_code([&] { p.vk->add(*d.s, hb::EltwiseArgs{foreign, b, out, 4, 16, {}}); }, halo::ErrorCode::Kernel,
                "buffer of the cpu backend");
    expect_code([&] { p.vk->add(*c.s, hb::EltwiseArgs{a, b, out, 4, 16, {}}); }, halo::ErrorCode::Api, "stream of the cpu backend");
    expect_code([&] { p.vk->add(*d.s, hb::EltwiseArgs{a, b, out, 4, 16, {1}}); }, halo::ErrorCode::Config, "variant 1");
    hb::GetRowsArgs gr{halo::DType::F32, a, b, out, 4, 16, 1, {}, {}};
    gr.status.word = 0;
    expect_code([&] { p.vk->get_rows(*d.s, gr); }, halo::ErrorCode::Unsupported, "non-empty StatusRef");
    // Range: a view past the buffer end.
    expect_code([&] { p.vk->add(*d.s, hb::EltwiseArgs{a, b, out.shifted(200), 4, 16, {}}); }, halo::ErrorCode::Kernel,
                "out past the buffer");
    d.sync();
}

// ---------------------------------------------------------------------------------------
// Op by op against the CPU backend
// ---------------------------------------------------------------------------------------

TEST(VulkanBackend, NormsAndElementwiseMatchCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    Pair p = make_pair(ctx);
    const std::uint32_t T = 3, nh = 4, hd = 64;
    const auto qg = randn(std::size_t{T} * nh * 2 * hd, 7, 2.0f);
    const auto w = randn(hd, 8), z = randn(std::size_t{T} * nh * hd, 9, 2.0f), resid = randn(std::size_t{T} * nh * hd, 10);
    const auto up = randn(std::size_t{T} * nh * hd, 11);
    auto run = [&](hb::Backend& be) {
        Dev d(be);
        const hb::TensorRef rqg = d.put(qg), rw = d.put(w), rz = d.put(z), rr = d.put(resid), ru = d.put(up);
        const std::size_t n = std::size_t{T} * nh * hd;
        const hb::TensorRef q = d.zeros(n * 4), h = d.put(resid), y = d.zeros(n * 4), gn = d.zeros(n * 4),
                            sg = d.zeros(n * 4), ms = d.zeros(n * 4);
        // Per-head q norm straight out of the interleaved [Q | gate] rows (stride 2 * hd).
        be.rms_norm(*d.s, hb::RmsNormArgs{rqg.with_stride(2 * hd * 4), rw, q, T * nh, hd, 1e-6f, {}});
        be.add_rms_norm(*d.s, hb::AddRmsNormArgs{h, q, h, rw, y, T * nh, hd, 1e-6f, {}});  // h == a
        be.gated_rms_norm(*d.s, hb::GatedNormArgs{y, rz, rw, gn, T * nh, hd, 1e-6f, {}});
        be.swiglu(*d.s, hb::EltwiseArgs{gn, ru, sg, T * nh, hd, {}});
        be.mul_sigmoid(*d.s, hb::EltwiseArgs{sg, rqg.shifted(hd * 4).with_stride(2 * hd * 4), ms, T * nh, hd, {}});
        be.add(*d.s, hb::EltwiseArgs{ms, rr, ms, T * nh, hd, {}});  // out == a
        (void)rr;
        return std::array<std::vector<float>, 6>{d.get<float>(q, n), d.get<float>(h, n), d.get<float>(y, n),
                                                 d.get<float>(gn, n), d.get<float>(sg, n), d.get<float>(ms, n)};
    };
    const auto c = run(*p.cpu), v = run(*p.vk);
    EXPECT_TRUE(close(v[0], c[0], k_wiring, "rms_norm strided"));
    EXPECT_TRUE(close(v[1], c[1], k_wiring, "add_rms_norm h"));
    EXPECT_TRUE(close(v[2], c[2], k_wiring, "add_rms_norm y"));
    EXPECT_TRUE(close(v[3], c[3], k_wiring, "gated_rms_norm"));
    EXPECT_TRUE(close(v[4], c[4], k_wiring, "swiglu"));
    EXPECT_TRUE(close(v[5], c[5], k_wiring, "mul_sigmoid(gate half) + add"));
}

TEST(VulkanBackend, GemvGetRowsLmHeadArgmaxTopKCopyMatchCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    Pair p = make_pair(ctx);
    const std::uint32_t rows = 300, cols = 256, n_vec = 3;
    // A Q8_0 table with normal fp16 scales (bytes shared by both backends).
    std::mt19937 rng(12);
    const std::size_t rb = halo::tensor::row_bytes(halo::DType::Q8_0, cols);
    std::vector<std::uint8_t> q8(rb * rows);
    for (auto& b : q8) b = static_cast<std::uint8_t>(rng());
    for (std::size_t blk = 0; blk < q8.size() / 34; ++blk) {
        const std::uint16_t h = static_cast<std::uint16_t>(0x2000 + (rng() % 0x800));  // positive normal fp16 scale
        std::memcpy(&q8[blk * 34], &h, 2);
    }
    const auto x = randn(std::size_t{n_vec} * cols, 13);
    const std::vector<std::int32_t> ids{0, 299, 17, 17};
    auto run = [&](hb::Backend& be) {
        Dev d(be);
        const hb::TensorRef w = d.import_ro(std::as_bytes(std::span(q8)));
        const hb::TensorRef rx = d.put(x), rid = d.import_ro(std::as_bytes(std::span(ids)));
        const hb::TensorRef y = d.zeros(std::size_t{n_vec} * rows * 4), emb = d.zeros(ids.size() * cols * 4);
        const hb::TensorRef logits = d.zeros(std::size_t{n_vec} * rows * 4);
        const hb::TensorRef res = d.zeros(n_vec * hb::kArgmaxResultBytes), res2 = d.zeros(n_vec * hb::kArgmaxResultBytes);
        const hb::TensorRef tid = d.zeros(n_vec * 8 * 4), tval = d.zeros(n_vec * 8 * 4), cp = d.zeros(cols * 4);
        be.gemv(*d.s, hb::GemvArgs{halo::DType::Q8_0, w, rx, y, rows, cols, n_vec, {}});
        be.get_rows(*d.s, hb::GetRowsArgs{halo::DType::Q8_0, w, rid, emb, rows, cols, 4, {}, {}});
        be.lm_head(*d.s, hb::LmHeadArgs{hb::GemvArgs{halo::DType::Q8_0, w, rx, logits, rows, cols, n_vec, {}}, res, {}, {}});
        be.argmax(*d.s, hb::ArgmaxArgs{y, res2, rows, n_vec, {}, {}});
        be.top_k(*d.s, hb::TopKArgs{y, tid, tval, rows, 8, n_vec, {}, {}});
        be.copy(*d.s, hb::CopyArgs{rx.shifted(cols * 4), cp, cols * 4, {}});
        struct R {
            std::vector<float> y, emb, logits, tval, cp;
            std::vector<std::byte> res, res2;
            std::vector<std::int32_t> tid;
        } r;
        r.y = d.get<float>(y, std::size_t{n_vec} * rows);
        r.emb = d.get<float>(emb, ids.size() * cols);
        r.logits = d.get<float>(logits, std::size_t{n_vec} * rows);
        r.res = d.get<std::byte>(res, n_vec * hb::kArgmaxResultBytes);
        r.res2 = d.get<std::byte>(res2, n_vec * hb::kArgmaxResultBytes);
        r.tid = d.get<std::int32_t>(tid, n_vec * 8);
        r.tval = d.get<float>(tval, n_vec * 8);
        r.cp = d.get<float>(cp, cols);
        return r;
    };
    const auto c = run(*p.cpu);
    const auto v = run(*p.vk);
    EXPECT_TRUE(close(v.y, c.y, k_wiring, "gemv Q8_0 n_vec 3"));
    EXPECT_TRUE(bitwise(v.emb, c.emb, "get_rows Q8_0"));
    EXPECT_TRUE(close(v.logits, c.logits, k_wiring, "lm_head logits"));
    EXPECT_TRUE(bitwise(v.cp, c.cp, "copy"));
    // Argmax / top_k on the Vulkan logits: exact functions of their input, so compare with
    // the CPU functions applied to the Vulkan gemv output.
    const auto best = hb::decode_argmax(v.res2);
    for (std::uint32_t i = 0; i < n_vec; ++i) {
        const auto want = halo::cpu::argmax(std::span<const float>(v.y).subspan(std::size_t{i} * rows, rows));
        EXPECT_EQ(best[i].index, want.index) << "argmax vector " << i;
        const auto tk = halo::cpu::top_k(std::span<const float>(v.y).subspan(std::size_t{i} * rows, rows), 8);
        for (std::size_t j = 0; j < 8; ++j) {
            EXPECT_EQ(v.tid[i * 8 + j], tk[j].index) << "top_k " << i << " rank " << j;
            EXPECT_EQ(std::bit_cast<std::uint32_t>(v.tval[i * 8 + j]), std::bit_cast<std::uint32_t>(tk[j].value));
        }
    }
    // LM head vs the CPU's own head: same token (the random table has no near-ties here).
    const auto vb = hb::decode_argmax(v.res), cb = hb::decode_argmax(c.res);
    for (std::uint32_t i = 0; i < n_vec; ++i) EXPECT_EQ(vb[i].index, cb[i].index) << "lm_head vector " << i;
}

TEST(VulkanBackend, RopeKvWriteAttentionMatchCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    Pair p = make_pair(ctx);
    const std::uint32_t T = 3, nh = 4, nkv = 2, hd = 64, rot = 32, bt = 4, layers = 2, layer = 1, blocks = 6;
    const std::uint32_t kvd = nkv * hd, start = 5;
    const auto qg = randn(std::size_t{T} * nh * 2 * hd, 20), kin = randn(std::size_t{T} * kvd, 21), vin = randn(std::size_t{T} * kvd, 22);
    const std::vector<std::int32_t> pos{5, 6, 7};
    const std::vector<std::uint32_t> table{4, 1, 3};  // fragmented; rows 0..7 in blocks 4, 1
    const std::size_t pool_floats = std::size_t{blocks} * layers * 2 * bt * kvd;
    const auto pool0 = randn(pool_floats, 23);
    auto run = [&](hb::Backend& be, std::vector<float>& pool) {
        Dev d(be);
        const hb::TensorRef rq = d.put(qg), rk = d.put(kin), rv = d.put(vin);
        const hb::TensorRef rpos = d.import_ro(std::as_bytes(std::span(pos)));
        const hb::TensorRef rt = d.import_ro(std::as_bytes(std::span(table)));
        const hb::TensorRef rp = d.import_rw(std::as_writable_bytes(std::span(pool)));
        const hb::TensorRef out = d.zeros(std::size_t{T} * nh * hd * 4);
        // Q in place inside [Q | gate] (head stride 2 * hd), K dense.
        be.partial_rope(*d.s, hb::RopeArgs{rq.with_stride(nh * 2 * hd * 4), rpos, T, nh, hd, rot, 1e7f, 2 * hd, {}});
        be.partial_rope(*d.s, hb::RopeArgs{rk, rpos, T, nkv, hd, rot, 1e7f, 0, {}});
        be.kv_write(*d.s, hb::KvWriteArgs{rp, rt, rk, rv, blocks, layers, layer, bt, kvd, 3, start, T, {}, {}});
        hb::AttentionArgs at;
        at.q = rq.with_stride(nh * 2 * hd * 4);
        at.q_head_stride = 2 * hd;
        at.kv_pool = rp;
        at.block_table = rt;
        at.out = out;
        at.n_pool_blocks = blocks;
        at.n_layers = layers;
        at.layer = layer;
        at.block_tokens = bt;
        at.n_block_table = 3;
        at.n_head = nh;
        at.n_kv_head = nkv;
        at.head_dim = hd;
        at.n_tokens = T;
        at.q_offset = start;
        at.scale = 0.125f;
        be.attention(*d.s, at);
        std::array<std::vector<float>, 3> r{d.get<float>(rq, qg.size()), d.get<float>(rk, kin.size()),
                                            d.get<float>(out, std::size_t{T} * nh * hd)};
        return r;
    };
    std::vector<float> pc = pool0, pv = pool0;
    const auto c = run(*p.cpu, pc);
    const auto v = run(*p.vk, pv);
    EXPECT_TRUE(bitwise(v[0], c[0], "rope Q in [Q|gate] (head stride)"));
    EXPECT_TRUE(bitwise(v[1], c[1], "rope K dense"));
    EXPECT_TRUE(bitwise(pv, pc, "KV pool after kv_write (import written back)"));
    EXPECT_TRUE(close(v[2], c[2], k_wiring, "attention (head-strided q, fragmented table, layer 1)"));
    // Positions that live only on the device cannot feed the host cos/sin table.
    Dev d(*p.vk);
    const hb::TensorRef dev_pos = d.put(pos), x = d.put(kin);
    expect_code([&] { p.vk->partial_rope(*d.s, hb::RopeArgs{x, dev_pos, T, nkv, hd, rot, 1e7f, 0, {}}); },
                halo::ErrorCode::Unsupported, "device-resident positions");
}

TEST(VulkanBackend, ConvAndGatedDeltaRuleBothFormsMatchCpu) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    Pair p = make_pair(ctx);
    const std::uint32_t T = 11, nk = 2, nv = 4, dk = 32, dv = 32, K = 4, n_slots = 3;
    const std::uint32_t C = 2 * nk * dk + nv * dv;
    const auto qkv = randn(std::size_t{T} * C, 30), cw = randn(std::size_t{C} * K, 31, 0.5f);
    const auto alpha = randn(std::size_t{T} * nv, 32), bin = randn(std::size_t{T} * nv, 33);
    const auto dt = randn(nv, 34);
    std::vector<float> ssm_a(nv);
    for (std::uint32_t j = 0; j < nv; ++j) ssm_a[j] = -0.5f - 0.1f * float(j);
    const auto cs0 = randn(std::size_t{K - 1} * C, 35), s0 = randn(std::size_t{nv} * dk * dv, 36, 0.3f);
    const std::size_t sn = std::size_t{nv} * dk * dv;
    std::array<std::vector<float>, 2> recurrent_out, chunked_out;  // {cpu, vk}
    for (const hb::GdnForm form : {hb::GdnForm::Recurrent, hb::GdnForm::Chunked}) {
        const bool ch = form == hb::GdnForm::Chunked;
        SCOPED_TRACE(ch ? "chunked" : "recurrent");
        struct Host {
            std::vector<float> conv_state, conv_slots, state, slots;
        };
        auto run = [&](hb::Backend& be, Host& h) {
            Dev d(be);
            const hb::TensorRef rqkv = d.put(qkv), rcw = d.put(cw), ral = d.put(alpha), rb = d.put(bin), rdt = d.put(dt),
                                ra = d.put(ssm_a);
            const hb::TensorRef rcs = d.import_rw(std::as_writable_bytes(std::span(h.conv_state)));
            const hb::TensorRef rcsl = d.import_rw(std::as_writable_bytes(std::span(h.conv_slots)));
            const hb::TensorRef rst = d.import_rw(std::as_writable_bytes(std::span(h.state)));
            const hb::TensorRef rsl = d.import_rw(std::as_writable_bytes(std::span(h.slots)));
            const hb::TensorRef conv = d.zeros(std::size_t{T} * C * 4), g = d.zeros(std::size_t{T} * nv * 4),
                                bs = d.zeros(std::size_t{T} * nv * 4), o = d.zeros(std::size_t{T} * nv * dv * 4);
            be.gdn_gates(*d.s, hb::GdnGateArgs{ral, rb, rdt, ra, g, bs, T, nv, {}});
            be.conv1d_silu(*d.s, hb::Conv1dArgs{rqkv, rcw, rcs, conv, ch ? hb::TensorRef{} : rcsl, T, C, K, ch ? 0 : n_slots, {}});
            hb::GdnArgs ga;
            ga.form = form;
            ga.q = conv.with_stride(C * 4);
            ga.k = conv.shifted(nk * dk * 4).with_stride(C * 4);
            ga.v = conv.shifted(2 * nk * dk * 4).with_stride(C * 4);
            ga.g = g;
            ga.beta = bs;
            ga.state = rst;
            ga.state_slots = rsl;
            ga.out = o;
            ga.n_k = nk;
            ga.n_v = nv;
            ga.d_k = dk;
            ga.d_v = dv;
            ga.n_tokens = T;
            ga.n_slots = n_slots;
            ga.qk_l2norm = true;
            ga.q_scale = 1.0f / std::sqrt(float(dk));
            ga.chunk_size = 4;  // several chunks + a partial one
            be.gated_delta_rule(*d.s, ga);
            std::array<std::vector<float>, 4> r{d.get<float>(g, std::size_t{T} * nv), d.get<float>(bs, std::size_t{T} * nv),
                                                d.get<float>(conv, std::size_t{T} * C), d.get<float>(o, std::size_t{T} * nv * dv)};
            return r;
        };
        Host hc{cs0, std::vector<float>(std::size_t{n_slots} * (K - 1) * C, 9.0f), s0, std::vector<float>(n_slots * sn, 9.0f)};
        Host hv_ = hc;
        const auto c = run(*p.cpu, hc);
        const auto v = run(*p.vk, hv_);
        EXPECT_TRUE(close(v[0], c[0], k_wiring, "gdn_gates g (alpha, dt_bias, a)"));
        EXPECT_TRUE(close(v[1], c[1], k_wiring, "gdn_gates beta (beta)"));
        EXPECT_TRUE(close(v[2], c[2], k_wiring, "conv1d_silu out"));
        EXPECT_TRUE(bitwise(hv_.conv_state, hc.conv_state, "conv state written back"));
        EXPECT_TRUE(bitwise(hv_.conv_slots, hc.conv_slots, "conv slots written back"));
        EXPECT_TRUE(close(v[3], c[3], k_wiring, "gated_delta_rule out"));
        EXPECT_TRUE(close(hv_.state, hc.state, k_wiring, "gated_delta_rule state written back"));
        EXPECT_TRUE(close(hv_.slots, hc.slots, k_wiring, "gated_delta_rule slots written back"));
        // The two forms are different computations (chunked is not bitwise the recurrent form on
        // either backend): guards against a form that is silently ignored.
        (ch ? chunked_out : recurrent_out) = {c[3], v[3]};
    }
    EXPECT_NE(std::memcmp(recurrent_out[0].data(), chunked_out[0].data(), recurrent_out[0].size() * 4), 0) << "cpu forms";
    EXPECT_NE(std::memcmp(recurrent_out[1].data(), chunked_out[1].data(), recurrent_out[1].size() * 4), 0) << "vulkan forms";
    for (std::size_t s = 0; s < 2; ++s) {
        double m = 0;
        for (std::size_t i = 0; i < recurrent_out[s].size(); ++i) {
            m = std::max(m, std::abs(double(recurrent_out[s][i]) - double(chunked_out[s][i])));
        }
        std::cout << "[vk-backend] " << (s == 0 ? "cpu" : "vulkan") << " max|recurrent - chunked| = " << m << "\n";
    }
}

TEST(VulkanBackend, DeviceDataErrorsRaiseAtWaitAndSkipWriteBack) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    Pair p = make_pair(ctx);
    const std::uint32_t T = 5, nv = 2, dk = 16, dv = 16;
    const auto q = randn(std::size_t{T} * dk, 40), k = randn(std::size_t{T} * dk, 41), v = randn(std::size_t{T} * nv * dv, 42);
    std::vector<float> g(std::size_t{T} * nv, -0.5f);
    g[7] = 0.25f;  // a positive g: cpu::gated_delta_rule_chunked throws
    const std::vector<float> beta(std::size_t{T} * nv, 0.5f);
    const std::vector<float> s0 = randn(std::size_t{nv} * dk * dv, 43);
    auto args = [&](Dev& d, std::vector<float>& state, hb::TensorRef out) {
        hb::GdnArgs a;
        a.form = hb::GdnForm::Chunked;
        a.q = d.put(q);
        a.k = d.put(k);
        a.v = d.put(v);
        a.g = d.put(g);
        a.beta = d.put(beta);
        a.state = d.import_rw(std::as_writable_bytes(std::span(state)));
        a.out = out;
        a.n_k = 1;
        a.n_v = nv;
        a.d_k = dk;
        a.d_v = dv;
        a.n_tokens = T;
        a.chunk_size = 2;
        return a;
    };
    {
        Dev c(*p.cpu);
        std::vector<float> st = s0;
        const hb::GdnArgs a = args(c, st, c.zeros(std::size_t{T} * nv * dv * 4));
        EXPECT_THROW(p.cpu->gated_delta_rule(*c.s, a), halo::Error) << "cpu: synchronous";
    }
    Dev d(*p.vk);
    std::vector<float> st = s0;
    const hb::GdnArgs a = args(d, st, d.zeros(std::size_t{T} * nv * dv * 4));
    ASSERT_NO_THROW(p.vk->gated_delta_rule(*d.s, a)) << "vulkan: detected on the device";
    d.s->submit();
    expect_code([&] { d.s->wait(); }, halo::ErrorCode::Kernel, "positive g at wait");
    EXPECT_TRUE(bitwise(st, s0, "state not written back after a data error"));
    // Bad block id in KV write / attention, bad row id in GET_ROWS: also at wait.
    const std::vector<std::uint32_t> table{0, 99};
    std::vector<float> pool(std::size_t{2} * 2 * 4 * 8, 0.0f);
    const hb::TensorRef rt = d.import_ro(std::as_bytes(std::span(table)));
    const hb::TensorRef rp = d.import_rw(std::as_writable_bytes(std::span(pool)));
    const hb::TensorRef kv = d.put(randn(8, 44));
    p.vk->kv_write(*d.s, hb::KvWriteArgs{rp, rt, kv, kv, 2, 1, 0, 4, 8, 2, 4, 1, {}, {}});
    d.s->submit();
    expect_code([&] { d.s->wait(); }, halo::ErrorCode::Kernel, "bad block id at wait");
    const std::vector<std::int32_t> bad{3, 10};
    const hb::TensorRef tb = d.put(randn(4 * 8, 45)), ids = d.import_ro(std::as_bytes(std::span(bad))),
                        emb = d.zeros(2 * 8 * 4);
    p.vk->get_rows(*d.s, hb::GetRowsArgs{halo::DType::F32, tb, ids, emb, 4, 8, 2, {}, {}});
    d.s->submit();
    expect_code([&] { d.s->wait(); }, halo::ErrorCode::Kernel, "bad row id at wait");
    // The stream stays usable.
    EXPECT_NO_THROW(d.sync());
}
