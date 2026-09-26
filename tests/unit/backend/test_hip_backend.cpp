// WS-BI-3 (ADR-001 §6.3, §5.5, §5.6): the HIP backend adapter in host emulation, op by op,
// against the CPU backend on the same input bytes.
//  (a) bitwise: every variant docs/hip.md marks bit-identical (every op's default, plus
//      gemv_generic_b64, gemm_t16x16_b256 and attn_exact_b128 chosen through KernelChoice) —
//      both emulation orders; a mismatch is a finding, never a tolerance to widen;
//  (b) bounded: the wave GEMV variants within docs/hip.md's derived summation bound, with the
//      LM-head argmax index equal when the CPU's top-2 gap exceeds twice the bound; the
//      online attention variants within the docs/hip.md attention bound;
//  - non-vacuity: the default (wave) GEMV differs from the CPU in at least one bit, so the
//    bitwise comparisons are not comparing a backend with itself;
//  - host validation parity: every rejection of the CPU backend (ownership, read-only output,
//    aliasing, range, alignment, foreign stream, unknown variant, status words) has the same
//    ErrorCode here; the HIP limits raise Unsupported (never Kernel);
//  - data errors (bad id, bad block, positive g) raise Error(Kernel) at the op call in
//    emulation, as on the CPU backend; a NaN logit in the fused LM head instead poisons only
//    its own row ({-1, NaN} in the result words, H1) on both backends, while standalone
//    argmax / top-k still raise Kernel on NaN;
//  - registry: the (id -> name) table is pinned; every HIP registry variant has an id; ids
//    select what they name; a wrong op/form id is Error(Config);
//  - modes: Kind, describe(), device mode without a device, poisoned allocations.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/backend/hip_backend.h"
#include "halo/backends/hip/registry.h"
#include "halo/backends/hip/runtime.h"
#include "halo/core/error.h"
#include "halo/tensor/fp16.h"
#include "halo/tensor/quant.h"

using namespace halo::backend;  // NOLINT(google-build-using-namespace)
using halo::DType;
using halo::ErrorCode;

namespace {

constexpr double kU = 5.9604644775390625e-08;  // 2^-24

std::vector<float> rnd(std::size_t n, std::uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
}

::testing::AssertionResult same_bits(std::span<const float> a, std::span<const float> b, const std::string& what) {
    if (a.size() != b.size()) return ::testing::AssertionFailure() << what << ": size " << a.size() << " vs " << b.size();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) {
            return ::testing::AssertionFailure() << what << ": first difference at " << i << " of " << a.size() << " (hip "
                                                 << a[i] << ", cpu " << b[i] << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

std::size_t differing(std::span<const float> a, std::span<const float> b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) n += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
    return n;
}

std::optional<ErrorCode> code_of(const std::function<void()>& fn, std::string* msg = nullptr) {
    try {
        fn();
    } catch (const halo::Error& e) {
        if (msg != nullptr) *msg = e.what();
        return e.code();
    }
    return std::nullopt;
}

/// Random but valid weight bytes (test_hip_gemv's recipe): random quants, finite f16 scales.
std::vector<std::byte> make_weights(DType type, std::uint32_t rows, std::uint32_t cols, std::uint64_t stride, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(-0.05f, 0.05f);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    const std::uint64_t rb = halo::tensor::row_bytes(type, cols);
    std::vector<std::byte> buf((rows - 1) * stride + rb);
    for (auto& b : buf) b = static_cast<std::byte>(byte(rng));
    auto put16 = [&](std::uint64_t at, float f) {
        const std::uint16_t h = halo::tensor::fp32_to_fp16(f);
        buf[at] = static_cast<std::byte>(h & 0xFF);
        buf[at + 1] = static_cast<std::byte>(h >> 8);
    };
    for (std::uint32_t n = 0; n < rows; ++n) {
        const std::uint64_t row = n * stride;
        switch (type) {
            case DType::F32:
                for (std::uint32_t i = 0; i < cols; ++i) {
                    const float f = val(rng);
                    std::memcpy(&buf[row + 4 * i], &f, 4);
                }
                break;
            case DType::Q8_0:
                for (std::uint32_t b = 0; b < cols / 32; ++b) put16(row + 34 * b, scale(rng));
                break;
            case DType::Q4_K:
            case DType::Q5_K: {
                const std::uint64_t bb = type == DType::Q4_K ? 144 : 176;
                for (std::uint32_t b = 0; b < cols / 256; ++b) {
                    put16(row + bb * b, scale(rng));
                    put16(row + bb * b + 2, std::fabs(scale(rng)));
                }
                break;
            }
            case DType::Q6_K:
                for (std::uint32_t b = 0; b < cols / 256; ++b) put16(row + 210 * b + 208, scale(rng));
                break;
            case DType::IQ4_XS:
                for (std::uint32_t b = 0; b < cols / 256; ++b) put16(row + 136 * b, scale(rng));
                break;
            default:
                break;
        }
    }
    return buf;
}

// ---------------------------------------------------------------------------------------
// Two backends driven with the same input bytes
// ---------------------------------------------------------------------------------------

/// One operand on both sides: separate host memory, identical contents, imported zero-copy.
struct Dual {
    std::vector<std::byte> c, h;
    TensorRef rc, rh;
    [[nodiscard]] std::span<const float> fc() const { return {reinterpret_cast<const float*>(c.data()), c.size() / 4}; }
    [[nodiscard]] std::span<const float> fh() const { return {reinterpret_cast<const float*>(h.data()), h.size() / 4}; }
};

class Pair {
public:
    explicit Pair(HipBackendOptions o = {}) : cpu_(make_cpu_backend(nullptr)), hip_(make_hip_backend(o)) {
        cs_ = cpu_->create_stream();
        hs_ = hip_->create_stream();
    }
    Backend& cpu() { return *cpu_; }
    Backend& hip() { return *hip_; }
    Stream& cs() { return *cs_; }
    Stream& hs() { return *hs_; }

    Dual& bytes(std::vector<std::byte> v, bool readonly = false) {
        auto d = std::make_unique<Dual>();
        d->c = v;
        d->h = std::move(v);
        if (readonly) {
            keep_.push_back(cpu_->import_host_readonly(d->c));
            d->rc = TensorRef::of(*keep_.back());
            keep_.push_back(hip_->import_host_readonly(d->h));
            d->rh = TensorRef::of(*keep_.back());
        } else {
            keep_.push_back(cpu_->import_host(d->c));
            d->rc = TensorRef::of(*keep_.back());
            keep_.push_back(hip_->import_host(d->h));
            d->rh = TensorRef::of(*keep_.back());
        }
        duals_.push_back(std::move(d));
        return *duals_.back();
    }
    Dual& f32(const std::vector<float>& v) {
        std::vector<std::byte> b(v.size() * 4);
        std::memcpy(b.data(), v.data(), b.size());
        return bytes(std::move(b));
    }
    template <class T>
    Dual& words(const std::vector<T>& v) {
        std::vector<std::byte> b(v.size() * sizeof(T));
        std::memcpy(b.data(), v.data(), b.size());
        return bytes(std::move(b));
    }
    /// An output of n floats prefilled with a sentinel on both sides.
    Dual& out(std::size_t n, float fill = -3.5f) { return f32(std::vector<float>(n, fill)); }

    /// Runs `op(backend, stream, side)`; side 0 = cpu, 1 = hip.
    void run(const std::function<void(Backend&, Stream&, int)>& op) {
        op(*cpu_, *cs_, 0);
        op(*hip_, *hs_, 1);
        hs_->submit();
        hs_->wait();
    }

private:
    std::unique_ptr<Backend> cpu_, hip_;
    std::unique_ptr<Stream> cs_, hs_;
    std::vector<std::unique_ptr<Buffer>> keep_;
    std::vector<std::unique_ptr<Dual>> duals_;
};

const TensorRef& S(const Dual& d, int side) { return side == 0 ? d.rc : d.rh; }

HipBackendOptions emu(bool reverse, bool bitwise_profile = false) {
    HipBackendOptions o;
    o.emulation_reverse_order = reverse;
    if (bitwise_profile) o.defaults = hip_bitwise_defaults();
    return o;
}

// Variant ids (pinned below).
constexpr std::uint32_t kWaveR4 = 5, kWaveR8 = 6, kGeneric = 7, kGemm = 8;
constexpr std::uint32_t kRec128 = 12, kRec64 = 13, kChunk64 = 14, kChunk32 = 15;
constexpr std::uint32_t kOnline128 = 20, kOnline64 = 21, kExact = 22;

class HipOps : public ::testing::TestWithParam<bool> {};  // param: reverse emulation order

// ---------------------------------------------------------------------------------------
// (a) bitwise
// ---------------------------------------------------------------------------------------

TEST_P(HipOps, NormsGatesEltwiseAreBitIdentical) {
    Pair p(emu(GetParam()));
    // Per-head RMS norm straight out of an interleaved [Q | gate] row (stride 2 * hd), as the forward.
    const std::uint32_t T = 3, nh = 4, hd = 64;
    Dual& qg = p.f32(rnd(T * 2 * nh * hd, 1, -3, 3));
    Dual& wq = p.f32(rnd(hd, 2, 0.5f, 1.5f));
    Dual& qn = p.out(T * nh * hd);
    p.run([&](Backend& b, Stream& s, int i) {
        b.rms_norm(s, RmsNormArgs{S(qg, i).with_stride(2 * hd * 4), S(wq, i), S(qn, i), T * nh, hd, 1e-6f, {}});
    });
    EXPECT_TRUE(same_bits(qn.fh(), qn.fc(), "per-head rms_norm"));

    // ADD + RMS_NORM with the residual accumulate h == a, over 37 columns (a tail).
    const std::uint32_t R = 5, E = 37;
    Dual& x = p.f32(rnd(R * E, 3, -2, 2));
    Dual& y = p.f32(rnd(R * E, 4, -2, 2));
    Dual& w = p.f32(rnd(E, 5, 0.5f, 1.5f));
    Dual& xn = p.out(R * E);
    p.run([&](Backend& b, Stream& s, int i) {
        b.add_rms_norm(s, AddRmsNormArgs{S(x, i), S(y, i), S(x, i), S(w, i), S(xn, i), R, E, 1e-6f, {}});
    });
    EXPECT_TRUE(same_bits(x.fh(), x.fc(), "add_rms_norm h"));
    EXPECT_TRUE(same_bits(xn.fh(), xn.fc(), "add_rms_norm out"));

    // GDN gates, including the softplus threshold branch (a + dt_bias > 20) and exp overflow.
    const std::uint32_t Tg = 5, nv = 48;
    std::vector<float> alpha = rnd(Tg * nv, 6, -8, 8), beta = rnd(Tg * nv, 7, -8, 8);
    alpha[3] = 30.0f;
    alpha[4] = 100.0f;
    alpha[5] = -40.0f;
    beta[6] = 95.0f;
    Dual& al = p.f32(alpha);
    Dual& be = p.f32(beta);
    Dual& dt = p.f32(rnd(nv, 8, -1, 1));
    Dual& av = p.f32(rnd(nv, 9, -2, -0.01f));
    Dual& g = p.out(Tg * nv);
    Dual& bs = p.out(Tg * nv);
    p.run([&](Backend& b, Stream& s, int i) {
        b.gdn_gates(s, GdnGateArgs{S(al, i), S(be, i), S(dt, i), S(av, i), S(g, i), S(bs, i), Tg, nv, {}});
    });
    EXPECT_TRUE(same_bits(g.fh(), g.fc(), "gdn_gates g"));
    EXPECT_TRUE(same_bits(bs.fh(), bs.fc(), "gdn_gates beta"));

    // Gated norm, SwiGLU, MUL_SIGMOID with a head-strided gate, ADD with out == a.
    Dual& z = p.f32(rnd(R * E, 10, -3, 3));
    Dual& on = p.out(R * E);
    p.run([&](Backend& b, Stream& s, int i) {
        b.gated_rms_norm(s, GatedNormArgs{S(y, i), S(z, i), S(w, i), S(on, i), R, E, 1e-6f, {}});
    });
    EXPECT_TRUE(same_bits(on.fh(), on.fc(), "gated_rms_norm"));
    Dual& sw = p.out(R * E);
    p.run([&](Backend& b, Stream& s, int i) { b.swiglu(s, EltwiseArgs{S(y, i), S(z, i), S(sw, i), R, E, {}}); });
    EXPECT_TRUE(same_bits(sw.fh(), sw.fc(), "swiglu"));
    Dual& att = p.f32(rnd(T * nh * hd, 11, -2, 2));
    Dual& gated = p.out(T * nh * hd);
    p.run([&](Backend& b, Stream& s, int i) {
        b.mul_sigmoid(s, EltwiseArgs{S(att, i), S(qg, i).shifted(hd * 4).with_stride(2 * hd * 4), S(gated, i), T * nh, hd, {}});
    });
    EXPECT_TRUE(same_bits(gated.fh(), gated.fc(), "mul_sigmoid (strided gate)"));
    p.run([&](Backend& b, Stream& s, int i) { b.add(s, EltwiseArgs{S(y, i), S(z, i), S(y, i), R, E, {}}); });
    EXPECT_TRUE(same_bits(y.fh(), y.fc(), "add out == a"));
}

TEST_P(HipOps, RopeDenseAndHeadStridedAreBitIdentical) {
    Pair p(emu(GetParam()));
    const std::uint32_t T = 5, nh = 4, hd = 64, rot = 16;
    const std::vector<std::int32_t> pos{0, 1, 17, 1000, 262143};
    Dual& ps = p.words(pos);
    Dual& dense = p.f32(rnd(T * nh * hd, 20, -2, 2));
    p.run([&](Backend& b, Stream& s, int i) { b.partial_rope(s, RopeArgs{S(dense, i), S(ps, i), T, nh, hd, rot, 1e7f, 0, {}}); });
    EXPECT_TRUE(same_bits(dense.fh(), dense.fc(), "rope dense"));
    Dual& inter = p.f32(rnd(T * 2 * nh * hd, 21, -2, 2));  // [Q | gate] per head; RoPE on Q in place
    p.run([&](Backend& b, Stream& s, int i) {
        b.partial_rope(s, RopeArgs{S(inter, i), S(ps, i), T, nh, hd, rot, 1e7f, 2 * hd, {}});
    });
    EXPECT_TRUE(same_bits(inter.fh(), inter.fc(), "rope head_stride 2*hd (gate halves untouched on both)"));
}

TEST_P(HipOps, ConvAndGdnRecurrentAndChunkedAreBitIdentical) {
    Pair p(emu(GetParam()));
    const std::uint32_t n_k = 2, n_v = 4, d_k = 32, d_v = 32, K = 4;
    const std::uint32_t key = n_k * d_k, val = n_v * d_v, C = 2 * key + val;
    for (const std::uint32_t T : {1u, 4u, 70u}) {
        SCOPED_TRACE("T=" + std::to_string(T));
        const std::uint32_t slots = T == 4 ? 3 : 0;
        Dual& qkv = p.f32(rnd(T * C, 30 + T, -2, 2));
        Dual& wc = p.f32(rnd(C * K, 31, -1, 1));
        Dual& cst = p.f32(rnd((K - 1) * C, 32, -1, 1));
        Dual& conv = p.out(T * C);
        Dual& csl = p.out(std::max<std::size_t>(1, slots * (K - 1) * C));
        p.run([&](Backend& b, Stream& s, int i) {
            b.conv1d_silu(s, Conv1dArgs{S(qkv, i), S(wc, i), S(cst, i), S(conv, i), slots ? S(csl, i) : TensorRef{}, T, C, K, slots, {}});
        });
        EXPECT_TRUE(same_bits(conv.fh(), conv.fc(), "conv out"));
        EXPECT_TRUE(same_bits(cst.fh(), cst.fc(), "conv state"));
        EXPECT_TRUE(same_bits(csl.fh(), csl.fc(), "conv slots"));

        Dual& g = p.f32(rnd(T * n_v, 33, -3, -0.01f));
        Dual& beta = p.f32(rnd(T * n_v, 34, 0.05f, 0.95f));
        const std::size_t sf = static_cast<std::size_t>(n_v) * d_k * d_v;
        Dual& st = p.f32(rnd(sf, 35, -0.5f, 0.5f));
        Dual& ssl = p.out(std::max<std::size_t>(1, slots * sf));
        Dual& o = p.out(T * val);
        const std::uint64_t cs = C * 4;
        for (const bool chunked : {false, true}) {
            if (chunked && slots) continue;  // the forward uses the recurrent form for slot-writing verify
            for (const std::uint32_t chunk : {64u, 8u}) {
                if (!chunked && chunk == 8) continue;
                SCOPED_TRACE(chunked ? "chunked " + std::to_string(chunk) : "recurrent");
                p.run([&](Backend& b, Stream& s, int i) {
                    GdnArgs a{};
                    a.form = chunked ? GdnForm::Chunked : GdnForm::Recurrent;
                    a.q = S(conv, i).with_stride(cs);
                    a.k = S(conv, i).shifted(key * 4).with_stride(cs);
                    a.v = S(conv, i).shifted(2 * key * 4).with_stride(cs);
                    a.g = S(g, i);
                    a.beta = S(beta, i);
                    a.state = S(st, i);
                    a.state_slots = slots ? S(ssl, i) : TensorRef{};
                    a.out = S(o, i);
                    a.n_k = n_k;
                    a.n_v = n_v;
                    a.d_k = d_k;
                    a.d_v = d_v;
                    a.n_tokens = T;
                    a.n_slots = slots;
                    a.qk_l2norm = true;
                    a.q_scale = 1.0f / std::sqrt(static_cast<float>(d_k));
                    a.chunk_size = chunk;
                    b.gated_delta_rule(s, a);
                });
                EXPECT_TRUE(same_bits(o.fh(), o.fc(), "gdn out"));
                EXPECT_TRUE(same_bits(st.fh(), st.fc(), "gdn state"));
                EXPECT_TRUE(same_bits(ssl.fh(), ssl.fc(), "gdn slots"));
            }
        }
    }
}

TEST_P(HipOps, GemvGenericAndGemmAndGetRowsAreBitIdentical) {
    Pair p(emu(GetParam()));
    for (const DType t : {DType::F32, DType::Q8_0, DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::IQ4_XS}) {
        SCOPED_TRACE(std::string(halo::traits(t).name));
        const std::uint32_t rows = 37, cols = 512, T = 17;
        const std::uint64_t rb = halo::tensor::row_bytes(t, cols);
        const std::uint64_t stride = rb + (t == DType::F32 ? 4 : 3);
        Dual& w = p.bytes(make_weights(t, rows, cols, stride, 40), true);
        Dual& x = p.f32(rnd(T * (cols + 5), 41, -1, 1));
        for (const std::uint32_t id : {kGeneric, kGemm}) {
            SCOPED_TRACE("variant " + std::to_string(id));
            Dual& y = p.out(T * (rows + 3));
            p.run([&](Backend& b, Stream& s, int i) {
                b.gemv(s, GemvArgs{t, S(w, i).with_stride(stride), S(x, i).with_stride((cols + 5) * 4),
                                   S(y, i).with_stride((rows + 3) * 4), rows, cols, T, {i == 1 ? id : 0}});
            });
            EXPECT_TRUE(same_bits(y.fh(), y.fc(), "gemv (padding sentinels included)"));
        }
        const std::vector<std::int32_t> ids{0, 36, 5, 5, 17};
        Dual& idv = p.words(ids);
        Dual& out = p.out(ids.size() * cols);
        p.run([&](Backend& b, Stream& s, int i) {
            b.get_rows(s, GetRowsArgs{t, S(w, i).with_stride(stride), S(idv, i), S(out, i), rows, cols,
                                      static_cast<std::uint32_t>(ids.size()), {}, {}});
        });
        EXPECT_TRUE(same_bits(out.fh(), out.fc(), "get_rows"));
    }
}

/// A KV pool image with a fragmented block table, shared by the KV and attention tests.
struct Kv {
    std::uint32_t n_blocks = 7, n_layers = 2, layer = 1, bt = 5, n_kv = 2, hd = 64, n_head = 4;
    [[nodiscard]] std::uint32_t kvd() const { return n_kv * hd; }
    [[nodiscard]] std::size_t floats() const { return static_cast<std::size_t>(n_blocks) * n_layers * 2 * bt * kvd(); }
};

TEST_P(HipOps, KvWriteAndAttentionAreBitIdentical) {
    Pair p(emu(GetParam()));
    const Kv kv;
    Dual& pool = p.f32(rnd(kv.floats(), 50, -1, 1));
    const std::vector<std::uint32_t> table{4, 1, 6, 0, 2};
    Dual& tab = p.words(table);
    // History of 12 rows (3 blocks, the last partial), then a 4-row verify appended.
    std::uint32_t len = 0;
    for (const std::uint32_t T : {12u, 4u}) {
        Dual& k = p.f32(rnd(T * kv.kvd(), 51 + T, -1, 1));
        Dual& v = p.f32(rnd(T * kv.kvd(), 52 + T, -1, 1));
        p.run([&](Backend& b, Stream& s, int i) {
            b.kv_write(s, KvWriteArgs{S(pool, i), S(tab, i), S(k, i), S(v, i), kv.n_blocks, kv.n_layers, kv.layer, kv.bt,
                                      kv.kvd(), static_cast<std::uint32_t>(table.size()), len, T, {}, {}});
        });
        EXPECT_TRUE(same_bits(pool.fh(), pool.fc(), "whole pool after kv_write"));
        len += T;
    }
    // Attention of the last 4 rows (q_offset 12), q read in place from interleaved [Q | gate].
    const std::uint32_t T = 4;
    Dual& qg = p.f32(rnd(T * 2 * kv.n_head * kv.hd, 55, -2, 2));
    for (const std::uint32_t qhs : {0u, 2 * kv.hd}) {
        Dual& out = p.out(T * kv.n_head * kv.hd);
        p.run([&](Backend& b, Stream& s, int i) {
            AttentionArgs a{};
            a.q = qhs == 0 ? S(qg, i) : S(qg, i);
            a.q_head_stride = qhs;
            a.kv_pool = S(pool, i);
            a.block_table = S(tab, i);
            a.out = S(out, i);
            a.n_pool_blocks = kv.n_blocks;
            a.n_layers = kv.n_layers;
            a.layer = kv.layer;
            a.block_tokens = kv.bt;
            a.n_block_table = static_cast<std::uint32_t>(table.size());
            a.n_head = kv.n_head;
            a.n_kv_head = kv.n_kv;
            a.head_dim = kv.hd;
            a.n_tokens = T;
            a.q_offset = 12;
            a.scale = 0.125f;
            a.kernel = {i == 1 ? kExact : 0};
            b.attention(s, a);
        });
        EXPECT_TRUE(same_bits(out.fh(), out.fc(), qhs == 0 ? "attention exact, dense q" : "attention exact, strided q"));
    }
}

TEST_P(HipOps, HeadOpsAndCopyAreBitIdentical) {
    Pair p(emu(GetParam()));
    // LM head, generic GEMV: index and value bits, and the written logits.
    const std::uint32_t vocab = 1003, E = 256, n = 2;
    for (const DType t : {DType::F32, DType::Q6_K}) {
        SCOPED_TRACE(std::string(halo::traits(t).name));
        std::vector<std::byte> wb = make_weights(t, vocab, E, halo::tensor::row_bytes(t, E), 60);
        if (t == DType::F32) std::memcpy(&wb[999 * E * 4], &wb[17 * E * 4], E * 4);  // exact tie: rows 17 and 999
        Dual& w = p.bytes(wb, true);
        std::vector<float> xs = rnd(n * E, 61, -1, 1);
        Dual& x = p.f32(xs);
        Dual& logits = p.out(n * vocab);
        Dual& res = p.out(n * 3);
        p.run([&](Backend& b, Stream& s, int i) {
            b.lm_head(s, LmHeadArgs{GemvArgs{t, S(w, i), S(x, i), S(logits, i), vocab, E, n, {i == 1 ? kGeneric : 0}}, S(res, i), {}, {}});
        });
        EXPECT_TRUE(same_bits(logits.fh(), logits.fc(), "lm_head logits"));
        EXPECT_TRUE(same_bits(res.fh(), res.fc(), "lm_head result words"));
        // Argmax over those logits, and top-k (k = 40 and k = n).
        Dual& res2 = p.out(n * 3);
        p.run([&](Backend& b, Stream& s, int i) { b.argmax(s, ArgmaxArgs{S(logits, i), S(res2, i), vocab, n, {}, {}}); });
        EXPECT_TRUE(same_bits(res2.fh(), res2.fc(), "argmax result words"));
        for (const std::uint32_t k : {1u, 40u, vocab}) {
            if (k > 1024) continue;
            Dual& ids = p.out(n * k);
            Dual& vals = p.out(n * k);
            p.run([&](Backend& b, Stream& s, int i) { b.top_k(s, TopKArgs{S(logits, i), S(ids, i), S(vals, i), vocab, k, n, {}, {}}); });
            EXPECT_TRUE(same_bits(ids.fh(), ids.fc(), "top_k ids k=" + std::to_string(k)));
            EXPECT_TRUE(same_bits(vals.fh(), vals.fc(), "top_k values k=" + std::to_string(k)));
        }
    }
    // COPY with offsets.
    Dual& src = p.f32(rnd(100, 62));
    Dual& dst = p.out(100);
    p.run([&](Backend& b, Stream& s, int i) { b.copy(s, CopyArgs{S(src, i).shifted(12), S(dst, i).shifted(40), 200, {}}); });
    EXPECT_TRUE(same_bits(dst.fh(), dst.fc(), "copy"));
}

// ---------------------------------------------------------------------------------------
// (b) bounded, and non-vacuity of (a)
// ---------------------------------------------------------------------------------------

/// docs/hip.md's derived GEMV bound: |y_hip - y_cpu| <= (d_cpu + d_wave) u sum|x_i w_i| (1 + 1e-3),
/// d_cpu = K/8 + 3 + K mod 8 + 1, d_wave = ceil(K/32) + 5.
double gemv_bound(std::uint32_t K, double mag) {
    const double d_cpu = K / 8 + 3 + K % 8 + 1;
    const double d_wave = (K + 31) / 32 + 5;
    return (d_cpu + d_wave) * kU * mag * (1.0 + 1e-3) + 1e-37;
}

TEST_P(HipOps, WaveGemvIsWithinTheDerivedBoundAndDiffersFromCpu) {
    Pair p(emu(GetParam()));
    std::size_t total_diff = 0;
    for (const DType t : {DType::F32, DType::Q4_K, DType::Q6_K}) {
        const std::uint32_t rows = 37, cols = 5120, T = 3;
        const std::uint64_t rb = halo::tensor::row_bytes(t, cols);
        const std::vector<std::byte> wb = make_weights(t, rows, cols, rb, 70);
        Dual& w = p.bytes(wb, true);
        const std::vector<float> xs = rnd(T * cols, 71, -1, 1);
        Dual& x = p.f32(xs);
        std::vector<float> wd(static_cast<std::size_t>(rows) * cols);
        halo::tensor::dequantize_row(t, wb.data(), wd.data(), static_cast<std::int64_t>(wd.size()));
        for (const std::uint32_t id : {0u, kWaveR4, kWaveR8}) {
            SCOPED_TRACE(std::string(halo::traits(t).name) + " variant " + std::to_string(id));
            Dual& y = p.out(T * rows);
            p.run([&](Backend& b, Stream& s, int i) {
                b.gemv(s, GemvArgs{t, S(w, i), S(x, i), S(y, i), rows, cols, T, {i == 1 ? id : 0}});
            });
            total_diff += differing(y.fh(), y.fc());
            for (std::uint32_t v = 0; v < T; ++v) {
                for (std::uint32_t r = 0; r < rows; ++r) {
                    double mag = 0;
                    for (std::uint32_t k = 0; k < cols; ++k) mag += std::fabs(double(xs[v * cols + k]) * double(wd[std::size_t(r) * cols + k]));
                    const std::size_t e = static_cast<std::size_t>(v) * rows + r;
                    ASSERT_LE(std::fabs(double(y.fh()[e]) - double(y.fc()[e])), gemv_bound(cols, mag)) << "row " << r << " vec " << v;
                }
            }
        }
    }
    // The default profile really runs a different summation order (the (a) tests compare two
    // different computations, not a backend with itself).
    EXPECT_GT(total_diff, 0u);
}

TEST_P(HipOps, OnlineAttentionIsWithinTheDerivedBound) {
    Pair p(emu(GetParam()));
    // docs/hip.md: |o_hip - o_cpu| <= u (4N + 3 n_tiles + 4R + 16) sum_s w_s |v_s[d]|.
    Kv kv;
    kv.n_blocks = 210;  // 1001 keys: 8 tiles of 128, 16 of 64
    kv.bt = 5;
    const std::uint32_t N = 1001;
    std::vector<std::uint32_t> table(kv.n_blocks);
    for (std::uint32_t b = 0; b < kv.n_blocks; ++b) table[b] = (b * 37) % kv.n_blocks;
    std::vector<float> pool = rnd(kv.floats(), 80, -1, 1);
    Dual& pl = p.f32(pool);
    Dual& tab = p.words(table);
    const std::vector<float> q = rnd(kv.n_head * kv.hd, 81, -2, 2);
    Dual& qd = p.f32(q);
    const float scale = 0.125f;
    for (const std::uint32_t id : {0u, kOnline128, kOnline64}) {
        SCOPED_TRACE("variant " + std::to_string(id));
        const double tiles = (N + (id == kOnline64 ? 63 : 127)) / (id == kOnline64 ? 64 : 128);
        Dual& out = p.out(kv.n_head * kv.hd);
        p.run([&](Backend& b, Stream& s, int i) {
            AttentionArgs a{};
            a.q = S(qd, i);
            a.kv_pool = S(pl, i);
            a.block_table = S(tab, i);
            a.out = S(out, i);
            a.n_pool_blocks = kv.n_blocks;
            a.n_layers = kv.n_layers;
            a.layer = kv.layer;
            a.block_tokens = kv.bt;
            a.n_block_table = kv.n_blocks;
            a.n_head = kv.n_head;
            a.n_kv_head = kv.n_kv;
            a.head_dim = kv.hd;
            a.n_tokens = 1;
            a.q_offset = N - 1;
            a.scale = scale;
            a.kernel = {i == 1 ? id : 0};
            b.attention(s, a);
        });
        const std::size_t kvd = kv.kvd();
        for (std::uint32_t h = 0; h < kv.n_head; ++h) {
            const std::uint32_t kvh = h / (kv.n_head / kv.n_kv);
            std::vector<double> sc(N);
            double m = -1e300;
            for (std::uint32_t sidx = 0; sidx < N; ++sidx) {
                const std::size_t blk = table[sidx / kv.bt];
                const float* kr = &pool[((blk * kv.n_layers + kv.layer) * 2 + 0) * kv.bt * kvd + (sidx % kv.bt) * kvd + kvh * kv.hd];
                double d = 0;
                for (std::uint32_t e = 0; e < kv.hd; ++e) d += double(q[h * kv.hd + e]) * double(kr[e]);
                sc[sidx] = d * double(scale);
                m = std::max(m, sc[sidx]);
            }
            double Z = 0, R = 0;
            for (double v : sc) {
                Z += std::exp(v - m);
                R = std::max(R, m - v);
            }
            for (std::uint32_t e = 0; e < kv.hd; ++e) {
                double wv = 0;
                for (std::uint32_t sidx = 0; sidx < N; ++sidx) {
                    const std::size_t blk = table[sidx / kv.bt];
                    const float* vr = &pool[((blk * kv.n_layers + kv.layer) * 2 + 1) * kv.bt * kvd + (sidx % kv.bt) * kvd + kvh * kv.hd];
                    wv += std::exp(sc[sidx] - m) / Z * std::fabs(double(vr[e]));
                }
                const double bound = kU * (4.0 * N + 3.0 * tiles + 4.0 * R + 16.0) * wv;
                const std::size_t i = static_cast<std::size_t>(h) * kv.hd + e;
                ASSERT_LE(std::fabs(double(out.fh()[i]) - double(out.fc()[i])), bound) << "head " << h << " dim " << e;
            }
        }
    }
}

TEST_P(HipOps, WaveLmHeadIndexIsEqualWhenTheGapIsDecisive) {
    Pair p(emu(GetParam()));
    const std::uint32_t vocab = 1003, E = 512, n = 2;
    const DType t = DType::Q6_K;
    const std::vector<std::byte> wb = make_weights(t, vocab, E, halo::tensor::row_bytes(t, E), 90);
    Dual& w = p.bytes(wb, true);
    const std::vector<float> xs = rnd(n * E, 91, -1, 1);
    Dual& x = p.f32(xs);
    Dual& logits = p.out(n * vocab);
    Dual& res = p.out(n * 3);
    p.run([&](Backend& b, Stream& s, int i) {
        b.lm_head(s, LmHeadArgs{GemvArgs{t, S(w, i), S(x, i), S(logits, i), vocab, E, n, {}}, S(res, i), {}, {}});
    });
    std::vector<float> wd(static_cast<std::size_t>(vocab) * E);
    halo::tensor::dequantize_row(t, wb.data(), wd.data(), static_cast<std::int64_t>(wd.size()));
    for (std::uint32_t v = 0; v < n; ++v) {
        std::vector<float> row(logits.fc().begin() + v * vocab, logits.fc().begin() + (v + 1) * vocab);
        std::vector<float> sorted = row;
        std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<>());
        double worst = 0;
        for (std::uint32_t r = 0; r < vocab; ++r) {
            double mag = 0;
            for (std::uint32_t k = 0; k < E; ++k) mag += std::fabs(double(xs[v * E + k]) * double(wd[std::size_t(r) * E + k]));
            const double bound = gemv_bound(E, mag);
            worst = std::max(worst, bound);
            ASSERT_LE(std::fabs(double(logits.fh()[v * vocab + r]) - double(row[r])), bound);
        }
        ASSERT_GT(double(sorted[0]) - double(sorted[1]), 2 * worst) << "precondition: the top-2 gap decides the index";
        std::uint32_t wc[3], wh[3];
        std::memcpy(wc, res.fc().data() + 3 * v, 12);
        std::memcpy(wh, res.fh().data() + 3 * v, 12);
        EXPECT_EQ(wh[0], wc[0]) << "vector " << v;
        EXPECT_LE(std::fabs(double(std::bit_cast<float>(wh[1])) - double(std::bit_cast<float>(wc[1]))), worst);
    }
}

INSTANTIATE_TEST_SUITE_P(Emulation, HipOps, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& i) { return i.param ? "reverse" : "forward"; });

// ---------------------------------------------------------------------------------------
// Validation parity, limits, data errors
// ---------------------------------------------------------------------------------------

/// Runs the same (bad) call on both backends and expects the same ErrorCode.
void expect_parity(Pair& p, const std::function<void(Backend&, Stream&, int)>& op, ErrorCode want, const std::string& what) {
    std::string mc, mh;
    const auto cc = code_of([&] { op(p.cpu(), p.cs(), 0); }, &mc);
    const auto ch = code_of([&] { op(p.hip(), p.hs(), 1); }, &mh);
    EXPECT_EQ(cc, std::optional(want)) << what << " (cpu: " << mc << ")";
    EXPECT_EQ(ch, std::optional(want)) << what << " (hip: " << mh << ")";
}

TEST(HipBackendValidation, RejectsExactlyWhatTheCpuBackendRejects) {
    Pair p;
    const std::uint32_t R = 4, E = 32;
    Dual& x = p.f32(rnd(R * E, 1));
    Dual& y = p.f32(rnd(R * E, 2));
    Dual& w = p.f32(rnd(E, 3, 0.5f, 1.5f));
    Dual& o = p.out(R * E);
    Dual& ro = p.bytes(std::vector<std::byte>(R * E * 4), true);
    // A buffer of the other backend.
    std::vector<float> foreign(R * E);
    auto fc = p.cpu().import_host(std::as_writable_bytes(std::span(foreign)));
    auto fh = p.hip().import_host(std::as_writable_bytes(std::span(foreign)));
    const TensorRef foreign_of[2] = {TensorRef::of(*fh), TensorRef::of(*fc)};

    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{foreign_of[i], S(w, i), S(o, i), R, E, 1e-6f, {}}); },
                  ErrorCode::Kernel, "operand of another backend");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{S(x, i), S(w, i), S(ro, i), R, E, 1e-6f, {}}); },
                  ErrorCode::Kernel, "read-only output");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{S(x, i), S(w, i), S(x, i), R, E, 1e-6f, {}}); },
                  ErrorCode::Kernel, "rms_norm out == x (HIP kernels allow it; the neutral rule does not)");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.swiglu(s, EltwiseArgs{S(x, i), S(y, i), S(x, i), R, E, {}}); },
                  ErrorCode::Kernel, "swiglu out == a");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.mul_sigmoid(s, EltwiseArgs{S(x, i), S(y, i), S(y, i), R, E, {}}); },
                  ErrorCode::Kernel, "mul_sigmoid out == b");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.add(s, EltwiseArgs{S(x, i), S(y, i), S(y, i), R, E, {}}); },
                  ErrorCode::Kernel, "add out == b (only out == a is allowed)");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.add(s, EltwiseArgs{S(x, i), S(y, i), S(x, i).shifted(4), R - 1, E, {}}); },
                  ErrorCode::Kernel, "add out partially overlapping a");
    expect_parity(p, [&](Backend& b, Stream& s, int i) {
        b.gated_rms_norm(s, GatedNormArgs{S(x, i), S(y, i), S(w, i), S(y, i), R, E, 1e-6f, {}});
    }, ErrorCode::Kernel, "gated_norm out == z");
    expect_parity(p, [&](Backend& b, Stream& s, int i) {
        b.add_rms_norm(s, AddRmsNormArgs{S(x, i), S(y, i), S(y, i), S(w, i), S(o, i), R, E, 1e-6f, {}});
    }, ErrorCode::Kernel, "add_rms_norm h == b");
    expect_parity(p, [&](Backend& b, Stream& s, int i) {
        b.gdn_gates(s, GdnGateArgs{S(x, i), S(y, i), S(w, i), S(w, i), S(x, i), S(o, i), 1, E, {}});
    }, ErrorCode::Kernel, "gdn_gates g_out == alpha (HIP allows in place)");
    expect_parity(p, [&](Backend& b, Stream& s, int i) {
        b.conv1d_silu(s, Conv1dArgs{S(x, i), S(y, i), S(w, i), S(x, i), {}, 1, E, 4, 0, {}});
    }, ErrorCode::Kernel, "conv out == x (HIP allows it)");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{S(x, i).shifted(2), S(w, i), S(o, i), 1, E, 1e-6f, {}}); },
                  ErrorCode::Kernel, "misaligned x");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{S(x, i), S(w, i), S(o, i), R + 1, E, 1e-6f, {}}); },
                  ErrorCode::Kernel, "view too small");
    expect_parity(p, [&](Backend& b, Stream& s, int i) {
        b.rms_norm(s, RmsNormArgs{S(x, i).with_stride(E * 2), S(w, i), S(o, i), 2, E, 1e-6f, {}});
    }, ErrorCode::Kernel, "row stride < row bytes");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{{}, S(w, i), S(o, i), R, E, 1e-6f, {}}); },
                  ErrorCode::Kernel, "empty operand");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{S(x, i), S(w, i), S(o, i), R, E, 1e-6f, {999}}); },
                  ErrorCode::Config, "unknown variant id");
    expect_parity(p, [&](Backend& b, Stream& s, int i) { b.rms_norm(s, RmsNormArgs{S(x, i), S(w, i), S(o, i), R, E, 1e-6f, {i == 1 ? kGeneric : 1u}}); },
                  ErrorCode::Config, "a variant id of another op");
    expect_parity(p, [&](Backend& b, Stream& s, int i) {
        b.get_rows(s, GetRowsArgs{DType::F32, S(x, i), S(y, i), S(o, i), R, E, 1, {}, StatusRef{0, 0}});
    }, ErrorCode::Unsupported, "a status word (WS-BI-2)");
    // A stream of the other backend.
    auto other = make_hip_backend();
    auto os = other->create_stream();
    auto cs2 = p.cpu().create_stream();
    const auto ch = code_of([&] { p.hip().rms_norm(*os, RmsNormArgs{x.rh, w.rh, o.rh, R, E, 1e-6f, {}}); });
    const auto ch2 = code_of([&] { p.hip().rms_norm(*cs2, RmsNormArgs{x.rh, w.rh, o.rh, R, E, 1e-6f, {}}); });
    EXPECT_EQ(ch, std::optional(ErrorCode::Api));
    EXPECT_EQ(ch2, std::optional(ErrorCode::Api));
    // run_op dispatches to the typed method (and its validation).
    EXPECT_EQ(code_of([&] { p.hip().run_op(p.hs(), OpInvocation{OpId::RmsNorm, RmsNormArgs{x.rh, w.rh, x.rh, R, E, 1e-6f, {}}}); }),
              std::optional(ErrorCode::Kernel));
}

TEST(HipBackendValidation, LimitsRaiseUnsupportedWhereTheCpuBackendAccepts) {
    Pair p;
    const Limits l = p.hip().limits();
    EXPECT_EQ(l.max_gdn_dk, 128u);
    EXPECT_EQ(l.max_gdn_chunk, 64u);
    EXPECT_EQ(l.max_head_dim, 256u);
    EXPECT_EQ(l.max_conv_k, 8u);
    EXPECT_EQ(l.max_top_k, 1024u);
    EXPECT_EQ(l.max_rope_dims, 128u);
    EXPECT_TRUE(l.gdn_chunked);
    auto both = [&](const std::function<void(Backend&, Stream&, int)>& op, const char* what) {
        std::string mc, mh;
        EXPECT_EQ(code_of([&] { op(p.cpu(), p.cs(), 0); }, &mc), std::nullopt) << what << ": cpu " << mc;
        EXPECT_EQ(code_of([&] { op(p.hip(), p.hs(), 1); }, &mh), std::optional(ErrorCode::Unsupported)) << what << ": hip " << mh;
    };
    // GDN d_k 129, chunk 65.
    {
        const std::uint32_t dk = 129, dv = 4, T = 2;
        Dual& q = p.f32(rnd(T * dk, 1));
        Dual& v = p.f32(rnd(T * dv, 2));
        Dual& g = p.f32(rnd(T, 3, -1, -0.1f));
        Dual& be = p.f32(rnd(T, 4, 0.1f, 0.9f));
        Dual& st = p.f32(rnd(dk * dv, 5));
        Dual& o = p.out(T * dv);
        both([&](Backend& b, Stream& s, int i) {
            GdnArgs a{};
            a.q = S(q, i);
            a.k = S(q, i);
            a.v = S(v, i);
            a.g = S(g, i);
            a.beta = S(be, i);
            a.state = S(st, i);
            a.out = S(o, i);
            a.n_k = a.n_v = 1;
            a.d_k = dk;
            a.d_v = dv;
            a.n_tokens = T;
            b.gated_delta_rule(s, a);
        }, "GDN d_k 129");
        Dual& q2 = p.f32(rnd(T * 8, 6));
        Dual& st2 = p.f32(rnd(8 * dv, 7));
        both([&](Backend& b, Stream& s, int i) {
            GdnArgs a{};
            a.form = GdnForm::Chunked;
            a.q = S(q2, i);
            a.k = S(q2, i);
            a.v = S(v, i);
            a.g = S(g, i);
            a.beta = S(be, i);
            a.state = S(st2, i);
            a.out = S(o, i);
            a.n_k = a.n_v = 1;
            a.d_k = 8;
            a.d_v = dv;
            a.n_tokens = T;
            a.chunk_size = 65;
            b.gated_delta_rule(s, a);
        }, "GDN chunk 65");
    }
    // Conv kernel 9.
    {
        const std::uint32_t C = 8, K = 9;
        Dual& x = p.f32(rnd(C, 1));
        Dual& w = p.f32(rnd(C * K, 2));
        Dual& st = p.f32(rnd((K - 1) * C, 3));
        Dual& o = p.out(C);
        both([&](Backend& b, Stream& s, int i) { b.conv1d_silu(s, Conv1dArgs{S(x, i), S(w, i), S(st, i), S(o, i), {}, 1, C, K, 0, {}}); },
             "conv kernel 9");
    }
    // Attention head_dim 264, rope rot 130, top-k 1025.
    {
        const Kv kv{1, 1, 0, 4, 1, 264, 1};
        Dual& pool = p.f32(rnd(kv.floats(), 1));
        Dual& tab = p.words(std::vector<std::uint32_t>{0});
        Dual& q = p.f32(rnd(kv.hd, 2));
        Dual& o = p.out(kv.hd);
        both([&](Backend& b, Stream& s, int i) {
            AttentionArgs a{};
            a.q = S(q, i);
            a.kv_pool = S(pool, i);
            a.block_table = S(tab, i);
            a.out = S(o, i);
            a.n_pool_blocks = 1;
            a.n_layers = 1;
            a.block_tokens = kv.bt;
            a.n_block_table = 1;
            a.n_head = a.n_kv_head = 1;
            a.head_dim = kv.hd;
            b.attention(s, a);
        }, "attention head_dim 264");
        Dual& x = p.f32(rnd(264, 3));
        Dual& pos = p.words(std::vector<std::int32_t>{3});
        both([&](Backend& b, Stream& s, int i) { b.partial_rope(s, RopeArgs{S(x, i), S(pos, i), 1, 1, 264, 130, 1e4f, 0, {}}); },
             "rope rot 130");
        Dual& lg = p.f32(rnd(2000, 4));
        Dual& ids = p.out(1025);
        Dual& vals = p.out(1025);
        both([&](Backend& b, Stream& s, int i) { b.top_k(s, TopKArgs{S(lg, i), S(ids, i), S(vals, i), 2000, 1025, 1, {}, {}}); },
             "top_k 1025");
    }
}

TEST(HipBackendValidation, DataErrorsRaiseKernelAtTheOpInEmulation) {
    Pair p;
    auto both = [&](const std::function<void(Backend&, Stream&, int)>& op, const char* what, const char* hip_contains) {
        std::string mc, mh;
        EXPECT_EQ(code_of([&] { op(p.cpu(), p.cs(), 0); }, &mc), std::optional(ErrorCode::Kernel)) << what << ": cpu " << mc;
        EXPECT_EQ(code_of([&] { op(p.hip(), p.hs(), 1); }, &mh), std::optional(ErrorCode::Kernel)) << what << ": hip " << mh;
        EXPECT_NE(mh.find(hip_contains), std::string::npos) << what << ": " << mh;
        // The stream is usable afterwards (the failed op's checks were consumed).
        EXPECT_NO_THROW(p.hs().wait()) << what;
    };
    const std::uint32_t E = 32;
    Dual& table = p.f32(rnd(4 * E, 1));
    Dual& bad_ids = p.words(std::vector<std::int32_t>{1, 4});
    Dual& out = p.out(2 * E);
    both([&](Backend& b, Stream& s, int i) { b.get_rows(s, GetRowsArgs{DType::F32, S(table, i), S(bad_ids, i), S(out, i), 4, E, 2, {}, {}}); },
         "get_rows id 4 of 4", "GET_ROWS");
    const Kv kv{3, 1, 0, 4, 1, 16, 1};
    Dual& pool = p.f32(rnd(kv.floats(), 2));
    Dual& tab = p.words(std::vector<std::uint32_t>{0, 3});  // block 3 is outside a pool of 3
    Dual& k = p.f32(rnd(2 * kv.kvd(), 3));
    both([&](Backend& b, Stream& s, int i) {
        b.kv_write(s, KvWriteArgs{S(pool, i), S(tab, i), S(k, i), S(k, i), kv.n_blocks, 1, 0, kv.bt, kv.kvd(), 2, 3, 2, {}, {}});
    }, "kv_write into block 3 of 3", "KV_WRITE");
    Dual& q = p.f32(rnd(kv.hd, 4));
    Dual& o = p.out(kv.hd);
    both([&](Backend& b, Stream& s, int i) {
        AttentionArgs a{};
        a.q = S(q, i);
        a.kv_pool = S(pool, i);
        a.block_table = S(tab, i);
        a.out = S(o, i);
        a.n_pool_blocks = kv.n_blocks;
        a.n_layers = 1;
        a.block_tokens = kv.bt;
        a.n_block_table = 2;
        a.n_head = a.n_kv_head = 1;
        a.head_dim = kv.hd;
        a.q_offset = 5;
        b.attention(s, a);
    }, "attention over block 3 of 3", "ATTENTION");
    // Positive g in chunked GDN.
    const std::uint32_t T = 3, dk = 8, dv = 4;
    Dual& qq = p.f32(rnd(T * dk, 5));
    Dual& v = p.f32(rnd(T * dv, 6));
    std::vector<float> gv = rnd(T, 7, -1, -0.1f);
    gv[1] = 0.5f;
    Dual& g = p.f32(gv);
    Dual& be = p.f32(rnd(T, 8, 0.1f, 0.9f));
    Dual& st = p.f32(rnd(dk * dv, 9));
    Dual& go = p.out(T * dv);
    both([&](Backend& b, Stream& s, int i) {
        GdnArgs a{};
        a.form = GdnForm::Chunked;
        a.q = S(qq, i);
        a.k = S(qq, i);
        a.v = S(v, i);
        a.g = S(g, i);
        a.beta = S(be, i);
        a.state = S(st, i);
        a.out = S(go, i);
        a.n_k = a.n_v = 1;
        a.d_k = dk;
        a.d_v = dv;
        a.n_tokens = T;
        a.chunk_size = 2;
        b.gated_delta_rule(s, a);
    }, "chunked GDN g > 0", "GATED_DELTANET");
    EXPECT_TRUE(same_bits(st.fh(), st.fc(), "state untouched by the failed chunked GDN (both sides)"));
    // NaN logits: the LM head poisons only its own row (H1) -- both backends report
    // {-1, NaN} in the result words instead of throwing for the batched call. Standalone
    // argmax and top-k still raise Error(Kernel), as on the CPU backend.
    std::vector<float> wv = rnd(20 * E, 10);
    wv[7 * E + 3] = std::numeric_limits<float>::quiet_NaN();
    Dual& w = p.f32(wv);
    Dual& x = p.f32(rnd(E, 11));
    Dual& res = p.out(3);
    p.run([&](Backend& b, Stream& s, int i) {
        b.lm_head(s, LmHeadArgs{GemvArgs{DType::F32, S(w, i), S(x, i), {}, 20, E, 1, {}}, S(res, i), {}, {}});
    });
    for (const std::span<const std::byte>& side : {std::as_bytes(res.fc()), std::as_bytes(res.fh())}) {
        const std::vector<ArgmaxResult> best = decode_argmax(side);
        ASSERT_EQ(best.size(), 1);
        EXPECT_EQ(best[0].index, -1);
        EXPECT_TRUE(std::isnan(best[0].value));
    }
    std::vector<float> lv = rnd(3000, 12);
    lv[2500] = std::numeric_limits<float>::quiet_NaN();
    Dual& lg = p.f32(lv);
    both([&](Backend& b, Stream& s, int i) { b.argmax(s, ArgmaxArgs{S(lg, i), S(res, i), 3000, 1, {}, {}}); }, "argmax NaN", "ARGMAX");
    Dual& ids = p.out(40);
    Dual& vals = p.out(40);
    both([&](Backend& b, Stream& s, int i) { b.top_k(s, TopKArgs{S(lg, i), S(ids, i), S(vals, i), 3000, 40, 1, {}, {}}); }, "top_k NaN",
         "TOP_K");
}

// ---------------------------------------------------------------------------------------
// Registry, modes, memory
// ---------------------------------------------------------------------------------------

TEST(HipBackendRegistry, IdsArePinnedAndCoverTheHipRegistry) {
    const auto be = make_hip_backend();
    struct Pin {
        std::uint32_t id;
        OpId op;
        const char* name;
        const char* form;
    };
    // Append-only: a new variant gets the next id; a retired id is never reused.
    const Pin pins[] = {
        {1, OpId::GetRows, "get_rows_b256", ""},          {2, OpId::RmsNorm, "rms_norm_b128", ""},
        {3, OpId::RmsNorm, "rms_norm_b32", ""},           {4, OpId::AddRmsNorm, "add_rms_norm_b128", ""},
        {5, OpId::Gemv, "gemv_wave32_r4", ""},            {6, OpId::Gemv, "gemv_wave32_r8", ""},
        {7, OpId::Gemv, "gemv_generic_b64", ""},          {8, OpId::Gemv, "gemm_t16x16_b256", ""},
        {9, OpId::GdnGates, "gdn_gate_b64", ""},          {10, OpId::Conv1dSilu, "conv1d_silu_b256", ""},
        {11, OpId::Conv1dSilu, "conv1d_silu_b64", ""},    {12, OpId::GatedDeltaRule, "gdn_recurrent_b128", "recurrent"},
        {13, OpId::GatedDeltaRule, "gdn_recurrent_b64", "recurrent"}, {14, OpId::GatedDeltaRule, "gdn_chunked_b64", "chunked"},
        {15, OpId::GatedDeltaRule, "gdn_chunked_b32", "chunked"},     {16, OpId::GatedRmsNorm, "gated_norm_b128", ""},
        {17, OpId::GatedRmsNorm, "gated_norm_b32", ""},   {18, OpId::PartialRope, "rope_neox_b128", ""},
        {19, OpId::KvWrite, "kv_write_b256", ""},         {20, OpId::Attention, "attn_online_b128", ""},
        {21, OpId::Attention, "attn_online_b64", ""},     {22, OpId::Attention, "attn_exact_b128", ""},
        {23, OpId::Swiglu, "swiglu_b256", ""},            {24, OpId::MulSigmoid, "mul_sigmoid_b256", ""},
        {25, OpId::Add, "add_b256", ""},                  {26, OpId::LmHead, "argmax_b256", ""},
        {27, OpId::Argmax, "argmax_b256", ""},            {28, OpId::TopK, "topk_bitonic_b256", ""},
        {29, OpId::Copy, "copy_async", ""},
    };
    std::size_t listed = 0;
    std::set<std::uint32_t> ids;
    for (std::size_t op = 0; op < kOpCount; ++op) {
        const auto vs = be->variants(static_cast<OpId>(op));
        EXPECT_FALSE(vs.empty()) << op_name(static_cast<OpId>(op));
        for (const VariantInfo& v : vs) {
            EXPECT_EQ(v.op, static_cast<OpId>(op));
            EXPECT_TRUE(ids.insert(v.id).second) << "duplicate id " << v.id;
            EXPECT_NE(v.id, 0u) << "0 is reserved for the default";
            ++listed;
        }
    }
    EXPECT_EQ(listed, std::size(pins));
    for (const Pin& pin : pins) {
        const auto vs = be->variants(pin.op);
        const auto it = std::find_if(vs.begin(), vs.end(), [&](const VariantInfo& v) { return v.id == pin.id; });
        ASSERT_NE(it, vs.end()) << "id " << pin.id;
        EXPECT_EQ(it->name, pin.name) << "id " << pin.id;
        EXPECT_EQ(it->form, pin.form) << "id " << pin.id;
    }
    // Forms filter.
    EXPECT_EQ(be->variants(OpId::GatedDeltaRule, "recurrent").size(), 2u);
    EXPECT_EQ(be->variants(OpId::GatedDeltaRule, "chunked").size(), 2u);
    EXPECT_EQ(be->variants(OpId::GatedDeltaRule, "nope").size(), 0u);
    // Every variant of the HIP kernel registry is reachable through some id.
    for (const halo::hip::KernelVariant& kv : halo::hip::kernel_variants()) {
        bool found = false;
        for (const Pin& pin : pins) found = found || kv.name == pin.name;
        EXPECT_TRUE(found) << "HIP variant " << kv.name << " has no backend id";
    }
}

TEST(HipBackendRegistry, AVariantIdSelectsWhatItNamesAndAWrongFormIsConfig) {
    Pair p(emu(false));
    // Recurrent ids on a chunked call (and the reverse) are Error(Config).
    const std::uint32_t T = 3, dk = 8, dv = 4;
    Dual& q = p.f32(rnd(T * dk, 1));
    Dual& v = p.f32(rnd(T * dv, 2));
    Dual& g = p.f32(rnd(T, 3, -1, -0.1f));
    Dual& be = p.f32(rnd(T, 4, 0.1f, 0.9f));
    Dual& st = p.f32(rnd(dk * dv, 5));
    Dual& o = p.out(T * dv);
    auto gdn = [&](GdnForm f, std::uint32_t id) {
        GdnArgs a{};
        a.form = f;
        a.q = q.rh;
        a.k = q.rh;
        a.v = v.rh;
        a.g = g.rh;
        a.beta = be.rh;
        a.state = st.rh;
        a.out = o.rh;
        a.n_k = a.n_v = 1;
        a.d_k = dk;
        a.d_v = dv;
        a.n_tokens = T;
        a.chunk_size = 2;
        a.kernel = {id};
        p.hip().gated_delta_rule(p.hs(), a);
    };
    EXPECT_EQ(code_of([&] { gdn(GdnForm::Chunked, kRec128); }), std::optional(ErrorCode::Config));
    EXPECT_EQ(code_of([&] { gdn(GdnForm::Recurrent, kChunk64); }), std::optional(ErrorCode::Config));
    for (const std::uint32_t id : {kRec128, kRec64}) EXPECT_EQ(code_of([&] { gdn(GdnForm::Recurrent, id); }), std::nullopt);
    for (const std::uint32_t id : {kChunk64, kChunk32}) EXPECT_EQ(code_of([&] { gdn(GdnForm::Chunked, id); }), std::nullopt);
    // The GEMM tile is not an LM-head variant; an LM-head id on gemv.kernel is not either.
    Dual& w = p.f32(rnd(16 * 32, 6));
    Dual& x = p.f32(rnd(32, 7));
    Dual& res = p.out(3);
    EXPECT_EQ(code_of([&] { p.hip().lm_head(p.hs(), LmHeadArgs{GemvArgs{DType::F32, w.rh, x.rh, {}, 16, 32, 1, {kGemm}}, res.rh, {}, {}}); }),
              std::optional(ErrorCode::Config));
    EXPECT_EQ(code_of([&] { p.hip().lm_head(p.hs(), LmHeadArgs{GemvArgs{DType::F32, w.rh, x.rh, {}, 16, 32, 1, {26}}, res.rh, {}, {}}); }),
              std::optional(ErrorCode::Config));
    EXPECT_EQ(code_of([&] { p.hip().lm_head(p.hs(), LmHeadArgs{GemvArgs{DType::F32, w.rh, x.rh, {}, 16, 32, 1, {kWaveR8}}, res.rh, {26}, {}}); }),
              std::nullopt);
    // Selection is real: over K = 5120 the generic id reproduces the CPU bits while the wave
    // ids do not (checked in WaveGemvIsWithinTheDerivedBoundAndDiffersFromCpu); here the two
    // wave ids must at least run and the generic id must equal the bitwise profile's default.
    const std::uint32_t rows = 13, cols = 5120;
    Dual& w2 = p.f32(rnd(rows * cols, 8));
    Dual& x2 = p.f32(rnd(cols, 9));
    Dual& y_id = p.out(rows);
    p.hip().gemv(p.hs(), GemvArgs{DType::F32, w2.rh, x2.rh, y_id.rh, rows, cols, 1, {kGeneric}});
    Pair pb(emu(false, true));
    Dual& w3 = pb.f32(rnd(rows * cols, 8));
    Dual& x3 = pb.f32(rnd(cols, 9));
    Dual& y_def = pb.out(rows);
    pb.run([&](Backend& b, Stream& s, int i) { b.gemv(s, GemvArgs{DType::F32, S(w3, i), S(x3, i), S(y_def, i), rows, cols, 1, {}}); });
    EXPECT_TRUE(same_bits(y_id.fh(), y_def.fh(), "id 7 == bitwise-profile default"));
    EXPECT_TRUE(same_bits(y_def.fh(), y_def.fc(), "bitwise-profile default == cpu"));
}

TEST(HipBackendModes, KindDescribeAndDeviceModeWithoutADevice) {
    const auto e = make_hip_backend();
    EXPECT_EQ(e->kind(), Kind::HipEmulation);
    EXPECT_EQ(to_string(e->kind()), "hip-emulation");
    EXPECT_NE(e->describe().find("hip-emulation"), std::string::npos) << e->describe();
    EXPECT_NE(e->describe().find("gemv_wave32_r4"), std::string::npos) << e->describe();
    const auto b = make_hip_backend(emu(true, true));
    EXPECT_NE(b->describe().find("reverse"), std::string::npos) << b->describe();
    EXPECT_NE(b->describe().find("gemv_generic_b64"), std::string::npos) << b->describe();
    HipBackendOptions bad;
    bad.defaults.gemv = "no_such_variant";
    EXPECT_EQ(code_of([&] { static_cast<void>(make_hip_backend(bad)); }), std::optional(ErrorCode::Config));
    HipBackendOptions dev;
    dev.mode = HipMode::Device;
    if (halo::hip::probe().available()) GTEST_SKIP() << "a HIP device is present; the no-device path cannot be exercised";
    EXPECT_EQ(code_of([&] { static_cast<void>(make_hip_backend(dev)); }), std::optional(ErrorCode::Device));
}

TEST(HipBackendModes, DeviceModeRunsTheSameAdapterOnADevice) {
    const halo::hip::Availability av = halo::hip::probe();
    if (!av.available()) GTEST_SKIP() << "no HIP device (dev host, D-001): " << av.reason;
    HipBackendOptions o;
    o.mode = HipMode::Device;
    o.defaults = hip_bitwise_defaults();
    const auto be = make_hip_backend(o);
    EXPECT_EQ(be->kind(), Kind::Hip);
    auto s = be->create_stream();
    const std::vector<float> x = rnd(64, 1), w = rnd(64 * 8, 2);
    auto bx = be->allocate(64 * 4, Tier::Vram), bw = be->allocate(64 * 8 * 4, Tier::Vram), by = be->allocate(8 * 4, Tier::Gtt);
    EXPECT_EQ(bx->host_data(), nullptr);
    be->upload(*s, TensorRef::of(*bx), std::as_bytes(std::span(x)));
    be->upload(*s, TensorRef::of(*bw), std::as_bytes(std::span(w)));
    be->gemv(*s, GemvArgs{DType::F32, TensorRef::of(*bw), TensorRef::of(*bx), TensorRef::of(*by), 8, 64, 1, {}});
    std::vector<float> y(8);
    be->download(*s, TensorRef::of(*by), std::as_writable_bytes(std::span(y)));
    s->submit();
    s->wait();
    Pair p(emu(false, true));
    Dual& dx = p.f32(x);
    Dual& dw = p.f32(w);
    Dual& dy = p.out(8);
    p.run([&](Backend& b, Stream& st, int i) { b.gemv(st, GemvArgs{DType::F32, S(dw, i), S(dx, i), S(dy, i), 8, 64, 1, {}}); });
    EXPECT_TRUE(same_bits(y, dy.fc(), "device generic gemv == cpu (IEEE fp32 mul/add only)"));
    EXPECT_EQ(code_of([&] { static_cast<void>(be->import_host(std::as_writable_bytes(std::span(y)))); }),
              std::optional(ErrorCode::Unsupported));
    // Read-only import = a copy into VRAM (D-017 copy at load): usable as an input, never an output.
    auto wro = be->import_host_readonly(std::as_bytes(std::span(w)));
    EXPECT_FALSE(wro->writable());
    EXPECT_EQ(wro->host_data(), nullptr);
    be->gemv(*s, GemvArgs{DType::F32, TensorRef::of(*wro), TensorRef::of(*bx), TensorRef::of(*by), 8, 64, 1, {}});
    std::vector<float> y2(8);
    be->download(*s, TensorRef::of(*by), std::as_writable_bytes(std::span(y2)));
    s->wait();
    EXPECT_TRUE(same_bits(y2, y, "gemv over the read-only VRAM copy"));
    EXPECT_EQ(code_of([&] { be->gemv(*s, GemvArgs{DType::F32, TensorRef::of(*bw), TensorRef::of(*bx), TensorRef::of(*wro), 8, 64, 1, {}}); }),
              std::optional(ErrorCode::Kernel));
}

TEST(HipBackendMemory, AllocationsAreHostAddressableAndPoisonedInEmulation) {
    const auto be = make_hip_backend();
    auto b = be->allocate(64, Tier::Vram);
    ASSERT_NE(b->host_data(), nullptr);
    EXPECT_EQ(b->bytes(), 64u);
    EXPECT_TRUE(b->writable());
    EXPECT_EQ(b->backend(), be.get());
    for (std::size_t i = 0; i < 64; ++i) ASSERT_EQ(b->host_data()[i], std::byte{0xFF}) << i;
    auto z = be->allocate(0, Tier::Host);
    EXPECT_EQ(z->bytes(), 0u);
    HipBackendOptions nopoison;
    nopoison.emulation_poison_alloc = false;
    const auto be2 = make_hip_backend(nopoison);
    auto c = be2->allocate(16, Tier::Gtt);
    for (std::size_t i = 0; i < 16; ++i) ASSERT_EQ(c->host_data()[i], std::byte{0}) << i;
    // Read-only import: host-readable, not host-writable, rejected as an output.
    std::vector<std::byte> ro(32);
    auto r = be->import_host_readonly(ro);
    EXPECT_FALSE(r->writable());
    EXPECT_NE(r->host_data(), nullptr);
    EXPECT_EQ(r->host_ptr(), nullptr);
    // Upload / download round trip, ordered in the stream.
    auto s = be->create_stream();
    const std::vector<float> v = rnd(16, 3);
    be->upload(*s, TensorRef::of(*b).shifted(0), std::as_bytes(std::span(v)));
    std::vector<float> back(16);
    be->download(*s, TensorRef::of(*b), std::as_writable_bytes(std::span(back)));
    s->submit();
    s->wait();
    EXPECT_TRUE(same_bits(back, v, "upload/download"));
    EXPECT_EQ(code_of([&] { be->upload(*s, TensorRef::of(*r), std::as_bytes(std::span(v)).first(16)); }), std::optional(ErrorCode::Kernel));
    EXPECT_EQ(code_of([&] { be->upload(*s, TensorRef::of(*b), std::as_bytes(std::span(back)).first(16)); }), std::nullopt);
    std::vector<float> big(17);
    EXPECT_EQ(code_of([&] { be->upload(*s, TensorRef::of(*b), std::as_bytes(std::span(big))); }), std::optional(ErrorCode::Kernel));
}

}  // namespace
