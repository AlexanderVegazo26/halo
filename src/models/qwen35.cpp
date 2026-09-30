// Qwen3.8 ("qwen35") forward (D-004) and MTP block (D-005), composed over the backend
// interface (ADR-001 §4): every op is a Backend call on TensorRefs, so this is the one model
// definition for every backend. On the CPU backend each call is the halo::cpu function the
// pre-backend forward called directly, on the same values (tested bitwise, WS-BI-1 gate).
//
// Differences from the pre-backend forward, none of which changes a value:
//  - every op writes a distinct output (ADR §5.2 neutral aliasing rule), except the residual
//    accumulate x += y (ADD / ADD+RMS_NORM) and in-place RoPE;
//  - the [Q | gate] de-interleave copy is gone: the per-head q RMSNorm reads Q with a head
//    stride of 2 * head_dim and writes a dense q, and MUL_SIGMOID reads the gate halves with
//    the same stride (review S-2);
//  - ADD + RMS_NORM pairs run as the fused interface op (= cpu::add then cpu::rms_norm).
// The cost model (D-011) counts the same traffic as before, minus the removed copy.

#include "halo/models/qwen35.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "halo/backend/backend.h"
#include "halo/backends/cpu/traffic.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"

namespace halo::models {

using backend::Backend;
using backend::Buffer;
using backend::TensorRef;
using model::LayerKind;
using model::WeightRef;

namespace {

constexpr std::uint64_t kF32 = 4;

std::uint32_t u32(std::size_t v) {
    HALO_CHECK(v <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Unsupported, "qwen35: size {} exceeds 32 bits", v);
    return static_cast<std::uint32_t>(v);
}

/// A weight matrix imported (read-only) into the backend: rows x cols of `type`.
struct Mat {
    DType type = DType::F32;
    std::unique_ptr<Buffer> buf;
    std::uint32_t rows = 0, cols = 0;
    std::uint64_t bytes = 0;  // stored bytes (cost model)
    [[nodiscard]] TensorRef ref() const { return TensorRef::of(*buf); }
};

/// A small fp32 vector (norm weights, dt_bias, ssm_a, conv taps), dequantized once.
struct Vec {
    std::unique_ptr<Buffer> buf;
    std::size_t n = 0;
    [[nodiscard]] TensorRef ref() const { return TensorRef::of(*buf); }
};

struct AttnW {
    Mat q, k, v, o;
    Vec q_norm, k_norm;
};

struct GdnW {
    Mat qkv, gate, beta, alpha, out;
    Vec conv;  // [C, K]
    Vec dt_bias, a, norm;
};

struct FfnW {
    Mat gate, up, down;
};

struct LayerW {
    LayerKind kind = LayerKind::GatedDeltaNet;
    std::size_t kind_index = 0;  // index among GDN layers (state) or attention layers (KV)
    Vec attn_norm, post_norm;
    std::optional<AttnW> attn;
    std::optional<GdnW> gdn;
    std::optional<FfnW> ffn;
    std::uint64_t bytes = 0;  // stored bytes of every weight of the layer
};

/// A token-row table (embedding / LM head) imported read-only.
struct Table {
    WeightRef ref;
    std::unique_ptr<Buffer> buf;
};

struct MtpW {
    LayerW block;
    Mat eh_proj;
    Vec enorm, hnorm, head_norm;
    Table embedding;
    Table lm_head;
    std::uint64_t bytes = 0;  // block + eh_proj + norms (not embedding / head)
};

/// Rows of one sequence inside the concatenated batch.
struct SeqRows {
    std::size_t r0 = 0;
    std::size_t n = 0;
    kv_cache::SequenceKv* kv = nullptr;
    std::int32_t pos0 = 0;
    TensorRef pool{};   // the KV pool storage image
    TensorRef table{};  // the block table (uint32)
};

std::uint64_t vec_bytes(const WeightRef& w) { return w.present() ? w.n_bytes() : 0; }

}  // namespace

StepCost& StepCost::operator+=(const StepCost& o) noexcept {
    weight_bytes += o.weight_bytes;
    weight_passes += o.weight_passes;
    embedding_bytes += o.embedding_bytes;
    activation_bytes += o.activation_bytes;
    state_bytes += o.state_bytes;
    kv_bytes += o.kv_bytes;
    return *this;
}

struct Qwen35::Impl {
    std::unique_ptr<Backend> owned;  // the internal CPU backend of the pool constructors
    Backend* be = nullptr;
    backend::Limits limits{};
    const model::Qwen35HParams* hp = nullptr;
    std::size_t E = 0, n_vocab = 0;
    float eps = 0.0f;
    // attention
    std::size_t nh = 0, nkv = 0, hd = 0, rot = 0;
    float theta = 0.0f;
    // GDN
    std::size_t n_k = 0, n_v = 0, d_k = 0, d_v = 0;
    std::size_t conv_k = 0, conv_c = 0, key_dim = 0, value_dim = 0;
    std::size_t gdn_chunk = 64;
    std::size_t valid_vocab = 0;  // M2: 0 = no clamp

    std::vector<LayerW> layers;
    Vec output_norm;
    Table embd;
    Table lm_head;
    std::uint64_t head_bytes = 0;
    std::uint64_t trunk_bytes = 0;
    std::optional<MtpW> mtp;
    /// Rows of the MTP head scored per draft (HALO_MTP_DRAFT_VOCAB; 0 = full vocab).
    std::size_t draft_vocab = 32768;

    // ---- loading -----------------------------------------------------------------------

    Mat make_mat(const WeightRef& w, std::size_t expect_rows, std::size_t expect_cols) const {
        HALO_CHECK(w.present(), ErrorCode::Model, "missing weight matrix");
        const cpu::WeightMatrix m = weight_matrix(w);  // validates type, shape, size, alignment
        HALO_CHECK(m.rows() == expect_rows && m.cols() == expect_cols, ErrorCode::Model, "tensor {}: {}x{}, expected {}x{}",
                   w.name(), m.rows(), m.cols(), expect_rows, expect_cols);
        Mat r;
        r.type = w.type();
        r.buf = be->import_host_readonly(w.data());
        r.rows = u32(m.rows());
        r.cols = u32(m.cols());
        r.bytes = w.n_bytes();
        return r;
    }

    Vec upload_vec(const std::vector<float>& v) const {
        Vec r;
        r.n = v.size();
        r.buf = be->allocate(v.size() * sizeof(float), backend::Tier::Vram);
        const auto s = be->create_stream();
        be->upload(*s, r.ref(), std::as_bytes(std::span(v)));
        s->submit();
        s->wait();
        return r;
    }

    Vec make_vec(const WeightRef& w, std::size_t expect) const {
        HALO_CHECK(w.present(), ErrorCode::Model, "missing weight vector");
        const std::vector<float> v = weight_vector(w);
        HALO_CHECK(v.size() == expect, ErrorCode::Model, "tensor {}: {} elements, expected {}", w.name(), v.size(), expect);
        return upload_vec(v);
    }

    Table make_table(const WeightRef& w) const { return Table{w, be->import_host_readonly(w.data())}; }

    AttnW load_attn(const model::AttentionWeights& w) const {
        return AttnW{make_mat(w.q, 2 * nh * hd, E), make_mat(w.k, nkv * hd, E), make_mat(w.v, nkv * hd, E),
                     make_mat(w.output, E, nh * hd), make_vec(w.q_norm, hd), make_vec(w.k_norm, hd)};
    }

    LayerW load_layer(const model::LayerWeights& w, std::size_t kind_index) const {
        LayerW l;
        l.kind = w.kind;
        l.kind_index = kind_index;
        l.attn_norm = make_vec(w.attn_norm, E);
        l.post_norm = make_vec(w.post_attention_norm, E);
        l.bytes = vec_bytes(w.attn_norm) + vec_bytes(w.post_attention_norm);
        if (w.kind == LayerKind::FullAttention) {
            l.attn = load_attn(w.attn);
            const auto& a = w.attn;
            l.bytes += a.q.n_bytes() + a.k.n_bytes() + a.v.n_bytes() + a.output.n_bytes() + a.q_norm.n_bytes() +
                       a.k_norm.n_bytes();
        } else {
            const auto& g = w.gdn;
            GdnW gw{make_mat(g.qkv, conv_c, E),
                    make_mat(g.gate, value_dim, E),
                    make_mat(g.beta, n_v, E),
                    make_mat(g.alpha, n_v, E),
                    make_mat(g.out, E, value_dim),
                    make_vec(g.conv1d, conv_c * conv_k),
                    make_vec(g.dt_bias, n_v),
                    make_vec(g.a, n_v),
                    make_vec(g.norm, d_v)};
            HALO_CHECK(g.conv1d.ne(0) == static_cast<std::int64_t>(conv_k), ErrorCode::Model, "tensor {}: kernel {}, expected {}",
                       g.conv1d.name(), g.conv1d.ne(0), conv_k);
            for (const float v : weight_vector(g.a)) {
                HALO_CHECK(v <= 0.0f, ErrorCode::Model, "tensor {}: ssm_a must be -exp(A_log) <= 0, found {}", g.a.name(), v);
            }
            l.gdn = std::move(gw);
            l.bytes += g.qkv.n_bytes() + g.gate.n_bytes() + g.beta.n_bytes() + g.alpha.n_bytes() + g.out.n_bytes() +
                       g.conv1d.n_bytes() + g.dt_bias.n_bytes() + g.a.n_bytes() + g.norm.n_bytes();
        }
        l.ffn = FfnW{make_mat(w.ffn_gate, hp->n_ff, E), make_mat(w.ffn_up, hp->n_ff, E), make_mat(w.ffn_down, E, hp->n_ff)};
        l.bytes += w.ffn_gate.n_bytes() + w.ffn_up.n_bytes() + w.ffn_down.n_bytes();
        return l;
    }

    // ---- one call's scratch: stream + step arena ------------------------------------------

    struct Step {
        const Impl& I;
        std::unique_ptr<backend::Stream> s;
        std::vector<std::unique_ptr<Buffer>> arena;  // lives until the call returns (after wait)
        StepCost& cost;

        Step(const Impl& impl, StepCost& c) : I(impl), s(impl.be->create_stream()), cost(c) {}
        Step(const Step&) = delete;
        Step& operator=(const Step&) = delete;
        ~Step() {
            if (s) s->abort();
        }

        TensorRef alloc(std::size_t floats) {
            arena.push_back(I.be->allocate(floats * kF32, backend::Tier::Vram));
            return TensorRef::of(*arena.back());
        }
        TensorRef import_ro(std::span<const std::byte> v) {
            arena.push_back(I.be->import_host_readonly(v));
            return TensorRef::of(*arena.back());
        }
        void sync() {
            s->submit();
            s->wait();
        }
        void download(TensorRef src, std::span<float> dst) { I.be->download(*s, src, std::as_writable_bytes(dst)); }
    };

    /// Row r0 of a dense [*, cols] fp32 tensor.
    static TensorRef rows_at(TensorRef t, std::size_t r0, std::size_t cols) { return t.shifted(r0 * cols * kF32); }

    // ---- blocks ------------------------------------------------------------------------

    void gemv(Step& st, TensorRef x, std::size_t R, const Mat& w, TensorRef y) const {
        be->gemv(*st.s, backend::GemvArgs{w.type, w.ref(), x, y, w.rows, w.cols, u32(R), {}});
        st.cost.weight_bytes += w.bytes;
        st.cost.activation_bytes += cpu::traffic_matmul(R, w.cols, w.rows, 0).total();
    }

    void rms_norm(Step& st, TensorRef x, std::size_t rows, std::size_t cols, const Vec& w, TensorRef out) const {
        be->rms_norm(*st.s, backend::RmsNormArgs{x, w.ref(), out, u32(rows), u32(cols), eps, {}});
        st.cost.activation_bytes += cpu::traffic_rms_norm(rows, cols).total();
    }

    /// x += y; out = rms_norm(x) * w (the fused ADD + RMS_NORM).
    void add_rms_norm(Step& st, TensorRef x, TensorRef y, std::size_t R, const Vec& w, TensorRef out) const {
        be->add_rms_norm(*st.s, backend::AddRmsNormArgs{x, y, x, w.ref(), out, u32(R), u32(E), eps, {}});
        st.cost.activation_bytes += cpu::traffic_binary(R * E).total() + cpu::traffic_rms_norm(R, E).total();
    }

    /// Scratch activations of one call, sized for R rows and reused by every layer (stream
    /// order makes the reuse safe on any backend).
    struct Acts {
        TensorRef x, xn, y;
        // attention
        TensorRef qg, q, k, kn, v, att, gated;
        // GDN
        TensorRef qkv, conv, z, beta, alpha, g, bs, o, on;
        // FFN
        TensorRef fg, fu, fs;
    };

    Acts make_acts(Step& st, std::size_t R, bool with_gdn) const {
        Acts a;
        const std::size_t qd = nh * hd, kvd = nkv * hd, ff = hp->n_ff;
        a.x = st.alloc(R * E);
        a.xn = st.alloc(R * E);
        a.y = st.alloc(R * E);
        a.qg = st.alloc(R * 2 * qd);
        a.q = st.alloc(R * qd);
        a.k = st.alloc(R * kvd);
        a.kn = st.alloc(R * kvd);
        a.v = st.alloc(R * kvd);
        a.att = st.alloc(R * qd);
        a.gated = st.alloc(R * qd);
        if (with_gdn) {
            a.qkv = st.alloc(R * conv_c);
            a.conv = st.alloc(R * conv_c);
            a.z = st.alloc(R * value_dim);
            a.beta = st.alloc(R * n_v);
            a.alpha = st.alloc(R * n_v);
            a.g = st.alloc(R * n_v);
            a.bs = st.alloc(R * n_v);
            a.o = st.alloc(R * value_dim);
            a.on = st.alloc(R * value_dim);
        }
        a.fg = st.alloc(R * ff);
        a.fu = st.alloc(R * ff);
        a.fs = st.alloc(R * ff);
        return a;
    }

    /// Full-attention mixer (D-004): y[R, E] from normed input xn[R, E]. Writes the K/V
    /// rows of every sequence into its KV cache at rows [kv.length(), + n) of kv_layer.
    void attention(Step& st, const Acts& a, const AttnW& w, std::size_t kv_layer, std::size_t R, std::span<const SeqRows> seqs,
                   TensorRef pos) const {
        StepCost& cost = st.cost;
        const std::size_t qd = nh * hd;
        const std::size_t kvd = nkv * hd;
        gemv(st, a.xn, R, w.q, a.qg);
        gemv(st, a.xn, R, w.k, a.k);
        gemv(st, a.xn, R, w.v, a.v);
        // Per-head q norm straight out of the interleaved [Q_h | gate_h] rows (head stride
        // 2 * hd) into a dense q: row r of [R * nh, hd] is token r / nh, head r % nh.
        rms_norm(st, a.qg.with_stride(2 * hd * kF32), R * nh, hd, w.q_norm, a.q);
        rms_norm(st, a.k, R * nkv, hd, w.k_norm, a.kn);
        be->partial_rope(*st.s, backend::RopeArgs{a.q, pos, u32(R), u32(nh), u32(hd), u32(rot), theta, 0, {}});
        be->partial_rope(*st.s, backend::RopeArgs{a.kn, pos, u32(R), u32(nkv), u32(hd), u32(rot), theta, 0, {}});
        cost.activation_bytes += cpu::traffic_partial_rope(R, nh, rot).total() + cpu::traffic_partial_rope(R, nkv, rot).total();

        const cpu::AttentionDims dims{nh, nkv, hd};
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        for (const SeqRows& s : seqs) {
            const std::size_t len = s.kv->length();
            const kv_cache::KvLayout& l = s.kv->pool().layout();
            const auto n_pool_blocks = u32(s.kv->pool().total_blocks());
            const auto n_table = u32(s.kv->blocks().size());
            be->kv_write(*st.s, backend::KvWriteArgs{s.pool, s.table, rows_at(a.kn, s.r0, kvd), rows_at(a.v, s.r0, kvd),
                                                     n_pool_blocks, u32(l.n_layers), u32(kv_layer), u32(l.block_tokens),
                                                     u32(kvd), n_table, u32(len), u32(s.n), {}, {}});
            cost.kv_bytes += 2ULL * 4 * s.n * kvd;
            backend::AttentionArgs at{};
            at.q = rows_at(a.q, s.r0, qd);
            at.kv_pool = s.pool;
            at.block_table = s.table;
            at.out = rows_at(a.att, s.r0, qd);
            at.n_pool_blocks = n_pool_blocks;
            at.n_layers = u32(l.n_layers);
            at.layer = u32(kv_layer);
            at.block_tokens = u32(l.block_tokens);
            at.n_block_table = n_table;
            at.n_head = u32(nh);
            at.n_kv_head = u32(nkv);
            at.head_dim = u32(hd);
            at.n_tokens = u32(s.n);
            at.q_offset = u32(len);
            at.scale = scale;
            be->attention(*st.s, at);
            const cpu::OpTraffic t = cpu::traffic_attention(dims, s.n, len);
            cost.kv_bytes += 2ULL * 4 * (len + s.n) * kvd;
            cost.activation_bytes += t.total() - 2ULL * 4 * (len + s.n) * kvd;
        }
        // out = att * sigmoid(gate): the gate halves read in place with the head stride.
        be->mul_sigmoid(*st.s, backend::EltwiseArgs{a.att, a.qg.shifted(hd * kF32).with_stride(2 * hd * kF32), a.gated,
                                                    u32(R * nh), u32(hd), {}});
        cost.activation_bytes += cpu::traffic_binary(R * qd).total();
        gemv(st, a.gated, R, w.o, a.y);
    }

    struct GdnSeq {
        state::GdnState* gdn = nullptr;
        std::size_t n_slots = 0;
        bool chunked = false;
    };

    /// Gated DeltaNet mixer (D-004 items 1-7).
    void gdn(Step& st, const Acts& a, const GdnW& w, std::size_t gdn_layer, std::size_t R, std::span<const SeqRows> seqs,
             std::span<const GdnSeq> gseqs) const {
        StepCost& cost = st.cost;
        const std::size_t nv = n_v;
        const std::size_t dv = d_v;
        gemv(st, a.xn, R, w.qkv, a.qkv);
        gemv(st, a.xn, R, w.gate, a.z);
        gemv(st, a.xn, R, w.beta, a.beta);
        gemv(st, a.xn, R, w.alpha, a.alpha);
        // beta = sigmoid(beta); g = ssm_a * softplus(alpha + dt_bias), ssm_a = -exp(A_log) (GGUF).
        be->gdn_gates(*st.s, backend::GdnGateArgs{a.alpha, a.beta, w.dt_bias.ref(), w.a.ref(), a.g, a.bs, u32(R), u32(nv), {}});
        cost.activation_bytes += 4 * cpu::traffic_unary(R * nv).total();

        const cpu::GdnDims dims{n_k, n_v, d_k, d_v, cpu::GdnHeadMapping::Tiled};
        const std::uint64_t cstride = conv_c * kF32;
        for (std::size_t si = 0; si < seqs.size(); ++si) {
            const SeqRows& s = seqs[si];
            const GdnSeq& g = gseqs[si];
            state::GdnState& gdn_st = *g.gdn;
            // ADR-001 §5.3: the conv and recurrent state are one device-resident ring slab
            // per sequence; the kernels read slab[live] and write the final state plus the
            // rollback slots to other physical slots (no host import/export per forward).
            const backend::StateRing ring{u32(gdn_st.ring_size()), gdn_st.live()};
            const TensorRef xs = rows_at(a.conv, s.r0, conv_c);
            be->conv1d_silu(*st.s, backend::Conv1dArgs{rows_at(a.qkv, s.r0, conv_c), w.conv.ref(),
                                                       gdn_st.conv_ref(gdn_layer), xs, {},
                                                       u32(s.n), u32(conv_c), u32(conv_k), u32(g.n_slots), ring, {}});
            const cpu::OpTraffic tc = cpu::traffic_causal_conv1d(s.n, conv_c, conv_k, g.n_slots);
            const std::uint64_t conv_state_bytes = 4ULL * (conv_k - 1) * conv_c;
            const std::uint64_t conv_slot_bytes = std::min(s.n, g.n_slots) * conv_state_bytes;
            cost.state_bytes += 2 * conv_state_bytes + conv_slot_bytes;
            cost.activation_bytes += tc.total() - 2 * conv_state_bytes - conv_slot_bytes;

            backend::GdnArgs ga{};
            ga.form = g.chunked ? backend::GdnForm::Chunked : backend::GdnForm::Recurrent;
            ga.q = xs.with_stride(cstride);
            ga.k = xs.shifted(key_dim * kF32).with_stride(cstride);
            ga.v = xs.shifted(2 * key_dim * kF32).with_stride(cstride);
            ga.g = rows_at(a.g, s.r0, nv);
            ga.beta = rows_at(a.bs, s.r0, nv);
            ga.state = gdn_st.recurrent_ref(gdn_layer);
            ga.out = rows_at(a.o, s.r0, value_dim);
            ga.n_k = u32(n_k);
            ga.n_v = u32(n_v);
            ga.d_k = u32(d_k);
            ga.d_v = u32(d_v);
            ga.n_tokens = u32(s.n);
            ga.n_slots = u32(g.n_slots);
            ga.mapping = backend::GdnHeadMapping::Tiled;  // D-004 item 5: GGUF tiled V-head order
            // D-016: raw q/k after conv+SiLU; the kernel L2-normalizes q and k per head and
            // scales q by 1/sqrt(d_k) because we say so here.
            ga.qk_l2norm = true;
            ga.q_scale = 1.0f / std::sqrt(static_cast<float>(d_k));
            ga.chunk_size = u32(gdn_chunk);
            ga.ring = ring;
            be->gated_delta_rule(*st.s, ga);
            const cpu::OpTraffic tg = cpu::traffic_gated_delta_rule(dims, s.n, g.n_slots);
            const std::uint64_t stb = 4ULL * nv * d_k * dv;
            const std::uint64_t slots = std::min(s.n, g.n_slots) * stb;
            cost.state_bytes += 2 * stb + slots;
            cost.activation_bytes += tg.total() - 2 * stb - slots;
        }
        be->gated_rms_norm(*st.s, backend::GatedNormArgs{a.o, a.z, w.norm.ref(), a.on, u32(R * nv), u32(dv), eps, {}});
        cost.activation_bytes += cpu::traffic_gated_rms_norm(R * nv, dv).total();
        gemv(st, a.on, R, w.out, a.y);
    }

    void ffn(Step& st, const Acts& a, const FfnW& w, std::size_t R) const {
        gemv(st, a.xn, R, w.gate, a.fg);
        gemv(st, a.xn, R, w.up, a.fu);
        be->swiglu(*st.s, backend::EltwiseArgs{a.fg, a.fu, a.fs, u32(R), u32(hp->n_ff), {}});
        st.cost.activation_bytes += cpu::traffic_binary(R * hp->n_ff).total();
        gemv(st, a.fs, R, w.down, a.y);
    }

    /// One decoder layer on a.x whose input norm a.xn = rms_norm(x) * attn_norm is already
    /// computed: x += mixer(xn); xn = post_norm(x); x += ffn(xn); then the next norm
    /// (the following layer's attn_norm or the output norm) fused with the last add into
    /// `next_out`.
    template <class Mixer>
    void decoder_layer(Step& st, const Acts& a, const LayerW& l, std::size_t R, Mixer&& mixer, const Vec& next_norm,
                       TensorRef next_out) const {
        mixer();
        add_rms_norm(st, a.x, a.y, R, l.post_norm, a.xn);
        ffn(st, a, *l.ffn, R);
        add_rms_norm(st, a.x, a.y, R, next_norm, next_out);
    }

    struct HeadReq {
        std::size_t row = 0;  // row in the hidden matrix
        SeqOutput* out = nullptr;
        std::size_t slot = 0;  // index into out->argmax / out->logits
        bool full = false;
    };

    /// LM head over the requested rows: one pass over the head matrix (the LM_HEAD op);
    /// ties resolve to the lowest index. H1: a NaN logit poisons only its own row -- the
    /// engine sees it as argmax.index == -1 (backend::decode_argmax's poisoned sentinel) and
    /// must fail only that request, not every sequence in the tick.
    ///
    /// `vocab_rows` < n_vocab scores only the first `vocab_rows` token rows of the head (a
    /// contiguous prefix of the row-major matrix, so no copy): the FR-Spec-style reduced
    /// draft head. Only valid when no request wants full logits.
    void head(Step& st, const Table& head_w, std::uint64_t bytes, TensorRef h, std::span<const HeadReq> reqs,
              std::size_t vocab_rows = 0) const {
        if (reqs.empty()) return;
        if (vocab_rows == 0 || vocab_rows > n_vocab) vocab_rows = n_vocab;
        const std::size_t n = reqs.size();
        const TensorRef xs = st.alloc(n * E);
        for (std::size_t i = 0; i < n; ++i) {
            be->copy(*st.s, backend::CopyArgs{rows_at(h, reqs[i].row, E), rows_at(xs, i, E), E * kF32, {}});
        }
        const bool any_full = std::any_of(reqs.begin(), reqs.end(), [](const HeadReq& r) { return r.full; });
        const TensorRef logits = any_full ? st.alloc(n * n_vocab) : TensorRef{};
        st.arena.push_back(be->allocate(n * backend::kArgmaxResultBytes, backend::Tier::Host));
        const TensorRef res = TensorRef::of(*st.arena.back());
        const WeightRef& w = head_w.ref;
        be->lm_head(*st.s, backend::LmHeadArgs{backend::GemvArgs{w.type(), TensorRef::of(*head_w.buf), xs, logits, u32(vocab_rows),
                                                                 u32(E), u32(n), {}},
                                               res,
                                               u32(std::min<std::size_t>(valid_vocab, vocab_rows)),
                                               {},
                                               {}});
        std::vector<std::byte> words(n * backend::kArgmaxResultBytes);
        be->download(*st.s, res, std::span(words));
        st.sync();
        const std::vector<backend::ArgmaxResult> best = backend::decode_argmax(words);
        for (std::size_t i = 0; i < n; ++i) {
            reqs[i].out->argmax[reqs[i].slot] = cpu::TopKEntry{best[i].index, best[i].value};
            if (reqs[i].full) {
                be->download(*st.s, rows_at(logits, i, n_vocab),
                             std::as_writable_bytes(std::span(&reqs[i].out->logits[reqs[i].slot * n_vocab], n_vocab)));
            }
        }
        if (any_full) st.sync();
        st.cost.weight_bytes += vocab_rows == n_vocab ? bytes : bytes / n_vocab * vocab_rows;
        st.cost.activation_bytes += cpu::traffic_matmul(n, E, vocab_rows, 0).total();
    }

    /// The KV pool storage image is the pool's State-arena buffer (ADR-001 §5.2, WS-BI-2
    /// stage 2: device-resident on GPU backends, attached once per pool); the host-owned
    /// block table is uploaded per step.
    void import_kv(Step& st, SeqRows& s) const {
        kv_cache::KvPool& pool = s.kv->pool();
        if (!pool.attached()) pool.attach(*be);  // tests build pools by hand; the engine attaches up front
        s.pool = pool.storage_ref();
        s.table = st.import_ro(std::as_bytes(s.kv->blocks()));
    }
};

// ---------------------------------------------------------------------------------------

Qwen35::Qwen35(const model::NormalizedModel& m, cpu::ThreadPool* pool) : Qwen35(m, pool, Qwen35Options{}) {}

Qwen35::Qwen35(const model::NormalizedModel& m, cpu::ThreadPool* pool, const Qwen35Options& options)
    : Qwen35(m, backend::make_cpu_backend(pool), options) {}

Qwen35::Qwen35(const model::NormalizedModel& m, backend::Backend& be, const Qwen35Options& options)
    : Qwen35(m, std::unique_ptr<backend::Backend>{}, options, &be) {}

Qwen35::Qwen35(const model::NormalizedModel& m, std::unique_ptr<backend::Backend> owned, const Qwen35Options& options,
               backend::Backend* borrowed)
    : model_(&m), impl_(std::make_unique<Impl>()) {
    HALO_CHECK(options.gdn_chunk >= 1 && options.gdn_chunk <= 4096, ErrorCode::Config, "qwen35: GDN chunk {} outside [1, 4096]",
               options.gdn_chunk);
    Impl& I = *impl_;
    I.owned = std::move(owned);
    I.be = borrowed != nullptr ? borrowed : I.owned.get();
    I.limits = I.be->limits();
    I.gdn_chunk = options.gdn_chunk;
    const auto& hp = m.hparams();
    I.hp = &hp;
    I.E = hp.n_embd;
    I.n_vocab = static_cast<std::size_t>(hp.n_vocab);
    HALO_CHECK(options.valid_vocab <= I.n_vocab, ErrorCode::Config, "qwen35: valid_vocab {} exceeds the head's {} rows",
               options.valid_vocab, I.n_vocab);
    I.valid_vocab = options.valid_vocab;
    I.eps = hp.rms_eps;
    HALO_CHECK(hp.key_length == hp.value_length, ErrorCode::Unsupported, "qwen35: key_length {} != value_length {}",
               hp.key_length, hp.value_length);
    HALO_CHECK(hp.rope_dim % 2 == 0 && hp.rope_dim <= hp.key_length, ErrorCode::Model, "qwen35: rope dim {} for head dim {}",
               hp.rope_dim, hp.key_length);
    I.nh = hp.n_head;
    I.nkv = hp.n_head_kv;
    I.hd = hp.key_length;
    I.rot = hp.rope_dim;
    I.theta = hp.rope_freq_base;
    I.n_k = hp.gdn_n_k_heads;
    I.n_v = hp.gdn_n_v_heads;
    I.d_k = hp.gdn_head_k_dim;
    I.d_v = hp.gdn_head_v_dim;
    I.conv_k = hp.ssm_conv_kernel;
    I.conv_c = hp.gdn_conv_channels;
    I.key_dim = hp.gdn_key_dim;
    I.value_dim = hp.gdn_value_dim;
    HALO_CHECK(I.conv_k >= 1 && I.conv_c == 2 * I.key_dim + I.value_dim, ErrorCode::Model,
               "qwen35: conv channels {} != 2 * {} + {}", I.conv_c, I.key_dim, I.value_dim);
    // Backend operand limits (ADR §5.2): fail at construction, not at op N.
    const backend::Limits& lim = I.limits;
    const std::string be_name(backend::to_string(I.be->kind()));
    HALO_CHECK(I.d_k <= lim.max_gdn_dk, ErrorCode::Unsupported, "qwen35: GDN d_k {} exceeds the {} backend's limit {}", I.d_k,
               be_name, lim.max_gdn_dk);
    HALO_CHECK(I.hd <= lim.max_head_dim, ErrorCode::Unsupported, "qwen35: head dim {} exceeds the {} backend's limit {}", I.hd,
               be_name, lim.max_head_dim);
    HALO_CHECK(I.conv_k <= lim.max_conv_k, ErrorCode::Unsupported, "qwen35: conv kernel {} exceeds the {} backend's limit {}",
               I.conv_k, be_name, lim.max_conv_k);
    HALO_CHECK(I.rot <= lim.max_rope_dims, ErrorCode::Unsupported, "qwen35: rope dims {} exceed the {} backend's limit {}", I.rot,
               be_name, lim.max_rope_dims);
    HALO_CHECK(!lim.gdn_chunked || I.gdn_chunk <= lim.max_gdn_chunk, ErrorCode::Unsupported,
               "qwen35: GDN chunk {} exceeds the {} backend's limit {}", I.gdn_chunk, be_name, lim.max_gdn_chunk);

    std::size_t n_gdn = 0, n_attn = 0;
    for (const auto& lw : m.layers()) {
        const std::size_t idx = lw.kind == LayerKind::FullAttention ? n_attn++ : n_gdn++;
        I.layers.push_back(I.load_layer(lw, idx));
        I.trunk_bytes += I.layers.back().bytes;
    }
    I.output_norm = I.make_vec(m.output_norm(), I.E);
    I.trunk_bytes += m.output_norm().n_bytes();
    I.embd = I.make_table(m.token_embd());
    I.head_bytes = m.output().n_bytes();
    const WeightRef head_ref = m.output();
    HALO_CHECK(head_ref.ne(0) == static_cast<std::int64_t>(I.E) && head_ref.ne(1) == hp.n_vocab, ErrorCode::Model,
               "qwen35: LM head is {}x{}", head_ref.ne(0), head_ref.ne(1));
    (void)weight_matrix(head_ref);  // validates type/shape once
    I.lm_head = I.make_table(head_ref);

    if (const model::MtpWeights* mw = m.mtp()) {
        MtpW t{I.load_layer(mw->block, 0), I.make_mat(mw->eh_proj, I.E, 2 * I.E), I.make_vec(mw->enorm, I.E),
               I.make_vec(mw->hnorm, I.E), I.make_vec(mw->head_norm, I.E), Table{}, Table{}, 0};
        HALO_CHECK(t.block.kind == LayerKind::FullAttention, ErrorCode::Model, "qwen35: MTP block is not full attention");
        HALO_CHECK(mw->embedding.ne(0) == static_cast<std::int64_t>(I.E) && mw->lm_head.ne(0) == static_cast<std::int64_t>(I.E) &&
                       mw->lm_head.ne(1) == hp.n_vocab,
                   ErrorCode::Model, "qwen35: MTP embedding/head shape mismatch");
        (void)weight_matrix(mw->lm_head);
        t.embedding = I.make_table(mw->embedding);
        t.lm_head = I.make_table(mw->lm_head);
        t.bytes = t.block.bytes + mw->eh_proj.n_bytes() + mw->enorm.n_bytes() + mw->hnorm.n_bytes() +
                  (mw->head_norm_origin == model::MtpTensorOrigin::NextnBlock ? mw->head_norm.n_bytes() : 0);
        if (const char* e = std::getenv("HALO_MTP_DRAFT_VOCAB")) I.draft_vocab = std::strtoull(e, nullptr, 10);
        I.mtp = std::move(t);
    }
}

Qwen35::~Qwen35() = default;

const model::Qwen35HParams& Qwen35::hparams() const noexcept { return *impl_->hp; }
bool Qwen35::has_mtp() const noexcept { return impl_->mtp.has_value(); }
std::size_t Qwen35::n_vocab() const noexcept { return impl_->n_vocab; }
std::size_t Qwen35::n_embd() const noexcept { return impl_->E; }
std::size_t Qwen35::gdn_chunk() const noexcept { return impl_->gdn_chunk; }
backend::Backend& Qwen35::backend() const noexcept { return *impl_->be; }
std::uint64_t Qwen35::trunk_weight_bytes() const noexcept { return impl_->trunk_bytes; }
std::uint64_t Qwen35::lm_head_bytes() const noexcept { return impl_->head_bytes; }
std::uint64_t Qwen35::mtp_block_bytes() const noexcept { return impl_->mtp ? impl_->mtp->bytes : 0; }
std::uint64_t Qwen35::mtp_head_bytes() const noexcept { return impl_->mtp ? impl_->mtp->lm_head.ref.n_bytes() : 0; }
std::uint64_t Qwen35::embedding_row_bytes() const noexcept {
    return tensor::row_bytes(impl_->embd.ref.type(), static_cast<std::int64_t>(impl_->E));
}

kv_cache::KvLayout Qwen35::kv_layout(std::size_t block_tokens) const {
    return {impl_->hp->n_attn_layers, impl_->nkv * impl_->hd, block_tokens};
}

kv_cache::KvLayout Qwen35::mtp_kv_layout(std::size_t block_tokens) const {
    return {1, impl_->nkv * impl_->hd, block_tokens};
}

state::GdnShape Qwen35::gdn_shape() const {
    const Impl& I = *impl_;
    return {I.hp->n_gdn_layers, I.n_v, I.d_k, I.d_v, I.conv_k, I.conv_c};
}

namespace {

void check_kv(const kv_cache::SequenceKv* kv, const kv_cache::KvLayout& want, const char* what, std::size_t seq) {
    HALO_CHECK(kv != nullptr, ErrorCode::Api, "{} step {}: no KV cache", what, seq);
    const auto& l = kv->pool().layout();
    HALO_CHECK(l.n_layers == want.n_layers && l.kv_dim == want.kv_dim, ErrorCode::Api,
               "{} step {}: KV pool has {} layers x {} floats, model needs {} x {}", what, seq, l.n_layers, l.kv_dim,
               want.n_layers, want.kv_dim);
}

void check_rows(std::span<const std::size_t> rows, std::size_t n, const char* what, std::size_t seq) {
    for (const std::size_t r : rows) {
        HALO_CHECK(r < n, ErrorCode::Api, "{} step {}: logit row {} of {} rows", what, seq, r, n);
    }
}

void check_tokens(std::span<const std::int32_t> tokens, std::size_t n_vocab, const char* what, std::size_t seq) {
    HALO_CHECK(!tokens.empty(), ErrorCode::Api, "{} step {}: no tokens", what, seq);
    for (const std::int32_t t : tokens) {
        HALO_CHECK(t >= 0 && static_cast<std::size_t>(t) < n_vocab, ErrorCode::Api, "{} step {}: token id {} outside [0, {})",
                   what, seq, t, n_vocab);
    }
}

// O(n log n) duplicate-pointer check, replacing the O(n^2) pairwise scan every batch step used to
// do: a pair of steps is invalid if they share the *same* state pointer, so sorting once and
// scanning for adjacent equal entries finds any violation without comparing every pair.
template <typename T>
void check_all_distinct(std::span<T* const> ptrs, const char* what, const char* field) {
    std::vector<std::pair<T*, std::size_t>> by_ptr(ptrs.size());
    for (std::size_t i = 0; i < ptrs.size(); ++i) by_ptr[i] = {ptrs[i], i};
    std::sort(by_ptr.begin(), by_ptr.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (std::size_t i = 1; i < by_ptr.size(); ++i) {
        HALO_CHECK(by_ptr[i].first != by_ptr[i - 1].first, ErrorCode::Api, "{}: steps {} and {} share a sequence's {}", what,
                   by_ptr[i - 1].second, by_ptr[i].second, field);
    }
}

}  // namespace

void Qwen35::forward(std::span<const SeqStep> steps, StepResult& out, const ForwardOptions& opts) const {
    const Impl& I = *impl_;
    const state::GdnShape shape = gdn_shape();
    const kv_cache::KvLayout want_kv = kv_layout();
    // ---- validate everything before touching any state ---------------------------------
    std::vector<SeqRows> seqs(steps.size());
    std::vector<Impl::GdnSeq> gseqs(steps.size());
    std::vector<kv_cache::SequenceKv*> kv_ptrs(steps.size());
    std::vector<state::GdnState*> gdn_ptrs(steps.size());
    std::size_t R = 0;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const SeqStep& s = steps[si];
        check_tokens(s.tokens, I.n_vocab, "forward", si);
        check_kv(s.kv, want_kv, "forward", si);
        HALO_CHECK(s.gdn != nullptr, ErrorCode::Api, "forward step {}: no GDN state", si);
        const auto& gs = s.gdn->shape();
        HALO_CHECK(gs.n_layers == shape.n_layers && gs.recurrent_floats() == shape.recurrent_floats() &&
                       gs.conv_floats() == shape.conv_floats(),
                   ErrorCode::Api, "forward step {}: GDN state shape does not match the model", si);
        HALO_CHECK(s.n_state_slots <= s.gdn->max_slots(), ErrorCode::Api, "forward step {}: {} state slots, state has {}", si,
                   s.n_state_slots, s.gdn->max_slots());
        kv_ptrs[si] = s.kv;
        gdn_ptrs[si] = s.gdn;
        const std::size_t len = s.kv->length();
        HALO_CHECK(len + s.tokens.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()), ErrorCode::Api,
                   "forward step {}: position overflow", si);
        if (!s.positions.empty()) {
            HALO_CHECK(s.positions.size() == s.tokens.size(), ErrorCode::Api, "forward step {}: {} positions for {} tokens", si,
                       s.positions.size(), s.tokens.size());
            for (std::size_t i = 0; i < s.positions.size(); ++i) {
                HALO_CHECK(s.positions[i] == static_cast<std::int32_t>(len + i), ErrorCode::Unsupported,
                           "forward step {}: position {} at row {} (text-only 1-D RoPE needs {})", si, s.positions[i], i,
                           len + i);
            }
        }
        check_rows(s.logit_rows, s.tokens.size(), "forward", si);
        HALO_CHECK(s.gdn_path != GdnPath::Chunked || I.limits.gdn_chunked, ErrorCode::Unsupported,
                   "forward step {}: the {} backend has no chunked GDN form", si, backend::to_string(I.be->kind()));
        seqs[si] = {R, s.tokens.size(), s.kv, static_cast<std::int32_t>(len), {}, {}};
        const bool chunked = s.gdn_path == GdnPath::Chunked ||
                             (s.gdn_path == GdnPath::Auto && I.limits.gdn_chunked && s.tokens.size() > 1 && s.n_state_slots == 0);
        gseqs[si] = {s.gdn, s.n_state_slots, chunked};
        R += s.tokens.size();
    }
    check_all_distinct<kv_cache::SequenceKv>(kv_ptrs, "forward", "KV cache");
    check_all_distinct<state::GdnState>(gdn_ptrs, "forward", "GDN state");
    // ADR-001 §5.3: attach the ring slab to this backend on first use, then record the
    // step's base (live) — an uncommitted verify on this sequence is Error(Api) here. Done
    // after the distinctness check so a duplicated state is never begun twice.
    for (const SeqStep& s : steps) {
        if (!s.gdn->attached()) s.gdn->attach(*I.be);
        s.gdn->begin_step(s.tokens.size(), s.n_state_slots);
    }
    out.seqs.assign(steps.size(), {});
    out.cost = {};
    out.layer_inputs.clear();
    out.gdn_paths.clear();
    for (const auto& g : gseqs) out.gdn_paths.push_back(g.chunked ? GdnPath::Chunked : GdnPath::Recurrent);
    if (steps.empty()) return;
    for (const SeqStep& s : steps) s.kv->reserve(s.tokens.size());  // Error(Memory); lengths unchanged

    // ---- trunk -----------------------------------------------------------------------------
    StepCost& cost = out.cost;
    cost.weight_passes = 1;
    const std::size_t E = I.E;
    Impl::Step st(I, cost);
    std::vector<std::int32_t> ids(R), pos(R);
    for (std::size_t si = 0; si < steps.size(); ++si) {
        std::copy(steps[si].tokens.begin(), steps[si].tokens.end(), &ids[seqs[si].r0]);
        for (std::size_t i = 0; i < seqs[si].n; ++i) pos[seqs[si].r0 + i] = seqs[si].pos0 + static_cast<std::int32_t>(i);
    }
    const TensorRef rids = st.import_ro(std::as_bytes(std::span(ids)));
    const TensorRef rpos = st.import_ro(std::as_bytes(std::span(pos)));
    for (SeqRows& s : seqs) I.import_kv(st, s);
    const bool any_gdn = std::any_of(I.layers.begin(), I.layers.end(), [](const LayerW& l) { return l.gdn.has_value(); });
    const Impl::Acts a = I.make_acts(st, R, any_gdn);
    const TensorRef hbuf = st.alloc(R * E);

    I.be->get_rows(*st.s, backend::GetRowsArgs{I.embd.ref.type(), TensorRef::of(*I.embd.buf), rids, a.x,
                                               u32(I.embd.ref.ne(1)), u32(E), u32(R), {}, {}});
    cost.embedding_bytes += R * embedding_row_bytes();
    if (!I.layers.empty()) I.rms_norm(st, a.x, R, E, I.layers.front().attn_norm, a.xn);
    for (std::size_t li = 0; li < I.layers.size(); ++li) {
        const LayerW& l = I.layers[li];
        if (opts.capture_layer_inputs) {
            out.layer_inputs.emplace_back(R * E);
            st.download(a.x, out.layer_inputs.back());
        }
        const bool last = li + 1 == I.layers.size();
        const Vec& next_norm = last ? I.output_norm : I.layers[li + 1].attn_norm;
        const TensorRef next_out = last ? hbuf : a.xn;
        if (l.kind == LayerKind::FullAttention) {
            I.decoder_layer(st, a, l, R, [&] { I.attention(st, a, *l.attn, l.kind_index, R, seqs, rpos); }, next_norm, next_out);
        } else {
            I.decoder_layer(st, a, l, R, [&] { I.gdn(st, a, *l.gdn, l.kind_index, R, seqs, gseqs); }, next_norm, next_out);
        }
        // attn_norm (layer 0) / post_norm / next norm accounting as one rms_norm per norm:
        // add_rms_norm counted the add and the norm; the 2 adds of the layer are counted there.
    }
    if (I.layers.empty()) I.rms_norm(st, a.x, R, E, I.output_norm, hbuf);
    // No sync here: the hidden download and the head are recorded into the same command
    // buffer, so one submit and one fence wait (in head() / below) cover the whole forward.

    // ---- head ------------------------------------------------------------------------------
    // H1/M1 (ADR-001 WS-BI-2 step 1): KV/GDN commit happens AFTER head() succeeds, not before.
    // head() can throw (NaN logit, Error(Kernel); allocation failure, Error(Memory)/bad_alloc)
    // after this point but before any state below is touched, so on that exception pre-tick
    // KV/GDN state is genuinely intact — the precondition the engine's Memory-error retry
    // already assumes (M1) and a future per-sequence fault domain (H1) will need.
    std::vector<Impl::HeadReq> reqs;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const SeqStep& s = steps[si];
        SeqOutput& o = out.seqs[si];
        if (s.want_hidden) {
            o.hidden.resize(s.tokens.size() * E);
            st.download(Impl::rows_at(hbuf, seqs[si].r0, E), o.hidden);
        }
        if (s.logits == LogitsMode::None) continue;
        o.argmax.resize(s.logit_rows.size());
        if (s.logits == LogitsMode::Full) o.logits.resize(s.logit_rows.size() * I.n_vocab);
        for (std::size_t j = 0; j < s.logit_rows.size(); ++j) {
            reqs.push_back({seqs[si].r0 + s.logit_rows[j], &o, j, s.logits == LogitsMode::Full});
        }
    }
    I.head(st, I.lm_head, I.head_bytes, hbuf, reqs);
    st.sync();

    for (std::size_t si = 0; si < steps.size(); ++si) {
        steps[si].kv->commit(steps[si].tokens.size());
        steps[si].gdn->mark_slots_written(steps[si].tokens.size(), steps[si].n_state_slots);
    }
}

void Qwen35::mtp_forward(std::span<const MtpStep> steps, StepResult& out) const {
    const Impl& I = *impl_;
    HALO_CHECK(I.mtp.has_value(), ErrorCode::Unsupported, "mtp_forward: the model has no MTP block");
    const MtpW& M = *I.mtp;
    const kv_cache::KvLayout want_kv = mtp_kv_layout();
    const std::size_t E = I.E;
    std::vector<SeqRows> seqs(steps.size());
    std::vector<kv_cache::SequenceKv*> kv_ptrs(steps.size());
    std::size_t R = 0;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const MtpStep& s = steps[si];
        check_tokens(s.tokens, I.n_vocab, "mtp_forward", si);
        check_kv(s.kv, want_kv, "mtp_forward", si);
        HALO_CHECK(s.hidden.size() == s.tokens.size() * E, ErrorCode::Api, "mtp_forward step {}: {} hidden floats for {} rows",
                   si, s.hidden.size(), s.tokens.size());
        HALO_CHECK(s.first_position >= 0 &&
                       static_cast<std::size_t>(s.first_position) + s.tokens.size() <=
                           static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
                   ErrorCode::Api, "mtp_forward step {}: bad first position {}", si, s.first_position);
        kv_ptrs[si] = s.kv;
        check_rows(s.logit_rows, s.tokens.size(), "mtp_forward", si);
        seqs[si] = {R, s.tokens.size(), s.kv, s.first_position, {}, {}};
        R += s.tokens.size();
    }
    check_all_distinct<kv_cache::SequenceKv>(kv_ptrs, "mtp_forward", "KV cache");
    out.seqs.assign(steps.size(), {});
    out.cost = {};
    out.layer_inputs.clear();
    out.gdn_paths.clear();
    if (steps.empty()) return;
    for (const MtpStep& s : steps) s.kv->reserve(s.tokens.size());

    StepCost& cost = out.cost;
    cost.weight_passes = 1;
    Impl::Step st(I, cost);
    std::vector<std::int32_t> ids(R), pos(R);
    for (std::size_t si = 0; si < steps.size(); ++si) {
        std::copy(steps[si].tokens.begin(), steps[si].tokens.end(), &ids[seqs[si].r0]);
        for (std::size_t i = 0; i < seqs[si].n; ++i) pos[seqs[si].r0 + i] = seqs[si].pos0 + static_cast<std::int32_t>(i);
    }
    const TensorRef rids = st.import_ro(std::as_bytes(std::span(ids)));
    const TensorRef rpos = st.import_ro(std::as_bytes(std::span(pos)));
    for (SeqRows& s : seqs) I.import_kv(st, s);
    const Impl::Acts a = I.make_acts(st, R, false);
    // concat(rmsnorm(embed(tok); enorm), rmsnorm(h; hnorm)) — embedding first (D-005): both
    // norms write straight into their half of the [R, 2E] concat rows.
    const TensorRef e = st.alloc(R * E), hh = st.alloc(R * E), cat = st.alloc(R * 2 * E), hbuf = st.alloc(R * E);
    I.be->get_rows(*st.s, backend::GetRowsArgs{M.embedding.ref.type(), TensorRef::of(*M.embedding.buf), rids, e,
                                               u32(M.embedding.ref.ne(1)), u32(E), u32(R), {}, {}});
    for (std::size_t si = 0; si < steps.size(); ++si) {
        I.be->upload(*st.s, Impl::rows_at(hh, seqs[si].r0, E), std::as_bytes(steps[si].hidden));
    }
    cost.embedding_bytes += R * tensor::row_bytes(M.embedding.ref.type(), static_cast<std::int64_t>(E));
    I.rms_norm(st, e, R, E, M.enorm, cat.with_stride(2 * E * kF32));
    I.rms_norm(st, hh, R, E, M.hnorm, cat.shifted(E * kF32).with_stride(2 * E * kF32));
    I.gemv(st, cat, R, M.eh_proj, a.x);
    I.rms_norm(st, a.x, R, E, M.block.attn_norm, a.xn);
    I.decoder_layer(st, a, M.block, R, [&] { I.attention(st, a, *M.block.attn, 0, R, seqs, rpos); }, M.head_norm, hbuf);
    // No sync: the head is recorded into the same command buffer (see Qwen35::forward).

    // H1/M1: see the matching comment in Qwen35::forward -- commit after head() succeeds.
    std::vector<Impl::HeadReq> reqs;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const MtpStep& s = steps[si];
        SeqOutput& o = out.seqs[si];
        if (s.want_hidden) {
            o.hidden.resize(s.tokens.size() * E);
            st.download(Impl::rows_at(hbuf, seqs[si].r0, E), o.hidden);
        }
        if (s.logits == LogitsMode::None) continue;
        o.argmax.resize(s.logit_rows.size());
        if (s.logits == LogitsMode::Full) o.logits.resize(s.logit_rows.size() * I.n_vocab);
        for (std::size_t j = 0; j < s.logit_rows.size(); ++j) {
            reqs.push_back({seqs[si].r0 + s.logit_rows[j], &o, j, s.logits == LogitsMode::Full});
        }
    }
    // Drafts only need a good guess (the trunk head verifies every token), so score a
    // frequent-token prefix of the MTP head unless a request wants full logits.
    const bool any_full = std::any_of(reqs.begin(), reqs.end(), [](const Impl::HeadReq& r) { return r.full; });
    I.head(st, M.lm_head, M.lm_head.ref.n_bytes(), hbuf, reqs, any_full ? 0 : I.draft_vocab);
    st.sync();
    for (const MtpStep& s : steps) s.kv->commit(s.tokens.size());
}

}  // namespace halo::models
