// Qwen3.8 ("qwen35") CPU reference forward (D-004) and MTP block (D-005).

#include "halo/models/qwen35.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

#include "halo/backends/cpu/traffic.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"

namespace halo::models {

using cpu::ConstRows;
using cpu::Rows;
using model::LayerKind;
using model::WeightRef;

namespace {

struct Mat {
    cpu::WeightMatrix m;
    std::uint64_t bytes = 0;
};

Mat make_mat(const WeightRef& w, std::size_t expect_rows, std::size_t expect_cols) {
    HALO_CHECK(w.present(), ErrorCode::Model, "missing weight matrix");
    Mat r{weight_matrix(w), w.n_bytes()};
    HALO_CHECK(r.m.rows() == expect_rows && r.m.cols() == expect_cols, ErrorCode::Model,
               "tensor {}: {}x{}, expected {}x{}", w.name(), r.m.rows(), r.m.cols(), expect_rows, expect_cols);
    return r;
}

std::vector<float> make_vec(const WeightRef& w, std::size_t expect) {
    HALO_CHECK(w.present(), ErrorCode::Model, "missing weight vector");
    std::vector<float> v = weight_vector(w);
    HALO_CHECK(v.size() == expect, ErrorCode::Model, "tensor {}: {} elements, expected {}", w.name(), v.size(), expect);
    return v;
}

struct AttnW {
    Mat q, k, v, o;
    std::vector<float> q_norm, k_norm;
};

struct GdnW {
    Mat qkv, gate, beta, alpha, out;
    std::vector<float> conv;  // [C, K]
    std::vector<float> dt_bias, a, norm;
};

struct FfnW {
    Mat gate, up, down;
};

struct LayerW {
    LayerKind kind = LayerKind::GatedDeltaNet;
    std::size_t kind_index = 0;  // index among GDN layers (state) or attention layers (KV)
    std::vector<float> attn_norm, post_norm;
    std::optional<AttnW> attn;
    std::optional<GdnW> gdn;
    std::optional<FfnW> ffn;
    std::uint64_t bytes = 0;  // stored bytes of every weight of the layer
};

struct MtpW {
    LayerW block;
    Mat eh_proj;
    std::vector<float> enorm, hnorm, head_norm;
    WeightRef embedding;
    WeightRef head_ref;
    std::uint64_t bytes = 0;  // block + eh_proj + norms (not embedding / head)
};

/// Rows of one sequence inside the concatenated batch.
struct SeqRows {
    std::size_t r0 = 0;
    std::size_t n = 0;
    kv_cache::SequenceKv* kv = nullptr;
    std::int32_t pos0 = 0;
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
    const model::Qwen35HParams* hp = nullptr;
    cpu::ThreadPool* pool = nullptr;
    std::size_t E = 0, n_vocab = 0;
    float eps = 0.0f;
    // attention
    std::size_t nh = 0, nkv = 0, hd = 0, rot = 0;
    float theta = 0.0f;
    // GDN
    cpu::GdnDims gdn_dims;
    std::size_t conv_k = 0, conv_c = 0, key_dim = 0, value_dim = 0;
    std::size_t gdn_chunk = 64;

    std::vector<LayerW> layers;
    std::vector<float> output_norm;
    WeightRef embd;
    WeightRef head_ref;
    std::uint64_t head_bytes = 0;
    std::uint64_t trunk_bytes = 0;
    std::optional<MtpW> mtp;

    // ---------------------------------------------------------------------------------
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
            const std::size_t nv = gdn_dims.n_v_heads;
            GdnW gw{make_mat(g.qkv, conv_c, E),
                    make_mat(g.gate, value_dim, E),
                    make_mat(g.beta, nv, E),
                    make_mat(g.alpha, nv, E),
                    make_mat(g.out, E, value_dim),
                    make_vec(g.conv1d, conv_c * conv_k),
                    make_vec(g.dt_bias, nv),
                    make_vec(g.a, nv),
                    make_vec(g.norm, gdn_dims.d_v)};
            HALO_CHECK(g.conv1d.ne(0) == static_cast<std::int64_t>(conv_k), ErrorCode::Model, "tensor {}: kernel {}, expected {}",
                       g.conv1d.name(), g.conv1d.ne(0), conv_k);
            for (const float v : gw.a) {
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

    // ---------------------------------------------------------------------------------
    // Blocks
    // ---------------------------------------------------------------------------------

    void matmul(ConstRows x, const Mat& w, Rows y, StepCost& cost) const {
        cpu::matmul(x, w.m, y, pool);
        cost.weight_bytes += w.bytes;
        const cpu::OpTraffic t = cpu::traffic_matmul(x.rows(), x.cols(), y.cols(), 0);
        cost.activation_bytes += t.total();
    }

    void rms_norm(ConstRows x, std::span<const float> w, Rows out, StepCost& cost) const {
        cpu::rms_norm(x, w, eps, out, pool);
        cost.activation_bytes += cpu::traffic_rms_norm(x.rows(), x.cols()).total();
    }

    /// Full-attention mixer (D-004): out[R, E] from normed input xn[R, E]. Writes the K/V
    /// rows of every sequence into its KV cache at rows [kv.length(), + n) of kv_layer.
    void attention(const AttnW& w, std::size_t kv_layer, ConstRows xn, std::span<const SeqRows> seqs, Rows out,
                   StepCost& cost) const {
        const std::size_t R = xn.rows();
        const std::size_t qd = nh * hd;
        const std::size_t kvd = nkv * hd;
        std::vector<float> qg(R * 2 * qd), q(R * qd), gate(R * qd), k(R * kvd), v(R * kvd), att(R * qd);
        matmul(xn, w.q, Rows(std::span(qg), R, 2 * qd), cost);
        matmul(xn, w.k, Rows(std::span(k), R, kvd), cost);
        matmul(xn, w.v, Rows(std::span(v), R, kvd), cost);
        // De-interleave [Q_h | gate_h] per head into dense buffers (review S-2: token-major
        // head views cannot express the interleave; an in-kernel stride is a later
        // optimization).
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t h = 0; h < nh; ++h) {
                const float* src = &qg[r * 2 * qd + h * 2 * hd];
                std::copy(src, src + hd, &q[r * qd + h * hd]);
                std::copy(src + hd, src + 2 * hd, &gate[r * qd + h * hd]);
            }
        }
        cost.activation_bytes += 4ULL * 4 * R * qd;
        rms_norm(ConstRows(std::span<const float>(q), R * nh, hd), w.q_norm, Rows(std::span(q), R * nh, hd), cost);
        rms_norm(ConstRows(std::span<const float>(k), R * nkv, hd), w.k_norm, Rows(std::span(k), R * nkv, hd), cost);
        std::vector<std::int32_t> pos(R);
        for (const SeqRows& s : seqs) {
            for (std::size_t i = 0; i < s.n; ++i) pos[s.r0 + i] = s.pos0 + static_cast<std::int32_t>(i);
        }
        cpu::partial_rope_neox(Rows(std::span(q), R, qd), nh, hd, pos, rot, theta, pool);
        cpu::partial_rope_neox(Rows(std::span(k), R, kvd), nkv, hd, pos, rot, theta, pool);
        cost.activation_bytes += cpu::traffic_partial_rope(R, nh, rot).total() + cpu::traffic_partial_rope(R, nkv, rot).total();

        const cpu::AttentionDims dims{nh, nkv, hd};
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        std::vector<const float*> tk, tv;
        for (const SeqRows& s : seqs) {
            const std::size_t len = s.kv->length();
            for (std::size_t i = 0; i < s.n; ++i) {
                const std::size_t r = s.r0 + i;
                s.kv->write(kv_layer, len + i, std::span<const float>(&k[r * kvd], kvd), std::span<const float>(&v[r * kvd], kvd));
            }
            cost.kv_bytes += 2ULL * 4 * s.n * kvd;
            const cpu::PagedRows keys = s.kv->keys(kv_layer, len + s.n, tk);
            const cpu::PagedRows vals = s.kv->values(kv_layer, len + s.n, tv);
            cpu::attention_gqa(dims, ConstRows(&q[s.r0 * qd], s.n, qd, qd), keys, vals, len, scale,
                               Rows(&att[s.r0 * qd], s.n, qd, qd), pool);
            const cpu::OpTraffic t = cpu::traffic_attention(dims, s.n, len);
            cost.kv_bytes += 2ULL * 4 * (len + s.n) * kvd;
            cost.activation_bytes += t.total() - 2ULL * 4 * (len + s.n) * kvd;
        }
        cpu::mul_sigmoid(ConstRows(std::span<const float>(att), R, qd), ConstRows(std::span<const float>(gate), R, qd),
                         Rows(std::span(att), R, qd), pool);
        cost.activation_bytes += cpu::traffic_binary(R * qd).total();
        matmul(ConstRows(std::span<const float>(att), R, qd), w.o, out, cost);
    }

    /// The one call into the GATED_DELTANET op (D-016): raw q/k after conv+SiLU, the kernel
    /// L2-normalizes q and k per head and scales q by 1/sqrt(d_k) because we say so here.
    void run_gdn(const cpu::GdnInputs& in, std::span<float> state, Rows o, bool chunked, std::span<float> slots) const {
        const cpu::GdnQkParams qk{.qk_l2norm = true,
                                  .q_scale = 1.0f / std::sqrt(static_cast<float>(gdn_dims.d_k))};
        if (chunked) {
            cpu::gated_delta_rule_chunked(gdn_dims, in, state, o, qk, gdn_chunk, pool, slots);
        } else {
            cpu::gated_delta_rule_recurrent(gdn_dims, in, state, o, qk, pool, slots);
        }
    }

    struct GdnSeq {
        state::GdnState* gdn = nullptr;
        std::size_t n_slots = 0;
        bool chunked = false;
    };

    /// Gated DeltaNet mixer (D-004 items 1-7).
    void gdn(const GdnW& w, std::size_t gdn_layer, ConstRows xn, std::span<const SeqRows> seqs,
             std::span<const GdnSeq> gseqs, Rows out, StepCost& cost) const {
        const std::size_t R = xn.rows();
        const std::size_t nv = gdn_dims.n_v_heads;
        const std::size_t dv = gdn_dims.d_v;
        std::vector<float> qkv(R * conv_c), z(R * value_dim), beta(R * nv), alpha(R * nv), o(R * value_dim);
        matmul(xn, w.qkv, Rows(std::span(qkv), R, conv_c), cost);
        matmul(xn, w.gate, Rows(std::span(z), R, value_dim), cost);
        matmul(xn, w.beta, Rows(std::span(beta), R, nv), cost);
        matmul(xn, w.alpha, Rows(std::span(alpha), R, nv), cost);
        cpu::sigmoid(ConstRows(std::span<const float>(beta), R, nv), Rows(std::span(beta), R, nv), pool);
        // g = ssm_a * softplus(alpha + dt_bias), ssm_a = -exp(A_log) (GGUF).
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t j = 0; j < nv; ++j) alpha[r * nv + j] += w.dt_bias[j];
        }
        cpu::softplus(ConstRows(std::span<const float>(alpha), R, nv), Rows(std::span(alpha), R, nv), pool);
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t j = 0; j < nv; ++j) alpha[r * nv + j] = w.a[j] * alpha[r * nv + j];
        }
        cost.activation_bytes += 4 * cpu::traffic_unary(R * nv).total();

        const ConstRows conv_w(std::span<const float>(w.conv), conv_c, conv_k);
        for (std::size_t si = 0; si < seqs.size(); ++si) {
            const SeqRows& s = seqs[si];
            const GdnSeq& g = gseqs[si];
            const Rows xs(&qkv[s.r0 * conv_c], s.n, conv_c, conv_c);
            cpu::causal_conv1d_silu(xs, conv_w, g.gdn->conv(gdn_layer), xs, pool, g.gdn->conv_slots(gdn_layer, g.n_slots));
            const cpu::OpTraffic tc = cpu::traffic_causal_conv1d(s.n, conv_c, conv_k, g.n_slots);
            const std::uint64_t conv_state = 4ULL * (conv_k - 1) * conv_c;
            const std::uint64_t conv_slots = std::min(s.n, g.n_slots) * conv_state;
            cost.state_bytes += 2 * conv_state + conv_slots;
            cost.activation_bytes += tc.total() - 2 * conv_state - conv_slots;
            const cpu::GdnInputs in{
                ConstRows(xs.data(), s.n, key_dim, conv_c),
                ConstRows(xs.data() + key_dim, s.n, key_dim, conv_c),
                ConstRows(xs.data() + 2 * key_dim, s.n, value_dim, conv_c),
                ConstRows(&alpha[s.r0 * nv], s.n, nv, nv),
                ConstRows(&beta[s.r0 * nv], s.n, nv, nv),
            };
            run_gdn(in, g.gdn->recurrent(gdn_layer), Rows(&o[s.r0 * value_dim], s.n, value_dim, value_dim), g.chunked,
                    g.gdn->recurrent_slots(gdn_layer, g.n_slots));
            const cpu::OpTraffic tg = cpu::traffic_gated_delta_rule(gdn_dims, s.n, g.n_slots);
            const std::uint64_t st = 4ULL * nv * gdn_dims.d_k * dv;
            const std::uint64_t slots = std::min(s.n, g.n_slots) * st;
            cost.state_bytes += 2 * st + slots;
            cost.activation_bytes += tg.total() - 2 * st - slots;
        }
        cpu::gated_rms_norm(ConstRows(std::span<const float>(o), R * nv, dv), w.norm, ConstRows(std::span<const float>(z), R * nv, dv),
                            eps, Rows(std::span(o), R * nv, dv), pool);
        cost.activation_bytes += cpu::traffic_gated_rms_norm(R * nv, dv).total();
        matmul(ConstRows(std::span<const float>(o), R, value_dim), w.out, out, cost);
    }

    void ffn(const FfnW& w, ConstRows xn, Rows out, StepCost& cost) const {
        const std::size_t R = xn.rows();
        const std::size_t ff = hp->n_ff;
        std::vector<float> g(R * ff), u(R * ff);
        matmul(xn, w.gate, Rows(std::span(g), R, ff), cost);
        matmul(xn, w.up, Rows(std::span(u), R, ff), cost);
        cpu::swiglu(ConstRows(std::span<const float>(g), R, ff), ConstRows(std::span<const float>(u), R, ff),
                    Rows(std::span(g), R, ff), pool);
        cost.activation_bytes += cpu::traffic_binary(R * ff).total();
        matmul(ConstRows(std::span<const float>(g), R, ff), w.down, out, cost);
    }

    /// x += attn_or_gdn(norm(x)); x += ffn(norm(x)).
    template <class Mixer>
    void decoder_layer(const LayerW& l, Rows x, Mixer&& mixer, StepCost& cost) const {
        const std::size_t R = x.rows();
        std::vector<float> xn(R * E), y(R * E);
        rms_norm(x, l.attn_norm, Rows(std::span(xn), R, E), cost);
        mixer(ConstRows(std::span<const float>(xn), R, E), Rows(std::span(y), R, E));
        cpu::add(x, ConstRows(std::span<const float>(y), R, E), x, pool);
        rms_norm(x, l.post_norm, Rows(std::span(xn), R, E), cost);
        ffn(*l.ffn, ConstRows(std::span<const float>(xn), R, E), Rows(std::span(y), R, E), cost);
        cpu::add(x, ConstRows(std::span<const float>(y), R, E), x, pool);
        cost.activation_bytes += 2 * cpu::traffic_binary(R * E).total();
    }

    struct HeadReq {
        std::size_t row = 0;  // row in the hidden matrix
        SeqOutput* out = nullptr;
        std::size_t slot = 0;  // index into out->argmax / out->logits
        bool full = false;
    };

    /// LM head over the requested rows: one pass over the head matrix in slabs; the
    /// argmax is taken over exactly the values matmul produces (ties: lowest index).
    void head(const WeightRef& head_w, std::uint64_t bytes, ConstRows h, std::span<const HeadReq> reqs, StepCost& cost) const {
        if (reqs.empty()) return;
        const std::size_t n = reqs.size();
        std::vector<float> xs(n * E);
        for (std::size_t i = 0; i < n; ++i) std::copy_n(h.row_ptr(reqs[i].row), E, &xs[i * E]);
        const ConstRows x(std::span<const float>(xs), n, E);
        constexpr std::size_t kSlab = 8192;
        const std::size_t slab = std::min(kSlab, n_vocab);
        std::vector<float> buf(n * slab);
        std::vector<cpu::TopKEntry> best(n);
        for (std::size_t n0 = 0; n0 < n_vocab; n0 += slab) {
            const std::size_t ns = std::min(slab, n_vocab - n0);
            const cpu::WeightMatrix sub = weight_matrix(head_w, n0, ns);
            cpu::matmul(x, sub, Rows(buf.data(), n, ns, slab), pool);
            for (std::size_t i = 0; i < n; ++i) {
                const float* row = &buf[i * slab];
                for (std::size_t c = 0; c < ns; ++c) {
                    HALO_CHECK(!std::isnan(row[c]), ErrorCode::Kernel, "LM head: NaN logit at vocab index {}", n0 + c);
                    if (best[i].index < 0 || row[c] > best[i].value) best[i] = {static_cast<std::int32_t>(n0 + c), row[c]};
                }
                if (reqs[i].full) std::copy_n(row, ns, &reqs[i].out->logits[reqs[i].slot * n_vocab + n0]);
            }
        }
        for (std::size_t i = 0; i < n; ++i) reqs[i].out->argmax[reqs[i].slot] = best[i];
        cost.weight_bytes += bytes;
        cost.activation_bytes += cpu::traffic_matmul(n, E, n_vocab, 0).total();
    }
};

// ---------------------------------------------------------------------------------------

Qwen35::Qwen35(const model::NormalizedModel& m, cpu::ThreadPool* pool) : Qwen35(m, pool, Qwen35Options{}) {}

Qwen35::Qwen35(const model::NormalizedModel& m, cpu::ThreadPool* pool, const Qwen35Options& options)
    : model_(&m), impl_(std::make_unique<Impl>()) {
    HALO_CHECK(options.gdn_chunk >= 1 && options.gdn_chunk <= 4096, ErrorCode::Config, "qwen35: GDN chunk {} outside [1, 4096]",
               options.gdn_chunk);
    Impl& I = *impl_;
    I.gdn_chunk = options.gdn_chunk;
    const auto& hp = m.hparams();
    I.hp = &hp;
    I.pool = pool;
    I.E = hp.n_embd;
    I.n_vocab = static_cast<std::size_t>(hp.n_vocab);
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
    I.gdn_dims = cpu::GdnDims{hp.gdn_n_k_heads, hp.gdn_n_v_heads, hp.gdn_head_k_dim, hp.gdn_head_v_dim,
                              cpu::GdnHeadMapping::Tiled};  // D-004 item 5: GGUF tiled V-head order
    I.conv_k = hp.ssm_conv_kernel;
    I.conv_c = hp.gdn_conv_channels;
    I.key_dim = hp.gdn_key_dim;
    I.value_dim = hp.gdn_value_dim;
    HALO_CHECK(I.conv_k >= 1 && I.conv_c == 2 * I.key_dim + I.value_dim, ErrorCode::Model,
               "qwen35: conv channels {} != 2 * {} + {}", I.conv_c, I.key_dim, I.value_dim);

    std::size_t n_gdn = 0, n_attn = 0;
    for (const auto& lw : m.layers()) {
        const std::size_t idx = lw.kind == LayerKind::FullAttention ? n_attn++ : n_gdn++;
        I.layers.push_back(I.load_layer(lw, idx));
        I.trunk_bytes += I.layers.back().bytes;
    }
    I.output_norm = make_vec(m.output_norm(), I.E);
    I.trunk_bytes += m.output_norm().n_bytes();
    I.embd = m.token_embd();
    I.head_ref = m.output();
    I.head_bytes = m.output().n_bytes();
    HALO_CHECK(I.head_ref.ne(0) == static_cast<std::int64_t>(I.E) && I.head_ref.ne(1) == hp.n_vocab, ErrorCode::Model,
               "qwen35: LM head is {}x{}", I.head_ref.ne(0), I.head_ref.ne(1));
    (void)weight_matrix(I.head_ref);  // validates type/shape once

    if (const model::MtpWeights* mw = m.mtp()) {
        MtpW t{I.load_layer(mw->block, 0), make_mat(mw->eh_proj, I.E, 2 * I.E), make_vec(mw->enorm, I.E),
               make_vec(mw->hnorm, I.E), make_vec(mw->head_norm, I.E), mw->embedding, mw->lm_head, 0};
        HALO_CHECK(t.block.kind == LayerKind::FullAttention, ErrorCode::Model, "qwen35: MTP block is not full attention");
        HALO_CHECK(t.embedding.ne(0) == static_cast<std::int64_t>(I.E) && t.head_ref.ne(0) == static_cast<std::int64_t>(I.E) &&
                       t.head_ref.ne(1) == hp.n_vocab,
                   ErrorCode::Model, "qwen35: MTP embedding/head shape mismatch");
        (void)weight_matrix(t.head_ref);
        t.bytes = t.block.bytes + mw->eh_proj.n_bytes() + mw->enorm.n_bytes() + mw->hnorm.n_bytes() +
                  (mw->head_norm_origin == model::MtpTensorOrigin::NextnBlock ? mw->head_norm.n_bytes() : 0);
        I.mtp = std::move(t);
    }
}

Qwen35::~Qwen35() = default;

const model::Qwen35HParams& Qwen35::hparams() const noexcept { return *impl_->hp; }
bool Qwen35::has_mtp() const noexcept { return impl_->mtp.has_value(); }
std::size_t Qwen35::n_vocab() const noexcept { return impl_->n_vocab; }
std::size_t Qwen35::n_embd() const noexcept { return impl_->E; }
std::size_t Qwen35::gdn_chunk() const noexcept { return impl_->gdn_chunk; }
std::uint64_t Qwen35::trunk_weight_bytes() const noexcept { return impl_->trunk_bytes; }
std::uint64_t Qwen35::lm_head_bytes() const noexcept { return impl_->head_bytes; }
std::uint64_t Qwen35::mtp_block_bytes() const noexcept { return impl_->mtp ? impl_->mtp->bytes : 0; }
std::uint64_t Qwen35::mtp_head_bytes() const noexcept { return impl_->mtp ? impl_->mtp->head_ref.n_bytes() : 0; }
std::uint64_t Qwen35::embedding_row_bytes() const noexcept {
    return tensor::row_bytes(impl_->embd.type(), static_cast<std::int64_t>(impl_->E));
}

kv_cache::KvLayout Qwen35::kv_layout(std::size_t block_tokens) const {
    return {impl_->hp->n_attn_layers, impl_->nkv * impl_->hd, block_tokens};
}

kv_cache::KvLayout Qwen35::mtp_kv_layout(std::size_t block_tokens) const {
    return {1, impl_->nkv * impl_->hd, block_tokens};
}

state::GdnShape Qwen35::gdn_shape() const {
    const Impl& I = *impl_;
    return {I.hp->n_gdn_layers, I.gdn_dims.n_v_heads, I.gdn_dims.d_k, I.gdn_dims.d_v, I.conv_k, I.conv_c};
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

}  // namespace

void Qwen35::forward(std::span<const SeqStep> steps, StepResult& out, const ForwardOptions& opts) const {
    const Impl& I = *impl_;
    const state::GdnShape shape = gdn_shape();
    const kv_cache::KvLayout want_kv = kv_layout();
    // ---- validate everything before touching any state ---------------------------------
    std::vector<SeqRows> seqs(steps.size());
    std::vector<Impl::GdnSeq> gseqs(steps.size());
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
        for (std::size_t sj = 0; sj < si; ++sj) {
            HALO_CHECK(steps[sj].kv != s.kv && steps[sj].gdn != s.gdn, ErrorCode::Api,
                       "forward: steps {} and {} share a sequence's state", sj, si);
        }
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
        seqs[si] = {R, s.tokens.size(), s.kv, static_cast<std::int32_t>(len)};
        const bool chunked = s.gdn_path == GdnPath::Chunked ||
                             (s.gdn_path == GdnPath::Auto && s.tokens.size() > 1 && s.n_state_slots == 0);
        gseqs[si] = {s.gdn, s.n_state_slots, chunked};
        R += s.tokens.size();
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
    std::vector<float> xbuf(R * E);
    const Rows x(std::span(xbuf), R, E);
    for (std::size_t si = 0; si < steps.size(); ++si) {
        for (std::size_t i = 0; i < steps[si].tokens.size(); ++i) {
            embedding_row(I.embd, steps[si].tokens[i], x.row(seqs[si].r0 + i));
        }
    }
    cost.embedding_bytes += R * embedding_row_bytes();
    for (const LayerW& l : I.layers) {
        if (opts.capture_layer_inputs) out.layer_inputs.push_back(xbuf);
        if (l.kind == LayerKind::FullAttention) {
            I.decoder_layer(l, x, [&](ConstRows xn, Rows y) { I.attention(*l.attn, l.kind_index, xn, seqs, y, cost); }, cost);
        } else {
            I.decoder_layer(l, x, [&](ConstRows xn, Rows y) { I.gdn(*l.gdn, l.kind_index, xn, seqs, gseqs, y, cost); }, cost);
        }
    }
    for (std::size_t si = 0; si < steps.size(); ++si) {
        steps[si].kv->commit(steps[si].tokens.size());
        steps[si].gdn->mark_slots_written(steps[si].tokens.size(), steps[si].n_state_slots);
    }

    // ---- head ------------------------------------------------------------------------------
    std::vector<float> hbuf(R * E);
    I.rms_norm(x, I.output_norm, Rows(std::span(hbuf), R, E), cost);
    const ConstRows h(std::span<const float>(hbuf), R, E);
    std::vector<Impl::HeadReq> reqs;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const SeqStep& s = steps[si];
        SeqOutput& o = out.seqs[si];
        if (s.want_hidden) o.hidden.assign(&hbuf[seqs[si].r0 * E], &hbuf[(seqs[si].r0 + s.tokens.size()) * E]);
        if (s.logits == LogitsMode::None) continue;
        o.argmax.resize(s.logit_rows.size());
        if (s.logits == LogitsMode::Full) o.logits.resize(s.logit_rows.size() * I.n_vocab);
        for (std::size_t j = 0; j < s.logit_rows.size(); ++j) {
            reqs.push_back({seqs[si].r0 + s.logit_rows[j], &o, j, s.logits == LogitsMode::Full});
        }
    }
    I.head(I.head_ref, I.head_bytes, h, reqs, cost);
}

void Qwen35::mtp_forward(std::span<const MtpStep> steps, StepResult& out) const {
    const Impl& I = *impl_;
    HALO_CHECK(I.mtp.has_value(), ErrorCode::Unsupported, "mtp_forward: the model has no MTP block");
    const MtpW& M = *I.mtp;
    const kv_cache::KvLayout want_kv = mtp_kv_layout();
    const std::size_t E = I.E;
    std::vector<SeqRows> seqs(steps.size());
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
        for (std::size_t sj = 0; sj < si; ++sj) {
            HALO_CHECK(steps[sj].kv != s.kv, ErrorCode::Api, "mtp_forward: steps {} and {} share a KV cache", sj, si);
        }
        check_rows(s.logit_rows, s.tokens.size(), "mtp_forward", si);
        seqs[si] = {R, s.tokens.size(), s.kv, s.first_position};
        R += s.tokens.size();
    }
    out.seqs.assign(steps.size(), {});
    out.cost = {};
    out.layer_inputs.clear();
    out.gdn_paths.clear();
    if (steps.empty()) return;
    for (const MtpStep& s : steps) s.kv->reserve(s.tokens.size());

    StepCost& cost = out.cost;
    cost.weight_passes = 1;
    // concat(rmsnorm(embed(tok); enorm), rmsnorm(h; hnorm)) — embedding first (D-005).
    std::vector<float> e(R * E), hh(R * E), cat(R * 2 * E), xbuf(R * E);
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const MtpStep& s = steps[si];
        for (std::size_t i = 0; i < s.tokens.size(); ++i) {
            embedding_row(M.embedding, s.tokens[i], std::span(&e[(seqs[si].r0 + i) * E], E));
        }
        std::copy(s.hidden.begin(), s.hidden.end(), &hh[seqs[si].r0 * E]);
    }
    cost.embedding_bytes += R * tensor::row_bytes(M.embedding.type(), static_cast<std::int64_t>(E));
    I.rms_norm(ConstRows(std::span<const float>(e), R, E), M.enorm, Rows(std::span(e), R, E), cost);
    I.rms_norm(ConstRows(std::span<const float>(hh), R, E), M.hnorm, Rows(std::span(hh), R, E), cost);
    for (std::size_t r = 0; r < R; ++r) {
        std::copy_n(&e[r * E], E, &cat[r * 2 * E]);
        std::copy_n(&hh[r * E], E, &cat[r * 2 * E + E]);
    }
    const Rows x(std::span(xbuf), R, E);
    I.matmul(ConstRows(std::span<const float>(cat), R, 2 * E), M.eh_proj, x, cost);
    I.decoder_layer(M.block, x, [&](ConstRows xn, Rows y) { I.attention(*M.block.attn, 0, xn, seqs, y, cost); }, cost);
    for (const MtpStep& s : steps) s.kv->commit(s.tokens.size());

    std::vector<float> hbuf(R * E);
    I.rms_norm(x, M.head_norm, Rows(std::span(hbuf), R, E), cost);
    const ConstRows h(std::span<const float>(hbuf), R, E);
    std::vector<Impl::HeadReq> reqs;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const MtpStep& s = steps[si];
        SeqOutput& o = out.seqs[si];
        if (s.want_hidden) o.hidden.assign(&hbuf[seqs[si].r0 * E], &hbuf[(seqs[si].r0 + s.tokens.size()) * E]);
        if (s.logits == LogitsMode::None) continue;
        o.argmax.resize(s.logit_rows.size());
        if (s.logits == LogitsMode::Full) o.logits.resize(s.logit_rows.size() * I.n_vocab);
        for (std::size_t j = 0; j < s.logit_rows.size(); ++j) {
            reqs.push_back({seqs[si].r0 + s.logit_rows[j], &o, j, s.logits == LogitsMode::Full});
        }
    }
    I.head(M.head_ref, M.head_ref.n_bytes(), h, reqs, cost);
}

}  // namespace halo::models
