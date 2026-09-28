// GATED_DELTANET: recurrent and chunked gated delta rule (D-003, D-004 items 5-6).
//
// State layout: state[j][i][c] = S_j[i][c], j = value head, i < d_k, c < d_v (d_v fastest).
// Both forms parallelize over value heads only; each head is computed start-to-finish by
// one thread with fixed-order arithmetic, so results do not depend on the thread count.

#include <algorithm>
#include <cmath>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "kernel_common.h"

namespace halo::cpu {

namespace {

constexpr float kQkL2Eps = 1e-6f;  // HF use_qk_l2norm_in_kernel eps (fixed)

void validate(const GdnDims& d, const GdnInputs& in, std::span<const float> state, ConstRows out,
              const char* op) {
    HALO_CHECK(d.n_k_heads > 0 && d.n_v_heads > 0 && d.d_k > 0 && d.d_v > 0, ErrorCode::Kernel,
               "{}: zero dimension (n_k {}, n_v {}, d_k {}, d_v {})", op, d.n_k_heads, d.n_v_heads,
               d.d_k, d.d_v);
    HALO_CHECK(d.n_v_heads % d.n_k_heads == 0, ErrorCode::Kernel,
               "{}: n_v_heads {} is not a multiple of n_k_heads {}", op, d.n_v_heads, d.n_k_heads);
    const std::size_t t = in.q.rows();
    const std::size_t qk_cols = detail::checked_mul(d.n_k_heads, d.d_k, op);
    const std::size_t v_cols = detail::checked_mul(d.n_v_heads, d.d_v, op);
    auto shape = [&](ConstRows r, std::size_t cols, const char* what) {
        HALO_CHECK(r.rows() == t && r.cols() == cols, ErrorCode::Kernel, "{}: {} is {}x{}, expected {}x{}",
                   op, what, r.rows(), r.cols(), t, cols);
    };
    shape(in.k, qk_cols, "k");
    shape(in.q, qk_cols, "q");
    shape(in.v, v_cols, "v");
    shape(in.g, d.n_v_heads, "g");
    shape(in.beta, d.n_v_heads, "beta");
    shape(out, v_cols, "out");
    const std::size_t state_n = detail::checked_mul(detail::checked_mul(d.n_v_heads, d.d_k, op), d.d_v, op);
    HALO_CHECK(state.size() == state_n, ErrorCode::Kernel, "{}: state has {} floats, expected {}", op,
               state.size(), state_n);
    const ConstRows st(state);
    for (const auto& [r, what] : {std::pair{in.q, "q"}, std::pair{in.k, "k"}, std::pair{in.v, "v"},
                                  std::pair{in.g, "g"}, std::pair{in.beta, "beta"}, std::pair{st, "state"}}) {
        detail::check_disjoint(r, out, op, what);
    }
    for (const auto& [r, what] : {std::pair{in.q, "q"}, std::pair{in.k, "k"}, std::pair{in.v, "v"},
                                  std::pair{in.g, "g"}, std::pair{in.beta, "beta"}}) {
        detail::check_disjoint(r, st, op, what);
    }
}

/// Validates the optional slot buffer; returns the slot count.
std::size_t validate_slots(const GdnInputs& in, std::span<const float> state, ConstRows out,
                           std::span<const float> slots, const char* op) {
    if (slots.empty()) return 0;
    const std::size_t state_n = state.size();
    HALO_CHECK(slots.size() % state_n == 0, ErrorCode::Kernel,
               "{}: state_slots has {} floats, not a multiple of the state size {}", op, slots.size(), state_n);
    const ConstRows sl(slots);
    const ConstRows st(state);
    for (const auto& [r, what] : {std::pair{in.q, "q"}, std::pair{in.k, "k"}, std::pair{in.v, "v"},
                                  std::pair{in.g, "g"}, std::pair{in.beta, "beta"}, std::pair{st, "state"},
                                  std::pair{out, "out"}}) {
        HALO_CHECK(!detail::overlaps(r, sl), ErrorCode::Kernel, "{}: state_slots overlaps {}", op, what);
    }
    return slots.size() / state_n;
}

/// dst = (optionally l2-normalized) src, then * scale.
void prep_qk(const float* src, std::size_t n, bool l2, float scale, float* dst) noexcept {
    float inv = 1.0f;
    if (l2) inv = 1.0f / std::sqrt(detail::sum_squares(src, n) + kQkL2Eps);
    for (std::size_t i = 0; i < n; ++i) dst[i] = (l2 ? src[i] * inv : src[i]) * scale;
}

/// Ring validation (ADR-001 §5.3): P >= 2, live < P, the slab holds exactly P states,
/// n_slots <= P - 1, and the whole slab is disjoint from every input and from out. Returns
/// the number of slots this call writes, min(T, n_slots).
std::size_t validate_ring(const GdnDims& d, const GdnInputs& in, const StateRingView& ring, ConstRows out,
                          const char* op) {
    const std::size_t state_n = detail::checked_mul(detail::checked_mul(d.n_v_heads, d.d_k, op), d.d_v, op);
    HALO_CHECK(ring.P >= 2, ErrorCode::Kernel, "{}: state ring needs P >= 2, got {}", op, ring.P);
    HALO_CHECK(ring.live < ring.P, ErrorCode::Kernel, "{}: ring live {} outside P {}", op, ring.live, ring.P);
    HALO_CHECK(ring.slab.size() == detail::checked_mul(ring.P, state_n, op), ErrorCode::Kernel,
               "{}: ring slab has {} floats, expected {} x {} states", op, ring.slab.size(), ring.P, state_n);
    HALO_CHECK(ring.n_slots <= ring.P - 1, ErrorCode::Kernel, "{}: {} ring slots requested, P - 1 = {}", op,
               ring.n_slots, ring.P - 1);
    // validate() over the live state: shapes plus its disjointness from inputs and out.
    validate(d, in, ring.slab.subspan(ring.live * state_n, state_n), out, op);
    const ConstRows sl(ring.slab);
    for (const auto& [r, what] : {std::pair{in.q, "q"}, std::pair{in.k, "k"}, std::pair{in.v, "v"},
                                  std::pair{in.g, "g"}, std::pair{in.beta, "beta"}, std::pair{out, "out"}}) {
        HALO_CHECK(!detail::overlaps(r, sl), ErrorCode::Kernel, "{}: ring slab overlaps {}", op, what);
    }
    return std::min(in.q.rows(), ring.n_slots);
}

}  // namespace

float gdn_q_scale(const GdnQkParams& qk, std::size_t d_k) {
    HALO_CHECK(d_k > 0, ErrorCode::Kernel, "gdn_q_scale: d_k == 0");
    const float s = qk.q_scale.value_or(1.0f / std::sqrt(static_cast<float>(d_k)));
    HALO_CHECK(std::isfinite(s), ErrorCode::Kernel, "gdn: q_scale {} is not finite", s);
    return s;
}

std::size_t gdn_k_head(const GdnDims& dims, std::size_t v_head) {
    HALO_CHECK(dims.n_k_heads > 0 && dims.n_v_heads % dims.n_k_heads == 0 && v_head < dims.n_v_heads,
               ErrorCode::Kernel, "gdn_k_head: bad head {} for n_v {} / n_k {}", v_head, dims.n_v_heads,
               dims.n_k_heads);
    if (dims.mapping == GdnHeadMapping::Tiled) return v_head % dims.n_k_heads;
    return v_head / (dims.n_v_heads / dims.n_k_heads);
}

void gated_delta_rule_recurrent(const GdnDims& dims, const GdnInputs& in, std::span<float> state,
                                Rows out, const GdnQkParams& qk, ThreadPool* pool, std::span<float> state_slots) {
    constexpr const char* kOp = "gated_delta_rule_recurrent";
    validate(dims, in, state, out, kOp);
    const std::size_t n_slots = validate_slots(in, state, out, state_slots, kOp);
    const std::size_t dk = dims.d_k;
    const std::size_t dv = dims.d_v;
    const std::size_t n_tok = in.q.rows();
    const std::size_t state_n = state.size();
    const float scale = gdn_q_scale(qk, dk);
    const bool qk_l2norm = qk.qk_l2norm;

    parallel_for(pool, dims.n_v_heads, [&](std::size_t h0, std::size_t h1) {
        std::vector<float> qn(dk), kn(dk), kv(dv), delta(dv);
        for (std::size_t j = h0; j < h1; ++j) {
            const std::size_t kh = gdn_k_head(dims, j);
            float* s = state.data() + j * dk * dv;
            for (std::size_t t = 0; t < n_tok; ++t) {
                prep_qk(in.q.row_ptr(t) + kh * dk, dk, qk_l2norm, scale, qn.data());
                prep_qk(in.k.row_ptr(t) + kh * dk, dk, qk_l2norm, 1.0f, kn.data());
                const float* vt = in.v.row_ptr(t) + j * dv;
                const float decay = std::exp(in.g(t, j));
                const float beta = in.beta(t, j);
                // S <- S * exp(g); kv = S^T k
                std::fill(kv.begin(), kv.end(), 0.0f);
                for (std::size_t i = 0; i < dk; ++i) {
                    float* row = s + i * dv;
                    const float ki = kn[i];
                    for (std::size_t c = 0; c < dv; ++c) {
                        row[c] *= decay;
                        kv[c] += row[c] * ki;
                    }
                }
                for (std::size_t c = 0; c < dv; ++c) delta[c] = (vt[c] - kv[c]) * beta;
                // S <- S + k delta^T; o = S^T q
                float* o = out.row_ptr(t) + j * dv;
                std::fill(o, o + dv, 0.0f);
                for (std::size_t i = 0; i < dk; ++i) {
                    float* row = s + i * dv;
                    const float ki = kn[i];
                    const float qi = qn[i];
                    for (std::size_t c = 0; c < dv; ++c) {
                        row[c] += ki * delta[c];
                        o[c] += row[c] * qi;
                    }
                }
                const std::size_t slot = n_tok - 1 - t;  // slot 0 = most recent row
                if (slot < n_slots) {
                    std::copy_n(s, dk * dv, state_slots.data() + slot * state_n + j * dk * dv);
                }
            }
        }
    });
}

void gated_delta_rule_chunked(const GdnDims& dims, const GdnInputs& in, std::span<float> state, Rows out,
                              const GdnQkParams& qk, std::size_t chunk_size, ThreadPool* pool,
                              std::span<float> state_slots) {
    constexpr const char* kOp = "gated_delta_rule_chunked";
    validate(dims, in, state, out, kOp);
    const std::size_t n_slots = validate_slots(in, state, out, state_slots, kOp);
    const std::size_t state_n = state.size();
    HALO_CHECK(chunk_size >= 1 && chunk_size <= 1024, ErrorCode::Kernel, "{}: chunk_size {} not in [1, 1024]",
               kOp, chunk_size);
    const std::size_t n_tok = in.q.rows();
    for (std::size_t t = 0; t < n_tok; ++t) {
        for (std::size_t j = 0; j < dims.n_v_heads; ++j) {
            HALO_CHECK(in.g(t, j) <= 0.0f, ErrorCode::Kernel, "{}: g[{}][{}] = {} must be <= 0", kOp, t, j,
                       in.g(t, j));
        }
    }
    const std::size_t dk = dims.d_k;
    const std::size_t dv = dims.d_v;
    const std::size_t cs = chunk_size;
    const float scale = gdn_q_scale(qk, dk);
    const bool qk_l2norm = qk.qk_l2norm;

    parallel_for(pool, dims.n_v_heads, [&](std::size_t h0, std::size_t h1) {
        // Per-chunk scratch, row-major. Names follow torch_chunk_gated_delta_rule.
        std::vector<float> q(cs * dk), k(cs * dk), kb(cs * dk), vb(cs * dv), gc(cs);
        std::vector<float> ut(cs * cs), attn(cs * cs);       // strictly lower / lower-incl-diag
        std::vector<float> nv(cs * dv), kcd(cs * dk);        // new_values, k_cumdecay
        std::vector<float> vnew(cs * dv), upd(dk * dv), tmp(std::max(dk, dv));
        for (std::size_t j = h0; j < h1; ++j) {
            const std::size_t kh = gdn_k_head(dims, j);
            float* s = state.data() + j * dk * dv;
            for (std::size_t t0 = 0; t0 < n_tok; t0 += cs) {
                const std::size_t len = std::min(cs, n_tok - t0);
                // Inputs of this chunk; gc = cumulative log decay inside the chunk.
                float cum = 0.0f;
                for (std::size_t a = 0; a < len; ++a) {
                    const std::size_t t = t0 + a;
                    prep_qk(in.q.row_ptr(t) + kh * dk, dk, qk_l2norm, scale, &q[a * dk]);
                    prep_qk(in.k.row_ptr(t) + kh * dk, dk, qk_l2norm, 1.0f, &k[a * dk]);
                    const float beta = in.beta(t, j);
                    const float* vt = in.v.row_ptr(t) + j * dv;
                    for (std::size_t i = 0; i < dk; ++i) kb[a * dk + i] = k[a * dk + i] * beta;
                    for (std::size_t c = 0; c < dv; ++c) vb[a * dv + c] = vt[c] * beta;
                    cum += in.g(t, j);
                    gc[a] = cum;
                }
                // ut[a][b] = (k_beta_a . k_b) * exp(gc_a - gc_b), b < a;
                // attn[a][b] = (q_a . k_b) * exp(gc_a - gc_b), b <= a.
                for (std::size_t a = 0; a < len; ++a) {
                    for (std::size_t b = 0; b <= a; ++b) {
                        const float dec = std::exp(gc[a] - gc[b]);
                        if (b < a) ut[a * cs + b] = detail::dot(&kb[a * dk], &k[b * dk], dk) * dec;
                        attn[a * cs + b] = detail::dot(&q[a * dk], &k[b * dk], dk) * dec;
                    }
                }
                // Solve (I + ut) X = B by forward substitution (unit lower triangular):
                //   new_values = X for B = v_beta; k_cumdecay = X for B = k_beta * exp(gc).
                for (std::size_t a = 0; a < len; ++a) {
                    float* nva = &nv[a * dv];
                    float* kca = &kcd[a * dk];
                    const float ea = std::exp(gc[a]);
                    for (std::size_t c = 0; c < dv; ++c) nva[c] = vb[a * dv + c];
                    for (std::size_t i = 0; i < dk; ++i) kca[i] = kb[a * dk + i] * ea;
                    for (std::size_t b = 0; b < a; ++b) {
                        const float m = ut[a * cs + b];
                        const float* nvb = &nv[b * dv];
                        const float* kcb = &kcd[b * dk];
                        for (std::size_t c = 0; c < dv; ++c) nva[c] -= m * nvb[c];
                        for (std::size_t i = 0; i < dk; ++i) kca[i] -= m * kcb[i];
                    }
                }
                // v_new = new_values - k_cumdecay @ S
                for (std::size_t a = 0; a < len; ++a) {
                    float* vn = &vnew[a * dv];
                    std::fill(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(dv), 0.0f);
                    for (std::size_t i = 0; i < dk; ++i) {
                        const float kc = kcd[a * dk + i];
                        const float* row = s + i * dv;
                        for (std::size_t c = 0; c < dv; ++c) tmp[c] += kc * row[c];
                    }
                    for (std::size_t c = 0; c < dv; ++c) vn[c] = nv[a * dv + c] - tmp[c];
                }
                // out_a = (q_a * exp(gc_a)) @ S + sum_{b<=a} attn[a][b] * v_new_b
                for (std::size_t a = 0; a < len; ++a) {
                    float* o = out.row_ptr(t0 + a) + j * dv;
                    const float ea = std::exp(gc[a]);
                    std::fill(o, o + dv, 0.0f);
                    for (std::size_t i = 0; i < dk; ++i) {
                        const float qi = q[a * dk + i] * ea;
                        const float* row = s + i * dv;
                        for (std::size_t c = 0; c < dv; ++c) o[c] += qi * row[c];
                    }
                    std::fill(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(dv), 0.0f);
                    for (std::size_t b = 0; b <= a; ++b) {
                        const float w = attn[a * cs + b];
                        const float* vn = &vnew[b * dv];
                        for (std::size_t c = 0; c < dv; ++c) tmp[c] += w * vn[c];
                    }
                    for (std::size_t c = 0; c < dv; ++c) o[c] += tmp[c];
                }
                // State after chunk row a (a = len-1: the chunk-end update):
                //   S_a = S * exp(gc_a) + sum_{b<=a} (k_b * exp(gc_a - gc_b))^T v_new_b.
                // dst may be s itself (element-wise read-then-write).
                auto state_at = [&](std::size_t a, float* dst) {
                    const float ga = gc[a];
                    std::fill(upd.begin(), upd.end(), 0.0f);
                    for (std::size_t b = 0; b <= a; ++b) {
                        const float w = std::exp(ga - gc[b]);
                        const float* vn = &vnew[b * dv];
                        for (std::size_t i = 0; i < dk; ++i) {
                            const float ki = k[b * dk + i] * w;
                            float* u = &upd[i * dv];
                            for (std::size_t c = 0; c < dv; ++c) u[c] += ki * vn[c];
                        }
                    }
                    const float decay = std::exp(ga);
                    for (std::size_t e = 0; e < dk * dv; ++e) dst[e] = s[e] * decay + upd[e];
                };
                // Rollback slots for rows of this chunk (before s is overwritten).
                for (std::size_t a = 0; a < len; ++a) {
                    const std::size_t slot = n_tok - 1 - (t0 + a);
                    if (slot < n_slots) state_at(a, state_slots.data() + slot * state_n + j * dk * dv);
                }
                state_at(len - 1, s);
            }
        }
    });
}

void gated_delta_rule_recurrent(const GdnDims& dims, const GdnInputs& in, const StateRingView& ring, Rows out,
                                const GdnQkParams& qk, ThreadPool* pool) {
    constexpr const char* kOp = "gated_delta_rule_recurrent (ring)";
    const std::size_t n_slots = validate_ring(dims, in, ring, out, kOp);
    const std::size_t dk = dims.d_k;
    const std::size_t dv = dims.d_v;
    const std::size_t n_tok = in.q.rows();
    const std::size_t state_n = ring.slab.size() / ring.P;
    const float scale = gdn_q_scale(qk, dk);
    const bool qk_l2norm = qk.qk_l2norm;
    // Ring placement (ADR-001 §5.3): read slab[live]; final state -> slab[(live+1) mod P];
    // logical slot s -> slab[(live+1+s) mod P].
    const float* sin = ring.slab.data() + ring.live * state_n;
    float* sout = ring.slab.data() + (ring.live + 1) % ring.P * state_n;
    const auto slot_at = [&](std::size_t s) { return ring.slab.data() + (ring.live + 1 + s) % ring.P * state_n; };

    // The per-row arithmetic is exactly the in-place form's: row 0 reads the input state,
    // later rows read the (already updated) output state; every write goes to the output
    // state. s_d = round(s*decay), then round(s_d + round(k*delta)) element-wise, as there.
    parallel_for(pool, dims.n_v_heads, [&](std::size_t h0, std::size_t h1) {
        std::vector<float> qn(dk), kn(dk), kv(dv), delta(dv);
        for (std::size_t j = h0; j < h1; ++j) {
            const std::size_t kh = gdn_k_head(dims, j);
            const float* sr = sin + j * dk * dv;
            float* sw = sout + j * dk * dv;
            for (std::size_t t = 0; t < n_tok; ++t) {
                prep_qk(in.q.row_ptr(t) + kh * dk, dk, qk_l2norm, scale, qn.data());
                prep_qk(in.k.row_ptr(t) + kh * dk, dk, qk_l2norm, 1.0f, kn.data());
                const float* vt = in.v.row_ptr(t) + j * dv;
                const float decay = std::exp(in.g(t, j));
                const float beta = in.beta(t, j);
                const float* s = t == 0 ? sr : sw;
                // kv = (S * exp(g))^T k
                std::fill(kv.begin(), kv.end(), 0.0f);
                for (std::size_t i = 0; i < dk; ++i) {
                    const float* row = s + i * dv;
                    const float ki = kn[i];
                    for (std::size_t c = 0; c < dv; ++c) {
                        const float sd = row[c] * decay;
                        kv[c] += sd * ki;
                    }
                }
                for (std::size_t c = 0; c < dv; ++c) delta[c] = (vt[c] - kv[c]) * beta;
                // S <- S * exp(g) + k delta^T; o = S^T q
                float* o = out.row_ptr(t) + j * dv;
                std::fill(o, o + dv, 0.0f);
                for (std::size_t i = 0; i < dk; ++i) {
                    const float* row = s + i * dv;
                    float* wrow = sw + i * dv;
                    const float ki = kn[i];
                    const float qi = qn[i];
                    for (std::size_t c = 0; c < dv; ++c) {
                        const float sv = row[c] * decay + ki * delta[c];
                        wrow[c] = sv;
                        o[c] += sv * qi;
                    }
                }
                const std::size_t slot = n_tok - 1 - t;  // slot 0 = most recent row
                if (slot < n_slots) {
                    std::copy_n(sw, dk * dv, slot_at(slot) + j * dk * dv);
                }
            }
        }
    });
}

void gated_delta_rule_chunked(const GdnDims& dims, const GdnInputs& in, const StateRingView& ring, Rows out,
                              const GdnQkParams& qk, std::size_t chunk_size, ThreadPool* pool) {
    constexpr const char* kOp = "gated_delta_rule_chunked (ring)";
    const std::size_t n_slots = validate_ring(dims, in, ring, out, kOp);
    const std::size_t state_n = ring.slab.size() / ring.P;
    HALO_CHECK(chunk_size >= 1 && chunk_size <= 1024, ErrorCode::Kernel, "{}: chunk_size {} not in [1, 1024]",
               kOp, chunk_size);
    const std::size_t n_tok = in.q.rows();
    for (std::size_t t = 0; t < n_tok; ++t) {
        for (std::size_t j = 0; j < dims.n_v_heads; ++j) {
            HALO_CHECK(in.g(t, j) <= 0.0f, ErrorCode::Kernel, "{}: g[{}][{}] = {} must be <= 0", kOp, t, j,
                       in.g(t, j));
        }
    }
    const std::size_t dk = dims.d_k;
    const std::size_t dv = dims.d_v;
    const std::size_t cs = chunk_size;
    const float scale = gdn_q_scale(qk, dk);
    const bool qk_l2norm = qk.qk_l2norm;
    const float* sin = ring.slab.data() + ring.live * state_n;
    float* sout = ring.slab.data() + (ring.live + 1) % ring.P * state_n;
    const auto slot_at = [&](std::size_t s) { return ring.slab.data() + (ring.live + 1 + s) % ring.P * state_n; };

    parallel_for(pool, dims.n_v_heads, [&](std::size_t h0, std::size_t h1) {
        // Per-chunk scratch, row-major. Names follow torch_chunk_gated_delta_rule.
        std::vector<float> q(cs * dk), k(cs * dk), kb(cs * dk), vb(cs * dv), gc(cs);
        std::vector<float> ut(cs * cs), attn(cs * cs);       // strictly lower / lower-incl-diag
        std::vector<float> nv(cs * dv), kcd(cs * dk);        // new_values, k_cumdecay
        std::vector<float> vnew(cs * dv), upd(dk * dv), tmp(std::max(dk, dv));
        for (std::size_t j = h0; j < h1; ++j) {
            const std::size_t kh = gdn_k_head(dims, j);
            const float* sr = sin + j * dk * dv;
            float* sw = sout + j * dk * dv;
            for (std::size_t t0 = 0; t0 < n_tok; t0 += cs) {
                const std::size_t len = std::min(cs, n_tok - t0);
                // Inputs of this chunk; gc = cumulative log decay inside the chunk.
                float cum = 0.0f;
                for (std::size_t a = 0; a < len; ++a) {
                    const std::size_t t = t0 + a;
                    prep_qk(in.q.row_ptr(t) + kh * dk, dk, qk_l2norm, scale, &q[a * dk]);
                    prep_qk(in.k.row_ptr(t) + kh * dk, dk, qk_l2norm, 1.0f, &k[a * dk]);
                    const float beta = in.beta(t, j);
                    const float* vt = in.v.row_ptr(t) + j * dv;
                    for (std::size_t i = 0; i < dk; ++i) kb[a * dk + i] = k[a * dk + i] * beta;
                    for (std::size_t c = 0; c < dv; ++c) vb[a * dv + c] = vt[c] * beta;
                    cum += in.g(t, j);
                    gc[a] = cum;
                }
                // ut[a][b] = (k_beta_a . k_b) * exp(gc_a - gc_b), b < a;
                // attn[a][b] = (q_a . k_b) * exp(gc_a - gc_b), b <= a.
                for (std::size_t a = 0; a < len; ++a) {
                    for (std::size_t b = 0; b <= a; ++b) {
                        const float dec = std::exp(gc[a] - gc[b]);
                        if (b < a) ut[a * cs + b] = detail::dot(&kb[a * dk], &k[b * dk], dk) * dec;
                        attn[a * cs + b] = detail::dot(&q[a * dk], &k[b * dk], dk) * dec;
                    }
                }
                // The chunk's starting state: the input state for the first chunk, the output
                // state (already updated by previous chunks) afterwards.
                const float* s = t0 == 0 ? sr : sw;
                // Solve (I + ut) X = B by forward substitution (unit lower triangular):
                //   new_values = X for B = v_beta; k_cumdecay = X for B = k_beta * exp(gc).
                for (std::size_t a = 0; a < len; ++a) {
                    float* nva = &nv[a * dv];
                    float* kca = &kcd[a * dk];
                    const float ea = std::exp(gc[a]);
                    for (std::size_t c = 0; c < dv; ++c) nva[c] = vb[a * dv + c];
                    for (std::size_t i = 0; i < dk; ++i) kca[i] = kb[a * dk + i] * ea;
                    for (std::size_t b = 0; b < a; ++b) {
                        const float m = ut[a * cs + b];
                        const float* nvb = &nv[b * dv];
                        const float* kcb = &kcd[b * dk];
                        for (std::size_t c = 0; c < dv; ++c) nva[c] -= m * nvb[c];
                        for (std::size_t i = 0; i < dk; ++i) kca[i] -= m * kcb[i];
                    }
                }
                // v_new = new_values - k_cumdecay @ S
                for (std::size_t a = 0; a < len; ++a) {
                    float* vn = &vnew[a * dv];
                    std::fill(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(dv), 0.0f);
                    for (std::size_t i = 0; i < dk; ++i) {
                        const float kc = kcd[a * dk + i];
                        const float* row = s + i * dv;
                        for (std::size_t c = 0; c < dv; ++c) tmp[c] += kc * row[c];
                    }
                    for (std::size_t c = 0; c < dv; ++c) vn[c] = nv[a * dv + c] - tmp[c];
                }
                // out_a = (q_a * exp(gc_a)) @ S + sum_{b<=a} attn[a][b] * v_new_b
                for (std::size_t a = 0; a < len; ++a) {
                    float* o = out.row_ptr(t0 + a) + j * dv;
                    const float ea = std::exp(gc[a]);
                    std::fill(o, o + dv, 0.0f);
                    for (std::size_t i = 0; i < dk; ++i) {
                        const float qi = q[a * dk + i] * ea;
                        const float* row = s + i * dv;
                        for (std::size_t c = 0; c < dv; ++c) o[c] += qi * row[c];
                    }
                    std::fill(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(dv), 0.0f);
                    for (std::size_t b = 0; b <= a; ++b) {
                        const float w = attn[a * cs + b];
                        const float* vn = &vnew[b * dv];
                        for (std::size_t c = 0; c < dv; ++c) tmp[c] += w * vn[c];
                    }
                    for (std::size_t c = 0; c < dv; ++c) o[c] += tmp[c];
                }
                // State after chunk row a (a = len-1: the chunk-end update):
                //   S_a = S * exp(gc_a) + sum_{b<=a} (k_b * exp(gc_a - gc_b))^T v_new_b.
                auto state_at = [&](std::size_t a, float* dst) {
                    const float ga = gc[a];
                    std::fill(upd.begin(), upd.end(), 0.0f);
                    for (std::size_t b = 0; b <= a; ++b) {
                        const float w = std::exp(ga - gc[b]);
                        const float* vn = &vnew[b * dv];
                        for (std::size_t i = 0; i < dk; ++i) {
                            const float ki = k[b * dk + i] * w;
                            float* u = &upd[i * dv];
                            for (std::size_t c = 0; c < dv; ++c) u[c] += ki * vn[c];
                        }
                    }
                    const float decay = std::exp(ga);
                    for (std::size_t e = 0; e < dk * dv; ++e) dst[e] = s[e] * decay + upd[e];
                };
                // Rollback slots for rows of this chunk (before the state is overwritten).
                // Slot 0 is the state after the last row overall — exactly what the final
                // chunk's chunk-end update writes into the output state, which IS slot 0's
                // ring target; writing it here too would clobber the chunk-start state the
                // chunk-end update still reads (they alias once s == the output region).
                for (std::size_t a = 0; a < len; ++a) {
                    const std::size_t slot = n_tok - 1 - (t0 + a);
                    if (slot > 0 && slot < n_slots) state_at(a, slot_at(slot) + j * dk * dv);
                }
                state_at(len - 1, sw);
            }
        }
    });
}

}  // namespace halo::cpu
