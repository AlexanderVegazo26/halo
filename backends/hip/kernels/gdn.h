#pragma once
// GATED_DELTANET kernel bodies (DECISIONS D-003, D-004 items 5-6, D-012 state slots, D-016
// q/k contract). Shared by the device kernels (src/launch_gdn.hip) and the host emulation
// (src/emulate.cpp); see hd.h for the execution model.
//
// Every expression below reproduces backends/cpu/gated_delta_rule.cpp operation by
// operation (same operands, same association, same summation order, fp32 throughout, no
// contraction), so the host emulation is bit-identical to halo::cpu and the device differs
// only by its transcendental functions (expf) — see docs/hip.md.
//
// State layout (identical to the CPU op): state[j][i][c], j = value head, i < d_k (key
// dim), c < d_v (value dim, fastest). Thread mapping in every kernel that touches the state:
// one thread per value column c of one head, holding S[:, c] (d_k floats) in registers.

#include "hd.h"

namespace halo::hip::kern {

inline constexpr unsigned kGdnMaxDk = 128;     // register-resident state column
inline constexpr unsigned kGdnMaxChunk = 64;   // LDS tiles of the chunked form
inline constexpr float kGdnQkEps = 1e-6f;      // HF use_qk_l2norm_in_kernel eps (fixed)

/// Per-token inputs; strides are in elements between consecutive tokens.
struct GdnIn {
    const float* q = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    const float* g = nullptr;
    const float* beta = nullptr;
    std::uint64_t q_stride = 0;
    std::uint64_t k_stride = 0;
    std::uint64_t v_stride = 0;
    std::uint64_t g_stride = 0;
    std::uint64_t beta_stride = 0;
};

struct GdnDimsK {
    unsigned n_k = 0;
    unsigned n_v = 0;
    unsigned dk = 0;
    unsigned dv = 0;
    unsigned grouped = 0;  // 0 = GGUF tiled (j % n_k), 1 = HF grouped (j / (n_v / n_k))
};

HALO_HD inline unsigned gdn_k_head(const GdnDimsK& d, unsigned j) {
    return d.grouped != 0u ? j / (d.n_v / d.n_k) : j % d.n_k;
}

/// 1 / sqrt(sum(x^2) + eps) with the CPU reference's dot order (cpu prep_qk).
HALO_HD inline float gdn_inv_l2(const float* x, unsigned n) {
    const At a{x};
    return 1.0f / hsqrt(dot8(a, a, n) + kGdnQkEps);
}

// =======================================================================================
// Recurrent form: grid (n_v, ceil(d_v / block)), block threads = value columns.
// =======================================================================================

struct GdnRecParams {
    GdnIn in;
    GdnDimsK dims;
    unsigned n_tok = 0;
    const float* state_in = nullptr;
    float* state_out = nullptr;  // may equal state_in (in place)
    float* slots = nullptr;      // n_slots states back to back, or null; ring: the slab base
    unsigned n_slots = 0;
    float* out = nullptr;
    std::uint64_t out_stride = 0;
    unsigned l2 = 1;
    float q_scale = 1.0f;
    unsigned ring_p = 0;     ///< ADR-001 §5.3: slab size (0 = contiguous slots region)
    unsigned ring_live = 0;  ///< the slab slot read this call
};

struct GdnRecRegs {
    float s[kGdnMaxDk];
};

struct GdnRecShared {
    float qn[kGdnMaxDk];
    float kn[kGdnMaxDk];
    float inv_q;
    float inv_k;
    float decay;
    float beta;
};

// The state column lives in kGdnMaxDk registers. Rows i >= d_k are zero and every loop over
// i runs to kGdnMaxDk without a guard (a guarded loop body makes the compiler spill the
// column to scratch; measured). The padded rows contribute exactly +0 to every sum as long
// as the values are finite (0*x = ±0, and a sum that starts at +0 stays unchanged by ±0
// terms), so results are identical to the unpadded CPU loops; only loads and stores are
// guarded.
HALO_HD inline void gdn_load_column(GdnRecRegs& r, const float* col, unsigned dk, unsigned dv) {
#pragma unroll
    for (unsigned i = 0; i < kGdnMaxDk; ++i) r.s[i] = i < dk ? col[static_cast<std::uint64_t>(i) * dv] : 0.0f;
}

HALO_HD inline void gdn_store_column(const GdnRecRegs& r, float* col, unsigned dk, unsigned dv) {
#pragma unroll
    for (unsigned i = 0; i < kGdnMaxDk; ++i) {
        if (i < dk) col[static_cast<std::uint64_t>(i) * dv] = r.s[i];
    }
}

inline Launch gdn_recurrent_launch(const GdnDimsK& d, unsigned block) {
    return Launch{d.n_v, (d.dv + block - 1) / block, block};
}

template <class Exec>
HALO_HD void gdn_recurrent_body(Exec& ex, GdnRecShared& sh, const GdnRecParams& p, unsigned bx,
                                unsigned by) {
    const GdnDimsK& d = p.dims;
    const unsigned j = bx;
    const unsigned kh = gdn_k_head(d, j);
    const unsigned dk = d.dk;
    const unsigned dv = d.dv;
    const unsigned col0 = by * ex.block_dim();
    const std::uint64_t head = static_cast<std::uint64_t>(j) * dk * dv;
    const std::uint64_t state_n = static_cast<std::uint64_t>(d.n_v) * dk * dv;

    ex.phase([&](unsigned tid, GdnRecRegs& r) {
        const unsigned c = col0 + tid;
        if (c >= dv) return;
        gdn_load_column(r, p.state_in + head + c, dk, dv);
    });

    for (unsigned t = 0; t < p.n_tok; ++t) {
        const float* qt = p.in.q + t * p.in.q_stride + static_cast<std::uint64_t>(kh) * dk;
        const float* kt = p.in.k + t * p.in.k_stride + static_cast<std::uint64_t>(kh) * dk;
        const float* vt = p.in.v + t * p.in.v_stride + static_cast<std::uint64_t>(j) * dv;
        // Per-token scalars (one thread): the CPU's prep_qk norms, exp(g) and beta.
        ex.phase([&](unsigned tid, GdnRecRegs&) {
            if (tid != 0) return;
            sh.inv_q = p.l2 != 0u ? gdn_inv_l2(qt, dk) : 1.0f;
            sh.inv_k = p.l2 != 0u ? gdn_inv_l2(kt, dk) : 1.0f;
            sh.decay = hexp(p.in.g[t * p.in.g_stride + j]);
            sh.beta = p.in.beta[t * p.in.beta_stride + j];
        });
        // Normalized q (then * q_scale) and k into LDS.
        // Padded to kGdnMaxDk with zeros (see gdn_load_column).
        ex.phase([&](unsigned tid, GdnRecRegs&) {
            for (unsigned i = tid; i < kGdnMaxDk; i += ex.block_dim()) {
                sh.qn[i] = i < dk ? (p.l2 != 0u ? qt[i] * sh.inv_q : qt[i]) * p.q_scale : 0.0f;
                sh.kn[i] = i < dk ? (p.l2 != 0u ? kt[i] * sh.inv_k : kt[i]) * 1.0f : 0.0f;
            }
        });
        // S <- S*exp(g); kv = S^T k; delta = (v - kv)*beta; S <- S + k delta^T; o = S^T q.
        ex.phase([&](unsigned tid, GdnRecRegs& r) {
            const unsigned c = col0 + tid;
            if (c >= dv) return;
            float kv = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < kGdnMaxDk; ++i) {
                r.s[i] = r.s[i] * sh.decay;
                kv = kv + r.s[i] * sh.kn[i];
                if ((i & 15u) == 15u) sched_fence();
            }
            const float delta = (vt[c] - kv) * sh.beta;
            float o = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < kGdnMaxDk; ++i) {
                r.s[i] = r.s[i] + sh.kn[i] * delta;
                o = o + r.s[i] * sh.qn[i];
                if ((i & 15u) == 15u) sched_fence();
            }
            p.out[t * p.out_stride + static_cast<std::uint64_t>(j) * dv + c] = o;
            const unsigned slot = p.n_tok - 1u - t;  // slot 0 = most recent row
            if (slot < p.n_slots)
                gdn_store_column(r, p.slots + ring_slot_off(p.ring_p, p.ring_live, slot, state_n) + head + c, dk, dv);
        });
    }

    ex.phase([&](unsigned tid, GdnRecRegs& r) {
        const unsigned c = col0 + tid;
        if (c >= dv) return;
        gdn_store_column(r, p.state_out + head + c, dk, dv);
    });
}

// =======================================================================================
// Chunked form (transformers torch_chunk_gated_delta_rule, as cpu::gated_delta_rule_chunked)
//
//   K0 gdn_check_g      : every g must satisfy g <= 0 (NaN fails, as the CPU check) —
//                         otherwise sets kStatusPositiveG and K1/K2 do nothing.
//   K1 gdn_chunk_intra  : per (head, chunk): prepped q/k, cumulative decay gc, decay matrix,
//                         intra-chunk attention, and the UT forward substitution
//                         (new_values, k_cumdecay). Independent of the state, so all chunks
//                         of a group run in parallel.
//   K2 gdn_chunk_state  : per (head, column block): sequential over the group's chunks;
//                         v_new, outputs, rollback slots and the chunk-end state update.
// The host runs K0 once over all T rows, then (K1, K2) per group of chunks that fits the
// workspace. Group 0's K2 reads `state`, later groups read `state_out`.
// =======================================================================================

/// Workspace record of one (head, chunk), in floats (cs = chunk size).
struct GdnChunkRecord {
    std::uint64_t q = 0;      // [cs, dk] prepped q (normalized, * q_scale)
    std::uint64_t k = 0;      // [cs, dk] prepped k
    std::uint64_t kcd = 0;    // [cs, dk] k_cumdecay
    std::uint64_t nv = 0;     // [cs, dv] new_values; K2 overwrites it with v_new
    std::uint64_t attn = 0;   // [cs, cs] (q_a . k_b) exp(gc_a - gc_b), b <= a
    std::uint64_t dec = 0;    // [cs, cs] exp(gc_a - gc_b), b <= a
    std::uint64_t gc = 0;     // [cs] cumulative log decay
    std::uint64_t floats = 0; // record size
};

HALO_HD inline GdnChunkRecord gdn_chunk_record(unsigned cs, unsigned dk, unsigned dv) {
    GdnChunkRecord r;
    const std::uint64_t cdk = static_cast<std::uint64_t>(cs) * dk;
    const std::uint64_t cc = static_cast<std::uint64_t>(cs) * cs;
    r.q = 0;
    r.k = r.q + cdk;
    r.kcd = r.k + cdk;
    r.nv = r.kcd + cdk;
    r.attn = r.nv + static_cast<std::uint64_t>(cs) * dv;
    r.dec = r.attn + cc;
    r.gc = r.dec + cc;
    r.floats = r.gc + cs;
    return r;
}

struct GdnCheckParams {
    const float* g = nullptr;
    std::uint64_t g_stride = 0;
    unsigned n_tok = 0;
    unsigned n_v = 0;
    std::uint32_t* status = nullptr;
};

struct GdnNoRegs {
    float unused;
};
struct GdnNoShared {
    float unused;
};

inline Launch gdn_check_launch(unsigned n_tok, unsigned n_v, unsigned block) {
    const std::uint64_t n = static_cast<std::uint64_t>(n_tok) * n_v;
    return Launch{static_cast<unsigned>((n + block - 1) / block), 1, block};
}

template <class Exec>
HALO_HD void gdn_check_g_body(Exec& ex, GdnNoShared&, const GdnCheckParams& p, unsigned bx, unsigned) {
    ex.phase([&](unsigned tid, GdnNoRegs&) {
        const std::uint64_t idx = static_cast<std::uint64_t>(bx) * ex.block_dim() + tid;
        if (idx >= static_cast<std::uint64_t>(p.n_tok) * p.n_v) return;
        const std::uint64_t t = idx / p.n_v;
        const std::uint64_t j = idx % p.n_v;
        const float g = p.g[t * p.g_stride + j];
        if (!(g <= 0.0f)) ex.atomic_or(p.status, kStatusPositiveG);
    });
}

struct GdnChunkParams {
    GdnIn in;
    GdnDimsK dims;
    unsigned n_tok = 0;
    unsigned cs = 0;           // chunk size (<= kGdnMaxChunk)
    unsigned chunk0 = 0;       // first chunk of this group
    unsigned n_chunks = 0;     // chunks in this group
    const float* state_in = nullptr;
    float* state_out = nullptr;
    float* slots = nullptr;      // ring: the slab base
    unsigned n_slots = 0;
    float* out = nullptr;
    std::uint64_t out_stride = 0;
    unsigned l2 = 1;
    float q_scale = 1.0f;
    float* ws = nullptr;       // n_chunks * n_v records; record (ci, j) at (ci * n_v + j)
    std::uint32_t* status = nullptr;
    unsigned ring_p = 0;     ///< ADR-001 §5.3: slab size (0 = contiguous slots region)
    unsigned ring_live = 0;  ///< the slab slot read this call
};

struct GdnIntraShared {
    float gc[kGdnMaxChunk];
    float beta[kGdnMaxChunk];
    float inv_q[kGdnMaxChunk];
    float inv_k[kGdnMaxChunk];
    float ut[kGdnMaxChunk * kGdnMaxChunk];  // strictly lower: (k_beta_a . k_b) exp(gc_a - gc_b)
};

inline Launch gdn_intra_launch(const GdnDimsK& d, unsigned n_chunks, unsigned block) {
    return Launch{d.n_v, n_chunks, block};
}

/// k_beta accessor: k[i] * beta (the CPU's kb array, computed on the fly — same product).
struct KBeta {
    const float* k;
    float beta;
    HALO_HD float operator()(unsigned i) const { return k[i] * beta; }
};

template <class Exec>
HALO_HD void gdn_chunk_intra_body(Exec& ex, GdnIntraShared& sh, const GdnChunkParams& p, unsigned bx,
                                  unsigned by) {
    if ((*p.status & kStatusPositiveG) != 0u) return;  // block-uniform
    const GdnDimsK& d = p.dims;
    const unsigned j = bx;
    const unsigned kh = gdn_k_head(d, j);
    const unsigned dk = d.dk;
    const unsigned dv = d.dv;
    const unsigned cs = p.cs;
    const unsigned t0 = (p.chunk0 + by) * cs;
    const unsigned len = (p.n_tok - t0) < cs ? (p.n_tok - t0) : cs;
    const GdnChunkRecord rec = gdn_chunk_record(cs, dk, dv);
    float* base = p.ws + (static_cast<std::uint64_t>(by) * d.n_v + j) * rec.floats;
    float* qw = base + rec.q;
    float* kw = base + rec.k;
    float* kcw = base + rec.kcd;
    float* nvw = base + rec.nv;
    float* attn = base + rec.attn;
    float* dec = base + rec.dec;
    float* gcw = base + rec.gc;
    const unsigned bd = ex.block_dim();

    // P1: per-row scalars: q/k inverse norms, beta, cumulative decay (sequential prefix,
    // exactly the CPU's `cum += g`).
    ex.phase([&](unsigned tid, GdnNoRegs&) {
        for (unsigned a = tid; a < len; a += bd) {
            const std::uint64_t t = t0 + a;
            const float* qt = p.in.q + t * p.in.q_stride + static_cast<std::uint64_t>(kh) * dk;
            const float* kt = p.in.k + t * p.in.k_stride + static_cast<std::uint64_t>(kh) * dk;
            sh.inv_q[a] = p.l2 != 0u ? gdn_inv_l2(qt, dk) : 1.0f;
            sh.inv_k[a] = p.l2 != 0u ? gdn_inv_l2(kt, dk) : 1.0f;
            sh.beta[a] = p.in.beta[t * p.in.beta_stride + j];
            float cum = 0.0f;
            for (unsigned b = 0; b <= a; ++b) cum = cum + p.in.g[(static_cast<std::uint64_t>(t0) + b) * p.in.g_stride + j];
            sh.gc[a] = cum;
            gcw[a] = cum;
        }
    });
    // P2: prepped q (normalized, then * q_scale) and k.
    ex.phase([&](unsigned tid, GdnNoRegs&) {
        for (unsigned e = tid; e < len * dk; e += bd) {
            const unsigned a = e / dk;
            const unsigned i = e % dk;
            const std::uint64_t t = t0 + a;
            const float qv = p.in.q[t * p.in.q_stride + static_cast<std::uint64_t>(kh) * dk + i];
            const float kv = p.in.k[t * p.in.k_stride + static_cast<std::uint64_t>(kh) * dk + i];
            qw[e] = (p.l2 != 0u ? qv * sh.inv_q[a] : qv) * p.q_scale;
            kw[e] = (p.l2 != 0u ? kv * sh.inv_k[a] : kv) * 1.0f;
        }
    });
    // P3: decay matrix, UT matrix (strictly lower) and intra-chunk attention (lower incl.).
    ex.phase([&](unsigned tid, GdnNoRegs&) {
        for (unsigned e = tid; e < len * len; e += bd) {
            const unsigned a = e / len;
            const unsigned b = e % len;
            if (b > a) continue;
            const float dc = hexp(sh.gc[a] - sh.gc[b]);
            dec[a * cs + b] = dc;
            const At kb{kw + static_cast<std::uint64_t>(b) * dk};
            if (b < a) sh.ut[a * cs + b] = dot8(KBeta{kw + static_cast<std::uint64_t>(a) * dk, sh.beta[a]}, kb, dk) * dc;
            attn[a * cs + b] = dot8(At{qw + static_cast<std::uint64_t>(a) * dk}, kb, dk) * dc;
        }
    });
    // P4: forward substitution, one thread per column (columns are independent):
    //   new_values[a][c] = v_beta[a][c] - sum_{b<a} ut[a][b] new_values[b][c]
    //   k_cumdecay[a][i] = k_beta[a][i] exp(gc_a) - sum_{b<a} ut[a][b] k_cumdecay[b][i]
    ex.phase([&](unsigned tid, GdnNoRegs&) {
        for (unsigned col = tid; col < dv + dk; col += bd) {
            if (col < dv) {
                const unsigned c = col;
                for (unsigned a = 0; a < len; ++a) {
                    const std::uint64_t t = t0 + a;
                    float x = p.in.v[t * p.in.v_stride + static_cast<std::uint64_t>(j) * dv + c] * sh.beta[a];
                    for (unsigned b = 0; b < a; ++b) x = x - sh.ut[a * cs + b] * nvw[static_cast<std::uint64_t>(b) * dv + c];
                    nvw[static_cast<std::uint64_t>(a) * dv + c] = x;
                }
            } else {
                const unsigned i = col - dv;
                for (unsigned a = 0; a < len; ++a) {
                    const float ea = hexp(sh.gc[a]);
                    float x = (kw[static_cast<std::uint64_t>(a) * dk + i] * sh.beta[a]) * ea;
                    for (unsigned b = 0; b < a; ++b) x = x - sh.ut[a * cs + b] * kcw[static_cast<std::uint64_t>(b) * dk + i];
                    kcw[static_cast<std::uint64_t>(a) * dk + i] = x;
                }
            }
        }
    });
}

// K2 keeps each thread's state column in LDS (thread-private: column tid of the tile, so no
// barrier is needed), not in registers: a register-resident column spilled heavily next to
// the chunk loops (ScratchSize 2744 B/lane measured at build time; see docs/hip.md).
inline constexpr unsigned kGdnStateMaxBlock = 64;  // 128 x 64 x 4 B = 32 KiB LDS

struct GdnStateShared {
    float s[kGdnMaxDk * kGdnStateMaxBlock];  // s[i * kGdnStateMaxBlock + tid]
};

inline Launch gdn_state_launch(const GdnDimsK& d, unsigned block) {
    return Launch{d.n_v, (d.dv + block - 1) / block, block};
}

template <class Exec>
HALO_HD void gdn_chunk_state_body(Exec& ex, GdnStateShared& sh, const GdnChunkParams& p, unsigned bx,
                                  unsigned by) {
    if ((*p.status & kStatusPositiveG) != 0u) return;  // block-uniform
    const GdnDimsK& d = p.dims;
    const unsigned j = bx;
    const unsigned dk = d.dk;
    const unsigned dv = d.dv;
    const unsigned cs = p.cs;
    const unsigned col0 = by * ex.block_dim();
    const std::uint64_t head = static_cast<std::uint64_t>(j) * dk * dv;
    const std::uint64_t state_n = static_cast<std::uint64_t>(d.n_v) * dk * dv;
    const GdnChunkRecord rec = gdn_chunk_record(cs, dk, dv);
    constexpr unsigned kB = kGdnStateMaxBlock;

    ex.phase([&](unsigned tid, GdnNoRegs&) {
        const unsigned c = col0 + tid;
        if (c >= dv) return;
        float* s = sh.s + tid;  // s[i * kB] = S[i][c]
        for (unsigned i = 0; i < dk; ++i) s[i * kB] = p.state_in[head + static_cast<std::uint64_t>(i) * dv + c];
        for (unsigned ci = 0; ci < p.n_chunks; ++ci) {
            const unsigned t0 = (p.chunk0 + ci) * cs;
            const unsigned len = (p.n_tok - t0) < cs ? (p.n_tok - t0) : cs;
            float* base = p.ws + (static_cast<std::uint64_t>(ci) * d.n_v + j) * rec.floats;
            const float* qw = base + rec.q;
            const float* kw = base + rec.k;
            const float* kcw = base + rec.kcd;
            float* vnw = base + rec.nv;
            const float* attn = base + rec.attn;
            const float* dec = base + rec.dec;
            const float* gc = base + rec.gc;
            // v_new = new_values - k_cumdecay @ S (in place over new_values, own column)
            for (unsigned a = 0; a < len; ++a) {
                float tmp = 0.0f;
                for (unsigned i = 0; i < dk; ++i) tmp = tmp + kcw[static_cast<std::uint64_t>(a) * dk + i] * s[i * kB];
                float* vn = vnw + static_cast<std::uint64_t>(a) * dv + c;
                *vn = *vn - tmp;
            }
            // out_a = (q_a exp(gc_a)) @ S + sum_{b<=a} attn[a][b] v_new_b
            for (unsigned a = 0; a < len; ++a) {
                const float ea = hexp(gc[a]);
                float o = 0.0f;
                for (unsigned i = 0; i < dk; ++i) {
                    const float qi = qw[static_cast<std::uint64_t>(a) * dk + i] * ea;
                    o = o + qi * s[i * kB];
                }
                float tmp = 0.0f;
                for (unsigned b = 0; b <= a; ++b) tmp = tmp + attn[a * cs + b] * vnw[static_cast<std::uint64_t>(b) * dv + c];
                o = o + tmp;
                p.out[(static_cast<std::uint64_t>(t0) + a) * p.out_stride + static_cast<std::uint64_t>(j) * dv + c] = o;
            }
            // State after chunk row a: S_a = S exp(gc_a) + sum_{b<=a} (k_b exp(gc_a-gc_b))^T v_new_b.
            // Rollback slots of rows in this chunk first (they read the pre-chunk S); the
            // chunk-end row (a = len-1) then updates S in place (element-wise).
            auto state_at = [&](unsigned a, float* dst, std::uint64_t dst_stride) {
                const float decay = hexp(gc[a]);
                for (unsigned i = 0; i < dk; ++i) {
                    float upd = 0.0f;
                    for (unsigned b = 0; b <= a; ++b) {
                        const float ki = kw[static_cast<std::uint64_t>(b) * dk + i] * dec[a * cs + b];
                        upd = upd + ki * vnw[static_cast<std::uint64_t>(b) * dv + c];
                    }
                    dst[i * dst_stride] = s[i * kB] * decay + upd;
                }
            };
            for (unsigned a = 0; a < len; ++a) {
                const unsigned slot = p.n_tok - 1u - (t0 + a);
                if (slot < p.n_slots) state_at(a, p.slots + ring_slot_off(p.ring_p, p.ring_live, slot, state_n) + head + c, dv);
            }
            state_at(len - 1u, s, kB);
        }
        for (unsigned i = 0; i < dk; ++i) p.state_out[head + static_cast<std::uint64_t>(i) * dv + c] = s[i * kB];
    });
}

}  // namespace halo::hip::kern
