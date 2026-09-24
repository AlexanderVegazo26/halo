#pragma once
// ATTENTION: causal GQA softmax attention over a paged K/V history (cpu::attention_gqa;
// DECISIONS D-004 GQA 24 Q / 4 KV heads, scale 1/sqrt(256)). See hd.h for the execution
// model.
//
// Paged KV layout: the halo::kv_cache pool, byte for byte. Block `id` of a pool with L
// layers, block_tokens rows and kv_dim = n_kv_head * head_dim floats per row holds
//   block[layer][K|V][token][kv_dim]     at float offset id * block_floats
//   + (layer * 2 + kv) * block_tokens * kv_dim   (kv_cache::KvPool::offset)
// and history row s of a sequence lives in block table[s / block_tokens], row
// s % block_tokens. Query t attends keys 0 ..= q_offset + t. Query head h uses KV head
// h / (n_head / n_kv_head) (cpu::attention_gqa's group mapping).
//
// Two variants, one workgroup per (query row t, query head h):
//   exact  : the CPU's order exactly — scores (8-lane dot, * scale), max, exp(s - m), a
//            sequential sum over s, p /= sum, then o[d] = sum_s p_s v_s[d] sequentially.
//            Bit-identical to cpu::attention_gqa (host libm). Serial sum: slow on purpose.
//   online : single pass over key tiles of `block` keys with a running max (flash-decoding
//            style). Per tile: scores; m' = max(m, tile max); alpha = exp(m - m');
//            l = l * alpha + sum_j exp(s_j - m') (j ascending); acc[d] = acc[d] * alpha +
//            sum_j exp(s_j - m') v_j[d] (j ascending); finally o = acc / l. Numerically
//            stable (every exponent <= 0), deterministic, but a different rounding sequence
//            from the CPU: accepted within a derived bound (tests, docs/hip.md).

#include "hd.h"

namespace halo::hip::kern {

inline constexpr unsigned kAttnMaxHeadDim = 256;
inline constexpr unsigned kAttnMaxBlock = 256;

struct AttnParams {
    const float* q = nullptr;
    std::uint64_t q_stride = 0;       // elements between query rows
    std::uint64_t q_head_stride = 0;  // elements between query heads (>= head_dim; TD-9 layout)
    const float* pool = nullptr;      // KV pool base
    std::uint64_t block_floats = 0;
    std::uint64_t k_off = 0;          // (layer * 2 + 0) * block_tokens * kv_dim
    std::uint64_t v_off = 0;          // (layer * 2 + 1) * block_tokens * kv_dim
    unsigned block_tokens = 0;
    unsigned kv_dim = 0;
    unsigned n_pool_blocks = 0;
    const std::uint32_t* table = nullptr;
    unsigned n_table = 0;
    unsigned n_head = 0;
    unsigned n_kv_head = 0;
    unsigned head_dim = 0;
    unsigned q_offset = 0;
    float scale = 1.0f;
    float* out = nullptr;
    std::uint64_t out_stride = 0;
    float* scores = nullptr;          // exact variant: [T][n_head][history]
    std::uint64_t scores_stride = 0;  // floats per (t, h)
    std::uint32_t* status = nullptr;
};

/// Row pointer of history row s (K or V), or null when its block id is outside the pool.
HALO_HD inline const float* attn_row(const AttnParams& p, unsigned s, bool v) {
    const unsigned bi = s / p.block_tokens;
    const std::uint32_t id = p.table[bi];
    if (id >= p.n_pool_blocks) return nullptr;
    return p.pool + static_cast<std::uint64_t>(id) * p.block_floats + (v ? p.v_off : p.k_off) +
           static_cast<std::uint64_t>(s % p.block_tokens) * p.kv_dim;
}

struct AttnNoRegs {
    float unused;
};

// ---- block-table check (all entries the call can touch) -------------------------------

inline Launch attn_check_launch(unsigned n_table, unsigned block) {
    return Launch{(n_table + block - 1) / block, 1, block};
}

template <class Exec>
HALO_HD void attn_check_body(Exec& ex, AttnNoRegs&, const AttnParams& p, unsigned bx, unsigned) {
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        const unsigned i = bx * ex.block_dim() + tid;
        if (i < p.n_table && p.table[i] >= p.n_pool_blocks) ex.atomic_or(p.status, kStatusBadBlock);
    });
}

inline Launch attn_launch(unsigned n_tok, unsigned n_head, unsigned block) { return Launch{n_tok, n_head, block}; }

// ---- exact variant --------------------------------------------------------------------

struct AttnExactShared {
    float q[kAttnMaxHeadDim];
    float m[kAttnMaxBlock];
    float mx;
    float sum;
};

template <class Exec>
HALO_HD void attn_exact_body(Exec& ex, AttnExactShared& sh, const AttnParams& p, unsigned bx, unsigned by) {
    const unsigned t = bx;
    const unsigned h = by;
    const unsigned hd = p.head_dim;
    const unsigned kvh = h / (p.n_head / p.n_kv_head);
    const unsigned n_keys = p.q_offset + t + 1;
    const unsigned bd = ex.block_dim();
    float* sc = p.scores + (static_cast<std::uint64_t>(t) * p.n_head + h) * p.scores_stride;
    const float* qh = p.q + static_cast<std::uint64_t>(t) * p.q_stride + static_cast<std::uint64_t>(h) * p.q_head_stride;
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        for (unsigned d = tid; d < hd; d += bd) sh.q[d] = qh[d];
    });
    // Scores and per-thread maxima (max is exact in any order for non-NaN values).
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        float m = -__builtin_huge_valf();
        for (unsigned s = tid; s < n_keys; s += bd) {
            const float* kr = attn_row(p, s, false);
            const float v = kr == nullptr ? -__builtin_huge_valf()
                                          : dot8(At{sh.q}, At{kr + static_cast<std::uint64_t>(kvh) * hd}, hd) * p.scale;
            sc[s] = v;
            m = v > m ? v : m;  // std::max(m, v) for non-NaN v
        }
        sh.m[tid] = m;
    });
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        if (tid != 0) return;
        float m = -__builtin_huge_valf();
        for (unsigned i = 0; i < bd; ++i) m = sh.m[i] > m ? sh.m[i] : m;
        sh.mx = m;
    });
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        for (unsigned s = tid; s < n_keys; s += bd) sc[s] = hexp(sc[s] - sh.mx);
    });
    ex.phase([&](unsigned tid, AttnNoRegs&) {  // the CPU's sequential sum
        if (tid != 0) return;
        float sum = 0.0f;
        for (unsigned s = 0; s < n_keys; ++s) sum = sum + sc[s];
        sh.sum = sum;
    });
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        for (unsigned s = tid; s < n_keys; s += bd) sc[s] = sc[s] / sh.sum;
    });
    ex.phase([&](unsigned tid, AttnNoRegs&) {
        float* o = p.out + static_cast<std::uint64_t>(t) * p.out_stride + static_cast<std::uint64_t>(h) * hd;
        for (unsigned d = tid; d < hd; d += bd) {
            float acc = 0.0f;
            for (unsigned s = 0; s < n_keys; ++s) {
                const float* vr = attn_row(p, s, true);
                if (vr != nullptr) acc = acc + sc[s] * vr[static_cast<std::uint64_t>(kvh) * hd + d];
            }
            o[d] = acc;
        }
    });
}

// ---- online (running max) variant -----------------------------------------------------

struct AttnOnlineShared {
    float q[kAttnMaxHeadDim];
    float p[kAttnMaxBlock];  // tile scores, then exp(s - m')
    float m;                 // running max
    float l;                 // running sum
    float alpha;             // exp(m_old - m_new) of the current tile
};

struct AttnOnlineRegs {
    float acc[kAttnMaxHeadDim / 32];  // dims tid, tid + bd, ... (bd >= 32)
};

template <class Exec>
HALO_HD void attn_online_body(Exec& ex, AttnOnlineShared& sh, const AttnParams& p, unsigned bx, unsigned by) {
    const unsigned t = bx;
    const unsigned h = by;
    const unsigned hd = p.head_dim;
    const unsigned kvh = h / (p.n_head / p.n_kv_head);
    const unsigned n_keys = p.q_offset + t + 1;
    const unsigned bd = ex.block_dim();
    const unsigned per = (hd + bd - 1) / bd;  // dims per thread (<= kAttnMaxHeadDim / 32)
    const float* qh = p.q + static_cast<std::uint64_t>(t) * p.q_stride + static_cast<std::uint64_t>(h) * p.q_head_stride;
    ex.phase([&](unsigned tid, AttnOnlineRegs& r) {
        for (unsigned d = tid; d < hd; d += bd) sh.q[d] = qh[d];
        for (unsigned i = 0; i < kAttnMaxHeadDim / 32; ++i) r.acc[i] = 0.0f;
        if (tid == 0) {
            sh.m = -__builtin_huge_valf();
            sh.l = 0.0f;
        }
    });
    for (unsigned s0 = 0; s0 < n_keys; s0 += bd) {
        const unsigned len = n_keys - s0 < bd ? n_keys - s0 : bd;
        ex.phase([&](unsigned tid, AttnOnlineRegs&) {
            if (tid >= len) return;
            const float* kr = attn_row(p, s0 + tid, false);
            sh.p[tid] = kr == nullptr ? -__builtin_huge_valf()
                                      : dot8(At{sh.q}, At{kr + static_cast<std::uint64_t>(kvh) * hd}, hd) * p.scale;
        });
        ex.phase([&](unsigned tid, AttnOnlineRegs&) {
            if (tid != 0) return;
            float mt = sh.m;
            for (unsigned j = 0; j < len; ++j) mt = sh.p[j] > mt ? sh.p[j] : mt;
            // First tile: m = -inf, so alpha = exp(-inf) = 0 (l and acc are 0 anyway).
            sh.alpha = hexp(sh.m - mt);
            sh.m = mt;
        });
        ex.phase([&](unsigned tid, AttnOnlineRegs&) {
            if (tid < len) sh.p[tid] = hexp(sh.p[tid] - sh.m);
        });
        ex.phase([&](unsigned tid, AttnOnlineRegs& r) {
            if (tid == 0) {
                float l = sh.l * sh.alpha;
                for (unsigned j = 0; j < len; ++j) l = l + sh.p[j];
                sh.l = l;
            }
            for (unsigned i = 0; i < per; ++i) {
                const unsigned d = tid + i * bd;
                if (d >= hd) break;
                float a = r.acc[i] * sh.alpha;
                for (unsigned j = 0; j < len; ++j) {
                    const float* vr = attn_row(p, s0 + j, true);
                    if (vr != nullptr) a = a + sh.p[j] * vr[static_cast<std::uint64_t>(kvh) * hd + d];
                }
                r.acc[i] = a;
            }
        });
        // (the next tile's first phase writes sh.p only after this barrier)
    }
    ex.phase([&](unsigned tid, AttnOnlineRegs& r) {
        float* o = p.out + static_cast<std::uint64_t>(t) * p.out_stride + static_cast<std::uint64_t>(h) * hd;
        for (unsigned i = 0; i < per; ++i) {
            const unsigned d = tid + i * bd;
            if (d >= hd) break;
            o[d] = r.acc[i] / sh.l;
        }
    });
}

}  // namespace halo::hip::kern
