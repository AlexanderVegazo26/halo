#pragma once
// LM head reductions (TRD §19; DECISIONS D-016 ARGMAX raises on NaN), TOP_K, PARTIAL_ROPE,
// SWIGLU and MUL_SIGMOID kernel bodies. See hd.h for the execution model.
//
// ARGMAX and TOP_K use the CPU reference's strict total order — larger value first, then
// lower index (cpu::matmul.cpp `better`) — which is associative and commutative, so any
// reduction tree gives exactly the CPU's answer. NaN is never ordered: it only sets a flag
// that the host turns into Error(Kernel).

#include "hd.h"

namespace halo::hip::kern {

inline constexpr std::uint32_t kNoIndex = 0xFFFFFFFFu;

/// The CPU's order: (va, ia) ranks before (vb, ib). kNoIndex ranks after everything.
HALO_HD inline bool ranks_before(float va, std::uint32_t ia, float vb, std::uint32_t ib) {
    if (ia == kNoIndex) return false;
    if (ib == kNoIndex) return true;
    return va > vb || (va == vb && ia < ib);
}

/// Per-workgroup argmax partial.
struct ArgPart {
    float value;
    std::uint32_t index;  // kNoIndex = no candidate
    std::uint32_t nan;    // 1 if a NaN was seen
    std::uint32_t pad;
};

/// Result words of one vector: {index, value bits, nan} (the Vulkan backend's layout).
struct ArgResult {
    std::uint32_t index;
    float value;
    std::uint32_t nan;
};

// ---- stage 1 over an existing logits vector ------------------------------------------

inline constexpr unsigned kArgPerThread = 16;

struct ArgmaxParams {
    const float* x = nullptr;   // logits
    std::uint64_t x_stride = 0; // elements between vectors
    unsigned n = 0;             // logits per vector
    ArgPart* part = nullptr;    // [n_vec][n_parts]
    unsigned n_parts = 0;       // stage-1 workgroups per vector
    ArgResult* result = nullptr;
};

struct ArgNoRegs {
    float unused;
};

inline constexpr unsigned kArgMaxBlock = 256;

struct ArgShared {
    float v[kArgMaxBlock];
    std::uint32_t i[kArgMaxBlock];
    std::uint32_t nan[kArgMaxBlock];
};

inline Launch argmax_partial_launch(unsigned n, unsigned n_vec, unsigned block) {
    const unsigned per_wg = block * kArgPerThread;
    return Launch{(n + per_wg - 1) / per_wg, n_vec, block};
}

/// Tree-reduces sh[0..bd) into sh[0] (fixed order; lanes < s merge lane + s).
template <class Exec>
HALO_HD void arg_tree(Exec& ex, ArgShared& sh) {
    for (unsigned s = ex.block_dim() / 2; s > 0; s /= 2) {
        ex.phase([&](unsigned tid, ArgNoRegs&) {
            if (tid >= s) return;
            if (ranks_before(sh.v[tid + s], sh.i[tid + s], sh.v[tid], sh.i[tid])) {
                sh.v[tid] = sh.v[tid + s];
                sh.i[tid] = sh.i[tid + s];
            }
            sh.nan[tid] = sh.nan[tid] | sh.nan[tid + s];
        });
    }
}

template <class Exec>
HALO_HD void argmax_partial_body(Exec& ex, ArgShared& sh, const ArgmaxParams& p, unsigned bx, unsigned by) {
    const float* x = p.x + static_cast<std::uint64_t>(by) * p.x_stride;
    const unsigned base = bx * ex.block_dim() * kArgPerThread;
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        float bv = 0.0f;
        std::uint32_t bi = kNoIndex;
        std::uint32_t nan = 0;
        for (unsigned k = 0; k < kArgPerThread; ++k) {
            const unsigned idx = base + k * ex.block_dim() + tid;
            if (idx >= p.n) break;
            const float v = x[idx];
            if (hisnan(v)) {
                nan = 1;
                continue;
            }
            if (ranks_before(v, idx, bv, bi)) {
                bv = v;
                bi = idx;
            }
        }
        sh.v[tid] = bv;
        sh.i[tid] = bi;
        sh.nan[tid] = nan;
    });
    arg_tree(ex, sh);
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        if (tid != 0) return;
        ArgPart& o = p.part[static_cast<std::uint64_t>(by) * p.n_parts + bx];
        o.value = sh.v[0];
        o.index = sh.i[0];
        o.nan = sh.nan[0];
        o.pad = 0;
    });
}

// ---- stage 2: partials -> one result per vector (one workgroup per vector) -----------

inline Launch argmax_reduce_launch(unsigned n_vec, unsigned block) { return Launch{n_vec, 1, block}; }

template <class Exec>
HALO_HD void argmax_reduce_body(Exec& ex, ArgShared& sh, const ArgmaxParams& p, unsigned bx, unsigned) {
    const ArgPart* part = p.part + static_cast<std::uint64_t>(bx) * p.n_parts;
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        float bv = 0.0f;
        std::uint32_t bi = kNoIndex;
        std::uint32_t nan = 0;
        for (unsigned j = tid; j < p.n_parts; j += ex.block_dim()) {
            nan = nan | part[j].nan;
            if (ranks_before(part[j].value, part[j].index, bv, bi)) {
                bv = part[j].value;
                bi = part[j].index;
            }
        }
        sh.v[tid] = bv;
        sh.i[tid] = bi;
        sh.nan[tid] = nan;
    });
    arg_tree(ex, sh);
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        if (tid != 0) return;
        ArgResult& r = p.result[bx];
        r.index = sh.i[0];
        r.value = sh.v[0];
        r.nan = sh.nan[0];
    });
}

// ---- TOP_K: bitonic sort of 2048-element chunks in LDS, repeated until one chunk -------

inline constexpr unsigned kTopkChunk = 2048;
inline constexpr unsigned kTopkMaxK = 1024;  // <= kTopkChunk / 2: every round halves the count

struct TopkPair {
    float value;
    std::uint32_t index;
};

struct TopkParams {
    const float* src_logits = nullptr;  // round 0 (index = position), else null
    const TopkPair* src_pairs = nullptr;// later rounds
    std::uint64_t src_stride = 0;       // elements (logits) or pairs between vectors
    unsigned m = 0;                     // candidates per vector in this round
    unsigned k = 0;
    TopkPair* dst = nullptr;            // non-final rounds: [n_vec][n_blocks * k]
    std::uint64_t dst_stride = 0;       // pairs between vectors
    std::int32_t* out_ids = nullptr;    // final round
    float* out_vals = nullptr;
    std::uint64_t out_stride = 0;       // elements between vectors (ids and values)
    std::uint32_t* status = nullptr;    // kStatusNaN
};

struct TopkShared {
    float v[kTopkChunk];
    std::uint32_t i[kTopkChunk];
};

inline Launch topk_launch(unsigned m, unsigned n_vec, unsigned block) {
    return Launch{(m + kTopkChunk - 1) / kTopkChunk, n_vec, block};
}

template <class Exec>
HALO_HD void topk_body(Exec& ex, TopkShared& sh, const TopkParams& p, unsigned bx, unsigned by) {
    const unsigned bd = ex.block_dim();
    const unsigned base = bx * kTopkChunk;
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        for (unsigned e = tid; e < kTopkChunk; e += bd) {
            const unsigned g = base + e;
            float v = 0.0f;
            std::uint32_t idx = kNoIndex;
            if (g < p.m) {
                if (p.src_logits != nullptr) {
                    v = p.src_logits[static_cast<std::uint64_t>(by) * p.src_stride + g];
                    idx = g;
                } else {
                    const TopkPair q = p.src_pairs[static_cast<std::uint64_t>(by) * p.src_stride + g];
                    v = q.value;
                    idx = q.index;
                }
                if (hisnan(v)) {
                    ex.atomic_or(p.status, kStatusNaN);
                    idx = kNoIndex;  // never ranked; the host raises Error(Kernel)
                }
            }
            sh.v[e] = v;
            sh.i[e] = idx;
        }
    });
    // Bitonic sort into rank order (best first). Step (size, stride): every element is in
    // exactly one pair (e, e ^ stride), handled by one thread — no intra-phase hazard.
    for (unsigned size = 2; size <= kTopkChunk; size *= 2) {
        for (unsigned stride = size / 2; stride > 0; stride /= 2) {
            ex.phase([&](unsigned tid, ArgNoRegs&) {
                for (unsigned e = tid; e < kTopkChunk / 2; e += bd) {
                    const unsigned a = 2 * stride * (e / stride) + e % stride;
                    const unsigned b = a + stride;
                    const bool best_first = (a & size) == 0;
                    const bool b_before_a = ranks_before(sh.v[b], sh.i[b], sh.v[a], sh.i[a]);
                    const bool a_before_b = ranks_before(sh.v[a], sh.i[a], sh.v[b], sh.i[b]);
                    if (best_first ? b_before_a : a_before_b) {
                        const float tv = sh.v[a];
                        const std::uint32_t ti = sh.i[a];
                        sh.v[a] = sh.v[b];
                        sh.i[a] = sh.i[b];
                        sh.v[b] = tv;
                        sh.i[b] = ti;
                    }
                }
            });
        }
    }
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        for (unsigned e = tid; e < p.k; e += bd) {
            if (p.out_ids != nullptr) {
                const std::uint64_t o = static_cast<std::uint64_t>(by) * p.out_stride + e;
                p.out_ids[o] = static_cast<std::int32_t>(sh.i[e]);
                p.out_vals[o] = sh.v[e];
            } else {
                TopkPair& d = p.dst[static_cast<std::uint64_t>(by) * p.dst_stride + static_cast<std::uint64_t>(bx) * p.k + e];
                d.value = sh.v[e];
                d.index = sh.i[e];
            }
        }
    });
}

// ---- PARTIAL_ROPE (NeoX rotate-half, in place; cpu::partial_rope_neox) ---------------

inline constexpr unsigned kRopeMaxHalf = 64;

struct RopeParams {
    float* x = nullptr;
    std::uint64_t x_stride = 0;
    const std::int32_t* pos = nullptr;
    unsigned n_heads = 0;
    unsigned head_dim = 0;
    std::uint64_t head_stride = 0; // elements between consecutive heads (>= head_dim)
    unsigned half = 0;             // rot_dims / 2
    float inv[kRopeMaxHalf] = {};  // cpu::rope_inv_freq, computed on the host
};

struct RopeShared {
    float cs[kRopeMaxHalf];
    float sn[kRopeMaxHalf];
};

inline Launch rope_launch(unsigned n_tok, unsigned n_heads, unsigned half, unsigned block) {
    return Launch{n_tok, (n_heads * half + block - 1) / block, block};
}

template <class Exec>
HALO_HD void rope_body(Exec& ex, RopeShared& sh, const RopeParams& p, unsigned bx, unsigned by) {
    const float pos = static_cast<float>(p.pos[bx]);
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        for (unsigned i = tid; i < p.half; i += ex.block_dim()) {
            const float angle = pos * p.inv[i];  // one fp32 multiply, as the CPU
            sh.cs[i] = static_cast<float>(hcos(static_cast<double>(angle)));
            sh.sn[i] = static_cast<float>(hsin(static_cast<double>(angle)));
        }
    });
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        const unsigned e = by * ex.block_dim() + tid;
        if (e >= p.n_heads * p.half) return;
        const unsigned h = e / p.half;
        const unsigned i = e % p.half;
        float* xh = p.x + static_cast<std::uint64_t>(bx) * p.x_stride + static_cast<std::uint64_t>(h) * p.head_stride;
        const float x1 = xh[i];
        const float x2 = xh[i + p.half];
        xh[i] = x1 * sh.cs[i] + (-x2) * sh.sn[i];
        xh[i + p.half] = x2 * sh.cs[i] + x1 * sh.sn[i];
    });
}

// ---- SWIGLU / MUL_SIGMOID (element-wise; out may alias an input exactly) --------------

enum class EwOp : unsigned { SwiGlu = 0, MulSigmoid = 1 };

struct EwParams {
    EwOp op = EwOp::SwiGlu;
    const float* a = nullptr;  // SwiGlu: gate; MulSigmoid: x
    std::uint64_t a_stride = 0;
    const float* b = nullptr;  // SwiGlu: up;   MulSigmoid: gate
    std::uint64_t b_stride = 0;
    float* out = nullptr;
    std::uint64_t out_stride = 0;
    unsigned rows = 0;
    unsigned cols = 0;
};

struct EwShared {
    float unused;
};

inline Launch ew_launch(unsigned rows, unsigned cols, unsigned block) {
    return Launch{rows, (cols + block - 1) / block, block};
}

template <class Exec>
HALO_HD void ew_body(Exec& ex, EwShared&, const EwParams& p, unsigned bx, unsigned by) {
    ex.phase([&](unsigned tid, ArgNoRegs&) {
        const unsigned c = by * ex.block_dim() + tid;
        if (c >= p.cols) return;
        const float a = p.a[static_cast<std::uint64_t>(bx) * p.a_stride + c];
        const float b = p.b[static_cast<std::uint64_t>(bx) * p.b_stride + c];
        p.out[static_cast<std::uint64_t>(bx) * p.out_stride + c] =
            p.op == EwOp::SwiGlu ? siluf(a) * b : a * sigmoidf(b);
    });
}

}  // namespace halo::hip::kern
