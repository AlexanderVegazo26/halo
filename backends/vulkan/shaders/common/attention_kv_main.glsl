// [ATTENTION] over a quantized KV pool (attention_f16.comp / attention_q8.comp include this with
// KV_TYPE = 1 / 2; the fp32 pool has its own shader, attention/attention.comp).
//
// Identical to attention.comp (causal GQA online softmax, one workgroup per (token, query head),
// the CPU's 8-lane fixed-order dot) except that K and V elements are dequantized on read from
// the pool's uint words (ops.h "The paged KV pool", kv_write_quant_main.glsl for the layouts):
//   KV_TYPE 1 (f16): element i of a head = half (i & 1) of word head_base + i / 2.
//   KV_TYPE 2 (q8):  element i = scale(word head_base + (i/32)*9) * int8 (i & 31) of the 8 words
//     after it. The host guarantees head_dim % 32 == 0 (q8) / head_dim even (f16), so a head
//     starts on a group / word boundary.
// The dequantized value is exact in fp32 (f16 -> f32 widening; q * scale is a single fp32
// multiply), so the only error vs the fp32 pool is the storage rounding done by kv_write.
#include "halo_common.glsl"

#ifndef KV_TYPE
#error "KV_TYPE must be 1 (f16) or 2 (q8)"
#endif

layout(constant_id = 0) const uint WG = 128;  // power of two (host-validated)
layout(local_size_x_id = 0) in;

const uint MAX_HD = 256u;
const uint DIMS = 4u;
const uint STATUS_BAD_BLOCK = 4u;
const uint NO_ROW = 0xFFFFFFFFu;

layout(std430, binding = 0) readonly buffer Q { float q[]; };
layout(std430, binding = 1) readonly buffer Pool { uint pool[]; };
layout(std430, binding = 2) readonly buffer Tab { uint tab[]; };
layout(std430, binding = 3) writeonly buffer O { float o[]; };
layout(std430, binding = 4) buffer St { uint st[]; };

layout(push_constant) uniform Push {
    uint n_head;
    uint n_kv_head;
    uint head_dim;
    uint n_tokens;
    uint q_offset;
    float scale;
    uint n_pool_blocks;
    uint block_words;  // n_layers * 2 * bt * row_words
    uint layer_off;    // layer * 2 * bt * row_words
    uint bt;
    uint row_words;    // words of one K (or V) row in this format
    uint q_off, q_stride, q_head_stride, pool_off, tab_off, o_off, o_stride, st_off;
} pc;

shared float qs[MAX_HD];
shared float sc[WG];
shared float red[WG];
shared uint rowb[WG];

// Element i of the head whose first word is hb.
float kv_at(uint hb, uint i) {
#if KV_TYPE == 1
    return unpackHalf2x16(pool[hb + (i >> 1)])[i & 1u];
#else
    const uint w = hb + (i >> 5) * 9u;
    const uint j = i & 31u;
    const int qv = bitfieldExtract(int(pool[w + 1u + (j >> 2)]), int((j & 3u) << 3), 8);
    return float(qv) * uintBitsToFloat(pool[w]);
#endif
}

// cpu::detail::dot: 8 interleaved partial sums, pairwise combine, sequential tail.
float dot8(uint kb, uint n) {
    precise float acc[8] = float[8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    uint i = 0u;
    for (; i + 8u <= n; i += 8u) {
        for (uint j = 0u; j < 8u; ++j) {
            precise float p = qs[i + j] * kv_at(kb, i + j);
            acc[j] = acc[j] + p;
        }
    }
    precise float tail = 0.0;
    for (; i < n; ++i) {
        precise float p = qs[i] * kv_at(kb, i);
        tail = tail + p;
    }
    precise float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    precise float r = s + tail;
    return r;
}

void main() {
    const uint item = halo_group_id();
    if (item >= pc.n_tokens * pc.n_head) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint t = item / pc.n_head;
    const uint h = item % pc.n_head;
    const uint kvh = h / (pc.n_head / pc.n_kv_head);
    const uint hd = pc.head_dim;
#if KV_TYPE == 1
    const uint head_words = hd >> 1;
#else
    const uint head_words = (hd >> 5) * 9u;
#endif
    const uint qb = pc.q_off + t * pc.q_stride + h * pc.q_head_stride;
    for (uint d = tid; d < hd; d += WG) qs[d] = q[qb + d];
    barrier();

    const uint n_keys = pc.q_offset + t + 1u;
    const uint v_delta = pc.bt * pc.row_words;
    const float NEG_INF = uintBitsToFloat(0xFF800000u);
    float m = NEG_INF;
    float l = 0.0;
    float acc[DIMS] = float[DIMS](0.0, 0.0, 0.0, 0.0);
    for (uint base = 0u; base < n_keys; base += WG) {
        const uint s = base + tid;
        float x = NEG_INF;
        uint rb = NO_ROW;
        if (s < n_keys) {
            const uint id = tab[pc.tab_off + s / pc.bt];
            if (id >= pc.n_pool_blocks) {
                atomicOr(st[pc.st_off], STATUS_BAD_BLOCK);
            } else {
                rb = pc.pool_off + id * pc.block_words + pc.layer_off + (s % pc.bt) * pc.row_words + kvh * head_words;
                precise float xs = dot8(rb, hd) * pc.scale;
                x = xs;
            }
        }
        rowb[tid] = rb;
        red[tid] = x;
        barrier();
        for (uint w = WG / 2u; w > 0u; w >>= 1) {
            if (tid < w) red[tid] = max(red[tid], red[tid + w]);
            barrier();
        }
        const float m_new = max(m, red[0]);
        barrier();
        if (m_new == NEG_INF) continue;  // uniform: every key so far was skipped
        precise float alpha = m == NEG_INF ? 0.0 : exp(m - m_new);
        precise float p = rb == NO_ROW ? 0.0 : exp(x - m_new);
        sc[tid] = p;
        red[tid] = p;
        barrier();
        for (uint w = WG / 2u; w > 0u; w >>= 1) {
            if (tid < w) red[tid] += red[tid + w];
            barrier();
        }
        precise float ls = l * alpha;
        l = ls + red[0];
        m = m_new;
        const uint tile_n = min(WG, n_keys - base);
        for (uint j = 0u; j < DIMS; ++j) {
            const uint d = tid + j * WG;
            if (d >= hd) break;
            precise float a = acc[j] * alpha;
            for (uint i = 0u; i < tile_n; ++i) {
                const uint r = rowb[i];
                if (r == NO_ROW) continue;
                precise float pv = sc[i] * kv_at(r + v_delta, d);
                a = a + pv;
            }
            acc[j] = a;
        }
        barrier();
    }
    const uint ob = pc.o_off + t * pc.o_stride + h * hd;
    for (uint j = 0u; j < DIMS; ++j) {
        const uint d = tid + j * WG;
        if (d >= hd) break;
        o[ob + d] = l > 0.0 ? acc[j] / l : 0.0;
    }
}
