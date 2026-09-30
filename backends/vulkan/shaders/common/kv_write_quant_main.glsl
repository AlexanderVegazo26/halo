// [KV write], quantized pool formats (kv_write_f16.comp / kv_write_q8.comp include this with
// KV_TYPE = 1 / 2; the fp32 pool has its own shader, attention/kv_write.comp).
//
// Same contract as kv_write.comp (ops.h "The paged KV pool"): K and V rows of n_tokens tokens
// go to block[layer][K|V][token] of the paged pool, one workgroup per token; history row
// s = start + t lives in block table[s / bt], token s % bt. A table entry >= n_pool_blocks sets
// status bit 4 (k_status_bad_block; the host zeroed the word) and that row is not written.
//
// The pool is bound as uint words; no invocation writes a word another one writes (there is no
// 8/16-bit storage here, so a sub-word read-modify-write would be a race):
//   KV_TYPE 1 (f16): two halves per word (packHalf2x16, round-to-nearest-even); row = kv_dim/2
//     words. One invocation owns one word. Values beyond +-65504 become +-inf.
//   KV_TYPE 2 (q8):  each 32-element group is 9 words [fp32 scale | 32 x int8, 4 per word];
//     row = kv_dim/32*9 words. One invocation owns one group: scale = max|x| / 127,
//     q = clamp(roundEven(x * 127 / max|x|), -127, 127) (all zero when max|x| = 0).
#include "halo_common.glsl"

#ifndef KV_TYPE
#error "KV_TYPE must be 1 (f16) or 2 (q8)"
#endif

layout(constant_id = 0) const uint WG = 256;
layout(local_size_x_id = 0) in;

const uint STATUS_BAD_BLOCK = 4u;

layout(std430, binding = 0) writeonly buffer Pool { uint pool[]; };
layout(std430, binding = 1) readonly buffer Tab { uint tab[]; };
layout(std430, binding = 2) readonly buffer K { float k[]; };
layout(std430, binding = 3) readonly buffer V { float v[]; };
layout(std430, binding = 4) buffer St { uint st[]; };

layout(push_constant) uniform Push {
    uint n_pool_blocks;
    uint block_words;  // n_layers * 2 * bt * row_words
    uint layer_off;    // layer * 2 * bt * row_words
    uint bt;
    uint kv_dim;
    uint row_words;    // words of one K (or V) row in this format
    uint start;
    uint n_tokens;
    uint pool_off, tab_off, k_off, k_stride, v_off, v_stride, st_off;
} pc;

#if KV_TYPE == 1
// Copies one row (src element base sb, dst word base db).
void write_row(uint db, uint sb, bool is_k) {
    for (uint p = gl_LocalInvocationID.x; p < pc.kv_dim / 2u; p += WG) {
        const vec2 x = is_k ? vec2(k[sb + 2u * p], k[sb + 2u * p + 1u]) : vec2(v[sb + 2u * p], v[sb + 2u * p + 1u]);
        pool[db + p] = packHalf2x16(x);
    }
}
#else
float src_at(uint i, bool is_k) { return is_k ? k[i] : v[i]; }

void write_row(uint db, uint sb, bool is_k) {
    for (uint g = gl_LocalInvocationID.x; g < pc.kv_dim / 32u; g += WG) {
        const uint s0 = sb + g * 32u;
        float amax = 0.0;
        for (uint j = 0u; j < 32u; ++j) amax = max(amax, abs(src_at(s0 + j, is_k)));
        const float inv = amax > 0.0 ? 127.0 / amax : 0.0;
        const uint w0 = db + g * 9u;
        pool[w0] = floatBitsToUint(amax / 127.0);
        for (uint w = 0u; w < 8u; ++w) {
            uint word = 0u;
            for (uint b = 0u; b < 4u; ++b) {
                const int q = clamp(int(roundEven(src_at(s0 + w * 4u + b, is_k) * inv)), -127, 127);
                word |= (uint(q) & 0xFFu) << (b * 8u);
            }
            pool[w0 + 1u + w] = word;
        }
    }
}
#endif

void main() {
    const uint t = halo_group_id();
    if (t >= pc.n_tokens) return;
    const uint s = pc.start + t;
    const uint id = tab[pc.tab_off + s / pc.bt];
    if (id >= pc.n_pool_blocks) {
        if (gl_LocalInvocationID.x == 0u) atomicOr(st[pc.st_off], STATUS_BAD_BLOCK);
        return;
    }
    const uint kb = pc.pool_off + id * pc.block_words + pc.layer_off + (s % pc.bt) * pc.row_words;
    const uint vb = kb + pc.bt * pc.row_words;
    write_row(kb, pc.k_off + t * pc.k_stride, true);
    write_row(vb, pc.v_off + t * pc.v_stride, false);
}
