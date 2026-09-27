// Per-type body of matvec_q3_k (ggml Q3_K GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_q3_k.comp). Port of llama.cpp's Vulkan
// mul_mat_vec_q3_k.comp (MIT) to HALO's binding model, on the matvec_q5_k_main.glsl
// pattern: 16 threads per 110-byte block, each thread covers 16 elements (2 per
// 16-element sub-block) with SWAR unpacks: 2-bit qs + inverted hmask bits as (q - 4*!bit),
// and the 8 sub-block scales of the thread's 128-half unpacked directly from 3 words
// (~8 ops) instead of llama's shared sccache + barrier. The 6-bit scales are applied as
// (x*sc)*(q-hmk) products accumulated per sub-block, d hoisted to one fma per block:
//   part = d * sum16((x*sc) * (q - hmk))
// The float expression in q3k_apply is identical in the decode and batched paths and in
// both LD32 instantiations, so vector t of a batch, the n_vec = 1 matvec, and the
// dense/arena views are bitwise identical (VkHead / VkViews gates). The shared-memory
// reduction tree is the same one as matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at even (fast) or arbitrary (slow) byte offset
//   Q3K_FN        -> function-name token (pasted as q3k_<Q3K_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[] from the .comp.

#define Q3K_PASTE2(a, b) a##b
#define Q3K_PASTE(a, b) Q3K_PASTE2(a, b)

vec4 Q3K_PASTE(q3k_b4_, Q3K_FN)(const uint v) {
    return vec4(uvec4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24));
}

// SWAR dequant of this thread's 16 elements of the 110-byte block at byte offset bo.
// Layout: hmask[32] @ 0, qs[64] @ 32, scales[12] @ 96, fp16 d @ 108. v_in even byte lane
// (l0 = 2*v_in), v_im the 128-half, v_im4 = 4*v_im.
Q3KBlock Q3K_PASTE(q3k_dequant_, Q3K_FN)(const uint bo, const uint v_im, const uint v_in,
                                         const uint q_off) {
    Q3KBlock b;
    const uint v_im4 = 4u * v_im;

    // Sign masks: hmask u16 at bytes 2*v_in and 2*v_in+16, inverted (bit set = NO -4).
    const uint hmk = ~((LD32(bo + 2u * v_in) & 0xFFFFu) | ((LD32(bo + 2u * v_in + 16u) & 0xFFFFu) << 16));
    const uint m = 0x01010101u << v_im4;
    b.h0 = Q3K_PASTE(q3k_b4_, Q3K_FN)(((hmk & (m)) >> (v_im4)) << 2);
    b.h1 = Q3K_PASTE(q3k_b4_, Q3K_FN)(((hmk & (m << 1u)) >> (1u + v_im4)) << 2);
    b.h2 = Q3K_PASTE(q3k_b4_, Q3K_FN)(((hmk & (m << 2u)) >> (2u + v_im4)) << 2);
    b.h3 = Q3K_PASTE(q3k_b4_, Q3K_FN)(((hmk & (m << 3u)) >> (3u + v_im4)) << 2);

    // qs 2-bit values: bytes q_off, q_off+1 (elements l0, l0+1) and q_off+16, +17 (+16).
    const uint qs_u32 =
        (LD32(bo + 32u + q_off) & 0xFFFFu) | ((LD32(bo + 32u + q_off + 16u) & 0xFFFFu) << 16);
    b.q0 = Q3K_PASTE(q3k_b4_, Q3K_FN)(qs_u32 & 0x03030303u);
    b.q2 = Q3K_PASTE(q3k_b4_, Q3K_FN)((qs_u32 >> 2) & 0x03030303u);
    b.q4 = Q3K_PASTE(q3k_b4_, Q3K_FN)((qs_u32 >> 4) & 0x03030303u);
    b.q6 = Q3K_PASTE(q3k_b4_, Q3K_FN)((qs_u32 >> 6) & 0x03030303u);

    // All 8 scales of the half: lo 4 bits from scales[j] >> v_im4, hi 2 bits from
    // scales[8 + j%4] >> (v_im4 + 2*(j/4)), biased by 32.
    const uint s0_3 = LD32(bo + 96u);
    const uint s4_7 = LD32(bo + 100u);
    const uint s8_11 = LD32(bo + 104u);
    const uint h01 = (s8_11 >> v_im4) & 0x03030303u;
    const uint h23 = (s8_11 >> (v_im4 + 2u)) & 0x03030303u;
    b.scA = Q3K_PASTE(q3k_b4_, Q3K_FN)(((s0_3 >> v_im4) & 0x0F0F0F0Fu) | (h01 << 4)) - 32.0;
    b.scB = Q3K_PASTE(q3k_b4_, Q3K_FN)(((s4_7 >> v_im4) & 0x0F0F0F0Fu) | (h23 << 4)) - 32.0;
    b.d = unpackHalf2x16(LD32(bo + 108u)).x;
    return b;
}

// Dot this thread's 16 elements against x at element offset xb, add to acc. Every float
// op is precise (same expression in decode and batched paths -> bitwise equal). Elements
// xb+{0,1,16,17,...,112,113}; two per sub-block, llama's exact fma nesting.
float Q3K_PASTE(q3k_apply_, Q3K_FN)(const Q3KBlock b, const uint xb, const float acc) {
    const float x0 = x[xb], x1 = x[xb + 1u], x2 = x[xb + 16u], x3 = x[xb + 17u];
    const float x4 = x[xb + 32u], x5 = x[xb + 33u], x6 = x[xb + 48u], x7 = x[xb + 49u];
    const float x8 = x[xb + 64u], x9 = x[xb + 65u], x10 = x[xb + 80u], x11 = x[xb + 81u];
    const float x12 = x[xb + 96u], x13 = x[xb + 97u], x14 = x[xb + 112u], x15 = x[xb + 113u];

    precise float sum = 0.0;
    // l = 0
    sum = fma(x0 * b.scA.x, b.q0.x - b.h0.x,
          fma(x2 * b.scA.y, b.q0.z - b.h0.z,
          fma(x4 * b.scA.z, b.q2.x - b.h1.x,
          fma(x6 * b.scA.w, b.q2.z - b.h1.z,
          fma(x8 * b.scB.x, b.q4.x - b.h2.x,
          fma(x10 * b.scB.y, b.q4.z - b.h2.z,
          fma(x12 * b.scB.z, b.q6.x - b.h3.x,
          fma(x14 * b.scB.w, b.q6.z - b.h3.z, sum))))))));
    // l = 1
    sum = fma(x1 * b.scA.x, b.q0.y - b.h0.y,
          fma(x3 * b.scA.y, b.q0.w - b.h0.w,
          fma(x5 * b.scA.z, b.q2.y - b.h1.y,
          fma(x7 * b.scA.w, b.q2.w - b.h1.w,
          fma(x9 * b.scB.x, b.q4.y - b.h2.y,
          fma(x11 * b.scB.y, b.q4.w - b.h2.w,
          fma(x13 * b.scB.z, b.q6.y - b.h3.y,
          fma(x15 * b.scB.w, b.q6.w - b.h3.w, sum))))))));
    precise float part = fma(b.d, sum, acc);
    return part;
}

void Q3K_PASTE(q3k_, Q3K_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint itid = tid & 15u;
    const uint v_im = itid >> 3;   // 0: elements 0.., 1: 128..
    const uint v_in = itid & 7u;
    const uint l0 = 2u * v_in;
    const uint q_off = 32u * v_im + l0;
    const uint y_off = 128u * v_im + l0;
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
            const Q3KBlock b = Q3K_PASTE(q3k_dequant_, Q3K_FN)(row_off + blk * 110u, v_im, v_in, q_off);
            acc1 = Q3K_PASTE(q3k_apply_, Q3K_FN)(b, pc.x_off + blk * 256u + y_off, acc1);
        }
        red[tid] = acc1;
        barrier();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            barrier();
        }
        if (tid == 0u) y[pc.y_off + row] = red[0];
        return;
    }

    // Batched form: dequantize once, apply every vector with the SAME expression (bitwise
    // VkHead gate). Uniform predicate: t >= n_vec lanes are inert.
    float acc[MAX_VEC] = float[MAX_VEC](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
        const Q3KBlock b = Q3K_PASTE(q3k_dequant_, Q3K_FN)(row_off + blk * 110u, v_im, v_in, q_off);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = Q3K_PASTE(q3k_apply_, Q3K_FN)(b, xb + t * pc.x_stride, acc[t]);
    }
    for (uint t = 0u; t < pc.n_vec; ++t) {
        red[tid] = acc[t];
        barrier();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            barrier();
        }
        if (tid == 0u) y[pc.y_off + t * pc.y_stride + row] = red[0];
        barrier();
    }
}

#undef Q3K_PASTE
#undef Q3K_PASTE2
