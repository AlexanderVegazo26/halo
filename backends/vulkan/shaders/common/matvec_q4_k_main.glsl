// Per-type body of matvec_q4_k (ggml Q4_K GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_q4_k.comp). Port of llama.cpp's Vulkan
// mul_mat_vec_q4_k.comp (MIT) to HALO's binding model, on the matvec_q5_k_main.glsl
// pattern: 16 threads per 144-byte block, each thread dequantizes 16 elements with SWAR
// unpacks (scales for the whole sub-block group in ~8 ops, 4 nibbles per u32), and applies
// d/dmin/scales ONCE per 32-element sub-block instead of per element:
//   part = d * (sc0*sx + sc1*sy + sc4*sz + sc5*sw) - dmin * smin
// where sx..sw are 4-element sums of q*x and smin the min-weighted x sum.
// The float expression in q4k_apply is identical in the decode and batched paths and in
// both LD32 instantiations, so vector t of a batch, the n_vec = 1 matvec, and the
// dense/arena views are bitwise identical (VkHead / VkViews gates). The shared-memory
// reduction tree is the same one as matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at 4-aligned (fast) or arbitrary (slow) byte off
//   Q4K_FN        -> function-name token (pasted as q4k_<Q4K_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[] from the .comp.

#define Q4K_PASTE2(a, b) a##b
#define Q4K_PASTE(a, b) Q4K_PASTE2(a, b)

vec4 Q4K_PASTE(q4k_b4_, Q4K_FN)(const uint v) {
    return vec4(uvec4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24));
}

// SWAR dequant of this thread's 16 elements of the 144-byte block at byte offset bo.
// q_off (= 32*v_im + l0) is a multiple of 4, so the qs loads are directly 4-aligned.
Q4KBlock Q4K_PASTE(q4k_dequant_, Q4K_FN)(const uint bo, const uint v_im, const uint q_off) {
    Q4KBlock b;
    b.dm = unpackHalf2x16(LD32(bo));

    // Sub-block scales/mins for j in {2*v_im, 2*v_im+1, 2*v_im+4, 2*v_im+5}: llama's SWAR
    // (scale0/4/8 u16 lanes -> 4 scales + 4 mins in ~8 ops). sc_lo = sc0, sc1, m0, m1;
    // sc_hi = sc4, sc5, m4, m5.
    const uint s0123 = LD32(bo + 4u);
    const uint s4567 = LD32(bo + 8u);
    const uint s89ab = LD32(bo + 12u);
    const uint sh16 = v_im << 4;
    const uint scale0_u32 = (s0123 >> sh16) & 0xFFFFu;
    const uint scale4_u32 = (s4567 >> sh16) & 0xFFFFu;
    const uint scale8_u32 = (s89ab >> sh16) & 0xFFFFu;
    const uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
    const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;
    b.sc_lo = Q4K_PASTE(q4k_b4_, Q4K_FN)(scale_0_4_l & 0x3F3F3F3Fu);
    b.sc_hi = Q4K_PASTE(q4k_b4_, Q4K_FN)((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0Fu) |
                                         scale_0_4_h);

    // qs at byte 16: lo nibbles = elements y_off+{0..3} and +128, hi nibbles = +32 / +160.
    const uint qs0_u32 = LD32(bo + 16u + q_off);
    const uint qs64_u32 = LD32(bo + 80u + q_off);
    b.qA = Q4K_PASTE(q4k_b4_, Q4K_FN)(qs0_u32 & 0x0F0F0F0Fu);
    b.qB = Q4K_PASTE(q4k_b4_, Q4K_FN)((qs0_u32 >> 4) & 0x0F0F0F0Fu);
    b.qC = Q4K_PASTE(q4k_b4_, Q4K_FN)(qs64_u32 & 0x0F0F0F0Fu);
    b.qD = Q4K_PASTE(q4k_b4_, Q4K_FN)((qs64_u32 >> 4) & 0x0F0F0F0Fu);
    return b;
}

// Dot this thread's 16 dequantized elements against x at element offset xb, add to acc.
// Every float op is precise (same expression in decode and batched paths -> bitwise equal).
float Q4K_PASTE(q4k_apply_, Q4K_FN)(const Q4KBlock b, const uint xb, const float acc) {
    const float x0 = x[xb], x1 = x[xb + 1u], x2 = x[xb + 2u], x3 = x[xb + 3u];
    const float x4 = x[xb + 32u], x5 = x[xb + 33u], x6 = x[xb + 34u], x7 = x[xb + 35u];
    const float x8 = x[xb + 128u], x9 = x[xb + 129u], x10 = x[xb + 130u], x11 = x[xb + 131u];
    const float x12 = x[xb + 160u], x13 = x[xb + 161u], x14 = x[xb + 162u], x15 = x[xb + 163u];

    const float m0 = b.sc_lo.z, m1 = b.sc_lo.w, m4 = b.sc_hi.z, m5 = b.sc_hi.w;
    precise float sx = fma(x0, b.qA.x, fma(x1, b.qA.y, fma(x2, b.qA.z, x3 * b.qA.w)));
    precise float sy = fma(x4, b.qB.x, fma(x5, b.qB.y, fma(x6, b.qB.z, x7 * b.qB.w)));
    precise float sz = fma(x8, b.qC.x, fma(x9, b.qC.y, fma(x10, b.qC.z, x11 * b.qC.w)));
    precise float sw = fma(x12, b.qD.x, fma(x13, b.qD.y, fma(x14, b.qD.z, x15 * b.qD.w)));
    precise float smin = fma(x0, m0, fma(x4, m1, fma(x8, m4, fma(x12, m5,
                         fma(x1, m0, fma(x5, m1, fma(x9, m4, fma(x13, m5,
                         fma(x2, m0, fma(x6, m1, fma(x10, m4, fma(x14, m5,
                         fma(x3, m0, fma(x7, m1, fma(x11, m4, x15 * m5)))))))))))))));
    precise float part = fma(b.dm.x, fma(sx, b.sc_lo.x, fma(sy, b.sc_lo.y, fma(sz, b.sc_hi.x, sw * b.sc_hi.y))),
                             fma(-b.dm.y, smin, acc));
    return part;
}

void Q4K_PASTE(q4k_, Q4K_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint itid = tid & 15u;
    const uint il = itid >> 2;
    const uint ir = itid & 3u;
    const uint v_im = il >> 1;
    const uint v_in = il & 1u;
    const uint l0 = 4u * (2u * ir + v_in);
    const uint q_off = 32u * v_im + l0;
    const uint y_off = 64u * v_im + l0;
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
            const Q4KBlock b = Q4K_PASTE(q4k_dequant_, Q4K_FN)(row_off + blk * 144u, v_im, q_off);
            acc1 = Q4K_PASTE(q4k_apply_, Q4K_FN)(b, pc.x_off + blk * 256u + y_off, acc1);
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
        const Q4KBlock b = Q4K_PASTE(q4k_dequant_, Q4K_FN)(row_off + blk * 144u, v_im, q_off);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = Q4K_PASTE(q4k_apply_, Q4K_FN)(b, xb + t * pc.x_stride, acc[t]);
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

#undef Q4K_PASTE
#undef Q4K_PASTE2
