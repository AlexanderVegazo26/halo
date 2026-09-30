// Per-type body of matvec_q5_k (ggml Q5_K GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_q5_k.comp). Port of llama.cpp's Vulkan
// mul_mat_vec_q5_k.comp (MIT) to HALO's binding model: 16 threads per 176-byte block,
// each thread dequantizes 16 elements with SWAR unpacks (scales for the whole sub-block
// group in ~8 ops, 4 nibbles + 5th bits per u32), and applies d/dmin/scales ONCE per
// 32-element sub-block instead of per element:
//   part = d * (sc0*sx + sc1*sy + sc4*sz + sc5*sw) - dmin * (m0*Sx + m1*Sy + m4*Sz + m5*Sw)
// where sx..sw are 4-element sums of q*x and Sx..Sw the matching sums of x.
// The float expression in q5k_apply is identical in the decode and batched paths and in
// both LD32 instantiations, so vector t of a batch, the n_vec = 1 matvec, and the
// dense/arena views are bitwise identical (VkHead / VkViews gates). The shared-memory
// reduction tree is the same one as matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at 4-aligned (fast) or arbitrary (slow) byte off
//   Q5K_FN        -> function-name token (pasted as q5k_<Q5K_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[] from the .comp.

#define Q5K_PASTE2(a, b) a##b
#define Q5K_PASTE(a, b) Q5K_PASTE2(a, b)

vec4 Q5K_PASTE(q5k_b4_, Q5K_FN)(const uint v) {
    return vec4(uvec4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24));
}

// SWAR dequant of this thread's 16 elements of the 176-byte block at byte offset bo.
Q5KBlock Q5K_PASTE(q5k_dequant_, Q5K_FN)(const uint bo, const uint v_im, const uint l0,
                                         const uint q_off) {
    Q5KBlock b;
    b.dm = unpackHalf2x16(LD32(bo));

    // Sub-block scales/mins for j in {2*v_im, 2*v_im+1, 2*v_im+4, 2*v_im+5}: llama's SWAR
    // (scale0/4/8 u16 lanes -> 4 scales + 4 mins in ~8 ops).
    const uint s0123 = LD32(bo + 4u);
    const uint s4567 = LD32(bo + 8u);
    const uint s89ab = LD32(bo + 12u);
    const uint sh16 = v_im << 4;
    const uint scale0_u32 = (s0123 >> sh16) & 0xFFFFu;
    const uint scale4_u32 = (s4567 >> sh16) & 0xFFFFu;
    const uint scale8_u32 = (s89ab >> sh16) & 0xFFFFu;
    const uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
    const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;
    b.sc_lo = Q5K_PASTE(q5k_b4_, Q5K_FN)(scale_0_4_l & 0x3F3F3F3Fu);
    b.sc_hi = Q5K_PASTE(q5k_b4_, Q5K_FN)((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0Fu) |
                                         scale_0_4_h);

    // qs/qh packed loads: q_off and l0 are 2-mod-4 for half the threads, so load the
    // containing u32 and shift (aligned path; the slow LD32 takes any byte offset).
    const uint qsh = (q_off & 2u) << 3;
    const uint qb = q_off & ~3u;
    const uint qs0 = LD32(bo + 48u + qb) >> qsh;
    const uint qs16 = LD32(bo + 48u + 16u + qb) >> qsh;
    const uint qs64 = LD32(bo + 48u + 64u + qb) >> qsh;
    const uint qs80 = LD32(bo + 48u + 80u + qb) >> qsh;
    const uint qs0_16_u32 = (qs0 & 0xFFFFu) | (qs16 << 16);
    const uint qs64_80_u32 = (qs64 & 0xFFFFu) | (qs80 << 16);
    const uint lsh = (l0 & 2u) << 3;
    const uint lb = l0 & ~3u;
    const uint qh = ((LD32(bo + 16u + lb) >> lsh) & 0xFFFFu) | ((LD32(bo + 32u + lb) >> lsh) << 16);

    const uint shv = 2u * v_im;
    const uint q_lo0 = (qs0_16_u32 & 0x0F0F0F0Fu) + (((qh >> shv) & 0x01010101u) << 4);
    const uint q_hi0 = ((qs0_16_u32 >> 4) & 0x0F0F0F0Fu) + (((qh >> shv) & 0x02020202u) << 3);
    const uint q_lo1 = (qs64_80_u32 & 0x0F0F0F0Fu) + ((qh >> shv) & 0x10101010u);
    const uint q_hi1 = ((qs64_80_u32 >> 4) & 0x0F0F0F0Fu) + (((qh >> shv) & 0x20202020u) >> 1);
    b.qA = Q5K_PASTE(q5k_b4_, Q5K_FN)(q_lo0);
    b.qB = Q5K_PASTE(q5k_b4_, Q5K_FN)(q_hi0);
    b.qC = Q5K_PASTE(q5k_b4_, Q5K_FN)(q_lo1);
    b.qD = Q5K_PASTE(q5k_b4_, Q5K_FN)(q_hi1);
    return b;
}

// Dot this thread's 16 dequantized elements against x at element offset xb, add to acc.
// Every float op is precise (same expression in decode and batched paths -> bitwise equal).
float Q5K_PASTE(q5k_apply_, Q5K_FN)(const Q5KBlock b, const uint xb, const float acc) {
    const float x0 = x[xb], x1 = x[xb + 1u], x2 = x[xb + 16u], x3 = x[xb + 17u];
    const float x4 = x[xb + 32u], x5 = x[xb + 33u], x6 = x[xb + 48u], x7 = x[xb + 49u];
    const float x8 = x[xb + 128u], x9 = x[xb + 129u], x10 = x[xb + 144u], x11 = x[xb + 145u];
    const float x12 = x[xb + 160u], x13 = x[xb + 161u], x14 = x[xb + 176u], x15 = x[xb + 177u];

    precise float sx = fma(x0, b.qA.x, fma(x1, b.qA.y, fma(x2, b.qA.z, x3 * b.qA.w)));
    precise float sy = fma(x4, b.qB.x, fma(x5, b.qB.y, fma(x6, b.qB.z, x7 * b.qB.w)));
    precise float sz = fma(x8, b.qC.x, fma(x9, b.qC.y, fma(x10, b.qC.z, x11 * b.qC.w)));
    precise float sw = fma(x12, b.qD.x, fma(x13, b.qD.y, fma(x14, b.qD.z, x15 * b.qD.w)));
    precise float smin = fma(x0 + x1 + x2 + x3, b.sc_lo.z,
                             fma(x4 + x5 + x6 + x7, b.sc_lo.w,
                                 fma(x8 + x9 + x10 + x11, b.sc_hi.z,
                                     (x12 + x13 + x14 + x15) * b.sc_hi.w)));
    precise float part = fma(b.dm.x, fma(sx, b.sc_lo.x, fma(sy, b.sc_lo.y, fma(sz, b.sc_hi.x, sw * b.sc_hi.y))),
                             fma(-b.dm.y, smin, acc));
    return part;
}

void Q5K_PASTE(q5k_, Q5K_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint itid = tid & 15u;
    const uint il = itid >> 2;
    const uint ir = itid & 3u;
    const uint v_im = il >> 1;
    const uint v_in = il & 1u;
    const uint l0 = 4u * ir + 2u * v_in;
    const uint q_off = 32u * v_im + l0;
    const uint y_off = 64u * v_im + l0;
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
            const Q5KBlock b = Q5K_PASTE(q5k_dequant_, Q5K_FN)(row_off + blk * 176u, v_im, l0, q_off);
            acc1 = Q5K_PASTE(q5k_apply_, Q5K_FN)(b, pc.x_off + blk * 256u + y_off, acc1);
        }
        // Single-wave workgroups + barrier-free subgroupAdd were tried (llama's structure):
        // decode measured ~4% slower than this tree at WG=256 (batched ~30% faster at
        // WG=64/128). Decode wins overall; keep the tree (identical order in both paths).
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
        const Q5KBlock b = Q5K_PASTE(q5k_dequant_, Q5K_FN)(row_off + blk * 176u, v_im, l0, q_off);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = Q5K_PASTE(q5k_apply_, Q5K_FN)(b, xb + t * pc.x_stride, acc[t]);
    }
    // One shared reduction tree for all n_vec vectors (same per-vector add order as a
    // serial tree, so results are bitwise unchanged; barriers drop from n_vec*(log2 WG+2)
    // to log2 WG+1).
    for (uint t = 0u; t < MAX_VEC; ++t)
        if (t < pc.n_vec) redb[t * WG + tid] = acc[t];
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            for (uint t = 0u; t < MAX_VEC; ++t)
                if (t < pc.n_vec) redb[t * WG + tid] += redb[t * WG + tid + s];
        }
        barrier();
    }
    if (tid == 0u) {
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) y[pc.y_off + t * pc.y_stride + row] = redb[t * WG];
    }
}

#undef Q5K_PASTE
#undef Q5K_PASTE2
