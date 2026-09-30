// Per-type body of matvec_q6_k (ggml Q6_K GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_q6_k.comp). Port of llama.cpp's Vulkan
// mul_mat_vec_q6_k.comp (MIT) to HALO's binding model, on the matvec_q5_k_main.glsl
// pattern: 16 threads per 210-byte block, each thread dequantizes 16 elements with SWAR
// unpacks (4 nibbles + two 2-bit high parts per u32) and applies d and the four 16-element
// sub-block scales ONCE per group instead of per element:
//   part = d * (sc0*sum0 + sc2*sum1 + sc4*sum2 + sc6*sum3)
// Unlike llama this loads the four needed int8 scales directly (2 u32 loads) instead of
// staging them through a shared sccache + barrier. The float expression in q6k_apply is
// identical in the decode and batched paths and in both LD32 instantiations, so vector t
// of a batch, the n_vec = 1 matvec, and the dense/arena views are bitwise identical
// (VkHead / VkViews gates). The shared-memory reduction tree is the same one as
// matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at 4-aligned (fast) or arbitrary (slow) byte off
//   Q6K_FN        -> function-name token (pasted as q6k_<Q6K_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[] from the .comp.

#define Q6K_PASTE2(a, b) a##b
#define Q6K_PASTE(a, b) Q6K_PASTE2(a, b)

vec4 Q6K_PASTE(q6k_b4_, Q6K_FN)(const uint v) {
    return vec4(uvec4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24));
}

// SWAR dequant of this thread's 16 elements of the 210-byte block at byte offset bo.
// ql_offset/qh_offset are multiples of 4, so the loads are directly 4-aligned.
Q6KBlock Q6K_PASTE(q6k_dequant_, Q6K_FN)(const uint bo, const uint ql_off, const uint qh_off,
                                         const uint v_im, const uint is) {
    Q6KBlock b;
    // ql at byte 0, qh at byte 128, scales at 192, d at 208.
    const uint ql0_u32 = LD32(bo + ql_off);
    const uint ql32_u32 = LD32(bo + ql_off + 32u);
    const uint qh_u32 = LD32(bo + 128u + qh_off);

    const uint q0_u32 = (ql0_u32 & 0x0F0F0F0Fu) | ((qh_u32 & 0x03030303u) << 4);
    const uint q1_u32 = (ql32_u32 & 0x0F0F0F0Fu) | ((qh_u32 & 0x0C0C0C0Cu) << 2);
    const uint q2_u32 = ((ql0_u32 >> 4) & 0x0F0F0F0Fu) | (qh_u32 & 0x30303030u);
    const uint q3_u32 = ((ql32_u32 >> 4) & 0x0F0F0F0Fu) | ((qh_u32 & 0xC0C0C0C0u) >> 2);
    b.qA = Q6K_PASTE(q6k_b4_, Q6K_FN)(q0_u32) - 32.0;  // elements y_off+{0..3}
    b.qB = Q6K_PASTE(q6k_b4_, Q6K_FN)(q1_u32) - 32.0;  // +32
    b.qC = Q6K_PASTE(q6k_b4_, Q6K_FN)(q2_u32) - 32.0;  // +64
    b.qD = Q6K_PASTE(q6k_b4_, Q6K_FN)(q3_u32) - 32.0;  // +96

    // int8 scales {s_offset, +2, +4, +6}, s_offset = 8*v_im + is: bytes 192+8*v_im+is and
    // +2 in the two u32 words at 192+8*v_im / 196+8*v_im.
    const uint wa = LD32(bo + 192u + 8u * v_im);
    const uint wb = LD32(bo + 196u + 8u * v_im);
    const int ish = int(is) << 3;
    b.sc = vec4(float(bitfieldExtract(int(wa), ish, 8)), float(bitfieldExtract(int(wa), ish + 16, 8)),
                float(bitfieldExtract(int(wb), ish, 8)), float(bitfieldExtract(int(wb), ish + 16, 8)));
    b.d = unpackHalf2x16(LD32(bo + 208u)).x;
    return b;
}

// Dot this thread's 16 dequantized elements against x at element offset xb, add to acc.
// Every float op is precise (same expression in decode and batched paths -> bitwise equal).
float Q6K_PASTE(q6k_apply_, Q6K_FN)(const Q6KBlock b, const uint xb, const float acc) {
    const float x0 = x[xb], x1 = x[xb + 1u], x2 = x[xb + 2u], x3 = x[xb + 3u];
    const float x4 = x[xb + 32u], x5 = x[xb + 33u], x6 = x[xb + 34u], x7 = x[xb + 35u];
    const float x8 = x[xb + 64u], x9 = x[xb + 65u], x10 = x[xb + 66u], x11 = x[xb + 67u];
    const float x12 = x[xb + 96u], x13 = x[xb + 97u], x14 = x[xb + 98u], x15 = x[xb + 99u];

    precise float sum0 = fma(x3, b.qA.w, fma(x2, b.qA.z, fma(x1, b.qA.y, fma(x0, b.qA.x, 0.0))));
    precise float sum1 = fma(x7, b.qB.w, fma(x6, b.qB.z, fma(x5, b.qB.y, fma(x4, b.qB.x, 0.0))));
    precise float sum2 = fma(x11, b.qC.w, fma(x10, b.qC.z, fma(x9, b.qC.y, fma(x8, b.qC.x, 0.0))));
    precise float sum3 = fma(x15, b.qD.w, fma(x14, b.qD.z, fma(x13, b.qD.y, fma(x12, b.qD.x, 0.0))));
    precise float part = fma(fma(sum0, b.sc.x, fma(sum1, b.sc.y, fma(sum2, b.sc.z, sum3 * b.sc.w))),
                             b.d, acc);
    return part;
}

void Q6K_PASTE(q6k_, Q6K_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint itid = tid & 15u;
    const uint v_im = itid >> 3;   // 0: elements 0.., 1: 128..
    const uint v_in = itid & 7u;
    const uint l0 = 4u * v_in;
    const uint is = v_in >> 2;
    const uint ql_off = 64u * v_im + l0;
    const uint qh_off = 32u * v_im + l0;
    const uint y_off = 128u * v_im + l0;
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
            const Q6KBlock b =
                Q6K_PASTE(q6k_dequant_, Q6K_FN)(row_off + blk * 210u, ql_off, qh_off, v_im, is);
            acc1 = Q6K_PASTE(q6k_apply_, Q6K_FN)(b, pc.x_off + blk * 256u + y_off, acc1);
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
        const Q6KBlock b = Q6K_PASTE(q6k_dequant_, Q6K_FN)(row_off + blk * 210u, ql_off, qh_off, v_im, is);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = Q6K_PASTE(q6k_apply_, Q6K_FN)(b, xb + t * pc.x_stride, acc[t]);
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

#undef Q6K_PASTE
#undef Q6K_PASTE2
