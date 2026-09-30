// Per-type body of matvec_iq3_s (ggml IQ3_S GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_iq3_s.comp). Port of llama.cpp's Vulkan
// mul_mat_vec_iq3_s.comp (MIT, TPB=8 / NUM_COLS=1 form) to HALO's binding model, on the
// matvec_q5_k_main.glsl pattern: 8 threads per 110-byte block, one 32-element group per
// thread; d*(1+2*scale) hoisted to one fma per 8-element l-group; grid lookups hit the
// workgroup-shared iq3s_grid_sh table (dequant_iq3_s.glsl; a const array indexed
// dynamically lowers to scratch and is ~100x slower). The batched path keeps the raw
// qs/qh/signs words and redoes the grid expansion per vector (same as iq4_xs). The float
// expression in iq3s_apply is identical in the decode and batched paths and in both LD32
// instantiations, so vector t of a batch, the n_vec = 1 matvec, and the dense/arena views
// are bitwise identical (VkHead / VkViews gates). The shared-memory reduction tree is the
// same one as matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at even (fast) or arbitrary (slow) byte offset
//   IQ3S_FN       -> function-name token (pasted as iq3s_<IQ3S_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[], iq3s_grid_sh[] from the
// .comp (via dequant_iq3_s.glsl).

#define IQ3S_PASTE2(a, b) a##b
#define IQ3S_PASTE(a, b) IQ3S_PASTE2(a, b)

vec4 IQ3S_PASTE(iq3s_b4_, IQ3S_FN)(const uint v) {
    return vec4(uvec4(v & 0xFFu, (v >> 8u) & 0xFFu, (v >> 16u) & 0xFFu, v >> 24u));
}

// This thread's 32-element group (grp = ib32) of the 110-byte block at byte offset bo.
// Layout: fp16 d @ 0, qs[64] @ 2, qh[8] @ 66, signs[32] @ 74, scales[4] @ 106.
QI3SBlock IQ3S_PASTE(iq3s_dequant_, IQ3S_FN)(const uint bo, const uint ib32) {
    QI3SBlock b;
    const float d = unpackHalf2x16(LD32(bo)).x;
    const uint sc4 = LD32(bo + 106u);
    const uint scale = (sc4 >> (8u * (ib32 >> 1) + 4u * (ib32 & 1u))) & 0xFu;
    b.dscale = d * float(1u + 2u * scale);
    b.qh = (LD32(bo + 66u + (ib32 & ~3u)) >> ((ib32 & 3u) << 3)) & 0xFFu;
    b.qsw = uvec2(LD32(bo + 2u + 8u * ib32), LD32(bo + 6u + 8u * ib32));
    b.signs = LD32(bo + 74u + 4u * ib32);
    return b;
}

// Dot this thread's 32 elements against x at element offset xb, add to acc. Every float
// op is precise (same expression in decode and batched paths -> bitwise equal). Per
// l-group (8 elements): two 9-bit grid indices (qs byte + qh bit), sign-selects, llama's
// exact fma nesting.
float IQ3S_PASTE(iq3s_apply_, IQ3S_FN)(const QI3SBlock b, const uint xb, float acc) {
    for (uint l = 0u; l < 4u; ++l) {
        const uint w = (l < 2u) ? b.qsw.x : b.qsw.y;
        const uint g0 = ((w >> (16u * (l & 1u))) & 0xFFu) | ((b.qh << (8u - 2u * l)) & 0x100u);
        const uint g1 = ((w >> (16u * (l & 1u) + 8u)) & 0xFFu) | ((b.qh << (7u - 2u * l)) & 0x100u);
        const vec4 grid0 = IQ3S_PASTE(iq3s_b4_, IQ3S_FN)(iq3s_grid_sh[g0]);
        const vec4 grid1 = IQ3S_PASTE(iq3s_b4_, IQ3S_FN)(iq3s_grid_sh[g1]);
        const uint sign = (b.signs >> (8u * l)) & 0xFFu;
        const float x0 = x[xb + 8u * l], x1 = x[xb + 8u * l + 1u];
        const float x2 = x[xb + 8u * l + 2u], x3 = x[xb + 8u * l + 3u];
        const float x4 = x[xb + 8u * l + 4u], x5 = x[xb + 8u * l + 5u];
        const float x6 = x[xb + 8u * l + 6u], x7 = x[xb + 8u * l + 7u];
        precise float sum =
            fma(x0, (sign & 1u) != 0u ? -grid0.x : grid0.x,
            fma(x1, (sign & 2u) != 0u ? -grid0.y : grid0.y,
            fma(x2, (sign & 4u) != 0u ? -grid0.z : grid0.z,
            fma(x3, (sign & 8u) != 0u ? -grid0.w : grid0.w,
            fma(x4, (sign & 16u) != 0u ? -grid1.x : grid1.x,
            fma(x5, (sign & 32u) != 0u ? -grid1.y : grid1.y,
            fma(x6, (sign & 64u) != 0u ? -grid1.z : grid1.z,
            fma(x7, (sign & 128u) != 0u ? -grid1.w : grid1.w, 0.0))))))));
        acc = fma(b.dscale, sum, acc);
    }
    return acc;
}

void IQ3S_PASTE(iq3s_, IQ3S_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint ib32 = tid & 7u;   // 8 threads per block, one 32-element group each
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 3; blk < nblocks; blk += WG / 8u) {
            const QI3SBlock b = IQ3S_PASTE(iq3s_dequant_, IQ3S_FN)(row_off + blk * 110u, ib32);
            acc1 = IQ3S_PASTE(iq3s_apply_, IQ3S_FN)(b, pc.x_off + blk * 256u + 32u * ib32, acc1);
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
    for (uint blk = tid >> 3; blk < nblocks; blk += WG / 8u) {
        const QI3SBlock b = IQ3S_PASTE(iq3s_dequant_, IQ3S_FN)(row_off + blk * 110u, ib32);
        const uint xb = pc.x_off + blk * 256u + 32u * ib32;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = IQ3S_PASTE(iq3s_apply_, IQ3S_FN)(b, xb + t * pc.x_stride, acc[t]);
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

#undef IQ3S_PASTE
#undef IQ3S_PASTE2
