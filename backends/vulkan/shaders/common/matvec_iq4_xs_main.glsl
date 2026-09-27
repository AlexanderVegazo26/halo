// Per-type body of matvec_iq4_xs (ggml IQ4_XS GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_iq4_xs.comp). Port of llama.cpp's Vulkan
// mul_mat_vec_iq4_xs.comp (MIT) to HALO's binding model, on the matvec_q5_k_main.glsl
// pattern: 8 threads per 136-byte block, each thread dequantizes one 32-element sub-block
// (4 packed qs words, SWAR nibble extract) and applies d*(scale-32) ONCE per sub-block:
//   part = dscale * sum32(kvalues_iq4nl[nibble] * x)
// The batched path keeps the raw qs words and redoes the nibble->kvalue expansion per
// vector (llama does the same for NUM_COLS > 1), keeping registers lean. The float
// expression in iq4_apply is identical in the decode and batched paths and in both LD32
// instantiations, so vector t of a batch, the n_vec = 1 matvec, and the dense/arena views
// are bitwise identical (VkHead / VkViews gates). The shared-memory reduction tree is the
// same one as matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at 4-aligned (fast) or arbitrary (slow) byte off
//   IQ4_FN        -> function-name token (pasted as iq4_<IQ4_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[] from the .comp.

#define IQ4_PASTE2(a, b) a##b
#define IQ4_PASTE(a, b) IQ4_PASTE2(a, b)

// This thread's 32-element sub-block (ib32) of the 136-byte block at byte offset bo:
// 4 packed qs words + the hoisted d*(scale-32).
QI4Block IQ4_PASTE(iq4_dequant_, IQ4_FN)(const uint bo, const uint ib32) {
    QI4Block b;
    const uint w0 = LD32(bo);        // fp16 d | uint16 scales_h
    const uint w1 = LD32(bo + 4u);   // uint8 scales_l[4]
    const float d = unpackHalf2x16(w0).x;
    const uint sl = (w1 >> ((ib32 >> 1) * 8u + (ib32 & 1u) * 4u)) & 0xFu;
    const uint sh = ((w0 >> 16) >> (2u * ib32)) & 3u;
    b.dscale = d * float(int(sl | (sh << 4)) - 32);
    b.qsw = uvec4(LD32(bo + 8u + 16u * ib32), LD32(bo + 12u + 16u * ib32),
                  LD32(bo + 16u + 16u * ib32), LD32(bo + 20u + 16u * ib32));
    return b;
}

// Dot this thread's 32 elements against x at element offset xb, add to acc. Every float
// op is precise (same expression in decode and batched paths -> bitwise equal).
float IQ4_PASTE(iq4_apply_, IQ4_FN)(const QI4Block b, const uint xb, const float acc) {
    precise float sum = 0.0;
    for (uint l = 0u; l < 4u; ++l) {
        const uint w = b.qsw[l];
        const uvec4 nib = uvec4(w, w >> 8u, w >> 16u, w >> 24u);
        const uvec4 q0 = nib & 0xFu;         // lo nibbles: elements 4l..4l+3
        const uvec4 q1 = (nib >> 4u) & 0xFu;  // hi nibbles: elements 16+4l..16+4l+3
        const float x0 = x[xb + 4u * l], x1 = x[xb + 4u * l + 1u];
        const float x2 = x[xb + 4u * l + 2u], x3 = x[xb + 4u * l + 3u];
        const float x4 = x[xb + 16u + 4u * l], x5 = x[xb + 16u + 4u * l + 1u];
        const float x6 = x[xb + 16u + 4u * l + 2u], x7 = x[xb + 16u + 4u * l + 3u];
        sum = fma(x0, float(sh_kv[q0.x]),
              fma(x1, float(sh_kv[q0.y]),
              fma(x2, float(sh_kv[q0.z]),
              fma(x3, float(sh_kv[q0.w]),
              fma(x4, float(sh_kv[q1.x]),
              fma(x5, float(sh_kv[q1.y]),
              fma(x6, float(sh_kv[q1.z]),
              fma(x7, float(sh_kv[q1.w]), sum))))))));
    }
    precise float part = fma(b.dscale, sum, acc);
    return part;
}

void IQ4_PASTE(iq4_, IQ4_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint ib32 = tid & 7u;   // 8 threads per block, one 32-element sub-block each
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 3; blk < nblocks; blk += WG / 8u) {
            const QI4Block b = IQ4_PASTE(iq4_dequant_, IQ4_FN)(row_off + blk * 136u, ib32);
            acc1 = IQ4_PASTE(iq4_apply_, IQ4_FN)(b, pc.x_off + blk * 256u + 32u * ib32, acc1);
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
        const QI4Block b = IQ4_PASTE(iq4_dequant_, IQ4_FN)(row_off + blk * 136u, ib32);
        const uint xb = pc.x_off + blk * 256u + 32u * ib32;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = IQ4_PASTE(iq4_apply_, IQ4_FN)(b, xb + t * pc.x_stride, acc[t]);
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

#undef IQ4_PASTE
#undef IQ4_PASTE2
