// Per-type body of matvec_iq4_nl (ggml IQ4_NL GEMV), instantiated twice by the .comp with
// different LD32 definitions (see matvec_iq4_nl.comp). Follows the matvec_iq4_xs_main.glsl
// port (llama.cpp's iq4_xs mat-vec, MIT) with IQ4_NL's simpler block: { fp16 d; uint8
// qs[16]; } = 18 bytes / 32 values, no scales: 8 threads per 256-value superblock, one
// 32-element block per thread, d hoisted to a single fma after register accumulation:
//   part = d * sum32(kvalues_iq4nl[nibble] * x)
// The batched path keeps the raw qs words and redoes the nibble->kvalue expansion per
// vector (same as iq4_xs). The float expression in iq4nl_apply is identical in the decode
// and batched paths and in both LD32 instantiations, so vector t of a batch, the n_vec = 1
// matvec, and the dense/arena views are bitwise identical (VkHead / VkViews gates). The
// shared-memory reduction tree is the same one as matvec_quant_main.glsl.
//
// Included twice WITHOUT an include guard. Expects from the includer:
//   LD32(bo)      -> uint: 4 weight bytes at even (fast) or arbitrary (slow) byte offset
//   IQ4NL_FN      -> function-name token (pasted as iq4nl_<IQ4NL_FN>)
// plus the bindings, push constants, BATCHED, MAX_VEC, red[], sh_kv[] from the .comp.

#define IQ4NL_PASTE2(a, b) a##b
#define IQ4NL_PASTE(a, b) IQ4NL_PASTE2(a, b)

// This thread's 32-element block (18 bytes) at byte offset bo: 4 packed qs words + d.
QI4NLBlock IQ4NL_PASTE(iq4nl_dequant_, IQ4NL_FN)(const uint bo) {
    QI4NLBlock b;
    b.d = unpackHalf2x16(LD32(bo)).x;
    // qs at byte 2: 2-mod-4 within the block; the fast LD32 funnel-shifts.
    b.qsw = uvec4(LD32(bo + 2u), LD32(bo + 6u), LD32(bo + 10u), LD32(bo + 14u));
    return b;
}

// Dot this thread's 32 elements against x at element offset xb, add to acc. Every float
// op is precise (same expression in decode and batched paths -> bitwise equal).
float IQ4NL_PASTE(iq4nl_apply_, IQ4NL_FN)(const QI4NLBlock b, const uint xb, const float acc) {
    precise float sum = 0.0;
    for (uint l = 0u; l < 4u; ++l) {
        const uint w = b.qsw[l];
        const uvec4 nib = uvec4(w, w >> 8u, w >> 16u, w >> 24u);
        const uvec4 q0 = nib & 0xFu;          // lo nibbles: elements 4l..4l+3
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
    precise float part = fma(b.d, sum, acc);
    return part;
}

void IQ4NL_PASTE(iq4nl_, IQ4NL_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint ib32 = tid & 7u;   // 8 threads per 256-value superblock, one block each
    const uint nblocks = pc.cols >> 5;  // 32-value blocks

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint g = tid >> 3; g * 8u < nblocks; g += WG / 8u) {
            const uint gb = g * 8u + ib32;
            if (gb < nblocks) {
                const QI4NLBlock b = IQ4NL_PASTE(iq4nl_dequant_, IQ4NL_FN)(row_off + gb * 18u);
                acc1 = IQ4NL_PASTE(iq4nl_apply_, IQ4NL_FN)(b, pc.x_off + gb * 32u, acc1);
            }
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
    for (uint g = tid >> 3; g * 8u < nblocks; g += WG / 8u) {
        const uint gb = g * 8u + ib32;
        if (gb < nblocks) {
            const QI4NLBlock b = IQ4NL_PASTE(iq4nl_dequant_, IQ4NL_FN)(row_off + gb * 18u);
            const uint xb = pc.x_off + gb * 32u;
            for (uint t = 0u; t < MAX_VEC; ++t)
                if (t < pc.n_vec)
                    acc[t] = IQ4NL_PASTE(iq4nl_apply_, IQ4NL_FN)(b, xb + t * pc.x_stride, acc[t]);
        }
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

#undef IQ4NL_PASTE
#undef IQ4NL_PASTE2
