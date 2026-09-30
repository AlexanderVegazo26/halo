// Per-type body of matvec_q6_k_rp: the Q6_K GEMV of matvec_q6_k_main.glsl over the REPACKED
// planar weight layout (include/halo/tensor/repack.h; opt-in HALO_REPACK=1). Same thread
// mapping (16 threads per block), same dequantized integers, same float expression in
// q6k_apply and the same reduction tree -- only the loads differ. The 210-byte blocks, and
// with them the funnel-shift ld32e path, are gone: a thread's ql words are one aligned
// uvec2, its qh word one aligned uint, the block's 16 scale bytes one uvec4, and d comes
// from a separate f16 plane. Results are intended to be bitwise identical to matvec_q6_k
// (by construction; needs a device run against the raw kernel to confirm).
//
// Row layout (nb = cols / 256, bytes from the row start):
//   [0, 128 nb)        ql' : per block 16 x uvec2, thread t = itid at 8 * itid
//                            (.x = ql[64 v_im + l0 ..+3], .y = ql[64 v_im + l0 + 32 ..+35])
//   [128 nb, 192 nb)   qh  : per block 16 x u32 at 4 * itid (original bytes 128..191)
//   [192 nb, 208 nb)   sc  : per block uvec4 = original bytes 192..207
//   [208 nb, 210 nb)   d   : f16 per block (2 * blk)
//
// Included once. Expects from the includer: the bindings wq / wq2 / wq4 (same buffer, u32 /
// uvec2 / uvec4 views), push constants, BATCHED, MAX_VEC, red[] / redb[], Q6KBlock, and
//   Q6K_FN        -> function-name token (pasted as q6k_<Q6K_FN>)

#define Q6K_PASTE2(a, b) a##b
#define Q6K_PASTE(a, b) Q6K_PASTE2(a, b)

vec4 Q6K_PASTE(q6k_b4_, Q6K_FN)(const uint v) {
    return vec4(uvec4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24));
}

// Unpack this thread's 16 elements of block blk of the row at byte offset row_off.
Q6KBlock Q6K_PASTE(q6k_dequant_, Q6K_FN)(const uint row_off, const uint blk, const uint nb, const uint itid,
                                         const uint v_im, const uint is) {
    Q6KBlock b;
    const uvec2 qlv = wq2[(row_off + blk * 128u + itid * 8u) >> 3];
    const uint ql0_u32 = qlv.x;
    const uint ql32_u32 = qlv.y;
    const uint qh_u32 = wq[(row_off + 128u * nb + blk * 64u + itid * 4u) >> 2];

    const uint q0_u32 = (ql0_u32 & 0x0F0F0F0Fu) | ((qh_u32 & 0x03030303u) << 4);
    const uint q1_u32 = (ql32_u32 & 0x0F0F0F0Fu) | ((qh_u32 & 0x0C0C0C0Cu) << 2);
    const uint q2_u32 = ((ql0_u32 >> 4) & 0x0F0F0F0Fu) | (qh_u32 & 0x30303030u);
    const uint q3_u32 = ((ql32_u32 >> 4) & 0x0F0F0F0Fu) | ((qh_u32 & 0xC0C0C0C0u) >> 2);
    b.qA = Q6K_PASTE(q6k_b4_, Q6K_FN)(q0_u32) - 32.0;  // elements y_off+{0..3}
    b.qB = Q6K_PASTE(q6k_b4_, Q6K_FN)(q1_u32) - 32.0;  // +32
    b.qC = Q6K_PASTE(q6k_b4_, Q6K_FN)(q2_u32) - 32.0;  // +64
    b.qD = Q6K_PASTE(q6k_b4_, Q6K_FN)(q3_u32) - 32.0;  // +96

    // int8 scales {s_offset, +2, +4, +6}, s_offset = 8*v_im + is: bytes 8*v_im+is and +2 in
    // the two u32 words 2*v_im / 2*v_im+1 of the block's 16 scale bytes.
    const uvec4 scv = wq4[(row_off + 192u * nb + blk * 16u) >> 4];
    const uint wa = v_im == 0u ? scv.x : scv.z;
    const uint wb = v_im == 0u ? scv.y : scv.w;
    const int ish = int(is) << 3;
    b.sc = vec4(float(bitfieldExtract(int(wa), ish, 8)), float(bitfieldExtract(int(wa), ish + 16, 8)),
                float(bitfieldExtract(int(wb), ish, 8)), float(bitfieldExtract(int(wb), ish + 16, 8)));
    // d: f16 at byte 208 nb + 2 blk of the row (even; both nb * 208 and the row start are 16-aligned).
    const uint dbo = row_off + 208u * nb + blk * 2u;
    b.d = unpackHalf2x16(wq[dbo >> 2] >> ((dbo & 2u) << 3)).x;
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
    const uint y_off = 128u * v_im + l0;
    const uint nblocks = pc.cols >> 8;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        float acc1 = 0.0;
        for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
            const Q6KBlock b = Q6K_PASTE(q6k_dequant_, Q6K_FN)(row_off, blk, nblocks, itid, v_im, is);
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
        const Q6KBlock b = Q6K_PASTE(q6k_dequant_, Q6K_FN)(row_off, blk, nblocks, itid, v_im, is);
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
