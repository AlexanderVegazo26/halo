// Per-type body of matvec_iq4_xs_rp: the IQ4_XS GEMV of matvec_iq4_xs_main.glsl over the
// REPACKED planar weight layout (include/halo/tensor/repack.h; opt-in HALO_REPACK=1). Same
// thread mapping (8 threads per block, one 32-element sub-block each), same words, same float
// expression in iq4_apply and the same reduction tree -- only the loads differ: a thread's 16
// qs bytes are ONE aligned uvec4 (in the 136-byte block they sat at byte 8 + 16 * ib32, i.e.
// only 8-aligned), and the 8 header bytes come from a separate uvec2 plane. Results are
// intended to be bitwise identical to matvec_iq4_xs (by construction; needs a device run
// against the raw kernel to confirm).
//
// Row layout (nb = cols / 256, bytes from the row start):
//   [0, 128 nb)        qs  : per block 8 x uvec4 (original bytes 8..135), thread ib32 at 16 * ib32
//   [128 nb, 136 nb)   hdr : per block uvec2 = original bytes 0..7 (d | scales_h, scales_l[4])
//
// Included once. Expects from the includer: the bindings wq2 / wq4 (same buffer, uvec2 /
// uvec4 views), push constants, BATCHED, MAX_VEC, red[] / redb[], sh_kv[], QI4Block, and
//   IQ4_FN        -> function-name token (pasted as iq4_<IQ4_FN>)

#define IQ4_PASTE2(a, b) a##b
#define IQ4_PASTE(a, b) IQ4_PASTE2(a, b)

// This thread's 32-element sub-block (ib32) of block blk of the row at byte offset row_off:
// 4 packed qs words + the hoisted d*(scale-32).
QI4Block IQ4_PASTE(iq4_dequant_, IQ4_FN)(const uint row_off, const uint blk, const uint nb, const uint ib32) {
    QI4Block b;
    const uvec2 hd = wq2[(row_off + 128u * nb + blk * 8u) >> 3];
    const uint w0 = hd.x;  // fp16 d | uint16 scales_h
    const uint w1 = hd.y;  // uint8 scales_l[4]
    const float d = unpackHalf2x16(w0).x;
    const uint sl = (w1 >> ((ib32 >> 1) * 8u + (ib32 & 1u) * 4u)) & 0xFu;
    const uint sh = ((w0 >> 16) >> (2u * ib32)) & 3u;
    b.dscale = d * float(int(sl | (sh << 4)) - 32);
    b.qsw = wq4[(row_off + blk * 128u + 16u * ib32) >> 4];
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
            const QI4Block b = IQ4_PASTE(iq4_dequant_, IQ4_FN)(row_off, blk, nblocks, ib32);
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
        const QI4Block b = IQ4_PASTE(iq4_dequant_, IQ4_FN)(row_off, blk, nblocks, ib32);
        const uint xb = pc.x_off + blk * 256u + 32u * ib32;
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] = IQ4_PASTE(iq4_apply_, IQ4_FN)(b, xb + t * pc.x_stride, acc[t]);
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

#undef IQ4_PASTE
#undef IQ4_PASTE2
