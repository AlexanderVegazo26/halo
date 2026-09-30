// HALO_SUBGROUP_REDUCE batched body of matvec_iq4_xs: see matvec_q5_k_sg_main.glsl.
#define IQ4_PASTE2(a, b) a##b
#define IQ4_PASTE(a, b) IQ4_PASTE2(a, b)

void IQ4_PASTE(iq4_sg_, IQ4_FN)(const uint row, const uint row_off) {
    const uint tid = gl_LocalInvocationID.x;
    const uint ib32 = tid & 7u;   // 8 threads per block, one 32-element sub-block each
    const uint nblocks = pc.cols >> 8;

    for (uint t = 0u; t < NV; ++t) g_acc[t] = 0.0;
    for (uint blk = tid >> 3; blk < nblocks; blk += WG / 8u) {
        const QI4Block b = IQ4_PASTE(iq4_dequant_, IQ4_FN)(row_off + blk * 136u, ib32);
        const uint xb = pc.x_off + blk * 256u + 32u * ib32;
        for (uint t = 0u; t < NV; ++t)
            g_acc[t] = IQ4_PASTE(iq4_apply_, IQ4_FN)(b, xb + t * pc.x_stride, g_acc[t]);
    }
    sg_finish(tid, row);
}

#undef IQ4_PASTE
#undef IQ4_PASTE2
