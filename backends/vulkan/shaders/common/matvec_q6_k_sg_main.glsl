// HALO_SUBGROUP_REDUCE batched body of matvec_q6_k: see matvec_q5_k_sg_main.glsl.
#define Q6K_PASTE2(a, b) a##b
#define Q6K_PASTE(a, b) Q6K_PASTE2(a, b)

void Q6K_PASTE(q6k_sg_, Q6K_FN)(const uint row, const uint row_off) {
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

    for (uint t = 0u; t < NV; ++t) g_acc[t] = 0.0;
    for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
        const Q6KBlock b = Q6K_PASTE(q6k_dequant_, Q6K_FN)(row_off + blk * 210u, ql_off, qh_off, v_im, is);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < NV; ++t)
            g_acc[t] = Q6K_PASTE(q6k_apply_, Q6K_FN)(b, xb + t * pc.x_stride, g_acc[t]);
    }
    sg_finish(tid, row);
}

#undef Q6K_PASTE
#undef Q6K_PASTE2
