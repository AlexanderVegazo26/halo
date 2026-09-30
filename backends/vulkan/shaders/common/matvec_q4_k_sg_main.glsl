// HALO_SUBGROUP_REDUCE batched body of matvec_q4_k: see matvec_q5_k_sg_main.glsl.
#define Q4K_PASTE2(a, b) a##b
#define Q4K_PASTE(a, b) Q4K_PASTE2(a, b)

void Q4K_PASTE(q4k_sg_, Q4K_FN)(const uint row, const uint row_off) {
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

    for (uint t = 0u; t < NV; ++t) g_acc[t] = 0.0;
    for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
        const Q4KBlock b = Q4K_PASTE(q4k_dequant_, Q4K_FN)(row_off + blk * 144u, v_im, q_off);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < NV; ++t)
            g_acc[t] = Q4K_PASTE(q4k_apply_, Q4K_FN)(b, xb + t * pc.x_stride, g_acc[t]);
    }
    sg_finish(tid, row);
}

#undef Q4K_PASTE
#undef Q4K_PASTE2
