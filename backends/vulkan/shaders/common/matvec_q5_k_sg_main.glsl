// HALO_SUBGROUP_REDUCE batched body of matvec_q5_k: same thread mapping, dequant and apply as
// the batched branch of matvec_q5_k_main.glsl (which the .comp includes unmodified just before
// this file, for q5k_dequant_<FN> / q5k_apply_<FN>), then the subgroupAdd tail
// (matvec_sg_reduce.glsl). Included twice (FAST / SLOW) like the original; same includer contract.
#define Q5K_PASTE2(a, b) a##b
#define Q5K_PASTE(a, b) Q5K_PASTE2(a, b)

void Q5K_PASTE(q5k_sg_, Q5K_FN)(const uint row, const uint row_off) {
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

    for (uint t = 0u; t < NV; ++t) g_acc[t] = 0.0;
    for (uint blk = tid >> 4; blk < nblocks; blk += WG / 16u) {
        const Q5KBlock b = Q5K_PASTE(q5k_dequant_, Q5K_FN)(row_off + blk * 176u, v_im, l0, q_off);
        const uint xb = pc.x_off + blk * 256u + y_off;
        for (uint t = 0u; t < NV; ++t)
            g_acc[t] = Q5K_PASTE(q5k_apply_, Q5K_FN)(b, xb + t * pc.x_stride, g_acc[t]);
    }
    sg_finish(tid, row);
}

#undef Q5K_PASTE
#undef Q5K_PASTE2
