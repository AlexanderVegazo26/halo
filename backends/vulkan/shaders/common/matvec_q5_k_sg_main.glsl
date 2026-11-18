// HALO_SUBGROUP_REDUCE entry of matvec_q5_k (every n_vec, 1..NV): same thread mapping, dequant and
// apply as the batched branch of the tree entry in matvec_q5_k_main.glsl, then the subgroupAdd tail
// (matvec_sg_reduce.glsl). matvec_q5_k_main.glsl includes this file at its end, after
// q5k_dequant_<FN> / q5k_apply_<FN>, and compiles no tree entry in this build. Included twice
// (FAST / SLOW) like the main. Includer contract: the bindings, push constants and WG from the
// .comp, LD32 / Q5K_FN, and NV, g_acc[], sg_finish() from matvec_sg_reduce.glsl; the .comp
// declares no BATCHED, MAX_VEC, red[] or redb[] in this build.
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
