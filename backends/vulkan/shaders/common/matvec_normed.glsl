// HALO_FUSE_NORM prologue shared by the matmul/matvec_<type>_normed.comp shaders
// (y = W * (rms_norm(x) * nw), decode n_vec == 1 only). Include it after the shader's bindings
// and push block and before the per-type main: it needs `xr` (raw input), `nw` (norm weight),
// `WG` and `pc.{cols, xr_off, nw_off, eps}`, and it defines HALO_X for the main.
//
// The row RMS is computed once per workgroup by norm_prologue(), and the included main reads the
// normalized input x[c] = (xr[c] * inv_rms) * nw[c] through HALO_X(c), computed as it is read. A workgroup
// covers one row and reads each input element once, so staging the normalized row in shared memory
// (cols x 4 bytes: 20 KiB at the 27B's 5120) bought no reuse, capped cols by the shared-memory size and cut
// how many workgroups fit on a CU. The host sends x_off = 0, so the main's indices are element indices c.
#ifndef HALO_MATVEC_NORMED_GLSL
#define HALO_MATVEC_NORMED_GLSL

#include "rms_norm_tree.glsl"

const uint HALO_NORM_LANES = 256u;  // = rms_norm.comp's default WG (the host fuses only at 256)
shared float rn[HALO_NORM_LANES];
float g_inv_rms;  // set by norm_prologue() before the main runs; the same value in every invocation
float normed_x(const uint c) {
    return halo_rms_scale(xr[pc.xr_off + c], g_inv_rms, nw[pc.nw_off + c]);
}
#define HALO_X(c) normed_x(c)

// The same steps as rms_norm.comp with its default WG = 256, taken from common/rms_norm_tree.glsl:
// 256 virtual lanes (lane l sums elements l, l+256, ...) folded by the same fixed tree, so inv_rms is
// bitwise equal to the separate norm's when the host's reduce_workgroup is 256, and halo_rms_scale is
// the same x*inv then *w.
void norm_prologue() {
    const uint tid = gl_LocalInvocationID.x;
    for (uint l = tid; l < HALO_NORM_LANES; l += WG)
        HALO_RMS_LANE_SUM(rn, l, HALO_NORM_LANES, pc.cols, i, xr[pc.xr_off + i])
    HALO_RMS_TREE_REDUCE(rn, HALO_NORM_LANES, tid, WG)
    g_inv_rms = halo_rms_inv(rn[0], pc.cols, pc.eps);
}

#endif  // HALO_MATVEC_NORMED_GLSL
