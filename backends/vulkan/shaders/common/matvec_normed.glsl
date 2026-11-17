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

shared float rn[256];
float g_inv_rms;  // set by norm_prologue() before the main runs; the same value in every invocation
float normed_x(const uint c) {
    precise float r = (xr[pc.xr_off + c] * g_inv_rms) * nw[pc.nw_off + c];
    return r;
}
#define HALO_X(c) normed_x(c)

// Same arithmetic as rms_norm.comp with its default WG = 256: 256 virtual lanes (lane l sums
// elements l, l+256, ...) reduced by the same fixed tree, so inv_rms is bitwise equal to the
// separate norm's when the host's reduce_workgroup is 256; x*inv then *w is the same op order. All of it
// is `precise`, as in rms_norm.comp: otherwise the compiler builds the two differently (1 ulp apart).
void norm_prologue() {
    const uint tid = gl_LocalInvocationID.x;
    for (uint l = tid; l < 256u; l += WG) {
        precise float a = 0.0;
        for (uint i = l; i < pc.cols; i += 256u) {
            const float v = xr[pc.xr_off + i];
            a += v * v;
        }
        rn[l] = a;
    }
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        for (uint l = tid; l < s; l += WG) rn[l] += rn[l + s];
        barrier();
    }
    precise float inv = 1.0 / sqrt(rn[0] / float(pc.cols) + pc.eps);
    g_inv_rms = inv;
}

#endif  // HALO_MATVEC_NORMED_GLSL
