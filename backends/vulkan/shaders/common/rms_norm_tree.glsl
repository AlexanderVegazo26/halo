// The RMS-norm arithmetic shared by norm/rms_norm.comp, norm/add_rms_norm.comp,
// norm/gated_rms_norm.comp (its reduction and inv only) and common/matvec_normed.glsl
// (HALO_FUSE_NORM). The fused gemv must produce bit for bit the row the separate norm produces
// (VkHead.GemvFusedNorm*), so they take each step from here instead of keeping hand copies in step:
//   lane l of LANES sums x[i]^2 for i = l, l + LANES, ...      (halo_rms_sq_add, in that order)
//   red[0..LANES) is folded by the fixed halving tree           (HALO_RMS_TREE_REDUCE)
//   inv = 1 / sqrt(red[0] / cols + eps)                         (halo_rms_inv)
//   y[i] = (x[i] * inv) * w[i]                                  (halo_rms_scale, = cpu::rms_norm)
// Every step is `precise`: left free, the compiler contracted and rewrote the same GLSL differently
// in different shaders (1 ulp apart).
#ifndef HALO_RMS_NORM_TREE_GLSL
#define HALO_RMS_NORM_TREE_GLSL

float halo_rms_sq_add(const float acc, const float v) {
    precise float r = acc + v * v;
    return r;
}

// One step of the fixed halving tree: red[l] += red[l + s]. The add is `precise` itself: glslang
// marks it only by tracing back from a precise use of red[0] in the same function, and red[0]
// reaches halo_rms_inv() as an argument, which that trace does not follow.
// The macros below are blocks with halo_rms_-prefixed locals, so they are safe as an unbraced
// if/for body and don't capture a caller's names.
#define HALO_RMS_TREE_ADD(red, l, s)                                     \
    {                                                                    \
        precise float halo_rms_t_ = red[l] + red[(l) + (s)];             \
        red[l] = halo_rms_t_;                                            \
    }

// Folds red[0..LANES) into red[0] (LANES a power of two) by the fixed halving tree; call it after
// red[] is written, it ends with a barrier. Invocation `tid` of a WG-wide workgroup adds lanes tid,
// tid + WG, ... below each half, so LANES may exceed WG (matvec_normed.glsl: HALO_NORM_LANES lanes
// over the matvec's workgroup).
#define HALO_RMS_TREE_REDUCE(red, LANES, tid, WG)                                        \
    {                                                                                    \
        barrier();                                                                       \
        for (uint halo_rms_s_ = (LANES) / 2u; halo_rms_s_ > 0u; halo_rms_s_ >>= 1) {     \
            for (uint halo_rms_l_ = (tid); halo_rms_l_ < halo_rms_s_; halo_rms_l_ += (WG)) \
                HALO_RMS_TREE_ADD(red, halo_rms_l_, halo_rms_s_)                         \
            barrier();                                                                   \
        }                                                                                \
    }

// The same tree when LANES == WG (one lane per invocation: the norm/ shaders), without the lane
// loop, so those kernels keep their one-add-per-invocation code. Same adds in the same pairs through
// the same HALO_RMS_TREE_ADD, so red[0] is bitwise HALO_RMS_TREE_REDUCE(red, WG, tid, WG).
#define HALO_RMS_TREE_REDUCE_WG(red, tid, WG)                                            \
    {                                                                                    \
        barrier();                                                                       \
        for (uint halo_rms_s_ = (WG) / 2u; halo_rms_s_ > 0u; halo_rms_s_ >>= 1) {        \
            if ((tid) < halo_rms_s_) HALO_RMS_TREE_ADD(red, (tid), halo_rms_s_)          \
            barrier();                                                                   \
        }                                                                                \
    }

float halo_rms_inv(const float sum_sq, const uint cols, const float eps) {
    precise float inv = 1.0 / sqrt(sum_sq / float(cols) + eps);
    return inv;
}

float halo_rms_scale(const float x, const float inv, const float w) {
    precise float r = (x * inv) * w;
    return r;
}

#endif  // HALO_RMS_NORM_TREE_GLSL
