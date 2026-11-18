// Bindings and push block of the per-type SWAR matvec shaders (matmul/matvec_{q3_k,q4_k,q5_k,q6_k,
// iq3_s,iq4_nl,iq4_xs}.comp). Not used by the F32/Q8_0 shaders (Wq in the .comp, X/Y + push in
// matvec_quant_main.glsl, with the _gates tail) nor the _rp shaders (uvec2/uvec4 weight views). One
// copy, so every variant of these 7 types matches the host's push structs in backends/vulkan/src/ops.cpp
// field for field:
//   plain / HALO_SUBGROUP_REDUCE build: the matvec push (common/matvec_push_core.glsl);
//   HALO_FUSE_NORM build (matvec_<type>_normed): x is replaced by the raw row xr (binding 1) and the
//   norm weight nw (binding 3), and the push block gains ops.cpp's NPush tail (xr_off, nw_off, eps).
// Include it before anything that reads pc, x, xr or nw (matvec_normed.glsl, the per-type mains).
#ifndef HALO_MATVEC_IO_GLSL
#define HALO_MATVEC_IO_GLSL

layout(std430, binding = 0) readonly buffer Wq { uint wq[]; };
#ifdef HALO_FUSE_NORM
layout(std430, binding = 1) readonly buffer XR { float xr[]; };  // raw (un-normed) input
layout(std430, binding = 3) readonly buffer NW { float nw[]; };  // norm weight
#else
layout(std430, binding = 1) readonly buffer X { float x[]; };
#endif
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
#include "matvec_push_core.glsl"
#ifdef HALO_FUSE_NORM
    uint xr_off;    // element remainder of the raw x view
    uint nw_off;    // element remainder of the norm-weight view
    float eps;
#endif
} pc;

#endif  // HALO_MATVEC_IO_GLSL
