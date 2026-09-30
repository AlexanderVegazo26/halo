// Shared tail of the matvec_*_sg shaders (HALO_SUBGROUP_REDUCE=1): batched (n_vec > 1)
// cross-thread reduction with subgroupAdd instead of the shared-memory tree of
// matvec_*_main.glsl. Included ONCE by a matvec_<type>_sg.comp, after its bindings / push
// constants / WG, and before the per-type matvec_<type>_sg_main.glsl bodies.
//
// The includer must have `#extension GL_KHR_shader_subgroup_basic : enable` and
// `#extension GL_KHR_shader_subgroup_arithmetic : enable` before any other token.
//
// Correct for ANY workgroup size and subgroup size >= 32 (the host gates on that): each
// subgroup reduces its own lanes with subgroupAdd; when the workgroup holds more than one
// subgroup (wave32 device, or HALO_VK_GEMV_WG > the wave size) the per-subgroup sums go
// through a tiny shared array and thread 0 adds them in subgroup order. With WG = 64 on a
// wave64 device that second step never runs (gl_NumSubgroups == 1, no barrier at all).
//
// NUMERICS: the per-thread partials are bitwise the ones of the tree shaders (the dequant
// and apply functions are the very same ones); only the ORDER of the WG-wide sum differs
// (subgroupAdd's hardware order instead of the halving tree), so results agree with the
// tree shaders and with the n_vec == 1 matvec to a few ulps, not bitwise.

// Exact vector count of this dispatch (host: 2..8, one pipeline per count): every vector loop
// below has a compile-time trip count and unrolls, keeping acc[] in registers.
layout(constant_id = 2) const uint NV = 4;

const uint SG_MAX = (WG + 31u) / 32u;  // subgroups per workgroup, at most (subgroupSize >= 32)
shared float sg_part[NV * SG_MAX];

float g_acc[NV];  // this thread's per-vector accumulators

void sg_finish(const uint tid, const uint row) {
    float tot[NV];
    for (uint t = 0u; t < NV; ++t) tot[t] = subgroupAdd(g_acc[t]);
    if (gl_NumSubgroups == 1u) {
        if (subgroupElect()) {
            for (uint t = 0u; t < NV; ++t) y[pc.y_off + t * pc.y_stride + row] = tot[t];
        }
        return;
    }
    if (subgroupElect()) {
        for (uint t = 0u; t < NV; ++t) sg_part[t * SG_MAX + gl_SubgroupID] = tot[t];
    }
    barrier();
    if (tid == 0u) {
        for (uint t = 0u; t < NV; ++t) {
            float s = 0.0;
            for (uint i = 0u; i < gl_NumSubgroups; ++i) s += sg_part[t * SG_MAX + i];
            y[pc.y_off + t * pc.y_stride + row] = s;
        }
    }
}
