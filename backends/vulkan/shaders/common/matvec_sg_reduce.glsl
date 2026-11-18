// Shared tail of the matvec_*_sg shaders (HALO_SUBGROUP_REDUCE=1; every chunk, n_vec 1..8):
// cross-thread reduction with subgroupAdd instead of the shared-memory tree of
// matvec_*_main.glsl. Included ONCE by matvec_<type>.comp when compiled with
// -DHALO_SUBGROUP_REDUCE (the matvec_<type>_sg variant in backends/vulkan/CMakeLists.txt), after
// its bindings / push constants / WG, and before the per-type matvec_<type>_sg_main.glsl bodies.
//
// The includer must enable GL_KHR_shader_subgroup_basic and GL_KHR_shader_subgroup_arithmetic
// before any other token (common/matvec_variant_prologue.glsl does).
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
// tree shaders to a few ulps, not bitwise. A vector's arithmetic doesn't depend on NV, so a batch
// row is still bitwise its own n_vec == 1 _sg matvec (the host never mixes the two, ops.cpp).

// Exact vector count of this dispatch (host: 1..8, one pipeline per count): every vector loop
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
