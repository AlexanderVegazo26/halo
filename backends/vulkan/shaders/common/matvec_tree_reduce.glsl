// Shared tail of the tree-reduction GEMV builds (every matvec_<type> / _rp / _normed / _gates
// shader that is NOT built with -DHALO_SUBGROUP_REDUCE): the fixed shared-memory halving tree
// that sums the WG per-thread partials of a row. It is the tree counterpart of sg_finish() in
// common/matvec_sg_reduce.glsl. Every gemv body calls these two functions instead of carrying
// its own copy, so the reduction order (and with it the bitwise VkHead / VkViews / batch ==
// single contracts) is the same for every weight type by construction.
//
// Included ONCE by the gemv shader (matvec_<type>.comp, matvec_<type>_rp.comp, or
// common/matvec_quant_main.glsl for the generic types), after it has declared WG, the y buffer,
// pc (y_off, y_stride, n_vec), BATCHED and MAX_VEC (and, in the _normed build, after
// common/matvec_normed.glsl: its rn[] stays declared before red[] / redb[]), and before the
// per-type bodies that call it. This file declares the shared red[] / redb[] itself. Optional
// hook, defined by the includer BEFORE this file (a later different definition is a macro
// redefinition error):
//   HALO_TREE_OUT(row, v) -> float: the value stored for row's sum v (default v; the
//                            matvec_<type>_gates build applies its GDN gate epilogue here).
//
// NUMERICS: tree_finish(t) adds vector t's partials in exactly the order tree_finish1 adds the
// single vector's (pairwise halving, thread tid += tid + s), so vector t of a batch is bitwise
// the n_vec = 1 matvec of the same vector. The batched form runs one tree for all vectors at
// once (barriers log2 WG + 1 instead of n_vec * (log2 WG + 2)); t >= n_vec lanes are inert
// (uniform predicate on the push constant, constant trip count MAX_VEC so the loops unroll).
#ifndef HALO_MATVEC_TREE_REDUCE_GLSL
#define HALO_MATVEC_TREE_REDUCE_GLSL

#ifndef HALO_TREE_OUT
#define HALO_TREE_OUT(row, v) (v)
#endif

shared float red[WG];                                  // decode form (n_vec == 1)
shared float redb[BATCHED != 0u ? MAX_VEC * WG : 1u];  // batched reduction (all vectors at once)

// Decode form (n_vec == 1): this thread's partial acc1 of row; thread 0 stores the row sum.
void tree_finish1(const uint tid, const uint row, const float acc1) {
    red[tid] = acc1;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    if (tid == 0u) y[pc.y_off + row] = HALO_TREE_OUT(row, red[0]);
}

// Batched form (BATCHED pipeline, n_vec 1..MAX_VEC): the caller has stored this thread's
// partial of row for every vector t < n_vec in redb[t * WG + tid] (its own unrolled loop over
// its register acc[]: an array parameter here would be a by-value copy that -O keeps as a
// second Function array per instantiation); thread 0 stores the n_vec row sums at y_stride apart.
void tree_finish(const uint tid, const uint row) {
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            for (uint t = 0u; t < MAX_VEC; ++t)
                if (t < pc.n_vec) redb[t * WG + tid] += redb[t * WG + tid + s];
        }
        barrier();
    }
    if (tid == 0u) {
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) y[pc.y_off + t * pc.y_stride + row] = HALO_TREE_OUT(row, redb[t * WG]);
    }
}

#endif  // HALO_MATVEC_TREE_REDUCE_GLSL
