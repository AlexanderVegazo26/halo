// Shared body of the GEMV kernels matvec_<type> (every weight type, F32 included):
//   y[t][r] = sum_c W[r,c] * x[t][c]   for t < n_vec (<= MAX_VEC per dispatch; the host splits).
// The including shader has defined the weight buffer `wq`, HALO_DEFINE_BYTE_READERS(wq) and
// halo_dequant() (common/dequant_<type>.glsl). W, x, y are host views (code review S-1): row r
// of W starts at w_off + r * w_stride (bytes; F32: elements). One workgroup per W row. Quantized
// types (HALO_DQ_QK >= 8) walk columns in the group-of-8 mapping below, identical in the decode
// and batched paths; F32 keeps the element-per-thread stride. Either way each thread's
// accumulation order and the fixed shared-memory tree are the same for every n_vec, so vector t
// of a batch is bitwise identical to the n_vec = 1 matvec.
#ifndef HALO_MATVEC_QUANT_MAIN_GLSL
#define HALO_MATVEC_QUANT_MAIN_GLSL

const uint MAX_VEC = 8u;

layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
#include "matvec_push_core.glsl"
#ifdef HALO_GDN_GATES
    uint gate_n_v;        // rows [0, gate_n_v) are beta rows, [gate_n_v, 2 gate_n_v) alpha rows
    uint dt_off, sa_off;  // element remainders of the dt_bias / ssm_a views
    uint gate_row0;       // row of the whole W this dispatch's row 0 is (row slabs of a W larger than one binding)
#endif
} pc;

// GDN gates epilogue (matvec_<type>_gates: matvec_<type>.comp built with -DHALO_GDN_GATES; next.md
// step 6): the gemv over qwen35's fused beta|alpha rows
// (HALO_FUSE_GEMV=ba) writes the gated values itself instead of a separate gdn_gates dispatch:
//   beta rows: sigmoid(y);  alpha row j: ssm_a[j] * softplus(y + dt_bias[j])
// with exactly eltwise/gdn_gates.comp's expressions, so the output is bit for bit gemv then gdn_gates.
#ifdef HALO_GDN_GATES
#include "halo_math.glsl"
layout(std430, binding = 3) readonly buffer GateDt { float gate_dt[]; };
layout(std430, binding = 4) readonly buffer GateSa { float gate_sa[]; };
float halo_gemv_out(uint row, float v) {
    const uint r = pc.gate_row0 + row;
    if (r < pc.gate_n_v) return halo_sigmoid(v);
    const uint j = r - pc.gate_n_v;
    precise float s = v + gate_dt[pc.dt_off + j];
    precise float gv = gate_sa[pc.sa_off + j] * halo_softplus(s);
    return gv;
}
#else
float halo_gemv_out(uint row, float v) { return v; }
#endif

// Specialization constant 1 (host: 0 when n_vec == 1). The decode pipeline variant is compiled
// without the batched body: the unrolled MAX_VEC accumulator loop triples register pressure,
// and sharing one kernel cost decode ~45%. The per-vector arithmetic is identical in both
// variants (VkHead bitwise gate).
layout(constant_id = 1) const uint BATCHED = 1;

// tree_finish1() / tree_finish() store halo_gemv_out(row, sum): GDN gates epilogue, else identity.
#define HALO_TREE_OUT(row, v) halo_gemv_out(row, v)
#include "matvec_tree_reduce.glsl"  // tree_finish1() / tree_finish()

// Per-type one-time workgroup setup (e.g. the IQ3_S shared grid); empty for most types.
#ifndef HALO_DEQUANT_SETUP
#define HALO_DEQUANT_SETUP
#endif

#ifndef HALO_DQ_QK
#define HALO_DQ_QK 1
#endif

void main() {
    HALO_DEQUANT_SETUP;
    const uint row = halo_group_id();
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint row_off = pc.w_off + row * pc.w_stride;

    if (BATCHED == 0u || pc.n_vec == 1u) {
        // Decode form: scalar accumulator, same column order and reduction tree as the
        // batched path below (bitwise identical for vector 0).
        float acc1 = 0.0;
#if HALO_DQ_QK >= 8
        // Group-of-8 column mapping: thread t owns 8 consecutive elements, c0 = blk*QK + g*8
        // with g = t % (QK/8), blk striding by WG/(QK/8) blocks. In the element-per-thread
        // mapping every thread paid the full block-header cost (d/dmin/scales) for one
        // element; here the unrolled 8 dequant calls share those loads (CSE) and consecutive
        // byte reads merge into single dword loads. A subgroup-shuffle reduction was tried
        // and was 2x slower (RDNA wave64 shuffles are LDS swizzles): keep the tree.
        const uint GPB = uint(HALO_DQ_QK) / 8u;
        const uint g = tid % GPB;
        const uint nblocks = pc.cols / uint(HALO_DQ_QK);
        for (uint blk = tid / GPB; blk < nblocks; blk += WG / GPB) {
            const uint c0 = blk * uint(HALO_DQ_QK) + g * 8u;
            for (uint j = 0u; j < 8u; ++j) acc1 += halo_dequant(row_off, c0 + j) * x[pc.x_off + c0 + j];
        }
#else
        for (uint c = tid; c < pc.cols; c += WG) acc1 += halo_dequant(row_off, c) * x[pc.x_off + c];
#endif
        tree_finish1(tid, row, acc1);
        return;
    }

    // Batched / prefill / MTP-verify form: each thread dequantizes its elements ONCE and
    // applies them to every x vector (W is read once for all n_vec rows).
    float acc[MAX_VEC] = float[MAX_VEC](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
#if HALO_DQ_QK >= 8
    const uint GPB = uint(HALO_DQ_QK) / 8u;
    const uint g = tid % GPB;
    const uint nblocks = pc.cols / uint(HALO_DQ_QK);
    for (uint blk = tid / GPB; blk < nblocks; blk += WG / GPB) {
        const uint c0 = blk * uint(HALO_DQ_QK) + g * 8u;
        for (uint j = 0u; j < 8u; ++j) {
            const float w = halo_dequant(row_off, c0 + j);
            // Constant trip count (unrolled, register accumulators; the runtime-bound form
            // spills acc[] to scratch). Uniform predicate: t >= n_vec lanes are inert.
            for (uint t = 0u; t < MAX_VEC; ++t)
                if (t < pc.n_vec) acc[t] += w * x[pc.x_off + t * pc.x_stride + c0 + j];
        }
    }
#else
    for (uint c = tid; c < pc.cols; c += WG) {
        const float w = halo_dequant(row_off, c);
        for (uint t = 0u; t < MAX_VEC; ++t)
            if (t < pc.n_vec) acc[t] += w * x[pc.x_off + t * pc.x_stride + c];
    }
#endif
    for (uint t = 0u; t < MAX_VEC; ++t)
        if (t < pc.n_vec) redb[t * WG + tid] = acc[t];
    tree_finish(tid, row);  // common/matvec_tree_reduce.glsl
}

#endif
