// Shared body of the cooperative-matrix prefill GEMMs matmul_cm_<type> (Q4_K / Q5_K / Q6_K):
//   y[t][r] = sum_c W[r,c] * x[t][c]   for t < n_vec (any n_vec; the host uses it for n_vec > 8).
// Same host contract as matvec_<type> (Push, bindings, y layout); NOT bit-identical to it: the
// activations and the dequantized weights are rounded to fp16 for the tensor cores (WMMA on
// RDNA3/3.5) and accumulated in fp32, so results differ from the fp32 matvec at the ~1e-3
// relative level. Host: opt-in (HALO_COOPMAT=1), only when the device enabled the extension.
//
// Structure: no workgroup-wide sharing. Every SUBGROUP owns one TM x TN (32 x 32) output tile
// (2 x 2 accumulators of 16x16x16, fp32) and stages its own A (x, fp16) and B (dequantized W,
// fp16) tiles in a private slice of shared memory, synchronised with subgroup barriers only.
// That keeps the kernel independent of the runtime subgroup size (RADV may run coopmat
// shaders as wave32 or wave64; the workgroup may hold 1 or several subgroups) and makes an
// early return of a whole subgroup (tile id past the grid) legal: no workgroup barrier()
// exists. The price is that W tiles are dequantized once per token tile and never shared
// between subgroups (measure before tuning).
//
// Tile id = workgroup_id * gl_NumSubgroups + gl_SubgroupID. The host launches one workgroup
// per tile (enough for a 1-subgroup workgroup); extra subgroups of larger workgroups run past
// the grid and return.
// The including shader has enabled GL_KHR_cooperative_matrix / GL_KHR_memory_scope_semantics /
// GL_KHR_shader_subgroup_basic / GL_EXT_shader_explicit_arithmetic_types_float16, defined the
// weight buffer `wq` (binding 0), HALO_DEFINE_BYTE_READERS(wq) and halo_dequant()
// (common/dequant_<type>.glsl: the exact fp32 element values of tensor::dequantize_row).
#ifndef HALO_MATMUL_CM_MAIN_GLSL
#define HALO_MATMUL_CM_MAIN_GLSL

layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint cols;
    uint n_vec;
    uint w_off;     // sub-alignment remainder of the W view (bytes)
    uint w_stride;  // row stride of W (bytes)
    uint x_off, x_stride, y_off, y_stride;  // element remainders / vector strides
} pc;

const uint TM = 32u;  // tokens per subgroup tile
const uint TN = 32u;  // weight rows per subgroup tile
const uint BK = 32u;  // K step (divides every K-quant block: cols % 256 == 0)
// Slices provided per workgroup: the host requires subgroupSize >= 32, so at most WG / 32.
const uint MAX_SG = WG / 32u;

shared float16_t sA[MAX_SG * TM * BK];  // [token][k]   (row-major, stride BK)
shared float16_t sB[MAX_SG * TN * BK];  // [wrow][k]    (= B(k, n) column-major, stride BK)
shared float sC[MAX_SG * 16u * 16u];    // one accumulator at a time, for the bounds-checked store

void main() {
    const uint sg = gl_SubgroupID;
    if (sg >= MAX_SG) return;  // unreachable when the host guard holds (subgroupSize >= 32)
    const uint lane = gl_SubgroupInvocationID;
    const uint sgsz = gl_SubgroupSize;
    const uint tiles_m = (pc.n_vec + TM - 1u) / TM;
    const uint tiles_n = (pc.rows + TN - 1u) / TN;
    const uint tile = halo_group_id() * gl_NumSubgroups + sg;
    if (tile >= tiles_m * tiles_n) return;  // uniform per subgroup
    const uint t0 = (tile % tiles_m) * TM;  // token tiles fastest: neighbours share the W rows
    const uint r0 = (tile / tiles_m) * TN;

    const uint baseA = sg * TM * BK;
    const uint baseB = sg * TN * BK;
    const uint baseC = sg * 256u;

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc00 =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc01 = acc00;  // tokens 0-15,  rows 16-31
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc10 = acc00;  // tokens 16-31, rows 0-15
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc11 = acc00;

    for (uint k0 = 0u; k0 < pc.cols; k0 += BK) {
        // Stage A: x[t0 .. t0+TM) x [k0, k0+BK) as fp16; tokens past n_vec are zero.
        for (uint idx = lane; idx < TM * BK; idx += sgsz) {
            const uint t = idx / BK;
            const uint k = idx % BK;
            float v = 0.0;
            if (t0 + t < pc.n_vec) v = x[pc.x_off + (t0 + t) * pc.x_stride + k0 + k];
            sA[baseA + idx] = float16_t(v);
        }
        // Stage B: dequantized W rows [r0, r0+TN) x [k0, k0+BK) as fp16; rows past `rows` are
        // zero (never read from W). Consecutive lanes take consecutive k of one row: the
        // block header / scale bytes of a row are shared by the whole 32-run.
        for (uint idx = lane; idx < TN * BK; idx += sgsz) {
            const uint n = idx / BK;
            const uint k = idx % BK;
            float v = 0.0;
            if (r0 + n < pc.rows) v = halo_dequant(pc.w_off + (r0 + n) * pc.w_stride, k0 + k);
            sB[baseB + idx] = float16_t(v);
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();

        for (uint kk = 0u; kk < BK; kk += 16u) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a0;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a1;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b0;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b1;
            coopMatLoad(a0, sA, baseA + kk, BK, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(a1, sA, baseA + 16u * BK + kk, BK, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(b0, sB, baseB + kk, BK, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(b1, sB, baseB + 16u * BK + kk, BK, gl_CooperativeMatrixLayoutColumnMajor);
            acc00 = coopMatMulAdd(a0, b0, acc00);
            acc01 = coopMatMulAdd(a0, b1, acc01);
            acc10 = coopMatMulAdd(a1, b0, acc10);
            acc11 = coopMatMulAdd(a1, b1, acc11);
        }
        // Everyone is done reading this K slice before the next one overwrites it.
        subgroupMemoryBarrierShared();
        subgroupBarrier();
    }

    // Epilogue: each accumulator goes through shared memory (16x16 fp32, row = token) so the
    // edge tiles (n_vec % TM, rows % TN) are written with per-element bounds checks; y is
    // token-major, lanes take consecutive weight rows (coalesced).
#define HALO_CM_STORE(ACC, TOK0, ROW0)                                                        \
    coopMatStore(ACC, sC, baseC, 16u, gl_CooperativeMatrixLayoutRowMajor);                     \
    subgroupMemoryBarrierShared();                                                             \
    subgroupBarrier();                                                                         \
    for (uint idx = lane; idx < 256u; idx += sgsz) {                                           \
        const uint m = idx / 16u;                                                              \
        const uint n = idx % 16u;                                                              \
        const uint tok = t0 + (TOK0) + m;                                                      \
        const uint row = r0 + (ROW0) + n;                                                      \
        if (tok < pc.n_vec && row < pc.rows) y[pc.y_off + tok * pc.y_stride + row] = sC[baseC + idx]; \
    }                                                                                          \
    subgroupMemoryBarrierShared();                                                             \
    subgroupBarrier();
    HALO_CM_STORE(acc00, 0u, 0u)
    HALO_CM_STORE(acc01, 0u, 16u)
    HALO_CM_STORE(acc10, 16u, 0u)
    HALO_CM_STORE(acc11, 16u, 16u)
#undef HALO_CM_STORE
}

#endif
