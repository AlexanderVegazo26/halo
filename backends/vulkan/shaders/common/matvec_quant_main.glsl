// Shared body of the quantized QUANT_GEMV kernels matvec_<type>: y[r] = sum_c W[r,c] * x[c].
// The including shader has defined the weight buffer `wq`, HALO_DEFINE_BYTE_READERS(wq) and
// halo_dequant() (common/dequant_<type>.glsl). W, x, y are host views (code review S-1):
// row r of W starts at byte w_off + r * w_stride. Correctness-first: one workgroup per row,
// each thread dequantizes individual elements, then a fixed-order shared-memory tree.
#ifndef HALO_MATVEC_QUANT_MAIN_GLSL
#define HALO_MATVEC_QUANT_MAIN_GLSL

layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint cols;
    uint w_off;     // sub-alignment remainder of the W view, in bytes
    uint w_stride;  // row stride of W, in bytes
    uint x_off;     // element remainders of the x / y views (code review S-1)
    uint y_off;
} pc;

shared float red[WG];

void main() {
    const uint row = halo_group_id();
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint row_off = pc.w_off + row * pc.w_stride;

    float acc = 0.0;
    for (uint c = tid; c < pc.cols; c += WG) acc += halo_dequant(row_off, c) * x[pc.x_off + c];
    red[tid] = acc;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    if (tid == 0u) y[pc.y_off + row] = red[0];
}

#endif
