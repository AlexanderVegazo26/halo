// Shared body of the GEMV kernels matvec_<type> (every weight type, F32 included):
//   y[t][r] = sum_c W[r,c] * x[t][c]   for t < n_vec (<= MAX_VEC per dispatch; the host splits).
// The including shader has defined the weight buffer `wq`, HALO_DEFINE_BYTE_READERS(wq) and
// halo_dequant() (common/dequant_<type>.glsl). W, x, y are host views (code review S-1): row r
// of W starts at w_off + r * w_stride (bytes; F32: elements). One workgroup per W row: each
// thread dequantizes its elements ONCE and applies them to every x vector (the batched /
// prefill / MTP-verify form reads W once for all n_vec rows), then one fixed-order
// shared-memory tree per vector. With n_vec = 1 this is the decode matvec, operation for
// operation.
#ifndef HALO_MATVEC_QUANT_MAIN_GLSL
#define HALO_MATVEC_QUANT_MAIN_GLSL

const uint MAX_VEC = 8u;

layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint cols;
    uint n_vec;     // 1..MAX_VEC
    uint w_off;     // sub-alignment remainder of the W view (quantized: bytes; F32: elements)
    uint w_stride;  // row stride of W (same unit)
    uint x_off, x_stride, y_off, y_stride;  // element remainders / vector strides (S-1)
} pc;

shared float red[WG];

void main() {
    const uint row = halo_group_id();
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint row_off = pc.w_off + row * pc.w_stride;

    float acc[MAX_VEC] = float[MAX_VEC](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (uint c = tid; c < pc.cols; c += WG) {
        const float w = halo_dequant(row_off, c);
        for (uint t = 0u; t < pc.n_vec; ++t) acc[t] += w * x[pc.x_off + t * pc.x_stride + c];
    }
    for (uint t = 0u; t < pc.n_vec; ++t) {
        red[tid] = acc[t];
        barrier();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            barrier();
        }
        if (tid == 0u) y[pc.y_off + t * pc.y_stride + row] = red[0];
        barrier();
    }
}

#endif
