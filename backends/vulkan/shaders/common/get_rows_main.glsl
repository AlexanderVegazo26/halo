// Shared body of the GET_ROWS kernels get_rows_<type>: out[t] = dequantize_row(W[ids[t]]),
// element by element with halo_dequant() (common/dequant_<type>.glsl, the expressions of
// tensor::dequantize_row, uncontracted: bit-identical). One workgroup per id.
// An id outside [0, n_rows) sets status bit 8 (k_status_bad_index; the host zeroed the word)
// and leaves that out row unwritten; W is never read through it.
#ifndef HALO_GET_ROWS_MAIN_GLSL
#define HALO_GET_ROWS_MAIN_GLSL

const uint STATUS_BAD_INDEX = 8u;

layout(std430, binding = 1) readonly buffer Ids { int ids[]; };
layout(std430, binding = 2) writeonly buffer O { float o[]; };
layout(std430, binding = 3) buffer St { uint st[]; };

layout(push_constant) uniform Push {
    uint n_rows;
    uint cols;
    uint n_ids;
    uint w_off;     // sub-alignment remainder of W (quantized: bytes; f32: elements)
    uint w_stride;  // row stride of W (same unit)
    uint ids_off, o_off, o_stride, st_off;
    // Row slab of this dispatch (a table larger than one binding is processed in slabs): W is
    // bound from row row_base on; ids in [row_base, row_base + slab_rows) are gathered, other
    // valid ids are left to the other slabs; flag_bad = 1 in exactly one slab dispatch.
    uint row_base, slab_rows, flag_bad;
} pc;

void main() {
    const uint t = halo_group_id();
    if (t >= pc.n_ids) return;
    const int id = ids[pc.ids_off + t];
    if (id < 0 || uint(id) >= pc.n_rows) {
        if (pc.flag_bad != 0u && gl_LocalInvocationID.x == 0u) atomicOr(st[pc.st_off], STATUS_BAD_INDEX);
        return;
    }
    const uint rel = uint(id) - pc.row_base;  // wraps past slab_rows for ids below the slab
    if (rel >= pc.slab_rows) return;
    const uint row_off = pc.w_off + rel * pc.w_stride;
    const uint ob = pc.o_off + t * pc.o_stride;
    for (uint c = gl_LocalInvocationID.x; c < pc.cols; c += WG) o[ob + c] = halo_dequant(row_off, c);
}

#endif
