// The 9 leading words of every gemv / matmul push block: #include it as the first lines inside
// `layout(push_constant) uniform Push { ... } pc;` (textual include, so no include guard: a shader
// declares one push block). Users: common/matvec_io.glsl (SWAR types, + the HALO_FUSE_NORM tail),
// common/matvec_quant_main.glsl (F32 / Q8_0, + the HALO_GDN_GATES tail), the matmul/matvec_<t>_rp.comp
// shaders and common/matmul_cm_main.glsl. Host mirror: GemvPushCore in backends/vulkan/src/ops.cpp (one struct,
// the first member of Ops::gemv_impl's Push and NPush, and CmPush itself); keep the two in the same field order.
    uint rows;
    uint cols;
    uint n_vec;     // vectors in this dispatch (matvec: 1..MAX_VEC; matmul_cm: any)
    uint w_off;     // sub-alignment remainder of the W view (quantized: bytes; F32: elements; _rp: multiple of 16)
    uint w_stride;  // row stride of W (same unit; _rp: multiple of 16)
    uint x_off, x_stride, y_off, y_stride;  // element remainders / vector strides (S-1)
