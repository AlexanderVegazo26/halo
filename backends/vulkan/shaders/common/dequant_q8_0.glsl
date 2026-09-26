// Element dequantization of ggml Q8_0 (block_q8_0, ggml-common.h, MIT): { fp16 d; int8 qs[32]; }
// = 34 bytes / 32 values, value = qs[i] * d — tensor::dequantize_row's expression, uncontracted.
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_q8_0 and get_rows_q8_0.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

const uint DQ_QK = 32u;
const uint DQ_BLOCK_BYTES = 34u;

// Element c of the row whose first block starts at byte `row_off`.
float halo_dequant(uint row_off, uint c) {
    const uint bo = row_off + (c / DQ_QK) * DQ_BLOCK_BYTES;
    const float d = read_f16(bo);
    const float q = float(read_i8(bo + 2u + (c % DQ_QK)));
    precise float r = q * d;
    return r;
}

#endif
