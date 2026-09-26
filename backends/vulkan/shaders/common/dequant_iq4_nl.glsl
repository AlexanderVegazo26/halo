// Element dequantization of ggml IQ4_NL. block_iq4_nl (ggml-common.h, MIT): { fp16 d;
// uint8 qs[16]; } = 18 bytes / 32 values; value j = d * kvalues_iq4nl[nibble] (low nibbles of
// qs for j < 16, high nibbles for j >= 16) — tensor::dequantize_row's expression, uncontracted.
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_iq4_nl and get_rows_iq4_nl.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

const uint DQ_QK = 32u;
const uint DQ_BLOCK_BYTES = 18u;

// ggml kvalues_iq4nl (ggml-common.h, MIT)
const int KVALUES[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

float halo_dequant(uint row_off, uint c) {
    const uint bo = row_off + (c / DQ_QK) * DQ_BLOCK_BYTES;
    const uint j = c % DQ_QK;
    const float d = read_f16(bo);
    const uint qb = read_u8(bo + 2u + (j % 16u));
    const uint nib = j < 16u ? (qb & 0xFu) : (qb >> 4);
    precise float r = d * float(KVALUES[nib]);
    return r;
}

#endif
