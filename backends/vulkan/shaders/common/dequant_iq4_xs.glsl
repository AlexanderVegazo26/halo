// Element dequantization of ggml IQ4_XS. block_iq4_xs (ggml-common.h, MIT), 136 bytes / 256
// values: fp16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]
// Ported from ggml-quants.c dequantize_row_iq4_xs (MIT): value = (d * (ls - 32)) *
// kvalues_iq4nl[nibble] — tensor::dequantize_row's expression, uncontracted.
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_iq4_xs and get_rows_iq4_xs.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

const uint DQ_QK = 256u;
const uint DQ_BLOCK_BYTES = 136u;

// ggml kvalues_iq4nl (ggml-common.h, MIT)
const int KVALUES[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

float halo_dequant(uint row_off, uint c) {
    const uint bo = row_off + (c / DQ_QK) * DQ_BLOCK_BYTES;
    const uint i = c % DQ_QK;
    const uint ib = i / 32u;
    const uint j = i % 32u;
    const float d = read_f16(bo);
    const uint scales_h = read_u8(bo + 2u) | (read_u8(bo + 3u) << 8);
    const uint ls = ((read_u8(bo + 4u + ib / 2u) >> (4u * (ib % 2u))) & 0xFu) | (((scales_h >> (2u * ib)) & 3u) << 4);
    const uint qb = read_u8(bo + 8u + 16u * ib + (j % 16u));
    const uint nib = j < 16u ? (qb & 0xFu) : (qb >> 4);
    precise float dl = d * float(int(ls) - 32);
    precise float r = dl * float(KVALUES[nib]);
    return r;
}

#endif
