// Element dequantization of ggml Q3_K. block_q3_K (ggml-common.h, MIT), 110 bytes / 256
// values: uint8 hmask[32]; uint8 qs[64]; uint8 scales[12] (6-bit, packed); fp16 d.
// Ported from ggml-quants.c dequantize_row_q3_K (MIT), as tensor::dequantize_row: value =
// (d * (scale - 32)) * (q2 - (hmask bit ? 0 : 4)) — uncontracted.
// Element i: half n = i/128, j = (i%128)/32 (shift 2j, hmask bit n*4+j), sub = (i%32)/16,
// l = i%16; scale index is = 8n + 2j + sub; q byte qs[32n + 16 sub + l]; hmask byte 16 sub + l.
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_q3_k and get_rows_q3_k.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

const uint DQ_QK = 256u;
const uint DQ_BLOCK_BYTES = 110u;

// The 6-bit scale `is` (0..15) from the 12 packed bytes at `so` (ggml's kmask1/kmask2 unpack).
int q3k_scale(uint so, uint is) {
    const uint g = is / 4u;
    const uint b = is % 4u;
    const uint lo = (read_u8(so + (g & 1u) * 4u + b) >> (4u * (g >> 1))) & 0xFu;
    const uint hi = (read_u8(so + 8u + b) >> (2u * g)) & 3u;
    return int(lo | (hi << 4)) - 32;
}

float halo_dequant(uint row_off, uint c) {
    const uint bo = row_off + (c / DQ_QK) * DQ_BLOCK_BYTES;
    const uint i = c % DQ_QK;
    const uint n = i / 128u;
    const uint j = (i % 128u) / 32u;
    const uint sub = (i % 32u) / 16u;
    const uint l = i % 16u;
    const float d = read_f16(bo + 108u);
    const int sc = q3k_scale(bo + 96u, 8u * n + 2u * j + sub);
    const uint qb = read_u8(bo + 32u + 32u * n + 16u * sub + l);
    const uint hm = read_u8(bo + 16u * sub + l);
    const int q = int((qb >> (2u * j)) & 3u) - (((hm >> (4u * n + j)) & 1u) != 0u ? 0 : 4);
    precise float dl = d * float(sc);
    precise float r = dl * float(q);
    return r;
}

#endif
