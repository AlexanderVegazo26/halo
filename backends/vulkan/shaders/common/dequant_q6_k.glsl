// Element dequantization of ggml Q6_K. block_q6_K (ggml-common.h, MIT), 210 bytes / 256 values:
//   uint8 ql[128] (low 4 bits); uint8 qh[64] (high 2 bits); int8 scales[16]; fp16 d
// Ported from ggml-quants.c dequantize_row_q6_K (MIT): value = (d * sc) * (q - 32) —
// tensor::dequantize_row's expression, uncontracted.
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_q6_k and get_rows_q6_k.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

#define HALO_DQ_QK 256
const uint DQ_QK = 256u;
const uint DQ_BLOCK_BYTES = 210u;

float halo_dequant(uint row_off, uint c) {
    const uint bo = row_off + (c / DQ_QK) * DQ_BLOCK_BYTES;
    const uint i = c % DQ_QK;
    const uint n = i / 128u;  // half
    const uint quarter = (i / 32u) & 3u;
    const uint l = i % 32u;
    const uint is = l / 16u;
    const uint ql = read_u8(bo + 64u * n + l + ((quarter & 1u) != 0u ? 32u : 0u));
    const uint qh = read_u8(bo + 128u + 32u * n + l);
    const uint lo4 = quarter >= 2u ? (ql >> 4) : (ql & 0xFu);
    const int q = int(lo4 | (((qh >> (2u * quarter)) & 3u) << 4)) - 32;
    const int sc = read_i8(bo + 192u + 8u * n + is + 2u * quarter);
    const float d = read_f16(bo + 208u);
    precise float ds = d * float(sc);
    precise float r = ds * float(q);
    return r;
}

#endif
