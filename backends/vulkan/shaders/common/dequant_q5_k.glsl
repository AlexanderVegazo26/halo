// Element dequantization of ggml Q5_K. block_q5_K (ggml-common.h, MIT), 176 bytes / 256 values:
//   fp16 d; fp16 dmin; uint8 scales[12]; uint8 qh[32] (5th bits); uint8 qs[128] (low 4 bits)
// Ported from ggml-quants.c dequantize_row_q5_K (MIT): value = (d*sc)*q - dmin*m with
// q = nibble + 16 * qh bit (2*chunk + hi) — tensor::dequantize_row's expression, uncontracted.
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_q5_k and get_rows_q5_k.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

const uint DQ_QK = 256u;
const uint DQ_BLOCK_BYTES = 176u;

// get_scale_min_k4(j, q, &d, &m) with q = block scales at byte offset `so`.
void scale_min_k4(uint j, uint so, out uint sc, out uint m) {
    if (j < 4u) {
        sc = read_u8(so + j) & 63u;
        m = read_u8(so + j + 4u) & 63u;
    } else {
        sc = (read_u8(so + j + 4u) & 0xFu) | ((read_u8(so + j - 4u) >> 6) << 4);
        m = (read_u8(so + j + 4u) >> 4) | ((read_u8(so + j) >> 6) << 4);
    }
}

float halo_dequant(uint row_off, uint c) {
    const uint bo = row_off + (c / DQ_QK) * DQ_BLOCK_BYTES;
    const uint i = c % DQ_QK;
    const uint chunk = i / 64u;
    const uint hi = (i / 32u) & 1u;
    const uint l = i % 32u;
    const float d = read_f16(bo);
    const float dmin = read_f16(bo + 2u);
    uint sc;
    uint m;
    scale_min_k4(2u * chunk + hi, bo + 4u, sc, m);
    const uint qb = read_u8(bo + 48u + 32u * chunk + l);
    const uint bit = (read_u8(bo + 16u + l) >> (2u * chunk + hi)) & 1u;
    const uint q = (hi == 1u ? (qb >> 4) : (qb & 0xFu)) + 16u * bit;
    precise float d1 = d * float(sc);
    precise float m1 = dmin * float(m);
    precise float p = d1 * float(q);
    precise float r = p - m1;
    return r;
}

#endif
