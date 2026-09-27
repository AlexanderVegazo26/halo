// Element dequantization of ggml Q4_K. block_q4_K (ggml-common.h, MIT), 144 bytes / 256 values:
//   fp16 d; fp16 dmin; uint8 scales[12] (6-bit scales + mins); uint8 qs[128] (4-bit)
// Ported from ggml-quants.c dequantize_row_q4_K / get_scale_min_k4 (MIT):
//   for chunk c in 0..3 (64 values): sub-blocks 2c (low nibbles) and 2c+1 (high nibbles)
//   of qs[32c .. 32c+31];  value = (d*sc[sub]) * q - dmin*m[sub]  (tensor::dequantize_row's
//   expression, uncontracted).
// Include after HALO_DEFINE_BYTE_READERS(<weight buffer>). Shared by matvec_q4_k and get_rows_q4_k.
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

#define HALO_DQ_QK 256
const uint DQ_QK = 256u;
const uint DQ_BLOCK_BYTES = 144u;

// get_scale_min_k4(j, q, &d, &m) with q = block scales at byte offset `so`.
// Branchless: j = 2*chunk+hi differs between lanes of the wave, so the original j < 4
// branch executed both sides under exec-masking. One extra byte load, no divergence.
void scale_min_k4(uint j, uint so, out uint sc, out uint m) {
    const uint sj = read_u8(so + j);        // s[j]
    const uint sj4 = read_u8(so + j + 4u);  // s[j+4]
    const uint sjm4 = read_u8(so + max(j, 4u) - 4u);  // s[j-4] for j >= 4 (s[j] otherwise: unused)
    const uint sc_lo = sj & 63u;
    const uint m_lo = sj4 & 63u;
    const uint sc_hi = (sj4 & 0xFu) | ((sjm4 >> 6) << 4);
    const uint m_hi = (sj4 >> 4) | ((sj >> 6) << 4);
    const bool hi = j >= 4u;
    sc = hi ? sc_hi : sc_lo;
    m = hi ? m_hi : m_lo;
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
    const uint qb = read_u8(bo + 16u + 32u * chunk + l);
    const uint q = hi == 1u ? (qb >> 4) : (qb & 0xFu);
    precise float d1 = d * float(sc);
    precise float m1 = dmin * float(m);
    precise float p = d1 * float(q);
    precise float r = p - m1;
    return r;
}

#endif
