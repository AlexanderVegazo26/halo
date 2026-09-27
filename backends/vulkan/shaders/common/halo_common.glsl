// HALO shared GLSL helpers (included by compute shaders; compiled with -I shaders/common).
#ifndef HALO_COMMON_GLSL
#define HALO_COMMON_GLSL

// Linear workgroup id for a 2-D dispatch grid (host: grid_1d()). Callers must discard
// ids >= their item count; the whole workgroup takes the same branch, so returning
// before barrier() is uniform.
uint halo_group_id() { return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x; }

// Byte-granular readers over a `uint[]` storage buffer. ggml block layouts are not
// 4-byte aligned in general (Q8_0 = 34 B, Q6_K = 210 B), so quantized weights are bound
// as uint words and unpacked here (little-endian, no 8/16-bit storage features needed).
// Offsets are arbitrary bytes: a weight view inside an arena may start at any byte (code
// review S-1), so an fp16 field can be odd-aligned or straddle two words.
// read_f16 is branchless: the offset's low bits are uniform per workgroup in the matvec
// mapping, but the compiler cannot prove it and would emit both sides of the alignment
// branch under exec-masking. The min() clamp keeps the spare word inside the buffer.
#define HALO_DEFINE_BYTE_READERS(BUF)                                                     \
    uint read_u8(uint off) { return (BUF[off >> 2] >> ((off & 3u) << 3)) & 0xFFu; }        \
    int read_i8(uint off) { return bitfieldExtract(int(read_u8(off)), 0, 8); }            \
    float read_f16(uint off) {                                                           \
        const uint i = off >> 2;                                                         \
        const uint sh = (off & 3u) << 3;                                                 \
        const uint w0 = BUF[i];                                                          \
        const uint w1 = BUF[min(i + 1u, uint(BUF.length()) - 1u)];                       \
        const uint h = (w0 >> sh) | (sh != 0u ? (w1 << (32u - sh)) : 0u);                \
        return unpackHalf2x16(h & 0xFFFFu).x;                                            \
    }

#endif
