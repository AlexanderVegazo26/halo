// "Dequantization" of fp32 weights read through the uint weight buffer: the bits, unchanged.
// Offsets are in 4-byte elements (the host binds fp32 weights with float units).
#ifndef HALO_DEQUANT_GLSL
#define HALO_DEQUANT_GLSL

float halo_dequant(uint row_off, uint c) { return uintBitsToFloat(wq[row_off + c]); }

#endif
