#pragma once
// Optional load-time repack of GGUF K-quant / IQ4_XS weight rows into a planar,
// 16-byte-aligned layout for the Vulkan SWAR matvec kernels (next.md step 8; opt-in via
// HALO_REPACK=1 in the qwen35 loader).
//
// The repacked bytes are ONLY valid as the W operand of a Vulkan gemv issued with
// KernelChoice{backend::kGemvRepacked}. They are not GGUF blocks: dequantize_row,
// get_rows, the CPU backend and anything else that reads the raw layout must keep the
// original bytes.
//
// Contract. A row of `cols` elements has nb = cols / 256 blocks and occupies the SAME
// number of bytes as the GGUF row (176 / 210 / 136 * nb) -- only the byte order inside the
// row changes -- but consecutive rows are `gemv_repack_row_stride(t, cols)` bytes apart
// (row footprint rounded up to 16), so every row starts 16-byte aligned. Inside a row the
// planes start at 16-byte-aligned offsets; b = block index, t = the thread index of the
// matvec_<type>_main.glsl this layout was designed for (16 threads per Q5_K/Q6_K block,
// 8 per IQ4_XS block). Words are little endian, exactly the u32 the original shader built
// with its LD32 loads, so the unpack integers (and thus every float op) are unchanged.
//
//  Q5_K   (176 B/block)  [ qs'  nb*128 ][ qh'  nb*32 ][ hdr nb*16 ]
//     qs'[b]  16 x uvec2 : thread t = 8*v_im+4*v_in+ir, q_off = 32*v_im + 4*ir + 2*v_in
//                          .x = qs[q_off], qs[q_off+1], qs[q_off+16], qs[q_off+17]
//                          .y = qs[q_off+64], +65, +80, +81
//     qh'[b]  8 x u32    : j = 2*ir+v_in, l0 = 2j: qh[l0], qh[l0+1], qh[l0+16], qh[l0+17]
//     hdr[b]  uvec4      : the original bytes 0..15 (d, dmin, scales[12])
//  Q6_K   (210 B/block)  [ ql'  nb*128 ][ qh nb*64 ][ scales nb*16 ][ d nb*2 ]
//     ql'[b]  16 x uvec2 : thread t = 8*v_im+v_in, l0 = 4*v_in
//                          .x = ql[64*v_im+l0 .. +3], .y = ql[64*v_im+l0+32 .. +35]
//     qh[b], scales[b]   : the original bytes 128..191 and 192..207 unchanged
//     d[b]               : f16, the original bytes 208..209
//  IQ4_XS (136 B/block)  [ qs nb*128 ][ hdr nb*8 ]
//     qs[b]   8 x uvec4  : the original bytes 8..135 (thread ib32 reads 16 B at 16*ib32)
//     hdr[b]  uvec2      : the original bytes 0..7 (d | scales_h, scales_l)
//
// Layout constants here and in backends/vulkan/shaders/common/matvec_*_rp_main.glsl must
// change together.

#include <cstddef>
#include <cstdint>
#include <span>

#include "halo/tensor/dtype.h"

namespace halo::tensor {

/// True for the types with a repacked layout: Q5_K, Q6_K, IQ4_XS.
[[nodiscard]] bool gemv_repack_supported(DType t) noexcept;

/// Distance in bytes between consecutive repacked rows: row_bytes(t, cols) rounded up to a
/// multiple of 16. Error(Unsupported) for an unsupported type, Error(Model) if cols is not a
/// multiple of 256.
[[nodiscard]] std::size_t gemv_repack_row_stride(DType t, std::size_t cols);

/// Repack `rows` dense GGUF rows (src: rows * row_bytes(t, cols) bytes, any alignment) into
/// dst (rows * gemv_repack_row_stride(t, cols) bytes; the row padding is zeroed). Pure host
/// function, no shared state.
void gemv_repack(DType t, std::span<const std::byte> src, std::size_t rows, std::size_t cols,
                 std::span<std::byte> dst);

}  // namespace halo::tensor
