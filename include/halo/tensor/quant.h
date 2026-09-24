#pragma once
// Row-level (de)quantization reference for the CPU path (DECISIONS.md D-007).
//
// A "row" is `n` consecutive elements stored as whole blocks of `traits(t).block_elems`
// elements (`traits(t).block_bytes` bytes each). Blocks never straddle a row in GGUF because
// every tensor's ne[0] is a multiple of the block size, so several contiguous rows can be
// passed as one row of n = rows * ne[0].
//
// Dequantization is a scalar port of ggml's reference code (ggml/src/ggml-quants.c at
// llama.cpp bd4f514db, MIT). It evaluates floating point in the same order as ggml and
// gguf-py and is compiled without FP contraction, so results are bit-identical to
// `gguf.quants.dequantize` (verified by tests/unit/tensor on real blocks).
//
// Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ4_NL, IQ4_XS, IQ3_S (see `dequant_supported`). Other types throw UNSUPPORTED_ERROR.
//
// All functions are thread-safe (no shared state). Source bytes may be arbitrarily aligned.

#include <cstddef>
#include <cstdint>
#include <span>

#include "halo/tensor/dtype.h"

namespace halo::tensor {

/// Bytes occupied by `n` elements of type `t`.
/// Throws Error(Model) if n < 0, n is not a multiple of the block size, or the byte count
/// overflows std::size_t / int64. Throws Error(Unsupported) for an unknown type.
[[nodiscard]] std::size_t row_bytes(DType t, std::int64_t n);

/// Dequantize `n` elements from `src` (row_bytes(t, n) bytes) into `dst` (n floats).
/// Preconditions are checked: n >= 0 and n % block_elems == 0 (Error(Model) otherwise);
/// unsupported types throw Error(Unsupported). `src` and `dst` must not overlap.
void dequantize_row(DType t, const std::byte* src, float* dst, std::int64_t n);

/// Span form: requires src.size() == row_bytes(t, dst.size()) (Error(Model) otherwise).
void dequantize_row(DType t, std::span<const std::byte> src, std::span<float> dst);

/// Reference dot product of one stored row with a float vector: sum_i w[i] * x[i].
/// Dequantizes block-wise and accumulates in double, returning the rounded float. This is
/// the accuracy reference, not a fast kernel. Same preconditions as dequantize_row.
[[nodiscard]] float vec_dot_row(DType t, const std::byte* row, const float* x, std::int64_t n);

/// Span form: requires row.size() == row_bytes(t, x.size()).
[[nodiscard]] float vec_dot_row(DType t, std::span<const std::byte> row, std::span<const float> x);

}  // namespace halo::tensor
