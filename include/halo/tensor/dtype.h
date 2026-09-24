#pragma once
// Tensor element types. Numeric values are identical to ggml's `enum ggml_type` so GGUF
// tensor-info type ids map 1:1 (DECISIONS.md D-007). Block geometry and dequantization
// live in the tensor module (src/tensor).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace halo {

enum class DType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
};

struct DTypeTraits {
    std::string_view name;
    std::size_t block_elems;  // elements per block (1 for plain types)
    std::size_t block_bytes;  // bytes per block
    bool quantized;
};

// Returns nullopt for ids HALO does not know (a GGUF may carry any uint32).
[[nodiscard]] std::optional<DType> dtype_from_id(std::uint32_t id) noexcept;
[[nodiscard]] const DTypeTraits& traits(DType t);  // throws Error(Unsupported) if unknown

}  // namespace halo
