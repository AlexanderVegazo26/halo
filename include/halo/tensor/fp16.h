#pragma once
// Scalar half-precision <-> single-precision conversions.
//
// fp16 conversions are bit-for-bit ports of ggml's portable reference
// (ggml/src/ggml-impl.h `ggml_compute_fp16_to_fp32` / `ggml_compute_fp32_to_fp16`, from the
// FP16 library by Marat Dukhan; ggml is MIT, Copyright (c) 2023-2026 The ggml authors):
//   - fp16 -> fp32 is exact for every input (subnormals, +-inf; NaN keeps sign + payload).
//   - fp32 -> fp16 rounds to nearest-even, overflows to +-inf, and maps NaN to 0x7E00 | sign.
// bf16 conversions follow ggml `ggml_compute_*_bf16`: widening is a 16-bit shift (exact);
// narrowing rounds to nearest-even and forces NaN to quiet (bit 6 of the bf16 set).
//
// All functions are constexpr, noexcept, and thread-safe (pure).

#include <bit>
#include <cstdint>

namespace halo::tensor {

[[nodiscard]] constexpr float fp16_to_fp32(std::uint16_t h) noexcept {
    const std::uint32_t w = static_cast<std::uint32_t>(h) << 16;
    const std::uint32_t sign = w & 0x80000000u;
    const std::uint32_t two_w = w + w;

    constexpr std::uint32_t exp_offset = 0xE0u << 23;
    constexpr float exp_scale = 0x1.0p-112f;
    const float normalized_value = std::bit_cast<float>((two_w >> 4) + exp_offset) * exp_scale;

    constexpr std::uint32_t magic_mask = 126u << 23;
    constexpr float magic_bias = 0.5f;
    const float denormalized_value = std::bit_cast<float>((two_w >> 17) | magic_mask) - magic_bias;

    constexpr std::uint32_t denormalized_cutoff = 1u << 27;
    const std::uint32_t result =
        sign | (two_w < denormalized_cutoff ? std::bit_cast<std::uint32_t>(denormalized_value)
                                            : std::bit_cast<std::uint32_t>(normalized_value));
    return std::bit_cast<float>(result);
}

[[nodiscard]] constexpr std::uint16_t fp32_to_fp16(float f) noexcept {
    constexpr float scale_to_inf = 0x1.0p+112f;
    constexpr float scale_to_zero = 0x1.0p-110f;
    const float abs_f = std::bit_cast<float>(std::bit_cast<std::uint32_t>(f) & 0x7FFFFFFFu);
    float base = (abs_f * scale_to_inf) * scale_to_zero;

    const std::uint32_t w = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t shl1_w = w + w;
    const std::uint32_t sign = w & 0x80000000u;
    std::uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = std::bit_cast<float>((bias >> 1) + 0x07800000u) + base;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(base);
    const std::uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const std::uint32_t mantissa_bits = bits & 0x00000FFFu;
    const std::uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<std::uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

[[nodiscard]] constexpr float bf16_to_fp32(std::uint16_t h) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
}

[[nodiscard]] constexpr std::uint16_t fp32_to_bf16(float f) noexcept {
    const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {  // NaN: truncate and force quiet
        return static_cast<std::uint16_t>((u >> 16) | 64u);
    }
    return static_cast<std::uint16_t>((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

}  // namespace halo::tensor
