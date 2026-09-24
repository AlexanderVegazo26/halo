#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "halo/core/error.h"
#include "halo/tensor/fp16.h"
#include "halo/tensor/quant.h"

using halo::DType;
using halo::ErrorCode;
namespace ht = halo::tensor;

namespace {

template <typename F>
ErrorCode code_of(F&& f) {
    try {
        f();
    } catch (const halo::Error& e) {
        return e.code();
    }
    ADD_FAILURE() << "expected halo::Error";
    return ErrorCode::Cancelled;
}

void put_f16(std::vector<std::byte>& b, std::size_t off, float v) {
    const std::uint16_t h = ht::fp32_to_fp16(v);
    std::memcpy(b.data() + off, &h, 2);
}

}  // namespace

TEST(RowBytes, GeometryAndChecks) {
    EXPECT_EQ(ht::row_bytes(DType::F32, 5120), 20480u);
    EXPECT_EQ(ht::row_bytes(DType::Q6_K, 5120), 20u * 210u);
    EXPECT_EQ(ht::row_bytes(DType::Q4_0, 5120), 160u * 18u);
    EXPECT_EQ(ht::row_bytes(DType::IQ4_NL, 17408), 544u * 18u);
    EXPECT_EQ(ht::row_bytes(DType::Q8_0, 0), 0u);
    EXPECT_EQ(code_of([] { (void)ht::row_bytes(DType::Q4_K, 100); }), ErrorCode::Model);  // not /256
    EXPECT_EQ(code_of([] { (void)ht::row_bytes(DType::F32, -1); }), ErrorCode::Model);
    constexpr auto big = std::numeric_limits<std::int64_t>::max() / 32 * 32;
    EXPECT_EQ(code_of([&] { (void)ht::row_bytes(DType::Q8_0, big); }), ErrorCode::Model);  // overflow
    EXPECT_EQ(code_of([] { (void)ht::row_bytes(static_cast<DType>(5), 32); }), ErrorCode::Unsupported);
}

TEST(Dequant, Q8_0HandBuiltBlock) {
    std::vector<std::byte> blk(34);
    put_f16(blk, 0, 0.5f);
    for (int j = 0; j < 32; ++j) blk[2 + static_cast<std::size_t>(j)] = static_cast<std::byte>(static_cast<std::int8_t>(j - 16));
    std::array<float, 32> y{};
    ht::dequantize_row(DType::Q8_0, blk, y);
    for (int j = 0; j < 32; ++j) EXPECT_EQ(y[static_cast<std::size_t>(j)], 0.5f * static_cast<float>(j - 16));
}

TEST(Dequant, Q4_0HandBuiltBlockNibbleOrder) {
    // low nibbles -> elements 0..15, high nibbles -> 16..31, value = (q - 8) * d
    std::vector<std::byte> blk(18);
    put_f16(blk, 0, 2.0f);
    for (int j = 0; j < 16; ++j) blk[2 + static_cast<std::size_t>(j)] = static_cast<std::byte>((15 - j) << 4 | j);
    std::array<float, 32> y{};
    ht::dequantize_row(DType::Q4_0, blk, y);
    for (int j = 0; j < 16; ++j) {
        EXPECT_EQ(y[static_cast<std::size_t>(j)], 2.0f * static_cast<float>(j - 8));
        EXPECT_EQ(y[static_cast<std::size_t>(j + 16)], 2.0f * static_cast<float>(7 - j));
    }
}

TEST(Dequant, RejectsBadArguments) {
    std::vector<std::byte> src(210);
    std::vector<float> dst(256);
    EXPECT_EQ(code_of([&] { ht::dequantize_row(DType::Q6_K, src.data(), dst.data(), 255); }), ErrorCode::Model);
    EXPECT_EQ(code_of([&] { ht::dequantize_row(DType::Q6_K, src.data(), dst.data(), -256); }), ErrorCode::Model);
    EXPECT_EQ(code_of([&] { ht::dequantize_row(DType::IQ2_XXS, src.data(), dst.data(), 256); }),
              ErrorCode::Unsupported);
    EXPECT_EQ(code_of([&] {
                  ht::dequantize_row(DType::Q6_K, std::span<const std::byte>(src).first(209), std::span<float>(dst));
              }),
              ErrorCode::Model);
    EXPECT_NO_THROW(ht::dequantize_row(DType::Q6_K, std::span<const std::byte>(src), std::span<float>(dst)));
    EXPECT_NO_THROW(ht::dequantize_row(DType::Q6_K, src.data(), dst.data(), 0));
}

// vec_dot_row must equal the double-precision dot of the dequantized row.
TEST(VecDot, MatchesDequantThenDot) {
    std::mt19937 rng(7);
    for (DType t : {DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q8_0, DType::Q4_K, DType::Q6_K,
                    DType::IQ4_NL, DType::IQ3_S}) {
        const std::int64_t n = 1024;
        std::vector<std::byte> row(ht::row_bytes(t, n));
        for (auto& b : row) b = static_cast<std::byte>(rng() & 0xFF);
        // Keep every value finite: clearing bit 6 of each odd byte caps the exponent of every
        // little-endian f16/bf16/f32 word, and all fp16 scale fields of the tested block types
        // start at even offsets (their high byte is odd), so no Inf/NaN can appear.
        for (std::size_t i = 1; i < row.size(); i += 2) row[i] &= std::byte{0xBF};
        std::vector<float> x(static_cast<std::size_t>(n));
        for (auto& v : x) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f;
        std::vector<float> w(static_cast<std::size_t>(n));
        ht::dequantize_row(t, row, w);
        double ref = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) ref += static_cast<double>(w[i]) * static_cast<double>(x[i]);
        EXPECT_EQ(ht::vec_dot_row(t, row, x), static_cast<float>(ref)) << halo::traits(t).name;
    }
    std::vector<std::byte> row(34);
    std::vector<float> x(31);
    EXPECT_EQ(code_of([&] { (void)ht::vec_dot_row(DType::Q8_0, row, x); }), ErrorCode::Model);
}
