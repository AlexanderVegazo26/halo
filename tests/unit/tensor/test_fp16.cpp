#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "halo/tensor/fp16.h"

using halo::tensor::bf16_to_fp32;
using halo::tensor::fp16_to_fp32;
using halo::tensor::fp32_to_bf16;
using halo::tensor::fp32_to_fp16;

namespace {

// Independent definition of IEEE binary16 -> binary32 via ldexp (different code path from
// the ported bit trick under test).
double fp16_reference(std::uint16_t h) {
    const int sign = (h >> 15) ? -1 : 1;
    const int exp = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    if (exp == 0) return sign * std::ldexp(static_cast<double>(mant), -24);
    if (exp == 31) {
        return mant == 0 ? sign * std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(mant | 0x400), exp - 25);
}

}  // namespace

TEST(Fp16, AllBitPatternsWidenExactly) {
    for (std::uint32_t i = 0; i < 65536; ++i) {
        const auto h = static_cast<std::uint16_t>(i);
        const float f = fp16_to_fp32(h);
        const double ref = fp16_reference(h);
        if (std::isnan(ref)) {
            ASSERT_TRUE(std::isnan(f)) << std::hex << i;
            ASSERT_EQ(std::signbit(f), (h >> 15) != 0) << std::hex << i;  // sign kept
        } else {
            ASSERT_EQ(static_cast<double>(f), ref) << std::hex << i;
            ASSERT_EQ(std::signbit(f), (h >> 15) != 0) << std::hex << i;  // -0 stays -0
        }
    }
}

TEST(Fp16, RoundTripIsIdentityExceptNaN) {
    for (std::uint32_t i = 0; i < 65536; ++i) {
        const auto h = static_cast<std::uint16_t>(i);
        const bool nan = ((h >> 10) & 0x1F) == 31 && (h & 0x3FF) != 0;
        const std::uint16_t back = fp32_to_fp16(fp16_to_fp32(h));
        if (nan) {
            EXPECT_EQ(back & 0x7FFF, 0x7E00) << std::hex << i;  // canonical quiet NaN, sign kept
        } else {
            ASSERT_EQ(back, h) << std::hex << i;
        }
    }
}

TEST(Fp16, NarrowingRoundsToNearestEvenAndSaturatesToInf) {
    EXPECT_EQ(fp32_to_fp16(1.0f), 0x3C00);
    EXPECT_EQ(fp32_to_fp16(1.0f + 0x1p-11f), 0x3C00);          // tie -> even (down)
    EXPECT_EQ(fp32_to_fp16(1.0f + 3 * 0x1p-11f), 0x3C02);      // tie -> even (up)
    EXPECT_EQ(fp32_to_fp16(1.0f + 0x1p-11f + 0x1p-20f), 0x3C01);  // above tie -> up
    EXPECT_EQ(fp32_to_fp16(65504.0f), 0x7BFF);
    EXPECT_EQ(fp32_to_fp16(65520.0f), 0x7C00);  // rounds to inf
    EXPECT_EQ(fp32_to_fp16(1e10f), 0x7C00);
    EXPECT_EQ(fp32_to_fp16(-1e10f), 0xFC00);
    EXPECT_EQ(fp32_to_fp16(0x1p-24f), 0x0001);  // smallest subnormal
    EXPECT_EQ(fp32_to_fp16(0x1p-26f), 0x0000);  // underflow to +0
    EXPECT_EQ(fp32_to_fp16(-0.0f), 0x8000);
    EXPECT_EQ(fp32_to_fp16(std::numeric_limits<float>::quiet_NaN()), 0x7E00);
}

TEST(Bf16, WidenIsShiftAndNarrowRoundsNearestEven) {
    for (std::uint32_t i = 0; i < 65536; ++i) {
        const auto h = static_cast<std::uint16_t>(i);
        const float f = bf16_to_fp32(h);
        ASSERT_EQ(std::bit_cast<std::uint32_t>(f), i << 16);
        const bool nan = ((h >> 7) & 0xFF) == 0xFF && (h & 0x7F) != 0;
        const std::uint16_t back = fp32_to_bf16(f);
        if (nan) {
            EXPECT_EQ(back, h | 64) << std::hex << i;  // forced quiet
        } else {
            ASSERT_EQ(back, h) << std::hex << i;
        }
    }
    EXPECT_EQ(fp32_to_bf16(std::bit_cast<float>(0x3F808000u)), 0x3F80);  // tie -> even (down)
    EXPECT_EQ(fp32_to_bf16(std::bit_cast<float>(0x3F818000u)), 0x3F82);  // tie -> even (up)
    EXPECT_EQ(fp32_to_bf16(std::bit_cast<float>(0x3F808001u)), 0x3F81);  // above tie
}

static_assert(fp16_to_fp32(0x3C00) == 1.0f);
static_assert(fp32_to_fp16(-2.0f) == 0xC000);
static_assert(bf16_to_fp32(0x3F80) == 1.0f);
