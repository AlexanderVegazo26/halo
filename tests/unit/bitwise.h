#pragma once
// Bit-for-bit float comparison for the tests, as gtest assertions that say where two arrays
// diverge: EXPECT_TRUE(halo::test::bitwise(got, want)) fails with the number of differing
// floats and the first one's index, values and bit patterns instead of a bare "false".
// Header-only; include it as "../bitwise.h" from a tests/unit/<area>/ file.

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <span>
#include <string>
#include <string_view>

namespace halo::test {

// Bit patterns, not values: +0 != -0 and a NaN equals itself only with the same payload.
inline bool same_bits(float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

// Bit-for-bit equality of two float arrays. A failure names the size mismatch, or the number of
// differing floats and the first one's index with both values and bit patterns.
inline ::testing::AssertionResult bitwise(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << "sizes differ: " << a.size() << " vs " << b.size();
    }
    std::size_t diffs = 0, first = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_bits(a[i], b[i]) && diffs++ == 0) first = i;
    }
    if (diffs == 0) return ::testing::AssertionSuccess() << "all " << a.size() << " floats bit-identical";
    const auto bits = [](float v) {
        std::ostringstream s;
        s << "0x" << std::hex << std::setw(8) << std::setfill('0') << std::bit_cast<std::uint32_t>(v);
        return s.str();
    };
    return ::testing::AssertionFailure()
           << diffs << " of " << a.size() << " floats differ; first at [" << first << "]: "
           << std::setprecision(9) << a[first] << " (" << bits(a[first]) << ") vs " << b[first] << " ("
           << bits(b[first]) << ")";
}

// The pointer form, for rows inside larger arrays: n floats at a vs n floats at b.
inline ::testing::AssertionResult bitwise(const float* a, const float* b, std::size_t n) {
    return bitwise(std::span<const float>(a, n), std::span<const float>(b, n));
}

// The same check with the compared thing named first, e.g. "trunk KV pool storage: 3 of 96 ...".
inline ::testing::AssertionResult bitwise(std::span<const float> a, std::span<const float> b, std::string_view what) {
    ::testing::AssertionResult r = bitwise(a, b);
    if (r) return r;
    return ::testing::AssertionFailure() << what << ": " << r.message();
}

}  // namespace halo::test
