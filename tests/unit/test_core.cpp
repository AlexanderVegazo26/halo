#include <gtest/gtest.h>

#include <format>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo/core/error.h"
#include "halo/core/log.h"

#include "bitwise.h"

TEST(Error, CarriesTypedCodeAndPrefix) {
    try {
        halo::throw_error(halo::ErrorCode::Model, "tensor {} missing", "blk.0.attn_q.weight");
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Model);
        EXPECT_STREQ(e.what(), "MODEL_ERROR: tensor blk.0.attn_q.weight missing");
    }
}

TEST(Error, CheckMacroThrowsOnlyWhenFalse) {
    EXPECT_NO_THROW(HALO_CHECK(1 + 1 == 2, halo::ErrorCode::Config, "never"));
    EXPECT_THROW(HALO_CHECK(false, halo::ErrorCode::Config, "bad {}", 1), halo::Error);
}

TEST(Log, ParsesLevelsCaseInsensitively) {
    halo::log::Level l{};
    ASSERT_TRUE(halo::log::parse_level("warn", l));
    EXPECT_EQ(l, halo::log::Level::Warn);
    EXPECT_FALSE(halo::log::parse_level("loud", l));
}

namespace {
// A value whose formatter throws, standing in for anything that makes std::format fail at run
// time (std::bad_alloc under memory pressure, a user formatter that throws).
struct ThrowsWhenFormatted {};
}  // namespace

template <>
struct std::formatter<ThrowsWhenFormatted> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    auto format(const ThrowsWhenFormatted&, std::format_context& ctx) const -> std::format_context::iterator {
        throw std::runtime_error("formatter \"boom\"");
        return ctx.out();
    }
};

// Logging runs in destructors and noexcept paths (sqlite Connection/Transaction, GdnState::reset),
// where an escaping exception is std::terminate. A message that can't be built is dropped with a
// one-line notice instead, in the configured format.
TEST(Log, NeverThrowsAndReportsADroppedMessage) {
    // Arguments built first: constructing a std::format_string from a literal is consteval but
    // not noexcept, which would hide what logf itself promises.
    static constexpr std::format_string<ThrowsWhenFormatted> fmt = "{}";
    static constexpr std::string_view comp = "c";
    static_assert(noexcept(halo::log::logf(halo::log::Level::Error, comp, fmt, ThrowsWhenFormatted{})),
                  "halo::log::logf must be noexcept: destructors log");
    static_assert(noexcept(halo::log::write(halo::log::Level::Error, comp, comp)), "halo::log::write must be noexcept");
    halo::log::set_level(halo::log::Level::Info);

    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(HALO_ERROR("autotune", "close failed: {}", ThrowsWhenFormatted{}));
    std::string err = testing::internal::GetCapturedStderr();
    EXPECT_EQ(err, "[ERROR] autotune: (log message dropped: formatter \"boom\")\n")
        << "text-mode notice for a message whose formatting threw";

    halo::log::set_json(true);
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(HALO_WARN("auto\"tune\n", "x {}", ThrowsWhenFormatted{}));
    err = testing::internal::GetCapturedStderr();
    halo::log::set_json(false);
    ASSERT_FALSE(err.empty()) << "json-mode notice missing";
    ASSERT_EQ(err.back(), '\n') << "json-mode notice must be one line: " << err;
    const auto j = nlohmann::json::parse(err, nullptr, /*allow_exceptions=*/false);
    ASSERT_FALSE(j.is_discarded()) << "json-mode notice is not valid JSON: " << err;
    EXPECT_EQ(j.value("level", ""), "WARN") << err;
    EXPECT_EQ(j.value("component", ""), "auto\"tune\n") << err;
    EXPECT_EQ(j.value("msg", ""), "(log message dropped: formatter \"boom\")") << err;
    EXPECT_TRUE(j.contains("ts_ms") && j["ts_ms"].is_number_integer()) << err;

    testing::internal::CaptureStderr();
    HALO_INFO("ok", "still logging {}", 1);
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "[INFO] ok: still logging 1\n")
        << "a normal message after a dropped one";
}

// halo::test::bitwise (tests/unit/bitwise.h) is the bit-for-bit gate of the kernel and device tests: it must compare bit patterns (not
// values) and, on failure, say where the arrays diverge so a log is enough to start debugging.
TEST(TestUtil, BitwiseComparesBitPatternsAndNamesTheFirstDifferingFloat) {
    using halo::test::bitwise;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> a{1.0f, 0.0f, nan, 3.0f, 4.0f};

    const ::testing::AssertionResult same = bitwise(a, std::vector<float>(a));
    EXPECT_TRUE(same) << "identical arrays (incl. the same NaN) must compare equal: " << same.message();
    EXPECT_EQ(std::string(same.message()), "all 5 floats bit-identical");

    std::vector<float> b = a;
    b[1] = -0.0f;  // == 0.0f as a value, a different bit pattern
    b[4] = 5.0f;
    const ::testing::AssertionResult diff = bitwise(a, b);
    EXPECT_FALSE(diff) << "+0 vs -0 and 4 vs 5 must fail the bitwise gate";
    EXPECT_EQ(std::string(diff.message()), "2 of 5 floats differ; first at [1]: 0 (0x00000000) vs -0 (0x80000000)");

    std::vector<float> c = a;
    c[2] = -nan;  // a NaN with the sign bit set: unequal bits even though both are NaN
    EXPECT_FALSE(bitwise(a, c)) << "NaNs with different bits must fail";

    const ::testing::AssertionResult sizes = bitwise(a, std::span(a).first(4));
    EXPECT_FALSE(sizes);
    EXPECT_EQ(std::string(sizes.message()), "sizes differ: 5 vs 4");

    EXPECT_TRUE(bitwise(a.data() + 3, b.data() + 3, 1)) << "pointer form over a matching 1-float row";
    EXPECT_FALSE(bitwise(a.data() + 3, b.data() + 3, 2)) << "pointer form over a row with 4 vs 5";
}

TEST(TestUtil, BitwiseWithANamePrefixesOnlyAFailure) {
    using halo::test::bitwise;
    const std::vector<float> a{1.0f, 2.0f};
    const std::vector<float> b{1.0f, 2.5f};
    const ::testing::AssertionResult same = bitwise(a, a, "logits");
    EXPECT_TRUE(same);
    EXPECT_EQ(std::string(same.message()), "all 2 floats bit-identical");
    const ::testing::AssertionResult diff = bitwise(a, b, "logits of seq 1");
    EXPECT_FALSE(diff) << "2 vs 2.5 must fail";
    EXPECT_EQ(std::string(diff.message()), "logits of seq 1: 1 of 2 floats differ; first at [1]: 2 (0x40000000) vs 2.5 (0x40200000)");
}
