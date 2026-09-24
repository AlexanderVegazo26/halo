#include <gtest/gtest.h>

#include "halo/core/error.h"
#include "halo/core/log.h"

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
