#pragma once
// Shared helpers for the profiling tests.

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>

#include "halo/core/error.h"

namespace halo::profiling::test {

/// Expects `fn` to throw halo::Error with `code`.
inline void expect_error(ErrorCode code, const std::function<void()>& fn) {
    try {
        fn();
        ADD_FAILURE() << "expected halo::Error(" << to_string(code) << "), nothing was thrown";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), code) << e.what();
    } catch (const std::exception& e) {
        ADD_FAILURE() << "expected halo::Error, got std::exception: " << e.what();
    }
}

inline std::filesystem::path source_dir() { return std::filesystem::path(HALO_SOURCE_DIR); }
inline std::filesystem::path sysfs_fixture(const char* name) {
    return source_dir() / "tests" / "fixtures" / "sysfs" / name;
}

}  // namespace halo::profiling::test
