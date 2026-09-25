// GGUF metadata allocation budget (security review S-4; docs/security-hardening.md).
// Small-scale equivalents of the review's PoCs (20M empty nested arrays, a 64 MiB u8
// array) exceed a lowered budget; the real headers must stay far inside the default.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../tensor/ref_data.h"
#include "gguf_builder.h"
#include "halo/core/error.h"
#include "halo/model/gguf.h"

using halo::ErrorCode;
using halo::model::GgufArray;
using halo::model::GgufFile;
using halo::model::GgufLimits;
using halo::model::GgufMode;
using halo::model::GgufOptions;
using namespace halo::test;
using halo::test::ref_dir;
namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;

// One KV holding an array of `n` empty nested arrays (12 bytes each on disk).
std::vector<std::byte> nested_arrays(std::uint64_t n) {
    Bytes v;
    v.put<std::uint32_t>(9).put<std::uint32_t>(9).put<std::uint64_t>(n);
    for (std::uint64_t i = 0; i < n; ++i) v.put<std::uint32_t>(0).put<std::uint64_t>(0);  // u8[0]
    GgufSpec s;
    s.kv("evil.arrays", std::move(v));
    return s.build(false);
}

std::vector<std::byte> u8_array(std::uint64_t n) {
    Bytes v;
    v.put<std::uint32_t>(9).put<std::uint32_t>(0).put<std::uint64_t>(n);
    for (std::uint64_t i = 0; i < n; ++i) v.put<std::uint8_t>(static_cast<std::uint8_t>(i));
    GgufSpec s;
    s.kv("evil.bytes", std::move(v));
    return s.build(false);
}

std::vector<std::byte> empty_strings(std::uint64_t n) {
    Bytes v;
    v.put<std::uint32_t>(9).put<std::uint32_t>(8).put<std::uint64_t>(n);
    for (std::uint64_t i = 0; i < n; ++i) v.put<std::uint64_t>(0);
    GgufSpec s;
    s.kv("evil.strings", std::move(v));
    return s.build(false);
}

ErrorCode parse_code(std::vector<std::byte> img, std::uint64_t budget, std::string* what = nullptr) {
    try {
        (void)GgufFile::parse(std::move(img), GgufMode::HeaderOnly, "budget.gguf", GgufOptions{budget});
    } catch (const halo::Error& e) {
        if (what != nullptr) *what = e.what();
        return e.code();
    }
    return ErrorCode::Cancelled;  // "did not throw"
}

}  // namespace

TEST(GgufBudget, EmptyNestedArraysAreChargedAtTheirStoredSize) {
    // 100K empty arrays: 1.2 MB on disk, 100K x sizeof(GgufArray) (~15 MB) in memory.
    const auto img = nested_arrays(100000);
    std::printf("100000 empty arrays: %zu bytes on disk, %zu bytes each in memory\n", img.size(), sizeof(GgufArray));
    std::string what;
    EXPECT_EQ(parse_code(img, 4 * kMiB, &what), ErrorCode::Model);
    EXPECT_NE(what.find("metadata allocation budget of 4194304 bytes exceeded"), std::string::npos) << what;
    const GgufFile ok = GgufFile::parse(img, GgufMode::HeaderOnly, "ok.gguf");  // default budget: 1 GiB
    EXPECT_GE(ok.metadata_bytes(), 100000 * sizeof(GgufArray));
    EXPECT_EQ(ok.get_array("evil.arrays")->size(), 100000u);
}

TEST(GgufBudget, NarrowElementsAreChargedAtEightBytes) {
    const auto bytes = u8_array(1000000);  // 1 MB on disk, 8 MB stored as uint64_t
    EXPECT_EQ(parse_code(bytes, 4 * kMiB), ErrorCode::Model);
    EXPECT_EQ(parse_code(bytes, 16 * kMiB), ErrorCode::Cancelled);
    const auto strings = empty_strings(1000000);  // 8 MB on disk, 32 MB of std::string
    EXPECT_EQ(parse_code(strings, 16 * kMiB), ErrorCode::Model);
    EXPECT_EQ(parse_code(strings, 64 * kMiB), ErrorCode::Cancelled);
    // Heap-allocated strings are charged their bytes too: 100K x 100 B = 10 MB of text
    // plus 3.2 MB of std::string slots.
    Bytes v;
    v.put<std::uint32_t>(9).put<std::uint32_t>(8).put<std::uint64_t>(100000);
    for (int i = 0; i < 100000; ++i) v.str(std::string(100, 'x'));
    GgufSpec s;
    s.kv("evil.long", std::move(v));
    EXPECT_EQ(parse_code(s.build(false), 8 * kMiB), ErrorCode::Model);
    EXPECT_EQ(parse_code(s.build(false), 16 * kMiB), ErrorCode::Cancelled);
}

TEST(GgufBudget, DefaultAndDisabledBudget) {
    EXPECT_EQ(GgufOptions{}.max_metadata_bytes, GgufLimits::kMaxMetadataBytes);
    EXPECT_EQ(GgufLimits::kMaxMetadataBytes, std::uint64_t{1} << 30);
    EXPECT_EQ(parse_code(nested_arrays(100000), 0), ErrorCode::Cancelled);  // 0 disables
    // The budget covers tensor infos as well.
    GgufSpec s;
    for (int i = 0; i < 2000; ++i) s.tensor("t" + std::to_string(i), {32});
    EXPECT_EQ(parse_code(s.build(false), 64 * 1024), ErrorCode::Model);
    EXPECT_EQ(parse_code(s.build(false), 4 * kMiB), ErrorCode::Cancelled);
}

namespace {

long vm_hwm_kib() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::strtol(line.c_str() + 6, nullptr, 10);
    }
    return -1;
}

// Child: 2M empty nested arrays (24 MB image, ~300 MB if stored) against a 64 MiB budget.
// Exits 0 only if rejected with Model while peak RSS grew by less than 96 MiB.
[[noreturn]] void amplification_child() {
    std::vector<std::byte> img = nested_arrays(2000000);
    {
        std::ofstream clear("/proc/self/clear_refs");
        clear << "5";  // reset VmHWM
    }
    const long before = vm_hwm_kib();
    if (before < 0) std::_Exit(4);
    int rc = 1;
    try {
        (void)GgufFile::parse(std::move(img), GgufMode::HeaderOnly, "amp.gguf", GgufOptions{64 * kMiB});
    } catch (const halo::Error& e) {
        rc = e.code() == ErrorCode::Model ? 0 : 2;
    }
    const long grew = vm_hwm_kib() - before;
    std::fprintf(stderr, "peak RSS grew by %ld KiB\n", grew);
    if (rc == 0 && grew > 96 * 1024) rc = 3;
    std::_Exit(rc);
}

}  // namespace

TEST(GgufBudget, BudgetBoundsPeakMemory) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(amplification_child(), ::testing::ExitedWithCode(0), "");
}

TEST(GgufBudget, RealHeadersUseAFractionOfTheDefault) {
    const fs::path ref = ref_dir();
    int seen = 0;
    for (const char* k : {"ggml-org-q4km", "ggml-org-mtp-q4_0", "unsloth-ud-q4kxl"}) {
        const fs::path p = ref / (std::string(k) + ".header.gguf");
        if (!fs::exists(p)) continue;
        ++seen;
        const GgufFile f = GgufFile::open(p, GgufMode::HeaderOnly);
        std::printf("%s: header %llu bytes, metadata charged %llu bytes\n", k,
                    static_cast<unsigned long long>(f.header_size()), static_cast<unsigned long long>(f.metadata_bytes()));
        EXPECT_LE(f.metadata_bytes() * 8, GgufLimits::kMaxMetadataBytes) << k;
        // The charge is not trivially small: it covers the 248K-token vocabulary.
        EXPECT_GE(f.metadata_bytes(), f.header_size()) << k;
    }
    if (seen == 0) GTEST_SKIP() << "real GGUF headers missing under " << ref;
}
