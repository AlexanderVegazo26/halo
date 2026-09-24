// Differential test: HALO dequantization vs gguf-py (`gguf.quants.dequantize`, v0.19.0) on
// real blocks range-fetched from the published Qwen3.8-27B GGUF files, real llama-quantize
// output from the tiny models, and seeded synthetic blocks. Data: python/tools/fetch_quant_blocks.py.
//
// Tolerance: zero. HALO keeps ggml's (== gguf-py's) float evaluation order and disables FP
// contraction, and every step is a correctly rounded IEEE float32 operation in both, so the
// outputs must be bit-identical. NaN is compared as "both NaN" (payloads may legitimately
// differ between numpy's and our fp16 widening of signalling NaNs).

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>

#include "halo/tensor/quant.h"
#include "ref_data.h"

namespace ht = halo::tensor;
using halo::test::read_bytes;
using halo::test::ref_dir;

namespace {

std::size_t compare(const std::vector<float>& got, const std::vector<float>& want, std::string& first_diff) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const bool both_nan = std::isnan(got[i]) && std::isnan(want[i]);
        if (!both_nan && std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i])) {
            if (bad++ == 0) {
                first_diff = "index " + std::to_string(i) + ": got " + std::to_string(got[i]) + " want " +
                             std::to_string(want[i]);
            }
        }
    }
    return bad;
}

}  // namespace

TEST(QuantBlocks, BitExactVsGgufPy) {
    const auto dir = ref_dir() / "quant_blocks";
    const auto m = halo::test::read_json(dir / "manifest.json");
    if (!m) GTEST_SKIP() << "no " << (dir / "manifest.json") << " (run python/tools/fetch_quant_blocks.py)";

    std::map<std::string, std::set<std::string>> kinds_by_type;
    std::size_t elements = 0;
    for (const auto& e : (*m)["entries"]) {
        const auto id = e["id"].get<std::string>();
        SCOPED_TRACE(id);
        const auto t = halo::dtype_from_id(e["type_id"].get<std::uint32_t>());
        ASSERT_TRUE(t.has_value());
        const auto rows = e["rows"].get<std::size_t>();
        const auto ne0 = e["ne0"].get<std::int64_t>();
        const auto raw = read_bytes(dir / e["raw"].get<std::string>());
        const auto ref_bytes = read_bytes(dir / e["ref"].get<std::string>());
        ASSERT_EQ(raw.size(), rows * ht::row_bytes(*t, ne0));
        ASSERT_EQ(raw.size(), rows * e["row_bytes"].get<std::size_t>());
        const std::size_t n = rows * static_cast<std::size_t>(ne0);
        ASSERT_EQ(ref_bytes.size(), n * sizeof(float));
        std::vector<float> want(n);
        std::memcpy(want.data(), ref_bytes.data(), ref_bytes.size());

        // Aligned buffer, all rows at once.
        std::vector<float> got(n);
        ht::dequantize_row(*t, raw, got);
        std::string diff;
        EXPECT_EQ(compare(got, want, diff), 0u) << e["type"].get<std::string>() << " first diff at " << diff;

        // Deliberately misaligned source (odd address), row by row: same bits.
        std::vector<std::byte> shifted(raw.size() + 1);
        std::memcpy(shifted.data() + 1, raw.data(), raw.size());
        std::vector<float> got2(n);
        const std::size_t rb = ht::row_bytes(*t, ne0);
        for (std::size_t r = 0; r < rows; ++r) {
            ht::dequantize_row(*t, shifted.data() + 1 + r * rb, got2.data() + r * static_cast<std::size_t>(ne0), ne0);
        }
        EXPECT_EQ(compare(got2, want, diff), 0u) << "misaligned: " << diff;

        kinds_by_type[e["type"].get<std::string>()].insert(e["kind"].get<std::string>());
        elements += n;
    }

    // Coverage: every type HALO dequantizes has a reference; the real-file types have real data.
    for (const char* ty : {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K",
                           "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "IQ3_S"}) {
        EXPECT_TRUE(kinds_by_type.contains(ty)) << "no reference blocks for " << ty;
    }
    for (const char* ty : {"F32", "Q4_0", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ3_S", "IQ4_NL", "IQ4_XS"}) {
        EXPECT_TRUE(kinds_by_type[ty].contains("remote")) << ty << " lacks real (remote) blocks";
    }
    for (const char* ty : {"Q5_0", "Q5_1"}) {
        EXPECT_TRUE(kinds_by_type[ty].contains("local")) << ty << " lacks llama-quantize (tiny) blocks";
    }
    RecordProperty("elements_compared", std::to_string(elements));
    std::printf("[ info ] compared %zu elements over %zu entries\n", elements, (*m)["entries"].size());
}
