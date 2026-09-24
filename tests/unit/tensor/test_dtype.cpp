#include <gtest/gtest.h>

#include <string>

#include "halo/core/error.h"
#include "halo/tensor/dtype.h"
#include "ref_data.h"

using halo::DType;

TEST(DType, IdsMapOneToOneWithGgml) {
    for (std::uint32_t id = 0; id <= 30; ++id) {
        const auto t = halo::dtype_from_id(id);
        if (id == 4 || id == 5) {
            EXPECT_FALSE(t.has_value()) << id;  // removed Q4_2 / Q4_3
            EXPECT_TRUE(halo::ggml_type_name(id).empty());
            continue;
        }
        ASSERT_TRUE(t.has_value()) << id;
        EXPECT_EQ(static_cast<std::uint32_t>(*t), id);
        EXPECT_EQ(halo::traits(*t).name, halo::ggml_type_name(id));
    }
    // ggml types newer than the DType enum: named, but not a DType.
    for (std::uint32_t id : {34u, 35u, 39u, 40u, 41u, 42u}) {
        EXPECT_FALSE(halo::dtype_from_id(id).has_value()) << id;
        EXPECT_FALSE(halo::ggml_type_name(id).empty()) << id;
    }
    EXPECT_EQ(halo::ggml_type_name(39), "MXFP4");
    // Never-valid ids.
    for (std::uint32_t id : {31u, 32u, 33u, 36u, 37u, 38u, 43u, 1000u, 0xFFFFFFFFu}) {
        EXPECT_FALSE(halo::dtype_from_id(id).has_value()) << id;
        EXPECT_TRUE(halo::ggml_type_name(id).empty()) << id;
    }
    try {
        (void)halo::traits(static_cast<DType>(4));
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
    }
}

// Geometry oracle: gguf-py's GGML_QUANT_SIZES, recorded by python/tools/fetch_quant_blocks.py.
TEST(DType, GeometryMatchesGgufPy) {
    const auto m = halo::test::read_json(halo::test::ref_dir() / "quant_blocks" / "manifest.json");
    if (!m) GTEST_SKIP() << "no quant_blocks/manifest.json under " << HALO_REF_DIR
                         << " (run python/tools/fetch_quant_blocks.py)";
    int checked = 0;
    for (const auto& [name, q] : (*m)["quant_sizes"].items()) {
        const auto id = q["id"].get<std::uint32_t>();
        const auto t = halo::dtype_from_id(id);
        if (!t) {
            // Only ggml types HALO deliberately lacks may be missing.
            EXPECT_FALSE(halo::ggml_type_name(id).empty()) << name << " id " << id << " unknown to HALO";
            EXPECT_EQ(halo::ggml_type_name(id), name);
            continue;
        }
        const auto& tr = halo::traits(*t);
        EXPECT_EQ(tr.name, name);
        EXPECT_EQ(tr.block_elems, q["block_elems"].get<std::size_t>()) << name;
        if (*t == DType::Q8_1) {
            // Known oracle defect: gguf-py 0.19.0 lists Q8_1 as 40 bytes (4-byte d, s), but ggml
            // defines block_q8_1 { ggml_half d, s; int8 qs[32]; } with
            // static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1) == 36
            // (ggml/src/ggml-common.h, llama.cpp bd4f514db). ggml defines the file layout.
            // If gguf-py is ever fixed this branch fails and should be removed.
            EXPECT_EQ(tr.block_bytes, 36u);
            EXPECT_EQ(q["block_bytes"].get<std::size_t>(), 40u);
        } else {
            EXPECT_EQ(tr.block_bytes, q["block_bytes"].get<std::size_t>()) << name;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 29);  // every DType enumerator
}

TEST(DType, DequantSupportedSetMatchesD007) {
    for (auto t : {DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1,
                   DType::Q8_0, DType::Q2_K, DType::Q3_K, DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::IQ4_NL,
                   DType::IQ4_XS, DType::IQ3_S}) {
        EXPECT_TRUE(halo::dequant_supported(t)) << halo::traits(t).name;
    }
    for (auto t : {DType::Q8_1, DType::Q8_K, DType::IQ2_XXS, DType::IQ2_XS, DType::IQ3_XXS, DType::IQ1_S,
                   DType::IQ2_S, DType::IQ1_M, DType::I8, DType::I16, DType::I32, DType::I64, DType::F64}) {
        EXPECT_FALSE(halo::dequant_supported(t)) << halo::traits(t).name;
    }
}
