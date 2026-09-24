// GGUF -> CPU adapters (code review S-4): WeightRef -> WeightMatrix, fp32 vectors,
// embedding rows, TokenizerMetadata -> VocabSpec -> Tokenizer.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "halo/models/qwen35.h"
#include "halo/tensor/quant.h"
#include "halo/tokenizer/tokenizer.h"
#include "tiny_golden.h"

namespace fs = std::filesystem;
using halo::models::weight_matrix;
using halo::models::weight_vector;

namespace {

std::unique_ptr<halo::model::NormalizedModel> load(const char* file) {
    const auto p = halo::test::tiny_dir() / file;
    if (!fs::exists(p)) return nullptr;
    return std::make_unique<halo::model::NormalizedModel>(halo::model::NormalizedModel::load(p));
}

std::vector<float> dequant_raw(const halo::model::WeightRef& w, std::size_t first, std::size_t n) {
    const std::size_t rb = halo::tensor::row_bytes(w.type(), w.ne(0));
    std::vector<float> out(n * static_cast<std::size_t>(w.ne(0)));
    halo::tensor::dequantize_row(w.type(), w.data().subspan(first * rb, n * rb), out);
    return out;
}

std::string from_hex(const std::string& h) {
    std::string s;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2) s.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    return s;
}

}  // namespace

TEST(ModelsAdapters, WeightMatrixRowsEqualDequantizeRowForEveryType) {
    std::size_t files = 0;
    for (const char* f : {"tiny-f32.gguf", "tiny-q8_0.gguf", "tiny-q4_k_m.gguf", "tiny-q6_k.gguf"}) {
        const auto m = load(f);
        if (!m) continue;
        ++files;
        SCOPED_TRACE(f);
        for (const auto* w : {&m->layer(0).gdn.qkv, &m->layer(3).attn.q, &m->layer(0).ffn_down, &m->output()}) {
            const auto wm = weight_matrix(*w);
            ASSERT_EQ(wm.rows(), static_cast<std::size_t>(w->ne(1)));
            ASSERT_EQ(wm.cols(), static_cast<std::size_t>(w->ne(0)));
            EXPECT_EQ(wm.is_dense(), w->type() == halo::DType::F32);
            std::vector<float> scratch;
            const std::size_t first = wm.rows() / 3, n = std::min<std::size_t>(5, wm.rows() - first);
            const auto blk = wm.row_block(first, n, scratch);
            const auto ref = dequant_raw(*w, first, n);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < wm.cols(); ++c) ASSERT_EQ(blk(r, c), ref[r * wm.cols() + c]) << w->name();
            }
            // Sub-range matrix starts at its first row.
            const auto sub = weight_matrix(*w, first, n);
            const auto sb = sub.row_block(0, n, scratch);
            for (std::size_t c = 0; c < wm.cols(); ++c) ASSERT_EQ(sb(n - 1, c), ref[(n - 1) * wm.cols() + c]);
            EXPECT_THROW((void)weight_matrix(*w, wm.rows(), 1), halo::Error);
        }
    }
    if (files == 0) GTEST_SKIP() << "no tiny GGUF under " << halo::test::tiny_dir();
}

TEST(ModelsAdapters, WeightVectorAndEmbeddingRow) {
    const auto m = load("tiny-q8_0.gguf");
    if (!m) GTEST_SKIP() << "tiny-q8_0.gguf missing";
    const auto v = weight_vector(m->layer(0).gdn.a);
    EXPECT_EQ(v.size(), 6u);
    for (const float x : v) EXPECT_LT(x, 0.0f);
    EXPECT_THROW((void)weight_matrix(m->layer(0).gdn.a), halo::Error) << "1-D tensor is not a matrix";
    std::vector<float> row(256);
    halo::models::embedding_row(m->token_embd(), 9707, row);
    EXPECT_EQ(row, dequant_raw(m->token_embd(), 9707, 1));
    EXPECT_THROW(halo::models::embedding_row(m->token_embd(), -1, row), halo::Error);
    EXPECT_THROW(halo::models::embedding_row(m->token_embd(), 248320, row), halo::Error);
}

TEST(ModelsAdapters, VocabSpecFromGgufReproducesTokenizerGolden) {
    const auto m = load("tiny-f32.gguf");
    const fs::path cases = fs::path(HALO_REF_DIR) / "tokenizer_golden" / "cases.json";
    if (!m) GTEST_SKIP() << "tiny-f32.gguf missing";
    if (!fs::exists(cases)) GTEST_SKIP() << "tokenizer golden missing: " << cases;
    const auto tok = halo::tokenizer::Tokenizer::from_spec(halo::models::vocab_spec(m->tokenizer()));
    EXPECT_EQ(tok.vocab_size(), 248320u);
    EXPECT_EQ(tok.bos(), 248044);
    EXPECT_EQ(tok.eos(), 248046);
    std::ifstream f(cases);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto j = nlohmann::json::parse(ss.str())["cases"];
    ASSERT_GT(j.size(), 40u);
    for (const auto& c : j) {
        const std::string in = c.contains("bytes_hex") ? from_hex(c["bytes_hex"].get<std::string>()) : c["text"].get<std::string>();
        EXPECT_EQ(tok.encode(in, true), c["ids"].get<std::vector<std::int32_t>>()) << c["name"];
    }
}

TEST(ModelsAdapters, VocabSpecRejectsForeignTokenizers) {
    halo::model::TokenizerMetadata meta;
    std::vector<std::string> toks = {"a", "b"};
    meta.tokens = toks;
    meta.model = "llama";
    meta.pre = "qwen35";
    EXPECT_THROW((void)halo::models::vocab_spec(meta), halo::Error);
    meta.model = "gpt2";
    meta.pre = "default";
    EXPECT_THROW((void)halo::models::vocab_spec(meta), halo::Error);
    meta.pre = "qwen35";
    meta.eos_id = 7;  // outside the 2-token vocabulary
    EXPECT_THROW((void)halo::models::vocab_spec(meta), halo::Error);
    meta.eos_id.reset();
    std::vector<std::int64_t> types = {1, 9};
    meta.token_types = types;
    EXPECT_THROW((void)halo::models::vocab_spec(meta), halo::Error);
}
