#include <gtest/gtest.h>
#include "halo/sampling/sampler.h"
#include "halo/sampling/structured.h"
using namespace halo::sampling;
TEST(Smoke, ByteMatcher) {
  const auto g = Grammar::from_json_schema(R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})");
  ByteMatcher m(g);
  EXPECT_TRUE(m.accept_bytes(R"({"a": 12})"));
  EXPECT_TRUE(m.is_accepting());
  ByteMatcher m2(g);
  EXPECT_FALSE(m2.accept_bytes(R"({"a":1,})"));
}
#include <chrono>
#include <cstdio>
#include <filesystem>
#include "halo/tokenizer/tokenizer.h"
TEST(Smoke, RealVocabFacts) {
  namespace fs = std::filesystem;
  const fs::path ref = HALO_REF_DIR;
  if (!fs::exists(ref / "tokenizer.json")) GTEST_SKIP() << "no tokenizer";
  auto tok = halo::tokenizer::Tokenizer::from_hf_dir(ref);
  std::printf("vocab=%zu eos=%d bos=%d pad=%d\n", tok.vocab_size(), tok.eos().value_or(-1), tok.bos().value_or(-1), tok.pad().value_or(-1));
  for (int id : {248044, 248045, 248046, 248047, 248068, 248069}) {
    std::printf("id %d type %d piece '%s'\n", id, static_cast<int>(tok.token_type(id)), tok.token_to_piece(id).c_str());
  }
  int single = 0;
  for (int b = 0; b < 256; ++b) {
    auto id = tok.piece_to_id(std::string(1, static_cast<char>(b)));
    if (id && tok.token_type(*id) == halo::tokenizer::TokenType::Normal) ++single;
  }
  std::printf("single-byte normal tokens: %d\n", single);
  std::size_t types[8] = {};
  for (std::size_t i = 0; i < tok.vocab_size(); ++i) types[static_cast<int>(tok.token_type(static_cast<int>(i)))]++;
  for (int t = 0; t < 8; ++t) std::printf("type %d: %zu\n", t, types[t]);
  for (int rep = 0; rep < 2; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    auto v = TokenVocab::build(tok);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("build %.1f ms nodes=%zu maxdepth=%zu\n", std::chrono::duration<double, std::milli>(t1 - t0).count(), v->trie_nodes(), v->max_depth_);
  }
}
