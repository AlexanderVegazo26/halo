// Tokenizer tests. Synthetic-vocab tests need no data; golden tests compare against
// HF tokenizers output written by python/tools/make_tokenizer_golden.py golden and skip
// (never pass) when $HALO_REF_DIR/tokenizer_golden is missing.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "halo/core/error.h"
#include "halo/tokenizer/tokenizer.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HALO_TEST_ASAN 1
#endif
#endif
#ifndef HALO_TEST_ASAN
#define HALO_TEST_ASAN 0
#endif

namespace fs = std::filesystem;
using halo::ErrorCode;
using halo::tokenizer::Encoding;
using halo::tokenizer::StreamDecoder;
using halo::tokenizer::Tokenizer;
using halo::tokenizer::TokenType;
using halo::tokenizer::VocabSpec;
using Json = nlohmann::json;

namespace {

const fs::path kRef = HALO_REF_DIR;
const fs::path kGolden = kRef / "tokenizer_golden";

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string from_hex(const std::string& h) {
    std::string out;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
        out.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    }
    return out;
}

// GPT-2 bytes_to_unicode, re-derived here independently of the implementation.
std::string mapped(unsigned char b) {
    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
    for (int c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
    for (int c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);
    std::vector<int> cs = bs;
    int n = 0;
    for (int c = 0; c < 256; ++c) {
        if (std::find(bs.begin(), bs.end(), c) == bs.end()) {
            bs.push_back(c);
            cs.push_back(256 + n++);
        }
    }
    for (std::size_t i = 0; i < bs.size(); ++i) {
        if (bs[i] == b) {
            std::string s;
            halo::tokenizer::unicode::append_utf8(s, static_cast<char32_t>(cs[i]));
            return s;
        }
    }
    return {};
}

// 256 byte tokens + merges over lowercase ASCII + a few added tokens.
VocabSpec tiny_spec() {
    VocabSpec s;
    for (int b = 0; b < 256; ++b) s.tokens.push_back(mapped(static_cast<unsigned char>(b)));
    s.types.assign(256, TokenType::Normal);
    auto add = [&](std::string tok, TokenType t) {
        s.tokens.push_back(std::move(tok));
        s.types.push_back(t);
        return static_cast<std::int32_t>(s.tokens.size() - 1);
    };
    add("aa", TokenType::Normal);        // 256
    add("ab", TokenType::Normal);        // 257
    add(mapped(' ') + "a", TokenType::Normal);  // 258 " a"
    add("aab", TokenType::Normal);       // 259
    s.merges = {"a a", "a b", mapped(' ') + " a", "aa b"};
    s.bos = add("<|s|>", TokenType::Control);    // 260
    add("<t>", TokenType::UserDefined);          // 261
    add("<|s|>x", TokenType::Control);           // 262: longest match must win
    add("[PAD]", TokenType::Unused);             // 263
    return s;
}

struct Golden {
    std::unique_ptr<Tokenizer> tok;
    Json cases, fuzz, sweep, nfc, decode, long_text, gguf;
    std::string skip;
};

// Loaded lazily (never at static-init time: gtest discovery must stay fast).
Golden& golden() {
    static Golden g = [] {
        Golden r;
        if (!fs::exists(kRef / "tiny" / "hf" / "tokenizer.json")) {
            r.skip = "reference tokenizer missing: " + (kRef / "tiny/hf/tokenizer.json").string();
            return r;
        }
        for (const char* f : {"cases.json", "fuzz.json", "sweep.json", "nfc.json", "decode.json", "long.json"}) {
            if (!fs::exists(kGolden / f)) {
                r.skip = "golden data missing (" + (kGolden / f).string() +
                         "): run python/tools/make_tokenizer_golden.py golden";
                return r;
            }
        }
        // A present-but-corrupt file is a failure, not a skip: the parse below throws inside
        // the test body and gtest reports it.
        r.tok = std::make_unique<Tokenizer>(Tokenizer::from_hf_dir(kRef / "tiny" / "hf"));
        r.cases = Json::parse(read_file(kGolden / "cases.json"))["cases"];
        r.fuzz = Json::parse(read_file(kGolden / "fuzz.json"))["cases"];
        r.sweep = Json::parse(read_file(kGolden / "sweep.json"))["chunks"];
        r.nfc = Json::parse(read_file(kGolden / "nfc.json"))["chunks"];
        r.decode = Json::parse(read_file(kGolden / "decode.json"))["cases"];
        r.long_text = Json::parse(read_file(kGolden / "long.json"));
        return r;
    }();
    return g;
}

#define REQUIRE_GOLDEN()                              \
    auto& G = golden();                               \
    if (!G.tok) GTEST_SKIP() << G.skip;               \
    const Tokenizer& tok = *G.tok

std::string case_input(const Json& c) {
    return c.contains("bytes_hex") ? from_hex(c["bytes_hex"].get<std::string>()) : c["text"].get<std::string>();
}

std::vector<std::int32_t> ids_of(const Json& j) { return j.get<std::vector<std::int32_t>>(); }

}  // namespace

// ---- synthetic vocabulary ----------------------------------------------------------------

TEST(TokenizerSpec, BpeMergesLowestRankThenLeftmost) {
    const Tokenizer t = Tokenizer::from_spec(tiny_spec());
    // "aaa": only "a a" applies at pos 0 and 1 with equal rank; HF picks the leftmost.
    EXPECT_EQ(t.encode("aaa", true), (std::vector<std::int32_t>{256, 'a'}));
    // "aab": "a a"(rank 0) first, then "aa b"(rank 3) -> one token.
    EXPECT_EQ(t.encode("aab", true), (std::vector<std::int32_t>{259}));
    // " ab": " a" (rank 2) vs "a b" (rank 1): "a b" wins -> [" ", "ab"].
    EXPECT_EQ(t.encode(" ab", true), (std::vector<std::int32_t>{' ', 257}));
}

TEST(TokenizerSpec, AddedTokensSplitFirstLongestWins) {
    const Tokenizer t = Tokenizer::from_spec(tiny_spec());
    EXPECT_EQ(t.encode("aa<|s|>xaa", true), (std::vector<std::int32_t>{256, 262, 256}));
    EXPECT_EQ(t.encode("aa<|s|>aa", true), (std::vector<std::int32_t>{256, 260, 256}));
    // parse_special=false: Control tokens become text, UserDefined still match.
    const auto ids = t.encode("<|s|><t>", false);
    EXPECT_EQ(ids.back(), 261);
    EXPECT_EQ(std::count(ids.begin(), ids.end(), 260), 0);
    EXPECT_EQ(t.decode(ids, false), "<|s|><t>");
    EXPECT_EQ(t.decode(std::vector<std::int32_t>{260, 'a', 261, 263}, true), "a<t>");
    EXPECT_EQ(t.decode(std::vector<std::int32_t>{260, 'a', 263}, false), "<|s|>a");
    EXPECT_TRUE(t.is_special(260));
    EXPECT_FALSE(t.is_special(261));
    EXPECT_EQ(t.bos(), 260);
    EXPECT_EQ(t.piece_to_id("ab"), 257);
    EXPECT_EQ(t.token_to_piece(263), "");
}

TEST(TokenizerSpec, RejectsMalformedSpecsWithTypedErrors) {
    const auto code_of = [](VocabSpec s) {
        try {
            (void)Tokenizer::from_spec(std::move(s));
        } catch (const halo::Error& e) {
            return e.code();
        }
        return ErrorCode::Cancelled;  // "did not throw"
    };
    auto bad_merge = tiny_spec();
    bad_merge.merges.push_back("a zz");
    EXPECT_EQ(code_of(bad_merge), ErrorCode::Model);
    auto bad_form = tiny_spec();
    bad_form.merges.push_back("a a a");
    EXPECT_EQ(code_of(bad_form), ErrorCode::Model);
    auto no_byte = tiny_spec();
    no_byte.tokens[65] = "not-a-byte";
    EXPECT_EQ(code_of(no_byte), ErrorCode::Model);
    auto bad_type = tiny_spec();
    bad_type.types[5] = TokenType::Byte;
    EXPECT_EQ(code_of(bad_type), ErrorCode::Unsupported);
    auto bad_bos = tiny_spec();
    bad_bos.bos = 100000;
    EXPECT_EQ(code_of(bad_bos), ErrorCode::Model);
    auto size_mismatch = tiny_spec();
    size_mismatch.types.pop_back();
    EXPECT_EQ(code_of(size_mismatch), ErrorCode::Model);
    auto garbage_type = tiny_spec();
    garbage_type.types[7] = static_cast<TokenType>(42);
    EXPECT_EQ(code_of(garbage_type), ErrorCode::Model);
}

TEST(TokenizerSpec, DecodeRejectsOutOfRangeIds) {
    const Tokenizer t = Tokenizer::from_spec(tiny_spec());
    try {
        (void)t.decode(std::vector<std::int32_t>{-1}, false);
        FAIL() << "expected throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Api);
    }
    EXPECT_THROW((void)t.token_to_piece(static_cast<std::int32_t>(t.vocab_size())), halo::Error);
}

TEST(TokenizerSpec, InvalidUtf8IsReplacedBeforeEncoding) {
    const Tokenizer t = Tokenizer::from_spec(tiny_spec());
    const std::string r = "\xef\xbf\xbd";
    EXPECT_EQ(t.decode(t.encode("a\xff" "b", true), false), "a" + r + "b");
    const Encoding e = t.encode_with_offsets("a\xff" "b", true);
    // U+FFFD is 3 byte tokens that all map back to the single invalid input byte at 1.
    EXPECT_EQ(e.offsets, (std::vector<std::size_t>{0, 1, 1, 1, 2}));
}

TEST(StreamDecoder, HoldsPartialUtf8AndFlushesReplacement) {
    const Tokenizer t = Tokenizer::from_spec(tiny_spec());
    StreamDecoder d(t);
    // U+20AC (e2 82 ac) fed one byte token at a time.
    EXPECT_EQ(d.push(0xE2), "");
    EXPECT_EQ(d.push(0x82), "");
    EXPECT_EQ(d.pending_bytes(), 2u);
    EXPECT_EQ(d.push(0xAC), "\xe2\x82\xac");
    EXPECT_EQ(d.push('a'), "a");
    // An impossible continuation is emitted as U+FFFD immediately, the new byte kept.
    EXPECT_EQ(d.push(0xE2), "");
    EXPECT_EQ(d.push('x'), "\xef\xbf\xbdx");
    EXPECT_EQ(d.push(0xF0), "");
    EXPECT_EQ(d.flush(), "\xef\xbf\xbd");
    EXPECT_EQ(d.pending_bytes(), 0u);
    StreamDecoder skip(t, true);
    EXPECT_EQ(skip.push(260), "");
    EXPECT_EQ(skip.push(261), "<t>");
}

TEST(TokenizerSpec, OffsetsForAddedTokensAndBytes) {
    const Tokenizer t = Tokenizer::from_spec(tiny_spec());
    const Encoding e = t.encode_with_offsets("aa<|s|> ab<t>", true);
    EXPECT_EQ(e.ids, (std::vector<std::int32_t>{256, 260, ' ', 257, 261}));
    EXPECT_EQ(e.offsets, (std::vector<std::size_t>{0, 2, 7, 8, 10}));
    EXPECT_EQ(e.token_index_at(7), 2u);
    EXPECT_EQ(e.token_index_at(2), 1u);
    EXPECT_EQ(e.token_index_at(100), 5u);
}

// ---- HF tokenizer.json ingestion checks ----------------------------------------------------

namespace {

Json tiny_hf_json() {
    Json vocab = Json::object();
    for (int b = 0; b < 256; ++b) vocab[mapped(static_cast<unsigned char>(b))] = b;
    vocab["aa"] = 256;
    const std::string regex =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    return Json{
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens",
         Json::array({{{"id", 257}, {"content", "<|end|>"}, {"single_word", false}, {"lstrip", false},
                       {"rstrip", false}, {"normalized", false}, {"special", true}}})},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           Json::array({{{"type", "Split"}, {"pattern", {{"Regex", regex}}}, {"behavior", "Isolated"}, {"invert", false}},
                        {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", false}}})}}},
        {"post_processor", {{"type", "ByteLevel"}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"model",
         {{"type", "BPE"}, {"dropout", nullptr}, {"unk_token", nullptr}, {"continuing_subword_prefix", ""},
          {"end_of_word_suffix", ""}, {"fuse_unk", false}, {"byte_fallback", false}, {"ignore_merges", false},
          {"vocab", vocab}, {"merges", Json::array({Json::array({"a", "a"})})}}}};
}

ErrorCode hf_error(const Json& j, const std::string& cfg = {}) {
    try {
        (void)Tokenizer::from_hf_json(j.dump(), cfg);
    } catch (const halo::Error& e) {
        return e.code();
    }
    return ErrorCode::Cancelled;
}

}  // namespace

TEST(TokenizerHfJson, AcceptsSupportedConfig) {
    const Tokenizer t = Tokenizer::from_hf_json(tiny_hf_json().dump(), R"({"eos_token": {"content": "<|end|>"}})");
    EXPECT_EQ(t.vocab_size(), 258u);
    EXPECT_EQ(t.eos(), 257);
    EXPECT_EQ(t.encode("aaa<|end|>", true), (std::vector<std::int32_t>{256, 'a', 257}));
    EXPECT_TRUE(t.is_special(257));
}

TEST(TokenizerHfJson, RejectsUnimplementedConfigs) {
    auto j = tiny_hf_json();
    j["normalizer"] = {{"type", "NFKC"}};
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\p{L}+";
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = true;
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["model"]["byte_fallback"] = true;
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["model"]["dropout"] = 0.1;
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["model"]["type"] = "WordPiece";
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["added_tokens"][0]["lstrip"] = true;
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["post_processor"] = {{"type", "TemplateProcessing"}};
    EXPECT_EQ(hf_error(j), ErrorCode::Unsupported);
    j = tiny_hf_json();
    j["model"]["merges"] = Json::array({"a q"});
    EXPECT_EQ(hf_error(j), ErrorCode::Model);
    j = tiny_hf_json();
    j["model"]["vocab"]["neg"] = -5;
    EXPECT_EQ(hf_error(j), ErrorCode::Model);
    EXPECT_EQ(hf_error(tiny_hf_json(), R"({"eos_token": "<|nope|>"})"), ErrorCode::Model);
    try {
        (void)Tokenizer::from_hf_json("{not json", {});
        FAIL();
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Model);
    }
}

// ---- golden: HF tokenizers on the real 248K vocabulary -------------------------------------

TEST(TokenizerGolden, VocabularyShape) {
    REQUIRE_GOLDEN();
    EXPECT_EQ(tok.vocab_size(), 248077u);  // tokenizer.json: 248044 + 33 added (GGUF pads to 248320)
    EXPECT_EQ(tok.eos(), 248046);
    EXPECT_EQ(tok.pad(), 248044);
    EXPECT_FALSE(tok.bos().has_value());
    EXPECT_TRUE(tok.is_special(248045));    // <|im_start|>
    EXPECT_FALSE(tok.is_special(248068));   // <think> is added but not special
    EXPECT_EQ(tok.normalizer(), halo::tokenizer::Normalizer::Nfc);
}

TEST(TokenizerGolden, CorpusMatchesHfTokenForToken) {
    REQUIRE_GOLDEN();
    ASSERT_GT(G.cases.size(), 40u);
    for (const auto& c : G.cases) {
        const std::string name = c["name"].get<std::string>();
        const std::string in = case_input(c);
        EXPECT_EQ(tok.encode(in, true), ids_of(c["ids"])) << name;
        EXPECT_EQ(tok.encode(in, false), ids_of(c["ids_no_special"])) << name;
        const auto ids = ids_of(c["ids"]);
        EXPECT_EQ(tok.decode(ids, false), c["decoded"].get<std::string>()) << name;
        EXPECT_EQ(tok.decode(ids, true), c["decoded_skip"].get<std::string>()) << name;
        const std::string expect_rt = c.contains("replaced") ? c["replaced"].get<std::string>() : in;
        EXPECT_EQ(tok.decode(tok.encode(in, true), false), halo::tokenizer::unicode::nfc_utf8(expect_rt)) << name;
    }
}

TEST(TokenizerGolden, SeededFuzzMatchesHf) {
    REQUIRE_GOLDEN();
    ASSERT_EQ(G.fuzz.size(), 3000u);
    int failures = 0;
    for (const auto& c : G.fuzz) {
        const std::string in = c["text"].get<std::string>();
        if (tok.encode(in, true) != ids_of(c["ids"]) || tok.encode(in, false) != ids_of(c["ids_no_special"])) {
            if (++failures <= 5) ADD_FAILURE() << c["name"].get<std::string>() << ": " << Json(in).dump();
        }
    }
    EXPECT_EQ(failures, 0);
}

TEST(TokenizerGolden, CodepointSweepMatchesHf) {
    REQUIRE_GOLDEN();
    int failures = 0;
    for (const auto& c : G.sweep) {
        const std::string text = c["text"].get<std::string>();
        // Pre-tokenizer pieces (codepoint lengths after NFC) must match HF's Split exactly.
        ASSERT_TRUE(c.contains("piece_lens")) << "stale golden: regenerate with make_tokenizer_golden.py golden";
        std::vector<char32_t> cps;
        const std::string norm = halo::tokenizer::unicode::nfc_utf8(text);
        for (std::size_t i = 0; i < norm.size();) {
            const auto st = halo::tokenizer::unicode::decode_step(norm, i);
            cps.push_back(st.cp);
            i += st.len;
        }
        std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
        halo::tokenizer::split_qwen35(cps, ranges);
        std::vector<std::size_t> lens;
        for (auto [a, b] : ranges) lens.push_back(b - a);
        if (lens != c["piece_lens"].get<std::vector<std::size_t>>()) {
            if (++failures <= 5) {
                ADD_FAILURE() << "sweep pieces U+" << std::hex << c["first"].get<int>() << "..U+" << c["last"].get<int>();
            }
        }
        if (tok.encode(text, true) != ids_of(c["ids"])) {
            if (++failures <= 5) {
                ADD_FAILURE() << "sweep chunk U+" << std::hex << c["first"].get<int>() << "..U+" << c["last"].get<int>();
            }
        }
    }
    EXPECT_EQ(failures, 0);
}

TEST(TokenizerGolden, NfcSweepMatchesHf) {
    REQUIRE_GOLDEN();
    int failures = 0;
    for (const auto& c : G.nfc) {
        const std::string in = c["text"].get<std::string>();
        const bool ok = tok.encode(in, true) == ids_of(c["ids"]) &&
                        halo::tokenizer::unicode::nfc_utf8(in) == c["nfc"].get<std::string>();
        if (!ok && ++failures <= 5) ADD_FAILURE() << "nfc chunk U+" << std::hex << c["first"].get<int>();
    }
    EXPECT_EQ(failures, 0);
}

TEST(TokenizerGolden, DecodeAndStreamDecodeMatchHf) {
    REQUIRE_GOLDEN();
    int failures = 0;
    for (const auto& c : G.decode) {
        const auto ids = ids_of(c["ids"]);
        const std::string expect = c["decoded"].get<std::string>();
        std::string streamed;
        StreamDecoder d(tok);
        for (auto id : ids) streamed += d.push(id);
        streamed += d.flush();
        std::string streamed_skip;
        StreamDecoder ds(tok, true);
        for (auto id : ids) streamed_skip += ds.push(id);
        streamed_skip += ds.flush();
        const bool ok = tok.decode(ids, false) == expect && streamed == expect &&
                        tok.decode(ids, true) == c["decoded_skip"].get<std::string>() &&
                        streamed_skip == c["decoded_skip"].get<std::string>();
        if (!ok && ++failures <= 5) ADD_FAILURE() << "decode case " << c["ids"].dump();
    }
    EXPECT_EQ(failures, 0);
}

TEST(TokenizerGolden, OffsetsPointAtTokenSources) {
    REQUIRE_GOLDEN();
    for (const auto& c : G.cases) {
        const std::string in = case_input(c);
        const Encoding e = tok.encode_with_offsets(in, true);
        ASSERT_EQ(e.ids, ids_of(c["ids"])) << c["name"];
        ASSERT_EQ(e.offsets.size(), e.ids.size());
        for (std::size_t i = 0; i < e.ids.size(); ++i) {
            ASSERT_LE(e.offsets[i], in.size());
            if (i > 0) ASSERT_LE(e.offsets[i - 1], e.offsets[i]) << c["name"];
            if (tok.token_type(e.ids[i]) != TokenType::Normal) {
                // Added tokens are matched on raw text: exact.
                EXPECT_EQ(in.compare(e.offsets[i], tok.token_to_piece(e.ids[i]).size(), tok.token_to_piece(e.ids[i])), 0)
                    << c["name"] << " token " << i;
            }
        }
        // For valid NFC-stable text, every token's bytes sit exactly at its offset.
        if (!c.contains("bytes_hex") && c["nfc"].get<std::string>() == in) {
            for (std::size_t i = 0; i < e.ids.size(); ++i) {
                const std::string& p = tok.token_to_piece(e.ids[i]);
                EXPECT_EQ(in.compare(e.offsets[i], p.size(), p), 0) << c["name"] << " token " << i;
            }
        }
    }
}

TEST(TokenizerGolden, LongPromptMatchesAndReportsThroughput) {
    REQUIRE_GOLDEN();
    const std::string text = G.long_text["text"].get<std::string>();
    const auto expect = ids_of(G.long_text["ids"]);
    ASSERT_GE(expect.size(), 32768u);
    std::vector<std::int32_t> ids;
    double best_ms = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        ids = tok.encode(text, true);
        const auto t1 = std::chrono::steady_clock::now();
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    EXPECT_EQ(ids, expect);
    // Cache-hostile second number: the codepoint sweep text (every BMP + plane-1 scalar),
    // where almost no word repeats.
    std::string sweep_text;
    for (const auto& c : G.sweep) sweep_text += c["text"].get<std::string>();
    const auto s0 = std::chrono::steady_clock::now();
    const auto sweep_ids = tok.encode(sweep_text, true);
    const double sweep_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
    std::printf("[ perf ] sweep text %zu bytes -> %zu tokens: %.2f ms (%.0f tokens/s)\n", sweep_text.size(),
                sweep_ids.size(), sweep_ms, static_cast<double>(sweep_ids.size()) / (sweep_ms / 1000.0));
    std::printf("[ perf ] encode %zu bytes -> %zu tokens: best of 5 = %.2f ms (%.0f tokens/s)\n", text.size(),
                ids.size(), best_ms, static_cast<double>(ids.size()) / (best_ms / 1000.0));
#if defined(NDEBUG) && !HALO_TEST_ASAN
    EXPECT_LT(best_ms, 100.0);  // loose dev-host bound for optimized builds; the number above is the evidence
#endif
}

// The GGUF path: build from the unsloth file's raw tokenizer arrays (gguf-py dump) and
// compare with the HF-json tokenizer.
TEST(TokenizerGolden, GgufVocabularyMatchesHfJson) {
    REQUIRE_GOLDEN();
    const fs::path p = kGolden / "gguf_unsloth_vocab.json";
    if (!fs::exists(p)) GTEST_SKIP() << "missing " << p;
    const Json g = Json::parse(read_file(p));
    ASSERT_EQ(g["pre"], "qwen35");
    ASSERT_EQ(g["model"], "gpt2");
    VocabSpec spec;
    spec.tokens = g["tokens"].get<std::vector<std::string>>();
    spec.merges = g["merges"].get<std::vector<std::string>>();
    for (int t : g["token_type"].get<std::vector<int>>()) spec.types.push_back(static_cast<TokenType>(t));
    spec.bos = g["bos"].get<std::int32_t>();
    spec.eos = g["eos"].get<std::int32_t>();
    spec.pad = g["pad"].get<std::int32_t>();
    const Tokenizer gg = Tokenizer::from_spec(std::move(spec));
    EXPECT_EQ(gg.vocab_size(), 248320u);
    EXPECT_EQ(gg.bos(), 248044);
    EXPECT_EQ(gg.eos(), 248046);
    EXPECT_EQ(gg.pad(), 248055);
    EXPECT_EQ(gg.token_type(248077), TokenType::Unused);
    // Known, asserted divergence: the GGUF marks <|fim_*|>, <|repo_name|>, <|file_sep|>
    // CONTROL while tokenizer.json has them special=false.
    const std::vector<std::int32_t> gguf_only_control = {248060, 248061, 248062, 248063, 248064, 248065};
    for (std::int32_t id = 248044; id < 248077; ++id) {
        const bool diverges = std::find(gguf_only_control.begin(), gguf_only_control.end(), id) != gguf_only_control.end();
        EXPECT_EQ(gg.is_special(id), diverges || tok.is_special(id)) << id;
        EXPECT_EQ(gg.token_to_piece(id), tok.token_to_piece(id)) << id;
    }
    for (const auto& c : G.cases) {
        const std::string in = case_input(c);
        EXPECT_EQ(gg.encode(in, true), ids_of(c["ids"])) << c["name"];
        const auto ids = ids_of(c["ids"]);
        EXPECT_EQ(gg.decode(ids, false), c["decoded"].get<std::string>()) << c["name"];
    }
    // parse_special=false differs exactly on the fim-type tokens.
    const std::string fim = "<|fim_prefix|>x<think>";
    EXPECT_EQ(tok.encode(fim, false).front(), 248060);
    EXPECT_NE(gg.encode(fim, false).front(), 248060);
    EXPECT_EQ(gg.encode(fim, false).back(), 248068);
    for (const auto& c : G.fuzz) {
        EXPECT_EQ(gg.encode(c["text"].get<std::string>(), true), ids_of(c["ids"])) << c["name"];
    }
}
