// Hostile-vocabulary tests (security review S-6, S-7; docs/security-hardening.md).
//
// Timing thresholds are dev-host smoke checks (D-001), not performance claims; under ASan
// only the measured times are printed.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "halo/core/error.h"
#include "halo/tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HALO_LIMITS_TEST_ASAN 1
#endif
#endif

using halo::ErrorCode;
using halo::tokenizer::Tokenizer;
using halo::tokenizer::TokenizerLimits;
using halo::tokenizer::TokenType;
using halo::tokenizer::VocabSpec;
using Json = nlohmann::json;

namespace {

#if defined(HALO_LIMITS_TEST_ASAN)
constexpr bool kCheckTiming = false;
#else
constexpr bool kCheckTiming = true;
#endif

// GPT-2 bytes_to_unicode (independent of the implementation).
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

VocabSpec byte_spec() {
    VocabSpec s;
    for (int b = 0; b < 256; ++b) s.tokens.push_back(mapped(static_cast<unsigned char>(b)));
    s.types.assign(256, TokenType::Normal);
    return s;
}

void add(VocabSpec& s, std::string content, TokenType t) {
    s.tokens.push_back(std::move(content));
    s.types.push_back(t);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

ErrorCode code_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const halo::Error& e) {
        return e.code();
    }
    return ErrorCode::Cancelled;  // "did not throw"
}

// The S-6 review vocabulary: 80,000 Control tokens that all start with '<'.
VocabSpec s6_vocabulary() {
    VocabSpec s = byte_spec();
    for (int i = 0; i < 80000; ++i) add(s, "<|tok_" + std::to_string(i) + "|>", TokenType::Control);
    s.limits.max_added_tokens = 0;  // far beyond the default cap on purpose
    return s;
}

}  // namespace

// ---- S-6 ---------------------------------------------------------------------------------

TEST(TokenizerLimits, EightyThousandAddedTokensBuildAndEncodeQuickly) {
    auto t0 = std::chrono::steady_clock::now();
    const Tokenizer tok = Tokenizer::from_spec(s6_vocabulary());
    const double build = seconds_since(t0);
    const std::string text(20000, '<');
    t0 = std::chrono::steady_clock::now();
    const auto ids = tok.encode(text, true);
    const double enc = seconds_since(t0);
    std::printf("80000 added tokens: build %.3f s, encode 20 KB of '<' %.4f s\n", build, enc);
    EXPECT_EQ(ids.size(), 20000u);  // no added token matches; every '<' is its byte token
    // One real match in the middle is still found.
    const auto hit = tok.encode(std::string(100, '<') + "<|tok_79999|>" + std::string(100, '<'), true);
    EXPECT_EQ(std::count(hit.begin(), hit.end(), 256 + 79999), 1);
    if (kCheckTiming) {  // review: 9.46 s to build, 6.69 s to encode
        EXPECT_LT(build, 1.0);
        EXPECT_LT(enc, 0.1);
    }
}

TEST(TokenizerLimits, SharedPrefixVocabularyEncodesInBoundedTime) {
    // Worst case for a trie walk inside the default limits: 4096 tokens of up to 256 bytes,
    // all '<'-prefixes of the input that never complete, so every position walks ~255 deep.
    VocabSpec s = byte_spec();
    for (int i = 0; i < 4096; ++i) {
        const std::size_t k = 1 + static_cast<std::size_t>(i) % 250;
        add(s, std::string(k, '<') + "#" + std::to_string(i), TokenType::Control);
    }
    const Tokenizer tok = Tokenizer::from_spec(std::move(s));
    const std::string text(100000, '<');
    const auto t0 = std::chrono::steady_clock::now();
    const auto ids = tok.encode(text, true);
    const double enc = seconds_since(t0);
    std::printf("4096 shared-prefix tokens: encode 100 KB of '<' %.3f s\n", enc);
    EXPECT_EQ(ids.size(), text.size());
    if (kCheckTiming) EXPECT_LT(enc, 1.0);
}

TEST(TokenizerLimits, AddedTokenCountAndLengthAreCapped) {
    const TokenizerLimits d;
    EXPECT_EQ(d.max_added_tokens, 4096u);
    EXPECT_EQ(d.max_added_token_bytes, 256u);
    const auto with_added = [](std::size_t count, std::size_t bytes) {
        VocabSpec s = byte_spec();
        for (std::size_t i = 0; i < count; ++i) {
            std::string c = "<" + std::to_string(i) + ">";
            c.resize(std::max(c.size(), bytes), 'x');
            add(s, c, i % 2 == 0 ? TokenType::Control : TokenType::UserDefined);
        }
        return s;
    };
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_spec(with_added(4096, 0)); }), ErrorCode::Cancelled);
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_spec(with_added(4097, 0)); }), ErrorCode::Model);
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_spec(with_added(1, 256)); }), ErrorCode::Cancelled);
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_spec(with_added(1, 257)); }), ErrorCode::Model);
    VocabSpec raised = with_added(5000, 300);
    raised.limits = {6000, 300};
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_spec(std::move(raised)); }), ErrorCode::Cancelled);
}

TEST(TokenizerLimits, DuplicateAddedTokensStillRejected) {
    VocabSpec s = byte_spec();
    add(s, "<a>", TokenType::Control);
    add(s, "<b>", TokenType::UserDefined);
    add(s, "<a>", TokenType::UserDefined);
    try {
        (void)Tokenizer::from_spec(std::move(s));
        FAIL() << "duplicate accepted";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Model);
        EXPECT_NE(std::string(e.what()).find("'<a>' appears twice (ids 256 and 258)"), std::string::npos) << e.what();
    }
    // A prefix of another added token is not a duplicate.
    VocabSpec p = byte_spec();
    add(p, "<a>", TokenType::Control);
    add(p, "<a", TokenType::UserDefined);
    add(p, "<", TokenType::UserDefined);
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_spec(std::move(p)); }), ErrorCode::Cancelled);
}

namespace {

// The matching rule, written naively: at the leftmost position where any allowed added
// token matches, take the longest; text between matches is encoded on its own.
std::vector<std::int32_t> reference_encode(const Tokenizer& tok, const std::vector<std::pair<std::string, std::int32_t>>& added,
                                           const std::vector<bool>& special, std::string_view text, bool parse_special) {
    std::vector<std::int32_t> out;
    std::size_t seg = 0;
    std::size_t p = 0;
    while (p < text.size()) {
        std::size_t best = SIZE_MAX;
        for (std::size_t i = 0; i < added.size(); ++i) {
            if (special[i] && !parse_special) continue;
            const std::string& c = added[i].first;
            if (text.substr(p, c.size()) == c && (best == SIZE_MAX || c.size() > added[best].first.size())) best = i;
        }
        if (best == SIZE_MAX) {
            ++p;
            continue;
        }
        const auto part = tok.encode(text.substr(seg, p - seg), parse_special);
        out.insert(out.end(), part.begin(), part.end());
        out.push_back(added[best].second);
        p += added[best].first.size();
        seg = p;
    }
    const auto tail = tok.encode(text.substr(seg), parse_special);
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

}  // namespace

TEST(TokenizerLimits, TrieMatchingEqualsLeftmostLongestReference) {
    // Random vocabularies over a small alphabet so added tokens overlap, nest and share
    // prefixes, with Control and UserDefined mixed (parse_special=false must fall back to
    // a shorter UserDefined token under a longer Control one).
    std::mt19937 rng(20260925);
    const std::string alphabet = "<>|ab";
    std::size_t matched = 0;
    for (int round = 0; round < 60; ++round) {
        VocabSpec s = byte_spec();
        std::vector<std::pair<std::string, std::int32_t>> added;
        std::vector<bool> special;
        const int n_added = 1 + static_cast<int>(rng() % 24);
        while (static_cast<int>(added.size()) < n_added) {
            std::string c;
            const std::size_t len = 1 + rng() % 6;
            for (std::size_t k = 0; k < len; ++k) c.push_back(alphabet[rng() % alphabet.size()]);
            if (std::any_of(added.begin(), added.end(), [&](const auto& a) { return a.first == c; })) continue;
            const bool sp = rng() % 2 == 0;
            added.emplace_back(c, static_cast<std::int32_t>(s.tokens.size()));
            special.push_back(sp);
            add(s, c, sp ? TokenType::Control : TokenType::UserDefined);
        }
        const Tokenizer tok = Tokenizer::from_spec(std::move(s));
        for (int t = 0; t < 40; ++t) {
            std::string text;
            const std::size_t len = rng() % 64;
            for (std::size_t k = 0; k < len; ++k) text.push_back((alphabet + " x")[rng() % (alphabet.size() + 2)]);
            for (const bool ps : {true, false}) {
                const auto want = reference_encode(tok, added, special, text, ps);
                ASSERT_EQ(tok.encode(text, ps), want) << "round " << round << " text '" << text << "' parse_special " << ps;
                matched += static_cast<std::size_t>(std::count_if(want.begin(), want.end(), [](std::int32_t id) { return id >= 256; }));
            }
        }
    }
    EXPECT_GT(matched, 1000u);  // the fuzz really exercises matching
}

// ---- S-7 ---------------------------------------------------------------------------------

namespace {

std::string s7_file() {
    // The review's PoC shape: a complete, supported tokenizer.json whose vocabulary is the
    // single entry {"a": 16777215}.
    const std::string regex =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    const Json j{
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", Json::array()},
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
          {"vocab", {{"a", 16777215}}}, {"merges", Json::array()}}}};
    return j.dump();
}

// Peak resident set (VmHWM) in KiB.
long vm_hwm_kib() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::strtol(line.c_str() + 6, nullptr, 10);
    }
    return -1;
}

// Child: resets the peak-RSS counter, parses the file, and exits 0 only if it was rejected
// with Model while the peak grew by less than 64 MiB. 1: accepted, 2: wrong error,
// 3: large allocation, 4: cannot measure.
[[noreturn]] void s7_child(const std::string& file) {
    {
        std::ofstream clear("/proc/self/clear_refs");
        clear << "5";  // reset VmHWM to the current RSS
    }
    const long before = vm_hwm_kib();
    if (before < 0) std::_Exit(4);
    int rc = 1;
    try {
        (void)Tokenizer::from_hf_json(file);
    } catch (const halo::Error& e) {
        rc = e.code() == ErrorCode::Model ? 0 : 2;
        std::fprintf(stderr, "rejected: %s\n", e.what());
    } catch (const std::bad_alloc&) {
        rc = 3;
    }
    const long grew = vm_hwm_kib() - before;
    std::fprintf(stderr, "peak RSS grew by %ld KiB\n", grew);
    if (rc == 0 && grew > 64 * 1024) rc = 3;
    std::_Exit(rc);
}

}  // namespace

TEST(TokenizerLimits, TinyFileNamingAHugeIdIsRejectedWithoutAllocating) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string file = s7_file();
    std::printf("S-7 file: %zu bytes\n", file.size());
    EXPECT_LT(file.size(), 1024u);
    EXPECT_EXIT(s7_child(file), ::testing::ExitedWithCode(0), "");
    try {
        (void)Tokenizer::from_hf_json(file);
    } catch (const halo::Error& e) {
        EXPECT_NE(std::string(e.what()).find("not below the entry count"), std::string::npos) << e.what();
    }
}

TEST(TokenizerLimits, DenseIdsWithAddedTokenOverlapStillLoad) {
    // An added token that repeats a vocab entry's id (allowed when the content matches)
    // leaves the largest id below the entry count.
    Json vocab = Json::object();
    for (int b = 0; b < 256; ++b) vocab[mapped(static_cast<unsigned char>(b))] = b;
    vocab["aa"] = 256;
    Json j = Json::parse(s7_file());
    j["model"]["vocab"] = vocab;
    j["added_tokens"] = Json::array({{{"id", 257}, {"content", "<|end|>"}, {"special", true}},
                                     {{"id", 256}, {"content", "aa"}, {"special", false}}});
    const Tokenizer t = Tokenizer::from_hf_json(j.dump());
    EXPECT_EQ(t.vocab_size(), 258u);  // 257 vocab + 2 added entries, largest id 257
    // A hole (id 300 with only 259 entries) is rejected.
    j["added_tokens"][0]["id"] = 300;
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_hf_json(j.dump()); }), ErrorCode::Model);
    // Limits reach the HF path too.
    j["added_tokens"][0]["id"] = 257;
    EXPECT_EQ(code_of([&] { (void)Tokenizer::from_hf_json(j.dump(), {}, TokenizerLimits{1, 256}); }),
              ErrorCode::Model);  // two added tokens, limit 1
}
