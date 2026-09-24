// Unit tests for the tokenizer's internal Unicode layer and pre-tokenizer (no reference
// data needed). Expected splits were produced by HF tokenizers 0.23.2's Split regex and
// UTF-8 replacements by Python 3.12 `bytes.decode('utf-8', 'replace')`.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace u = halo::tokenizer::unicode;

namespace {

std::vector<char32_t> to_cps(std::string_view s) {
    std::vector<char32_t> out;
    for (std::size_t i = 0; i < s.size();) {
        const auto st = u::decode_step(s, i);
        out.push_back(st.cp);
        i += st.len;
    }
    return out;
}

std::vector<std::string> split(std::string_view s) {
    const auto cps = to_cps(s);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    halo::tokenizer::split_qwen35(cps, ranges);
    std::vector<std::string> out;
    for (auto [a, b] : ranges) {
        std::string piece;
        for (auto k = a; k < b; ++k) u::append_utf8(piece, cps[k]);
        out.push_back(piece);
    }
    return out;
}

}  // namespace

TEST(Utf8, ReplacesMaximalSubparts) {
    // Python: b'\xed\xa0\x80\xe2\x82x\xf0\x9f\x98\xc0\xff'.decode('utf-8','replace')
    //   == '����x���'
    const std::string in = "\xed\xa0\x80\xe2\x82x\xf0\x9f\x98\xc0\xff";
    const std::string r = "\xef\xbf\xbd";
    EXPECT_EQ(u::to_valid_utf8(in), r + r + r + r + "x" + r + r + r);
    EXPECT_FALSE(u::is_valid_utf8(in));
    EXPECT_TRUE(u::is_valid_utf8("h\xc3\xa9llo \xf0\x9f\x98\x80"));
    EXPECT_EQ(u::to_valid_utf8("end\xf0\x9f\x98"), "end" + r);            // incomplete tail
    EXPECT_EQ(u::to_valid_utf8("o\xc0\xafp"), "o" + r + r + "p");          // overlong
    EXPECT_EQ(u::to_valid_utf8("m\xf4\x90\x80\x80n"), "m" + r + r + r + r + "n");  // > U+10FFFF
}

TEST(Utf8, DecodeStepStatuses) {
    using St = u::Utf8Step::Status;
    EXPECT_EQ(u::decode_step("\xe2\x82", 0).status, St::Incomplete);
    EXPECT_EQ(u::decode_step("\xe2\x82", 0).len, 2u);
    EXPECT_EQ(u::decode_step("\xe2\x28", 0).status, St::Invalid);
    EXPECT_EQ(u::decode_step("\xe2\x28", 0).len, 1u);
    const auto ok = u::decode_step("\xe2\x82\xac", 0);
    EXPECT_EQ(ok.status, St::Ok);
    EXPECT_EQ(ok.cp, U'€');
}

TEST(Nfc, ComposesReordersAndHandlesHangul) {
    EXPECT_EQ(u::nfc_utf8("cafe\xcc\x81"), "caf\xc3\xa9");                    // e + U+0301
    EXPECT_EQ(u::nfc_utf8("a\xcc\x81\xcc\xa3"), "\xe1\xba\xa1\xcc\x81");     // reorder, then a+0323
    EXPECT_EQ(u::nfc_utf8("\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8"), "\xea\xb0\x81");  // jamo -> U+AC01
    EXPECT_EQ(u::nfc_utf8("\xe2\x84\xab"), "\xc3\x85");                       // U+212B -> U+00C5
    // HF's NFC data predates Unicode 13: U+11935 U+11930 stays decomposed.
    EXPECT_EQ(u::nfc_utf8("\xf0\x91\xa4\xb5\xf0\x91\xa4\xb0"), "\xf0\x91\xa4\xb5\xf0\x91\xa4\xb0");
    EXPECT_TRUE(u::nfc_quick_check("plain ascii \xc3\xa9 \xe4\xb8\xad"));
    EXPECT_FALSE(u::nfc_quick_check("e\xcc\x81"));
}

TEST(Nfc, OffsetsStayMonotonic) {
    // "xa" + U+0301 + U+0323 + "b": the mark run is reordered and a+0323 composes.
    std::vector<u::CpOff> cps = {{U'x', 0}, {U'a', 1}, {0x301, 2}, {0x323, 4}, {U'b', 6}};
    u::nfc(cps);
    ASSERT_EQ(cps.size(), 4u);
    EXPECT_EQ(cps[1].cp, 0x1EA1u);
    EXPECT_EQ(cps[1].off, 1u);
    EXPECT_EQ(cps[2].cp, 0x301u);
    EXPECT_EQ(cps[2].off, 2u);  // min offset of the reordered run
    EXPECT_EQ(cps[3].off, 6u);
}

TEST(Classes, MatchHfRegexClasses) {
    EXPECT_EQ(u::classify(U'a'), u::CpClass::LetterMark);
    EXPECT_EQ(u::classify(0x301), u::CpClass::LetterMark);
    EXPECT_EQ(u::classify(U'7'), u::CpClass::Number);
    EXPECT_EQ(u::classify(0xBD), u::CpClass::Number);     // ½ (No)
    EXPECT_EQ(u::classify(0x3000), u::CpClass::Space);
    EXPECT_EQ(u::classify(0x200B), u::CpClass::Other);    // ZWSP is not \s
    EXPECT_EQ(u::classify(0x0897), u::CpClass::LetterMark);  // Unicode 16 mark, known to HF
    EXPECT_TRUE(u::fold_single(0x17F));                    // ſ folds to s
    EXPECT_FALSE(u::fold_single(0x212A));                  // Kelvin sign folds to k: not a contraction
}

TEST(PreTokenizer, RegexSemantics) {
    using V = std::vector<std::string>;
    EXPECT_EQ(split("Hello world"), (V{"Hello", " world"}));
    EXPECT_EQ(split("  hello"), (V{" ", " hello"}));
    EXPECT_EQ(split("don't DON'T it'\xc5\xbf"), (V{"don", "'t", " DON", "'T", " it", "'\xc5\xbf"}));
    EXPECT_EQ(split("we're 'LLx"), (V{"we", "'re", " '", "LLx"}));
    EXPECT_EQ(split("'LLx"), (V{"'LL", "x"}));
    // U+017F (long s) case-folds to s under (?i): a contraction, unlike Kelvin sign K.
    EXPECT_EQ(split("'\xc5\xbf" "a"), (V{"'\xc5\xbf", "a"}));
    EXPECT_EQ(split("'\xe2\x84\xaa" "a"), (V{"'\xe2\x84\xaa" "a"}));
    EXPECT_EQ(split("12345"), (V{"1", "2", "3", "4", "5"}));
    EXPECT_EQ(split("a  \n  b"), (V{"a", "  \n", " ", " b"}));
    EXPECT_EQ(split("x !!!\n\ny"), (V{"x", " !!!\n\n", "y"}));
    EXPECT_EQ(split("end   "), (V{"end", "   "}));
    EXPECT_EQ(split("\t!"), (V{"\t", "!"}));
    EXPECT_EQ(split("$var"), (V{"$var"}));
    EXPECT_EQ(split("a\r\n\r\nb"), (V{"a", "\r\n\r\n", "b"}));
    EXPECT_TRUE(split("").empty());
}
