#include "tokenizer/pretokenize.h"

#include "tokenizer/unicode.h"

namespace halo::tokenizer {
namespace {

using unicode::CpClass;

bool is_crlf(char32_t c) noexcept { return c == U'\r' || c == U'\n'; }

// Length of the match starting at `i` (always >= 1).
std::size_t match_at(std::span<const char32_t> s, std::span<const CpClass> cls, std::size_t i) {
    const std::size_t n = s.size();
    const char32_t c = s[i];
    const CpClass k = cls[i];

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c == U'\'' && i + 1 < n) {
        const char32_t c1 = s[i + 1];
        if (unicode::fold_single(c1)) return 2;
        if (i + 2 < n) {
            const char32_t c2 = s[i + 2];
            if ((unicode::fold_re_ve_first(c1) && unicode::fold_e_second(c2)) ||
                (unicode::fold_l_first(c1) && unicode::fold_l_second(c2))) {
                return 3;
            }
        }
    }
    const auto run = [&](std::size_t from, CpClass want) {
        std::size_t j = from;
        while (j < n && cls[j] == want) ++j;
        return j;
    };
    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ : a mark as the optional prefix gives the same
    //    match as the mark starting the run, so only non-letter prefixes are special.
    if (k == CpClass::LetterMark) return run(i, CpClass::LetterMark) - i;
    if (!is_crlf(c) && k != CpClass::Number && i + 1 < n && cls[i + 1] == CpClass::LetterMark) {
        return run(i + 1, CpClass::LetterMark) - i;
    }
    // 3. \p{N}
    if (k == CpClass::Number) return 1;
    // 4. " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    std::size_t j = i;
    if (c == U' ' && i + 1 < n && cls[i + 1] == CpClass::Other) j = i + 1;
    if (cls[j] == CpClass::Other) {
        j = run(j, CpClass::Other);
        while (j < n && is_crlf(s[j])) ++j;
        return j - i;
    }
    // Only whitespace remains (k == Space).
    const std::size_t end = run(i, CpClass::Space);
    // 5. \s*[\r\n]+ : \s* backtracks to the last CR/LF of the run.
    for (std::size_t m = end; m > i; --m) {
        if (is_crlf(s[m - 1])) return m - i;
    }
    // 6. \s+(?!\S)
    if (end == n) return end - i;
    if (end - i >= 2) return end - 1 - i;
    // 7. \s+
    return end - i;
}

}  // namespace

void split_qwen35(std::span<const char32_t> cps, std::vector<std::pair<std::uint32_t, std::uint32_t>>& out) {
    std::vector<CpClass> cls(cps.size());
    for (std::size_t i = 0; i < cps.size(); ++i) cls[i] = unicode::classify(cps[i]);
    std::size_t i = 0;
    while (i < cps.size()) {
        const std::size_t len = match_at(cps, cls, i);
        out.emplace_back(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(i + len));
        i += len;
    }
}

}  // namespace halo::tokenizer
