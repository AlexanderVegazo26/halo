// Prompt-lookup (n-gram) drafts (next.md step 11).

#include "halo/speculative/ngram.h"

#include <algorithm>

namespace halo::speculative {

std::vector<std::int32_t> ngram_draft(std::span<const std::int32_t> prompt, std::span<const std::int32_t> generated,
                                      std::size_t k, std::size_t min_match, std::size_t max_match) {
    const std::size_t n = prompt.size() + generated.size();
    if (k == 0 || min_match == 0 || n < min_match + 1) return {};
    const auto at = [&](std::size_t i) { return i < prompt.size() ? prompt[i] : generated[i - prompt.size()]; };
    for (std::size_t m = std::min(max_match, n - 1); m >= min_match; --m) {
        // Scan back for the latest start p < n - m whose m tokens equal the last m tokens.
        for (std::size_t p = n - m; p-- > 0;) {
            std::size_t j = 0;
            while (j < m && at(p + j) == at(n - m + j)) ++j;
            if (j < m) continue;
            std::vector<std::int32_t> out;
            for (std::size_t i = p + m; i < n && out.size() < k; ++i) out.push_back(at(i));
            return out;  // p + m < n always: p < n - m
        }
    }
    return {};
}

}  // namespace halo::speculative
