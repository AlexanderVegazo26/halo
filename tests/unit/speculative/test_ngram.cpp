// next.md step 11: prompt-lookup (n-gram) drafts. Pure function over token ids; the engine
// wiring and the on-device contract (greedy output unchanged, drafts accepted) are covered by
// VulkanEngine.Ngram* in tests/unit/runtime/test_engine.cpp.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "halo/speculative/ngram.h"

using halo::speculative::ngram_draft;
using Toks = std::vector<std::int32_t>;

namespace {

Toks draft(const Toks& prompt, const Toks& generated, std::size_t k, std::size_t min_match = 3,
           std::size_t max_match = 8) {
    return ngram_draft(prompt, generated, k, min_match, max_match);
}

}  // namespace

TEST(NgramDraft, NoRecurringSuffixDraftsNothing) {
    EXPECT_EQ(draft({1, 2, 3, 4, 5, 6}, {7, 8}, 4), Toks{}) << "no 3-token suffix of 1..8 recurs";
    // Only a 2-token suffix recurs, below min_match = 3.
    EXPECT_EQ(draft({1, 2, 9, 4, 1}, {2}, 4), Toks{}) << "suffix (1, 2) recurs but is shorter than min_match";
    EXPECT_EQ(draft({1, 2, 9, 4, 1}, {2}, 4, 2), (Toks{9, 4, 1, 2})) << "with min_match = 2 the same context drafts";
}

TEST(NgramDraft, ProposesWhatFollowedTheLatestOccurrenceCappedAtK) {
    // suffix (11, 12, 13) occurs at 1 (followed by 14) and at 6 (followed by 15 16 30 ...).
    const Toks p = {10, 11, 12, 13, 14, 20, 11, 12, 13, 15, 16, 30};
    EXPECT_EQ(draft(p, {11, 12, 13}, 2), (Toks{15, 16})) << "latest occurrence wins, k caps the draft";
    EXPECT_EQ(draft(p, {11, 12, 13}, 1), (Toks{15}));
    EXPECT_EQ(draft(p, {11, 12, 13}, 0), Toks{}) << "k == 0 drafts nothing";
}

TEST(NgramDraft, LongestRecurringSuffixWinsOverALaterShorterMatch) {
    // m = 4 (5 11 12 13) recurs at 0 -> 40 41; m = 3 (11 12 13) recurs later at 7 -> 50 51.
    const Toks p = {5, 11, 12, 13, 40, 41, 7, 11, 12, 13, 50, 51};
    EXPECT_EQ(draft(p, {5, 11, 12, 13}, 2), (Toks{40, 41}));
    EXPECT_EQ(draft(p, {5, 11, 12, 13}, 2, 3, 3), (Toks{50, 51})) << "max_match = 3 only sees the shorter suffix";
}

TEST(NgramDraft, MatchesAndDraftsAcrossThePromptGeneratedBoundary) {
    // The recurring suffix is in `generated`, its earlier occurrence and continuation in the prompt,
    // and the draft runs on into `generated` up to the end of the context (never past it).
    EXPECT_EQ(draft({1, 2, 3, 4, 5, 6}, {9, 1, 2, 3}, 8), (Toks{4, 5, 6, 9, 1, 2, 3}));
    // The suffix itself straddles the boundary: (7, 8 | 9) recurs at 0.
    EXPECT_EQ(draft({7, 8, 9, 60, 61, 7, 8}, {9}, 3), (Toks{60, 61, 7}));
    // Only the generated part holds the context.
    EXPECT_EQ(draft({}, {4, 5, 6, 70, 4, 5, 6}, 2), (Toks{70, 4}));
}

TEST(NgramDraft, PeriodicRunProposesUpToTheEndOfTheContext) {
    // A match may overlap the suffix: (3 3 3) recurs at 0 inside 3 3 3 3, and only one token
    // follows it before the end of the context.
    EXPECT_EQ(draft({3, 3, 3}, {3}, 4), (Toks{3}));
    // Period 2: the longest recurring suffix (8 tokens) starts 2 earlier and proposes one period.
    EXPECT_EQ(draft({1, 2, 1, 2, 1, 2, 1, 2}, {1, 2}, 6), (Toks{1, 2}));
}

TEST(NgramDraft, ShortContextsAndDegenerateArgumentsDraftNothing) {
    EXPECT_EQ(draft({}, {}, 4), Toks{});
    EXPECT_EQ(draft({3, 3}, {3}, 4), Toks{}) << "3 tokens cannot hold a recurring 3-token suffix";
    EXPECT_EQ(draft({3, 3, 3}, {3}, 4, 0), Toks{}) << "min_match = 0 is rejected, not an endless loop";
    EXPECT_EQ(draft({1, 2, 3, 9, 1, 2}, {3}, 4, 3, 2), Toks{}) << "max_match < min_match tries no length";
}
