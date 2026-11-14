#pragma once
// Prompt-lookup (n-gram) drafts for speculative decoding (next.md step 11, opt-in:
// CpuEngineOptions::ngram_draft / HALO_NGRAM_DRAFT=1). Agent and code-editing output repeats
// long spans of its context; a draft copied from the context costs no model pass, and the
// trunk still verifies every drafted token, so greedy output is unchanged by construction.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace halo::speculative {

/// Finds the most recent earlier occurrence of the last m tokens of (prompt ++ generated),
/// trying m from `max_match` down to `min_match` (the longest suffix that recurs wins; among
/// equal m the latest start wins), and returns the up-to-`k` tokens that followed it. A match
/// may overlap the suffix itself (a periodic run then proposes its period). Empty when k == 0,
/// min_match == 0, the context is shorter than min_match + 1, or no suffix recurs: the caller
/// drafts with MTP instead. The last token of `generated` is the pending (not yet fed) token,
/// so the result is the draft that would follow it.
[[nodiscard]] std::vector<std::int32_t> ngram_draft(std::span<const std::int32_t> prompt,
                                                    std::span<const std::int32_t> generated, std::size_t k,
                                                    std::size_t min_match = 3, std::size_t max_match = 8);

}  // namespace halo::speculative
