#pragma once
// qwen35 pre-tokenizer: a hand-written matcher with the exact semantics (leftmost-first
// alternation, greedy quantifiers with backtracking) of the HF Split regex
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// with behavior "Isolated". Every codepoint matches some alternative, so the pieces tile
// the input.

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace halo::tokenizer {

/// Appends [begin, end) codepoint-index ranges of the pieces of `cps` to `out`.
void split_qwen35(std::span<const char32_t> cps, std::vector<std::pair<std::uint32_t, std::uint32_t>>& out);

}  // namespace halo::tokenizer
