#pragma once
// Structured output (TRD §25, FR-013): JSON Schema -> byte-level grammar -> per-step
// token masks over the full vocabulary. Implementation: src/sampling (WS-H).
//
// Pipeline
//   Grammar::from_json_schema / Grammar::any_json_object compile to an immutable byte-level
//   context-free grammar. TokenMatcher walks a precomputed trie of the vocabulary's token
//   byte strings against the grammar's incremental matcher to produce the allowed-token
//   bitset for the current step (plus EOS iff the document is complete), and advances on
//   accept_token(). Matching is byte-level, so a token that ends inside a multi-byte UTF-8
//   character is allowed exactly when some valid continuation exists.
//
// Supported JSON Schema subset (anything else is a typed error, never silently ignored)
//   type (string or array of types): object, array, string, number, integer, boolean, null
//   properties, required, additionalProperties (true / false / schema)
//   enum, const, items (single schema), minItems, maxItems
//   minLength, maxLength (counted in Unicode codepoints), pattern (ECMA-262 subset below)
//   anyOf, oneOf (oneOf only when the branches are provably disjoint: disjoint JSON types,
//   disjoint enum/const literal sets, or an object discriminator — a required property whose
//   const/enum values differ; otherwise Unsupported), $ref to "#", "#/$defs/<name>" or
//   "#/definitions/<name>" (recursion allowed when it passes through an object or array),
//   $defs / definitions.
//   Annotations are ignored: title, description, default, examples, $schema, $id,
//   $comment, deprecated, readOnly, writeOnly.
//
// Generation policy (every output validates against the schema; these choices only
// narrow *which* valid documents can be produced — stated so nobody mistakes them for the
// schema's full language):
//   * Properties are emitted in schema order; optional ones may be skipped.
//   * additionalProperties absent or true: extra keys are allowed after the declared ones
//     (JSON Schema's default). Extra keys use unescaped characters only and can never
//     equal a declared name. additionalProperties: {schema} constrains their values.
//   * Strings use literal UTF-8 except where JSON requires an escape; generic strings also
//     accept every JSON escape (\uXXXX surrogate pairs count as one codepoint; lone
//     surrogates are never produced). Pattern-constrained strings produce only the
//     canonical escapes (\" \\ \b \f \n \r \t \u00XX).
//   * integer produces -?(0|[1-9][0-9]*); number adds optional fraction and exponent.
//   * Whitespace between tokens is optional and limited to `max_whitespace` bytes per gap.
//   * Container nesting is limited to `max_nesting` (a schema that *requires* deeper
//     nesting is rejected with Api; recursion is unrolled up to the limit).
//
// Pattern subset (ECMA-262, no flags, matched against decoded codepoints; unanchored
// patterns are searched, i.e. wrapped in .* on the unanchored side):
//   literals, ., [...] and [^...] with ranges, \d \D \w \W \s \S (ECMA definitions),
//   escapes \t \n \r \f \v \0 \xHH \uHHHH and escaped syntax characters, groups (...),
//   (?:...), (?<name>...), alternation |, quantifiers * + ? {n} {n,} {n,m} (lazy ? suffix
//   accepted; laziness does not change the matched language), ^ at the start and $ at the
//   end of a top-level alternative (`^a|b$` is (^a)|(b$), as in ECMA-262). Unsupported (Error Unsupported): backreferences, lookaround, \b \B, \p{..},
//   \u{...}, anchors elsewhere, and a pattern combined with minLength/maxLength.
//
// Errors: halo::Error with ErrorCode::Api for malformed or unsatisfiable schemas,
// ErrorCode::Unsupported for valid-but-unsupported keywords/features/limits.
//
// Thread safety: Grammar and TokenVocab are immutable and may be shared across threads.
// TokenMatcher holds per-sequence state and lazily grown caches: one per sequence, not
// thread-safe.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace halo::tokenizer { class Tokenizer; }

namespace halo::sampling {

namespace detail {
struct GrammarData;
class Runtime;
}  // namespace detail

struct SchemaOptions {
    std::size_t max_nesting = 32;            ///< object/array nesting levels
    std::size_t max_whitespace = 16;         ///< bytes of optional whitespace per gap
    std::size_t max_repeat = 10000;          ///< bound for min/max Items/Length and {n,m}
    std::size_t max_grammar_elements = std::size_t{1} << 22;
    std::size_t max_schema_depth = 128;      ///< nesting of the schema document itself
};

/// Immutable compiled grammar. Cheap to copy (shared).
class Grammar {
public:
    /// Compiles a JSON Schema document (JSON text). See the header comment for the subset.
    static Grammar from_json_schema(std::string_view schema_json, const SchemaOptions& opts = {});
    /// `json_object` mode: any JSON object.
    static Grammar any_json_object(const SchemaOptions& opts = {});
    /// Any JSON value (used for tests and schema `{}`).
    static Grammar any_json_value(const SchemaOptions& opts = {});

    [[nodiscard]] std::size_t num_rules() const noexcept;
    [[nodiscard]] std::size_t num_elements() const noexcept;
    [[nodiscard]] std::string debug_string() const;

    [[nodiscard]] const std::shared_ptr<const detail::GrammarData>& data() const noexcept { return data_; }
    explicit Grammar(std::shared_ptr<const detail::GrammarData> d) : data_(std::move(d)) {}

private:
    std::shared_ptr<const detail::GrammarData> data_;
};

/// Byte-level incremental matcher (the pushdown automaton), independent of any vocabulary.
class ByteMatcher {
public:
    explicit ByteMatcher(const Grammar& g);
    ByteMatcher(ByteMatcher&&) noexcept;
    ByteMatcher& operator=(ByteMatcher&&) noexcept;
    ~ByteMatcher();

    /// Consumes all of `bytes` or none: returns false (state unchanged) if some prefix
    /// leads to a dead state.
    bool accept_bytes(std::string_view bytes);
    /// The bytes consumed so far form a complete document.
    [[nodiscard]] bool is_accepting() const;
    /// At least one more byte can be consumed.
    [[nodiscard]] bool can_continue() const;
    void reset();

private:
    std::unique_ptr<detail::Runtime> rt_;
    std::int32_t state_;
};

/// Token byte strings of a vocabulary arranged as a preorder trie. Built once per
/// tokenizer and shared by every TokenMatcher. Only Normal tokens with non-empty pieces
/// are included: Control, UserDefined (e.g. <think>, <tool_call>) and Unused tokens are
/// never allowed inside structured output; EOS is handled separately.
class TokenVocab {
    struct PrivateKey {
        explicit PrivateKey() = default;
    };

public:
    /// Use build() / from_pieces(); the key keeps the constructor private in effect.
    explicit TokenVocab(PrivateKey /*key*/) {}

    static std::shared_ptr<const TokenVocab> build(const tokenizer::Tokenizer& tok);
    /// For tests: pieces[id] = raw bytes; `included[id]` false excludes the id.
    static std::shared_ptr<const TokenVocab> from_pieces(std::vector<std::string> pieces,
                                                         std::vector<bool> included);

    [[nodiscard]] std::size_t vocab_size() const noexcept { return pieces_.size(); }
    [[nodiscard]] std::size_t trie_nodes() const noexcept { return byte_.size(); }
    [[nodiscard]] bool included(std::int32_t id) const;
    [[nodiscard]] const std::string& piece(std::int32_t id) const;

    // Flattened preorder trie (root excluded): node i has `byte_[i]` at depth `depth_[i]`
    // (1-based), its subtree spans [i, subtree_end_[i]), and tokens ending at it are
    // toks_[tok_begin_[i] .. tok_begin_[i+1]).
    std::vector<std::uint8_t> byte_;
    std::vector<std::uint16_t> depth_;
    std::vector<std::uint32_t> subtree_end_;
    std::vector<std::uint32_t> tok_begin_;
    std::vector<std::int32_t> toks_;
    std::size_t max_depth_ = 0;

private:
    std::vector<std::string> pieces_;
    std::vector<bool> included_;
};

struct MaskStats {
    std::uint64_t masks = 0;            ///< mask() calls
    std::uint64_t cache_hits = 0;       ///< served from the per-state mask cache
    std::uint64_t trie_nodes_visited = 0;
    std::size_t states = 0;             ///< interned matcher states
    std::size_t compactions = 0;
};

/// Snapshot of a TokenMatcher's position (for speculative rollback). Valid across cache
/// compaction; only for the TokenMatcher (grammar) that produced it.
struct MatcherSnapshot {
    std::vector<std::vector<std::uint32_t>> stacks;
    bool accepting = false;
    bool finished = false;
};

/// Token-mask engine for one sequence.
class TokenMatcher {
public:
    /// `eos_ids` must be non-empty; they are allowed only when the document is complete.
    TokenMatcher(const Grammar& g, std::shared_ptr<const TokenVocab> vocab, std::vector<std::int32_t> eos_ids);
    TokenMatcher(TokenMatcher&&) noexcept;
    TokenMatcher& operator=(TokenMatcher&&) noexcept;
    ~TokenMatcher();

    /// Allowed-token bitset for the current step over [0, vocab_size): bit i of word i/64.
    /// The reference stays valid until the next non-const call.
    [[nodiscard]] std::span<const std::uint64_t> mask();
    [[nodiscard]] bool is_allowed(std::int32_t id);
    /// Advances by one token. Throws Error(Api) if the token is not allowed (state
    /// unchanged). Accepting an EOS id finishes the matcher.
    void accept_token(std::int32_t id);
    /// The text so far is a complete document (EOS allowed).
    [[nodiscard]] bool is_complete() const;
    /// An EOS token has been accepted.
    [[nodiscard]] bool is_finished() const noexcept { return finished_; }
    void reset();

    [[nodiscard]] MatcherSnapshot snapshot() const;
    void restore(const MatcherSnapshot& s);

    [[nodiscard]] MaskStats stats() const;
    [[nodiscard]] const TokenVocab& vocab() const noexcept { return *vocab_; }
    [[nodiscard]] std::span<const std::int32_t> eos_ids() const noexcept { return eos_; }

    /// Reference implementation for tests: the same mask computed by walking every token
    /// independently with uncached transitions (a full vocabulary scan).
    [[nodiscard]] std::vector<std::uint64_t> mask_naive();

    /// Mask computed by the trie walk, bypassing the per-state mask cache (for timing).
    [[nodiscard]] std::span<const std::uint64_t> mask_uncached();

private:
    void maybe_compact();
    void compute_trie_mask(std::vector<std::uint64_t>& out);

    Grammar grammar_;
    std::shared_ptr<const TokenVocab> vocab_;
    std::vector<std::int32_t> eos_;
    std::unique_ptr<detail::Runtime> rt_;
    std::int32_t state_;
    bool finished_ = false;
    std::vector<std::uint64_t> scratch_;
    std::vector<std::int32_t> dfs_states_;
    struct Cache;
    std::unique_ptr<Cache> cache_;
    MaskStats stats_;
};

}  // namespace halo::sampling
