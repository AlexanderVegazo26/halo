#pragma once
// Internal representation of a compiled byte-level grammar and its incremental matcher
// runtime (WS-H, TRD §25). Not a public header: include/halo/sampling/structured.h is the
// contract; tests may include this for white-box checks.
//
// Grammar model
//   A context-free grammar over *bytes*. A rule is a list of alternatives; an alternative
//   is a sequence of elements, each either a 256-bit byte set or a reference to a rule,
//   terminated by End. Everything JSON- or UTF-8-specific (codepoints, escapes, nesting)
//   is compiled into this form, so a token that ends in the middle of a UTF-8 sequence is
//   simply a prefix of some byte path.
//
//   Invariants established by Builder::finish():
//     * every rule is productive (derives at least one finite byte string) — so any
//       non-empty matcher stack can be completed, which is what makes "the mask never
//       allows a token that leads to a dead state" hold;
//     * no left recursion (expansion always terminates);
//     * only rules reachable from the root are kept.
//
// Matcher runtime
//   A matcher configuration is a set of stacks (nondeterministic pushdown automaton, as in
//   llama.cpp's grammar engine). Stacks are persistent linked lists of Nodes that are
//   hash-consed, so equal stacks share one node id; a configuration is interned as a
//   sorted node-id set -> state id. Transitions (state, byte) -> state are memoized in a
//   dense 256-entry row per state, filled lazily. A reference in tail position does not
//   push a frame (tail-call elimination), so repetition loops keep a constant stack and
//   revisit the same state id — which is what makes the transition and mask caches hit.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace halo::sampling::detail {

using ByteSet = std::array<std::uint64_t, 4>;

[[nodiscard]] inline bool byteset_has(const ByteSet& s, std::uint8_t b) noexcept {
    return ((s[b >> 6U] >> (b & 63U)) & 1U) != 0;
}
inline void byteset_add(ByteSet& s, std::uint8_t lo, std::uint8_t hi) noexcept {
    for (unsigned b = lo; b <= hi; ++b) s[b >> 6U] |= std::uint64_t{1} << (b & 63U);
}
[[nodiscard]] inline bool byteset_empty(const ByteSet& s) noexcept {
    return (s[0] | s[1] | s[2] | s[3]) == 0;
}

enum class ElemKind : std::uint8_t { End, Bytes, Ref };

struct Elem {
    ElemKind kind = ElemKind::End;
    std::uint32_t arg = 0;  // Bytes: set index; Ref: rule index
};

/// Immutable, flattened grammar.
struct GrammarData {
    std::vector<Elem> elems;                        // all alternatives, End-terminated
    std::vector<ByteSet> sets;                      // byte sets referenced by Bytes elems
    std::vector<std::vector<std::uint32_t>> alts;   // per rule: start offsets into elems
    std::vector<std::string> names;                 // per rule, for debug_string()
    std::uint32_t root = 0;

    [[nodiscard]] std::string debug_string() const;
};

/// A grammar symbol while building: a byte set or a rule.
struct Sym {
    bool is_rule = false;
    std::uint32_t id = 0;
};

/// Grammar construction with size limits. Throws halo::Error(Unsupported) when the
/// grammar would exceed `max_elems` elements.
class Builder {
public:
    explicit Builder(std::size_t max_elems) : max_elems_(max_elems) {}

    std::uint32_t new_rule(std::string name);
    void add_alt(std::uint32_t rule, std::vector<Sym> seq);
    [[nodiscard]] std::size_t num_rules() const noexcept { return rules_.size(); }

    Sym set(const ByteSet& s);
    Sym byte(std::uint8_t b);
    Sym range(std::uint8_t lo, std::uint8_t hi);
    static Sym ref(std::uint32_t rule) { return Sym{true, rule}; }

    /// Rule deriving exactly the empty string.
    Sym empty();
    /// Rule deriving nothing (pruned by finish()).
    Sym never(std::string name = "never");
    /// Exact byte string (empty string -> empty()).
    Sym literal(std::string_view bytes);
    /// Sequence (a single element is returned as-is).
    Sym seq(std::vector<Sym> items, std::string name = "seq");
    /// Alternatives, each a sequence.
    Sym alt(std::vector<std::vector<Sym>> alternatives, std::string name = "alt");
    /// item{min,max}; max = nullopt is unbounded. Separator (if given) goes between items.
    Sym repeat(Sym item, std::size_t min, std::optional<std::size_t> max, std::string name,
               std::optional<Sym> separator = std::nullopt);
    /// Any one of a set of byte strings, built as a trie (keeps stacks deterministic for
    /// large enums).
    Sym literal_set(std::vector<std::string> strings, std::string name = "literals");

    /// Prunes unproductive alternatives, checks for left recursion, keeps reachable rules
    /// and flattens. Throws Error(Api) if the root derives nothing.
    [[nodiscard]] GrammarData finish(Sym root);

private:
    void charge(std::size_t n);

    std::size_t max_elems_;
    std::size_t total_ = 0;
    std::vector<ByteSet> sets_;
    std::unordered_map<std::string, std::uint32_t> set_ids_;
    std::vector<std::vector<std::vector<Sym>>> rules_;
    std::vector<std::string> names_;
    std::optional<std::uint32_t> empty_rule_;
};

// ---------------------------------------------------------------------------------------
// UTF-8 range compilation
// ---------------------------------------------------------------------------------------

/// Sorted, non-overlapping closed codepoint ranges.
using CpRanges = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

/// Normalizes (sorts, merges) and removes the surrogate block D800-DFFF.
[[nodiscard]] CpRanges normalize_ranges(CpRanges r);
[[nodiscard]] CpRanges complement_ranges(const CpRanges& r);  // within [0, 0x10FFFF]
[[nodiscard]] CpRanges subtract_ranges(const CpRanges& a, const CpRanges& b);

/// Byte-level sequences (each a list of byte ranges) whose union is exactly the UTF-8
/// encodings of the codepoints in `r` (surrogates excluded).
[[nodiscard]] std::vector<std::vector<std::pair<std::uint8_t, std::uint8_t>>> utf8_sequences(
    const CpRanges& r);

/// Grammar symbol matching one UTF-8 encoded codepoint from `r` (raw, not JSON-escaped).
Sym utf8_class(Builder& b, const CpRanges& r, std::string name);

/// Appends the UTF-8 encoding of cp.
void append_utf8(std::string& out, std::uint32_t cp);

// ---------------------------------------------------------------------------------------
// Matcher runtime
// ---------------------------------------------------------------------------------------

inline constexpr std::int32_t kDead = -1;

/// A stack materialized as grammar positions, top first. Used for snapshots and
/// compaction (state ids are only meaningful inside one Runtime).
using Stack = std::vector<std::uint32_t>;
struct Materialized {
    std::vector<Stack> stacks;
    bool accepting = false;
};

class Runtime {
public:
    explicit Runtime(std::shared_ptr<const GrammarData> g);

    /// State after expanding the root (before any byte).
    [[nodiscard]] std::int32_t initial() const noexcept { return initial_; }
    /// Memoized transition; kDead if no stack accepts `b`.
    [[nodiscard]] std::int32_t step(std::int32_t state, std::uint8_t b);
    /// Same result without reading or writing the transition memo (reference path for
    /// differential tests).
    [[nodiscard]] std::int32_t step_uncached(std::int32_t state, std::uint8_t b);

    [[nodiscard]] bool accepting(std::int32_t state) const { return states_[idx(state)].accepting; }
    /// True if some stack can consume another byte.
    [[nodiscard]] bool can_continue(std::int32_t state) const { return states_[idx(state)].count > 0; }
    /// Bytes some stack of `state` can consume next.
    [[nodiscard]] ByteSet next_bytes(std::int32_t state) const;

    [[nodiscard]] std::size_t num_states() const noexcept { return states_.size(); }
    [[nodiscard]] std::size_t num_nodes() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t max_stack_depth() const noexcept { return max_depth_seen_; }

    [[nodiscard]] Materialized materialize(std::int32_t state) const;
    [[nodiscard]] std::int32_t intern(const Materialized& m);

    [[nodiscard]] const GrammarData& grammar() const noexcept { return *g_; }
    [[nodiscard]] const std::shared_ptr<const GrammarData>& grammar_ptr() const noexcept { return g_; }

private:
    struct Node {
        std::uint32_t pos;     // element index to match next at this level
        std::int32_t parent;   // -1 = bottom
        std::uint32_t depth;
    };
    struct State {
        std::uint32_t begin = 0;  // into state_nodes_
        std::uint32_t count = 0;
        bool accepting = false;
    };

    [[nodiscard]] std::size_t idx(std::int32_t s) const;
    std::int32_t node(std::uint32_t pos, std::int32_t parent);
    /// Expands position `pos` under `parent` until every stack has a Bytes element on
    /// top; appends node ids to out_ and sets accepting_ when a stack empties.
    void expand(std::uint32_t pos, std::int32_t parent);
    std::int32_t intern_current();
    std::int32_t compute_step(std::int32_t state, std::uint8_t b);

    std::shared_ptr<const GrammarData> g_;
    std::vector<Node> nodes_;
    std::unordered_map<std::uint64_t, std::int32_t> node_ids_;
    std::vector<State> states_;
    std::vector<std::int32_t> state_nodes_;
    struct VecHash {
        std::size_t operator()(const std::vector<std::int32_t>& v) const noexcept;
    };
    std::unordered_map<std::vector<std::int32_t>, std::int32_t, VecHash> state_ids_;
    std::vector<std::int32_t> trans_;  // 256 per state; -2 = unknown
    std::int32_t initial_ = kDead;
    std::size_t max_depth_seen_ = 0;

    // scratch for expand()
    std::vector<std::int32_t> out_;
    bool accepting_ = false;
    std::vector<std::pair<std::uint32_t, std::int32_t>> work_;
    std::unordered_map<std::uint64_t, bool> visited_;
};

/// Hard cap on stack depth. The compiler bounds nesting by construction (containers are
/// unrolled per level, repetition is tail-recursive), so reaching this is an internal bug
/// and throws Error(Kernel) rather than silently dropping a stack.
inline constexpr std::uint32_t kMaxStackDepth = 4096;

}  // namespace halo::sampling::detail
