// Byte-level grammar builder, UTF-8 range compilation and the incremental matcher
// runtime. See grammar_impl.h for the model and invariants.

#include <algorithm>
#include <format>
#include <map>
#include <utility>

#include "halo/core/error.h"
#include "sampling/grammar_impl.h"

namespace halo::sampling::detail {

// ---------------------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------------------

void Builder::charge(std::size_t n) {
    total_ += n;
    HALO_CHECK(total_ <= max_elems_, ErrorCode::Unsupported,
               "structured output: grammar exceeds {} elements (schema too large)", max_elems_);
}

std::uint32_t Builder::new_rule(std::string name) {
    charge(1);
    rules_.emplace_back();
    names_.push_back(std::move(name));
    return static_cast<std::uint32_t>(rules_.size() - 1);
}

void Builder::add_alt(std::uint32_t rule, std::vector<Sym> seq) {
    charge(seq.size() + 1);
    rules_.at(rule).push_back(std::move(seq));
}

Sym Builder::set(const ByteSet& s) {
    std::string key(reinterpret_cast<const char*>(s.data()), sizeof(ByteSet));
    auto [it, inserted] = set_ids_.try_emplace(std::move(key), static_cast<std::uint32_t>(sets_.size()));
    if (inserted) sets_.push_back(s);
    return Sym{false, it->second};
}

Sym Builder::byte(std::uint8_t b) { return range(b, b); }

Sym Builder::range(std::uint8_t lo, std::uint8_t hi) {
    ByteSet s{};
    byteset_add(s, lo, hi);
    return set(s);
}

Sym Builder::empty() {
    if (!empty_rule_) {
        empty_rule_ = new_rule("empty");
        add_alt(*empty_rule_, {});
    }
    return ref(*empty_rule_);
}

Sym Builder::never(std::string name) { return ref(new_rule(std::move(name))); }

Sym Builder::literal(std::string_view bytes) {
    if (bytes.empty()) return empty();
    std::vector<Sym> seq;
    seq.reserve(bytes.size());
    for (const char c : bytes) seq.push_back(byte(static_cast<std::uint8_t>(c)));
    if (seq.size() == 1) return seq[0];
    const auto r = new_rule("lit");
    add_alt(r, std::move(seq));
    return ref(r);
}

Sym Builder::seq(std::vector<Sym> items, std::string name) {
    if (items.empty()) return empty();
    if (items.size() == 1) return items[0];
    const auto r = new_rule(std::move(name));
    add_alt(r, std::move(items));
    return ref(r);
}

Sym Builder::alt(std::vector<std::vector<Sym>> alternatives, std::string name) {
    const auto r = new_rule(std::move(name));
    for (auto& a : alternatives) add_alt(r, std::move(a));
    return ref(r);
}

Sym Builder::repeat(Sym item, std::size_t min, std::optional<std::size_t> max, std::string name,
                    std::optional<Sym> separator) {
    HALO_CHECK(!max || *max >= min, ErrorCode::Api, "structured output: {} has min {} > max {}", name, min,
               *max);
    // item{min,max} without separator: min copies then a tail-recursive optional chain.
    const auto plain = [&](Sym x, std::size_t lo, std::optional<std::size_t> hi) -> Sym {
        std::vector<Sym> parts(lo, x);
        if (!hi) {
            const auto s = new_rule(name + "_star");
            add_alt(s, {});
            add_alt(s, {x, ref(s)});
            parts.push_back(ref(s));
        } else if (*hi > lo) {
            // O_k = "" | x O_{k-1}; O_0 = "".
            Sym prev = empty();
            for (std::size_t k = 1; k <= *hi - lo; ++k) {
                const auto o = new_rule(std::format("{}_opt{}", name, k));
                add_alt(o, {});
                add_alt(o, {x, prev});
                prev = ref(o);
            }
            parts.push_back(prev);
        }
        return seq(std::move(parts), name);
    };
    if (max && *max == 0) return empty();
    if (!separator) return plain(item, min, max);
    const Sym tail = seq({*separator, item}, name + "_sep_item");
    const std::optional<std::size_t> tail_max = max ? std::optional<std::size_t>(*max - 1) : std::nullopt;
    if (min == 0) {
        return alt({{}, {item, plain(tail, 0, tail_max)}}, name);
    }
    return seq({item, plain(tail, min - 1, tail_max)}, name);
}

Sym Builder::literal_set(std::vector<std::string> strings, std::string name) {
    std::sort(strings.begin(), strings.end());
    strings.erase(std::unique(strings.begin(), strings.end()), strings.end());
    if (strings.empty()) return never(name + "_none");
    // Recursive trie construction over the sorted range [lo, hi) sharing prefix `depth`.
    struct Rec {
        Builder& b;
        const std::vector<std::string>& s;
        const std::string& name;
        Sym build(std::size_t lo, std::size_t hi, std::size_t depth) {
            std::vector<Sym> chain;
            // Compress single-child, non-terminal chains.
            while (true) {
                const bool terminal = s[lo].size() == depth;
                if (terminal) break;
                const auto c = static_cast<unsigned char>(s[lo][depth]);
                if (static_cast<unsigned char>(s[hi - 1][depth]) != c) break;
                chain.push_back(b.byte(c));
                ++depth;
            }
            const auto r = b.new_rule(name);
            std::size_t i = lo;
            if (s[i].size() == depth) {
                b.add_alt(r, {});
                ++i;
            }
            while (i < hi) {
                const auto c = static_cast<unsigned char>(s[i][depth]);
                std::size_t j = i;
                while (j < hi && static_cast<unsigned char>(s[j][depth]) == c) ++j;
                const Sym child = build(i, j, depth + 1);
                b.add_alt(r, {b.byte(c), child});
                i = j;
            }
            chain.push_back(Builder::ref(r));
            return b.seq(std::move(chain), name);
        }
    };
    Rec rec{*this, strings, name};
    return rec.build(0, strings.size(), 0);
}

GrammarData Builder::finish(Sym root) {
    const std::size_t n = rules_.size();
    const auto sym_ok = [&](const Sym& s, const std::vector<char>& prod) {
        return s.is_rule ? prod[s.id] != 0 : !byteset_empty(sets_[s.id]);
    };
    // 1. Productivity fixpoint.
    std::vector<char> prod(n, 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t r = 0; r < n; ++r) {
            if (prod[r]) continue;
            for (const auto& a : rules_[r]) {
                if (std::all_of(a.begin(), a.end(), [&](const Sym& s) { return sym_ok(s, prod); })) {
                    prod[r] = 1;
                    changed = true;
                    break;
                }
            }
        }
    }
    const Sym root_rule = root.is_rule ? root : seq({root, empty()}, "root");
    HALO_CHECK(prod[root_rule.id] != 0, ErrorCode::Api,
               "structured output: the schema admits no finite JSON value within the nesting limit");
    // 2. Prune unproductive alternatives.
    for (std::size_t r = 0; r < n; ++r) {
        auto& alts = rules_[r];
        std::erase_if(alts, [&](const std::vector<Sym>& a) {
            return !std::all_of(a.begin(), a.end(), [&](const Sym& s) { return sym_ok(s, prod); });
        });
    }
    // 3. Reachability.
    std::vector<char> reach(n, 0);
    std::vector<std::uint32_t> todo{root_rule.id};
    reach[root_rule.id] = 1;
    while (!todo.empty()) {
        const auto r = todo.back();
        todo.pop_back();
        for (const auto& a : rules_[r]) {
            for (const auto& s : a) {
                if (s.is_rule && !reach[s.id]) {
                    reach[s.id] = 1;
                    todo.push_back(s.id);
                }
            }
        }
    }
    // 4. Left recursion through a non-tail reference would make expansion push forever.
    //    Tail-position cycles are harmless (expansion dedupes (pos, parent) pairs).
    std::vector<char> nullable(n, 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t r = 0; r < n; ++r) {
            if (nullable[r] || !reach[r]) continue;
            for (const auto& a : rules_[r]) {
                if (std::all_of(a.begin(), a.end(), [&](const Sym& s) { return s.is_rule && nullable[s.id]; })) {
                    nullable[r] = 1;
                    changed = true;
                    break;
                }
            }
        }
    }
    struct Edge {
        std::uint32_t to;
        bool tail;
    };
    std::vector<std::vector<Edge>> left(n);
    for (std::size_t r = 0; r < n; ++r) {
        if (!reach[r]) continue;
        for (const auto& a : rules_[r]) {
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (!a[i].is_rule) break;
                left[r].push_back(Edge{a[i].id, i + 1 == a.size()});
                if (!nullable[a[i].id]) break;
            }
        }
    }
    {  // Iterative Tarjan SCC.
        std::vector<std::int32_t> index(n, -1), low(n, 0), comp(n, -1);
        std::vector<char> on(n, 0);
        std::vector<std::uint32_t> st;
        std::int32_t counter = 0, ncomp = 0;
        for (std::uint32_t s0 = 0; s0 < n; ++s0) {
            if (!reach[s0] || index[s0] >= 0) continue;
            std::vector<std::pair<std::uint32_t, std::size_t>> call{{s0, 0}};
            index[s0] = low[s0] = counter++;
            st.push_back(s0);
            on[s0] = 1;
            while (!call.empty()) {
                auto& [v, ei] = call.back();
                if (ei < left[v].size()) {
                    const auto w = left[v][ei++].to;
                    if (index[w] < 0) {
                        index[w] = low[w] = counter++;
                        st.push_back(w);
                        on[w] = 1;
                        call.emplace_back(w, 0);
                    } else if (on[w]) {
                        low[v] = std::min(low[v], index[w]);
                    }
                } else {
                    const auto vv = v;
                    if (low[vv] == index[vv]) {
                        while (true) {
                            const auto w = st.back();
                            st.pop_back();
                            on[w] = 0;
                            comp[w] = ncomp;
                            if (w == vv) break;
                        }
                        ++ncomp;
                    }
                    call.pop_back();
                    if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[vv]);
                }
            }
        }
        for (std::size_t r = 0; r < n; ++r) {
            for (const auto& e : left[r]) {
                HALO_CHECK(e.tail || comp[r] != comp[e.to], ErrorCode::Unsupported,
                           "structured output: left-recursive grammar (rule '{}')", names_[r]);
            }
        }
    }
    // 5. Flatten reachable rules.
    GrammarData g;
    g.sets = sets_;
    std::vector<std::uint32_t> remap(n, 0);
    for (std::size_t r = 0; r < n; ++r) {
        if (reach[r]) {
            remap[r] = static_cast<std::uint32_t>(g.alts.size());
            g.alts.emplace_back();
            g.names.push_back(names_[r]);
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        if (!reach[r]) continue;
        auto& out = g.alts[remap[r]];
        for (const auto& a : rules_[r]) {
            out.push_back(static_cast<std::uint32_t>(g.elems.size()));
            for (const auto& s : a) {
                g.elems.push_back(s.is_rule ? Elem{ElemKind::Ref, remap[s.id]} : Elem{ElemKind::Bytes, s.id});
            }
            g.elems.push_back(Elem{ElemKind::End, 0});
        }
    }
    g.root = remap[root_rule.id];
    return g;
}

namespace {
std::string set_to_string(const ByteSet& s) {
    std::string out = "[";
    for (unsigned b = 0; b < 256;) {
        if (!byteset_has(s, static_cast<std::uint8_t>(b))) {
            ++b;
            continue;
        }
        unsigned e = b;
        while (e + 1 < 256 && byteset_has(s, static_cast<std::uint8_t>(e + 1))) ++e;
        const auto one = [](unsigned c) {
            return (c >= 0x21 && c < 0x7F && c != '\\' && c != ']' && c != '-') ? std::string(1, static_cast<char>(c))
                                                                               : std::format("\\x{:02X}", c);
        };
        out += one(b);
        if (e > b) out += "-" + one(e);
        b = e + 1;
    }
    return out + "]";
}
}  // namespace

std::string GrammarData::debug_string() const {
    std::string out;
    for (std::size_t r = 0; r < alts.size(); ++r) {
        out += std::format("{}{}_{} ::=", r == root ? "*" : "", names[r], r);
        for (std::size_t a = 0; a < alts[r].size(); ++a) {
            out += a == 0 ? " " : " | ";
            std::size_t p = alts[r][a];
            if (elems[p].kind == ElemKind::End) out += "\"\"";
            for (; elems[p].kind != ElemKind::End; ++p) {
                if (elems[p].kind == ElemKind::Bytes) {
                    out += set_to_string(sets[elems[p].arg]) + " ";
                } else {
                    out += std::format("{}_{} ", names[elems[p].arg], elems[p].arg);
                }
            }
        }
        out += "\n";
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// UTF-8 ranges
// ---------------------------------------------------------------------------------------

CpRanges normalize_ranges(CpRanges r) {
    for (auto& [lo, hi] : r) {
        if (lo > hi) std::swap(lo, hi);
        hi = std::min<std::uint32_t>(hi, 0x10FFFF);
    }
    std::erase_if(r, [](const auto& p) { return p.first > 0x10FFFF; });
    std::sort(r.begin(), r.end());
    CpRanges merged;
    for (const auto& p : r) {
        if (!merged.empty() && p.first <= merged.back().second + 1) {
            merged.back().second = std::max(merged.back().second, p.second);
        } else {
            merged.push_back(p);
        }
    }
    return subtract_ranges(merged, {{0xD800, 0xDFFF}});
}

CpRanges complement_ranges(const CpRanges& r) {
    CpRanges out;
    std::uint32_t next = 0;
    for (const auto& [lo, hi] : r) {
        if (lo > next) out.emplace_back(next, lo - 1);
        next = hi + 1;
    }
    if (next <= 0x10FFFF) out.emplace_back(next, 0x10FFFF);
    return subtract_ranges(out, {{0xD800, 0xDFFF}});
}

CpRanges subtract_ranges(const CpRanges& a, const CpRanges& b) {
    CpRanges out;
    for (auto [lo, hi] : a) {
        std::uint64_t cur = lo;
        for (const auto& [blo, bhi] : b) {
            if (bhi < cur || blo > hi) continue;
            if (blo > cur) out.emplace_back(static_cast<std::uint32_t>(cur), blo - 1);
            cur = std::uint64_t{bhi} + 1;
            if (cur > hi) break;
        }
        if (cur <= hi) out.emplace_back(static_cast<std::uint32_t>(cur), hi);
    }
    return out;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

namespace {
using ByteRanges = std::vector<std::pair<std::uint8_t, std::uint8_t>>;

void utf8_split(std::uint32_t lo, std::uint32_t hi, std::vector<ByteRanges>& out) {
    // Same encoded length (the utf8-ranges algorithm, as in RE2 / regex-syntax).
    constexpr std::array<std::uint32_t, 3> kMax{0x7F, 0x7FF, 0xFFFF};
    for (const auto m : kMax) {
        if (lo <= m && hi > m) {
            utf8_split(lo, m, out);
            utf8_split(m + 1, hi, out);
            return;
        }
    }
    if (hi <= 0x7F) {
        out.push_back({{static_cast<std::uint8_t>(lo), static_cast<std::uint8_t>(hi)}});
        return;
    }
    const int len = lo <= 0x7FF ? 2 : lo <= 0xFFFF ? 3 : 4;
    for (int i = 1; i < len; ++i) {
        const std::uint32_t m = (1U << (6 * i)) - 1;
        if ((lo & ~m) != (hi & ~m)) {
            if ((lo & m) != 0) {
                utf8_split(lo, lo | m, out);
                utf8_split((lo | m) + 1, hi, out);
                return;
            }
            if ((hi & m) != m) {
                utf8_split(lo, (hi & ~m) - 1, out);
                utf8_split(hi & ~m, hi, out);
                return;
            }
        }
    }
    std::string a, b;
    append_utf8(a, lo);
    append_utf8(b, hi);
    ByteRanges seq;
    for (std::size_t i = 0; i < a.size(); ++i) {
        seq.emplace_back(static_cast<std::uint8_t>(a[i]), static_cast<std::uint8_t>(b[i]));
    }
    out.push_back(std::move(seq));
}
}  // namespace

std::vector<std::vector<std::pair<std::uint8_t, std::uint8_t>>> utf8_sequences(const CpRanges& r) {
    std::vector<ByteRanges> out;
    for (const auto& [lo, hi] : normalize_ranges(r)) utf8_split(lo, hi, out);
    return out;
}

Sym utf8_class(Builder& b, const CpRanges& r, std::string name) {
    const auto seqs = utf8_sequences(r);
    ByteSet single{};
    std::vector<std::vector<Sym>> alts;
    for (const auto& s : seqs) {
        if (s.size() == 1) {
            byteset_add(single, s[0].first, s[0].second);
            continue;
        }
        std::vector<Sym> seq;
        for (const auto& [lo, hi] : s) seq.push_back(b.range(lo, hi));
        alts.push_back(std::move(seq));
    }
    if (alts.empty()) return byteset_empty(single) ? b.never(name + "_none") : b.set(single);
    if (!byteset_empty(single)) alts.insert(alts.begin(), {b.set(single)});
    return b.alt(std::move(alts), std::move(name));
}

// ---------------------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------------------

std::size_t Runtime::VecHash::operator()(const std::vector<std::int32_t>& v) const noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const auto x : v) {
        h ^= static_cast<std::uint32_t>(x);
        h *= 0x100000001b3ULL;
    }
    return static_cast<std::size_t>(h);
}

Runtime::Runtime(std::shared_ptr<const GrammarData> g) : g_(std::move(g)) {
    out_.clear();
    accepting_ = false;
    visited_.clear();
    for (const auto a : g_->alts[g_->root]) work_.emplace_back(a, -1);
    expand(0, 0);  // processes work_ only (see expand)
    initial_ = intern_current();
    HALO_CHECK(initial_ != kDead, ErrorCode::Kernel, "structured output: grammar has no start state");
}

std::size_t Runtime::idx(std::int32_t s) const {
    HALO_CHECK(s >= 0 && static_cast<std::size_t>(s) < states_.size(), ErrorCode::Kernel,
               "structured output: invalid matcher state {}", s);
    return static_cast<std::size_t>(s);
}

std::int32_t Runtime::node(std::uint32_t pos, std::int32_t parent) {
    const std::uint64_t key = (std::uint64_t{pos} << 32) | static_cast<std::uint32_t>(parent + 1);
    const auto it = node_ids_.find(key);
    if (it != node_ids_.end()) return it->second;
    const std::uint32_t depth = parent < 0 ? 1 : nodes_[static_cast<std::size_t>(parent)].depth + 1;
    HALO_CHECK(depth <= kMaxStackDepth, ErrorCode::Kernel,
               "structured output: matcher stack depth {} exceeds {} (grammar bug)", depth, kMaxStackDepth);
    max_depth_seen_ = std::max<std::size_t>(max_depth_seen_, depth);
    const auto id = static_cast<std::int32_t>(nodes_.size());
    nodes_.push_back(Node{pos, parent, depth});
    node_ids_.emplace(key, id);
    return id;
}

// Processes work_ (seeded by the caller); `pos`/`parent` are unused placeholders kept for
// symmetry with the header comment.
void Runtime::expand(std::uint32_t /*pos*/, std::int32_t /*parent*/) {
    const auto& elems = g_->elems;
    while (!work_.empty()) {
        const auto [pos, parent] = work_.back();
        work_.pop_back();
        const std::uint64_t key = (std::uint64_t{pos} << 32) | static_cast<std::uint32_t>(parent + 1);
        if (!visited_.emplace(key, true).second) continue;
        const Elem e = elems[pos];
        switch (e.kind) {
            case ElemKind::End:
                if (parent < 0) {
                    accepting_ = true;
                } else {
                    const Node& p = nodes_[static_cast<std::size_t>(parent)];
                    work_.emplace_back(p.pos, p.parent);
                }
                break;
            case ElemKind::Bytes:
                out_.push_back(node(pos, parent));
                break;
            case ElemKind::Ref: {
                // Tail call: nothing follows the reference, so the callee returns straight
                // to our parent and no frame is pushed.
                const std::int32_t cont = elems[pos + 1].kind == ElemKind::End ? parent : node(pos + 1, parent);
                for (const auto a : g_->alts[e.arg]) work_.emplace_back(a, cont);
                break;
            }
        }
    }
}

std::int32_t Runtime::intern_current() {
    std::sort(out_.begin(), out_.end());
    out_.erase(std::unique(out_.begin(), out_.end()), out_.end());
    if (out_.empty() && !accepting_) return kDead;
    std::vector<std::int32_t> key = out_;
    if (accepting_) key.push_back(-1);
    const auto it = state_ids_.find(key);
    if (it != state_ids_.end()) return it->second;
    const auto id = static_cast<std::int32_t>(states_.size());
    states_.push_back(State{static_cast<std::uint32_t>(state_nodes_.size()), static_cast<std::uint32_t>(out_.size()),
                            accepting_});
    state_nodes_.insert(state_nodes_.end(), out_.begin(), out_.end());
    trans_.resize(trans_.size() + 256, -2);
    state_ids_.emplace(std::move(key), id);
    return id;
}

std::int32_t Runtime::compute_step(std::int32_t state, std::uint8_t b) {
    const State st = states_[idx(state)];
    out_.clear();
    accepting_ = false;
    visited_.clear();
    work_.clear();
    for (std::uint32_t i = 0; i < st.count; ++i) {
        const Node& n = nodes_[static_cast<std::size_t>(state_nodes_[st.begin + i])];
        const Elem e = g_->elems[n.pos];
        if (byteset_has(g_->sets[e.arg], b)) work_.emplace_back(n.pos + 1, n.parent);
    }
    if (work_.empty()) return kDead;
    expand(0, 0);
    return intern_current();
}

std::int32_t Runtime::step(std::int32_t state, std::uint8_t b) {
    const std::size_t slot = idx(state) * 256 + b;
    const std::int32_t t = trans_[slot];
    if (t != -2) return t;
    const std::int32_t r = compute_step(state, b);
    trans_[slot] = r;  // trans_ may have grown; slot index is still valid
    return r;
}

std::int32_t Runtime::step_uncached(std::int32_t state, std::uint8_t b) { return compute_step(state, b); }

ByteSet Runtime::next_bytes(std::int32_t state) const {
    const State st = states_[idx(state)];
    ByteSet out{};
    for (std::uint32_t i = 0; i < st.count; ++i) {
        const Node& n = nodes_[static_cast<std::size_t>(state_nodes_[st.begin + i])];
        const auto& s = g_->sets[g_->elems[n.pos].arg];
        for (int w = 0; w < 4; ++w) out[static_cast<std::size_t>(w)] |= s[static_cast<std::size_t>(w)];
    }
    return out;
}

Materialized Runtime::materialize(std::int32_t state) const {
    const State st = states_[idx(state)];
    Materialized m;
    m.accepting = st.accepting;
    for (std::uint32_t i = 0; i < st.count; ++i) {
        Stack s;
        for (std::int32_t n = state_nodes_[st.begin + i]; n >= 0; n = nodes_[static_cast<std::size_t>(n)].parent) {
            s.push_back(nodes_[static_cast<std::size_t>(n)].pos);
        }
        m.stacks.push_back(std::move(s));
    }
    return m;
}

std::int32_t Runtime::intern(const Materialized& m) {
    out_.clear();
    accepting_ = m.accepting;
    for (const auto& s : m.stacks) {
        std::int32_t parent = -1;
        for (auto it = s.rbegin(); it != s.rend(); ++it) {
            HALO_CHECK(*it < g_->elems.size(), ErrorCode::Kernel, "structured output: bad snapshot");
            parent = node(*it, parent);
        }
        out_.push_back(parent);
    }
    return intern_current();
}

}  // namespace halo::sampling::detail
