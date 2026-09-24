// JSON text building blocks and the ECMA-262 pattern subset compiler.

#include "sampling/json_grammar.h"

#include <format>
#include <vector>

#include "halo/core/error.h"

namespace halo::sampling::detail {

// ---------------------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------------------

Sym JsonGrammar::ws() {
    if (!ws_) {
        ByteSet s{};
        for (const char c : {' ', '\t', '\n', '\r'}) byteset_add(s, static_cast<std::uint8_t>(c), static_cast<std::uint8_t>(c));
        ws_ = b_.repeat(b_.set(s), 0, opts_.max_whitespace, "ws");
    }
    return *ws_;
}

Sym JsonGrammar::plain_char() {
    if (!plain_) plain_ = utf8_class(b_, normalize_ranges({{0x20, 0x21}, {0x23, 0x5B}, {0x5D, 0x10FFFF}}), "plain_char");
    return *plain_;
}

Sym JsonGrammar::string_char() {
    if (!char_) {
        const Sym hex = [&] {
            ByteSet s{};
            byteset_add(s, '0', '9');
            byteset_add(s, 'A', 'F');
            byteset_add(s, 'a', 'f');
            return b_.set(s);
        }();
        const auto set_of = [&](std::string_view chars) {
            ByteSet s{};
            for (const char c : chars) byteset_add(s, static_cast<std::uint8_t>(c), static_cast<std::uint8_t>(c));
            return b_.set(s);
        };
        // \uXXXX that is not a surrogate: [0-9A-Ca-c]XXX | [Dd][0-7]XX | [EFef]XXX
        const Sym u_bmp = b_.alt({{set_of("0123456789ABCabc"), hex, hex, hex},
                                  {set_of("Dd"), b_.range('0', '7'), hex, hex},
                                  {set_of("EFef"), hex, hex, hex}},
                                 "u_bmp");
        // surrogate pair: [Dd][89ABab]XX \u [Dd][C-Fc-f]XX (one codepoint)
        const Sym u_pair = b_.seq({set_of("Dd"), set_of("89ABab"), hex, hex, b_.byte('\\'), b_.byte('u'),
                                   set_of("Dd"), set_of("CDEFcdef"), hex, hex},
                                  "u_pair");
        const Sym esc = b_.alt({{set_of("\"\\/bfnrt")}, {b_.byte('u'), u_bmp}, {b_.byte('u'), u_pair}}, "escape");
        char_ = b_.alt({{plain_char()}, {b_.byte('\\'), esc}}, "string_char");
    }
    return *char_;
}

Sym JsonGrammar::string_any() {
    if (!string_) {
        string_ = b_.seq({b_.byte('"'), b_.repeat(string_char(), 0, std::nullopt, "chars"), b_.byte('"')}, "string");
    }
    return *string_;
}

Sym JsonGrammar::string_bounded(std::size_t min, std::optional<std::size_t> max) {
    if (min == 0 && !max) return string_any();
    return b_.seq({b_.byte('"'), b_.repeat(string_char(), min, max, "chars_n"), b_.byte('"')}, "string_n");
}

Sym JsonGrammar::integer() {
    if (!integer_) {
        const Sym digit = b_.range('0', '9');
        const Sym int_part = b_.alt({{b_.byte('0')}, {b_.range('1', '9'), b_.repeat(digit, 0, std::nullopt, "digits")}},
                                    "int_part");
        integer_ = b_.alt({{int_part}, {b_.byte('-'), int_part}}, "integer");
    }
    return *integer_;
}

Sym JsonGrammar::number() {
    if (!number_) {
        const Sym digit = b_.range('0', '9');
        const Sym digits1 = b_.repeat(digit, 1, std::nullopt, "digits1");
        ByteSet e{}, pm{};
        byteset_add(e, 'e', 'e');
        byteset_add(e, 'E', 'E');
        byteset_add(pm, '+', '+');
        byteset_add(pm, '-', '-');
        const Sym frac = b_.alt({{}, {b_.byte('.'), digits1}}, "frac");
        const Sym exp = b_.alt({{}, {b_.set(e), digits1}, {b_.set(e), b_.set(pm), digits1}}, "exp");
        number_ = b_.seq({integer(), frac, exp}, "number");
    }
    return *number_;
}

Sym JsonGrammar::boolean() {
    if (!boolean_) boolean_ = b_.literal_set({"true", "false"}, "boolean");
    return *boolean_;
}

Sym JsonGrammar::null() {
    if (!null_) null_ = b_.literal("null");
    return *null_;
}

Sym JsonGrammar::value_any(std::size_t level) {
    if (const auto it = value_.find(level); it != value_.end()) return it->second;
    std::vector<std::vector<Sym>> alts{{string_any()}, {number()}, {boolean()}, {null()}};
    if (level < opts_.max_nesting) {
        alts.push_back({object_any(level)});
        alts.push_back({array_any(level)});
    }
    const Sym v = b_.alt(std::move(alts), std::format("value{}", level));
    value_.emplace(level, v);
    return v;
}

Sym JsonGrammar::object_any(std::size_t level) {
    if (const auto it = object_.find(level); it != object_.end()) return it->second;
    if (level >= opts_.max_nesting) return b_.never("object_too_deep");
    // "{" ws ( "}" | member rest );  rest = ws ( "}" | "," ws member rest )
    const Sym member = b_.seq({string_any(), ws(), b_.byte(':'), ws(), value_any(level + 1)}, "member");
    const auto rest = b_.new_rule(std::format("obj_rest{}", level));
    const Sym tail = b_.alt({{b_.byte('}')}, {b_.byte(','), ws(), member, Builder::ref(rest)}}, "obj_tail");
    b_.add_alt(rest, {ws(), tail});
    const Sym body = b_.alt({{b_.byte('}')}, {member, Builder::ref(rest)}}, "obj_body");
    const Sym o = b_.seq({b_.byte('{'), ws(), body}, std::format("object{}", level));
    object_.emplace(level, o);
    return o;
}

Sym JsonGrammar::array_any(std::size_t level) {
    if (const auto it = array_.find(level); it != array_.end()) return it->second;
    if (level >= opts_.max_nesting) return b_.never("array_too_deep");
    const Sym item = value_any(level + 1);
    const auto rest = b_.new_rule(std::format("arr_rest{}", level));
    const Sym tail = b_.alt({{b_.byte(']')}, {b_.byte(','), ws(), item, Builder::ref(rest)}}, "arr_tail");
    b_.add_alt(rest, {ws(), tail});
    const Sym body = b_.alt({{b_.byte(']')}, {item, Builder::ref(rest)}}, "arr_body");
    const Sym a = b_.seq({b_.byte('['), ws(), body}, std::format("array{}", level));
    array_.emplace(level, a);
    return a;
}

Sym JsonGrammar::json_class(const CpRanges& cps_in, std::string name) {
    const CpRanges cps = normalize_ranges(cps_in);
    const CpRanges need_escape = normalize_ranges({{0x00, 0x1F}, {0x22, 0x22}, {0x5C, 0x5C}});
    const CpRanges raw = subtract_ranges(cps, need_escape);
    std::vector<std::string> escapes;
    for (const auto& [lo, hi] : cps) {
        for (std::uint32_t c = lo; c <= hi && c <= 0x5C; ++c) {
            switch (c) {
                case '"': escapes.emplace_back("\\\""); break;
                case '\\': escapes.emplace_back("\\\\"); break;
                case '\b': escapes.emplace_back("\\b"); break;
                case '\f': escapes.emplace_back("\\f"); break;
                case '\n': escapes.emplace_back("\\n"); break;
                case '\r': escapes.emplace_back("\\r"); break;
                case '\t': escapes.emplace_back("\\t"); break;
                default:
                    if (c < 0x20) escapes.push_back(std::format("\\u{:04x}", c));
            }
        }
    }
    std::vector<std::vector<Sym>> alts;
    if (!raw.empty()) alts.push_back({utf8_class(b_, raw, name + "_raw")});
    if (!escapes.empty()) alts.push_back({b_.literal_set(std::move(escapes), name + "_esc")});
    if (alts.empty()) return b_.never(name + "_empty");
    if (alts.size() == 1) return alts[0][0];
    return b_.alt(std::move(alts), std::move(name));
}

std::u32string decode_utf8(std::string_view s) {
    std::u32string out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp = 0;
        std::size_t n = 0;
        if (c < 0x80) {
            cp = c;
            n = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1FU;
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0FU;
            n = 3;
        } else {
            cp = c & 0x07U;
            n = 4;
        }
        HALO_CHECK(i + n <= s.size(), ErrorCode::Api, "structured output: invalid UTF-8");
        for (std::size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3FU);
        out.push_back(static_cast<char32_t>(cp));
        i += n;
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// Pattern (ECMA-262 subset) -> grammar
// ---------------------------------------------------------------------------------------

namespace {

struct Re {
    enum class Kind { Empty, Class, Concat, Alt, Repeat } kind = Kind::Empty;
    CpRanges cls;
    std::vector<Re> kids;
    std::size_t min = 0;
    std::optional<std::size_t> max;
};

struct Branch {
    Re re;
    bool start = false;  // anchored with ^
    bool end = false;    // anchored with $
};

const CpRanges& class_digit() {
    static const CpRanges r{{'0', '9'}};
    return r;
}
const CpRanges& class_word() {
    static const CpRanges r = normalize_ranges({{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}});
    return r;
}
const CpRanges& class_space() {
    // ECMA-262 WhiteSpace + LineTerminator.
    static const CpRanges r = normalize_ranges({{0x09, 0x0D},
                                                {0x20, 0x20},
                                                {0xA0, 0xA0},
                                                {0x1680, 0x1680},
                                                {0x2000, 0x200A},
                                                {0x2028, 0x2029},
                                                {0x202F, 0x202F},
                                                {0x205F, 0x205F},
                                                {0x3000, 0x3000},
                                                {0xFEFF, 0xFEFF}});
    return r;
}
const CpRanges& class_dot() {
    static const CpRanges r = complement_ranges({{0x0A, 0x0A}, {0x0D, 0x0D}, {0x2028, 0x2029}});
    return r;
}

class PatternParser {
public:
    PatternParser(std::u32string p, std::size_t max_repeat) : p_(std::move(p)), max_repeat_(max_repeat) {}

    /// Top-level alternatives. ECMA-262 alternation binds loosest, so in `^a|b$` the `^`
    /// belongs to the first branch and the `$` to the second: anchors are tracked per
    /// top-level branch (`^` at its start, `$` at its end); anywhere else they are
    /// Unsupported.
    std::vector<Branch> parse() {
        end_ = p_.size();
        std::vector<Branch> out;
        while (true) {
            Branch br;
            if (!at_end() && peek() == U'^') {
                br.start = true;
                ++i_;
            }
            br.re = parse_concat(0, &br.end);
            out.push_back(std::move(br));
            if (at_end()) break;
            if (peek() != U'|') fail_api("unexpected ')'");
            ++i_;
        }
        return out;
    }

private:
    [[noreturn]] void unsupported(std::string_view what) const {
        throw_error(ErrorCode::Unsupported, "structured output: pattern feature not supported: {}", what);
    }
    [[noreturn]] void fail_api(std::string_view what) const {
        throw_error(ErrorCode::Api, "structured output: invalid pattern: {}", what);
    }
    bool at_end() const { return i_ >= end_; }
    char32_t peek() const { return p_[i_]; }

    Re parse_alt(std::size_t depth) {
        if (depth > 256) unsupported("nesting deeper than 256");
        std::vector<Re> alts;
        alts.push_back(parse_concat(depth));
        while (!at_end() && peek() == U'|') {
            ++i_;
            alts.push_back(parse_concat(depth));
        }
        if (alts.size() == 1) return std::move(alts[0]);
        Re r;
        r.kind = Re::Kind::Alt;
        r.kids = std::move(alts);
        return r;
    }

    /// `top_end` (top-level branches only) is set when the branch ends with a `$` anchor.
    Re parse_concat(std::size_t depth, bool* top_end = nullptr) {
        Re r;
        r.kind = Re::Kind::Concat;
        while (!at_end() && peek() != U'|' && peek() != U')') {
            if (top_end != nullptr && peek() == U'$' && (i_ + 1 == end_ || p_[i_ + 1] == U'|')) {
                ++i_;
                *top_end = true;
                break;
            }
            r.kids.push_back(parse_quantified(depth));
        }
        if (r.kids.empty()) return Re{};
        if (r.kids.size() == 1) return std::move(r.kids[0]);
        return r;
    }

    std::optional<std::size_t> read_int() {
        std::size_t v = 0;
        bool any = false;
        while (!at_end() && peek() >= U'0' && peek() <= U'9') {
            v = v * 10 + static_cast<std::size_t>(peek() - U'0');
            if (v > max_repeat_) unsupported(std::format("repetition count above {}", max_repeat_));
            ++i_;
            any = true;
        }
        return any ? std::optional<std::size_t>(v) : std::nullopt;
    }

    // Tries to parse {n}, {n,}, {n,m} at i_ (pointing at '{'); restores i_ on failure.
    bool try_braces(std::size_t& lo, std::optional<std::size_t>& hi) {
        const std::size_t save = i_;
        ++i_;
        const auto a = read_int();
        if (!a) {
            i_ = save;
            return false;
        }
        lo = *a;
        hi = a;
        if (!at_end() && peek() == U',') {
            ++i_;
            hi = read_int();
        }
        if (at_end() || peek() != U'}') {
            i_ = save;
            return false;
        }
        ++i_;
        if (hi && *hi < lo) fail_api("{n,m} with m < n");
        return true;
    }

    Re parse_quantified(std::size_t depth) {
        Re atom = parse_atom(depth);
        bool quantified = false;
        while (!at_end()) {
            std::size_t lo = 0;
            std::optional<std::size_t> hi;
            const char32_t c = peek();
            if (c == U'*') {
                lo = 0;
                hi.reset();
                ++i_;
            } else if (c == U'+') {
                lo = 1;
                hi.reset();
                ++i_;
            } else if (c == U'?') {
                lo = 0;
                hi = 1;
                ++i_;
            } else if (c == U'{' && try_braces(lo, hi)) {
            } else {
                break;
            }
            if (quantified) fail_api("nothing to repeat");
            quantified = true;
            if (!at_end() && peek() == U'?') ++i_;  // lazy: same language
            Re r;
            r.kind = Re::Kind::Repeat;
            r.min = lo;
            r.max = hi;
            r.kids.push_back(std::move(atom));
            atom = std::move(r);
        }
        return atom;
    }

    Re cls(CpRanges r) {
        Re x;
        x.kind = Re::Kind::Class;
        x.cls = normalize_ranges(std::move(r));
        return x;
    }

    std::uint32_t hex_digits(std::size_t n) {
        std::uint32_t v = 0;
        for (std::size_t k = 0; k < n; ++k) {
            if (at_end()) fail_api("truncated escape");
            const char32_t c = peek();
            std::uint32_t d = 0;
            if (c >= U'0' && c <= U'9') {
                d = static_cast<std::uint32_t>(c - U'0');
            } else if (c >= U'a' && c <= U'f') {
                d = static_cast<std::uint32_t>(c - U'a' + 10);
            } else if (c >= U'A' && c <= U'F') {
                d = static_cast<std::uint32_t>(c - U'A' + 10);
            } else {
                fail_api("bad hex escape");
            }
            v = v * 16 + d;
            ++i_;
        }
        return v;
    }

    // Parses an escape after '\'. Returns a class (single codepoint or \d-style class).
    // `in_class` changes \b to backspace.
    CpRanges parse_escape(bool in_class, bool& is_multi) {
        if (at_end()) fail_api("trailing backslash");
        const char32_t c = peek();
        ++i_;
        is_multi = false;
        const auto one = [](std::uint32_t cp) { return CpRanges{{cp, cp}}; };
        switch (c) {
            case U'd': is_multi = true; return class_digit();
            case U'D': is_multi = true; return complement_ranges(class_digit());
            case U'w': is_multi = true; return class_word();
            case U'W': is_multi = true; return complement_ranges(class_word());
            case U's': is_multi = true; return class_space();
            case U'S': is_multi = true; return complement_ranges(class_space());
            case U't': return one(0x09);
            case U'n': return one(0x0A);
            case U'r': return one(0x0D);
            case U'f': return one(0x0C);
            case U'v': return one(0x0B);
            case U'0':
                if (!at_end() && peek() >= U'0' && peek() <= U'9') unsupported("octal escape");
                return one(0);
            case U'x': return one(hex_digits(2));
            case U'u':
                if (!at_end() && peek() == U'{') unsupported("\\u{...} (unicode mode)");
                {
                    const auto v = hex_digits(4);
                    if (v >= 0xD800 && v <= 0xDFFF) unsupported("surrogate escape");
                    return one(v);
                }
            case U'b':
                if (in_class) return one(0x08);
                unsupported("\\b word boundary");
            case U'B': unsupported("\\B");
            case U'p':
            case U'P': unsupported("\\p{...} property escape");
            case U'c': unsupported("\\c control escape");
            case U'k': unsupported("\\k named backreference");
            default: break;
        }
        if (c >= U'1' && c <= U'9') unsupported("backreference");
        const bool alnum = (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9');
        if (alnum) unsupported(std::format("escape \\{}", static_cast<char>(c)));
        return one(static_cast<std::uint32_t>(c));  // identity escape
    }

    Re parse_class() {
        // at '[' consumed
        bool negate = false;
        if (!at_end() && peek() == U'^') {
            negate = true;
            ++i_;
        }
        CpRanges acc;
        bool first = true;
        while (true) {
            if (at_end()) fail_api("unterminated character class");
            char32_t c = peek();
            if (c == U']' && !first) {
                ++i_;
                break;
            }
            if (c == U']' && first) {  // "[]" matches nothing / "[^]" anything (ECMA)
                ++i_;
                break;
            }
            first = false;
            // one class atom
            CpRanges lo_set;
            bool lo_multi = false;
            std::uint32_t lo_cp = 0;
            ++i_;
            if (c == U'\\') {
                lo_set = parse_escape(true, lo_multi);
                lo_cp = lo_set.front().first;
            } else {
                lo_cp = static_cast<std::uint32_t>(c);
                lo_set = {{lo_cp, lo_cp}};
            }
            // range?
            if (!lo_multi && i_ + 1 < end_ && peek() == U'-' && p_[i_ + 1] != U']') {
                ++i_;
                const char32_t d = peek();
                ++i_;
                std::uint32_t hi_cp = 0;
                if (d == U'\\') {
                    bool hi_multi = false;
                    const auto hs = parse_escape(true, hi_multi);
                    if (hi_multi) fail_api("class escape as range end");
                    hi_cp = hs.front().first;
                } else {
                    hi_cp = static_cast<std::uint32_t>(d);
                }
                if (hi_cp < lo_cp) fail_api("range out of order in character class");
                acc.emplace_back(lo_cp, hi_cp);
            } else {
                acc.insert(acc.end(), lo_set.begin(), lo_set.end());
            }
        }
        CpRanges r = normalize_ranges(acc);
        if (negate) r = complement_ranges(r);
        return cls(std::move(r));
    }

    Re parse_atom(std::size_t depth) {
        const char32_t c = peek();
        ++i_;
        switch (c) {
            case U'(': {
                if (!at_end() && peek() == U'?') {
                    ++i_;
                    if (at_end()) fail_api("bad group");
                    const char32_t g = peek();
                    if (g == U':') {
                        ++i_;
                    } else if (g == U'=' || g == U'!') {
                        unsupported("lookahead");
                    } else if (g == U'<') {
                        ++i_;
                        if (!at_end() && (peek() == U'=' || peek() == U'!')) unsupported("lookbehind");
                        while (!at_end() && peek() != U'>') ++i_;  // named group
                        if (at_end()) fail_api("bad group name");
                        ++i_;
                    } else {
                        unsupported("group modifier");
                    }
                }
                Re inner = parse_alt(depth + 1);
                if (at_end() || peek() != U')') fail_api("missing ')'");
                ++i_;
                return inner;
            }
            case U'[': return parse_class();
            case U'.': return cls(class_dot());
            case U'^': unsupported("'^' anchor not at the start of the pattern");
            case U'$': unsupported("'$' anchor not at the end of the pattern");
            case U'*':
            case U'+':
            case U'?': fail_api("nothing to repeat");
            case U'\\': {
                bool multi = false;
                return cls(parse_escape(false, multi));
            }
            case U'{': {
                --i_;
                std::size_t lo = 0;
                std::optional<std::size_t> hi;
                if (try_braces(lo, hi)) fail_api("nothing to repeat");
                ++i_;
                return cls({{'{', '{'}});
            }
            default: return cls({{static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(c)}});
        }
    }

    std::u32string p_;
    std::size_t max_repeat_;
    std::size_t i_ = 0;
    std::size_t end_ = 0;
};

Sym emit(JsonGrammar& jg, const Re& r) {
    Builder& b = jg.b();
    switch (r.kind) {
        case Re::Kind::Empty: return b.empty();
        case Re::Kind::Class: return jg.json_class(r.cls, "re_class");
        case Re::Kind::Concat: {
            std::vector<Sym> s;
            for (const auto& k : r.kids) s.push_back(emit(jg, k));
            return b.seq(std::move(s), "re_seq");
        }
        case Re::Kind::Alt: {
            std::vector<std::vector<Sym>> a;
            for (const auto& k : r.kids) a.push_back({emit(jg, k)});
            return b.alt(std::move(a), "re_alt");
        }
        case Re::Kind::Repeat: return b.repeat(emit(jg, r.kids[0]), r.min, r.max, "re_rep");
    }
    return b.never();
}

}  // namespace

Sym compile_pattern(JsonGrammar& jg, std::string_view pattern) {
    PatternParser parser(decode_utf8(pattern), jg.opts().max_repeat);
    const std::vector<Branch> branches = parser.parse();
    Builder& b = jg.b();
    std::optional<Sym> any;  // unanchored side: any string content (search semantics)
    const auto any_star = [&] {
        if (!any) any = b.repeat(jg.string_char(), 0, std::nullopt, "re_any");
        return *any;
    };
    std::vector<std::vector<Sym>> alts;
    for (const auto& br : branches) {
        std::vector<Sym> s;
        if (!br.start) s.push_back(any_star());
        s.push_back(emit(jg, br.re));
        if (!br.end) s.push_back(any_star());
        alts.push_back(std::move(s));
    }
    const Sym body = alts.size() == 1 ? b.seq(std::move(alts[0]), "re_branch") : b.alt(std::move(alts), "re_branches");
    return b.seq({b.byte('"'), body, b.byte('"')}, "pattern_string");
}

}  // namespace halo::sampling::detail
