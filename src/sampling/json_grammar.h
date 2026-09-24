#pragma once
// JSON grammar primitives shared by the JSON-Schema and regex compilers (internal).

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "halo/sampling/structured.h"
#include "sampling/grammar_impl.h"

namespace halo::sampling::detail {

/// Memoizing factory for the fixed JSON building blocks. Every symbol it returns
/// matches JSON *text* (bytes as they appear in the document).
class JsonGrammar {
public:
    JsonGrammar(Builder& b, const SchemaOptions& opts) : b_(b), opts_(opts) {}

    Builder& b() noexcept { return b_; }
    const SchemaOptions& opts() const noexcept { return opts_; }

    Sym ws();                 ///< [ \t\n\r]{0,max_whitespace}
    Sym plain_char();         ///< one codepoint that needs no escape (>= 0x20, not " or \)
    Sym string_char();        ///< one codepoint, literal or any JSON escape
    Sym string_any();         ///< "..." any content
    Sym string_bounded(std::size_t min, std::optional<std::size_t> max);  ///< codepoint count
    Sym number();
    Sym integer();
    Sym boolean();
    Sym null();
    Sym value_any(std::size_t level);
    Sym object_any(std::size_t level);
    Sym array_any(std::size_t level);

    /// One codepoint from `cps`, JSON-encoded canonically (escapes only where required).
    Sym json_class(const CpRanges& cps, std::string name);

private:
    Builder& b_;
    SchemaOptions opts_;
    std::optional<Sym> ws_, plain_, char_, string_, number_, integer_, boolean_, null_;
    std::map<std::size_t, Sym> value_, object_, array_;
};

/// Compiles an ECMA-262 `pattern` (subset in structured.h) into a symbol matching the
/// complete JSON string literal, quotes included. Throws Error(Unsupported) for
/// unsupported features and Error(Api) for malformed patterns.
Sym compile_pattern(JsonGrammar& jg, std::string_view pattern);

/// Decodes UTF-8 (already validated by the JSON parser) into codepoints.
std::u32string decode_utf8(std::string_view s);

}  // namespace halo::sampling::detail
