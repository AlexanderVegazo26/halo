// JSON-Schema -> byte grammar tests (WS-H, TRD §25). No reference data needed.
//
// Hand-written accept/reject documents per feature (random generation alone cannot show
// that a grammar rejects nothing valid), typed errors for unsupported / malformed schemas,
// a differential check of the pattern compiler against std::regex (ECMAScript), and seeded
// byte-level random walks whose outputs are validated by the independent validator in
// json_check.h.

#include <gtest/gtest.h>

#include <deque>
#include <optional>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "halo/core/error.h"
#include "halo/sampling/structured.h"
#include "json_check.h"
#include "sampling/grammar_impl.h"

namespace {

using halo::Error;
using halo::ErrorCode;
using halo::sampling::ByteMatcher;
using halo::sampling::Grammar;
using halo::sampling::SchemaOptions;
using halo::sampling::test::OJson;
using halo::sampling::test::validate_text;

bool accepts(const Grammar& g, std::string_view doc) {
    ByteMatcher m(g);
    return m.accept_bytes(doc) && m.is_accepting();
}

ErrorCode compile_error(std::string_view schema, const SchemaOptions& o = {}) {
    try {
        (void)Grammar::from_json_schema(schema, o);
    } catch (const Error& e) {
        return e.code();
    }
    ADD_FAILURE() << "schema compiled but should not: " << schema;
    return ErrorCode::Cancelled;
}

void expect_language(const Grammar& g, const std::vector<std::string>& good, const std::vector<std::string>& bad) {
    for (const auto& d : good) EXPECT_TRUE(accepts(g, d)) << "should accept: " << d;
    for (const auto& d : bad) EXPECT_FALSE(accepts(g, d)) << "should reject: " << d;
}

TEST(JsonGrammar, AnyValueAcceptsValidJsonRejectsInvalid) {
    const auto g = Grammar::any_json_value();
    expect_language(
        g,
        {"null", "true", "false", "0", "-0", "12.5e-3", "1E+2", "-7", R"("")",
         R"("aé\"\\\/\b\f\n\r\t")", R"("😀")", "\"\xC3\xA9\xE6\x97\xA5\xF0\x9F\x98\x80\"", "[]",
         R"([1, "a", {"k": [null]}])", "{}", R"({"a":{"b":{}}})", " { \"a\" : 1 } ", "\t[\n1\r]"},
        {"", "01", "1.", ".5", "-", "+1", "1e", "1e+", R"("\ud800")", R"("\udc00")", R"("\ud800A")", R"("\x")",
         R"("a)", "\"\n\"", "[1,]", R"({"a":1,})", "{a:1}", R"({"a"})", "tru", "nul", "'a'", "[1 2]", "\"\xFF\"",
         "\"\xC0\xAF\"", "\"\xED\xA0\x80\"", "\"\xF4\x90\x80\x80\"", "\"\xE6\x97\"", "[", "{\"a\":1}}", "nulll"});
    // Whitespace per gap is bounded (generation policy): 16 bytes ok, 17 not.
    EXPECT_TRUE(accepts(g, std::string(16, ' ') + "1"));
    EXPECT_FALSE(accepts(g, std::string(17, ' ') + "1"));
}

TEST(JsonGrammar, JsonObjectModeAcceptsOnlyObjects) {
    const auto g = Grammar::any_json_object();
    expect_language(g, {"{}", R"({"a":[1,{"b":null}],"c":"d"})"}, {"[]", "1", R"("s")", "null", "{"});
}

TEST(JsonSchema, ObjectRequiredOptionalNoAdditional) {
    const auto g = Grammar::from_json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
            "required":["name"],"additionalProperties":false})");
    expect_language(g, {R"({"name":"x"})", R"({"name":"x","age":3})", R"( { "name" : "" , "age" : -12 } )"},
                    {"{}", R"({"age":3})", R"({"name":1})", R"({"name":"x","extra":1})", R"({"name":"x","age":1.5})",
                     R"({"name":"x",})", R"({"name":"x","name":"y"})"});
    // Generation policy, not schema semantics: declared properties come in schema order.
    EXPECT_FALSE(accepts(g, R"({"age":3,"name":"x"})"));
}

TEST(JsonSchema, AdditionalPropertiesDefaultTrueNeverShadowsDeclared) {
    const auto g = Grammar::from_json_schema(R"({"type":"object","properties":{"a":{"type":"boolean"}}})");
    expect_language(g, {R"({"a":true,"zz":[1]})", R"({"b":1})", "{}", R"({"ab":1})", R"({"":1})"},
                    {R"({"a":1})", R"({"a":true,"a":false})", R"({"b":1,"a":true})"});
    const auto h = Grammar::from_json_schema(R"({"type":"object","additionalProperties":{"type":"integer"}})");
    expect_language(h, {R"({"x":1,"y":-2})", "{}"}, {R"({"x":"s"})", R"({"x":1.5})"});
}

TEST(JsonSchema, EnumConstAndTypeLists) {
    const auto g = Grammar::from_json_schema(R"({"enum":["red","green",1,null,{"a":1}]})");
    expect_language(g, {R"("red")", "1", "null", R"({"a":1})", R"( "green" )"}, {R"("blue")", "2", R"("re")", "true"});
    const auto c = Grammar::from_json_schema(R"({"const":"x"})");
    expect_language(c, {R"("x")"}, {R"("y")", "1"});
    const auto t = Grammar::from_json_schema(R"({"type":["string","null"]})");
    expect_language(t, {R"("s")", "null"}, {"1", "true"});
    const auto n = Grammar::from_json_schema(R"({"type":"number"})");
    expect_language(n, {"1", "-0.5", "1e10", "3.25E-2"}, {R"("1")", "NaN", "Infinity"});
    const auto i = Grammar::from_json_schema(R"({"type":"integer"})");
    expect_language(i, {"0", "-3", "123456789012345678901234567890"}, {"1.0", "1e2", "-"});
    // enum filtered by type.
    const auto f = Grammar::from_json_schema(R"({"type":"string","enum":["a",1]})");
    expect_language(f, {R"("a")"}, {"1"});
}

TEST(JsonSchema, ArraysItemsAndBounds) {
    const auto g = Grammar::from_json_schema(R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3})");
    expect_language(g, {"[1,2]", "[1, 2, 3]", "[ 1 ,2 ]"}, {"[1]", "[1,2,3,4]", "[]", R"(["a",1])", "[1,2,]"});
    const auto e = Grammar::from_json_schema(R"({"type":"array","maxItems":0})");
    expect_language(e, {"[]", "[ ]"}, {"[1]"});
    const auto any = Grammar::from_json_schema(R"({"type":"array"})");
    expect_language(any, {"[]", R"([1,"a",[{}]])"}, {"{}"});
}

TEST(JsonSchema, StringLengthCountsCodepoints) {
    const auto g = Grammar::from_json_schema(R"({"type":"string","minLength":2,"maxLength":3})");
    expect_language(g,
                    {R"("ab")", R"("abc")", "\"\xC3\xA9\xE6\x97\xA5\"", R"("éx")", "\"\xF0\x9F\x98\x80\xF0\x9F\x98\x80\"",
                     R"("😀a")", R"("\n\t")"},
                    {R"("a")", R"("abcd")", R"("")", R"("😀")"});
}

TEST(JsonSchema, AnyOfOneOfAndRefs) {
    const auto a = Grammar::from_json_schema(R"({"anyOf":[{"type":"integer"},{"type":"string","maxLength":1}]})");
    expect_language(a, {"5", R"("a")"}, {R"("ab")", "true"});
    const auto o = Grammar::from_json_schema(
        R"({"oneOf":[{"type":"object","properties":{"kind":{"const":"a"},"x":{"type":"integer"}},"required":["kind"],"additionalProperties":false},
                     {"type":"object","properties":{"kind":{"const":"b"},"y":{"type":"string"}},"required":["kind"],"additionalProperties":false}]})");
    expect_language(o, {R"({"kind":"a","x":1})", R"({"kind":"b","y":"s"})"}, {R"({"kind":"a","y":"s"})", R"({"kind":"c"})"});
    const auto r = Grammar::from_json_schema(
        R"({"$defs":{"pos":{"type":"integer"}},"type":"object","properties":{"p":{"$ref":"#/$defs/pos"}},
            "required":["p"],"additionalProperties":false})");
    expect_language(r, {R"({"p":4})"}, {R"({"p":"4"})"});
    const auto tree = Grammar::from_json_schema(
        R"({"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},
            "required":["v"],"additionalProperties":false}},"$ref":"#/$defs/node"})");
    expect_language(tree, {R"({"v":1,"kids":[{"v":2,"kids":[]},{"v":3}]})"}, {R"({"v":1,"kids":[{"kids":[]}]})"});
    const auto legacy = Grammar::from_json_schema(R"({"definitions":{"b":{"type":"boolean"}},"$ref":"#/definitions/b"})");
    expect_language(legacy, {"true"}, {"1"});
    EXPECT_TRUE(accepts(Grammar::from_json_schema("true"), R"({"any":[1]})"));
    EXPECT_TRUE(accepts(Grammar::from_json_schema("{}"), "3"));
}

TEST(JsonSchema, UnsupportedFeaturesAreTypedErrors) {
    for (const char* s : {R"({"type":"string","format":"date"})", R"({"minimum":1})", R"({"allOf":[{}]})",
                          R"({"not":{}})", R"({"patternProperties":{}})", R"({"items":[{}]})", R"({"if":{}})",
                          R"({"$ref":"http://example.com/s"})", R"({"$ref":"#/properties/a"})",
                          R"({"oneOf":[{"type":"integer"},{"type":"number"}]})",
                          R"({"oneOf":[{"type":"string"},{"type":"string","maxLength":2}]})",
                          R"({"type":"string","pattern":"(?=a)b"})", R"({"type":"string","pattern":"(a)\\1"})",
                          R"({"type":"string","pattern":"\\bword"})", R"({"type":"string","pattern":"\\p{L}"})",
                          R"({"type":"string","pattern":"a^b"})", R"({"type":"string","pattern":"a$b"})",
                          R"({"type":"string","pattern":"(?<=a)b"})", R"({"type":"string","pattern":"\\u{1F600}"})",
                          R"({"type":"string","pattern":"^a$","minLength":1})", R"({"type":"array","minItems":100000})",
                          R"({"$defs":{"a":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"})",
                          R"({"$defs":{"a":{"anyOf":[{"$ref":"#/$defs/a"},{"type":"null"}]}},"$ref":"#/$defs/a"})"}) {
        EXPECT_EQ(compile_error(s), ErrorCode::Unsupported) << s;
    }
}

TEST(JsonSchema, MalformedOrUnsatisfiableSchemasAreApiErrors) {
    for (const char* s : {"{", "\"string\"", "[]", R"({"type":"strin"})", R"({"minLength":-1})", R"({"minLength":1.5})",
                          R"({"type":"array","minItems":3,"maxItems":1})", R"({"required":["x"],"additionalProperties":false})",
                          R"({"$ref":"#/$defs/missing"})", R"({"type":"string","pattern":"("})",
                          R"({"type":"string","pattern":"[z-a]"})", R"({"type":"string","pattern":"a**"})",
                          R"({"enum":[]})", R"({"anyOf":[]})", "false", R"({"type":"string","maxLength":1,"minLength":2})"}) {
        EXPECT_EQ(compile_error(s), ErrorCode::Api) << s;
    }
}

TEST(JsonSchema, AdversarialNestingAndSize) {
    // 20 nested arrays fit the default max_nesting (32); 40 cannot and must be rejected.
    const auto nested = [](int depth) {
        std::string s = R"({"type":"integer"})";
        for (int i = 0; i < depth; ++i) s = R"({"type":"array","minItems":1,"maxItems":1,"items":)" + s + "}";
        return s;
    };
    const auto g = Grammar::from_json_schema(nested(20));
    EXPECT_TRUE(accepts(g, std::string(20, '[') + "7" + std::string(20, ']')));
    EXPECT_FALSE(accepts(g, std::string(19, '[') + "7" + std::string(19, ']')));
    EXPECT_EQ(compile_error(nested(40)), ErrorCode::Api);
    // Schema-document nesting limit (anyOf chains do not consume container levels).
    std::string deep = R"({"type":"null"})";
    for (int i = 0; i < 200; ++i) deep = R"({"anyOf":[)" + deep + "]}";
    EXPECT_EQ(compile_error(deep), ErrorCode::Unsupported);
    // Large enum: 5000 literals compile to a trie.
    std::string e = R"({"enum":[)";
    for (int i = 0; i < 5000; ++i) e += (i ? "," : "") + std::string("\"v") + std::to_string(i) + "\"";
    e += "]}";
    const auto ge = Grammar::from_json_schema(e);
    expect_language(ge, {R"("v0")", R"("v4999")", R"("v123")"}, {R"("v5000")", R"("v")", R"("v01")"});
    // Element budget is enforced with a typed error.
    SchemaOptions tiny;
    tiny.max_grammar_elements = 64;
    EXPECT_EQ(compile_error(e, tiny), ErrorCode::Unsupported);
    // Recursion is unrolled to the nesting limit: a deep but finite tree still compiles.
    const auto tree = Grammar::from_json_schema(
        R"({"$defs":{"n":{"type":"array","items":{"$ref":"#/$defs/n"}}},"$ref":"#/$defs/n"})");
    EXPECT_TRUE(accepts(tree, std::string(32, '[') + std::string(32, ']')));
    EXPECT_FALSE(accepts(tree, std::string(33, '[') + std::string(33, ']')));
}

// ---- pattern differential vs std::regex ---------------------------------------------

std::string json_quote(const std::string& s) { return OJson(s).dump(); }

/// Every string of length <= 4 over a small alphabet with two digits, letters, '-', ' ', '.'.
const std::vector<std::string>& enumerated_strings() {
    static const std::vector<std::string> v = [] {
        const std::string alphabet = "abcd-17x .";
        std::vector<std::string> s{""};
        for (std::size_t len = 1; len <= 4; ++len) {
            const std::size_t base = s.size();
            for (std::size_t i = 0; i < base; ++i) {
                if (s[i].size() != len - 1) continue;
                for (const char c : alphabet) s.push_back(s[i] + c);
            }
        }
        return s;
    }();
    return v;
}

/// Our pattern `ours` must accept exactly the strings std::regex (ECMAScript) finds with
/// `ref`. `ref` differs from `ours` only where libstdc++'s std::regex lacks an ECMA-262
/// feature (named groups), and then only by removing the group name, which does not change
/// the matched language.
void expect_same_language(const std::string& ours, const std::string& ref) {
    const auto g = Grammar::from_json_schema(OJson{{"type", "string"}, {"pattern", ours}}.dump());
    const std::regex re(ref, std::regex::ECMAScript);
    std::size_t matches = 0;
    for (const auto& s : enumerated_strings()) {
        const bool want = std::regex_search(s, re);
        matches += want ? 1 : 0;
        ASSERT_EQ(accepts(g, json_quote(s)), want) << "pattern " << ours << " string '" << s << "'";
    }
    EXPECT_GT(matches, 0U) << ours;  // the pattern is not trivially empty over the alphabet
}

TEST(JsonSchemaPattern, MatchesStdRegexOnEnumeratedStrings) {
    ASSERT_GT(enumerated_strings().size(), 11000U);
    for (const char* pat : {"^[a-c]+$", "^a{2,3}b?$", "^(ab|c)*d$", "ab", R"(^\d{2}-\d$)", R"(^\d\D\d$)",
                            "^[^b]c*$", "x$", "^(?:a|bc){1,2}$", "^.b$", R"(^[\w-]+$)", "^a|b$", R"(^\s?a$)",
                            "^(a+)+$", "^[a-]*1$", R"(^\x61b$)", "^a{2,}$", "^a+?$", R"(^\-\.$)", "c",
                            R"(^[1-7]{2}\.?$)"}) {
        expect_same_language(pat, pat);
    }
}

TEST(JsonSchemaPattern, NamedGroupsMatchTheUnnamedLanguage) {
    // libstdc++'s std::regex throws on (?<name>...), so the reference is the same pattern
    // with the name removed.
    expect_same_language("^(?<n>a|b)c$", "^(a|b)c$");
    expect_same_language(R"(^(?<d>\d)+-(?<w>x|7)$)", R"(^(\d)+-(x|7)$)");
}

TEST(JsonSchemaPattern, UnicodeAndEscapesInsideStrings) {
    const auto g = Grammar::from_json_schema(R"({"type":"string","pattern":"^é+$"})");
    expect_language(g, {"\"\xC3\xA9\xC3\xA9\""}, {R"("e")", R"("")", "\"\xC3\""});
    const auto r = Grammar::from_json_schema(R"({"type":"string","pattern":"^[α-ω]{2}$"})");
    expect_language(r, {"\"\xCE\xB1\xCF\x89\""}, {"\"\xCE\x91\xCF\x89\"", R"("ab")"});
    // A pattern class containing '"' or a control char produces the JSON escape.
    const auto q = Grammar::from_json_schema(R"({"type":"string","pattern":"^[\"\\n]$"})");
    expect_language(q, {R"("\"")", R"("\n")"}, {"\"\"\"", "\"\n\""});
}

// ---- byte-level random walks validated independently ----------------------------------

/// Shortest byte string leading from `from` to an accepting state (breadth-first over the
/// interned matcher states, bytes tried in ascending order). nullopt if none is found
/// within `max_states` visited states — for these grammars that means a live state that
/// can never complete, which is itself a defect.
std::optional<std::string> shortest_completion(halo::sampling::detail::Runtime& rt, std::int32_t from,
                                               std::size_t max_states = 200000) {
    using halo::sampling::detail::byteset_has;
    struct Prev {
        std::int32_t state;
        std::uint8_t byte;
    };
    std::unordered_map<std::int32_t, Prev> prev{{from, {from, 0}}};
    std::deque<std::int32_t> queue{from};
    while (!queue.empty() && prev.size() <= max_states) {
        const std::int32_t s = queue.front();
        queue.pop_front();
        if (rt.accepting(s)) {
            std::string path;
            for (std::int32_t t = s; t != from; t = prev.at(t).state) path.push_back(static_cast<char>(prev.at(t).byte));
            return std::string(path.rbegin(), path.rend());
        }
        const auto next = rt.next_bytes(s);
        for (unsigned b = 0; b < 256; ++b) {
            if (!byteset_has(next, static_cast<std::uint8_t>(b))) continue;
            const std::int32_t t = rt.step(s, static_cast<std::uint8_t>(b));
            if (t == halo::sampling::detail::kDead || prev.contains(t)) continue;
            prev.emplace(t, Prev{s, static_cast<std::uint8_t>(b)});
            queue.push_back(t);
        }
    }
    return std::nullopt;
}

/// Random walk over the byte automaton: random allowed bytes for `budget` steps, then the
/// shortest completion (so unanchored patterns and deep nesting close too). Returns the
/// document.
std::string random_document(const Grammar& g, std::mt19937_64& rng, std::size_t budget) {
    using halo::sampling::detail::byteset_has;
    halo::sampling::detail::Runtime rt(g.data());
    std::int32_t s = rt.initial();
    std::string out;
    for (std::size_t step = 0; step < budget; ++step) {
        if (rt.accepting(s) && (!rt.can_continue(s) || rng() % 8 == 0)) return out;
        const auto next = rt.next_bytes(s);
        std::vector<std::uint8_t> opts;
        for (unsigned b = 0; b < 256; ++b) {
            if (byteset_has(next, static_cast<std::uint8_t>(b))) opts.push_back(static_cast<std::uint8_t>(b));
        }
        if (opts.empty()) {
            ADD_FAILURE() << "dead state reached after '" << out << "'";
            return out;
        }
        const std::uint8_t b = opts[rng() % opts.size()];
        s = rt.step(s, b);
        if (s == halo::sampling::detail::kDead) {
            ADD_FAILURE() << "next_bytes offered a byte that kills the state";
            return out;
        }
        out.push_back(static_cast<char>(b));
    }
    const auto tail = shortest_completion(rt, s);
    if (!tail) {
        ADD_FAILURE() << "no completion reachable after '" << out << "'";
        return out;
    }
    for (const char c : *tail) s = rt.step(s, static_cast<std::uint8_t>(c));
    EXPECT_TRUE(s != halo::sampling::detail::kDead && rt.accepting(s)) << "completion path did not accept";
    return out + *tail;
}

const std::vector<std::string>& walk_schemas() {
    static const std::vector<std::string> v{
        R"({"type":"object","properties":{"name":{"type":"string","maxLength":8},"age":{"type":"integer"},"tags":{"type":"array","items":{"enum":["a","b","c"]},"maxItems":3}},"required":["name"],"additionalProperties":false})",
        R"({"type":"object","properties":{"id":{"type":"string","pattern":"^[A-Z]{2}-\\d{3}$"},"ok":{"type":"boolean"}},"required":["id","ok"]})",
        R"({"$defs":{"n":{"type":"object","properties":{"v":{"type":"number"},"kids":{"type":"array","items":{"$ref":"#/$defs/n"},"maxItems":2}},"required":["v"],"additionalProperties":false}},"$ref":"#/$defs/n"})",
        R"({"anyOf":[{"type":"null"},{"type":"array","items":{"type":"string","minLength":1,"maxLength":3},"minItems":1}]})",
        R"({"oneOf":[{"type":"object","properties":{"t":{"const":"x"},"n":{"type":"integer"}},"required":["t","n"],"additionalProperties":false},{"type":"object","properties":{"t":{"const":"y"}},"required":["t"]}]})",
        R"({"type":"object","additionalProperties":{"type":["integer","null"]}})",
        R"({"type":"string","pattern":"[a-f]+@x"})",
    };
    return v;
}

TEST(JsonSchemaWalk, ValidatorAcceptsOverflowingNumbersWithoutMaskingTypeErrors) {
    // The grammar follows RFC 8259, which puts no range on numbers; the test validator must
    // not reject those documents, but an overflowing number is still only a number.
    const auto num = OJson::parse(R"({"type":"number"})");
    const auto str = OJson::parse(R"({"type":"string"})");
    const auto grammar = Grammar::from_json_schema(R"({"type":"number"})");
    for (const char* doc : {"948131206E637", "-1e999", "1" /* control */}) {
        ASSERT_TRUE(accepts(grammar, doc)) << doc;
        EXPECT_EQ(validate_text(num, doc), "") << doc;
        EXPECT_NE(validate_text(str, doc), "") << doc;
    }
    const std::string big_int(400, '9');
    EXPECT_EQ(validate_text(OJson::parse(R"({"type":"integer"})"), big_int), "");
    EXPECT_TRUE(accepts(Grammar::from_json_schema(R"({"type":"integer"})"), big_int));
    // Digits inside strings are never rewritten.
    EXPECT_EQ(validate_text(OJson::parse(R"({"const":"1e999"})"), R"("1e999")"), "");
    EXPECT_NE(validate_text(OJson::parse(R"({"const":"1e999"})"), R"("1e308")"), "");
}

TEST(JsonSchemaWalk, RandomByteWalksAlwaysValidate) {
    std::mt19937_64 rng(2024);
    std::size_t docs = 0;
    for (const auto& schema_text : walk_schemas()) {
        const auto g = Grammar::from_json_schema(schema_text);
        const auto schema = OJson::parse(schema_text);
        for (int run = 0; run < 150; ++run) {
            const auto doc = random_document(g, rng, 10 + static_cast<std::size_t>(run % 60));
            const auto err = validate_text(schema, doc);
            ASSERT_TRUE(err.empty()) << schema_text << "\n" << doc << "\n" << err;
            ++docs;
        }
    }
    EXPECT_EQ(docs, walk_schemas().size() * 150);
    // json_object / any value.
    const auto obj = Grammar::any_json_object();
    for (int run = 0; run < 150; ++run) {
        const auto doc = random_document(obj, rng, 40);
        OJson v;
        ASSERT_NO_THROW(v = OJson::parse(doc)) << doc;
        ASSERT_TRUE(v.is_object()) << doc;
    }
}

}  // namespace
