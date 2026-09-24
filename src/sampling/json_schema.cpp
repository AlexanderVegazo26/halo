// JSON Schema (subset) -> byte-level grammar. Subset and generation policy: structured.h.

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "halo/core/error.h"
#include "halo/sampling/structured.h"
#include "sampling/grammar_impl.h"
#include "sampling/json_grammar.h"

namespace halo::sampling {

namespace {

using Json = nlohmann::ordered_json;
using detail::Builder;
using detail::JsonGrammar;
using detail::Sym;

// Type bits. `number` is INT|NONINT, so integer-vs-number overlap is detected.
enum : unsigned {
    kNull = 1U,
    kBool = 2U,
    kObj = 4U,
    kArr = 8U,
    kStr = 16U,
    kInt = 32U,
    kNonInt = 64U,
    kAll = 127U,
};

unsigned type_bit(std::string_view t) {
    if (t == "null") return kNull;
    if (t == "boolean") return kBool;
    if (t == "object") return kObj;
    if (t == "array") return kArr;
    if (t == "string") return kStr;
    if (t == "integer") return kInt;
    if (t == "number") return kInt | kNonInt;
    throw_error(ErrorCode::Api, "structured output: unknown type '{}'", t);
}

bool is_integral(const Json& v) {
    if (v.is_number_integer() || v.is_number_unsigned()) return true;
    if (v.is_number_float()) {
        const double d = v.get<double>();
        return std::isfinite(d) && std::floor(d) == d;
    }
    return false;
}

unsigned literal_bit(const Json& v) {
    if (v.is_null()) return kNull;
    if (v.is_boolean()) return kBool;
    if (v.is_object()) return kObj;
    if (v.is_array()) return kArr;
    if (v.is_string()) return kStr;
    return is_integral(v) ? kInt : kNonInt;
}

bool literal_has_type(const Json& v, unsigned types) {
    const unsigned b = literal_bit(v);
    return (b & types) != 0;
}

const std::set<std::string, std::less<>>& annotation_keywords() {
    static const std::set<std::string, std::less<>> k{"title",  "description", "default",  "examples", "$schema",
                                                      "$id",    "$comment",    "deprecated", "readOnly", "writeOnly"};
    return k;
}
const std::set<std::string, std::less<>>& supported_keywords() {
    static const std::set<std::string, std::less<>> k{
        "type",      "properties", "required", "additionalProperties", "enum",      "const",
        "items",     "anyOf",      "oneOf",    "minItems",             "maxItems",  "minLength",
        "maxLength", "pattern",    "$ref",     "$defs",                "definitions"};
    return k;
}
bool is_meta(std::string_view k) {
    return annotation_keywords().contains(k) || k == "$defs" || k == "definitions";
}

struct Kind {
    unsigned types = kAll;
    std::optional<std::vector<Json>> literals;
    std::map<std::string, std::vector<Json>> discriminators;  // object branches only
};

class SchemaCompiler {
public:
    SchemaCompiler(const Json& root, JsonGrammar& jg) : root_(root), jg_(jg), b_(jg.b()) {}

    Sym compile(const Json& s, std::size_t level, std::size_t sdepth) {
        HALO_CHECK(sdepth <= jg_.opts().max_schema_depth, ErrorCode::Unsupported,
                   "structured output: schema nesting exceeds {}", jg_.opts().max_schema_depth);
        if (s.is_boolean()) return s.get<bool>() ? jg_.value_any(level) : b_.never("false_schema");
        HALO_CHECK(s.is_object(), ErrorCode::Api, "structured output: a schema must be an object or a boolean");
        const auto key = std::make_pair(&s, level);
        if (const auto it = memo_.find(key); it != memo_.end()) return it->second;
        HALO_CHECK(!in_progress_.contains(key), ErrorCode::Unsupported,
                   "structured output: recursive $ref that does not pass through an object or array");
        for (const auto& [k, v] : s.items()) {
            if (annotation_keywords().contains(k)) continue;
            HALO_CHECK(supported_keywords().contains(k), ErrorCode::Unsupported,
                       "structured output: JSON Schema keyword '{}' is not supported", k);
        }
        in_progress_.insert(key);
        const Sym r = body(s, level, sdepth);
        in_progress_.erase(key);
        memo_.emplace(key, r);
        return r;
    }

    const Json& resolve(const Json& s) const {
        const auto& ref = s.at("$ref");
        HALO_CHECK(ref.is_string(), ErrorCode::Api, "structured output: $ref must be a string");
        const auto r = ref.get<std::string>();
        if (r == "#") return root_;
        for (const std::string_view prefix : {"#/$defs/", "#/definitions/"}) {
            if (r.starts_with(prefix)) {
                const std::string name = r.substr(prefix.size());
                HALO_CHECK(name.find('/') == std::string::npos && name.find('~') == std::string::npos,
                           ErrorCode::Unsupported, "structured output: $ref '{}' is not supported", r);
                const std::string container(prefix.substr(2, prefix.size() - 3));
                HALO_CHECK(root_.is_object() && root_.contains(container) && root_.at(container).is_object() &&
                               root_.at(container).contains(name),
                           ErrorCode::Api, "structured output: $ref '{}' not found", r);
                return root_.at(container).at(name);
            }
        }
        throw_error(ErrorCode::Unsupported, "structured output: $ref '{}' is not supported (only #, #/$defs/, "
                                            "#/definitions/)", r);
    }

private:
    void require_only(const Json& s, std::string_view owner, std::initializer_list<std::string_view> allowed) const {
        for (const auto& [k, v] : s.items()) {
            if (is_meta(k) || k == owner) continue;
            if (std::find(allowed.begin(), allowed.end(), k) != allowed.end()) continue;
            throw_error(ErrorCode::Unsupported, "structured output: '{}' combined with '{}' is not supported", owner, k);
        }
    }

    std::optional<std::size_t> count(const Json& s, std::string_view k) const {
        if (!s.contains(k)) return std::nullopt;
        const auto& v = s.at(k);
        HALO_CHECK(is_integral(v) && v.get<double>() >= 0.0, ErrorCode::Api,
                   "structured output: {} must be a non-negative integer", k);
        const double d = v.get<double>();
        HALO_CHECK(d <= static_cast<double>(jg_.opts().max_repeat), ErrorCode::Unsupported,
                   "structured output: {} = {} exceeds the supported maximum {}", k, d, jg_.opts().max_repeat);
        return static_cast<std::size_t>(d);
    }

    unsigned declared_types(const Json& s) const {
        if (!s.contains("type")) return 0;
        const auto& t = s.at("type");
        if (t.is_string()) return type_bit(t.get<std::string>());
        HALO_CHECK(t.is_array() && !t.empty(), ErrorCode::Api, "structured output: 'type' must be a string or array");
        unsigned m = 0;
        for (const auto& e : t) {
            HALO_CHECK(e.is_string(), ErrorCode::Api, "structured output: 'type' entries must be strings");
            m |= type_bit(e.get<std::string>());
        }
        return m;
    }

    static unsigned inferred_types(const Json& s) {
        unsigned m = 0;
        for (const char* k : {"properties", "required", "additionalProperties"}) {
            if (s.contains(k)) m |= kObj;
        }
        for (const char* k : {"items", "minItems", "maxItems"}) {
            if (s.contains(k)) m |= kArr;
        }
        for (const char* k : {"minLength", "maxLength", "pattern"}) {
            if (s.contains(k)) m |= kStr;
        }
        return m;
    }

    Sym body(const Json& s, std::size_t level, std::size_t sdepth) {
        if (s.contains("$ref")) {
            require_only(s, "$ref", {});
            return compile(resolve(s), level, sdepth + 1);
        }
        if (s.contains("enum") || s.contains("const")) return literals(s);
        for (const char* comb : {"anyOf", "oneOf"}) {
            if (!s.contains(comb)) continue;
            require_only(s, comb, {});
            const auto& branches = s.at(comb);
            HALO_CHECK(branches.is_array() && !branches.empty(), ErrorCode::Api,
                       "structured output: {} must be a non-empty array", comb);
            if (std::string_view(comb) == "oneOf") check_disjoint(branches);
            std::vector<std::vector<Sym>> alts;
            for (const auto& br : branches) alts.push_back({compile(br, level, sdepth + 1)});
            return b_.alt(std::move(alts), comb);
        }
        unsigned types = declared_types(s);
        if (types == 0) types = inferred_types(s);
        if (types == 0) return jg_.value_any(level);
        std::vector<std::vector<Sym>> alts;
        if (types & kNull) alts.push_back({jg_.null()});
        if (types & kBool) alts.push_back({jg_.boolean()});
        if ((types & kNonInt) != 0) {
            alts.push_back({jg_.number()});
        } else if (types & kInt) {
            alts.push_back({jg_.integer()});
        }
        if (types & kStr) alts.push_back({string(s)});
        if (types & kArr) alts.push_back({array(s, level, sdepth)});
        if (types & kObj) alts.push_back({object(s, level, sdepth)});
        if (alts.size() == 1) return alts[0][0];
        return b_.alt(std::move(alts), "types");
    }

    Sym literals(const Json& s) {
        require_only(s, s.contains("enum") ? "enum" : "const", {"type", "enum", "const"});
        std::vector<Json> vals;
        if (s.contains("enum")) {
            const auto& e = s.at("enum");
            HALO_CHECK(e.is_array() && !e.empty(), ErrorCode::Api, "structured output: enum must be a non-empty array");
            vals.assign(e.begin(), e.end());
        }
        if (s.contains("const")) {
            const Json& c = s.at("const");
            if (s.contains("enum")) {
                std::erase_if(vals, [&](const Json& v) { return v != c; });
            } else {
                vals = {c};
            }
        }
        if (const unsigned t = declared_types(s); t != 0) {
            std::erase_if(vals, [&](const Json& v) { return !literal_has_type(v, t); });
        }
        std::vector<std::string> texts;
        texts.reserve(vals.size());
        for (const auto& v : vals) texts.push_back(v.dump());
        return b_.literal_set(std::move(texts), "enum");
    }

    Sym string(const Json& s) {
        const auto min = count(s, "minLength");
        const auto max = count(s, "maxLength");
        if (s.contains("pattern")) {
            HALO_CHECK(s.at("pattern").is_string(), ErrorCode::Api, "structured output: pattern must be a string");
            HALO_CHECK(!min && !max, ErrorCode::Unsupported,
                       "structured output: pattern combined with minLength/maxLength is not supported");
            return detail::compile_pattern(jg_, s.at("pattern").get<std::string>());
        }
        if (max && *max < min.value_or(0)) return b_.never("string_unsat");
        return jg_.string_bounded(min.value_or(0), max);
    }

    Sym array(const Json& s, std::size_t level, std::size_t sdepth) {
        if (level >= jg_.opts().max_nesting) return b_.never("array_too_deep");
        const std::size_t min = count(s, "minItems").value_or(0);
        std::optional<std::size_t> max = count(s, "maxItems");
        Sym item = jg_.value_any(level + 1);
        if (s.contains("items")) {
            const auto& it = s.at("items");
            HALO_CHECK(!it.is_array(), ErrorCode::Unsupported,
                       "structured output: 'items' as an array (tuple validation) is not supported");
            if (it.is_boolean() && !it.get<bool>()) {
                max = 0;
            } else {
                item = compile(it, level + 1, sdepth + 1);
            }
        }
        if (max && *max < min) return b_.never("array_unsat");
        const Sym ws = jg_.ws();
        std::vector<std::vector<Sym>> body;
        if (min == 0) body.push_back({b_.byte(']')});
        if (!max || *max > 0) {
            const Sym sep = b_.seq({ws, b_.byte(','), ws}, "item_sep");
            body.push_back({b_.repeat(item, std::max<std::size_t>(min, 1), max, "items", sep), ws, b_.byte(']')});
        }
        return b_.seq({b_.byte('['), ws, b_.alt(std::move(body), "array_body")}, "array");
    }

    // Key strings that are not equal to any declared name. Extra keys use unescaped
    // characters only, so a name needing an escape can never collide and is skipped.
    Sym other_key(const std::vector<std::string>& names) {
        struct Node {
            std::map<std::uint32_t, std::size_t> kids;
            bool terminal = false;
        };
        std::vector<Node> trie(1);
        for (const auto& n : names) {
            const std::u32string cps = detail::decode_utf8(n);
            if (std::any_of(cps.begin(), cps.end(), [](char32_t c) { return c < 0x20 || c == U'"' || c == U'\\'; })) {
                continue;
            }
            std::size_t cur = 0;
            for (const char32_t c : cps) {
                const auto cp = static_cast<std::uint32_t>(c);
                auto it = trie[cur].kids.find(cp);
                if (it == trie[cur].kids.end()) {
                    trie.emplace_back();
                    it = trie[cur].kids.emplace(cp, trie.size() - 1).first;
                }
                cur = it->second;
            }
            trie[cur].terminal = true;
        }
        const Sym plain_star = b_.repeat(jg_.plain_char(), 0, std::nullopt, "key_free");
        const detail::CpRanges plain{{0x20, 0x21}, {0x23, 0x5B}, {0x5D, 0x10FFFF}};
        std::vector<Sym> rule_of(trie.size());
        for (std::size_t i = trie.size(); i-- > 0;) {  // children have larger indices
            std::vector<std::vector<Sym>> alts;
            if (!trie[i].terminal) alts.push_back({b_.byte('"')});
            detail::CpRanges used;
            for (const auto& [cp, child] : trie[i].kids) {
                std::string enc;
                detail::append_utf8(enc, cp);
                alts.push_back({b_.literal(enc), rule_of[child]});
                used.emplace_back(cp, cp);
            }
            const auto free = detail::subtract_ranges(detail::normalize_ranges(plain), detail::normalize_ranges(used));
            if (!free.empty()) {
                alts.push_back({detail::utf8_class(b_, free, "key_diverge"), plain_star, b_.byte('"')});
            }
            rule_of[i] = b_.alt(std::move(alts), "other_key");
        }
        return b_.seq({b_.byte('"'), rule_of[0]}, "other_key_string");
    }

    Sym object(const Json& s, std::size_t level, std::size_t sdepth) {
        if (level >= jg_.opts().max_nesting) return b_.never("object_too_deep");
        struct Prop {
            std::string name;
            const Json* schema;
            bool required;
        };
        std::vector<Prop> props;
        static const Json kTrue = true;
        if (s.contains("properties")) {
            const auto& p = s.at("properties");
            HALO_CHECK(p.is_object(), ErrorCode::Api, "structured output: properties must be an object");
            for (const auto& [k, v] : p.items()) props.push_back(Prop{k, &v, false});
        }
        const Json* extra = &kTrue;  // JSON Schema default: additional properties allowed
        if (s.contains("additionalProperties")) {
            const auto& a = s.at("additionalProperties");
            HALO_CHECK(a.is_boolean() || a.is_object(), ErrorCode::Api,
                       "structured output: additionalProperties must be a boolean or a schema");
            extra = (a.is_boolean() && !a.get<bool>()) ? nullptr : &a;
        }
        if (s.contains("required")) {
            const auto& r = s.at("required");
            HALO_CHECK(r.is_array(), ErrorCode::Api, "structured output: required must be an array");
            for (const auto& n : r) {
                HALO_CHECK(n.is_string(), ErrorCode::Api, "structured output: required entries must be strings");
                const auto name = n.get<std::string>();
                // Searching `props` *including* entries appended below for earlier required
                // names is what deduplicates repeated names (covered by
                // JsonSchema.DuplicateRequiredNamesAreDeduplicated).
                auto it = std::find_if(props.begin(), props.end(), [&](const Prop& p) { return p.name == name; });
                if (it != props.end()) {
                    it->required = true;
                } else if (extra == nullptr) {
                    return b_.never("object_unsat");  // required but not allowed
                } else {
                    props.push_back(Prop{name, extra, true});
                }
            }
        }
        const Sym ws = jg_.ws();
        const Sym comma = b_.seq({b_.byte(','), ws}, "comma");
        std::vector<Sym> kv;
        std::vector<std::string> names;
        for (const auto& p : props) {
            names.push_back(p.name);
            kv.push_back(b_.seq({b_.literal(Json(p.name).dump()), ws, b_.byte(':'), ws,
                                 compile(*p.schema, level + 1, sdepth + 1), ws},
                                std::format("prop_{}", p.name)));
        }
        // Tail: additional properties A^first.
        std::array<Sym, 2> next{};  // [first=false, first=true]
        if (extra != nullptr) {
            const Sym key = names.empty() ? jg_.string_any() : other_key(names);
            const Sym ekv = b_.seq({key, ws, b_.byte(':'), ws, compile(*extra, level + 1, sdepth + 1), ws}, "extra_kv");
            const auto a_later = b_.new_rule("extra_later");
            b_.add_alt(a_later, {});
            b_.add_alt(a_later, {comma, ekv, Builder::ref(a_later)});
            const auto a_first = b_.new_rule("extra_first");
            b_.add_alt(a_first, {});
            b_.add_alt(a_first, {ekv, Builder::ref(a_later)});
            next = {Builder::ref(a_later), Builder::ref(a_first)};
        } else {
            next = {b_.empty(), b_.empty()};
        }
        // S_i^first for i = n-1 .. 0.
        for (std::size_t i = props.size(); i-- > 0;) {
            std::array<Sym, 2> cur{};
            for (int first = 0; first < 2; ++first) {
                const auto r = b_.new_rule(std::format("props{}_{}", i, first));
                if (first) {
                    b_.add_alt(r, {kv[i], next[0]});
                } else {
                    b_.add_alt(r, {comma, kv[i], next[0]});
                }
                if (!props[i].required) b_.add_alt(r, {next[static_cast<std::size_t>(first)]});
                cur[static_cast<std::size_t>(first)] = Builder::ref(r);
            }
            next = cur;
        }
        return b_.seq({b_.byte('{'), ws, next[1], b_.byte('}')}, "object");
    }

    // ---- oneOf disjointness -------------------------------------------------------
    Kind kind_of(const Json& s, std::size_t depth) const {
        HALO_CHECK(depth < 64, ErrorCode::Unsupported, "structured output: oneOf analysis too deep");
        Kind k;
        if (s.is_boolean()) {
            k.types = s.get<bool>() ? kAll : 0U;
            return k;
        }
        if (!s.is_object()) return k;
        if (s.contains("$ref")) return kind_of(resolve(s), depth + 1);
        if (s.contains("enum") || s.contains("const")) {
            std::vector<Json> vals;
            if (s.contains("enum") && s.at("enum").is_array()) vals.assign(s.at("enum").begin(), s.at("enum").end());
            if (s.contains("const")) vals.push_back(s.at("const"));
            k.types = 0;
            for (const auto& v : vals) k.types |= literal_bit(v);
            k.literals = std::move(vals);
            return k;
        }
        for (const char* comb : {"anyOf", "oneOf"}) {
            if (!s.contains(comb) || !s.at(comb).is_array()) continue;
            k.types = 0;
            std::vector<Json> lits;
            bool all_lit = true;
            for (const auto& br : s.at(comb)) {
                const Kind bk = kind_of(br, depth + 1);
                k.types |= bk.types;
                if (bk.literals) {
                    lits.insert(lits.end(), bk.literals->begin(), bk.literals->end());
                } else {
                    all_lit = false;
                }
            }
            if (all_lit) k.literals = std::move(lits);
            return k;
        }
        unsigned t = declared_types(s);
        if (t == 0) t = inferred_types(s);
        k.types = t == 0 ? kAll : t;
        if (k.types == kObj && s.contains("properties") && s.contains("required") && s.at("required").is_array()) {
            for (const auto& n : s.at("required")) {
                if (!n.is_string()) continue;
                const auto name = n.get<std::string>();
                if (!s.at("properties").contains(name)) continue;
                const Kind pk = kind_of(s.at("properties").at(name), depth + 1);
                if (pk.literals) k.discriminators.emplace(name, *pk.literals);
            }
        }
        return k;
    }

    static bool disjoint_values(const std::vector<Json>& a, const std::vector<Json>& b) {
        for (const auto& x : a) {
            for (const auto& y : b) {
                if (x == y) return false;
                if (x.is_number() && y.is_number() && x.get<double>() == y.get<double>()) return false;
            }
        }
        return true;
    }

    void check_disjoint(const Json& branches) const {
        std::vector<Kind> kinds;
        for (const auto& br : branches) kinds.push_back(kind_of(br, 0));
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            for (std::size_t j = i + 1; j < kinds.size(); ++j) {
                const Kind& a = kinds[i];
                const Kind& b = kinds[j];
                if ((a.types & b.types) == 0) continue;
                if (a.literals && b.literals && disjoint_values(*a.literals, *b.literals)) continue;
                bool disc = false;
                if (a.types == kObj && b.types == kObj) {
                    for (const auto& [name, vals] : a.discriminators) {
                        const auto it = b.discriminators.find(name);
                        if (it != b.discriminators.end() && disjoint_values(vals, it->second)) disc = true;
                    }
                }
                HALO_CHECK(disc, ErrorCode::Unsupported,
                           "structured output: oneOf branches {} and {} may overlap (only provably disjoint "
                           "branches are supported; use anyOf)",
                           i, j);
            }
        }
    }

    const Json& root_;
    JsonGrammar& jg_;
    Builder& b_;
    std::map<std::pair<const Json*, std::size_t>, Sym> memo_;
    std::set<std::pair<const Json*, std::size_t>> in_progress_;
};

Grammar finish_root(Builder& b, JsonGrammar& jg, Sym value) {
    const Sym root = b.seq({jg.ws(), value, jg.ws()}, "root");
    return Grammar(std::make_shared<const detail::GrammarData>(b.finish(root)));
}

}  // namespace

Grammar Grammar::from_json_schema(std::string_view schema_json, const SchemaOptions& opts) {
    Json schema;
    try {
        schema = Json::parse(schema_json);
    } catch (const Json::exception& e) {
        throw_error(ErrorCode::Api, "structured output: json_schema is not valid JSON: {}", e.what());
    }
    Builder b(opts.max_grammar_elements);
    JsonGrammar jg(b, opts);
    SchemaCompiler c(schema, jg);
    const Sym v = c.compile(schema, 0, 0);
    return finish_root(b, jg, v);
}

Grammar Grammar::any_json_object(const SchemaOptions& opts) {
    Builder b(opts.max_grammar_elements);
    JsonGrammar jg(b, opts);
    return finish_root(b, jg, jg.object_any(0));
}

Grammar Grammar::any_json_value(const SchemaOptions& opts) {
    Builder b(opts.max_grammar_elements);
    JsonGrammar jg(b, opts);
    return finish_root(b, jg, jg.value_any(0));
}

std::size_t Grammar::num_rules() const noexcept { return data_->alts.size(); }
std::size_t Grammar::num_elements() const noexcept { return data_->elems.size(); }
std::string Grammar::debug_string() const { return data_->debug_string(); }

}  // namespace halo::sampling
