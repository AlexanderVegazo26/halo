#pragma once
// Test-side JSON Schema validator for the WS-H subset. Deliberately independent of the
// grammar compiler: it parses the document with nlohmann::json and checks the schema
// keywords directly (patterns via std::wregex, ECMAScript), so a compiler bug cannot be
// mirrored here.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

namespace halo::sampling::test {

using OJson = nlohmann::ordered_json;

inline std::wstring to_wide(std::string_view s) {
    std::wstring out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp = 0;
        std::size_t n = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0U) == 0xC0U) {
            cp = c & 0x1FU;
            n = 2;
        } else if ((c & 0xF0U) == 0xE0U) {
            cp = c & 0x0FU;
            n = 3;
        } else {
            cp = c & 0x07U;
            n = 4;
        }
        for (std::size_t k = 1; k < n && i + k < s.size(); ++k) {
            cp = (cp << 6U) | (static_cast<unsigned char>(s[i + k]) & 0x3FU);
        }
        out.push_back(static_cast<wchar_t>(cp));
        i += n;
    }
    return out;
}

inline std::size_t codepoints(std::string_view s) {
    std::size_t n = 0;
    for (const char c : s) n += (static_cast<unsigned char>(c) & 0xC0U) != 0x80U ? 1 : 0;
    return n;
}

class Validator {
public:
    explicit Validator(const OJson& root) : root_(root) {}

    /// Empty string = valid; otherwise the first violation found.
    std::string check(const OJson& v) const { return check(root_, v, "$"); }

private:
    const OJson& resolve(const std::string& ref) const {
        if (ref == "#") return root_;
        for (const std::string prefix : {"#/$defs/", "#/definitions/"}) {
            if (ref.rfind(prefix, 0) == 0) {
                return root_.at(prefix.substr(2, prefix.size() - 3)).at(ref.substr(prefix.size()));
            }
        }
        throw std::runtime_error("validator: unsupported $ref " + ref);
    }

    static bool type_ok(const std::string& t, const OJson& v) {
        if (t == "null") return v.is_null();
        if (t == "boolean") return v.is_boolean();
        if (t == "object") return v.is_object();
        if (t == "array") return v.is_array();
        if (t == "string") return v.is_string();
        if (t == "number") return v.is_number();
        if (t == "integer") {
            if (v.is_number_integer() || v.is_number_unsigned()) return true;
            if (v.is_number_float()) {
                const double d = v.get<double>();
                return std::isfinite(d) && std::floor(d) == d;
            }
            return false;
        }
        throw std::runtime_error("validator: unknown type " + t);
    }

    static bool json_equal(const OJson& a, const OJson& b) {
        if (a.is_number() && b.is_number()) return a.get<double>() == b.get<double>();
        return a == b;
    }

    std::string check(const OJson& s, const OJson& v, const std::string& path) const {
        if (s.is_boolean()) return s.get<bool>() ? "" : path + ": false schema";
        if (s.contains("$ref")) return check(resolve(s.at("$ref").get<std::string>()), v, path);
        if (s.contains("type")) {
            const auto& t = s.at("type");
            bool ok = false;
            if (t.is_string()) {
                ok = type_ok(t.get<std::string>(), v);
            } else {
                for (const auto& e : t) ok = ok || type_ok(e.get<std::string>(), v);
            }
            if (!ok) return path + ": wrong type " + v.dump();
        }
        if (s.contains("enum")) {
            bool ok = false;
            for (const auto& e : s.at("enum")) ok = ok || json_equal(e, v);
            if (!ok) return path + ": not in enum";
        }
        if (s.contains("const") && !json_equal(s.at("const"), v)) return path + ": const mismatch";
        if (s.contains("anyOf")) {
            bool ok = false;
            for (const auto& b : s.at("anyOf")) ok = ok || check(b, v, path).empty();
            if (!ok) return path + ": no anyOf branch matches";
        }
        if (s.contains("oneOf")) {
            int n = 0;
            for (const auto& b : s.at("oneOf")) n += check(b, v, path).empty() ? 1 : 0;
            if (n != 1) return path + ": " + std::to_string(n) + " oneOf branches match";
        }
        if (v.is_string()) {
            const auto str = v.get<std::string>();
            const std::size_t n = codepoints(str);
            if (s.contains("minLength") && n < s.at("minLength").get<std::size_t>()) return path + ": too short";
            if (s.contains("maxLength") && n > s.at("maxLength").get<std::size_t>()) return path + ": too long";
            if (s.contains("pattern")) {
                const std::wregex re(to_wide(s.at("pattern").get<std::string>()), std::regex::ECMAScript);
                if (!std::regex_search(to_wide(str), re)) return path + ": pattern mismatch " + v.dump();
            }
        }
        if (v.is_array()) {
            if (s.contains("minItems") && v.size() < s.at("minItems").get<std::size_t>()) return path + ": few items";
            if (s.contains("maxItems") && v.size() > s.at("maxItems").get<std::size_t>()) return path + ": many items";
            if (s.contains("items")) {
                for (std::size_t i = 0; i < v.size(); ++i) {
                    auto e = check(s.at("items"), v[i], path + "[" + std::to_string(i) + "]");
                    if (!e.empty()) return e;
                }
            }
        }
        if (v.is_object()) {
            if (s.contains("required")) {
                for (const auto& r : s.at("required")) {
                    if (!v.contains(r.get<std::string>())) return path + ": missing " + r.get<std::string>();
                }
            }
            const OJson* props = s.contains("properties") ? &s.at("properties") : nullptr;
            for (const auto& [k, x] : v.items()) {
                if (props != nullptr && props->contains(k)) {
                    auto e = check(props->at(k), x, path + "." + k);
                    if (!e.empty()) return e;
                } else if (s.contains("additionalProperties")) {
                    auto e = check(s.at("additionalProperties"), x, path + "." + k);
                    if (!e.empty()) return path + ": additional property " + k + " (" + e + ")";
                }
            }
        }
        return "";
    }

    const OJson& root_;
};

/// Numbers that overflow a double are valid JSON (RFC 8259 sets no range) and the grammar
/// deliberately stays faithful to that: `number` may produce 9E637, and an integer may have
/// 400 digits. nlohmann::json refuses such numbers (out_of_range.406), so before parsing,
/// every number token outside a string whose value is not finite as a double is replaced by
/// a same-sign finite stand-in (1e308, an integral value). This keeps type checks meaningful
/// (an overflowing number is still a number, and still not a string); it is sound only
/// because the supported subset has no numeric range keywords (minimum/maximum etc. are
/// rejected as Unsupported), so the magnitude never decides validity.
inline std::string clamp_overflowing_numbers(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    for (std::size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (in_string) {
            out.push_back(c);
            if (c == '\\' && i + 1 < text.size()) {
                out.push_back(text[i + 1]);
                i += 2;
                continue;
            }
            if (c == '"') in_string = false;
            ++i;
            continue;
        }
        if (c == '"') {
            in_string = true;
            out.push_back(c);
            ++i;
            continue;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            std::size_t j = i;
            while (j < text.size() && std::string_view("0123456789+-.eE").find(text[j]) != std::string_view::npos) ++j;
            const std::string tok(text.substr(i, j - i));
            const double d = std::strtod(tok.c_str(), nullptr);
            if (std::isfinite(d)) {
                out += tok;
            } else {
                out += tok[0] == '-' ? "-1e308" : "1e308";
            }
            i = j;
            continue;
        }
        out.push_back(c);
        ++i;
    }
    return out;
}

/// Parses and validates; empty string = valid. See clamp_overflowing_numbers().
inline std::string validate_text(const OJson& schema, std::string_view text) {
    OJson v;
    try {
        v = OJson::parse(clamp_overflowing_numbers(text));
    } catch (const std::exception& e) {
        return std::string("parse error: ") + e.what();
    }
    return Validator(schema).check(v);
}

}  // namespace halo::sampling::test
