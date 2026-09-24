#include "json_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <random>
#include <unordered_set>
#include <vector>

#include "halo/core/error.h"

namespace halo::api {

int http_status(ErrorKind k) noexcept {
    switch (k) {
        case ErrorKind::InvalidRequest: return 400;
        case ErrorKind::Unsupported: return 400;
        case ErrorKind::Authentication: return 401;
        case ErrorKind::Forbidden: return 403;
        case ErrorKind::NotFound: return 404;
        case ErrorKind::TooLarge: return 413;
        case ErrorKind::MediaType: return 415;
        case ErrorKind::RateLimited: return 429;
        case ErrorKind::Overloaded: return 503;
        case ErrorKind::Server: return 500;
    }
    return 500;
}

namespace {

std::string_view openai_type(ErrorKind k) noexcept {
    switch (k) {
        case ErrorKind::InvalidRequest:
        case ErrorKind::Unsupported:
        case ErrorKind::Authentication:  // OpenAI reports bad keys as invalid_request_error
        case ErrorKind::Forbidden:
        case ErrorKind::NotFound:
        case ErrorKind::TooLarge:
        case ErrorKind::MediaType: return "invalid_request_error";
        case ErrorKind::RateLimited: return "rate_limit_error";
        case ErrorKind::Overloaded:
        case ErrorKind::Server: return "server_error";
    }
    return "server_error";
}

std::string_view openai_default_code(ErrorKind k) noexcept {
    switch (k) {
        case ErrorKind::Unsupported: return "unsupported_parameter";
        case ErrorKind::Authentication: return "invalid_api_key";
        case ErrorKind::TooLarge: return "request_too_large";
        case ErrorKind::MediaType: return "unsupported_media_type";
        case ErrorKind::RateLimited: return "rate_limit_exceeded";
        case ErrorKind::Overloaded: return "overloaded";
        default: return {};
    }
}

std::string_view anthropic_type(ErrorKind k) noexcept {
    switch (k) {
        case ErrorKind::InvalidRequest:
        case ErrorKind::Unsupported:
        case ErrorKind::MediaType: return "invalid_request_error";
        case ErrorKind::Authentication: return "authentication_error";
        case ErrorKind::Forbidden: return "permission_error";
        case ErrorKind::NotFound: return "not_found_error";
        case ErrorKind::TooLarge: return "request_too_large";
        case ErrorKind::RateLimited: return "rate_limit_error";
        case ErrorKind::Overloaded: return "overloaded_error";
        case ErrorKind::Server: return "api_error";
    }
    return "api_error";
}

}  // namespace

Json openai_error_body(const ApiErrorInfo& e) {
    Json err = Json::object();
    err["message"] = e.message;
    err["type"] = openai_type(e.kind);
    err["param"] = e.param ? Json(*e.param) : Json(nullptr);
    if (e.code) {
        err["code"] = *e.code;
    } else if (const auto c = openai_default_code(e.kind); !c.empty()) {
        err["code"] = c;
    } else {
        err["code"] = nullptr;
    }
    return Json{{"error", std::move(err)}};
}

Json anthropic_error_body(const ApiErrorInfo& e) {
    std::string msg = e.message;
    if (e.param && msg.find(*e.param) == std::string::npos) msg = *e.param + ": " + msg;
    return Json{{"type", "error"}, {"error", {{"type", anthropic_type(e.kind)}, {"message", std::move(msg)}}}};
}

Json error_body(ApiFamily f, const ApiErrorInfo& e) {
    return f == ApiFamily::Anthropic ? anthropic_error_body(e) : openai_error_body(e);
}

namespace {

/// halo::Error::what() is "CODE_NAME: message"; drop the prefix for client messages.
std::string_view strip_code_prefix(std::string_view w) {
    const auto colon = w.find(": ");
    if (colon != std::string_view::npos && colon < 24) {
        const std::string_view head = w.substr(0, colon);
        if (head.find(' ') == std::string_view::npos && (head.ends_with("ERROR") || head == "CANCELLED")) {
            return w.substr(colon + 2);
        }
    }
    return w;
}

constexpr std::size_t kClientDetailBytes = 512;

}  // namespace

ApiErrorInfo classify_exception(const std::exception& ex) {
    if (const auto* re = dynamic_cast<const RequestError*>(&ex)) return re->info();
    if (const auto* he = dynamic_cast<const halo::Error*>(&ex)) {
        const auto detail = [&] { return sanitize_for_log(strip_code_prefix(he->what()), kClientDetailBytes); };
        switch (he->code()) {
            case ErrorCode::Api: return {ErrorKind::InvalidRequest, detail(), std::nullopt, std::nullopt};
            case ErrorCode::Unsupported: return {ErrorKind::Unsupported, detail(), std::nullopt, std::nullopt};
            case ErrorCode::Memory:
                return {ErrorKind::Overloaded, "insufficient memory for this request", std::nullopt, std::nullopt};
            case ErrorCode::Cancelled: return {ErrorKind::Overloaded, "request cancelled", std::nullopt, std::nullopt};
            default: return {ErrorKind::Server, "internal error", std::nullopt, std::nullopt};
        }
    }
    if (dynamic_cast<const std::bad_alloc*>(&ex) != nullptr) {
        return {ErrorKind::Overloaded, "out of memory", std::nullopt, std::nullopt};
    }
    return {ErrorKind::Server, "internal error", std::nullopt, std::nullopt};
}

std::string sanitize_for_log(std::string_view s, std::size_t max_bytes) {
    std::string out;
    out.reserve(std::min(s.size(), max_bytes) + 3);
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t len = 1;
        bool ok = true;
        if (c >= 0x80) {
            len = (c & 0xE0u) == 0xC0u ? 2 : (c & 0xF0u) == 0xE0u ? 3 : (c & 0xF8u) == 0xF0u ? 4 : 0;
            if (len == 0 || c == 0xC0 || c == 0xC1 || c > 0xF4 || i + len > s.size()) {
                ok = false;
                len = 1;
            } else {
                for (std::size_t k = 1; k < len; ++k) {
                    if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) ok = false;
                }
                if (!ok) len = 1;
            }
        }
        std::string_view repl = s.substr(i, len);
        if (!ok || (len == 1 && (c < 0x20 || c == 0x7F)) ||
            (len == 2 && c == 0xC2 && static_cast<unsigned char>(s[i + 1]) < 0xA0)) {  // C1 controls
            repl = "?";
        }
        if (out.size() + repl.size() > max_bytes) {
            out += "...";
            return out;
        }
        out += repl;
        i += len;
    }
    return out;
}

namespace {

Json parse_strict(std::string_view body, std::size_t max_depth, const std::string& what) {
    // One key set per open object; a duplicate key is rejected (two parsers disagreeing on
    // which value wins is a request-smuggling vector). The depth check runs inside the
    // parser callback, i.e. before any recursive copy / dump / template conversion of the
    // value exists (security review S-2); nlohmann's parser itself is iterative.
    std::vector<std::unordered_set<std::string>> keys;
    const auto cb = [&](int depth, Json::parse_event_t ev, Json& parsed) -> bool {
        if (depth < 0 || static_cast<std::size_t>(depth) > max_depth) {
            throw RequestError(ErrorKind::InvalidRequest,
                               what + " nests deeper than " + std::to_string(max_depth) + " levels");
        }
        switch (ev) {
            case Json::parse_event_t::object_start: keys.emplace_back(); break;
            case Json::parse_event_t::object_end:
                if (!keys.empty()) keys.pop_back();
                break;
            case Json::parse_event_t::key:
                if (!keys.empty() && parsed.is_string() &&
                    !keys.back().insert(parsed.get_ref<const std::string&>()).second) {
                    throw RequestError(ErrorKind::InvalidRequest,
                                       "duplicate JSON object key '" +
                                           sanitize_for_log(parsed.get_ref<const std::string&>(), 128) + "' in " +
                                           what);
                }
                break;
            default: break;
        }
        return true;
    };
    try {
        return Json::parse(body.begin(), body.end(), cb, /*allow_exceptions=*/true, /*ignore_comments=*/false);
    } catch (const Json::exception& e) {
        // parse_error for syntax, out_of_range for numbers such as 1e999 that overflow a
        // double: both are the client's malformed input.
        throw RequestError(ErrorKind::InvalidRequest, "malformed JSON in " + what + ": " + e.what());
    }
}

}  // namespace

Json parse_request_body(std::string_view body, std::size_t max_depth) {
    Json j = parse_strict(body, max_depth, "request body");
    if (!j.is_object()) throw RequestError(ErrorKind::InvalidRequest, "request body must be a JSON object");
    return j;
}

Json parse_embedded_json(std::string_view text, std::size_t max_depth, const std::string& where) {
    try {
        return parse_strict(text, max_depth, where);
    } catch (const RequestError& e) {
        throw RequestError(e.info().kind, e.info().message, where);
    }
}

std::string dump(const Json& j) { return j.dump(-1, ' ', false, Json::error_handler_t::replace); }

const Json* field(const Json& obj, std::string_view key) {
    if (!obj.is_object()) return nullptr;
    const auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) return nullptr;
    return &*it;
}

std::string join_path(const std::string& base, std::string_view key) {
    return base.empty() ? std::string(key) : base + "." + std::string(key);
}
std::string join_path(const std::string& base, std::size_t index) { return base + "[" + std::to_string(index) + "]"; }

std::optional<bool> opt_bool(const Json& obj, std::string_view key, const std::string& where) {
    const Json* v = field(obj, key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_boolean()) throw RequestError(ErrorKind::InvalidRequest, "must be a boolean", where);
    return v->get<bool>();
}

std::optional<std::string> opt_string(const Json& obj, std::string_view key, const std::string& where) {
    const Json* v = field(obj, key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_string()) throw RequestError(ErrorKind::InvalidRequest, "must be a string", where);
    return v->get<std::string>();
}

std::optional<double> opt_number(const Json& obj, std::string_view key, const std::string& where, double lo, double hi) {
    const Json* v = field(obj, key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_number()) throw RequestError(ErrorKind::InvalidRequest, "must be a number", where);
    const double d = v->get<double>();
    if (!std::isfinite(d) || d < lo || d > hi) {
        throw RequestError(ErrorKind::InvalidRequest,
                           std::format("must be between {} and {}", lo, hi), where);
    }
    return d;
}

std::optional<std::int64_t> opt_int(const Json& obj, std::string_view key, const std::string& where, std::int64_t lo,
                                    std::int64_t hi) {
    const Json* v = field(obj, key);
    if (v == nullptr) return std::nullopt;
    std::int64_t x = 0;
    if (v->is_number_unsigned()) {
        const auto u = v->get<std::uint64_t>();
        if (u > static_cast<std::uint64_t>(hi)) {
            throw RequestError(ErrorKind::InvalidRequest, std::format("must be between {} and {}", lo, hi), where);
        }
        x = static_cast<std::int64_t>(u);
    } else if (v->is_number_integer()) {
        x = v->get<std::int64_t>();
    } else if (v->is_number_float()) {
        // Accept integral floats such as 256.0 (some clients send them); reject 1.5.
        const double d = v->get<double>();
        if (!std::isfinite(d) || std::floor(d) != d || d < static_cast<double>(lo) || d > static_cast<double>(hi)) {
            throw RequestError(ErrorKind::InvalidRequest, std::format("must be an integer between {} and {}", lo, hi),
                               where);
        }
        x = static_cast<std::int64_t>(d);
    } else {
        throw RequestError(ErrorKind::InvalidRequest, "must be an integer", where);
    }
    if (x < lo || x > hi) {
        throw RequestError(ErrorKind::InvalidRequest, std::format("must be between {} and {}", lo, hi), where);
    }
    return x;
}

bool constant_time_equal(std::string_view expected, std::string_view given) noexcept {
    // Loops over the expected (configured) key only; the running time never depends on
    // where the first mismatching byte is.
    std::size_t diff = expected.size() ^ given.size();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto g = i < given.size() ? static_cast<unsigned char>(given[i]) : 0u;
        diff |= static_cast<std::size_t>(static_cast<unsigned char>(expected[i]) ^ g);
    }
    return diff == 0;
}

std::string random_hex(std::size_t n) {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    constexpr std::string_view hex = "0123456789abcdef";
    std::string s(n, '0');
    for (auto& c : s) c = hex[rng() & 0xF];
    return s;
}

}  // namespace halo::api
