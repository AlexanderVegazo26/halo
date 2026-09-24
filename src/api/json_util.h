#pragma once
// Request-side JSON helpers and the API error type (internal to halo_api).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace halo::api {

using Json = nlohmann::ordered_json;

/// Which API family a route belongs to; decides the native error body format.
enum class ApiFamily { OpenAI, Anthropic };

/// Error classes. Each maps to an HTTP status and to both APIs' `type` strings.
enum class ErrorKind {
    InvalidRequest,  // 400
    Unsupported,     // 400, feature not implemented in this server
    Authentication,  // 401
    Forbidden,       // 403
    NotFound,        // 404
    TooLarge,        // 413
    MediaType,       // 415 (POST body is not application/json, or an unsupported encoding)
    RateLimited,     // 429 (queue full)
    Overloaded,      // 503 (queue wait timed out / shutting down)
    Server,          // 500
};

struct ApiErrorInfo {
    ErrorKind kind = ErrorKind::Server;
    std::string message;
    std::optional<std::string> param;
    std::optional<std::string> code;
};

class RequestError : public std::runtime_error {
public:
    RequestError(ErrorKind kind, std::string message, std::optional<std::string> param = std::nullopt,
                 std::optional<std::string> code = std::nullopt)
        : std::runtime_error(message), info_{kind, std::move(message), std::move(param), std::move(code)} {}
    [[nodiscard]] const ApiErrorInfo& info() const noexcept { return info_; }

private:
    ApiErrorInfo info_;
};

[[nodiscard]] int http_status(ErrorKind k) noexcept;
/// {"error":{"message","type","param","code"}}
[[nodiscard]] Json openai_error_body(const ApiErrorInfo& e);
/// {"type":"error","error":{"type","message"}}
[[nodiscard]] Json anthropic_error_body(const ApiErrorInfo& e);
[[nodiscard]] Json error_body(ApiFamily f, const ApiErrorInfo& e);

/// Maps a halo::Error / std::exception escaping request handling to an API error. Client
/// messages never carry internal detail (security review A-9): RequestError keeps its own
/// (HALO-authored) text; halo::Error(Api/Unsupported) keeps its text sanitized and truncated
/// (it explains what was wrong with the request, e.g. a template raise_exception or a schema
/// error); every other error becomes a generic message. The full text belongs in the log
/// (see sanitize_for_log).
[[nodiscard]] ApiErrorInfo classify_exception(const std::exception& e);

/// Makes untrusted text (model-supplied strings, client strings, exception text) safe to
/// write to a log or terminal: C0/C1 control characters and DEL become '?', invalid UTF-8 is
/// replaced, and the result is cut to at most `max_bytes` (on a UTF-8 boundary, "..." added).
[[nodiscard]] std::string sanitize_for_log(std::string_view s, std::size_t max_bytes = 2048);

/// Strict JSON: RFC 8259 only (no comments, no trailing data), valid UTF-8, nesting depth
/// <= max_depth, duplicate object keys rejected, top level must be an object. Throws
/// RequestError(InvalidRequest).
[[nodiscard]] Json parse_request_body(std::string_view body, std::size_t max_depth);

/// The same strict, depth-limited parser for JSON embedded in a request as a string (e.g.
/// OpenAI tool_calls[].function.arguments); any top-level type. `where` names the field.
/// Throws RequestError(InvalidRequest).
[[nodiscard]] Json parse_embedded_json(std::string_view text, std::size_t max_depth, const std::string& where);

/// Serializes for the wire; never throws on invalid UTF-8 (replaced with U+FFFD).
[[nodiscard]] std::string dump(const Json& j);

// ---- typed field access; `where` names the field for error messages ------------------------

/// Pointer to obj[key], or nullptr when absent or JSON null.
[[nodiscard]] const Json* field(const Json& obj, std::string_view key);
[[nodiscard]] std::optional<bool> opt_bool(const Json& obj, std::string_view key, const std::string& where);
[[nodiscard]] std::optional<std::string> opt_string(const Json& obj, std::string_view key, const std::string& where);
[[nodiscard]] std::optional<double> opt_number(const Json& obj, std::string_view key, const std::string& where,
                                               double lo, double hi);
[[nodiscard]] std::optional<std::int64_t> opt_int(const Json& obj, std::string_view key, const std::string& where,
                                                  std::int64_t lo, std::int64_t hi);
[[nodiscard]] std::string join_path(const std::string& base, std::string_view key);
[[nodiscard]] std::string join_path(const std::string& base, std::size_t index);

/// Constant-time equality (time depends only on the lengths, not the contents).
[[nodiscard]] bool constant_time_equal(std::string_view a, std::string_view b) noexcept;

/// Random lowercase hex string for response/tool-call ids.
[[nodiscard]] std::string random_hex(std::size_t n);

}  // namespace halo::api
