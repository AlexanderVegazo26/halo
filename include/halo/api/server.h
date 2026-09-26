#pragma once
// HALO HTTP API server (PRD FR-013, TRD §25, PRD §12 security): OpenAI chat/completions,
// Anthropic Messages, /tokenize, /apply-template, /health, /v1/models, /metrics, with SSE
// streaming. Backed by cpp-httplib (kept out of this header).
//
// Security defaults (PRD §12, security review 2026-09-24 A-1..A-12): binds 127.0.0.1;
// optional API key (Bearer or x-api-key, constant-time compare) required on every route
// except GET /health and CORS preflight, and mandatory for a non-loopback bind; Host header
// checked against the bound names (DNS rebinding); a request carrying an Origin that is not
// configured is refused (403) on every route; POST bodies must be application/json (415);
// request-body cap applied before parsing (413); JSON nesting depth capped at parse time;
// caps on messages, content bytes, tools and tool-schema size; concurrency cap + bounded
// wait queue (429 when the queue is full, 503 when a queued request waits longer than
// queue_timeout); per-request max_tokens cap and wall-clock limit; client disconnect cancels
// generation; no request field names a file path; client strings cannot inject special
// tokens (prompt.h). Every handler converts errors into the calling API's native error body
// with a generic message for internal failures; one failing request never takes the process
// down.
//
// See docs/api.md for the endpoint reference and the field support matrix.
//
// Thread-safety: start()/stop()/listen() must be called from one controlling thread; the
// request handlers run on the server's worker pool and use the Engine concurrently (the
// Engine contract is thread-safe). The Engine must outlive the ApiServer.

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace halo::runtime { class Engine; }

namespace halo::api {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;                     ///< 0 = pick an ephemeral port (read it via port())
    std::optional<std::string> api_key;  ///< empty/absent = no authentication
    std::size_t max_body_bytes = 8u << 20;
    std::size_t max_json_depth = 64;     ///< nesting depth limit for request bodies
    std::size_t max_concurrent = 4;      ///< generations running at once (API-level cap)
    std::size_t max_queue = 16;          ///< generations waiting for a slot; beyond -> 429
    std::chrono::milliseconds queue_timeout{30000};  ///< waited longer -> 503
    std::size_t max_tokens_cap = 32768;  ///< a request asking for more is rejected (400)
    std::size_t default_max_tokens = 8192;  ///< when the request gives none (clamped to context)
    /// Output headroom reserved from max_tokens when thinking is on and the request gives no
    /// explicit reasoning budget: reasoning may use max_tokens - min(reserve, max_tokens/2)
    /// tokens before HALO closes the think block and lets the answer start (TRD §25).
    std::size_t reasoning_output_reserve = 512;
    std::vector<std::string> cors_origins;  ///< exact origins, or "*"; empty = CORS off
    std::string served_model_name;          ///< empty = Engine::model().id
    /// /v1/completions takes a raw prompt from the client; with true, special-token text in
    /// it is parsed as control tokens (the client authored the whole prompt). Chat routes
    /// never do this for client-supplied strings (see prompt.h).
    bool completions_parse_special = true;
    std::size_t http_threads = 0;  ///< 0 = max_concurrent + max_queue + utility_concurrency + utility_queue + 4
    /// /tokenize and /apply-template run behind their own small admission (security review
    /// S-16): at most `utility_concurrency` at once, `utility_queue` waiting; beyond -> 429.
    std::size_t utility_concurrency = 2;
    std::size_t utility_queue = 8;
    /// Per-connection deadlines for reading a request (security review S-13): the time from
    /// the connection opening (or the previous response) to the complete header block, and
    /// from the headers to the complete body. 0 = unlimited. Linux only (see conn_guard.h).
    std::chrono::milliseconds header_timeout{10000};
    std::chrono::milliseconds body_timeout{60000};
    std::chrono::seconds read_timeout{60};
    std::chrono::seconds write_timeout{60};
    std::chrono::seconds keep_alive_timeout{5};
    std::size_t max_messages = 4096;
    std::size_t max_tools = 256;
    std::size_t max_stop_sequences = 16;
    /// Sum of the bytes of every string in messages/system (chat routes), of the prompt
    /// (/v1/completions) or of the content (/tokenize); beyond -> 413.
    std::size_t max_content_bytes = 4u << 20;
    /// Serialized size of one tool's parameters / input_schema; beyond -> 400.
    std::size_t max_tool_schema_bytes = 64u << 10;
    /// Wall-clock limit for one generation (all engine calls); 0 = none. On expiry the
    /// generation ends like max_tokens (finish_reason "length" / stop_reason "max_tokens")
    /// with a warning.
    std::chrono::seconds request_timeout{600};
    /// Unmatched '[' / '{' allowed in parsed model output before generation is stopped
    /// (bounds the recursion of tool-call argument parsing; security review S-9).
    std::size_t max_output_nesting = 256;
    /// Host header values (host names, port ignored) accepted besides the bind address.
    /// When the server binds a loopback address, "localhost", "127.0.0.1" and "[::1]" are
    /// always accepted and every other Host is refused (DNS-rebinding defence). When it binds
    /// any other address and this list is empty, the Host header is not checked (the API
    /// key is then mandatory, see below).
    std::vector<std::string> allowed_hosts;
    /// D-017: v0.2 serving is loopback-only. Binding any non-loopback address is refused
    /// (Error(Config)) unless this is set, and such a deployment requires a reverse proxy
    /// that enforces per-client connection limits and header/body timeouts (docs/api.md).
    bool allow_remote = false;
    /// Binding a non-loopback address without an api_key is refused (Error(Config)) unless
    /// this is set, and then `allowed_hosts` must be non-empty so the Host check (DNS
    /// rebinding) stays on (security review S-20).
    bool allow_unauthenticated_remote = false;
};

/// Throws Error(Config) for a configuration ApiServer refuses (the checks its constructor
/// runs), so callers can fail before loading a model.
void validate_server_config(const ServerConfig& config);

/// True for 127.0.0.0/8, ::1 and "localhost".
[[nodiscard]] bool is_loopback_host(std::string_view host) noexcept;

class ApiServer {
public:
    /// Validates `config` (Error(Config) on nonsense such as max_concurrent == 0, and for a
    /// non-loopback host without api_key unless allow_unauthenticated_remote).
    ApiServer(runtime::Engine& engine, ServerConfig config);
    ~ApiServer();  ///< stop()s
    ApiServer(const ApiServer&) = delete;
    ApiServer& operator=(const ApiServer&) = delete;

    /// Binds the listening socket; returns the port. Error(Io) if binding fails.
    int bind();
    /// Serves on the calling thread until stop() (binds first if needed).
    void listen();
    /// bind() + serve on a background thread; returns once the server accepts connections.
    void start();
    /// Idempotent. In-flight generations are cancelled (their token callbacks return
    /// false), queued requests fail with 503, then the listener is closed and joined.
    void stop();

    [[nodiscard]] int port() const noexcept;
    [[nodiscard]] const ServerConfig& config() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::api
