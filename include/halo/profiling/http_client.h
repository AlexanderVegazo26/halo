#pragma once
// Minimal HTTP/1.1 client for the local baseline servers (llama-server, Ollama).
//
// Scope is deliberately narrow: the host must be the numeric loopback address 127.0.0.1
// or ::1 (no name resolution, no remote hosts, no TLS), so an adapter config cannot turn
// the harness into a network client for arbitrary endpoints. One request per connection
// ("Connection: close").
//
// Responses are untrusted input: the status line, header block (<= 64 KiB), Content-Length
// (strict decimal, overflow-checked), chunked encoding (hex sizes, per-chunk and total caps)
// and read-until-close bodies are all bounded by max_body_bytes. Every connect/send/recv
// honours the timeout. Violations are Error(Io) (transport) or Error(Api) (malformed
// response).

#include <chrono>
#include <cstdint>
#include <map>
#include <string>

namespace halo::profiling {

struct HttpRequest {
    std::string method = "GET";  ///< GET | POST
    std::string host = "127.0.0.1";
    std::uint16_t port = 0;
    std::string path = "/";
    std::string body;
    std::string content_type = "application/json";
    std::chrono::milliseconds timeout{std::chrono::seconds(30)};
    std::size_t max_body_bytes = 64U << 20;
};

struct HttpResponse {
    int status = 0;
    std::map<std::string, std::string> headers;  ///< keys lowercased
    std::string body;
};

[[nodiscard]] HttpResponse http_request(const HttpRequest& req);

}  // namespace halo::profiling
