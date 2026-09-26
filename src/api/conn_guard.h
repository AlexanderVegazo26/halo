#pragma once
// Per-connection read deadlines for the HTTP server (security review S-13).
//
// cpp-httplib 0.57.1 bounds each recv() (read_timeout) but not the total time spent
// reading a request's headers or body, and it serves each connection on one pool thread.
// A peer that sends one header byte every few seconds therefore holds a worker forever,
// before authentication runs; a handful of such connections makes the whole API,
// /health included, unreachable.
//
// ConnectionGuard closes that gap without patching httplib:
//   * The server reports, per peer (address, port), when a request's headers have been
//     read (pre-routing handler), when its handler starts (after the body was read) and
//     when its response is done (logger).
//   * A watchdog thread scans this process's descriptors (/proc/self/fd) for TCP sockets
//     connected to the server's port, starts a clock for each new one, and shuts down
//     (shutdown(2), SHUT_RDWR) any connection that has been waiting for headers longer than
//     `header_timeout`, or reading a body longer than `body_timeout`. httplib then sees a
//     read error, closes the socket and frees the worker.
//   * A socket is shut down only after getpeername() confirms it is still the same peer on
//     the server's port, so a reused descriptor is never touched.
// The keep-alive gap between two requests counts toward the header deadline (httplib's own
// keep-alive timeout, 5 s by default, normally closes idle connections first). Handlers
// (generation, SSE) are not limited here; request_timeout and write_timeout cover them.
//
// Linux only (the scan uses /proc); elsewhere start() is a no-op and the gap remains, which
// docs/api.md states. Per-peer connection caps are not implemented: put non-loopback
// deployments behind a reverse proxy with connection limits.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace halo::api {

class ConnectionGuard {
public:
    ConnectionGuard(std::chrono::milliseconds header_timeout, std::chrono::milliseconds body_timeout,
                    std::chrono::milliseconds scan_interval = std::chrono::milliseconds(200));
    ~ConnectionGuard();
    ConnectionGuard(const ConnectionGuard&) = delete;
    ConnectionGuard& operator=(const ConnectionGuard&) = delete;

    /// Starts the watchdog for connections to local `port`. No-op when both timeouts are 0
    /// or on non-Linux builds. Idempotent.
    void start(int port);
    void stop();

    void headers_done(const std::string& addr, int port);
    void handler_started(const std::string& addr, int port);
    void request_done(const std::string& addr, int port);

    /// Connections shut down for exceeding a deadline.
    [[nodiscard]] std::uint64_t timeouts() const noexcept { return timeouts_.load(); }
    [[nodiscard]] bool active() const noexcept { return running_.load(); }

private:
    enum class Phase { Header, Body, Handler };
    struct Conn {
        int fd = -1;
        Phase phase = Phase::Header;
        std::chrono::steady_clock::time_point since;
        bool seen = false;
    };
    using Key = std::pair<std::string, int>;

    void set_phase(const std::string& addr, int port, Phase p);
    void loop();
    void scan();

    const std::chrono::milliseconds header_timeout_, body_timeout_, interval_;
    int port_ = -1;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<Key, Conn> conns_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool quit_ = false;
    std::atomic<std::uint64_t> timeouts_{0};
};

}  // namespace halo::api
