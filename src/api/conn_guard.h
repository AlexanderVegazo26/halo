#pragma once
// Per-connection read deadlines and pre-auth load shedding for the HTTP server (security
// reviews S-13, S-38, S-39, S-45).
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
//     connected to the server's port and observes each one through the kernel: how many
//     bytes the application has read from it (TCP_INFO bytes received minus the unread
//     bytes, FIONREAD) and whether its unread bytes already hold a complete header block
//     (MSG_PEEK).
//   * Header deadline (S-38 a). The clock starts when a worker has read the first byte of
//     the request (or when the previous response on a keep-alive connection is done), not
//     at accept: a connection waiting in httplib's queue with its request already buffered
//     has no deadline running. A connection that has sent nothing is timed from accept.
//     Body deadline: from the end of the headers.
//   * Per-peer cap (S-38 b). At most `per_peer_header_limit` connections per peer address
//     may be stalled before their first request: in the header phase, never served, and
//     without a complete header block buffered. The oldest beyond the cap are shut down.
//   * Overload shedding (S-38 b). When a complete request has waited a full scan interval
//     without a worker picking it up, the pool is saturated; every header-phase connection
//     that is not a complete buffered request (dribbling, silent, or idle keep-alive) and
//     is older than `shed_grace` is shut down, so the queue drains to the waiting requests.
//     Complete buffered requests are never shed.
//   * Shutting down (shutdown(2), SHUT_RDWR) makes httplib see a read error, close the
//     socket and free the worker. It goes through a dup() of the descriptor whose inode
//     and peer are checked on the duplicate itself, so a descriptor number that was closed
//     and reused in the meantime is never touched (S-45).
// The keep-alive gap between two requests counts toward the header deadline (httplib's own
// keep-alive timeout, 5 s by default, normally closes idle connections first). Handlers
// (generation, SSE) are not limited here; request_timeout and write_timeout cover them.
//
// Limits: a peer that opens more connections than the HTTP pool plus its queue can hold
// still gets new connections refused at accept (by httplib) between two scans, and the
// guard cannot tell apart peers behind one address. Non-loopback deployments need the
// reverse proxy D-017 requires. Linux only (the scan uses /proc); elsewhere start() is a
// no-op and the gap remains, which docs/api.md states.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace halo::api {

struct GuardLimits {
    std::chrono::milliseconds header_timeout{10000};  ///< 0 = no header deadline
    std::chrono::milliseconds body_timeout{60000};    ///< 0 = no body deadline
    std::size_t per_peer_header_limit = 8;            ///< 0 = no per-peer cap
    std::chrono::milliseconds shed_grace{1000};       ///< 0 = no overload shedding
    std::chrono::milliseconds scan_interval{200};
};

class ConnectionGuard {
public:
    using Clock = std::chrono::steady_clock;
    using Key = std::pair<std::string, int>;  ///< peer (numeric address, port)

    /// What one scan saw of one connected socket (step 1, no lock held).
    struct Observed {
        Key key;
        int fd = -1;               ///< descriptor number at snapshot time (identity checked again before use)
        std::uint64_t ino = 0;     ///< socket inode at snapshot time
        bool tcp_info = false;     ///< TCP_INFO and FIONREAD succeeded (else: treated as read by a worker)
        std::uint64_t consumed = 0;  ///< bytes the application has read from the socket
        std::uint64_t unread = 0;    ///< bytes waiting in the receive buffer
        bool complete_header = false;  ///< the unread bytes hold a complete header block
    };
    enum class Reason { Deadline, PeerLimit, Overload };
    struct Victim {
        Key key;
        int fd = -1;
        std::uint64_t ino = 0;
        Reason reason = Reason::Deadline;
    };

    explicit ConnectionGuard(GuardLimits limits);
    ~ConnectionGuard();
    ConnectionGuard(const ConnectionGuard&) = delete;
    ConnectionGuard& operator=(const ConnectionGuard&) = delete;

    /// Starts the watchdog for connections to local `port`. No-op when every limit is off
    /// or on non-Linux builds. Idempotent.
    void start(int port);
    void stop();

    void headers_done(const std::string& addr, int port);
    void handler_started(const std::string& addr, int port);
    void request_done(const std::string& addr, int port);

    /// Merges one scan's observations (taken starting at `t_snap`) into the connection
    /// table at time `now` and returns the connections to shut down. Pure bookkeeping, no
    /// syscalls; public so tests can drive interleavings (S-39) deterministically.
    [[nodiscard]] std::vector<Victim> decide(const std::vector<Observed>& live, Clock::time_point t_snap,
                                             Clock::time_point now);

    /// Shuts `v` down through a duplicate of its descriptor, after checking on the
    /// duplicate that it is still the same socket (inode) with the same peer on local
    /// `port`. `between` runs after the check and before shutdown (tests use it to close
    /// and reuse the descriptor number). Returns true when the socket was shut down.
    static bool shutdown_verified(const Victim& v, int port, const std::function<void()>& between = {});

    /// Connections closed by the guard, total and per reason.
    [[nodiscard]] std::uint64_t timeouts() const noexcept { return timeouts_.load(); }
    [[nodiscard]] std::uint64_t closes(Reason r) const noexcept { return by_reason_[static_cast<int>(r)].load(); }
    [[nodiscard]] bool active() const noexcept { return running_.load(); }
    /// Phase of a tracked connection, for tests: 0 header, 1 body, 2 handler, -1 untracked.
    [[nodiscard]] int phase_of(const Key& k);

private:
    enum class Phase { Header, Body, Handler };
    struct Conn {
        int fd = -1;               ///< -1 until a scan has observed the socket
        std::uint64_t ino = 0;
        Phase phase = Phase::Header;
        Clock::time_point since;       ///< start of the running phase clock
        Clock::time_point first_seen;  ///< accept (first scan) or first report
        Clock::time_point touched;     ///< last create / phase report (S-39)
        Clock::time_point ready_since{};  ///< first scan that saw a complete buffered request
        bool started = false;  ///< header phase: a worker has read from it (clock running)
        bool served = false;   ///< at least one request's headers were read
        bool seen = false;
        bool ready = false;    ///< header phase: a complete request is buffered, unread
        std::uint64_t unread = 0;
    };

    void set_phase(const std::string& addr, int port, Phase p);
    void loop();
    void scan();

    const GuardLimits lim_;
    int port_ = -1;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<Key, Conn> conns_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool quit_ = false;
    std::atomic<std::uint64_t> timeouts_{0};
    std::atomic<std::uint64_t> by_reason_[3] = {};
    std::vector<char> peek_buf_;
};

}  // namespace halo::api
