#include "conn_guard.h"

#include <vector>

#if defined(__linux__)
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#endif

#include "halo/core/log.h"

namespace halo::api {

ConnectionGuard::ConnectionGuard(std::chrono::milliseconds header_timeout, std::chrono::milliseconds body_timeout,
                                 std::chrono::milliseconds scan_interval)
    : header_timeout_(header_timeout), body_timeout_(body_timeout), interval_(scan_interval) {}

ConnectionGuard::~ConnectionGuard() { stop(); }

void ConnectionGuard::start(int port) {
#if defined(__linux__)
    if (header_timeout_.count() <= 0 && body_timeout_.count() <= 0) return;
    std::lock_guard lk(mu_);
    if (running_.load()) return;
    port_ = port;
    quit_ = false;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
#else
    (void)port;
#endif
}

void ConnectionGuard::stop() {
    {
        std::lock_guard lk(mu_);
        if (!running_.load()) return;
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    running_ = false;
}

void ConnectionGuard::set_phase(const std::string& addr, int port, Phase p) {
    std::lock_guard lk(mu_);
    Conn& c = conns_[Key{addr, port}];
    c.phase = p;
    c.since = std::chrono::steady_clock::now();
}

void ConnectionGuard::headers_done(const std::string& addr, int port) { set_phase(addr, port, Phase::Body); }
void ConnectionGuard::handler_started(const std::string& addr, int port) { set_phase(addr, port, Phase::Handler); }
void ConnectionGuard::request_done(const std::string& addr, int port) { set_phase(addr, port, Phase::Header); }

void ConnectionGuard::loop() {
    std::unique_lock lk(mu_);
    while (!quit_) {
        lk.unlock();
        scan();
        lk.lock();
        cv_.wait_for(lk, interval_, [this] { return quit_; });
    }
}

#if defined(__linux__)
namespace {

/// Peer (numeric address, port) of `fd` when it is a TCP socket connected to local `port`.
bool peer_of(int fd, int port, std::string& addr, int& peer_port) {
    sockaddr_storage local{};
    socklen_t len = sizeof(local);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &len) != 0) return false;  // NOLINT: sockets API
    int lport = -1;
    if (local.ss_family == AF_INET) {
        lport = ntohs(reinterpret_cast<const sockaddr_in*>(&local)->sin_port);  // NOLINT: sockets API
    } else if (local.ss_family == AF_INET6) {
        lport = ntohs(reinterpret_cast<const sockaddr_in6*>(&local)->sin6_port);  // NOLINT: sockets API
    } else {
        return false;
    }
    if (lport != port) return false;
    sockaddr_storage peer{};
    len = sizeof(peer);
    if (::getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &len) != 0) return false;  // listener / not connected
    char buf[INET6_ADDRSTRLEN] = {};
    if (peer.ss_family == AF_INET) {
        const auto* p = reinterpret_cast<const sockaddr_in*>(&peer);  // NOLINT: sockets API
        if (::inet_ntop(AF_INET, &p->sin_addr, buf, sizeof(buf)) == nullptr) return false;
        peer_port = ntohs(p->sin_port);
    } else if (peer.ss_family == AF_INET6) {
        const auto* p = reinterpret_cast<const sockaddr_in6*>(&peer);  // NOLINT: sockets API
        if (::inet_ntop(AF_INET6, &p->sin6_addr, buf, sizeof(buf)) == nullptr) return false;
        peer_port = ntohs(p->sin6_port);
    } else {
        return false;
    }
    addr = buf;
    return true;
}

}  // namespace
#endif

void ConnectionGuard::scan() {
#if defined(__linux__)
    // 1. Snapshot the connected sockets on our port (no lock held during syscalls).
    std::vector<std::pair<Key, int>> live;
    if (DIR* d = ::opendir("/proc/self/fd")) {
        while (const dirent* e = ::readdir(d)) {
            char* end = nullptr;
            const long fd = std::strtol(e->d_name, &end, 10);
            if (end == e->d_name || *end != '\0' || fd < 0) continue;
            std::string addr;
            int pport = 0;
            if (peer_of(static_cast<int>(fd), port_, addr, pport)) live.push_back({Key{addr, pport}, static_cast<int>(fd)});
        }
        ::closedir(d);
    }
    // 2. Merge, and pick the connections past their deadline.
    const auto now = std::chrono::steady_clock::now();
    std::vector<std::pair<Key, int>> expired;
    {
        std::lock_guard lk(mu_);
        for (auto& [k, c] : conns_) c.seen = false;
        for (const auto& [k, fd] : live) {
            auto it = conns_.find(k);
            if (it == conns_.end()) it = conns_.emplace(k, Conn{fd, Phase::Header, now, false}).first;
            it->second.fd = fd;
            it->second.seen = true;
        }
        std::erase_if(conns_, [](const auto& kv) { return !kv.second.seen; });
        for (const auto& [k, c] : conns_) {
            const auto age = now - c.since;
            if ((c.phase == Phase::Header && header_timeout_.count() > 0 && age > header_timeout_) ||
                (c.phase == Phase::Body && body_timeout_.count() > 0 && age > body_timeout_)) {
                expired.push_back({k, c.fd});
            }
        }
    }
    // 3. Shut them down, after re-checking that the descriptor still belongs to that peer.
    for (const auto& [k, fd] : expired) {
        std::string addr;
        int pport = 0;
        if (!peer_of(fd, port_, addr, pport) || addr != k.first || pport != k.second) continue;
        ::shutdown(fd, SHUT_RDWR);
        timeouts_.fetch_add(1);
        HALO_WARN("api", "closed a connection from {}:{} that exceeded the request read deadline", addr, pport);
        std::lock_guard lk(mu_);
        conns_.erase(k);
    }
#endif
}

}  // namespace halo::api
