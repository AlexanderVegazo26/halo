#include "conn_guard.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string_view>

#if defined(__linux__)
#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <linux/tcp.h>  // struct tcp_info with tcpi_bytes_received (glibc's netinet/tcp.h lacks it)
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#endif

#include "halo/core/log.h"

namespace halo::api {

namespace {
/// httplib's CPPHTTPLIB_HEADER_MAX_LENGTH: a header block this long without its end is
/// rejected by httplib as soon as a worker reads it, so it cannot stall a worker.
constexpr std::size_t kPeekBytes = 8192;
}  // namespace

ConnectionGuard::ConnectionGuard(GuardLimits limits) : lim_(limits) {}

ConnectionGuard::~ConnectionGuard() { stop(); }

void ConnectionGuard::start(int port) {
#if defined(__linux__)
    if (lim_.header_timeout.count() <= 0 && lim_.body_timeout.count() <= 0 && lim_.per_peer_header_limit == 0 &&
        lim_.shed_grace.count() <= 0) {
        return;
    }
    std::lock_guard lk(mu_);
    if (running_.load()) return;
    port_ = port;
    quit_ = false;
    running_ = true;
    peek_buf_.resize(kPeekBytes);
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
    const auto now = Clock::now();
    std::lock_guard lk(mu_);
    auto [it, inserted] = conns_.try_emplace(Key{addr, port});
    Conn& c = it->second;
    if (inserted) c.first_seen = now;
    c.phase = p;
    c.since = now;
    c.touched = now;
    c.ready = false;
    c.ready_since = {};
    c.served = true;  // any report means a request's headers were read on it
    if (p == Phase::Header) c.started = true;  // a worker holds it: the previous response is done
}

void ConnectionGuard::headers_done(const std::string& addr, int port) { set_phase(addr, port, Phase::Body); }
void ConnectionGuard::handler_started(const std::string& addr, int port) { set_phase(addr, port, Phase::Handler); }
void ConnectionGuard::request_done(const std::string& addr, int port) { set_phase(addr, port, Phase::Header); }

int ConnectionGuard::phase_of(const Key& k) {
    std::lock_guard lk(mu_);
    const auto it = conns_.find(k);
    return it == conns_.end() ? -1 : static_cast<int>(it->second.phase);
}

void ConnectionGuard::loop() {
    std::unique_lock lk(mu_);
    while (!quit_) {
        lk.unlock();
        scan();
        lk.lock();
        cv_.wait_for(lk, lim_.scan_interval, [this] { return quit_; });
    }
}

std::vector<ConnectionGuard::Victim> ConnectionGuard::decide(const std::vector<Observed>& live, Clock::time_point t_snap,
                                                             Clock::time_point now) {
    std::vector<Victim> out;
    std::lock_guard lk(mu_);
    // 1. Merge the observations.
    for (auto& [k, c] : conns_) c.seen = false;
    for (const Observed& o : live) {
        auto [it, inserted] = conns_.try_emplace(o.key);
        Conn& c = it->second;
        if (inserted) {
            c.first_seen = now;
            c.since = now;
            c.touched = now;
        }
        c.fd = o.fd;
        c.ino = o.ino;
        c.seen = true;
        if (c.phase != Phase::Header) continue;
        // S-38 (a): the header clock starts once a worker has read from the connection.
        if (!c.started && (!o.tcp_info || o.consumed > 0)) {
            c.started = true;
            c.since = now;
        }
        c.unread = o.unread;
        c.ready = !c.started && o.complete_header;
        if (!c.ready) {
            c.ready_since = {};
        } else if (c.ready_since == Clock::time_point{}) {
            c.ready_since = now;
        }
    }
    // S-39: an entry that is missing from the snapshot is gone only if nothing reported it
    // after the snapshot started (a connection accepted during the directory walk is
    // reported by set_phase but only appears in the next scan).
    std::erase_if(conns_, [t_snap](const auto& kv) { return !kv.second.seen && kv.second.touched < t_snap; });

    // 2. Deadlines.
    auto victim = [&](const Key& k, const Conn& c, Reason r) { out.push_back(Victim{k, c.fd, c.ino, r}); };
    std::set<const Key*> marked;
    bool saturated = false;
    for (const auto& [k, c] : conns_) {
        if (c.fd < 0) continue;  // reported, not yet observed
        if (c.phase == Phase::Header && c.ready && now - c.ready_since >= lim_.scan_interval) saturated = true;
        bool expired = false;
        if (c.phase == Phase::Header && lim_.header_timeout.count() > 0) {
            if (c.started) {
                expired = now - c.since > lim_.header_timeout;
            } else if (c.unread == 0) {  // silent since accept
                expired = now - c.first_seen > lim_.header_timeout;
            }  // else: queued with its bytes unread; no worker has it yet
        } else if (c.phase == Phase::Body && lim_.body_timeout.count() > 0) {
            expired = now - c.since > lim_.body_timeout;
        }
        if (expired) {
            victim(k, c, Reason::Deadline);
            marked.insert(&k);
        }
    }
    auto is_marked = [&](const Key& k) { return marked.contains(&k); };

    // 3. Per-peer cap on connections stalled before their first request.
    if (lim_.per_peer_header_limit > 0) {
        std::map<std::string_view, std::vector<std::pair<const Key*, const Conn*>>> by_peer;
        for (const auto& [k, c] : conns_) {
            if (c.fd < 0 || c.phase != Phase::Header || c.served || c.ready || is_marked(k)) continue;
            by_peer[k.first].push_back({&k, &c});
        }
        for (auto& [addr, v] : by_peer) {
            if (v.size() <= lim_.per_peer_header_limit) continue;
            std::ranges::sort(v, [](const auto& a, const auto& b) { return a.second->first_seen < b.second->first_seen; });
            for (std::size_t i = 0; i + lim_.per_peer_header_limit < v.size(); ++i) {
                victim(*v[i].first, *v[i].second, Reason::PeerLimit);
                marked.insert(v[i].first);
            }
        }
    }

    // 4. Overload: a complete request has waited a whole interval for a worker.
    if (saturated && lim_.shed_grace.count() > 0) {
        for (const auto& [k, c] : conns_) {
            if (c.fd < 0 || c.phase != Phase::Header || c.ready || is_marked(k)) continue;
            const auto age = c.started ? now - c.since : now - c.first_seen;
            if (age >= lim_.shed_grace) victim(k, c, Reason::Overload);
        }
    }
    return out;
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

bool has_header_end(std::string_view s) {
    return s.find("\r\n\r\n") != std::string_view::npos || s.find("\n\n") != std::string_view::npos;
}

}  // namespace
#endif

bool ConnectionGuard::shutdown_verified(const Victim& v, int port, const std::function<void()>& between) {
#if defined(__linux__)
    // S-45: act on a duplicate. It names one open file description for as long as we hold
    // it, so the identity check below and the shutdown cannot land on different sockets.
    const int d = ::fcntl(v.fd, F_DUPFD_CLOEXEC, 0);
    if (d < 0) return false;
    struct stat st {};
    std::string addr;
    int pport = 0;
    const bool same = ::fstat(d, &st) == 0 && S_ISSOCK(st.st_mode) && static_cast<std::uint64_t>(st.st_ino) == v.ino &&
                      peer_of(d, port, addr, pport) && addr == v.key.first && pport == v.key.second;
    if (same) {
        if (between) between();
        ::shutdown(d, SHUT_RDWR);
    }
    ::close(d);
    return same;
#else
    (void)v;
    (void)port;
    (void)between;
    return false;
#endif
}

void ConnectionGuard::scan() {
#if defined(__linux__)
    // 1. Snapshot the connected sockets on our port (no lock held during syscalls).
    const auto t_snap = Clock::now();
    std::vector<Observed> live;
    if (DIR* d = ::opendir("/proc/self/fd")) {
        while (const dirent* e = ::readdir(d)) {
            char* end = nullptr;
            const long fd_l = std::strtol(e->d_name, &end, 10);
            if (end == e->d_name || *end != '\0' || fd_l < 0) continue;
            const int fd = static_cast<int>(fd_l);
            Observed o;
            if (!peer_of(fd, port_, o.key.first, o.key.second)) continue;
            struct stat st {};
            if (::fstat(fd, &st) != 0) continue;
            o.fd = fd;
            o.ino = static_cast<std::uint64_t>(st.st_ino);
            // Bytes read by the application = received - still unread. Read "received"
            // first: a byte arriving in between then under-counts (never a false start).
            tcp_info ti{};
            socklen_t tl = sizeof(ti);
            int unread = 0;
            if (::getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &tl) == 0 &&
                tl >= offsetof(tcp_info, tcpi_bytes_received) + sizeof(ti.tcpi_bytes_received) &&
                ::ioctl(fd, FIONREAD, &unread) == 0 && unread >= 0) {  // NOLINT: ioctl is variadic
                o.tcp_info = true;
                o.unread = static_cast<std::uint64_t>(unread);
                o.consumed = ti.tcpi_bytes_received > o.unread ? ti.tcpi_bytes_received - o.unread : 0;
                if (o.consumed == 0 && o.unread > 0) {
                    const ssize_t n = ::recv(fd, peek_buf_.data(), std::min<std::size_t>(o.unread, kPeekBytes),
                                             MSG_PEEK | MSG_DONTWAIT);
                    if (n > 0) {
                        const auto got = static_cast<std::size_t>(n);
                        o.complete_header = got >= kPeekBytes || has_header_end(std::string_view(peek_buf_.data(), got));
                    }
                }
            }
            live.push_back(std::move(o));
        }
        ::closedir(d);
    }
    // 2. Merge and decide.
    const std::vector<Victim> victims = decide(live, t_snap, Clock::now());
    // 3. Shut them down.
    std::size_t n[3] = {0, 0, 0};
    for (const Victim& v : victims) {
        if (!shutdown_verified(v, port_)) continue;
        ++n[static_cast<int>(v.reason)];
        by_reason_[static_cast<int>(v.reason)].fetch_add(1);
        timeouts_.fetch_add(1);
        std::lock_guard lk(mu_);
        const auto it = conns_.find(v.key);
        if (it != conns_.end() && it->second.ino == v.ino) conns_.erase(it);
    }
    if (n[0] + n[1] + n[2] > 0) {  // one line per scan, not one per connection
        HALO_WARN("api",
                  "closed {} connection(s) while reading a request: {} past the read deadline, {} over the per-peer "
                  "limit, {} shed under overload",
                  n[0] + n[1] + n[2], n[0], n[1], n[2]);
    }
#endif
}

}  // namespace halo::api
