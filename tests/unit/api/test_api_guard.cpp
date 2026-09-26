// ConnectionGuard (src/api/conn_guard.h): the header clock starts when a worker reads the
// request (S-38 a), the per-peer cap and overload shedding (S-38 b), the scan/report race
// (S-39) and the descriptor-reuse race on shutdown (S-45). Unit tests drive decide() with
// synthetic observations and times; the server tests run real slow clients.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <thread>

#include "api_test_util.h"
#include "conn_guard.h"

namespace halo::test {
namespace {

using namespace std::chrono_literals;
using api::ConnectionGuard;
using api::GuardLimits;
using Clock = ConnectionGuard::Clock;
using Reason = ConnectionGuard::Reason;

// ---- decide(): pure bookkeeping ------------------------------------------------------------

ConnectionGuard::Observed obs(const std::string& addr, int port, std::uint64_t consumed, std::uint64_t unread,
                              bool complete = false) {
    ConnectionGuard::Observed o;
    o.key = {addr, port};
    o.fd = 1000 + port;  // never used by decide()
    o.ino = static_cast<std::uint64_t>(port);
    o.tcp_info = true;
    o.consumed = consumed;
    o.unread = unread;
    o.complete_header = complete;
    return o;
}

GuardLimits limits(std::chrono::milliseconds header, std::size_t per_peer, std::chrono::milliseconds grace) {
    GuardLimits l;
    l.header_timeout = header;
    l.body_timeout = 0ms;
    l.per_peer_header_limit = per_peer;
    l.shed_grace = grace;
    l.scan_interval = 200ms;
    return l;
}

std::vector<std::pair<std::string, int>> keys_of(const std::vector<ConnectionGuard::Victim>& v, Reason r) {
    std::vector<std::pair<std::string, int>> out;
    for (const auto& x : v) {
        if (x.reason == r) out.push_back(x.key);
    }
    std::ranges::sort(out);
    return out;
}

TEST(ConnGuardUnit, ScanKeepsConnectionsReportedAfterItsSnapshot) {
    // S-39: a connection accepted after the /proc walk passed its descriptor, whose headers
    // and handler start are reported before the scan merges, is not in the snapshot. The
    // scan must not erase it (or it comes back as a fresh header-phase connection and a
    // long non-streaming handler is cut off at header_timeout).
    ConnectionGuard g(limits(1000ms, 0, 0ms));
    const ConnectionGuard::Key k{"10.0.0.1", 40001};
    const auto t_snap = Clock::now();
    g.headers_done(k.first, k.second);
    g.handler_started(k.first, k.second);
    EXPECT_TRUE(g.decide({}, t_snap, Clock::now()).empty());
    EXPECT_EQ(g.phase_of(k), 2) << "the entry reported after the snapshot was erased";
    // The next scans see the socket; the handler runs far past header_timeout untouched.
    for (auto later : {5s, 7s}) {
        const auto t = Clock::now() + later;
        EXPECT_TRUE(g.decide({obs(k.first, k.second, 300, 0)}, t, t).empty());
        EXPECT_EQ(g.phase_of(k), 2);
    }
    // An entry last reported before the snapshot and missing from it is gone.
    const ConnectionGuard::Key gone{"10.0.0.2", 40002};
    g.request_done(gone.first, gone.second);
    std::this_thread::sleep_for(2ms);
    const auto t2 = Clock::now();
    (void)g.decide({obs(k.first, k.second, 300, 0)}, t2, t2);
    EXPECT_EQ(g.phase_of(gone), -1);
}

TEST(ConnGuardUnit, HeaderClockStartsWhenAWorkerReads) {
    // S-38 a: a queued connection whose request is buffered and unread has no deadline; a
    // connection a worker has read from is timed from that point; a silent one from accept.
    ConnectionGuard g(limits(1000ms, 0, 0ms));
    const auto t = Clock::now();
    (void)g.decide({obs("10.0.0.1", 1, 0, 120, true), obs("10.0.0.1", 2, 0, 0)}, t, t);
    auto v = g.decide({obs("10.0.0.1", 1, 0, 120, true), obs("10.0.0.1", 2, 0, 0)}, t + 5s, t + 5s);
    EXPECT_EQ(keys_of(v, Reason::Deadline), (std::vector<std::pair<std::string, int>>{{"10.0.0.1", 2}}))
        << "only the silent connection is past its deadline; the queued request is not";
    // A worker picks the queued one up and reads part of it: now its clock runs.
    v = g.decide({obs("10.0.0.1", 1, 60, 0)}, t + 5s, t + 5s);
    EXPECT_TRUE(v.empty());
    v = g.decide({obs("10.0.0.1", 1, 70, 0)}, t + 5900ms, t + 5900ms);
    EXPECT_TRUE(v.empty());
    v = g.decide({obs("10.0.0.1", 1, 80, 0)}, t + 6100ms, t + 6100ms);
    EXPECT_EQ(keys_of(v, Reason::Deadline), (std::vector<std::pair<std::string, int>>{{"10.0.0.1", 1}}));
}

TEST(ConnGuardUnit, PerPeerCapClosesTheOldestStalledConnectionsOfThatPeer) {
    ConnectionGuard g(limits(0ms, 2, 0ms));
    const auto t = Clock::now();
    std::vector<ConnectionGuard::Observed> live = {obs("10.0.0.9", 10, 5, 0), obs("10.0.0.9", 11, 0, 0)};
    EXPECT_TRUE(g.decide(live, t, t).empty()) << "two stalled connections are within the cap";
    g.request_done("10.0.0.9", 30);  // a keep-alive connection that has served a request
    live.push_back(obs("10.0.0.9", 12, 5, 0));        // two newer dribblers of the same address
    live.push_back(obs("10.0.0.9", 13, 5, 0));
    live.push_back(obs("10.0.0.9", 20, 0, 90, true));  // a complete request, queued: exempt
    live.push_back(obs("10.0.0.8", 21, 5, 0));        // another address: its own budget
    live.push_back(obs("10.0.0.9", 30, 400, 0));      // served keep-alive: not "before its first request"
    const auto v = g.decide(live, t + 10s, t + 10s);
    EXPECT_EQ(keys_of(v, Reason::PeerLimit), (std::vector<std::pair<std::string, int>>{{"10.0.0.9", 10}, {"10.0.0.9", 11}}))
        << "the two oldest stalled connections of 10.0.0.9, and nothing else";
    EXPECT_EQ(v.size(), 2u);
}

TEST(ConnGuardUnit, OverloadShedsOnlyWhileACompleteRequestWaits) {
    ConnectionGuard g(limits(0ms, 0, 1000ms));
    const auto t = Clock::now();
    std::vector<ConnectionGuard::Observed> live = {obs("10.0.1.1", 1, 5, 0), obs("10.0.1.2", 2, 5, 0)};
    (void)g.decide(live, t, t);
    live.push_back(obs("10.0.1.3", 3, 5, 0));  // a younger dribbler
    (void)g.decide(live, t + 1000ms, t + 1000ms);
    // Nothing waits: no shedding however old the dribblers are.
    EXPECT_TRUE(g.decide(live, t + 1200ms, t + 1200ms).empty());
    live.push_back(obs("10.0.1.4", 4, 0, 200, true));  // a complete request, queued
    EXPECT_TRUE(g.decide(live, t + 1250ms, t + 1250ms).empty()) << "it has not waited a scan interval yet";
    const auto v = g.decide(live, t + 1500ms, t + 1500ms);
    EXPECT_EQ(keys_of(v, Reason::Overload),
              (std::vector<std::pair<std::string, int>>{{"10.0.1.1", 1}, {"10.0.1.2", 2}}))
        << "the dribblers older than the grace are shed; the younger one and the waiting request are not";
    EXPECT_EQ(v.size(), 2u);
    // The request keeps waiting past the grace itself: it is still never shed, while the
    // younger dribbler (now old enough) is.
    live.erase(live.begin(), live.begin() + 2);  // the two shed ones are gone
    const auto w = g.decide(live, t + 2600ms, t + 2600ms);
    EXPECT_EQ(keys_of(w, Reason::Overload), (std::vector<std::pair<std::string, int>>{{"10.0.1.3", 3}}));
    EXPECT_EQ(w.size(), 1u);
}

TEST(ConnGuardUnit, HttpQueueMustBeBounded) {
    api::ServerConfig cfg;
    cfg.http_queue = 0;  // httplib would treat 0 as unbounded
    EXPECT_THROW(api::validate_server_config(cfg), halo::Error);
    cfg.http_queue = 1;
    EXPECT_NO_THROW(api::validate_server_config(cfg));
}

// ---- shutdown through a verified duplicate (S-45) --------------------------------------------

struct Pair {
    int client = -1, server = -1;
};

int listen_loopback(int& port) {
    const int l = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (l < 0 || ::bind(l, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || ::listen(l, 8) != 0) return -1;  // NOLINT
    socklen_t len = sizeof(a);
    ::getsockname(l, reinterpret_cast<sockaddr*>(&a), &len);  // NOLINT: sockets API
    port = ntohs(a.sin_port);
    return l;
}

Pair connect_pair(int listener, int port) {
    Pair p;
    p.client = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<std::uint16_t>(port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(p.client, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) return p;  // NOLINT: sockets API
    p.server = ::accept(listener, nullptr, nullptr);
    return p;
}

ConnectionGuard::Victim victim_for(int server_fd) {
    ConnectionGuard::Victim v;
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    ::getpeername(server_fd, reinterpret_cast<sockaddr*>(&peer), &len);  // NOLINT: sockets API
    char buf[INET_ADDRSTRLEN] = {};
    ::inet_ntop(AF_INET, &peer.sin_addr, buf, sizeof(buf));
    v.key = {buf, ntohs(peer.sin_port)};
    v.fd = server_fd;
    struct stat st {};
    ::fstat(server_fd, &st);
    v.ino = static_cast<std::uint64_t>(st.st_ino);
    return v;
}

/// 0 = orderly EOF from the server, 1 = still open (nothing to read), -1 = error.
int client_state(int fd) {
    char c = 0;
    for (int i = 0; i < 50; ++i) {
        const ssize_t n = ::recv(fd, &c, 1, MSG_DONTWAIT);
        if (n == 0) return 0;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) return -1;
        std::this_thread::sleep_for(2ms);
    }
    return 1;
}

TEST(ConnGuardUnit, ShutdownActsOnTheCheckedSocketEvenIfTheNumberIsReused) {
    int port = 0;
    const int l = listen_loopback(port);
    ASSERT_GE(l, 0);
    const Pair a = connect_pair(l, port);
    const Pair b = connect_pair(l, port);
    ASSERT_GE(a.server, 0);
    ASSERT_GE(b.server, 0);
    const ConnectionGuard::Victim v = victim_for(a.server);
    // Between the identity check and the shutdown, the worker closes a's descriptor and the
    // number is reused for b's connection (dup2 does both atomically).
    const bool done = ConnectionGuard::shutdown_verified(v, port, [&] { ASSERT_EQ(::dup2(b.server, a.server), a.server); });
    EXPECT_TRUE(done);
    EXPECT_EQ(client_state(a.client), 0) << "the checked connection was shut down";
    EXPECT_EQ(client_state(b.client), 1) << "the connection that now owns the number was shut down (S-45)";
    // A stale identity (inode) is refused outright.
    const Pair c = connect_pair(l, port);
    ASSERT_GE(c.server, 0);
    ConnectionGuard::Victim stale = victim_for(c.server);
    stale.ino += 1;
    EXPECT_FALSE(ConnectionGuard::shutdown_verified(stale, port));
    EXPECT_EQ(client_state(c.client), 1);
    for (int fd : {a.client, a.server, b.client, b.server, c.client, c.server, l}) ::close(fd);
}

// ---- the server under slow clients (S-38) ----------------------------------------------------

/// A raw connection from source address `from` that sends a partial request and then one
/// header line every `every`; with `reconnect`, it reopens as soon as the server closes it.
class Dribbler {
public:
    Dribbler(int port, std::string from, std::chrono::milliseconds every, bool reconnect)
        : port_(port), from_(std::move(from)), every_(every), reconnect_(reconnect) {
        fd_ = open_one();
        thread_ = std::thread([this] { run(); });
    }
    ~Dribbler() {
        stop_ = true;
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0) ::close(fd_);
    }
    Dribbler(const Dribbler&) = delete;
    Dribbler& operator=(const Dribbler&) = delete;
    [[nodiscard]] bool connected() const { return first_ok_; }
    [[nodiscard]] int closed_by_server() const { return closed_.load(); }

private:
    int open_one() {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        sockaddr_in src{};
        src.sin_family = AF_INET;
        ::inet_pton(AF_INET, from_.c_str(), &src.sin_addr);
        sockaddr_in dst{};
        dst.sin_family = AF_INET;
        dst.sin_port = htons(static_cast<std::uint16_t>(port_));
        dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(fd, reinterpret_cast<sockaddr*>(&src), sizeof(src)) != 0 ||      // NOLINT: sockets API
            ::connect(fd, reinterpret_cast<sockaddr*>(&dst), sizeof(dst)) != 0) {  // NOLINT: sockets API
            ::close(fd);
            return -1;
        }
        static constexpr std::string_view head = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n";
        (void)::send(fd, head.data(), head.size(), MSG_NOSIGNAL);
        if (!first_done_) first_ok_ = true;
        first_done_ = true;
        return fd;
    }
    void run() {
        while (!stop_.load()) {
            if (fd_ < 0) {
                if (!reconnect_) return;
                fd_ = open_one();
                if (fd_ < 0) std::this_thread::sleep_for(20ms);
                continue;
            }
            // Sleep in small steps so a close by the server is noticed promptly.
            const auto until = std::chrono::steady_clock::now() + every_;
            bool closed = false;
            while (!closed && !stop_.load() && std::chrono::steady_clock::now() < until) {
                std::this_thread::sleep_for(10ms);
                char c = 0;
                const ssize_t n = ::recv(fd_, &c, 1, MSG_DONTWAIT);
                closed = n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
            }
            if (!closed && !stop_.load()) closed = ::send(fd_, "X-a: b\r\n", 8, MSG_NOSIGNAL) < 0;
            if (closed) {
                ++closed_;
                ::close(fd_);
                fd_ = -1;
            }
        }
    }
    int port_;
    std::string from_;
    std::chrono::milliseconds every_;
    bool reconnect_;
    int fd_ = -1;
    bool first_ok_ = false, first_done_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<int> closed_{0};
    std::thread thread_;
};

Script slow_answer() {
    Script s;
    s.text = "</think>\n\n" + std::string(24, 'z');
    s.delay = 100ms;
    return s;
}

int metric(const TestServer& ts, const std::string& series) {
    auto r = ts.get("/metrics");
    if (!r) return -1;
    const auto p = r->body.find("\n" + series + " ");  // the sample line, not "# HELP <series> ..."
    return p == std::string::npos ? -1 : std::stoi(r->body.substr(p + series.size() + 2));
}

TEST(ApiGuard, AQueuedCompleteRequestIsNotCutOffByTheHeaderDeadline) {
    // S-38 a: every worker runs a generation longer than header_timeout; a /health that
    // waits in httplib's queue with its request already sent must be served when a worker
    // frees up, not shut down because its connection was accepted header_timeout ago.
    api::ServerConfig cfg;
    cfg.http_threads = 2;
    cfg.header_timeout = 800ms;
    TestServer ts(cfg);
    ts.engine->script = [](const auto&, int) { return slow_answer(); };
    std::atomic<int> slow_ok{0};
    std::vector<std::thread> gens;
    for (int i = 0; i < 2; ++i) {
        gens.emplace_back([&] {
            auto r = make_client(ts.port, 30s)->Post("/v1/completions", Json{{"prompt", "x"}}.dump(), "application/json");
            if (r && r->status == 200) ++slow_ok;
        });
    }
    std::this_thread::sleep_for(300ms);
    const auto t0 = std::chrono::steady_clock::now();
    auto r = make_client(ts.port, 30s)->Get("/health");
    const auto waited = std::chrono::steady_clock::now() - t0;
    for (auto& t : gens) t.join();
    EXPECT_EQ(slow_ok.load(), 2);
    ASSERT_TRUE(r) << "queued /health was cut off: " << httplib::to_string(r.error());
    EXPECT_EQ(r->status, 200);
    EXPECT_GT(waited, 1200ms) << "premise: /health waited in the queue longer than header_timeout";
    EXPECT_EQ(metric(ts, "halo_api_connection_deadline_closes_total"), 0);
}

TEST(ApiGuard, PerPeerLimitFreesWorkersHeldByOneAddress) {
    // S-38 b: six dribblers from 127.0.0.2 hold every worker; the header deadline (30 s) and
    // shedding are out of the picture. The cap (3) closes the three oldest; /health from
    // another address is served, and the three under the cap are left alone.
    api::ServerConfig cfg;
    cfg.http_threads = 6;
    cfg.header_timeout = 30s;
    cfg.header_shed_grace = 0ms;
    cfg.max_header_connections_per_peer = 3;
    TestServer ts(cfg);
    std::vector<std::unique_ptr<Dribbler>> d;
    for (int i = 0; i < 6; ++i) {
        d.push_back(std::make_unique<Dribbler>(ts.port, "127.0.0.2", 300ms, false));
        ASSERT_TRUE(d.back()->connected());
        std::this_thread::sleep_for(20ms);
    }
    std::this_thread::sleep_for(300ms);
    const auto t0 = std::chrono::steady_clock::now();
    auto r = make_client(ts.port, 5s)->Get("/health");
    ASSERT_TRUE(r) << "/health starved by one address: " << httplib::to_string(r.error());
    EXPECT_EQ(r->status, 200);
    EXPECT_LT(std::chrono::steady_clock::now() - t0, 3s);
    std::this_thread::sleep_for(1500ms);
    int closed = 0;
    for (const auto& x : d) closed += x->closed_by_server();
    EXPECT_EQ(closed, 3) << "exactly the connections beyond the cap are closed";
    EXPECT_EQ(metric(ts, "halo_api_connection_closes_total{reason=\"peer_limit\"}"), 3);
}

TEST(ApiGuard, OverloadSheddingFreesWorkersHeldByManyAddresses) {
    // S-38 b: one dribbler per address (so no per-peer cap can help) holds every worker, the
    // header deadline is 30 s. A complete /health waits a scan interval, which triggers
    // shedding of the dribblers older than the grace.
    api::ServerConfig cfg;
    cfg.http_threads = 4;
    cfg.header_timeout = 30s;
    cfg.max_header_connections_per_peer = 0;
    cfg.header_shed_grace = 500ms;
    TestServer ts(cfg);
    std::vector<std::unique_ptr<Dribbler>> d;
    for (int i = 0; i < 4; ++i) {
        d.push_back(std::make_unique<Dribbler>(ts.port, "127.0.0." + std::to_string(10 + i), 300ms, false));
        ASSERT_TRUE(d.back()->connected());
    }
    std::this_thread::sleep_for(300ms);
    const auto t0 = std::chrono::steady_clock::now();
    auto r = make_client(ts.port, 5s)->Get("/health");
    ASSERT_TRUE(r) << "/health starved: " << httplib::to_string(r.error());
    EXPECT_EQ(r->status, 200);
    EXPECT_LT(std::chrono::steady_clock::now() - t0, 3s);
    EXPECT_GE(metric(ts, "halo_api_connection_closes_total{reason=\"overload\"}"), 1);
    EXPECT_EQ(metric(ts, "halo_api_connection_closes_total{reason=\"deadline\"}"), 0);
}

TEST(ApiGuard, RollingReconnectDribblersCannotStarveHealth) {
    // The reviewer's S-38 repro, scaled down: 40 dribblers on the same address as the
    // legitimate client, each reopened as soon as the server closes it, against 8 workers.
    api::ServerConfig cfg;
    cfg.http_threads = 8;
    cfg.http_queue = 64;
    TestServer ts(cfg);
    std::vector<std::unique_ptr<Dribbler>> d;
    for (int i = 0; i < 40; ++i) d.push_back(std::make_unique<Dribbler>(ts.port, "127.0.0.1", 200ms, true));
    std::this_thread::sleep_for(1s);
    int ok = 0, total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < 6s) {
        auto c = make_client(ts.port, 3s);
        c->set_connection_timeout(3s);
        auto r = c->Get("/health");
        ++total;
        if (r && r->status == 200) ++ok;
        std::this_thread::sleep_for(200ms);
    }
    int reconnects = 0;
    for (const auto& x : d) reconnects += x->closed_by_server();
    d.clear();
    EXPECT_GE(ok * 10, total * 9) << "/health ok " << ok << "/" << total << " under 40 rolling dribblers (" << reconnects
                                  << " closed by the server)";
    EXPECT_GT(reconnects, 40) << "premise: the dribblers were being closed and came back";
}

}  // namespace
}  // namespace halo::test
