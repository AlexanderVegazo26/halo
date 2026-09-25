#include "halo/profiling/http_client.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <format>

#include "halo/core/error.h"

namespace halo::profiling {

namespace {

constexpr std::size_t kMaxHeaderBytes = 64U << 10;

using Clock = std::chrono::steady_clock;

class Socket {
public:
    explicit Socket(int fd) : fd_(fd) {}
    ~Socket() {
        if (fd_ >= 0) ::close(fd_);
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    Socket& operator=(Socket&&) = delete;
    [[nodiscard]] int fd() const noexcept { return fd_; }

private:
    int fd_;
};

int ms_left(Clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return left <= 0 ? 0 : static_cast<int>(std::min<long long>(left, 1 << 30));
}

void wait_fd(int fd, short events, Clock::time_point deadline, const char* what) {
    while (true) {
        pollfd p{fd, events, 0};
        const int left = ms_left(deadline);
        HALO_CHECK(left > 0, ErrorCode::Io, "http: timeout while {}", what);
        const int rc = ::poll(&p, 1, left);
        if (rc < 0 && errno == EINTR) continue;
        HALO_CHECK(rc >= 0, ErrorCode::Io, "http: poll failed while {}: {}", what, std::strerror(errno));
        HALO_CHECK(rc > 0, ErrorCode::Io, "http: timeout while {}", what);
        return;
    }
}

Socket connect_loopback(const HttpRequest& req, Clock::time_point deadline) {
    sockaddr_storage ss{};
    socklen_t len = 0;
    int family = 0;
    if (req.host == "127.0.0.1") {
        auto* a = reinterpret_cast<sockaddr_in*>(&ss);
        a->sin_family = AF_INET;
        a->sin_port = htons(req.port);
        a->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        len = sizeof(sockaddr_in);
        family = AF_INET;
    } else if (req.host == "::1") {
        auto* a = reinterpret_cast<sockaddr_in6*>(&ss);
        a->sin6_family = AF_INET6;
        a->sin6_port = htons(req.port);
        a->sin6_addr = in6addr_loopback;
        len = sizeof(sockaddr_in6);
        family = AF_INET6;
    } else {
        throw_error(ErrorCode::Config, "http: host '{}' is not the loopback address 127.0.0.1 or ::1", req.host);
    }
    HALO_CHECK(req.port != 0, ErrorCode::Config, "http: port 0");
    Socket s(::socket(family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    HALO_CHECK(s.fd() >= 0, ErrorCode::Io, "http: socket failed: {}", std::strerror(errno));
    if (::connect(s.fd(), reinterpret_cast<const sockaddr*>(&ss), len) != 0) {
        HALO_CHECK(errno == EINPROGRESS, ErrorCode::Io, "http: connect {}:{} failed: {}", req.host, req.port,
                   std::strerror(errno));
        wait_fd(s.fd(), POLLOUT, deadline, "connecting");
        int err = 0;
        socklen_t el = sizeof(err);
        ::getsockopt(s.fd(), SOL_SOCKET, SO_ERROR, &err, &el);
        HALO_CHECK(err == 0, ErrorCode::Io, "http: connect {}:{} failed: {}", req.host, req.port, std::strerror(err));
    }
    return s;
}

void send_all(int fd, std::string_view data, Clock::time_point deadline) {
    std::size_t off = 0;
    while (off < data.size()) {
        wait_fd(fd, POLLOUT, deadline, "sending");
        const ssize_t n = ::send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        HALO_CHECK(n > 0, ErrorCode::Io, "http: send failed: {}", std::strerror(errno));
        off += static_cast<std::size_t>(n);
    }
}

/// Reads until close or until `limit` bytes are held (then stops reading).
bool read_some(int fd, std::string& buf, Clock::time_point deadline, std::size_t limit) {
    wait_fd(fd, POLLIN, deadline, "receiving");
    std::array<char, 65536> tmp{};
    while (true) {
        const ssize_t n = ::recv(fd, tmp.data(), std::min(tmp.size(), limit + 1 - std::min(limit, buf.size())), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && errno == EAGAIN) return true;
        HALO_CHECK(n >= 0, ErrorCode::Io, "http: recv failed: {}", std::strerror(errno));
        if (n == 0) return false;  // closed
        buf.append(tmp.data(), static_cast<std::size_t>(n));
        return true;
    }
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}

std::uint64_t parse_decimal(std::string_view s, const char* what) {
    std::uint64_t v = 0;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v, 10);
    HALO_CHECK(!s.empty() && ec == std::errc() && p == s.data() + s.size(), ErrorCode::Api, "http: bad {} '{}'", what,
               s);
    return v;
}

/// Decodes a complete chunked body; returns false if more bytes are needed.
bool decode_chunked(std::string_view raw, std::string& out, std::size_t max_body) {
    out.clear();
    std::size_t pos = 0;
    while (true) {
        const auto eol = raw.find("\r\n", pos);
        if (eol == std::string_view::npos) {
            HALO_CHECK(raw.size() - pos <= 1024, ErrorCode::Api, "http: chunk size line too long");
            return false;
        }
        std::string_view line = raw.substr(pos, eol - pos);
        if (const auto semi = line.find(';'); semi != std::string_view::npos) line = line.substr(0, semi);
        line = trim(line);
        std::uint64_t size = 0;
        const auto [p, ec] = std::from_chars(line.data(), line.data() + line.size(), size, 16);
        HALO_CHECK(!line.empty() && ec == std::errc() && p == line.data() + line.size(), ErrorCode::Api,
                   "http: bad chunk size '{}'", line);
        HALO_CHECK(size <= max_body && out.size() + size <= max_body, ErrorCode::Api,
                   "http: chunked body exceeds {} bytes", max_body);
        pos = eol + 2;
        if (size == 0) return raw.find("\r\n", pos) != std::string_view::npos;  // trailers end
        if (raw.size() - pos < size + 2) return false;
        out.append(raw.substr(pos, size));
        HALO_CHECK(raw.substr(pos + size, 2) == "\r\n", ErrorCode::Api, "http: chunk not terminated by CRLF");
        pos += size + 2;
    }
}

}  // namespace

HttpResponse http_request(const HttpRequest& req) {
    HALO_CHECK(req.method == "GET" || req.method == "POST", ErrorCode::Config, "http: method {}", req.method);
    HALO_CHECK(!req.path.empty() && req.path.front() == '/' && req.path.find_first_of("\r\n ") == std::string::npos,
               ErrorCode::Config, "http: invalid path");
    // Header values are written verbatim: CR, LF or NUL would inject header lines (S-35).
    HALO_CHECK(req.content_type.find_first_of(std::string_view("\r\n\0", 3)) == std::string::npos,
               ErrorCode::Config, "http: invalid content_type (CR/LF/NUL)");
    const auto deadline = Clock::now() + req.timeout;
    Socket s = connect_loopback(req, deadline);
    std::string head = std::format("{} {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\nAccept: application/json\r\n",
                                   req.method, req.path, req.host == "::1" ? "[::1]" : req.host, req.port);
    if (req.method == "POST") {
        head += std::format("Content-Type: {}\r\nContent-Length: {}\r\n", req.content_type, req.body.size());
    }
    head += "\r\n";
    send_all(s.fd(), head, deadline);
    if (req.method == "POST") send_all(s.fd(), req.body, deadline);

    // Header block.
    std::string buf;
    std::size_t hdr_end = std::string::npos;
    while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
        HALO_CHECK(buf.size() <= kMaxHeaderBytes, ErrorCode::Api, "http: header block exceeds {} bytes",
                   kMaxHeaderBytes);
        const bool open = read_some(s.fd(), buf, deadline, kMaxHeaderBytes + req.max_body_bytes);
        HALO_CHECK(open || buf.find("\r\n\r\n") != std::string::npos, ErrorCode::Api,
                   "http: connection closed before the end of the headers");
    }
    HttpResponse resp;
    const std::string_view hv(buf.data(), hdr_end);
    const auto first_eol = hv.find("\r\n");
    const std::string_view status_line = hv.substr(0, first_eol);
    HALO_CHECK(status_line.starts_with("HTTP/1.") && status_line.size() >= 12 && status_line[8] == ' ',
               ErrorCode::Api, "http: malformed status line '{}'", status_line.substr(0, 64));
    resp.status = static_cast<int>(parse_decimal(status_line.substr(9, 3), "status code"));
    HALO_CHECK(resp.status >= 100 && resp.status <= 599, ErrorCode::Api, "http: status {} out of range", resp.status);
    std::size_t pos = first_eol == std::string_view::npos ? hv.size() : first_eol + 2;
    while (pos < hv.size()) {
        const auto eol = std::min(hv.find("\r\n", pos), hv.size());
        const std::string_view line = hv.substr(pos, eol - pos);
        const auto colon = line.find(':');
        HALO_CHECK(colon != std::string_view::npos && colon > 0, ErrorCode::Api, "http: malformed header line");
        resp.headers[lower(std::string(trim(line.substr(0, colon))))] = std::string(trim(line.substr(colon + 1)));
        pos = eol + 2;
    }
    std::string raw = buf.substr(hdr_end + 4);
    const auto te = resp.headers.find("transfer-encoding");
    const auto cl = resp.headers.find("content-length");
    const bool chunked = te != resp.headers.end() && lower(te->second).find("chunked") != std::string::npos;
    if (chunked) {
        while (!decode_chunked(raw, resp.body, req.max_body_bytes)) {
            HALO_CHECK(raw.size() <= req.max_body_bytes + (1U << 20), ErrorCode::Api, "http: chunked body too large");
            const bool open = read_some(s.fd(), raw, deadline, req.max_body_bytes + (1U << 20));
            HALO_CHECK(open, ErrorCode::Api, "http: connection closed inside a chunked body");
        }
    } else if (cl != resp.headers.end()) {
        const std::uint64_t n = parse_decimal(trim(cl->second), "Content-Length");
        HALO_CHECK(n <= req.max_body_bytes, ErrorCode::Api, "http: Content-Length {} exceeds {}", n,
                   req.max_body_bytes);
        while (raw.size() < n) {
            const bool open = read_some(s.fd(), raw, deadline, static_cast<std::size_t>(n));
            HALO_CHECK(open || raw.size() >= n, ErrorCode::Api, "http: body truncated ({} of {} bytes)", raw.size(), n);
        }
        raw.resize(static_cast<std::size_t>(n));
        resp.body = std::move(raw);
    } else {
        while (read_some(s.fd(), raw, deadline, req.max_body_bytes)) {
            HALO_CHECK(raw.size() <= req.max_body_bytes, ErrorCode::Api, "http: body exceeds {} bytes",
                       req.max_body_bytes);
        }
        HALO_CHECK(raw.size() <= req.max_body_bytes, ErrorCode::Api, "http: body exceeds {} bytes", req.max_body_bytes);
        resp.body = std::move(raw);
    }
    return resp;
}

}  // namespace halo::profiling
