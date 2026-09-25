#pragma once
// In-process fake HTTP server on 127.0.0.1:<ephemeral> for the http_client / Ollama tests.
// Each accepted connection reads the request (headers + Content-Length body), calls the
// handler with (method, path, body) and writes the RAW response string it returns (so a
// test can send malformed responses), then closes. Runs until destroyed.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace halo::profiling::test {

class FakeHttpServer {
public:
    using Handler = std::function<std::string(const std::string& method, const std::string& path, const std::string& body)>;

    explicit FakeHttpServer(Handler h) : handler_(std::move(h)) {
        fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        int one = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        ::bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a));
        ::listen(fd_, 16);
        socklen_t len = sizeof(a);
        ::getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &len);
        port_ = ntohs(a.sin_port);
        thread_ = std::thread([this] { loop(); });
    }
    ~FakeHttpServer() {
        stop_ = true;
        thread_.join();
        ::close(fd_);
    }
    FakeHttpServer(const FakeHttpServer&) = delete;
    FakeHttpServer& operator=(const FakeHttpServer&) = delete;

    [[nodiscard]] std::uint16_t port() const { return port_; }
    [[nodiscard]] std::vector<std::string> requests() const {
        std::scoped_lock l(mu_);
        return requests_;
    }

    /// Standard JSON response with Content-Length.
    static std::string json(int status, const std::string& body) {
        return "HTTP/1.1 " + std::to_string(status) + " X\r\nContent-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    }

private:
    void loop() {
        while (!stop_) {
            pollfd p{fd_, POLLIN, 0};
            if (::poll(&p, 1, 50) <= 0) continue;
            const int c = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
            if (c < 0) continue;
            serve(c);
            ::close(c);
        }
    }
    void serve(int c) {
        std::string req;
        char buf[4096];
        std::size_t hdr_end = std::string::npos;
        while ((hdr_end = req.find("\r\n\r\n")) == std::string::npos) {
            const ssize_t n = ::recv(c, buf, sizeof(buf), 0);
            if (n <= 0) return;
            req.append(buf, static_cast<std::size_t>(n));
        }
        std::size_t want = 0;
        const auto cl = req.find("Content-Length: ");
        if (cl != std::string::npos && cl < hdr_end) want = std::stoul(req.substr(cl + 16));
        while (req.size() < hdr_end + 4 + want) {
            const ssize_t n = ::recv(c, buf, sizeof(buf), 0);
            if (n <= 0) return;
            req.append(buf, static_cast<std::size_t>(n));
        }
        const auto sp1 = req.find(' ');
        const auto sp2 = req.find(' ', sp1 + 1);
        const std::string method = req.substr(0, sp1);
        const std::string path = req.substr(sp1 + 1, sp2 - sp1 - 1);
        const std::string body = req.substr(hdr_end + 4, want);
        {
            std::scoped_lock l(mu_);
            requests_.push_back(method + " " + path + " " + body);
        }
        const std::string resp = handler_(method, path, body);
        std::size_t off = 0;
        while (off < resp.size()) {
            const ssize_t n = ::send(c, resp.data() + off, resp.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return;
            off += static_cast<std::size_t>(n);
        }
    }

    Handler handler_;
    int fd_ = -1;
    std::uint16_t port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    mutable std::mutex mu_;
    std::vector<std::string> requests_;
};

}  // namespace halo::profiling::test
