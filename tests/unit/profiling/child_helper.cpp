// Test helper executable for the subprocess and baseline tests (no shell involved).
// Behaviour is selected by the HALO_FAKE_MODE environment variable:
//   spam           write HALO_FAKE_BYTES bytes to stdout AND stderr, interleaved
//   exit           exit with HALO_FAKE_CODE
//   ignore_term    ignore SIGTERM and sleep 60 s (tests SIGKILL escalation)
//   grandchild     fork a child that sleeps 60 s, then sleep 60 s (tests group kill);
//                  writes the grandchild pid to HALO_FAKE_PIDFILE
//   close_sleep    close stdout/stderr, then sleep 60 s (tests the deadline after EOF)
//   fds            print every open fd number (S-33 fd-leak test)
//   fake_server    a grandchild serves HALO_FAKE_JSON on HALO_FAKE_PORT (S-29 ownership test)
//   argv           print each argv[i] (i >= 1) followed by '\n'
//   llama_bench    "--version" -> version text on stderr; otherwise cat HALO_FAKE_JSON to
//                  stdout (and exit HALO_FAKE_CODE, default 0)

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string env(const char* k, const char* def = "") {
    const char* v = std::getenv(k);
    return v != nullptr ? v : def;
}

void write_all(int fd, const char* p, std::size_t n) {
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w <= 0) return;
        p += w;
        n -= static_cast<std::size_t>(w);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = env("HALO_FAKE_MODE");
    if (mode == "spam") {
        const std::size_t total = std::stoul(env("HALO_FAKE_BYTES", "1048576"));
        const std::string block(4096, 'x');
        for (std::size_t done = 0; done < total; done += block.size()) {
            write_all(1, block.data(), block.size());
            write_all(2, block.data(), block.size());
        }
        return 0;
    }
    if (mode == "exit") return std::atoi(env("HALO_FAKE_CODE", "0").c_str());
    if (mode == "ignore_term") {
        ::signal(SIGTERM, SIG_IGN);
        ::sleep(60);
        return 0;
    }
    if (mode == "grandchild") {
        const pid_t g = ::fork();
        if (g == 0) {
            ::sleep(60);
            return 0;
        }
        std::ofstream(env("HALO_FAKE_PIDFILE")) << g << "\n";
        ::sleep(60);
        return 0;
    }
    if (mode == "close_sleep") {
        ::close(1);
        ::close(2);
        ::sleep(60);
        return 0;
    }
    if (mode == "fake_server") {
        // "--version" answers like llama.cpp; otherwise a GRANDCHILD listens on
        // 127.0.0.1:HALO_FAKE_PORT and answers every request with HALO_FAKE_JSON, while this
        // (direct) child holds no socket: an in-group impostor for the S-29 ownership check.
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--version") {
                std::cerr << "version: 0.5.0-dev (build 1, commit bd4f514)\n";
                return 0;
            }
        }
        std::ifstream in(env("HALO_FAKE_JSON"), std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string body = ss.str();
        const int port = std::atoi(env("HALO_FAKE_PORT", "0").c_str());
        const pid_t g = ::fork();
        if (g == 0) {
            ::alarm(30);  // never outlive a test, even if a mutation breaks the group kill
            const int s = ::socket(AF_INET, SOCK_STREAM, 0);
            int one = 1;
            ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<std::uint16_t>(port));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || ::listen(s, 8) != 0) return 3;
            for (int n = 0; n < 200; ++n) {
                const int c = ::accept(s, nullptr, nullptr);
                if (c < 0) continue;
                char buf[8192];
                (void)::recv(c, buf, sizeof(buf), 0);
                const std::string resp = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
                                         "\r\nConnection: close\r\n\r\n" + body;
                write_all(c, resp.data(), resp.size());
                ::close(c);
            }
            return 0;
        }
        ::sleep(60);
        return 0;
    }
    if (mode == "fds") {  // list open fds (one per line), excluding the listing's own dir fd
        for (int fd = 0; fd < 1024; ++fd) {
            if (::fcntl(fd, F_GETFD) != -1) {
                const std::string s = std::to_string(fd) + "\n";
                write_all(1, s.data(), s.size());
            }
        }
        return 0;
    }
    if (mode == "argv") {
        for (int i = 1; i < argc; ++i) {
            write_all(1, argv[i], std::strlen(argv[i]));
            write_all(1, "\n", 1);
        }
        return 0;
    }
    if (mode == "llama_bench") {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--version") {
                std::cerr << env("HALO_FAKE_VERSION", "version: 0.5.0-dev (build 1, commit bd4f514)\n"
                                                     "built with Clang 18.1.3 for Linux x86_64\n");
                return 0;
            }
        }
        std::ifstream in(env("HALO_FAKE_JSON"), std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        std::cout << ss.str();
        std::cout.flush();
        return std::atoi(env("HALO_FAKE_CODE", "0").c_str());
    }
    std::cerr << "child_helper: unknown HALO_FAKE_MODE '" << mode << "'\n";
    return 99;
}
