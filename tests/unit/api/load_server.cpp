// Test-only HTTP server for the k6 load layer (load/run.py, load/*.js): the real ApiServer
// (admission, connection guard, JSON parsing, chat templating, SSE streaming, metrics) over the
// deterministic FakeEngine, so load tests measure HALO's API layer without a model or a GPU.
//
//   halo_load_server [--concurrent=N] [--queue=N] [--token-delay-ms=N]
//
// Binds an ephemeral 127.0.0.1 port, prints "PORT <n>" on stdout, serves until SIGTERM/SIGINT,
// then stops the server (in-flight generations cancelled) and exits 0.

#include <pthread.h>
#include <signal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

#include "fake_engine.h"
#include "halo/api/server.h"

namespace {

std::size_t parse_size(std::string_view arg, std::string_view prefix) {
    const std::string v(arg.substr(prefix.size()));
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
    if (v.empty() || end == nullptr || *end != '\0') {
        std::fprintf(stderr, "halo_load_server: %.*s needs a number, got '%s'\n", static_cast<int>(prefix.size()),
                     prefix.data(), v.c_str());
        std::exit(2);
    }
    return static_cast<std::size_t>(n);
}

}  // namespace

int main(int argc, char** argv) {
    halo::api::ServerConfig cfg;
    cfg.port = 0;
    std::size_t token_delay_ms = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a.starts_with("--concurrent=")) {
            cfg.max_concurrent = parse_size(a, "--concurrent=");
        } else if (a.starts_with("--queue=")) {
            cfg.max_queue = parse_size(a, "--queue=");
        } else if (a.starts_with("--token-delay-ms=")) {
            token_delay_ms = parse_size(a, "--token-delay-ms=");
        } else {
            std::fprintf(stderr, "halo_load_server: unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }

    // Block the stop signals before any server thread exists, so every thread inherits the mask
    // and only sigwait() below receives them.
    sigset_t stop_signals;
    sigemptyset(&stop_signals);
    sigaddset(&stop_signals, SIGTERM);
    sigaddset(&stop_signals, SIGINT);
    pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

    try {
        auto engine = halo::test::FakeEngine::synthetic();
        // A reply long enough to stream as many SSE chunks (one per byte token).
        engine->script = [token_delay_ms](const halo::runtime::GenerateRequest&, int) {
            halo::test::Script s;
            s.text = "The quick brown fox jumps over the lazy dog.";
            s.delay = std::chrono::milliseconds(token_delay_ms);
            return s;
        };
        halo::api::ApiServer server(*engine, cfg);
        server.start();
        std::printf("PORT %d\n", server.port());
        std::fflush(stdout);

        int sig = 0;
        sigwait(&stop_signals, &sig);
        server.stop();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "halo_load_server: %s\n", e.what());
        return 1;
    }
    return 0;
}
