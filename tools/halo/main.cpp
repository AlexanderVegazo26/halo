// `halo` entry point: wires the process environment, stdio, the real engine factory and
// signal handling into halo::cli::run_cli.
//
// Signals: SIGPIPE is ignored (a client or pipe reader going away must surface as a write
// error, never kill the server; RR-005). SIGINT/SIGTERM are blocked in every thread and
// taken by one sigwait thread: while `serve` runs they stop the server gracefully (in-flight
// generations are cancelled); otherwise they end the process with 128 + signo.

#include <pthread.h>

#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cli.h"

extern char** environ;

namespace {

std::mutex g_mu;
std::condition_variable g_cv;
bool g_serving = false;
bool g_stop_requested = false;

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);  // inherited by every thread created below
    std::thread([set] {
        while (true) {
            int sig = 0;
            if (sigwait(&set, &sig) != 0) continue;
            std::lock_guard lk(g_mu);
            if (!g_serving) std::_Exit(128 + sig);
            std::cerr << "\nhalo: signal " << sig << ", stopping the server\n";
            g_stop_requested = true;
            g_cv.notify_all();
        }
    }).detach();

    halo::cli::Context ctx;
    ctx.out = &std::cout;
    ctx.err = &std::cerr;
    ctx.in = &std::cin;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const std::string kv = *e;
        const auto eq = kv.find('=');
        if (eq != std::string::npos) ctx.env[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    ctx.engine_factory = halo::cli::default_engine_factory();
    ctx.argv0 = {argc > 0 ? argv[0] : "halo"};
    ctx.on_serving = [](halo::api::ApiServer&, std::function<void()> stop) {
        std::unique_lock lk(g_mu);
        g_serving = true;
        g_stop_requested = false;
        g_cv.wait(lk, [] { return g_stop_requested; });
        g_serving = false;
        lk.unlock();
        stop();  // cancels in-flight generations, fails queued requests, joins the listener
    };
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    const int rc = halo::cli::run_cli(args, ctx);
    std::cout.flush();
    return rc;
}
