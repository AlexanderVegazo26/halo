// Child process for test_log_env: the HALO_LOG_* environment is read once per process, so every
// case runs this helper in a fresh process with its own environment (never through a shell).
//
//   halo_log_env_helper <mode>
//     emit           one line at each level trace..warn
//     explicit       set_level(Warn) before the first log line, then emit
//     explicit-json  set_json(false) before the first log line, then emit
//     late           emit, then set_level(Error) and log once at info and once at error
//     abort          emit, then std::abort() (lines written to HALO_LOG_FILE must survive)
//     fds            emit, then exec this binary as `listfds` (O_CLOEXEC check)
//     listfds        print the target of every open file descriptor, one per line, on stdout
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include "halo/core/log.h"

namespace {

void emit() {
    HALO_TRACE("t", "trace-line {}", 1);
    HALO_DEBUG("d", "debug-line");
    HALO_INFO("i", "info-line");
    HALO_WARN("w", "warn-line \"quoted\"");
}

int list_fds() {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/fd", ec)) {
        std::error_code ec2;
        const auto target = std::filesystem::read_symlink(e.path(), ec2);
        std::printf("%s -> %s\n", e.path().filename().c_str(), ec2 ? "?" : target.c_str());
    }
    return ec ? 3 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "emit";
    if (mode == "listfds") return list_fds();
    if (mode == "explicit") halo::log::set_level(halo::log::Level::Warn);
    if (mode == "explicit-json") halo::log::set_json(false);
    emit();
    if (mode == "late") {
        halo::log::set_level(halo::log::Level::Error);
        HALO_INFO("i", "late-info-line");
        HALO_ERROR("e", "late-error-line");
    }
    if (mode == "abort") std::abort();
    if (mode == "fds") {
        std::fflush(stdout);
        char self[] = "halo_log_env_helper";
        char list[] = "listfds";
        char* args[] = {self, list, nullptr};
        ::execv("/proc/self/exe", args);
        return 4;  // exec failed
    }
    return 0;
}
