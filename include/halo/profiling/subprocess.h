#pragma once
// Subprocess execution for the baseline adapters (TRD §51). No shell is ever involved:
// the program is exec'd directly with an argv vector (posix_spawn), so no argument is
// re-parsed, globbed or expanded. The program path must contain a '/' (no PATH search).
//
// * The child gets exactly `env` as its environment (plus, when inherit_env is set, the
//   parent's environment underneath it). Records store only `recorded_environment()`:
//   the variables passed explicitly plus inherited ones with a GPU/runtime prefix, never
//   the full parent environment (which may hold secrets).
// * stdout and stderr are read concurrently with poll() (reading them one after the other
//   deadlocks once the other pipe fills), each capped at max_output_bytes (then
//   `*_truncated` is set and the rest is drained and discarded).
// * The child runs in its own process group. On timeout the whole group gets SIGTERM,
//   then SIGKILL after kill_grace. The child is always reaped (waitpid) on every path,
//   including exceptions.
//
// Thread-safety: run_process may be called concurrently from several threads.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace halo::profiling {

struct ProcessOptions {
    std::map<std::string, std::string> env;
    bool inherit_env = false;
    std::chrono::milliseconds timeout{std::chrono::minutes(60)};
    std::chrono::milliseconds kill_grace{std::chrono::seconds(2)};
    std::size_t max_output_bytes = 16U << 20;
};

struct ProcessResult {
    int exit_code = -1;       ///< valid when !signaled
    bool signaled = false;    ///< terminated by a signal
    int term_signal = 0;
    bool timed_out = false;
    std::string out;
    std::string err;
    bool out_truncated = false;
    bool err_truncated = false;
    double wall_s = 0;
    [[nodiscard]] bool ok() const noexcept { return !timed_out && !signaled && exit_code == 0; }
};

/// Runs argv[0] with argv and waits. Throws Error(Config) for an empty argv or a program
/// without '/', Error(Io) when the program cannot be executed (missing, not executable).
[[nodiscard]] ProcessResult run_process(const std::vector<std::string>& argv, const ProcessOptions& options = {});

/// Environment variables that go into a record's invocation (see header comment).
[[nodiscard]] std::map<std::string, std::string> recorded_environment(const ProcessOptions& options);

/// A long-running child (e.g. llama-server). stdout+stderr go to `log_path` (or
/// /dev/null). The destructor terminates the process group (SIGTERM, grace, SIGKILL) and
/// reaps it, so a server never outlives its owner.
class ChildProcess {
public:
    ChildProcess(const std::vector<std::string>& argv, const ProcessOptions& options,
                 const std::optional<std::filesystem::path>& log_path = std::nullopt);
    ~ChildProcess();
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

    [[nodiscard]] int pid() const noexcept { return pid_; }
    /// True while the child has not exited (reaps it if it has).
    [[nodiscard]] bool running();
    /// SIGTERM the group, wait up to grace, then SIGKILL; reaps. Idempotent.
    void terminate();
    /// Exit status after it ended (nullopt while running).
    [[nodiscard]] std::optional<int> exit_code() const noexcept { return exit_code_; }

private:
    int pid_ = -1;
    bool reaped_ = false;
    std::optional<int> exit_code_;
    std::chrono::milliseconds grace_;
};

}  // namespace halo::profiling
