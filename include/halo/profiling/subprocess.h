#pragma once
// Subprocess execution for the baseline adapters (TRD §51). No shell is ever involved:
// the program is exec'd directly with an argv vector (posix_spawn), so no argument is
// re-parsed, globbed or expanded. The program path must contain a '/' (no PATH search).
//
// * The child gets exactly `env` as its environment (plus, when inherit_env is set, the
//   parent's environment underneath it). Records store only `recorded_environment()`
//   (security review S-31): the NAMES of the variables passed explicitly plus inherited
//   ones with a GPU/runtime prefix; a VALUE only for an allowlisted, non-secret name
//   (RADV_ AMD_ HSA_ ROCR_ HIP_ GGML_ VK_ MESA_ prefixes). Every other value, and any name
//   containing KEY, TOKEN, SECRET, PASSWORD, AUTH or CREDENTIAL (case-insensitive, also
//   inside allowlisted prefixes and inherited LLAMA_ / OLLAMA_ names), is kRedacted.
// * Every fd HALO opens is O_CLOEXEC, and the child closes every fd >= 3 before exec
//   (posix_spawn_file_actions_addclosefrom_np), so an embedding program's fds are not
//   inherited (S-33).
// * stdout and stderr are read concurrently with poll() (reading them one after the other
//   deadlocks once the other pipe fills), each capped at max_output_bytes (then
//   `*_truncated` is set and the rest is drained and discarded).
// * The child runs in its own process group. On timeout the whole group gets SIGTERM,
//   then SIGKILL after kill_grace. The child is always reaped (waitpid) on every path,
//   including exceptions. Signals go only to a group whose leader has not been reaped
//   yet: exit is detected with waitid(WNOWAIT), the group is signalled while the zombie
//   still pins its pid/pgid, and only then is it reaped (S-32).
// * A waitpid failure (e.g. SIGCHLD set to SIG_IGN by the embedding program, or another
//   thread reaping with waitpid(-1)) is reported as status_unknown, never as exit 0 (S-30).
// * Limitation (S-34): a descendant that calls setsid() leaves the process group and
//   escapes the group kill. llama.cpp tools do not do this; a hard guarantee would need
//   PR_SET_CHILD_SUBREAPER plus a sweep, or a transient cgroup.
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

/// Recorded in place of a value that must not be stored.
inline constexpr std::string_view kRedacted = "<redacted>";

struct ProcessResult {
    int exit_code = -1;       ///< valid when !signaled && !status_unknown
    bool status_unknown = false;  ///< waitpid failed: the exit status was lost
    bool signaled = false;    ///< terminated by a signal
    int term_signal = 0;
    bool timed_out = false;
    std::string out;
    std::string err;
    bool out_truncated = false;
    bool err_truncated = false;
    double wall_s = 0;
    [[nodiscard]] bool ok() const noexcept { return !timed_out && !signaled && !status_unknown && exit_code == 0; }
};

/// Runs argv[0] with argv and waits. Throws Error(Config) for an empty argv or a program
/// without '/', Error(Io) when the program cannot be executed (missing, not executable).
[[nodiscard]] ProcessResult run_process(const std::vector<std::string>& argv, const ProcessOptions& options = {});

/// Environment variables that go into a record's invocation (see header comment).
[[nodiscard]] std::map<std::string, std::string> recorded_environment(const ProcessOptions& options);

/// Whether the TCP socket listening on 127.0.0.1 / 0.0.0.0 / :: port `port` is held open
/// by process `pid` (matches the LISTEN socket inodes of /proc/net/tcp{,6} against the
/// socket links in /proc/<pid>/fd). nullopt when it cannot be determined (no /proc access,
/// or nothing listens). Used to verify that a server answering on a port is the child that
/// was spawned (S-29).
[[nodiscard]] std::optional<bool> port_listener_is(int pid, std::uint16_t port);

/// True when a TCP bind to 127.0.0.1:`port` succeeds right now (the port is free).
[[nodiscard]] bool loopback_port_free(std::uint16_t port);

namespace detail {
/// Number of signals that were refused because their target had already been reaped
/// (must stay 0; exposed for the S-32 regression test).
[[nodiscard]] std::uint64_t signals_refused_after_reap() noexcept;
}  // namespace detail

/// A long-running child (e.g. llama-server). stdout+stderr go to `log_path` (or
/// /dev/null); the log is created with mode 0600 and O_NOFOLLOW, so a symlink at that path
/// makes the spawn fail with Error(Io) instead of truncating the link target (S-37). The destructor terminates the process group (SIGTERM, grace, SIGKILL) and
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
    std::optional<int> exit_code_;  ///< 128+signal when signalled; nullopt if status was lost
    std::chrono::milliseconds grace_;
};

}  // namespace halo::profiling
