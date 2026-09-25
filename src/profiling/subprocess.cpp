#include "halo/profiling/subprocess.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#include "halo/core/error.h"
#include "halo/core/log.h"

extern char** environ;

namespace halo::profiling {

namespace {

std::atomic<std::uint64_t> g_signals_refused{0};

// ---- recorded environment (S-31) ----------------------------------------------------------

/// Inherited variables whose NAMES are recorded (GPU runtime / llama.cpp / Ollama knobs).
constexpr std::array<std::string_view, 10> kRecordedPrefixes{"RADV_", "AMD_",  "HSA_",   "ROCR_",  "HIP_",
                                                             "GGML_", "VK_",   "LLAMA_", "OLLAMA_", "MESA_"};
/// Names whose VALUES may be recorded (driver/runtime tuning knobs, never credentials).
/// LLAMA_ and OLLAMA_ are deliberately absent: llama-server reads LLAMA_API_KEY and
/// LLAMA_ARG_* may carry keys, so those are recorded by name only.
constexpr std::array<std::string_view, 8> kValueAllowlist{"RADV_", "AMD_", "HSA_", "ROCR_",
                                                          "HIP_",  "GGML_", "VK_", "MESA_"};
constexpr std::array<std::string_view, 6> kSecretMarkers{"KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL"};

bool has_prefix(std::string_view key, std::span<const std::string_view> prefixes) {
    return std::any_of(prefixes.begin(), prefixes.end(), [&](std::string_view p) { return key.starts_with(p); });
}

bool looks_secret(std::string_view key) {
    std::string up(key);
    std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return std::any_of(kSecretMarkers.begin(), kSecretMarkers.end(),
                       [&](std::string_view m) { return up.find(m) != std::string::npos; });
}

std::string recorded_value(std::string_view key, std::string_view value) {
    if (looks_secret(key) || !has_prefix(key, kValueAllowlist)) return std::string(kRedacted);
    return std::string(value);
}

/// Final environment: inherited (optional) overridden by options.env.
std::map<std::string, std::string> build_env(const ProcessOptions& o) {
    std::map<std::string, std::string> env;
    if (o.inherit_env) {
        for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
            const std::string_view kv(*e);
            const auto eq = kv.find('=');
            if (eq == std::string_view::npos || eq == 0) continue;
            env.emplace(std::string(kv.substr(0, eq)), std::string(kv.substr(eq + 1)));
        }
    }
    for (const auto& [k, v] : o.env) {
        HALO_CHECK(!k.empty() && k.find('=') == std::string::npos && k.find('\0') == std::string::npos &&
                       v.find('\0') == std::string::npos,
                   ErrorCode::Config, "subprocess: invalid environment variable '{}'", k);
        env[k] = v;
    }
    return env;
}

/// Owns a NUL-terminated char* array over strings it also owns.
struct CStrings {
    std::vector<std::string> storage;
    std::vector<char*> ptrs;
    void finish() {
        ptrs.clear();
        for (auto& s : storage) ptrs.push_back(s.data());
        ptrs.push_back(nullptr);
    }
};

void check_argv(const std::vector<std::string>& argv) {
    HALO_CHECK(!argv.empty(), ErrorCode::Config, "subprocess: empty argv");
    HALO_CHECK(argv[0].find('/') != std::string::npos, ErrorCode::Config,
               "subprocess: program '{}' must be a path (no PATH search)", argv[0]);
    for (const auto& a : argv) {
        HALO_CHECK(a.find('\0') == std::string::npos, ErrorCode::Config, "subprocess: NUL byte in an argument");
    }
    HALO_CHECK(::access(argv[0].c_str(), X_OK) == 0, ErrorCode::Io, "subprocess: cannot execute '{}': {}", argv[0],
               std::strerror(errno));
}

/// RAII over posix_spawn attribute/file-action objects.
struct SpawnSetup {
    posix_spawnattr_t attr{};
    posix_spawn_file_actions_t fa{};
    SpawnSetup() {
        posix_spawnattr_init(&attr);
        posix_spawn_file_actions_init(&fa);
        sigset_t none;
        sigemptyset(&none);
        sigset_t defaults;
        sigemptyset(&defaults);
        sigaddset(&defaults, SIGPIPE);
        sigaddset(&defaults, SIGTERM);
        sigaddset(&defaults, SIGINT);
        posix_spawnattr_setsigmask(&attr, &none);
        posix_spawnattr_setsigdefault(&attr, &defaults);
        posix_spawnattr_setpgroup(&attr, 0);  // own process group
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    }
    ~SpawnSetup() {
        posix_spawn_file_actions_destroy(&fa);
        posix_spawnattr_destroy(&attr);
    }
    SpawnSetup(const SpawnSetup&) = delete;
    SpawnSetup& operator=(const SpawnSetup&) = delete;
};

int spawn(const std::vector<std::string>& argv, const ProcessOptions& o, SpawnSetup& setup) {
    CStrings args;
    args.storage = argv;
    args.finish();
    CStrings envs;
    for (const auto& [k, v] : build_env(o)) envs.storage.push_back(k + "=" + v);
    envs.finish();
    // Last file action: close every fd >= 3 in the child, so fds the embedding program
    // opened without O_CLOEXEC never reach the benchmarked tool (S-33).
    const int crc = posix_spawn_file_actions_addclosefrom_np(&setup.fa, 3);
    HALO_CHECK(crc == 0, ErrorCode::Io, "subprocess: addclosefrom failed: {}", std::strerror(crc));
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, argv[0].c_str(), &setup.fa, &setup.attr, args.ptrs.data(), envs.ptrs.data());
    HALO_CHECK(rc == 0, ErrorCode::Io, "subprocess: posix_spawn '{}' failed: {}", argv[0], std::strerror(rc));
    return pid;
}

struct Fd {
    int fd = -1;
    Fd() = default;
    explicit Fd(int f) : fd(f) {}
    ~Fd() { reset(); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    void reset() {
        if (fd >= 0) ::close(fd);
        fd = -1;
    }
};

void make_pipe(Fd& r, Fd& w) {
    int p[2];
    HALO_CHECK(::pipe2(p, O_CLOEXEC) == 0, ErrorCode::Io, "subprocess: pipe2 failed: {}", std::strerror(errno));
    r.fd = p[0];
    w.fd = p[1];
}

// ---- child lifecycle (S-30, S-32) ---------------------------------------------------------

struct Exit {
    bool known = false;  ///< false: waitpid/waitid failed, status lost (S-30)
    int status = 0;
};

enum class State : std::uint8_t { Running, Exited, Lost };

/// A spawned child that is the leader of its own process group. Signals are only sent while
/// the leader has not been reaped: running, or a zombie that still pins the pid and pgid.
class Child {
public:
    Child(pid_t pid, bool reaped) : pid_(pid), reaped_(reaped) {}
    [[nodiscard]] bool reaped() const noexcept { return reaped_; }

    /// Non-blocking; does not reap (WNOWAIT).
    State state(bool block = false) {
        if (reaped_) return State::Lost;
        while (true) {
            siginfo_t si{};
            si.si_pid = 0;
            const int flags = WEXITED | WNOWAIT | (block ? 0 : WNOHANG);
            if (::waitid(P_PID, static_cast<id_t>(pid_), &si, flags) == 0) {
                return si.si_pid == 0 ? State::Running : State::Exited;
            }
            if (errno == EINTR) continue;
            // ECHILD: auto-reaped (SIGCHLD = SIG_IGN) or reaped by someone else. The pid may
            // already belong to another process, so it is never signalled again.
            reaped_ = true;
            return State::Lost;
        }
    }

    void signal_group(int sig) {
        if (reaped_) {
            g_signals_refused.fetch_add(1);
            return;
        }
        (void)::kill(-pid_, sig);  // no single-pid fallback: the child is the group leader
    }

    /// The child has exited (zombie): kill whatever is left in its group, then reap.
    Exit reap() {
        if (reaped_) return {};
        signal_group(SIGKILL);
        int status = 0;
        while (::waitpid(pid_, &status, 0) < 0) {
            if (errno == EINTR) continue;
            reaped_ = true;
            return {};
        }
        reaped_ = true;
        return {true, status};
    }

    /// SIGTERM the group, wait up to grace, then SIGKILL; reaps.
    Exit terminate(std::chrono::milliseconds grace) {
        State s = state();
        if (s == State::Exited) return reap();
        if (s == State::Lost) return {};
        signal_group(SIGTERM);
        const auto until = std::chrono::steady_clock::now() + grace;
        while (std::chrono::steady_clock::now() < until) {
            s = state();
            if (s == State::Exited) return reap();
            if (s == State::Lost) return {};
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        signal_group(SIGKILL);
        s = state(/*block=*/true);
        if (s == State::Exited) return reap();
        return {};
    }

private:
    pid_t pid_;
    bool reaped_;
};

/// Reaps a child if a scope is left by an exception.
struct ChildGuard {
    Child& child;
    std::chrono::milliseconds grace;
    bool done = false;
    ~ChildGuard() {
        if (!done) (void)child.terminate(grace);
    }
};

int exit_code_of(const Exit& e) {
    if (!e.known) return -1;
    if (WIFEXITED(e.status)) return WEXITSTATUS(e.status);
    if (WIFSIGNALED(e.status)) return 128 + WTERMSIG(e.status);
    return -1;
}

// ---- /proc socket ownership (S-29) --------------------------------------------------------

std::set<std::string> listen_inodes(std::uint16_t port) {
    std::set<std::string> out;
    for (const char* file : {"/proc/net/tcp", "/proc/net/tcp6"}) {
        std::ifstream in(file);
        std::string line;
        std::getline(in, line);  // header
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string sl, local, rem, st, txrx, trtm, retr, uid, timeout, inode;
            if (!(ls >> sl >> local >> rem >> st >> txrx >> trtm >> retr >> uid >> timeout >> inode)) continue;
            if (st != "0A") continue;  // TCP_LISTEN
            const auto colon = local.rfind(':');
            if (colon == std::string::npos) continue;
            unsigned long p = 0;
            try {
                p = std::stoul(local.substr(colon + 1), nullptr, 16);
            } catch (const std::exception&) {
                continue;
            }
            if (p == port) out.insert(inode);
        }
    }
    return out;
}

}  // namespace

namespace detail {
std::uint64_t signals_refused_after_reap() noexcept { return g_signals_refused.load(); }
}  // namespace detail

std::map<std::string, std::string> recorded_environment(const ProcessOptions& o) {
    std::map<std::string, std::string> out;
    if (o.inherit_env) {
        for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
            const std::string_view kv(*e);
            const auto eq = kv.find('=');
            if (eq == std::string_view::npos) continue;
            const std::string_view k = kv.substr(0, eq);
            if (has_prefix(k, kRecordedPrefixes)) out.emplace(std::string(k), recorded_value(k, kv.substr(eq + 1)));
        }
    }
    for (const auto& [k, v] : o.env) out[k] = recorded_value(k, v);
    return out;
}

std::optional<bool> port_listener_is(int pid, std::uint16_t port) {
    const auto inodes = listen_inodes(port);
    if (inodes.empty()) return std::nullopt;
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path("/proc") / std::to_string(pid) / "fd";
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) return std::nullopt;
    for (const auto& e : it) {
        std::error_code lec;
        const std::string target = std::filesystem::read_symlink(e.path(), lec).string();
        if (lec || !target.starts_with("socket:[") || target.back() != ']') continue;
        if (inodes.contains(target.substr(8, target.size() - 9))) return true;
    }
    return false;
}

bool loopback_port_free(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return false;
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));  // TIME_WAIT is fine; a listener is not
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok = ::bind(fd, reinterpret_cast<const sockaddr*>(&a), sizeof(a)) == 0;
    ::close(fd);
    return ok;
}

ProcessResult run_process(const std::vector<std::string>& argv, const ProcessOptions& o) {
    check_argv(argv);
    HALO_CHECK(o.max_output_bytes >= 1, ErrorCode::Config, "subprocess: max_output_bytes must be >= 1");
    Fd out_r, out_w, err_r, err_w;
    make_pipe(out_r, out_w);
    make_pipe(err_r, err_w);
    SpawnSetup setup;
    posix_spawn_file_actions_addopen(&setup.fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&setup.fa, out_w.fd, 1);
    posix_spawn_file_actions_adddup2(&setup.fa, err_w.fd, 2);
    const auto t0 = std::chrono::steady_clock::now();
    Child child(spawn(argv, o, setup), false);
    ChildGuard guard{child, o.kill_grace};
    out_w.reset();  // only the child holds the write ends now
    err_w.reset();

    ProcessResult r;
    const auto deadline = t0 + o.timeout;
    std::array<char, 65536> buf{};
    bool out_open = true, err_open = true;
    while (out_open || err_open) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            r.timed_out = true;
            break;
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        std::array<pollfd, 2> fds{{{out_open ? out_r.fd : -1, POLLIN, 0}, {err_open ? err_r.fd : -1, POLLIN, 0}}};
        const int pr = ::poll(fds.data(), fds.size(), static_cast<int>(std::min<long long>(left, 1000)));
        if (pr < 0) {
            if (errno == EINTR) continue;
            throw_error(ErrorCode::Io, "subprocess: poll failed: {}", std::strerror(errno));
        }
        for (int s = 0; s < 2; ++s) {
            auto& f = fds[static_cast<std::size_t>(s)];
            if (f.fd < 0 || (f.revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
            const ssize_t n = ::read(f.fd, buf.data(), buf.size());
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                (s == 0 ? out_open : err_open) = false;
                continue;
            }
            std::string& dst = s == 0 ? r.out : r.err;
            bool& trunc = s == 0 ? r.out_truncated : r.err_truncated;
            const std::size_t room = o.max_output_bytes > dst.size() ? o.max_output_bytes - dst.size() : 0;
            const std::size_t take = std::min(room, static_cast<std::size_t>(n));
            dst.append(buf.data(), take);
            if (take < static_cast<std::size_t>(n)) trunc = true;  // rest drained and dropped
        }
    }
    Exit ex;
    bool done = false;
    // Both pipes closed does not mean the child exited (it may have closed them itself):
    // keep honouring the deadline while waiting for the exit.
    while (!r.timed_out) {
        const State s = child.state();
        if (s == State::Exited) {
            ex = child.reap();  // kills stragglers in the group before reaping (S-32)
            done = true;
            break;
        }
        if (s == State::Lost) {
            done = true;  // ex.known stays false (S-30)
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            r.timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!done) ex = child.terminate(o.kill_grace);
    guard.done = true;
    r.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ex.known) {
        r.status_unknown = true;
        r.exit_code = -1;
    } else if (WIFEXITED(ex.status)) {
        r.exit_code = WEXITSTATUS(ex.status);
    } else if (WIFSIGNALED(ex.status)) {
        r.signaled = true;
        r.term_signal = WTERMSIG(ex.status);
    }
    return r;
}

// ---- ChildProcess -------------------------------------------------------------------------

ChildProcess::ChildProcess(const std::vector<std::string>& argv, const ProcessOptions& o,
                           const std::optional<std::filesystem::path>& log_path)
    : grace_(o.kill_grace) {
    check_argv(argv);
    // The log is opened here (not by a spawn file action) so that it is always 0600, never
    // follows a symlink, and a bad path is a clear Error(Io) before anything runs (S-37).
    Fd log;
    if (log_path) {
        log.fd = ::open(log_path->c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0600);
        HALO_CHECK(log.fd >= 0, ErrorCode::Io, "subprocess: cannot open log {}: {}", log_path->string(),
                   std::strerror(errno));
        HALO_CHECK(::fchmod(log.fd, 0600) == 0, ErrorCode::Io, "subprocess: fchmod log: {}", std::strerror(errno));
    }
    SpawnSetup setup;
    posix_spawn_file_actions_addopen(&setup.fa, 0, "/dev/null", O_RDONLY, 0);
    if (log.fd >= 0) {
        posix_spawn_file_actions_adddup2(&setup.fa, log.fd, 1);
    } else {
        posix_spawn_file_actions_addopen(&setup.fa, 1, "/dev/null", O_WRONLY, 0);
    }
    posix_spawn_file_actions_adddup2(&setup.fa, 1, 2);
    pid_ = spawn(argv, o, setup);
}

ChildProcess::~ChildProcess() {
    try {
        terminate();
    } catch (const std::exception& e) {
        HALO_ERROR("subprocess", "ChildProcess teardown: {}", e.what());
    }
}

bool ChildProcess::running() {
    if (reaped_) return false;
    Child c(pid_, false);
    const State s = c.state();
    if (s == State::Running) return true;
    const Exit e = s == State::Exited ? c.reap() : Exit{};
    reaped_ = true;
    exit_code_ = e.known ? std::optional<int>(exit_code_of(e)) : std::nullopt;
    return false;
}

void ChildProcess::terminate() {
    if (reaped_ || pid_ <= 0) return;
    Child c(pid_, false);
    const Exit e = c.terminate(grace_);
    reaped_ = true;
    exit_code_ = e.known ? std::optional<int>(exit_code_of(e)) : std::nullopt;
}

}  // namespace halo::profiling
