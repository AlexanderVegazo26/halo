#include "halo/profiling/subprocess.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <thread>

#include "halo/core/error.h"
#include "halo/core/log.h"

extern char** environ;

namespace halo::profiling {

namespace {

constexpr std::array<std::string_view, 10> kRecordedPrefixes{"RADV_", "AMD_",  "HSA_",   "ROCR_",  "HIP_",
                                                             "GGML_", "VK_",   "LLAMA_", "OLLAMA_", "MESA_"};

bool recorded_prefix(std::string_view key) {
    for (const auto p : kRecordedPrefixes) {
        if (key.starts_with(p)) return true;
    }
    return false;
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

/// Blocking waitpid, EINTR-safe. Returns the raw status.
int reap(pid_t pid) {
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return 0;
    }
    return status;
}

/// Non-blocking check; true + status when the child exited.
bool try_reap(pid_t pid, int& status) {
    while (true) {
        const pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) return true;
        if (r == 0) return false;
        if (errno != EINTR) return true;  // not our child any more: treat as gone
    }
}

void kill_group(pid_t pid, int sig) {
    if (::kill(-pid, sig) != 0 && errno == ESRCH) ::kill(pid, sig);
}

/// SIGTERM the group, wait up to grace for exit, then SIGKILL. Returns the raw status.
int terminate_group(pid_t pid, std::chrono::milliseconds grace) {
    int status = 0;
    if (try_reap(pid, status)) {
        kill_group(pid, SIGKILL);  // stragglers in the group
        return status;
    }
    kill_group(pid, SIGTERM);
    const auto until = std::chrono::steady_clock::now() + grace;
    while (std::chrono::steady_clock::now() < until) {
        if (try_reap(pid, status)) {
            kill_group(pid, SIGKILL);
            return status;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill_group(pid, SIGKILL);
    return reap(pid);
}

/// Reaps a child if a scope is left by an exception.
struct ChildGuard {
    pid_t pid;
    std::chrono::milliseconds grace;
    bool done = false;
    ~ChildGuard() {
        if (!done) (void)terminate_group(pid, grace);
    }
};

}  // namespace

std::map<std::string, std::string> recorded_environment(const ProcessOptions& o) {
    std::map<std::string, std::string> out;
    if (o.inherit_env) {
        for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
            const std::string_view kv(*e);
            const auto eq = kv.find('=');
            if (eq == std::string_view::npos) continue;
            const std::string_view k = kv.substr(0, eq);
            if (recorded_prefix(k)) out.emplace(std::string(k), std::string(kv.substr(eq + 1)));
        }
    }
    for (const auto& [k, v] : o.env) out[k] = v;
    return out;
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
    const pid_t pid = spawn(argv, o, setup);
    ChildGuard guard{pid, o.kill_grace};
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
    int status = 0;
    bool reaped = false;
    // Both pipes closed does not mean the child exited (it may have closed them itself):
    // keep honouring the deadline while waiting for the exit.
    while (!r.timed_out) {
        if (try_reap(pid, status)) {
            reaped = true;
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            r.timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (reaped) {
        kill_group(pid, SIGKILL);  // any grandchild left in the group
    } else {
        status = terminate_group(pid, o.kill_grace);
    }
    guard.done = true;
    r.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (WIFEXITED(status)) {
        r.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        r.signaled = true;
        r.term_signal = WTERMSIG(status);
    }
    return r;
}

// ---- ChildProcess -------------------------------------------------------------------------

ChildProcess::ChildProcess(const std::vector<std::string>& argv, const ProcessOptions& o,
                           const std::optional<std::filesystem::path>& log_path)
    : grace_(o.kill_grace) {
    check_argv(argv);
    SpawnSetup setup;
    const std::string log = log_path ? log_path->string() : std::string("/dev/null");
    posix_spawn_file_actions_addopen(&setup.fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&setup.fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
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
    int status = 0;
    if (try_reap(pid_, status)) {
        reaped_ = true;
        kill_group(pid_, SIGKILL);
        exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
        return false;
    }
    return true;
}

void ChildProcess::terminate() {
    if (reaped_ || pid_ <= 0) return;
    const int status = terminate_group(pid_, grace_);
    reaped_ = true;
    exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

}  // namespace halo::profiling
