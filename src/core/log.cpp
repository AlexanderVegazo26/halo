#include "halo/core/log.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <nlohmann/json.hpp>

namespace halo::log {
namespace {

std::atomic<Level> g_level{Level::Info};
std::atomic<bool> g_json{false};
std::mutex g_mutex;
std::once_flag g_env_once;
std::FILE* g_sink = nullptr;  // set once by apply_env() (nullptr = stderr); read under g_mutex

constexpr std::string_view level_name(Level l) noexcept {
    switch (l) {
        case Level::Trace: return "TRACE";
        case Level::Debug: return "DEBUG";
        case Level::Info: return "INFO";
        case Level::Warn: return "WARN";
        case Level::Error: return "ERROR";
        case Level::Fatal: return "FATAL";
        case Level::Off: return "OFF";
    }
    return "?";
}

// Empty counts as unset (the same rule as the CLI's HALO_* settings, review S-18).
const char* env_value(const char* name) noexcept {
    const char* v = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read once, under call_once
    return (v != nullptr && *v != '\0') ? v : nullptr;
}

bool iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        char x = a[i];
        char y = b[i];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y - 'A' + 'a');
        if (x != y) return false;
    }
    return true;
}

// Opens HALO_LOG_FILE for appending (review S-33/S-37/S-42/S-43):
// - never follows a symlink (O_NOFOLLOW), never blocks on a FIFO (O_NONBLOCK until checked),
//   never becomes a controlling terminal (O_NOCTTY), never leaks into children (O_CLOEXEC);
// - an existing path is lstat()ed first, so a device node or FIFO is refused without being opened;
// - accepts only a regular file owned by the effective user with exactly one link, so a mistyped
//   path under root cannot append to (or chmod) a system file or a hard link to one;
// - creates a new file 0600, and tightens an existing file of ours only if it is wider than 0600.
std::FILE* open_log_file(const char* path) noexcept {
    const auto refuse = [path](const char* why) -> std::FILE* {
        std::fprintf(stderr, "[WARN] log: HALO_LOG_FILE '%s' %s; logging to stderr\n", path, why);
        return nullptr;
    };
    constexpr int kFlags = O_WRONLY | O_APPEND | O_CLOEXEC | O_NOFOLLOW | O_NOCTTY | O_NONBLOCK;
    bool created = true;
    int fd = ::open(path, kFlags | O_CREAT | O_EXCL, 0600);
    if (fd < 0 && errno == EEXIST) {
        created = false;
        struct stat pre{};
        // A symlink is left to open() below, which refuses it with ELOOP (O_NOFOLLOW).
        if (::lstat(path, &pre) == 0 && !S_ISREG(pre.st_mode) && !S_ISLNK(pre.st_mode)) {
            return refuse("is not a regular file");
        }
        fd = ::open(path, kFlags);
    }
    if (fd < 0) {
        std::fprintf(stderr, "[WARN] log: cannot open HALO_LOG_FILE '%s': %s; logging to stderr\n", path,
                     std::strerror(errno));
        return nullptr;
    }
    struct stat st{};
    const char* why = nullptr;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        why = "is not a regular file";
    } else if (st.st_uid != ::geteuid()) {
        why = "is not owned by this user";
    } else if (st.st_nlink != 1) {
        why = "has more than one hard link";
    }
    if (why != nullptr) {
        ::close(fd);
        return refuse(why);
    }
    const int flags = ::fcntl(fd, F_GETFL);
    if (flags >= 0) (void)::fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
    if (!created && (st.st_mode & 077) != 0) (void)::fchmod(fd, 0600);  // ours (checked above)
    std::FILE* f = ::fdopen(fd, "a");
    if (f == nullptr) {
        std::fprintf(stderr, "[WARN] log: cannot use HALO_LOG_FILE '%s': %s; logging to stderr\n", path,
                     std::strerror(errno));
        ::close(fd);
    }
    return f;
}

// Runs once. Must not call level() or write(): those enter the same once_flag.
void apply_env() noexcept {
    if (const char* v = env_value("HALO_LOG_LEVEL")) {
        Level l = Level::Info;
        if (parse_level(v, l)) {
            g_level.store(l, std::memory_order_relaxed);
        } else {
            std::fprintf(stderr, "[WARN] log: HALO_LOG_LEVEL='%s' is not a level (trace|debug|info|warn|error|fatal|off); using info\n", v);
        }
    }
    if (const char* v = env_value("HALO_LOG_FORMAT")) {
        if (iequals(v, "json")) {
            g_json.store(true, std::memory_order_relaxed);
        } else if (!iequals(v, "text")) {
            std::fprintf(stderr, "[WARN] log: HALO_LOG_FORMAT='%s' is not text or json; using text\n", v);
        }
    }
    if (const char* v = env_value("HALO_LOG_FILE")) g_sink = open_log_file(v);
}

void ensure_env() noexcept {
    try {
        std::call_once(g_env_once, apply_env);
    } catch (...) {  // std::system_error from call_once: keep the defaults
    }
}

}  // namespace

// set_* read the environment first, so an explicit call made before the first log line is
// not overwritten by it later.
void set_level(Level l) noexcept {
    ensure_env();
    g_level.store(l, std::memory_order_relaxed);
}
Level level() noexcept {
    ensure_env();
    return g_level.load(std::memory_order_relaxed);
}
void set_json(bool json) noexcept {
    ensure_env();
    g_json.store(json, std::memory_order_relaxed);
}

bool parse_level(std::string_view name, Level& out) noexcept {
    for (int i = 0; i <= static_cast<int>(Level::Off); ++i) {
        auto l = static_cast<Level>(i);
        if (iequals(name, level_name(l))) {
            out = l;
            return true;
        }
    }
    return false;
}

void write(Level lvl, std::string_view component, std::string_view message) {
    if (lvl < level()) return;
    const auto now = std::chrono::system_clock::now();
    std::string line;
    if (g_json.load(std::memory_order_relaxed)) {
        nlohmann::json j = {
            {"ts_ms", std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()},
            {"level", level_name(lvl)},
            {"component", component},
            {"msg", message},
        };
        line = j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    } else {
        line = std::format("[{}] {}: {}", level_name(lvl), component, message);
    }
    std::lock_guard lock(g_mutex);
    std::FILE* out = g_sink != nullptr ? g_sink : stderr;
    std::fprintf(out, "%s\n", line.c_str());
    if (out != stderr) std::fflush(out);
}

}  // namespace halo::log
