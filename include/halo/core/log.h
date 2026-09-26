#pragma once
// Minimal leveled logger (TRD §46). Text by default; JSON lines in benchmark mode.
// Thread-safe; output goes to stderr unless HALO_LOG_FILE names a file.
//
// Environment (read once, at the first log call or level() query; an explicit set_level()
// or set_json() call always wins over it, whenever it happens):
//   HALO_LOG_LEVEL   trace | debug | info | warn | error | fatal | off (case-insensitive;
//                    default info)
//   HALO_LOG_FORMAT  text | json (default text). json writes one object per line:
//                    {"ts_ms": <Unix epoch milliseconds>, "level", "component", "msg"}
//   HALO_LOG_FILE    append log lines to this file instead of stderr. A new file is created
//                    with mode 0600; an existing one must be a regular file owned by the
//                    effective user with exactly one hard link (a symlink, device, FIFO,
//                    foreign-owned or hard-linked file is refused, without opening a device).
//                    Opened O_APPEND | O_CLOEXEC | O_NOFOLLOW | O_NOCTTY; an existing file of
//                    ours wider than 0600 is tightened to 0600. Every line is flushed. If it
//                    cannot be used, one warning goes to stderr and logging stays on stderr.
// An empty value counts as unset. An unknown value prints one warning on stderr and keeps
// the default; it never aborts the process.

#include <format>
#include <string>
#include <string_view>

namespace halo::log {

enum class Level { Trace = 0, Debug, Info, Warn, Error, Fatal, Off };

void set_level(Level level) noexcept;
[[nodiscard]] Level level() noexcept;
void set_json(bool json) noexcept;
[[nodiscard]] bool parse_level(std::string_view name, Level& out) noexcept;

void write(Level level, std::string_view component, std::string_view message);

template <typename... Args>
void logf(Level lvl, std::string_view component, std::format_string<Args...> fmt, Args&&... args) {
    if (lvl < level()) return;
    write(lvl, component, std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace halo::log

#define HALO_LOG(lvl, component, ...) ::halo::log::logf(::halo::log::Level::lvl, component, __VA_ARGS__)
#define HALO_TRACE(c, ...) HALO_LOG(Trace, c, __VA_ARGS__)
#define HALO_DEBUG(c, ...) HALO_LOG(Debug, c, __VA_ARGS__)
#define HALO_INFO(c, ...) HALO_LOG(Info, c, __VA_ARGS__)
#define HALO_WARN(c, ...) HALO_LOG(Warn, c, __VA_ARGS__)
#define HALO_ERROR(c, ...) HALO_LOG(Error, c, __VA_ARGS__)
