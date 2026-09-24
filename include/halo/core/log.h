#pragma once
// Minimal leveled logger (TRD §46). Text by default; JSON lines in benchmark mode.
// Thread-safe; output goes to stderr.

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
