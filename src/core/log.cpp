#include "halo/core/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>

#include <nlohmann/json.hpp>

namespace halo::log {
namespace {

std::atomic<Level> g_level{Level::Info};
std::atomic<bool> g_json{false};
std::mutex g_mutex;

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

}  // namespace

void set_level(Level l) noexcept { g_level.store(l, std::memory_order_relaxed); }
Level level() noexcept { return g_level.load(std::memory_order_relaxed); }
void set_json(bool json) noexcept { g_json.store(json, std::memory_order_relaxed); }

bool parse_level(std::string_view name, Level& out) noexcept {
    for (int i = 0; i <= static_cast<int>(Level::Off); ++i) {
        auto l = static_cast<Level>(i);
        std::string_view n = level_name(l);
        if (n.size() != name.size()) continue;
        bool eq = true;
        for (std::size_t j = 0; j < n.size(); ++j) {
            char c = name[j];
            if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
            if (c != n[j]) { eq = false; break; }
        }
        if (eq) { out = l; return true; }
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
    std::fprintf(stderr, "%s\n", line.c_str());
}

}  // namespace halo::log
