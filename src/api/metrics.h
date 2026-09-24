#pragma once
// Prometheus text exposition for GET /metrics (PRD §13, TRD §47). API-side counters and
// histograms are kept here; engine-side values (active/queued sequences, tokens generated,
// prefix-cache hits, speculative accepts) come from Engine::stats() at scrape time.

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

#include "halo/runtime/engine.h"

namespace halo::api {

class Histogram {
public:
    static constexpr std::array<double, 12> kBounds{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 120};
    void observe(double seconds);
    void render(std::string& out, const std::string& name, const std::string& help) const;

private:
    mutable std::mutex mu_;
    std::array<std::uint64_t, kBounds.size()> buckets_{};
    std::uint64_t count_ = 0;
    double sum_ = 0.0;
};

class Metrics {
public:
    void count_request(const std::string& route, int status);
    void count_rejection(const std::string& reason);

    std::atomic<std::int64_t> active{0};
    std::atomic<std::uint64_t> prompt_tokens{0};
    std::atomic<std::uint64_t> completion_tokens{0};
    std::atomic<std::uint64_t> reasoning_tokens{0};
    std::atomic<std::uint64_t> cancelled{0};
    std::atomic<std::uint64_t> budget_closes{0};
    std::atomic<double> last_decode_tps{0.0};
    Histogram ttft;
    Histogram duration;

    [[nodiscard]] std::string render(const runtime::EngineStats& engine, std::size_t queued, std::size_t admitted) const;

private:
    mutable std::mutex mu_;
    std::map<std::pair<std::string, int>, std::uint64_t> requests_;
    std::map<std::string, std::uint64_t> rejections_;
};

}  // namespace halo::api
