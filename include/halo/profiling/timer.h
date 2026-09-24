#pragma once
// High-resolution timers and a scoped operator-timing recorder (TRD §27, §34).
//
// All timing uses a `Clock` interface so that suites and tests can inject a controllable
// clock (tests never sleep to create timings). `SteadyClock` wraps std::chrono::steady_clock
// (monotonic, nanosecond ticks on Linux/libstdc++).
//
// Numbers measured on the development host are harness smoke tests only (D-001).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace halo::profiling {

/// Monotonic time source in nanoseconds. Implementations must be thread-safe.
class Clock {
public:
    virtual ~Clock() = default;
    /// Nanoseconds since an arbitrary, fixed epoch. Never decreases.
    [[nodiscard]] virtual std::int64_t now_ns() const = 0;
};

/// std::chrono::steady_clock.
class SteadyClock final : public Clock {
public:
    [[nodiscard]] std::int64_t now_ns() const override;
    /// Process-wide instance (stateless).
    [[nodiscard]] static const SteadyClock& instance() noexcept;
};

/// Clock that only moves when told to (tests, deterministic replays). Thread-safe.
class ManualClock final : public Clock {
public:
    explicit ManualClock(std::int64_t start_ns = 0) noexcept : ns_(start_ns) {}
    [[nodiscard]] std::int64_t now_ns() const override { return ns_.load(std::memory_order_acquire); }
    /// Advance by `ns` (>= 0; negative values are ignored so time never decreases).
    void advance(std::int64_t ns) noexcept {
        if (ns > 0) ns_.fetch_add(ns, std::memory_order_acq_rel);
    }

private:
    std::atomic<std::int64_t> ns_;
};

/// Measures elapsed time from construction (or the last restart()).
class Stopwatch {
public:
    explicit Stopwatch(const Clock& clock = SteadyClock::instance()) noexcept
        : clock_(&clock), start_(clock.now_ns()) {}
    void restart() noexcept { start_ = clock_->now_ns(); }
    [[nodiscard]] std::int64_t elapsed_ns() const noexcept { return clock_->now_ns() - start_; }
    [[nodiscard]] double elapsed_ms() const noexcept { return static_cast<double>(elapsed_ns()) * 1e-6; }
    [[nodiscard]] double elapsed_s() const noexcept { return static_cast<double>(elapsed_ns()) * 1e-9; }

private:
    const Clock* clock_;
    std::int64_t start_;
};

/// Per-operator aggregate kept by OpTimingRecorder.
struct OpTiming {
    std::uint64_t calls = 0;
    std::int64_t total_ns = 0;
    std::int64_t min_ns = 0;
    std::int64_t max_ns = 0;
    std::uint64_t bytes = 0;         ///< bytes moved as reported by the caller (0 = unknown)
    std::vector<double> samples_ns;  ///< per-call durations, capped at `max_samples`
};

/// Thread-safe accumulator of per-operator timings. One mutex guards the map; the
/// critical section is a map lookup plus a few adds, so it is intended for operator-level
/// (not per-element) granularity.
class OpTimingRecorder {
public:
    explicit OpTimingRecorder(std::size_t max_samples_per_op = 4096) : max_samples_(max_samples_per_op) {}

    /// Add one call of `op` that took `ns` nanoseconds and moved `bytes` (0 = unknown).
    void record(std::string_view op, std::int64_t ns, std::uint64_t bytes = 0);
    /// Snapshot, sorted by operator name.
    [[nodiscard]] std::map<std::string, OpTiming, std::less<>> snapshot() const;
    void clear();

private:
    std::size_t max_samples_;
    mutable std::mutex mu_;
    std::map<std::string, OpTiming, std::less<>> ops_;
};

/// RAII: records the lifetime of the scope into `recorder` under `op`.
///   { ScopedOpTimer t(rec, "MATMUL", bytes); matmul(...); }
class ScopedOpTimer {
public:
    ScopedOpTimer(OpTimingRecorder& recorder, std::string_view op, std::uint64_t bytes = 0,
                  const Clock& clock = SteadyClock::instance())
        : recorder_(&recorder), op_(op), bytes_(bytes), clock_(&clock), start_(clock.now_ns()) {}
    ~ScopedOpTimer();
    ScopedOpTimer(const ScopedOpTimer&) = delete;
    ScopedOpTimer& operator=(const ScopedOpTimer&) = delete;
    ScopedOpTimer(ScopedOpTimer&&) = delete;
    ScopedOpTimer& operator=(ScopedOpTimer&&) = delete;

private:
    OpTimingRecorder* recorder_;
    std::string op_;
    std::uint64_t bytes_;
    const Clock* clock_;
    std::int64_t start_;
};

void to_json(nlohmann::json& j, const OpTiming& v);

}  // namespace halo::profiling
