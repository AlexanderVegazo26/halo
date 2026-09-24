#include "halo/profiling/timer.h"

#include <algorithm>
#include <exception>

#include <nlohmann/json.hpp>

#include "halo/core/log.h"

namespace halo::profiling {

std::int64_t SteadyClock::now_ns() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const SteadyClock& SteadyClock::instance() noexcept {
    static const SteadyClock clock;
    return clock;
}

void OpTimingRecorder::record(std::string_view op, std::int64_t ns, std::uint64_t bytes) {
    const std::scoped_lock lock(mu_);
    auto it = ops_.find(op);
    if (it == ops_.end()) it = ops_.emplace(std::string(op), OpTiming{}).first;
    OpTiming& t = it->second;
    if (t.calls == 0) {
        t.min_ns = ns;
        t.max_ns = ns;
    } else {
        t.min_ns = std::min(t.min_ns, ns);
        t.max_ns = std::max(t.max_ns, ns);
    }
    ++t.calls;
    t.total_ns += ns;
    t.bytes += bytes;
    if (t.samples_ns.size() < max_samples_) t.samples_ns.push_back(static_cast<double>(ns));
}

std::map<std::string, OpTiming, std::less<>> OpTimingRecorder::snapshot() const {
    const std::scoped_lock lock(mu_);
    return ops_;
}

void OpTimingRecorder::clear() {
    const std::scoped_lock lock(mu_);
    ops_.clear();
}

ScopedOpTimer::~ScopedOpTimer() {
    try {
        recorder_->record(op_, clock_->now_ns() - start_, bytes_);
    } catch (const std::exception& e) {
        // A destructor must not throw; the lost sample is reported, not hidden.
        HALO_ERROR("profiling", "ScopedOpTimer: dropped sample for {}: {}", op_, e.what());
    }
}

void to_json(nlohmann::json& j, const OpTiming& v) {
    j = {{"calls", v.calls},
         {"total_ns", v.total_ns},
         {"min_ns", v.min_ns},
         {"max_ns", v.max_ns},
         {"mean_ns", v.calls == 0 ? 0.0 : static_cast<double>(v.total_ns) / static_cast<double>(v.calls)},
         {"bytes", v.bytes},
         {"samples_kept", v.samples_ns.size()}};
}

}  // namespace halo::profiling
