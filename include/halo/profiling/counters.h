#pragma once
// TRD §34 performance counters, as one plain value struct with Prometheus-text and JSON
// renderings (PRD §13: "/metrics (Prometheus format in v1)").
//
// The struct is a snapshot over a measurement window of `window_ns`. Counters are totals;
// mtp_acceptance_rate and effective_bandwidth_per_tier are *derived* from the totals (they
// are never stored independently, so they cannot disagree with them).
//
// Prometheus names (text exposition format 0.0.4): every metric is prefixed `halo_`;
// counters end in `_total`; times are exported in seconds; per-tier values carry a
// `tier="VRAM|GTT|PINNED|HOST"` label.

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/tier.h"

namespace halo::runtime { struct EngineStats; }

namespace halo::profiling {

using hardware::MemoryTier;

struct PerfCounters {
    std::uint64_t tokens_generated = 0;
    std::uint64_t kernel_calls = 0;
    std::uint64_t kernel_time_ns = 0;
    std::uint64_t gpu_time_ns = 0;
    std::uint64_t cpu_time_ns = 0;
    std::map<MemoryTier, std::uint64_t> memory_allocations_by_tier;  ///< allocation count
    std::uint64_t kv_hits = 0;
    std::uint64_t kv_misses = 0;
    std::uint64_t prefix_cache_hits = 0;
    std::uint64_t prefix_cache_misses = 0;
    std::uint64_t deltanet_state_checkpoints = 0;
    std::uint64_t state_evictions = 0;
    std::uint64_t speculative_attempts = 0;  ///< draft tokens proposed
    std::uint64_t speculative_accepts = 0;   ///< draft tokens accepted
    std::map<MemoryTier, std::uint64_t> tier_bytes_read;
    std::map<MemoryTier, std::uint64_t> tier_bytes_written;
    std::uint64_t window_ns = 0;  ///< measurement window for the derived bandwidths

    /// speculative_accepts / speculative_attempts; nullopt when no draft was proposed.
    [[nodiscard]] std::optional<double> mtp_acceptance_rate() const noexcept;
    /// (read + written) bytes of `tier` / window, in GB/s (1e9); nullopt when the window
    /// is 0 or the tier moved no bytes.
    [[nodiscard]] std::optional<double> effective_bandwidth_gbps(MemoryTier tier) const noexcept;

    /// Element-wise sum (windows add).
    PerfCounters& operator+=(const PerfCounters& o);
};

/// Fill the counters the Engine contract exposes (tokens, prefix cache, speculation).
[[nodiscard]] PerfCounters from_engine_stats(const runtime::EngineStats& s);

/// Prometheus text exposition (with # HELP / # TYPE lines).
[[nodiscard]] std::string to_prometheus(const PerfCounters& c);

void to_json(nlohmann::json& j, const PerfCounters& c);

}  // namespace halo::profiling
