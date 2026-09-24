#pragma once
// Per-tier bandwidth microbenchmark (TRD §13.2, statistics per TRD §50).
//
// Only the CPU-issued host-memory side is implemented here. GPU-issued measurements are
// produced by the backends and reported through the same TierBandwidth record.
//
// Results measured on the development host are NOT HALO performance claims (D-001); every
// result carries the caller-supplied `label` (e.g. "dev-host") so it cannot be mistaken
// for a target measurement.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/tier.h"

namespace halo::hardware {

/// Summary statistics over repeated measurements.
struct Stats {
    std::size_t n = 0;
    double min = 0;
    double max = 0;
    double mean = 0;
    double median = 0;  ///< average of the two middle values for even n
    double stddev = 0;  ///< sample standard deviation (n-1); 0 when n < 2
    double p95 = 0;     ///< nearest-rank: sorted[ceil(0.95·n) − 1]
};

/// Throws Error(Config) on an empty sample or a non-finite value.
[[nodiscard]] Stats compute_stats(std::span<const double> samples);

struct TierBandwidth {
    MemoryTier tier = MemoryTier::Host;
    Processor processor = Processor::Cpu;
    std::string label;             ///< where it was measured, e.g. "dev-host"
    std::size_t buffer_bytes = 0;  ///< per buffer
    unsigned threads = 0;
    unsigned warmup = 0;
    unsigned iterations = 0;
    Stats read_gbps;   ///< bytes read / s / 1e9
    Stats write_gbps;  ///< bytes written / s / 1e9
    Stats copy_gbps;   ///< (bytes read + bytes written) / s / 1e9  (STREAM convention)
};

struct BandwidthOptions {
    std::size_t buffer_bytes = 256ULL << 20;
    unsigned threads = 0;     ///< 0 = std::thread::hardware_concurrency()
    unsigned warmup = 5;      ///< TRD §50 minimum
    unsigned iterations = 20; ///< TRD §50 minimum
    std::string label;        ///< required, non-empty (e.g. "dev-host")
    /// Allow fewer than 5 warm-up / 20 measured runs (unit tests only; results are then
    /// not TRD §50-conformant).
    bool allow_nonconformant = false;
};

/// CPU-issued read / write / copy bandwidth of pageable host memory using `threads`
/// persistent worker threads. Throws Error(Config) on invalid options and Error(Memory) if
/// the buffers cannot be allocated.
[[nodiscard]] TierBandwidth measure_host_bandwidth(const BandwidthOptions& options);

void to_json(nlohmann::json& j, const Stats& v);
void to_json(nlohmann::json& j, const TierBandwidth& v);

}  // namespace halo::hardware
