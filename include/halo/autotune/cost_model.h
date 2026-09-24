#pragma once
// TRD §55 cost model: T = max(T_compute, T_memory) + T_sync + T_launch, with
// T_memory = BytesMoved / EffectiveBandwidth(tier) from *measured* tier bandwidth
// (halo::hardware::TierBandwidth, TRD §13.2) and T_compute from a per-thread throughput
// that is calibrated against HALO's own kernels (calibrate_gflops_per_thread); there is no
// built-in GFLOPS constant.
//
// The memory-bound verdict is the model's classification only. TRD §55 asks for it to be
// verified per kernel with occupancy/stall counters; the CPU reference path has no such
// counters, so `memory_bound` is an estimate and is labelled as one wherever it is stored.

#include <cstdint>
#include <map>
#include <span>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/bandwidth.h"
#include "halo/hardware/tier.h"

namespace halo::autotune {

/// What one candidate does, as reported by its TunableOp.
struct CostInputs {
    double flops = 0;
    std::uint64_t bytes = 0;
    hardware::MemoryTier tier = hardware::MemoryTier::Host;
    unsigned parallelism = 1;  ///< workers that actually get work (>= 1)
    unsigned launches = 1;     ///< kernel launches / parallel regions
    unsigned syncs = 0;        ///< barriers / fences
};

struct CostEstimate {
    double t_compute_ns = 0;
    double t_memory_ns = 0;
    double t_sync_ns = 0;
    double t_launch_ns = 0;
    double total_ns = 0;
    bool memory_bound = false;  ///< model estimate (t_memory >= t_compute), not counter-verified
};

struct CostModel {
    std::map<hardware::MemoryTier, double> bandwidth_gbps;  ///< per tier, GB/s (1e9)
    unsigned bandwidth_threads = 1;  ///< threads the bandwidth was measured with
    double gflops_per_thread = 0;    ///< calibrated; must be > 0
    double launch_ns = 0;
    double sync_ns = 0;

    /// Bandwidth per tier = the MEDIAN of the measured read bandwidth (hardware::Stats
    /// median; its p95 is nearest-rank and is not used). Throws Error(Config) on an empty
    /// list, a non-positive median, or inconsistent thread counts.
    [[nodiscard]] static CostModel from_measurements(std::span<const hardware::TierBandwidth> measured,
                                                     double gflops_per_thread, double launch_ns = 0,
                                                     double sync_ns = 0);

    /// Throws Error(Config) when the tier has no bandwidth or gflops_per_thread <= 0.
    /// Memory bandwidth scales as min(1, parallelism / bandwidth_threads); compute scales
    /// linearly with parallelism.
    [[nodiscard]] CostEstimate estimate(const CostInputs& in) const;
};

/// Per-thread GFLOP/s from one measured run of a HALO kernel: flops / (ns · threads).
/// Throws Error(Config) on non-positive inputs.
[[nodiscard]] double calibrate_gflops_per_thread(double flops, double measured_ns, unsigned threads);

void to_json(nlohmann::json& j, const CostEstimate& e);

}  // namespace halo::autotune
