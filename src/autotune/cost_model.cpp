#include "halo/autotune/cost_model.h"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"

namespace halo::autotune {

CostModel CostModel::from_measurements(std::span<const hardware::TierBandwidth> measured, double gflops_per_thread,
                                       double launch_ns, double sync_ns) {
    HALO_CHECK(!measured.empty(), ErrorCode::Config, "cost model: no tier bandwidth measurements");
    CostModel m;
    m.gflops_per_thread = gflops_per_thread;
    m.launch_ns = launch_ns;
    m.sync_ns = sync_ns;
    m.bandwidth_threads = measured.front().threads;
    for (const auto& t : measured) {
        HALO_CHECK(t.threads == m.bandwidth_threads && t.threads >= 1, ErrorCode::Config,
                   "cost model: tier bandwidths measured with different thread counts ({} vs {})", t.threads,
                   m.bandwidth_threads);
        HALO_CHECK(std::isfinite(t.read_gbps.median) && t.read_gbps.median > 0, ErrorCode::Config,
                   "cost model: tier {} median read bandwidth {} is not positive", hardware::to_string(t.tier),
                   t.read_gbps.median);
        m.bandwidth_gbps[t.tier] = t.read_gbps.median;
    }
    return m;
}

CostEstimate CostModel::estimate(const CostInputs& in) const {
    HALO_CHECK(gflops_per_thread > 0 && std::isfinite(gflops_per_thread), ErrorCode::Config,
               "cost model: gflops_per_thread {} is not calibrated", gflops_per_thread);
    const auto bw = bandwidth_gbps.find(in.tier);
    HALO_CHECK(bw != bandwidth_gbps.end() && bw->second > 0, ErrorCode::Config,
               "cost model: no measured bandwidth for tier {}", hardware::to_string(in.tier));
    HALO_CHECK(in.flops >= 0 && std::isfinite(in.flops), ErrorCode::Config, "cost model: invalid flops");
    const double par = std::max(1U, in.parallelism);
    const double bw_scale = std::min(1.0, par / std::max(1U, bandwidth_threads));
    CostEstimate e;
    e.t_compute_ns = in.flops / (gflops_per_thread * par);  // GFLOP/s == flop/ns
    e.t_memory_ns = static_cast<double>(in.bytes) / (bw->second * bw_scale);  // GB/s == B/ns
    e.t_sync_ns = sync_ns * in.syncs;
    e.t_launch_ns = launch_ns * in.launches;
    e.total_ns = std::max(e.t_compute_ns, e.t_memory_ns) + e.t_sync_ns + e.t_launch_ns;
    e.memory_bound = e.t_memory_ns >= e.t_compute_ns;
    return e;
}

double calibrate_gflops_per_thread(double flops, double measured_ns, unsigned threads) {
    HALO_CHECK(flops > 0 && measured_ns > 0 && threads > 0 && std::isfinite(flops) && std::isfinite(measured_ns),
               ErrorCode::Config, "calibrate_gflops_per_thread: invalid inputs");
    return flops / (measured_ns * threads);
}

void to_json(nlohmann::json& j, const CostEstimate& e) {
    j = {{"t_compute_ns", e.t_compute_ns}, {"t_memory_ns", e.t_memory_ns}, {"t_sync_ns", e.t_sync_ns},
         {"t_launch_ns", e.t_launch_ns},   {"total_ns", e.total_ns},       {"memory_bound_estimate", e.memory_bound}};
}

}  // namespace halo::autotune
