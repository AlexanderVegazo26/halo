#include "halo/profiling/counters.h"

#include <array>
#include <format>

#include <nlohmann/json.hpp>

#include "halo/runtime/engine.h"

namespace halo::profiling {

namespace {

constexpr std::array<MemoryTier, 4> kTiers{MemoryTier::Vram, MemoryTier::Gtt, MemoryTier::Pinned,
                                           MemoryTier::Host};

std::uint64_t get(const std::map<MemoryTier, std::uint64_t>& m, MemoryTier t) {
    const auto it = m.find(t);
    return it == m.end() ? 0 : it->second;
}

void add_map(std::map<MemoryTier, std::uint64_t>& a, const std::map<MemoryTier, std::uint64_t>& b) {
    for (const auto& [k, v] : b) a[k] += v;
}

void counter(std::string& out, std::string_view name, std::string_view help, std::uint64_t v) {
    out += std::format("# HELP halo_{} {}\n# TYPE halo_{} counter\nhalo_{} {}\n", name, help, name, name, v);
}

void seconds_counter(std::string& out, std::string_view name, std::string_view help, std::uint64_t ns) {
    out += std::format("# HELP halo_{} {}\n# TYPE halo_{} counter\nhalo_{} {:.9f}\n", name, help, name, name,
                       static_cast<double>(ns) * 1e-9);
}

void tier_counter(std::string& out, std::string_view name, std::string_view help,
                  const std::map<MemoryTier, std::uint64_t>& m) {
    out += std::format("# HELP halo_{} {}\n# TYPE halo_{} counter\n", name, help, name);
    for (const MemoryTier t : kTiers) {
        out += std::format("halo_{}{{tier=\"{}\"}} {}\n", name, hardware::to_string(t), get(m, t));
    }
}

nlohmann::json tier_json(const std::map<MemoryTier, std::uint64_t>& m) {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [k, v] : m) j[std::string(hardware::to_string(k))] = v;
    return j;
}

}  // namespace

std::optional<double> PerfCounters::mtp_acceptance_rate() const noexcept {
    if (speculative_attempts == 0) return std::nullopt;
    return static_cast<double>(speculative_accepts) / static_cast<double>(speculative_attempts);
}

std::optional<double> PerfCounters::effective_bandwidth_gbps(MemoryTier tier) const noexcept {
    const std::uint64_t bytes = get(tier_bytes_read, tier) + get(tier_bytes_written, tier);
    if (window_ns == 0 || bytes == 0) return std::nullopt;
    return static_cast<double>(bytes) / static_cast<double>(window_ns);  // bytes/ns == GB/s
}

PerfCounters& PerfCounters::operator+=(const PerfCounters& o) {
    tokens_generated += o.tokens_generated;
    kernel_calls += o.kernel_calls;
    kernel_time_ns += o.kernel_time_ns;
    gpu_time_ns += o.gpu_time_ns;
    cpu_time_ns += o.cpu_time_ns;
    add_map(memory_allocations_by_tier, o.memory_allocations_by_tier);
    kv_hits += o.kv_hits;
    kv_misses += o.kv_misses;
    prefix_cache_hits += o.prefix_cache_hits;
    prefix_cache_misses += o.prefix_cache_misses;
    deltanet_state_checkpoints += o.deltanet_state_checkpoints;
    state_evictions += o.state_evictions;
    speculative_attempts += o.speculative_attempts;
    speculative_accepts += o.speculative_accepts;
    add_map(tier_bytes_read, o.tier_bytes_read);
    add_map(tier_bytes_written, o.tier_bytes_written);
    window_ns += o.window_ns;
    return *this;
}

PerfCounters from_engine_stats(const runtime::EngineStats& s) {
    PerfCounters c;
    c.tokens_generated = s.tokens_generated;
    c.prefix_cache_hits = s.prefix_cache_hits;
    c.prefix_cache_misses = s.prefix_cache_misses;
    c.speculative_attempts = s.speculative_attempts;
    c.speculative_accepts = s.speculative_accepts;
    return c;
}

std::string to_prometheus(const PerfCounters& c) {
    std::string out;
    counter(out, "tokens_generated_total", "Tokens generated.", c.tokens_generated);
    counter(out, "kernel_calls_total", "Kernel invocations.", c.kernel_calls);
    seconds_counter(out, "kernel_time_seconds_total", "Time spent in kernels.", c.kernel_time_ns);
    seconds_counter(out, "gpu_time_seconds_total", "GPU busy time.", c.gpu_time_ns);
    seconds_counter(out, "cpu_time_seconds_total", "CPU time.", c.cpu_time_ns);
    tier_counter(out, "memory_allocations_total", "Allocations by memory tier.", c.memory_allocations_by_tier);
    counter(out, "kv_hits_total", "KV cache block hits.", c.kv_hits);
    counter(out, "kv_misses_total", "KV cache block misses.", c.kv_misses);
    counter(out, "prefix_cache_hits_total", "Prefix cache hits.", c.prefix_cache_hits);
    counter(out, "prefix_cache_misses_total", "Prefix cache misses.", c.prefix_cache_misses);
    counter(out, "deltanet_state_checkpoints_total", "DeltaNet state checkpoints taken.", c.deltanet_state_checkpoints);
    counter(out, "state_evictions_total", "State evictions.", c.state_evictions);
    counter(out, "speculative_attempts_total", "Draft tokens proposed.", c.speculative_attempts);
    counter(out, "speculative_accepts_total", "Draft tokens accepted.", c.speculative_accepts);
    out += "# HELP halo_mtp_acceptance_rate Accepted / proposed draft tokens (NaN when none proposed).\n"
           "# TYPE halo_mtp_acceptance_rate gauge\n";
    const auto rate = c.mtp_acceptance_rate();
    out += rate ? std::format("halo_mtp_acceptance_rate {:.6f}\n", *rate) : "halo_mtp_acceptance_rate NaN\n";
    tier_counter(out, "tier_bytes_read_total", "Bytes read per memory tier.", c.tier_bytes_read);
    tier_counter(out, "tier_bytes_written_total", "Bytes written per memory tier.", c.tier_bytes_written);
    out += "# HELP halo_effective_bandwidth_gbps Bytes moved per tier / window (GB/s, 1e9).\n"
           "# TYPE halo_effective_bandwidth_gbps gauge\n";
    for (const MemoryTier t : kTiers) {
        const auto bw = c.effective_bandwidth_gbps(t);
        out += bw ? std::format("halo_effective_bandwidth_gbps{{tier=\"{}\"}} {:.6f}\n", hardware::to_string(t), *bw)
                  : std::format("halo_effective_bandwidth_gbps{{tier=\"{}\"}} NaN\n", hardware::to_string(t));
    }
    return out;
}

void to_json(nlohmann::json& j, const PerfCounters& c) {
    nlohmann::json bw = nlohmann::json::object();
    for (const MemoryTier t : kTiers) {
        if (const auto v = c.effective_bandwidth_gbps(t)) bw[std::string(hardware::to_string(t))] = *v;
    }
    const auto rate = c.mtp_acceptance_rate();
    j = {{"tokens_generated", c.tokens_generated},
         {"kernel_calls", c.kernel_calls},
         {"kernel_time_ns", c.kernel_time_ns},
         {"GPU_time_ns", c.gpu_time_ns},
         {"CPU_time_ns", c.cpu_time_ns},
         {"memory_allocations_by_tier", tier_json(c.memory_allocations_by_tier)},
         {"KV_hits", c.kv_hits},
         {"KV_misses", c.kv_misses},
         {"prefix_cache_hits", c.prefix_cache_hits},
         {"prefix_cache_misses", c.prefix_cache_misses},
         {"deltanet_state_checkpoints", c.deltanet_state_checkpoints},
         {"state_evictions", c.state_evictions},
         {"speculative_attempts", c.speculative_attempts},
         {"speculative_accepts", c.speculative_accepts},
         {"mtp_acceptance_rate", rate ? nlohmann::json(*rate) : nlohmann::json(nullptr)},
         {"tier_bytes_read", tier_json(c.tier_bytes_read)},
         {"tier_bytes_written", tier_json(c.tier_bytes_written)},
         {"effective_bandwidth_per_tier_gbps", bw},
         {"window_ns", c.window_ns}};
}

}  // namespace halo::profiling
