#pragma once
// The TRD §28 benchmark record (one per run), its roll-up (TRD §50), and the D-011 decode
// byte model used for the D-014 efficiency fields.
//
// JSON contract (schema id "halo.bench.record/1"):
//   * Every TRD §28 field is present under its exact TRD name: model, model_hash, pack,
//     backend, driver, gpu, quantization, lm_head, context, batch, concurrency, ttft_ms,
//     ttft_cached_ms, prompt_tps, decode_tps, decode_effective_tps, mtp_acceptance,
//     prefix_cache_hit_rate, memory_by_tier_gb, load_time_s, temperature_c, clocks_mhz,
//     power_mode.
//   * A metric that was not measured by this run is JSON null, never 0 (a 0 would be
//     indistinguishable from a measured zero).
//   * Additions: schema, suite, mode, engine, pack_hash, mtp_hash, repetition, phase
//     ("cold" | "steady"), host_label (D-001: e.g. "dev-host"; only a record captured on
//     the target unit can back a HALO performance claim), measured_bandwidth_gbps,
//     predicted_bytes_per_token, efficiency, hardware_before, hardware_after, invocation
//     (binary, version, commit, argv, environment), extra (suite-specific numbers).

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/profiling/hw_state.h"
#include "halo/profiling/stats.h"

namespace halo::profiling {

inline constexpr std::string_view kRecordSchema = "halo.bench.record/1";
inline constexpr std::string_view kSummarySchema = "halo.bench.summary/1";

// ---- D-011 decode byte model ------------------------------------------------------------

/// B_step = W_trunk + (n+1)·W_mtp + n·W_head + S·[(1+K)·state_copy + ctx·kv_per_token]
/// tok/s ≤ S·τ·η·BW / B_step   (DECISIONS.md D-011)
///
/// Units: GB = 1e9 bytes; kv_bytes_per_token defaults to 64 KiB = 65536 B (FP16 KV of the
/// 16 attention layers); state_copy_gb defaults to 0.157 GB (one GDN recurrent + conv state
/// copy, D-003). W_trunk INCLUDES the LM head (n·W_head counts only the extra head reads of
/// draft tokens). Set w_mtp_gb = 0 when MTP is disabled. K defaults to n+1 (D-012 rollback
/// slots). With w_trunk 16.48, S=1, n=0 (K=1), ctx=64 and BW 256 GB/s the ceiling is
/// ~15.24 tok/s and a measured 12.0 tok/s gives η ≈ 0.787 — D-011's worked numbers
/// (ctx ≈ 0 there: the llama-cli run had a short prompt). At ctx = 4096 the ceiling is ~15.0.
struct DecodeByteModel {
    double w_trunk_gb = 0;
    double w_mtp_gb = 0;
    double w_head_gb = 0;
    std::uint32_t sequences = 1;           ///< S
    std::uint32_t draft_tokens = 0;        ///< n
    std::optional<std::uint32_t> state_copies;  ///< K; nullopt = n + 1
    std::uint64_t context_tokens = 0;      ///< ctx (KV rows read per sequence per step)
    double state_copy_gb = 0.157;
    std::uint64_t kv_bytes_per_token = 65536;

    /// B_step in bytes. Throws Error(Config) on S == 0 or negative/non-finite weights.
    [[nodiscard]] double bytes_per_step() const;
    /// B_step / (S·τ): bytes per generated token, τ = tokens per step per sequence (>= 1).
    [[nodiscard]] double bytes_per_token(double tokens_per_step = 1.0) const;
    /// S·τ·η·BW / B_step in tok/s (aggregate over S sequences).
    [[nodiscard]] double ceiling_tps(double bandwidth_gbps, double efficiency = 1.0,
                                     double tokens_per_step = 1.0) const;
};

/// η = measured_tps · bytes_per_token / BW. Throws Error(Config) on BW <= 0.
[[nodiscard]] double decode_efficiency(double measured_tps, double bytes_per_token, double bandwidth_gbps);

// ---- record -----------------------------------------------------------------------------

enum class Phase : std::uint8_t { Cold, Steady };
[[nodiscard]] std::string_view to_string(Phase p) noexcept;

struct Invocation {
    std::string binary;       ///< absolute path
    std::string version;      ///< e.g. "0.5.0-dev (build 1, commit bd4f514)"
    std::string commit;       ///< e.g. "bd4f514"
    std::vector<std::string> argv;  ///< full command line, argv[0] first
    std::map<std::string, std::string> environment;  ///< relevant variables only
};

struct BenchmarkRecord {
    // TRD §28 fields.
    std::string model;
    std::string model_hash;  ///< SHA-256 hex of the trunk GGUF (empty = not computed)
    std::string pack;
    std::string backend;     ///< "vulkan" | "hip" | "cpu"
    std::string driver;
    std::string gpu;
    std::string quantization;
    std::string lm_head;
    std::uint64_t context = 0;
    std::uint32_t batch = 1;
    std::uint32_t concurrency = 1;
    std::optional<double> ttft_ms;
    std::optional<double> ttft_cached_ms;
    std::optional<double> prompt_tps;
    std::optional<double> decode_tps;
    std::optional<double> decode_effective_tps;
    std::optional<double> mtp_acceptance;
    std::optional<double> prefix_cache_hit_rate;
    std::map<std::string, double> memory_by_tier_gb;
    std::optional<double> load_time_s;
    std::optional<double> temperature_c;
    std::map<std::string, std::uint32_t> clocks_mhz;
    std::string power_mode;
    // HALO additions.
    std::string suite;   ///< "micro" | "model" | "system" | "baseline"
    std::string mode;    ///< e.g. "decode", "prompt", "mtp_decode", "prefix_replay", ...
    std::string engine;  ///< "halo" | "llama-bench" | "llama-server" | "ollama"
    std::string pack_hash;  ///< PACK_ID: hash over trunk + MTP file hashes
    std::string mtp_hash;
    std::uint32_t repetition = 0;
    Phase phase = Phase::Steady;
    std::string host_label;
    std::optional<double> measured_bandwidth_gbps;
    std::optional<double> predicted_bytes_per_token;
    std::optional<double> efficiency;
    std::optional<HardwareState> hardware_before;
    std::optional<HardwareState> hardware_after;
    std::optional<Invocation> invocation;
    nlohmann::json extra = nlohmann::json::object();
};

/// Fill measured_bandwidth_gbps / predicted_bytes_per_token / efficiency from a model and a
/// measured bandwidth (no-op on efficiency if decode_tps is absent).
void apply_efficiency(BenchmarkRecord& r, const DecodeByteModel& model, double bandwidth_gbps,
                      double tokens_per_step = 1.0);

void to_json(nlohmann::json& j, const Invocation& v);
void from_json(const nlohmann::json& j, Invocation& v);
void to_json(nlohmann::json& j, const BenchmarkRecord& v);
/// Throws Error(Config) on a missing/ill-typed required field or a foreign schema id.
void from_json(const nlohmann::json& j, BenchmarkRecord& v);

// ---- roll-up ----------------------------------------------------------------------------

/// Statistics of one metric over the records of one configuration group.
struct MetricSummary {
    SampleStats stats;
    Stability stability = Stability::TooFewSamples;
};

/// One configuration group (suite, mode, engine, context, concurrency, phase): measured
/// repetitions rolled up per metric (TRD §50).
struct RunSummary {
    std::string suite;
    std::string mode;
    std::string engine;
    std::uint64_t context = 0;
    std::uint32_t concurrency = 1;
    Phase phase = Phase::Steady;
    std::uint32_t warmup_count = 0;
    std::uint32_t measured_count = 0;
    std::uint32_t required_measured = 0;  ///< methodology minimum for this suite
    bool conformant = false;              ///< measured_count >= required_measured
    std::map<std::string, MetricSummary> metrics;  ///< "decode_tps" -> stats, ...
};

/// Group records by (suite, mode, engine, context, concurrency, phase) and compute
/// per-metric statistics: every TRD §28 numeric metric plus `efficiency`, plus every
/// numeric top-level member of `extra` (as "extra.<name>").
///
/// Cold-phase records are the warm-up / first-after-load runs: they form their own (cold)
/// group, never mix into the steady statistics (TRD §50: never compare cold against warm),
/// and are also counted as `warmup_count` of the matching steady group.
/// `required_measured` = kMicroMinMeasured for suite "micro", kEndToEndMinRepetitions
/// otherwise; a steady micro group is conformant only with >= kMicroMinWarmup warm-ups.
/// The stability policy's min_samples is replaced by required_measured per group.
[[nodiscard]] std::vector<RunSummary> summarize(const std::vector<BenchmarkRecord>& records,
                                                const StabilityPolicy& policy = {});

void to_json(nlohmann::json& j, const RunSummary& v);

}  // namespace halo::profiling
