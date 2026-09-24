#pragma once
// Benchmark suites (TRD §28): micro (per-kernel callback), model (PRD §18 / PR-004 matrix),
// system (N concurrent agents). Model and system suites drive only the runtime::Engine
// interface (include/halo/runtime/engine.h); they never construct an Engine themselves
// except through the optional caller-supplied factory used to measure load time.
//
// Methodology (TRD §50, PRD PR-003): environment snapshot -> (cold) warm-up -> measured
// repetitions -> environment snapshot. Micro: >= 5 warm-up + >= 20 measured iterations;
// end-to-end: >= 3 repetitions. Configs below that minimum are refused with Error(Config)
// unless `allow_nonconformant` is set (unit tests only); such artifacts say so.
//
// Timing is taken by the harness from its injected Clock, never from the engine's self-
// reported numbers (those are kept under `extra.engine_*` for cross-checking):
//   ttft_ms     = first token callback - call start
//   prompt_tps  = prompt_tokens / ttft  (includes the first decode step; stated, not hidden)
//   decode_tps  = (generated - 1) / (last token callback - first token callback)
// for single-request modes. concurrency/system records report aggregate generated tokens /
// wall time (prefill, and for system think time, included) and leave efficiency null.
// Every artifact carries the hardware state before and after, the thermal-drift verdict and
// the power-mode comparability verdict. Numbers produced on the dev host are harness smoke
// tests only (D-001); `identity.host_label` says where a record came from.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/hardware.h"
#include "halo/profiling/record.h"
#include "halo/profiling/timer.h"
#include "halo/profiling/workload.h"
#include "halo/runtime/engine.h"

namespace halo::profiling {

/// Record fields every record of a suite shares (model identity, backend, host label).
struct RecordIdentity {
    std::string model;
    std::string model_hash;
    std::string pack;
    std::string pack_hash;
    std::string mtp_hash;
    std::string backend;
    std::string driver;
    std::string gpu;
    std::string quantization;
    std::string lm_head;
    std::string engine = "halo";
    std::string host_label;  ///< required, e.g. "dev-host" or "evo-x2"
};

/// Environment capture for the before/after hardware snapshots. `enabled = false` skips
/// discovery entirely (the artifact then says hardware state was not captured and is
/// marked invalid for comparison).
struct EnvironmentOptions {
    bool enabled = true;
    hardware::DiscoveryOptions discovery;
    std::string platform_power_mode;  ///< EVO-X2 BIOS/EC mode label (not visible in sysfs)
    double thermal_drift_threshold_c = kDefaultThermalDriftC;
    /// Test hook: called once both snapshots are taken and before the thermal/power-mode
    /// verdicts, so tests can patch them (e.g. simulate drift). Not used by production callers.
    std::function<void(HardwareState& before, HardwareState& after)> patch_states;
    /// Test hook: supplies `captured_at` for each snapshot (null = system clock, UTC).
    std::function<std::string()> utc_now;
};

/// Optional D-011 efficiency inputs (D-014 fields). Without them the efficiency fields
/// stay null.
///
/// Per record the harness overrides `sequences` and `context_tokens`. For every mode except
/// mtp_decode it also forces draft_tokens = 0 and w_mtp_gb = 0 (no draft is read). For
/// mtp_decode `byte_model.draft_tokens` (n) must be the engine's draft length, and τ
/// (tokens emitted per verification step) is derived from the engine's counters assuming
/// each step emits accepted + 1 tokens: τ = generated / (generated - accepted_draft_tokens).
/// The derived τ is recorded as extra.tau_observed.
struct EfficiencyInputs {
    DecodeByteModel byte_model;
    double measured_bandwidth_gbps = 0;
};

struct SuiteArtifact {
    std::string suite;
    std::optional<HardwareState> hardware_before;
    std::optional<HardwareState> hardware_after;
    ThermalCheck thermal;
    Comparability power_mode_pinned;  ///< before vs after comparability
    bool conformant = false;          ///< every steady group met the methodology minimum
    bool valid = false;               ///< thermal ok && power mode pinned && conformant
    std::vector<std::string> notes;   ///< skipped modes/contexts with the reason, etc.
    std::vector<BenchmarkRecord> records;  ///< cold (warm-up) + steady
    std::vector<RunSummary> summaries;
};

void to_json(nlohmann::json& j, const SuiteArtifact& v);

// ---- micro ------------------------------------------------------------------------------

struct MicroBenchmark {
    std::string name;             ///< e.g. "MATMUL/T=1,K=5120,N=17408"
    std::function<void()> run;    ///< one invocation
    std::uint64_t bytes_per_call = 0;  ///< compulsory traffic (0 = unknown)
};

struct MicroSuiteConfig {
    RecordIdentity identity;
    std::vector<MicroBenchmark> benchmarks;
    unsigned warmup = kMicroMinWarmup;
    unsigned iterations = kMicroMinMeasured;
    bool allow_nonconformant = false;
    StabilityPolicy stability;
    EnvironmentOptions environment;
};

/// One record per iteration: suite "micro", mode = benchmark name, extra.latency_ns and
/// (when bytes are known) extra.gbps. Warm-up iterations are Cold records.
[[nodiscard]] SuiteArtifact run_micro_suite(const MicroSuiteConfig& config,
                                            const Clock& clock = SteadyClock::instance());

// ---- model ------------------------------------------------------------------------------

enum class ModelMode : std::uint8_t {
    LoadTime,         ///< needs `engine_factory`
    PromptProcessing, ///< fresh prompt of `prompt_tokens`, 1 generated token
    Decode,           ///< per context: prompt to depth ctx - decode_tokens, then decode
    MtpDecode,        ///< as Decode, requires model().has_mtp; records acceptance
    Concurrency,      ///< per level: N simultaneous decode requests at `concurrency_context`
    PrefixReplay,     ///< cold prompt at replay context, then prefix + suffix
    CancellationStorm,///< N simultaneous requests cancelled after k tokens, then a probe
};
[[nodiscard]] std::string_view to_string(ModelMode m) noexcept;

struct ModelSuiteConfig {
    RecordIdentity identity;
    std::set<ModelMode> modes{ModelMode::LoadTime, ModelMode::PromptProcessing, ModelMode::Decode,
                              ModelMode::MtpDecode, ModelMode::Concurrency, ModelMode::PrefixReplay,
                              ModelMode::CancellationStorm};
    std::vector<std::uint64_t> contexts{4096, 32768, 131072, 262144};  ///< PRD §18
    std::vector<std::uint32_t> concurrency{1, 2, 4, 8};                ///< PRD §18
    std::uint64_t concurrency_context = 4096;
    std::size_t prompt_tokens = 512;   ///< PR-004 "Prefill, 512-token prompt"
    std::size_t decode_tokens = 128;
    std::uint64_t prefix_replay_context = 32768;  ///< PR-004 "Cached TTFT, 32K prefix replay"
    std::size_t prefix_replay_suffix = 256;
    std::uint32_t storm_requests = 8;
    std::size_t storm_cancel_after = 4;
    unsigned warmup = 1;
    unsigned repetitions = kEndToEndMinRepetitions;
    bool allow_nonconformant = false;
    std::uint64_t seed = 0x48414C4FULL;
    SamplingParams sampling = [] { SamplingParams p; p.temperature = 0.0f; return p; }();
    /// Creates a fresh engine (for LoadTime). Engines it returns are destroyed after timing.
    std::function<std::unique_ptr<runtime::Engine>()> engine_factory;
    std::optional<EfficiencyInputs> efficiency;
    StabilityPolicy stability;
    EnvironmentOptions environment;
};

[[nodiscard]] SuiteArtifact run_model_suite(runtime::Engine& engine, const ModelSuiteConfig& config,
                                            const Clock& clock = SteadyClock::instance());

// ---- system -----------------------------------------------------------------------------

struct SystemSuiteConfig {
    RecordIdentity identity;
    Workload workload;           ///< kind "concurrency" or "agent_replay"
    WorkloadEncoder encoder;     ///< required (synthetic_encoder() when no tokenizer)
    std::uint32_t agents = 4;    ///< agents run concurrently; scripts are reused round-robin
    unsigned warmup = 1;
    unsigned repetitions = kEndToEndMinRepetitions;
    bool allow_nonconformant = false;
    SamplingParams sampling = [] { SamplingParams p; p.temperature = 0.0f; return p; }();
    /// Called for an agent's think time between turns; null = std::this_thread::sleep_for.
    /// Called concurrently from the agent threads, so it must be thread-safe. Tests pass a
    /// hook that advances a ManualClock instead of sleeping.
    std::function<void(std::uint64_t ms)> think;
    std::optional<EfficiencyInputs> efficiency;
    StabilityPolicy stability;
    EnvironmentOptions environment;
};

/// Each agent replays its script as a growing conversation: turn k's prompt is turn k-1's
/// prompt + the tokens it generated + the tokens the encoder adds for turn k.
/// One record per repetition: suite "system", mode "agents", concurrency = agents;
/// decode_tps = aggregate generated tokens / wall time, ttft_ms = p95 over all turns,
/// ttft_cached_ms = mean TTFT of turns >= 1 (the prefix-reuse turns),
/// prefix_cache_hit_rate = sum(cached_prompt_tokens) / sum(prompt_tokens).
[[nodiscard]] SuiteArtifact run_system_suite(runtime::Engine& engine, const SystemSuiteConfig& config,
                                             const Clock& clock = SteadyClock::instance());

// ---- dispatch (entry point for `halo bench`) -----------------------------------------------

enum class SuiteKind : std::uint8_t { Micro, Model, System };

struct SuiteRequest {
    SuiteKind kind = SuiteKind::Model;
    std::optional<MicroSuiteConfig> micro;
    std::optional<ModelSuiteConfig> model;
    std::optional<SystemSuiteConfig> system;
    runtime::Engine* engine = nullptr;  ///< required for Model/System
};

/// Runs the requested suite. Throws Error(Config) when the matching config (or the engine
/// for Model/System) is missing.
[[nodiscard]] SuiteArtifact run_suite(const SuiteRequest& request, const Clock& clock = SteadyClock::instance());

}  // namespace halo::profiling
