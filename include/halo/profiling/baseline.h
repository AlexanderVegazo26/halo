#pragma once
// Baseline adapters (TRD §51): llama.cpp `llama-bench` (Vulkan/RADV and HIP builds; the
// backend is a config field), `llama-server` with MTP (`--spec-type draft-mtp`), and
// Ollama (HTTP API). Every run records binary version + commit, backend/driver versions,
// the full command line, the model hash (streamed SHA-256), the environment actually
// passed, the power mode and the results, in the same halo.bench.record/1 schema as the
// HALO suites (suite "baseline").
//
// Processes are exec'd without a shell (subprocess.h); HTTP goes only to loopback
// (http_client.h). Parsers treat tool output as untrusted: every field is type- and
// range-checked and a violation is Error(Api) (malformed output), never UB.
//
// Mode / context contract (what ProfileDb::latest_run is queried with):
//   llama-bench  pp test (n_prompt > 0, n_gen = 0): mode "pp<n_prompt>", or
//                "pp<n_prompt>@d<n_depth>" when n_depth > 0; prompt_tps; context =
//                n_depth + n_prompt.
//                tg test (n_gen > 0, n_prompt = 0): mode "tg<n_gen>" / "tg<n_gen>@d<n_depth>";
//                decode_tps; context = n_depth + n_gen.
//                pg test (both > 0): mode "pp<p>+tg<n>" (@d...), the combined rate in
//                extra.combined_tps only; context = n_depth + p + n.
//                One record per llama-bench sample (repetition = sample index).
//   llama-server mode "completion" (MTP off) or "mtp_completion" (MTP on); prompt_tps,
//                decode_tps (MTP off) / decode_effective_tps (MTP on), mtp_acceptance =
//                draft_n_accepted / draft_n (null when draft_n = 0); context =
//                prompt_n + predicted_n. ttft_ms is null: /completion without streaming does
//                not expose it.
//   ollama       mode "generate"; prompt_tps = prompt_eval_count / prompt_eval_duration,
//                decode_tps = eval_count / eval_duration (Ollama durations are NANOSECONDS);
//                context = prompt_eval_count + eval_count.
//
// Stored baselines (autotune::store_baseline) are the per-configuration aggregates of
// aggregate_records(): the MEDIAN of each metric over the measured samples.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/profiling/record.h"
#include "halo/profiling/subprocess.h"
#include "halo/profiling/suite.h"

namespace halo::profiling {

/// Parsed "version: 0.5.0-dev (build 1, commit bd4f514)" (llama.cpp --version, stderr).
struct BuildInfo {
    std::string version;  ///< full first line, e.g. "0.5.0-dev (build 1, commit bd4f514)"
    std::string build;
    std::string commit;
};
/// Throws Error(Api) when no "version:" line is found.
[[nodiscard]] BuildInfo parse_llama_version(std::string_view text);

/// Name the ggml backend registry reports for a HALO backend label: cpu -> "CPU",
/// vulkan -> "Vulkan" (GGML_VK_NAME), hip -> "ROCm" (GGML_CUDA_NAME under GGML_USE_HIP).
/// Throws Error(Config) for another label.
[[nodiscard]] std::string_view ggml_backend_name(std::string_view backend);

/// Fields shared by every adapter.
struct BaselineCommon {
    std::string model_name;                ///< e.g. "Qwen/Qwen3.8-27B"
    std::string pack;                      ///< e.g. "unsloth UD-Q4_K_XL"
    std::optional<std::string> model_hash; ///< precomputed SHA-256 hex; else hashed from the file
    std::string host_label;                ///< required (D-001)
    EnvironmentOptions environment;        ///< hardware snapshots before/after (M1)
    ProcessOptions process;                ///< env, timeout, output caps
    StabilityPolicy stability;
};

// ---- llama-bench ----------------------------------------------------------------------------

struct LlamaBenchConfig {
    std::filesystem::path binary;  ///< e.g. <llama.cpp vulkan build>/bin/llama-bench
    std::filesystem::path model;
    std::string backend = "vulkan";  ///< vulkan | hip | cpu (must match the binary's backends)
    std::vector<unsigned> n_prompt{512};
    std::vector<unsigned> n_gen{128};
    std::vector<unsigned> n_depth{0};
    unsigned repetitions = 5;
    int n_gpu_layers = 999;
    std::optional<unsigned> threads;
    std::string flash_attn = "on";  ///< on | off | auto (llama-bench -fa at bd4f514)
    bool no_warmup = false;
    std::vector<std::string> extra_args;
    BaselineCommon common;
};

[[nodiscard]] std::vector<std::string> llama_bench_argv(const LlamaBenchConfig& c);

/// Context the parser needs beyond the JSON (identity and the expected backend).
struct BaselineParseContext {
    std::string engine;   ///< "llama-bench" | "llama-server" | "ollama"
    std::string backend;  ///< expected backend label
    std::string model_name;
    std::string pack;
    std::string model_hash;
    std::string host_label;
    std::optional<std::string> expected_commit;  ///< from --version, cross-checked
};

struct BaselineParse {
    std::vector<BenchmarkRecord> records;
    std::vector<std::string> notes;     ///< informational cross-check results
    std::vector<std::string> failures;  ///< make the artifact invalid (e.g. backend mismatch)
};

/// Parses `llama-bench -o json` output (a JSON array of tests).
[[nodiscard]] BaselineParse parse_llama_bench_json(const nlohmann::json& j, const BaselineParseContext& ctx);

/// Hash model, read --version, run, parse; before/after hardware snapshots and verdicts as
/// in the M1 suites. Tool failures (non-zero exit, timeout, unparsable output, backend
/// mismatch) become FAILED notes and an invalid artifact rather than exceptions. Invalid
/// configuration throws Error(Config); a missing/unreadable model file or binary throws
/// Error(Io). pack_hash = make_pack_id(model_hash, none) when model_hash is a SHA-256.
[[nodiscard]] SuiteArtifact run_llama_bench(const LlamaBenchConfig& c);

// ---- llama-server (MTP) ---------------------------------------------------------------------

struct LlamaServerConfig {
    std::filesystem::path binary;
    std::filesystem::path model;
    std::string backend = "vulkan";
    std::uint16_t port = 18084;
    std::size_t ctx_size = 8192;
    int n_gpu_layers = 999;
    unsigned parallel = 1;
    std::string flash_attn = "on";
    bool mtp = true;
    unsigned spec_draft_n_max = 2;  ///< engine report: deeper than 2 loses acceptance
    std::vector<std::string> extra_args;
    std::string prompt = "Explain speculative decoding in one sentence.";
    unsigned n_predict = 128;
    unsigned warmup = 1;
    unsigned repetitions = 3;
    std::uint64_t seed = 42;
    std::chrono::milliseconds startup_timeout{std::chrono::minutes(10)};
    std::chrono::milliseconds request_timeout{std::chrono::minutes(10)};
    std::optional<std::filesystem::path> log_path;  ///< server stdout+stderr
    BaselineCommon common;
};

[[nodiscard]] std::vector<std::string> llama_server_argv(const LlamaServerConfig& c);

/// Parses one non-streaming `/completion` response. `mtp` = speculation was requested:
/// then a response without draft counters is a failure.
[[nodiscard]] BaselineParse parse_llama_server_completion(const nlohmann::json& j, const BaselineParseContext& ctx,
                                                          bool mtp);

/// Starts the server (argv, no shell), waits for GET /health == 200, sends warmup +
/// repetitions POST /completion requests (cache_prompt false), and always tears the
/// server's process group down before returning or throwing.
[[nodiscard]] SuiteArtifact run_llama_server(const LlamaServerConfig& c);

// ---- Ollama ---------------------------------------------------------------------------------

struct OllamaConfig {
    std::string host = "127.0.0.1";  ///< loopback only
    std::uint16_t port = 11434;
    std::string model = "qwen3.8:27b";
    std::string prompt = "Explain speculative decoding in one sentence.";
    unsigned num_predict = 128;
    unsigned warmup = 1;
    unsigned repetitions = 3;
    std::uint64_t seed = 42;
    std::chrono::milliseconds request_timeout{std::chrono::minutes(10)};
    BaselineCommon common;  ///< model_hash: Ollama's model digest is used when not given
};

/// Parses one non-streaming `/api/generate` response.
[[nodiscard]] BaselineParse parse_ollama_generate(const nlohmann::json& j, const BaselineParseContext& ctx);

/// GET /api/version, GET /api/tags (model digest), then warmup + repetitions
/// POST /api/generate (stream false, temperature 0, seed).
[[nodiscard]] SuiteArtifact run_ollama(const OllamaConfig& c);

// ---- stored form ----------------------------------------------------------------------------

/// One record per steady configuration group of `a.summaries`: the group's first record
/// with every summarised TRD metric replaced by the group MEDIAN, repetition 0, and
/// extra.aggregate = {"method": "median", "n": N, "metrics": {name: SampleStats}}.
[[nodiscard]] std::vector<BenchmarkRecord> aggregate_records(const SuiteArtifact& a);

}  // namespace halo::profiling
