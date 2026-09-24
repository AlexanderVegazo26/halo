#pragma once
// Frozen reference workloads (TRD §67 v0.1 "reference workloads frozen", PRD Phase 0) and
// deterministic synthetic token generation.
//
// Workload files live under bench/workloads/*.json (schema "halo.bench.workload/1"; see
// docs/benchmarks.md). They hold *text* (system prompt, tools, turns) plus, where real text
// is not needed, a synthetic segment `{"synthetic_tokens": N, "seed": S}`. Turning text
// into tokens needs the model's tokenizer and chat template, which the profiling library
// deliberately does not link: callers supply a `WorkloadEncoder` (the CLI wires the Engine's
// tokenizer + template; tests use a fake).
//
// Workload files are external input: sizes and counts are bounded (kMaxWorkloadBytes,
// kMaxSyntheticTokens, kMaxTurns, kMaxAgents) and a violation is Error(Config).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace halo::profiling {

inline constexpr std::string_view kWorkloadSchema = "halo.bench.workload/1";
inline constexpr std::uintmax_t kMaxWorkloadBytes = 16ULL << 20;
inline constexpr std::uint64_t kMaxSyntheticTokens = 1ULL << 20;  ///< 1M (YaRN experiment ceiling)
inline constexpr std::size_t kMaxTurns = 1024;
inline constexpr std::size_t kMaxAgents = 64;

/// One message. Exactly one of `text` / `synthetic_tokens` is set.
struct WorkloadMessage {
    std::string role;  ///< "system" | "user" | "assistant" | "tool"
    std::optional<std::string> text;
    std::optional<std::uint64_t> synthetic_tokens;
    std::uint64_t seed = 0;
};

/// A scripted agent: the turns are replayed in order; each turn appends the user message
/// to the running conversation and generates `max_tokens` (multi-turn replay reuses the
/// growing prefix, which is what the prefix cache is measured on).
struct AgentScript {
    std::string id;
    std::vector<WorkloadMessage> preamble;  ///< system prompt + tool definitions
    std::vector<WorkloadMessage> turns;     ///< user messages, one generation each
    std::size_t max_tokens = 128;
    std::uint64_t think_time_ms = 0;        ///< delay between turns (0 = back-to-back)
};

struct Workload {
    std::string id;
    std::string kind;  ///< "agent_replay" | "long_context" | "concurrency"
    std::string description;
    std::string provenance;
    nlohmann::json tools = nlohmann::json::array();  ///< OpenAI-style tool list (text only)
    std::vector<AgentScript> agents;  ///< >= 1
};

/// Parse + validate. Throws Error(Config) with the offending path/field.
[[nodiscard]] Workload parse_workload(const nlohmann::json& j);
/// Read (size-bounded) + parse. Throws Error(Io) if unreadable.
[[nodiscard]] Workload load_workload(const std::filesystem::path& path);

/// Deterministic token ids in [first_id, first_id + id_range): SplitMix64 over `seed`,
/// reduced with a 128-bit multiply (portable; independent of <random> distributions).
/// Throws Error(Config) if id_range == 0 or n > kMaxSyntheticTokens.
[[nodiscard]] std::vector<std::int32_t> synthetic_tokens(std::uint64_t seed, std::size_t n,
                                                         std::int32_t id_range, std::int32_t first_id = 0);

/// Turns a conversation (preamble + turns[0..=turn]) into prompt token ids.
using WorkloadEncoder =
    std::function<std::vector<std::int32_t>(const Workload&, const AgentScript&, std::size_t turn)>;

/// Encoder that needs no tokenizer: every message becomes its synthetic tokens, and text
/// messages become ceil(bytes/4) synthetic tokens seeded by a hash of the text (a stand-in
/// with roughly the right length, NOT a tokenization). Deterministic.
[[nodiscard]] WorkloadEncoder synthetic_encoder(std::int32_t vocab_size);

}  // namespace halo::profiling
