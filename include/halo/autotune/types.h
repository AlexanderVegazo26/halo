#pragma once
// Plain value types shared by the tuner, the profile database and the runtime lookup.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/autotune/cost_model.h"
#include "halo/profiling/stats.h"

namespace halo::autotune {

/// Identifies one tunable operator instance: family ("MATMUL") + canonical shape string
/// ("T=1,K=5120,N=17408,w=f32"). Both are stored verbatim.
struct OpKey {
    std::string family;
    std::string shape;
    friend bool operator==(const OpKey&, const OpKey&) = default;
    friend auto operator<=>(const OpKey&, const OpKey&) = default;
};

/// One kernel configuration: named integer parameters. The canonical text form is
/// "name=value;name=value" in name order (std::map), e.g. "chunk=64;threads=8".
/// Names are [a-z0-9_]{1,32}; values are signed 64-bit.
struct Candidate {
    std::map<std::string, std::int64_t> params;

    [[nodiscard]] std::string to_string() const;
    /// Throws Error(Config) on malformed text (also used on text read from the DB).
    [[nodiscard]] static Candidate parse(std::string_view text);
    /// Throws Error(Config) when absent.
    [[nodiscard]] std::int64_t get(std::string_view name) const;

    friend bool operator==(const Candidate&, const Candidate&) = default;
};

enum class Strategy : std::uint8_t { Exhaustive, Random, Grid, Heuristic, Bayesian };
[[nodiscard]] std::string_view to_string(Strategy s) noexcept;

/// The metric a winner is chosen by: median latency in ns (robust to outliers); ties go to
/// the lexicographically smallest canonical candidate string.
inline constexpr std::string_view kWinnerMetric = "median_ns";

struct CandidateResult {
    Candidate candidate;
    std::optional<CostEstimate> predicted;
    bool measured = false;
    profiling::SampleStats stats;  ///< valid when measured
    profiling::Stability stability = profiling::Stability::TooFewSamples;
    std::string rejected;  ///< empty = eligible; otherwise why it cannot win
};

struct OpTuneResult {
    OpKey key;
    std::string backend;
    Strategy strategy = Strategy::Heuristic;
    std::vector<CandidateResult> candidates;  ///< in evaluation order
    std::optional<Candidate> winner;
    double winner_median_ns = 0;
    bool persisted = false;  ///< winner written to winning_configuration
    std::string note;        ///< e.g. "no stable candidate; existing winner left unchanged"
};

void to_json(nlohmann::json& j, const CandidateResult& v);
void to_json(nlohmann::json& j, const OpTuneResult& v);

}  // namespace halo::autotune
