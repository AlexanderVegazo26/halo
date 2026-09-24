#pragma once
// Offline autotuner (TRD §58, §56 steps 4-5, §64). Explicit call only: it measures
// candidates and writes the profile database. The runtime never includes this header; it
// uses lookup.h.
//
// Candidates come from a generic interface so GPU backends plug in later. The task's
// sketch `struct TunableOp { OpKey key; candidates(); run(const Candidate&); }` is an
// abstract class here so that backends can hold device state; `run` returns an optional
// device-measured time (GPU timestamps) and otherwise the tuner times the call itself.
//
// Methodology (TRD §50, reused from halo_profiling): per measured candidate
// `warmup` (>= 5) untimed + `iterations` (>= 20) timed runs -> SampleStats; candidates that
// profiling::classify_stability does not call Stable are rejected, as are candidates that
// fail validate(). Winner = smallest median ns among eligible candidates (tie: smallest
// canonical string). If no candidate is eligible, no winner is persisted and any existing
// winner stays in place (the runs are still recorded).
//
// Strategies:
//   EXHAUSTIVE  every candidate.
//   RANDOM      `random_budget` candidates drawn without replacement by a SplitMix64-driven
//               partial Fisher-Yates over the candidate list (seeded; portable; does not
//               use std::shuffle, whose algorithm is implementation-defined).
//   GRID        per parameter, the distinct values sorted ascending; keep indices
//               0, s, 2s, ... and the last (s = grid_stride); measure the candidates whose
//               every value is kept. A strict subset once a dimension has > 2 values.
//   HEURISTIC   (default) cost-model prefilter: the `heuristic_keep` best predicted
//               candidates (ties: canonical string), then measured selection. Requires a
//               CostModel and cost() for every candidate.
//   BAYESIAN    not implemented: Error(Unsupported).

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "halo/autotune/cost_model.h"
#include "halo/autotune/db.h"
#include "halo/autotune/profile_key.h"
#include "halo/autotune/types.h"
#include "halo/profiling/stats.h"
#include "halo/profiling/timer.h"

namespace halo::autotune {

struct Measurement {
    std::optional<std::int64_t> device_ns;  ///< set by backends with their own timers
};

class TunableOp {
public:
    virtual ~TunableOp() = default;
    [[nodiscard]] virtual OpKey key() const = 0;
    [[nodiscard]] virtual std::string backend() const = 0;  ///< "cpu" | "vulkan" | "hip"
    [[nodiscard]] virtual std::vector<Candidate> candidates() const = 0;
    /// One invocation of the operator with `c`. Called warmup + iterations times.
    virtual Measurement run(const Candidate& c) = 0;
    /// Correctness gate: false (with a reason) makes the candidate ineligible to win.
    virtual bool validate(const Candidate& c, std::string& why) {
        (void)c;
        (void)why;
        return true;
    }
    /// Cost-model inputs (needed by HEURISTIC and for the predicted column).
    [[nodiscard]] virtual std::optional<CostInputs> cost(const Candidate& c) const {
        (void)c;
        return std::nullopt;
    }
};

struct TuneOptions {
    Strategy strategy = Strategy::Heuristic;
    std::uint64_t seed = 0x48414C4FULL;
    std::size_t random_budget = 8;
    std::size_t grid_stride = 2;
    std::size_t heuristic_keep = 4;
    unsigned warmup = profiling::kMicroMinWarmup;
    unsigned iterations = profiling::kMicroMinMeasured;
    bool allow_nonconformant = false;  ///< unit tests only
    profiling::StabilityPolicy stability;
    std::optional<CostModel> cost_model;
};

/// Which candidates a strategy measures, in evaluation order (pure; exposed for tests).
/// Throws Error(Unsupported) for BAYESIAN and Error(Config) for invalid options (e.g.
/// HEURISTIC without a cost model or with a candidate lacking cost inputs).
[[nodiscard]] std::vector<Candidate> plan_candidates(const std::vector<Candidate>& all, const TuneOptions& options,
                                                     const std::function<std::optional<CostInputs>(const Candidate&)>& cost);

struct TuneReport {
    ProfileKey key;
    std::vector<OpTuneResult> ops;
};

/// Tunes every op and persists each result (ProfileDb::persist_tune). Throws Error(Config)
/// below the TRD §50 minimum unless allow_nonconformant, Error(Unsupported) for BAYESIAN.
/// An op that throws from run() aborts the tune (the error propagates; results of ops
/// already finished are persisted).
[[nodiscard]] TuneReport tune(ProfileDb& db, const ProfileKey& key, std::span<TunableOp* const> ops,
                              const TuneOptions& options,
                              const profiling::Clock& clock = profiling::SteadyClock::instance());

void to_json(nlohmann::json& j, const TuneReport& r);

}  // namespace halo::autotune
