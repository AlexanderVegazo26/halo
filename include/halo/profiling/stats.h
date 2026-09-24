#pragma once
// Rolled-up benchmark statistics (TRD §50, PRD PR-003) and the instability classifier the
// autotuner reuses (TRD §58: "reject unstable candidates").
//
// Conventions (fixed; tests pin them against hand-computed values):
//   * mean: arithmetic mean.
//   * stddev: SAMPLE standard deviation (divisor n-1); 0 when n == 1. Same convention as
//     halo::hardware::Stats.
//   * percentiles p50/p95/p99 and median: LINEAR INTERPOLATION between closest ranks,
//     i.e. Hyndman & Fan type 7 (numpy.percentile default, Excel PERCENTILE.INC):
//         h = (n - 1) * q;  p = x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)])
//     on the ascending-sorted sample. median == p50 (for even n this is the mean of the two
//     middle values). NOTE: halo::hardware::Stats::p95 uses nearest-rank instead; the two
//     are different conventions and must not be compared value-for-value.
//   * cv (coefficient of variation) = stddev / |mean|; +inf when mean == 0 and stddev > 0,
//     0 when both are 0.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

namespace halo::profiling {

struct SampleStats {
    std::size_t n = 0;
    double min = 0;
    double max = 0;
    double mean = 0;
    double median = 0;
    double stddev = 0;
    double p50 = 0;
    double p95 = 0;
    double p99 = 0;
    double cv = 0;
};

/// Type-7 percentile of an ascending-sorted, non-empty sample; q in [0, 1].
/// Throws Error(Config) on an empty sample or q outside [0, 1].
[[nodiscard]] double percentile_sorted(std::span<const double> sorted, double q);

/// Throws Error(Config) on an empty sample or any non-finite value.
[[nodiscard]] SampleStats compute_sample_stats(std::span<const double> samples);

/// Methodology minimums (TRD §50 / PRD PR-003).
inline constexpr unsigned kMicroMinWarmup = 5;
inline constexpr unsigned kMicroMinMeasured = 20;
inline constexpr unsigned kEndToEndMinRepetitions = 3;

enum class Stability : std::uint8_t {
    Stable,
    Unstable,       ///< cv above threshold
    TooFewSamples,  ///< fewer measured samples than the methodology minimum
};
[[nodiscard]] std::string_view to_string(Stability s) noexcept;

struct StabilityPolicy {
    /// Coefficient-of-variation threshold. 0.05 (5 %) is a HALO policy default, not a
    /// measured property of any machine; the autotuner and the harness share it.
    double max_cv = 0.05;
    std::size_t min_samples = kMicroMinMeasured;
};

/// Stable iff n >= min_samples and cv <= max_cv.
[[nodiscard]] Stability classify_stability(const SampleStats& s, const StabilityPolicy& policy = {}) noexcept;

void to_json(nlohmann::json& j, const SampleStats& v);
void from_json(const nlohmann::json& j, SampleStats& v);

}  // namespace halo::profiling
