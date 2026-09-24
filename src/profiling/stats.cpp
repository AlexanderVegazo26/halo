#include "halo/profiling/stats.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"

namespace halo::profiling {

double percentile_sorted(std::span<const double> sorted, double q) {
    HALO_CHECK(!sorted.empty(), ErrorCode::Config, "percentile of an empty sample");
    HALO_CHECK(q >= 0.0 && q <= 1.0, ErrorCode::Config, "percentile q={} outside [0, 1]", q);
    const double h = static_cast<double>(sorted.size() - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    if (lo + 1 >= sorted.size()) return sorted.back();
    const double frac = h - static_cast<double>(lo);
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
}

SampleStats compute_sample_stats(std::span<const double> samples) {
    HALO_CHECK(!samples.empty(), ErrorCode::Config, "statistics of an empty sample");
    for (const double v : samples) {
        HALO_CHECK(std::isfinite(v), ErrorCode::Config, "non-finite sample value");
    }
    std::vector<double> s(samples.begin(), samples.end());
    std::sort(s.begin(), s.end());
    SampleStats r;
    r.n = s.size();
    r.min = s.front();
    r.max = s.back();
    double sum = 0;
    for (const double v : s) sum += v;
    r.mean = sum / static_cast<double>(r.n);
    if (r.n > 1) {
        double ss = 0;
        for (const double v : s) ss += (v - r.mean) * (v - r.mean);
        r.stddev = std::sqrt(ss / static_cast<double>(r.n - 1));
    }
    r.p50 = percentile_sorted(s, 0.50);
    r.median = r.p50;
    r.p95 = percentile_sorted(s, 0.95);
    r.p99 = percentile_sorted(s, 0.99);
    if (r.mean != 0.0) {
        r.cv = r.stddev / std::fabs(r.mean);
    } else {
        r.cv = r.stddev > 0 ? std::numeric_limits<double>::infinity() : 0.0;
    }
    return r;
}

std::string_view to_string(Stability s) noexcept {
    switch (s) {
        case Stability::Stable: return "stable";
        case Stability::Unstable: return "unstable";
        case Stability::TooFewSamples: return "too_few_samples";
    }
    return "?";
}

Stability classify_stability(const SampleStats& s, const StabilityPolicy& policy) noexcept {
    if (s.n < policy.min_samples) return Stability::TooFewSamples;
    if (!(s.cv <= policy.max_cv)) return Stability::Unstable;  // also catches NaN/inf
    return Stability::Stable;
}

void to_json(nlohmann::json& j, const SampleStats& v) {
    j = {{"n", v.n},       {"min", v.min},    {"max", v.max},       {"mean", v.mean},
         {"median", v.median}, {"stddev", v.stddev}, {"p50", v.p50}, {"p95", v.p95},
         {"p99", v.p99},   {"cv", std::isfinite(v.cv) ? nlohmann::json(v.cv) : nlohmann::json(nullptr)},
         {"percentile_method", "linear (Hyndman-Fan type 7)"}};
}

void from_json(const nlohmann::json& j, SampleStats& v) {
    v.n = j.at("n").get<std::size_t>();
    v.min = j.at("min").get<double>();
    v.max = j.at("max").get<double>();
    v.mean = j.at("mean").get<double>();
    v.median = j.at("median").get<double>();
    v.stddev = j.at("stddev").get<double>();
    v.p50 = j.at("p50").get<double>();
    v.p95 = j.at("p95").get<double>();
    v.p99 = j.at("p99").get<double>();
    v.cv = j.at("cv").is_null() ? std::numeric_limits<double>::infinity() : j.at("cv").get<double>();
}

}  // namespace halo::profiling
