#include "halo/autotune/tuner.h"

#include <algorithm>
#include <format>
#include <set>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"
#include "halo/core/log.h"

namespace halo::autotune {

namespace {

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint64_t mulhi64(std::uint64_t a, std::uint64_t b) noexcept {
    const std::uint64_t a_lo = a & 0xffffffffULL, a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xffffffffULL, b_hi = b >> 32;
    const std::uint64_t lo_lo = a_lo * b_lo;
    const std::uint64_t hi_lo = a_hi * b_lo;
    const std::uint64_t lo_hi = a_lo * b_hi;
    const std::uint64_t hi_hi = a_hi * b_hi;
    const std::uint64_t cross = (lo_lo >> 32) + (hi_lo & 0xffffffffULL) + lo_hi;
    return hi_hi + (hi_lo >> 32) + (cross >> 32);
}

std::vector<Candidate> plan_random(const std::vector<Candidate>& all, const TuneOptions& o) {
    HALO_CHECK(o.random_budget >= 1, ErrorCode::Config, "tune: random_budget must be >= 1");
    std::vector<Candidate> pool = all;
    const std::size_t k = std::min(o.random_budget, pool.size());
    std::uint64_t state = o.seed;
    for (std::size_t i = 0; i < k; ++i) {
        // j uniform-ish in [i, n): high 64 bits of r * (n - i).
        const std::size_t j = i + static_cast<std::size_t>(mulhi64(splitmix64(state), pool.size() - i));
        std::swap(pool[i], pool[j]);
    }
    pool.resize(k);
    return pool;
}

std::vector<Candidate> plan_grid(const std::vector<Candidate>& all, const TuneOptions& o) {
    HALO_CHECK(o.grid_stride >= 1, ErrorCode::Config, "tune: grid_stride must be >= 1");
    std::map<std::string, std::set<std::int64_t>> values;
    for (const auto& c : all) {
        for (const auto& [k, v] : c.params) values[k].insert(v);
    }
    std::map<std::string, std::set<std::int64_t>> kept;
    for (const auto& [name, vs] : values) {
        const std::vector<std::int64_t> sorted(vs.begin(), vs.end());
        for (std::size_t i = 0; i < sorted.size(); i += o.grid_stride) kept[name].insert(sorted[i]);
        kept[name].insert(sorted.back());
    }
    std::vector<Candidate> out;
    for (const auto& c : all) {
        bool on_grid = true;
        for (const auto& [k, v] : c.params) on_grid = on_grid && kept[k].contains(v);
        if (on_grid) out.push_back(c);
    }
    return out;
}

}  // namespace

std::vector<Candidate> plan_candidates(const std::vector<Candidate>& all, const TuneOptions& o,
                                       const std::function<std::optional<CostInputs>(const Candidate&)>& cost) {
    switch (o.strategy) {
        case Strategy::Exhaustive: return all;
        case Strategy::Random: return plan_random(all, o);
        case Strategy::Grid: return plan_grid(all, o);
        case Strategy::Bayesian:
            throw_error(ErrorCode::Unsupported, "tune: BAYESIAN strategy is not implemented in v1 (TRD §58: future)");
        case Strategy::Heuristic: {
            HALO_CHECK(o.cost_model.has_value(), ErrorCode::Config, "tune: HEURISTIC needs a cost model");
            HALO_CHECK(o.heuristic_keep >= 1, ErrorCode::Config, "tune: heuristic_keep must be >= 1");
            HALO_CHECK(static_cast<bool>(cost), ErrorCode::Config, "tune: HEURISTIC needs cost inputs");
            std::vector<std::pair<double, std::size_t>> ranked;
            for (std::size_t i = 0; i < all.size(); ++i) {
                const auto in = cost(all[i]);
                HALO_CHECK(in.has_value(), ErrorCode::Config, "tune: candidate {} has no cost inputs",
                           all[i].to_string());
                ranked.emplace_back(o.cost_model->estimate(*in).total_ns, i);
            }
            std::sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return all[a.second].to_string() < all[b.second].to_string();
            });
            std::vector<Candidate> out;
            for (std::size_t i = 0; i < std::min(o.heuristic_keep, ranked.size()); ++i) {
                out.push_back(all[ranked[i].second]);
            }
            return out;
        }
    }
    throw_error(ErrorCode::Config, "tune: unknown strategy");
}

namespace {

OpTuneResult tune_one(TunableOp& op, const TuneOptions& o, const profiling::Clock& clock) {
    OpTuneResult r;
    r.key = op.key();
    r.backend = op.backend();
    r.strategy = o.strategy;
    const std::vector<Candidate> all = op.candidates();
    if (all.empty()) {
        r.note = "operator offered no candidates";
        return r;
    }
    const auto cost = [&op](const Candidate& c) { return op.cost(c); };
    const std::vector<Candidate> plan = plan_candidates(all, o, cost);
    profiling::StabilityPolicy policy = o.stability;
    policy.min_samples = std::min<std::size_t>(policy.min_samples, o.iterations);
    for (const auto& c : plan) {
        CandidateResult cr;
        cr.candidate = c;
        if (const auto in = op.cost(c); in && o.cost_model) cr.predicted = o.cost_model->estimate(*in);
        std::string why;
        if (!op.validate(c, why)) {
            cr.rejected = "invalid: " + (why.empty() ? std::string("validation failed") : why);
            r.candidates.push_back(std::move(cr));
            continue;
        }
        for (unsigned i = 0; i < o.warmup; ++i) (void)op.run(c);
        std::vector<double> ns;
        ns.reserve(o.iterations);
        for (unsigned i = 0; i < o.iterations; ++i) {
            const std::int64_t t0 = clock.now_ns();
            const Measurement m = op.run(c);
            const std::int64_t t1 = clock.now_ns();
            ns.push_back(static_cast<double>(m.device_ns.value_or(t1 - t0)));
        }
        cr.measured = true;
        cr.stats = profiling::compute_sample_stats(ns);
        cr.stability = profiling::classify_stability(cr.stats, policy);
        if (cr.stability != profiling::Stability::Stable) {
            cr.rejected = std::format("{} (cv {:.4f}, max {:.4f})", profiling::to_string(cr.stability), cr.stats.cv,
                                      policy.max_cv);
        }
        r.candidates.push_back(std::move(cr));
    }
    const CandidateResult* best = nullptr;
    for (const auto& cr : r.candidates) {
        if (!cr.measured || !cr.rejected.empty()) continue;
        if (best == nullptr || cr.stats.median < best->stats.median ||
            (cr.stats.median == best->stats.median && cr.candidate.to_string() < best->candidate.to_string())) {
            best = &cr;
        }
    }
    if (best != nullptr) {
        r.winner = best->candidate;
        r.winner_median_ns = best->stats.median;
    } else {
        r.note = "no eligible (stable and valid) candidate; existing winner left unchanged";
    }
    return r;
}

}  // namespace

TuneReport tune(ProfileDb& db, const ProfileKey& key, std::span<TunableOp* const> ops, const TuneOptions& o,
                const profiling::Clock& clock) {
    if (o.strategy == Strategy::Bayesian) {
        throw_error(ErrorCode::Unsupported, "tune: BAYESIAN strategy is not implemented in v1 (TRD §58: future)");
    }
    HALO_CHECK(o.allow_nonconformant ||
                   (o.warmup >= profiling::kMicroMinWarmup && o.iterations >= profiling::kMicroMinMeasured),
               ErrorCode::Config, "tune: {} warm-up + {} measured is below the TRD §50 minimum ({} + {})", o.warmup,
               o.iterations, profiling::kMicroMinWarmup, profiling::kMicroMinMeasured);
    HALO_CHECK(o.iterations >= 1, ErrorCode::Config, "tune: at least one measured iteration is required");
    TuneReport report;
    report.key = key;
    for (TunableOp* op : ops) {
        HALO_CHECK(op != nullptr, ErrorCode::Config, "tune: null TunableOp");
        OpTuneResult r = tune_one(*op, o, clock);
        r.persisted = db.persist_tune(key, r);
        HALO_INFO("autotune", "{} {} [{}]: winner {} ({})", r.key.family, r.key.shape, r.backend,
                  r.winner ? r.winner->to_string() : std::string("none"), r.note);
        report.ops.push_back(std::move(r));
    }
    return report;
}

void to_json(nlohmann::json& j, const TuneReport& r) {
    j = {{"schema", "halo.tune.report/1"}, {"key", r.key}, {"ops", r.ops}};
}

}  // namespace halo::autotune
