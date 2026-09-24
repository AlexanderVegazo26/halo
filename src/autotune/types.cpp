#include "halo/autotune/types.h"

#include <charconv>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"

namespace halo::autotune {

namespace {

bool valid_name(std::string_view n) noexcept {
    if (n.empty() || n.size() > 32) return false;
    for (const char c : n) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    }
    return true;
}

}  // namespace

std::string Candidate::to_string() const {
    std::string out;
    for (const auto& [k, v] : params) {
        HALO_CHECK(valid_name(k), ErrorCode::Config, "candidate: invalid parameter name '{}'", k);
        if (!out.empty()) out += ';';
        out += k;
        out += '=';
        out += std::to_string(v);
    }
    return out;
}

Candidate Candidate::parse(std::string_view text) {
    Candidate c;
    HALO_CHECK(text.size() <= 4096, ErrorCode::Config, "candidate: text of {} bytes is too long", text.size());
    if (text.empty()) return c;
    std::size_t pos = 0;
    while (true) {
        const std::size_t semi = text.find(';', pos);
        const std::size_t end = semi == std::string_view::npos ? text.size() : semi;
        const std::string_view item = text.substr(pos, end - pos);
        const std::size_t eq = item.find('=');
        HALO_CHECK(eq != std::string_view::npos, ErrorCode::Config, "candidate: item '{}' in '{}' has no '='", item,
                   text);
        const std::string_view name = item.substr(0, eq);
        const std::string_view val = item.substr(eq + 1);
        HALO_CHECK(valid_name(name), ErrorCode::Config, "candidate: invalid parameter name '{}'", name);
        std::int64_t v = 0;
        const auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), v);
        HALO_CHECK(!val.empty() && ec == std::errc() && ptr == val.data() + val.size(), ErrorCode::Config,
                   "candidate: value '{}' of '{}' is not an integer", val, name);
        HALO_CHECK(c.params.emplace(std::string(name), v).second, ErrorCode::Config,
                   "candidate: duplicate parameter '{}'", name);
        if (semi == std::string_view::npos) break;
        pos = semi + 1;
    }
    return c;
}

std::int64_t Candidate::get(std::string_view name) const {
    const auto it = params.find(std::string(name));
    HALO_CHECK(it != params.end(), ErrorCode::Config, "candidate {} has no parameter '{}'", to_string(), name);
    return it->second;
}

std::string_view to_string(Strategy s) noexcept {
    switch (s) {
        case Strategy::Exhaustive: return "exhaustive";
        case Strategy::Random: return "random";
        case Strategy::Grid: return "grid";
        case Strategy::Heuristic: return "heuristic";
        case Strategy::Bayesian: return "bayesian";
    }
    return "?";
}

void to_json(nlohmann::json& j, const CandidateResult& v) {
    j = {{"candidate", v.candidate.to_string()},
         {"predicted", v.predicted ? nlohmann::json(*v.predicted) : nlohmann::json(nullptr)},
         {"measured", v.measured},
         {"stats", v.measured ? nlohmann::json(v.stats) : nlohmann::json(nullptr)},
         {"stability", v.measured ? nlohmann::json(profiling::to_string(v.stability)) : nlohmann::json(nullptr)},
         {"rejected", v.rejected}};
}

void to_json(nlohmann::json& j, const OpTuneResult& v) {
    j = {{"family", v.key.family},
         {"shape", v.key.shape},
         {"backend", v.backend},
         {"strategy", to_string(v.strategy)},
         {"candidates", v.candidates},
         {"winner", v.winner ? nlohmann::json(v.winner->to_string()) : nlohmann::json(nullptr)},
         {"winner_metric", kWinnerMetric},
         {"winner_median_ns", v.winner ? nlohmann::json(v.winner_median_ns) : nlohmann::json(nullptr)},
         {"persisted", v.persisted},
         {"note", v.note}};
}

}  // namespace halo::autotune
