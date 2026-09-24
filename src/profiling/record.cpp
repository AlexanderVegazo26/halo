#include "halo/profiling/record.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <functional>
#include <tuple>

#include "halo/core/error.h"

namespace halo::profiling {

namespace {

constexpr double kGB = 1e9;

template <class T>
nlohmann::json opt(const std::optional<T>& v) {
    return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
}

template <class T>
void read_opt(const nlohmann::json& j, const char* key, std::optional<T>& out) {
    if (j.contains(key) && !j.at(key).is_null()) {
        out = j.at(key).get<T>();
    } else {
        out.reset();
    }
}

/// Strict unsigned read: rejects negatives, floats and values above `max` (nlohmann's
/// get<unsigned> would silently wrap -1 to 2^64-1).
std::uint64_t read_uint(const nlohmann::json& j, const char* key, std::uint64_t max) {
    const auto& v = j.at(key);
    std::optional<std::uint64_t> out;
    if (v.is_number_unsigned()) out = v.get<std::uint64_t>();
    else if (v.is_number_integer() && v.get<std::int64_t>() >= 0) out = static_cast<std::uint64_t>(v.get<std::int64_t>());
    HALO_CHECK(out && *out <= max, ErrorCode::Config, "benchmark record: '{}' must be an integer in [0, {}]", key, max);
    return *out;
}

void check_weight(double v, const char* name) {
    HALO_CHECK(std::isfinite(v) && v >= 0.0, ErrorCode::Config, "DecodeByteModel: {} = {} is invalid", name, v);
}

}  // namespace

double DecodeByteModel::bytes_per_step() const {
    HALO_CHECK(sequences > 0, ErrorCode::Config, "DecodeByteModel: sequences == 0");
    check_weight(w_trunk_gb, "w_trunk_gb");
    check_weight(w_mtp_gb, "w_mtp_gb");
    check_weight(w_head_gb, "w_head_gb");
    check_weight(state_copy_gb, "state_copy_gb");
    const double n = draft_tokens;
    const double k = state_copies ? *state_copies : n + 1.0;
    const double s = sequences;
    const double weights = (w_trunk_gb + (n + 1.0) * w_mtp_gb + n * w_head_gb) * kGB;
    const double per_seq = (1.0 + k) * state_copy_gb * kGB +
                           static_cast<double>(context_tokens) * static_cast<double>(kv_bytes_per_token);
    return weights + s * per_seq;
}

double DecodeByteModel::bytes_per_token(double tokens_per_step) const {
    HALO_CHECK(std::isfinite(tokens_per_step) && tokens_per_step >= 1.0, ErrorCode::Config,
               "DecodeByteModel: tokens_per_step {} < 1", tokens_per_step);
    return bytes_per_step() / (static_cast<double>(sequences) * tokens_per_step);
}

double DecodeByteModel::ceiling_tps(double bandwidth_gbps, double efficiency, double tokens_per_step) const {
    HALO_CHECK(std::isfinite(bandwidth_gbps) && bandwidth_gbps > 0.0, ErrorCode::Config,
               "DecodeByteModel: bandwidth {} GB/s is invalid", bandwidth_gbps);
    return efficiency * bandwidth_gbps * kGB / bytes_per_token(tokens_per_step);
}

double decode_efficiency(double measured_tps, double bytes_per_token, double bandwidth_gbps) {
    HALO_CHECK(std::isfinite(bandwidth_gbps) && bandwidth_gbps > 0.0, ErrorCode::Config,
               "decode_efficiency: bandwidth {} GB/s is invalid", bandwidth_gbps);
    return measured_tps * bytes_per_token / (bandwidth_gbps * kGB);
}

std::string_view to_string(Phase p) noexcept {
    return p == Phase::Cold ? "cold" : "steady";
}

void apply_efficiency(BenchmarkRecord& r, const DecodeByteModel& model, double bandwidth_gbps,
                      double tokens_per_step) {
    r.measured_bandwidth_gbps = bandwidth_gbps;
    r.predicted_bytes_per_token = model.bytes_per_token(tokens_per_step);
    const std::optional<double> tps = r.decode_effective_tps ? r.decode_effective_tps : r.decode_tps;
    if (tps) r.efficiency = decode_efficiency(*tps, *r.predicted_bytes_per_token, bandwidth_gbps);
}

void to_json(nlohmann::json& j, const Invocation& v) {
    j = {{"binary", v.binary}, {"version", v.version}, {"commit", v.commit}, {"argv", v.argv},
         {"environment", v.environment}};
}

void from_json(const nlohmann::json& j, Invocation& v) {
    v.binary = j.value("binary", "");
    v.version = j.value("version", "");
    v.commit = j.value("commit", "");
    v.argv = j.value("argv", std::vector<std::string>{});
    v.environment = j.value("environment", std::map<std::string, std::string>{});
}

void to_json(nlohmann::json& j, const BenchmarkRecord& v) {
    j = {{"schema", kRecordSchema},
         {"model", v.model},
         {"model_hash", v.model_hash},
         {"pack", v.pack},
         {"backend", v.backend},
         {"driver", v.driver},
         {"gpu", v.gpu},
         {"quantization", v.quantization},
         {"lm_head", v.lm_head},
         {"context", v.context},
         {"batch", v.batch},
         {"concurrency", v.concurrency},
         {"ttft_ms", opt(v.ttft_ms)},
         {"ttft_cached_ms", opt(v.ttft_cached_ms)},
         {"prompt_tps", opt(v.prompt_tps)},
         {"decode_tps", opt(v.decode_tps)},
         {"decode_effective_tps", opt(v.decode_effective_tps)},
         {"mtp_acceptance", opt(v.mtp_acceptance)},
         {"prefix_cache_hit_rate", opt(v.prefix_cache_hit_rate)},
         {"memory_by_tier_gb", v.memory_by_tier_gb},
         {"load_time_s", opt(v.load_time_s)},
         {"temperature_c", opt(v.temperature_c)},
         {"clocks_mhz", v.clocks_mhz},
         {"power_mode", v.power_mode},
         {"suite", v.suite},
         {"mode", v.mode},
         {"engine", v.engine},
         {"pack_hash", v.pack_hash},
         {"mtp_hash", v.mtp_hash},
         {"repetition", v.repetition},
         {"phase", to_string(v.phase)},
         {"host_label", v.host_label},
         {"measured_bandwidth_gbps", opt(v.measured_bandwidth_gbps)},
         {"predicted_bytes_per_token", opt(v.predicted_bytes_per_token)},
         {"efficiency", opt(v.efficiency)},
         {"hardware_before", v.hardware_before ? nlohmann::json(*v.hardware_before) : nlohmann::json(nullptr)},
         {"hardware_after", v.hardware_after ? nlohmann::json(*v.hardware_after) : nlohmann::json(nullptr)},
         {"invocation", v.invocation ? nlohmann::json(*v.invocation) : nlohmann::json(nullptr)},
         {"extra", v.extra}};
}

void from_json(const nlohmann::json& j, BenchmarkRecord& v) {
    try {
        HALO_CHECK(j.is_object(), ErrorCode::Config, "benchmark record is not a JSON object");
        const std::string schema = j.value("schema", "");
        HALO_CHECK(schema == kRecordSchema, ErrorCode::Config, "benchmark record schema '{}' is not '{}'", schema,
                   kRecordSchema);
        v.model = j.at("model").get<std::string>();
        v.model_hash = j.at("model_hash").get<std::string>();
        v.pack = j.at("pack").get<std::string>();
        v.backend = j.at("backend").get<std::string>();
        v.driver = j.at("driver").get<std::string>();
        v.gpu = j.at("gpu").get<std::string>();
        v.quantization = j.at("quantization").get<std::string>();
        v.lm_head = j.at("lm_head").get<std::string>();
        constexpr std::uint64_t kU32 = 0xffffffffULL;
        v.context = read_uint(j, "context", ~0ULL);
        v.batch = static_cast<std::uint32_t>(read_uint(j, "batch", kU32));
        v.concurrency = static_cast<std::uint32_t>(read_uint(j, "concurrency", kU32));
        read_opt(j, "ttft_ms", v.ttft_ms);
        read_opt(j, "ttft_cached_ms", v.ttft_cached_ms);
        read_opt(j, "prompt_tps", v.prompt_tps);
        read_opt(j, "decode_tps", v.decode_tps);
        read_opt(j, "decode_effective_tps", v.decode_effective_tps);
        read_opt(j, "mtp_acceptance", v.mtp_acceptance);
        read_opt(j, "prefix_cache_hit_rate", v.prefix_cache_hit_rate);
        v.memory_by_tier_gb = j.value("memory_by_tier_gb", std::map<std::string, double>{});
        read_opt(j, "load_time_s", v.load_time_s);
        read_opt(j, "temperature_c", v.temperature_c);
        v.clocks_mhz = j.value("clocks_mhz", std::map<std::string, std::uint32_t>{});
        v.power_mode = j.at("power_mode").get<std::string>();
        v.suite = j.value("suite", "");
        v.mode = j.value("mode", "");
        v.engine = j.value("engine", "");
        v.pack_hash = j.value("pack_hash", "");
        v.mtp_hash = j.value("mtp_hash", "");
        v.repetition = j.contains("repetition") ? static_cast<std::uint32_t>(read_uint(j, "repetition", kU32)) : 0U;
        const std::string phase = j.value("phase", "steady");
        HALO_CHECK(phase == "cold" || phase == "steady", ErrorCode::Config,
                   "benchmark record: phase '{}' is not cold/steady", phase);
        v.phase = phase == "cold" ? Phase::Cold : Phase::Steady;
        v.host_label = j.value("host_label", "");
        read_opt(j, "measured_bandwidth_gbps", v.measured_bandwidth_gbps);
        read_opt(j, "predicted_bytes_per_token", v.predicted_bytes_per_token);
        read_opt(j, "efficiency", v.efficiency);
        read_opt(j, "hardware_before", v.hardware_before);
        read_opt(j, "hardware_after", v.hardware_after);
        read_opt(j, "invocation", v.invocation);
        v.extra = j.value("extra", nlohmann::json::object());
    } catch (const nlohmann::json::exception& e) {
        throw_error(ErrorCode::Config, "malformed benchmark record: {}", e.what());
    }
}

// ---- roll-up ----------------------------------------------------------------------------

namespace {

using GroupKey = std::tuple<std::string, std::string, std::string, std::uint64_t, std::uint32_t, int>;

GroupKey key_of(const BenchmarkRecord& r) {
    return {r.suite, r.mode, r.engine, r.context, r.concurrency, static_cast<int>(r.phase)};
}

using Getter = std::function<std::optional<double>(const BenchmarkRecord&)>;

const std::vector<std::pair<std::string, Getter>>& metric_getters() {
    static const std::vector<std::pair<std::string, Getter>> g = {
        {"ttft_ms", [](const BenchmarkRecord& r) { return r.ttft_ms; }},
        {"ttft_cached_ms", [](const BenchmarkRecord& r) { return r.ttft_cached_ms; }},
        {"prompt_tps", [](const BenchmarkRecord& r) { return r.prompt_tps; }},
        {"decode_tps", [](const BenchmarkRecord& r) { return r.decode_tps; }},
        {"decode_effective_tps", [](const BenchmarkRecord& r) { return r.decode_effective_tps; }},
        {"mtp_acceptance", [](const BenchmarkRecord& r) { return r.mtp_acceptance; }},
        {"prefix_cache_hit_rate", [](const BenchmarkRecord& r) { return r.prefix_cache_hit_rate; }},
        {"load_time_s", [](const BenchmarkRecord& r) { return r.load_time_s; }},
        {"efficiency", [](const BenchmarkRecord& r) { return r.efficiency; }},
    };
    return g;
}

}  // namespace

std::vector<RunSummary> summarize(const std::vector<BenchmarkRecord>& records, const StabilityPolicy& policy) {
    std::map<GroupKey, std::vector<const BenchmarkRecord*>> groups;
    std::map<GroupKey, std::uint32_t> warm;  // keyed by the *steady* key
    for (const auto& r : records) {
        groups[key_of(r)].push_back(&r);
        if (r.phase == Phase::Cold) {
            GroupKey k = key_of(r);
            std::get<5>(k) = static_cast<int>(Phase::Steady);
            ++warm[k];
        }
    }

    std::vector<RunSummary> out;
    for (const auto& [key, recs] : groups) {
        RunSummary s;
        s.suite = std::get<0>(key);
        s.mode = std::get<1>(key);
        s.engine = std::get<2>(key);
        s.context = std::get<3>(key);
        s.concurrency = std::get<4>(key);
        s.phase = static_cast<Phase>(std::get<5>(key));
        const auto w = warm.find(key);
        s.warmup_count = (s.phase == Phase::Steady && w != warm.end()) ? w->second : 0;
        s.measured_count = static_cast<std::uint32_t>(recs.size());
        s.required_measured = s.suite == "micro" ? kMicroMinMeasured : kEndToEndMinRepetitions;
        s.conformant = s.measured_count >= s.required_measured &&
                       (s.suite != "micro" || s.warmup_count >= kMicroMinWarmup);
        StabilityPolicy p = policy;
        p.min_samples = s.required_measured;
        for (const auto& [name, get] : metric_getters()) {
            std::vector<double> xs;
            for (const auto* r : recs) {
                if (const auto v = get(*r)) xs.push_back(*v);
            }
            if (xs.empty()) continue;
            MetricSummary m;
            m.stats = compute_sample_stats(xs);
            m.stability = classify_stability(m.stats, p);
            s.metrics.emplace(name, m);
        }
        std::map<std::string, std::vector<double>> extras;
        for (const auto* r : recs) {
            if (!r->extra.is_object()) continue;
            for (const auto& [k, val] : r->extra.items()) {
                if (val.is_number()) extras["extra." + k].push_back(val.get<double>());
            }
        }
        for (const auto& [name, xs] : extras) {
            MetricSummary m;
            m.stats = compute_sample_stats(xs);
            m.stability = classify_stability(m.stats, p);
            s.metrics.emplace(name, m);
        }
        out.push_back(std::move(s));
    }
    return out;
}

void to_json(nlohmann::json& j, const RunSummary& v) {
    nlohmann::json metrics = nlohmann::json::object();
    for (const auto& [name, m] : v.metrics) {
        metrics[name] = {{"stats", m.stats}, {"stability", to_string(m.stability)}};
    }
    j = {{"schema", kSummarySchema},
         {"suite", v.suite},
         {"mode", v.mode},
         {"engine", v.engine},
         {"context", v.context},
         {"concurrency", v.concurrency},
         {"phase", to_string(v.phase)},
         {"warmup_count", v.warmup_count},
         {"measured_count", v.measured_count},
         {"required_measured", v.required_measured},
         {"conformant", v.conformant},
         {"metrics", metrics}};
}

}  // namespace halo::profiling
