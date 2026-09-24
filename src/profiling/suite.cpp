#include "halo/profiling/suite.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <format>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"
#include "halo/core/log.h"

namespace halo::profiling {

namespace {

// ---- common plumbing --------------------------------------------------------------------

struct Timed {
    runtime::GenerateResult result;
    std::int64_t t_start = 0;
    std::int64_t t_first = -1;
    std::int64_t t_last = -1;
    std::int64_t t_cancel = -1;  ///< when the callback returned false
    std::int64_t t_end = 0;
    std::size_t tokens = 0;      ///< callbacks observed
    std::string error;           ///< non-empty when generate() threw
};

/// Runs generate() and timestamps the callbacks. `cancel_after` > 0 makes the callback
/// return false once that many tokens have been observed.
Timed timed_generate(runtime::Engine& engine, const runtime::GenerateRequest& req, const Clock& clock,
                     std::size_t cancel_after = 0) {
    Timed t;
    t.t_start = clock.now_ns();
    const runtime::TokenCallback cb = [&](const runtime::TokenEvent&) {
        const std::int64_t now = clock.now_ns();
        if (t.t_first < 0) t.t_first = now;
        t.t_last = now;
        ++t.tokens;
        if (cancel_after > 0 && t.tokens >= cancel_after) {
            t.t_cancel = clock.now_ns();
            return false;
        }
        return true;
    };
    try {
        t.result = engine.generate(req, cb);
    } catch (const std::exception& e) {
        t.error = e.what();
    }
    t.t_end = clock.now_ns();
    return t;
}

double ns_to_ms(std::int64_t ns) { return static_cast<double>(ns) * 1e-6; }

std::optional<double> ttft_ms(const Timed& t) {
    if (t.t_first < 0) return std::nullopt;
    return ns_to_ms(t.t_first - t.t_start);
}

std::optional<double> prompt_tps(const Timed& t, std::size_t prompt_tokens) {
    if (t.t_first < 0 || t.t_first <= t.t_start || prompt_tokens == 0) return std::nullopt;
    return static_cast<double>(prompt_tokens) * 1e9 / static_cast<double>(t.t_first - t.t_start);
}

std::optional<double> decode_tps(const Timed& t) {
    if (t.tokens < 2 || t.t_last <= t.t_first) return std::nullopt;
    return static_cast<double>(t.tokens - 1) * 1e9 / static_cast<double>(t.t_last - t.t_first);
}

void put_engine_self_report(BenchmarkRecord& r, const runtime::GenerateResult& g) {
    r.extra["engine_ttft_ms"] = g.ttft_ms;
    r.extra["engine_decode_tps"] = g.decode_tps;
    r.extra["engine_prompt_tokens"] = g.prompt_tokens;
    r.extra["engine_cached_prompt_tokens"] = g.cached_prompt_tokens;
    r.extra["generated_tokens"] = g.tokens.size();
}

BenchmarkRecord base_record(const RecordIdentity& id, std::string suite, std::string mode) {
    BenchmarkRecord r;
    r.model = id.model;
    r.model_hash = id.model_hash;
    r.pack = id.pack;
    r.pack_hash = id.pack_hash;
    r.mtp_hash = id.mtp_hash;
    r.backend = id.backend;
    r.driver = id.driver;
    r.gpu = id.gpu;
    r.quantization = id.quantization;
    r.lm_head = id.lm_head;
    r.engine = id.engine;
    r.host_label = id.host_label;
    r.suite = std::move(suite);
    r.mode = std::move(mode);
    return r;
}

class ArtifactBuilder {
public:
    ArtifactBuilder(std::string suite, const EnvironmentOptions& env, const RecordIdentity& id,
                    const StabilityPolicy& stability)
        : env_(env), stability_(stability) {
        HALO_CHECK(!id.host_label.empty(), ErrorCode::Config,
                   "suite {}: identity.host_label is required (D-001: every record says where it ran)", suite);
        art_.suite = std::move(suite);
        if (env_.enabled) {
            // Project immediately: make_hardware_state stamps captured_at from the wall
            // clock, so deferring it to finish() would give "before" the finish time.
            before_ = capture_hardware_state(env_.discovery, env_.platform_power_mode, stamp());
        }
    }

    SuiteArtifact& art() { return art_; }

    /// Hardware fields (power_mode, temperature, clocks, snapshots) are stamped in finish().
    void add(BenchmarkRecord r) { art_.records.push_back(std::move(r)); }

    void note(std::string s) {
        HALO_INFO("bench", "{}", s);
        art_.notes.push_back(std::move(s));
    }

    void failure(std::string s) {
        HALO_ERROR("bench", "{}", s);
        art_.notes.push_back("FAILED: " + s);
        failed_ = true;
    }

    SuiteArtifact finish() {
        if (env_.enabled && before_) {
            HardwareState before = std::move(*before_);
            HardwareState after = capture_hardware_state(env_.discovery, env_.platform_power_mode, stamp());
            if (env_.patch_states) env_.patch_states(before, after);
            art_.thermal = check_thermal_drift(before, after, env_.thermal_drift_threshold_c);
            art_.power_mode_pinned = check_comparable(before, after);
            for (auto& r : art_.records) {
                r.power_mode = before.power_mode;
                r.temperature_c = before.temperature_c;
                r.clocks_mhz = before.clocks_mhz;
                if (r.gpu.empty() && before.gpu_arch) r.gpu = *before.gpu_arch;
                r.hardware_before = before;
                r.hardware_after = after;
            }
            art_.hardware_before = std::move(before);
            art_.hardware_after = std::move(after);
        } else {
            art_.thermal.reason = "hardware state not captured";
            art_.power_mode_pinned = {false, "hardware state not captured"};
            for (auto& r : art_.records) r.power_mode = "unknown|unknown/unknown";
        }
        art_.summaries = summarize(art_.records, stability_);
        art_.conformant = !art_.summaries.empty();
        for (const auto& s : art_.summaries) {
            if (s.phase == Phase::Steady && !s.conformant) art_.conformant = false;
        }
        if (failed_) art_.conformant = false;
        art_.valid = art_.thermal.valid && art_.power_mode_pinned.comparable && art_.conformant;
        return std::move(art_);
    }

private:
    std::string stamp() const { return env_.utc_now ? env_.utc_now() : std::string(); }

    const EnvironmentOptions& env_;
    StabilityPolicy stability_;
    std::optional<HardwareState> before_;
    SuiteArtifact art_;
    bool failed_ = false;
};

void apply_eff(BenchmarkRecord& r, const std::optional<EfficiencyInputs>& eff, std::uint32_t sequences,
               std::uint64_t context, double tokens_per_step = 1.0, bool speculative = false) {
    if (!eff || eff->measured_bandwidth_gbps <= 0) return;
    DecodeByteModel m = eff->byte_model;
    m.sequences = std::max<std::uint32_t>(1, sequences);
    m.context_tokens = context;
    if (!speculative) {
        m.draft_tokens = 0;
        m.w_mtp_gb = 0;
        m.state_copies.reset();
    }
    apply_efficiency(r, m, eff->measured_bandwidth_gbps, tokens_per_step);
}

void check_counts(bool allow, unsigned warmup, unsigned measured, unsigned min_warmup, unsigned min_measured,
                  std::string_view suite) {
    HALO_CHECK(allow || (warmup >= min_warmup && measured >= min_measured), ErrorCode::Config,
               "{} suite: {} warm-up + {} measured is below the TRD §50 minimum ({} + {}); set allow_nonconformant "
               "only for tests",
               suite, warmup, measured, min_warmup, min_measured);
    HALO_CHECK(measured >= 1, ErrorCode::Config, "{} suite: at least one measured run is required", suite);
}

}  // namespace

std::string_view to_string(ModelMode m) noexcept {
    switch (m) {
        case ModelMode::LoadTime: return "load_time";
        case ModelMode::PromptProcessing: return "prompt";
        case ModelMode::Decode: return "decode";
        case ModelMode::MtpDecode: return "mtp_decode";
        case ModelMode::Concurrency: return "concurrency";
        case ModelMode::PrefixReplay: return "prefix_replay";
        case ModelMode::CancellationStorm: return "cancellation_storm";
    }
    return "?";
}

void to_json(nlohmann::json& j, const SuiteArtifact& v) {
    j = {{"schema", "halo.bench.artifact/1"},
         {"suite", v.suite},
         {"hardware_before", v.hardware_before ? nlohmann::json(*v.hardware_before) : nlohmann::json(nullptr)},
         {"hardware_after", v.hardware_after ? nlohmann::json(*v.hardware_after) : nlohmann::json(nullptr)},
         {"thermal", {{"valid", v.thermal.valid},
                      {"drift_c", v.thermal.drift_c ? nlohmann::json(*v.thermal.drift_c) : nlohmann::json(nullptr)},
                      {"reason", v.thermal.reason}}},
         {"power_mode_pinned", {{"comparable", v.power_mode_pinned.comparable}, {"reason", v.power_mode_pinned.reason}}},
         {"conformant", v.conformant},
         {"valid", v.valid},
         {"notes", v.notes},
         {"records", v.records},
         {"summaries", v.summaries}};
}

// ---- micro ------------------------------------------------------------------------------

SuiteArtifact run_micro_suite(const MicroSuiteConfig& c, const Clock& clock) {
    check_counts(c.allow_nonconformant, c.warmup, c.iterations, kMicroMinWarmup, kMicroMinMeasured, "micro");
    HALO_CHECK(!c.benchmarks.empty(), ErrorCode::Config, "micro suite: no benchmarks");
    ArtifactBuilder b("micro", c.environment, c.identity, c.stability);
    for (const auto& bench : c.benchmarks) {
        HALO_CHECK(static_cast<bool>(bench.run), ErrorCode::Config, "micro benchmark '{}' has no callback",
                   bench.name);
        const unsigned total = c.warmup + c.iterations;
        for (unsigned i = 0; i < total; ++i) {
            const std::int64_t t0 = clock.now_ns();
            bench.run();
            const std::int64_t ns = clock.now_ns() - t0;
            BenchmarkRecord r = base_record(c.identity, "micro", bench.name);
            r.phase = i < c.warmup ? Phase::Cold : Phase::Steady;
            r.repetition = i < c.warmup ? i : i - c.warmup;
            r.extra["latency_ns"] = ns;
            if (bench.bytes_per_call > 0) {
                r.extra["bytes"] = bench.bytes_per_call;
                if (ns > 0) r.extra["gbps"] = static_cast<double>(bench.bytes_per_call) / static_cast<double>(ns);
            }
            b.add(std::move(r));
        }
    }
    return b.finish();
}

// ---- model ------------------------------------------------------------------------------

namespace {

class ModelRunner {
public:
    ModelRunner(runtime::Engine& e, const ModelSuiteConfig& c, const Clock& clock, ArtifactBuilder& b)
        : e_(e), c_(c), clock_(clock), b_(b) {
        const auto vocab = e.model().vocab_size;
        HALO_CHECK(vocab > 0 && vocab <= 0x7fffffffU, ErrorCode::Config, "model suite: engine vocab_size {} invalid",
                   vocab);
        vocab_ = static_cast<std::int32_t>(vocab);
    }

    void run() {
        for (const ModelMode m : c_.modes) {
            switch (m) {
                case ModelMode::LoadTime: load_time(); break;
                case ModelMode::PromptProcessing: prompt(); break;
                case ModelMode::Decode: decode(false); break;
                case ModelMode::MtpDecode: decode(true); break;
                case ModelMode::Concurrency: concurrency(); break;
                case ModelMode::PrefixReplay: prefix_replay(); break;
                case ModelMode::CancellationStorm: storm(); break;
            }
        }
    }

private:
    std::vector<std::int32_t> tokens(std::size_t n) { return synthetic_tokens(c_.seed + next_seed_++, n, vocab_); }

    unsigned total_runs() const { return c_.warmup + c_.repetitions; }
    Phase phase_of(unsigned i) const { return i < c_.warmup ? Phase::Cold : Phase::Steady; }
    std::uint32_t rep_of(unsigned i) const { return i < c_.warmup ? i : i - c_.warmup; }

    runtime::GenerateRequest request(std::vector<std::int32_t> prompt, std::size_t max_tokens) const {
        runtime::GenerateRequest r;
        r.prompt = std::move(prompt);
        r.max_tokens = max_tokens;
        r.sampling = c_.sampling;
        return r;
    }

    bool fits(std::uint64_t ctx, ModelMode m) {
        const auto limit = e_.model().context_length;
        if (limit != 0 && ctx > limit) {
            b_.note(std::format("{}: context {} skipped: exceeds engine context_length {}", to_string(m), ctx, limit));
            return false;
        }
        return true;
    }

    bool ok(const Timed& t, std::string_view what) {
        if (!t.error.empty()) {
            b_.failure(std::format("{}: generate() threw: {}", what, t.error));
            return false;
        }
        if (t.result.finish == runtime::FinishReason::Error) {
            b_.failure(std::format("{}: engine reported error: {}", what, t.result.error));
            return false;
        }
        return true;
    }

    void load_time() {
        if (!c_.engine_factory) {
            b_.note("load_time skipped: no engine_factory supplied");
            return;
        }
        for (unsigned i = 0; i < total_runs(); ++i) {
            const std::int64_t t0 = clock_.now_ns();
            std::unique_ptr<runtime::Engine> fresh;
            try {
                fresh = c_.engine_factory();
            } catch (const std::exception& ex) {
                b_.failure(std::format("load_time: engine_factory threw: {}", ex.what()));
                return;
            }
            const std::int64_t t1 = clock_.now_ns();
            if (!fresh) {
                b_.failure("load_time: engine_factory returned null");
                return;
            }
            BenchmarkRecord r = base_record(c_.identity, "model", "load_time");
            r.phase = phase_of(i);
            r.repetition = rep_of(i);
            r.load_time_s = static_cast<double>(t1 - t0) * 1e-9;
            b_.add(std::move(r));
        }
    }

    void prompt() {
        if (!fits(c_.prompt_tokens + 1, ModelMode::PromptProcessing)) return;
        for (unsigned i = 0; i < total_runs(); ++i) {
            const Timed t = timed_generate(e_, request(tokens(c_.prompt_tokens), 1), clock_);
            if (!ok(t, "prompt")) return;
            BenchmarkRecord r = base_record(c_.identity, "model", "prompt");
            r.phase = phase_of(i);
            r.repetition = rep_of(i);
            r.context = c_.prompt_tokens;
            r.ttft_ms = ttft_ms(t);
            r.prompt_tps = prompt_tps(t, c_.prompt_tokens);
            put_engine_self_report(r, t.result);
            b_.add(std::move(r));
        }
    }

    void decode(bool mtp) {
        const ModelMode mode = mtp ? ModelMode::MtpDecode : ModelMode::Decode;
        if (mtp && !e_.model().has_mtp) {
            b_.note("mtp_decode skipped: engine model has no MTP block");
            return;
        }
        for (const std::uint64_t ctx : c_.contexts) {
            if (!fits(ctx, mode)) continue;
            if (ctx <= c_.decode_tokens) {
                b_.note(std::format("{}: context {} skipped: not larger than decode_tokens {}", to_string(mode), ctx,
                                    c_.decode_tokens));
                continue;
            }
            const std::size_t prompt_n = ctx - c_.decode_tokens;
            for (unsigned i = 0; i < total_runs(); ++i) {
                const Timed t = timed_generate(e_, request(tokens(prompt_n), c_.decode_tokens), clock_);
                if (!ok(t, to_string(mode))) return;
                BenchmarkRecord r = base_record(c_.identity, "model", std::string(to_string(mode)));
                r.phase = phase_of(i);
                r.repetition = rep_of(i);
                r.context = ctx;
                r.ttft_ms = ttft_ms(t);
                r.prompt_tps = prompt_tps(t, prompt_n);
                const auto tps = decode_tps(t);
                double tau = 1.0;
                bool tau_known = true;
                if (mtp) {
                    r.decode_effective_tps = tps;
                    if (t.result.draft_tokens > 0) {
                        r.mtp_acceptance = static_cast<double>(t.result.accepted_draft_tokens) /
                                           static_cast<double>(t.result.draft_tokens);
                    }
                    r.extra["draft_tokens"] = t.result.draft_tokens;
                    r.extra["accepted_draft_tokens"] = t.result.accepted_draft_tokens;
                    // τ = generated / verification steps, steps = generated - accepted (each
                    // step emits accepted + 1 tokens). Not derivable when the counters are
                    // inconsistent; then η is left null rather than guessed.
                    const std::size_t gen = t.result.tokens.size();
                    const std::size_t acc = t.result.accepted_draft_tokens;
                    if (gen > acc && gen > 0) {
                        tau = static_cast<double>(gen) / static_cast<double>(gen - acc);
                        r.extra["tau_observed"] = tau;
                    } else {
                        tau_known = false;
                        b_.note(std::format("mtp_decode ctx {}: accepted_draft_tokens {} >= generated {}; "
                                            "tau and efficiency not derived", ctx, acc, gen));
                    }
                } else {
                    r.decode_tps = tps;
                    if (t.result.draft_tokens > 0 && !warned_spec_in_decode_) {
                        warned_spec_in_decode_ = true;
                        b_.note("decode: engine proposed draft tokens during plain decode; decode_tps includes "
                                "speculation (PR-004 single decode needs an MTP-disabled engine)");
                    }
                }
                put_engine_self_report(r, t.result);
                if (tau_known) apply_eff(r, c_.efficiency, 1, ctx, tau, mtp);
                b_.add(std::move(r));
            }
        }
    }

    /// Runs `n` requests concurrently; returns their timings (index = request).
    std::vector<Timed> concurrent(const std::vector<runtime::GenerateRequest>& reqs, std::size_t cancel_after) {
        std::vector<Timed> out(reqs.size());
        {
            std::vector<std::jthread> threads;
            threads.reserve(reqs.size());
            for (std::size_t k = 0; k < reqs.size(); ++k) {
                threads.emplace_back([&, k] { out[k] = timed_generate(e_, reqs[k], clock_, cancel_after); });
            }
        }  // join
        return out;
    }

    void concurrency() {
        const std::uint64_t ctx = c_.concurrency_context;
        if (!fits(ctx, ModelMode::Concurrency)) return;
        if (ctx <= c_.decode_tokens) {
            b_.note("concurrency skipped: concurrency_context not larger than decode_tokens");
            return;
        }
        for (const std::uint32_t n : c_.concurrency) {
            if (n == 0) continue;
            for (unsigned i = 0; i < total_runs(); ++i) {
                std::vector<runtime::GenerateRequest> reqs;
                for (std::uint32_t k = 0; k < n; ++k) reqs.push_back(request(tokens(ctx - c_.decode_tokens), c_.decode_tokens));
                const std::int64_t t0 = clock_.now_ns();
                const auto ts = concurrent(reqs, 0);
                const std::int64_t t1 = clock_.now_ns();
                std::size_t generated = 0;
                std::vector<double> ttfts;
                for (const auto& t : ts) {
                    if (!ok(t, "concurrency")) return;
                    generated += t.tokens;
                    if (const auto v = ttft_ms(t)) ttfts.push_back(*v);
                }
                BenchmarkRecord r = base_record(c_.identity, "model", "concurrency");
                r.phase = phase_of(i);
                r.repetition = rep_of(i);
                r.context = ctx;
                r.concurrency = n;
                r.batch = n;
                if (t1 > t0) r.decode_tps = static_cast<double>(generated) * 1e9 / static_cast<double>(t1 - t0);
                if (!ttfts.empty()) {
                    const auto st = compute_sample_stats(ttfts);
                    r.ttft_ms = st.p95;
                    r.extra["ttft_ms_mean"] = st.mean;
                }
                r.extra["generated_tokens"] = generated;
                r.extra["aggregate_includes_prefill"] = true;
                // No efficiency here: this aggregate rate includes prefill, so eta computed from
                // it would be a quietly understated number under the D-014 name.
                if (c_.efficiency && !warned_conc_eff_) {
                    warned_conc_eff_ = true;
                    b_.note("concurrency: efficiency not computed (aggregate decode_tps includes prefill)");
                }
                b_.add(std::move(r));
            }
        }
    }

    void prefix_replay() {
        const std::uint64_t ctx = c_.prefix_replay_context;
        if (!fits(ctx + c_.prefix_replay_suffix + c_.decode_tokens, ModelMode::PrefixReplay)) return;
        for (unsigned i = 0; i < total_runs(); ++i) {
            auto prefix = tokens(ctx);
            const Timed cold = timed_generate(e_, request(prefix, c_.decode_tokens), clock_);
            if (!ok(cold, "prefix_replay (cold)")) return;
            auto replay = prefix;
            const auto suffix = tokens(c_.prefix_replay_suffix);
            replay.insert(replay.end(), suffix.begin(), suffix.end());
            const std::size_t replay_n = replay.size();
            const Timed warm = timed_generate(e_, request(std::move(replay), c_.decode_tokens), clock_);
            if (!ok(warm, "prefix_replay (replay)")) return;
            BenchmarkRecord r = base_record(c_.identity, "model", "prefix_replay");
            r.phase = phase_of(i);
            r.repetition = rep_of(i);
            r.context = ctx;
            r.ttft_ms = ttft_ms(cold);
            r.ttft_cached_ms = ttft_ms(warm);
            if (replay_n > 0) {
                r.prefix_cache_hit_rate =
                    static_cast<double>(warm.result.cached_prompt_tokens) / static_cast<double>(replay_n);
            }
            r.decode_tps = decode_tps(warm);
            r.extra["replay_prompt_tokens"] = replay_n;
            r.extra["engine_cached_prompt_tokens"] = warm.result.cached_prompt_tokens;
            b_.add(std::move(r));
        }
    }

    void storm() {
        if (c_.storm_requests == 0 || c_.storm_cancel_after == 0) {
            b_.note("cancellation_storm skipped: storm_requests or storm_cancel_after is 0");
            return;
        }
        const std::size_t max_tokens = c_.storm_cancel_after + c_.decode_tokens;
        for (unsigned i = 0; i < total_runs(); ++i) {
            std::vector<runtime::GenerateRequest> reqs;
            for (std::uint32_t k = 0; k < c_.storm_requests; ++k) reqs.push_back(request(tokens(c_.prompt_tokens), max_tokens));
            const auto ts = concurrent(reqs, c_.storm_cancel_after);
            std::size_t cancelled = 0;
            std::size_t overshoot = 0;  // tokens delivered after the callback said stop
            std::vector<double> latency_ms;
            for (const auto& t : ts) {
                if (!ok(t, "cancellation_storm")) return;
                if (t.result.finish == runtime::FinishReason::Cancelled) ++cancelled;
                if (t.tokens > c_.storm_cancel_after) overshoot += t.tokens - c_.storm_cancel_after;
                if (t.t_cancel >= 0) latency_ms.push_back(ns_to_ms(t.t_end - t.t_cancel));
            }
            // Probe: the engine must still serve a normal request after the storm.
            const Timed probe = timed_generate(e_, request(tokens(c_.prompt_tokens), 2), clock_);
            const bool probe_ok = probe.error.empty() && probe.result.finish != runtime::FinishReason::Error &&
                                  probe.tokens >= 1;
            BenchmarkRecord r = base_record(c_.identity, "model", "cancellation_storm");
            r.phase = phase_of(i);
            r.repetition = rep_of(i);
            r.context = c_.prompt_tokens;
            r.concurrency = c_.storm_requests;
            r.extra["requests"] = c_.storm_requests;
            r.extra["cancelled"] = cancelled;
            r.extra["not_cancelled"] = c_.storm_requests - cancelled;
            r.extra["tokens_after_cancel"] = overshoot;
            if (!latency_ms.empty()) {
                const auto st = compute_sample_stats(latency_ms);
                r.extra["cancel_latency_ms_max"] = st.max;
                r.extra["cancel_latency_ms_p95"] = st.p95;
            }
            r.extra["probe_ok"] = probe_ok;
            if (!probe_ok) b_.failure("cancellation_storm: engine failed the post-storm probe request");
            if (cancelled != c_.storm_requests) {
                b_.failure(std::format("cancellation_storm: {} of {} requests did not finish as Cancelled",
                                       c_.storm_requests - cancelled, c_.storm_requests));
            }
            b_.add(std::move(r));
        }
    }

    runtime::Engine& e_;
    const ModelSuiteConfig& c_;
    const Clock& clock_;
    ArtifactBuilder& b_;
    std::int32_t vocab_ = 0;
    std::uint64_t next_seed_ = 0;
    bool warned_spec_in_decode_ = false;
    bool warned_conc_eff_ = false;
};

}  // namespace

SuiteArtifact run_model_suite(runtime::Engine& engine, const ModelSuiteConfig& c, const Clock& clock) {
    check_counts(c.allow_nonconformant, c.warmup, c.repetitions, 0, kEndToEndMinRepetitions, "model");
    ArtifactBuilder b("model", c.environment, c.identity, c.stability);
    ModelRunner(engine, c, clock, b).run();
    return b.finish();
}

// ---- system -----------------------------------------------------------------------------

SuiteArtifact run_system_suite(runtime::Engine& engine, const SystemSuiteConfig& c, const Clock& clock) {
    check_counts(c.allow_nonconformant, c.warmup, c.repetitions, 0, kEndToEndMinRepetitions, "system");
    HALO_CHECK(static_cast<bool>(c.encoder), ErrorCode::Config, "system suite: encoder is required");
    HALO_CHECK(c.agents >= 1 && c.agents <= kMaxAgents, ErrorCode::Config, "system suite: agents {} not in [1, {}]",
               c.agents, kMaxAgents);
    HALO_CHECK(!c.workload.agents.empty(), ErrorCode::Config, "system suite: workload has no agent scripts");
    ArtifactBuilder b("system", c.environment, c.identity, c.stability);

    struct TurnResult {
        Timed t;
        std::size_t turn = 0;
        std::size_t prompt_tokens = 0;
        bool prefix_unstable = false;
    };

    const unsigned total = c.warmup + c.repetitions;
    bool warned_unstable = false;
    for (unsigned i = 0; i < total; ++i) {
        std::vector<std::vector<TurnResult>> per_agent(c.agents);
        const std::int64_t t0 = clock.now_ns();
        {
            std::vector<std::jthread> threads;
            for (std::uint32_t a = 0; a < c.agents; ++a) {
                threads.emplace_back([&, a] {
                    const AgentScript& script = c.workload.agents[a % c.workload.agents.size()];
                    // Conversation replay: turn k's prompt = turn k-1's prompt + its generated
                    // tokens + the new turn's delta (enc(k) minus the enc(k-1) prefix), so each
                    // turn extends the previous context the way a chat client resends it. An
                    // encoder that is not prefix-stable falls back to enc(k) alone (noted).
                    std::vector<std::int32_t> prev_encoded;
                    std::vector<std::int32_t> context;
                    for (std::size_t turn = 0; turn < script.turns.size(); ++turn) {
                        TurnResult tr;
                        tr.turn = turn;
                        runtime::GenerateRequest req;
                        std::vector<std::int32_t> full;
                        try {
                            full = c.encoder(c.workload, script, turn);
                        } catch (const std::exception& e) {
                            tr.t.error = std::format("encoder: {}", e.what());
                            per_agent[a].push_back(std::move(tr));
                            return;
                        }
                        if (turn > 0 && full.size() >= prev_encoded.size() &&
                            std::equal(prev_encoded.begin(), prev_encoded.end(), full.begin())) {
                            req.prompt = context;
                            req.prompt.insert(req.prompt.end(),
                                              full.begin() + static_cast<std::ptrdiff_t>(prev_encoded.size()), full.end());
                        } else {
                            if (turn > 0) tr.prefix_unstable = true;
                            req.prompt = full;
                        }
                        prev_encoded = std::move(full);
                        req.max_tokens = script.max_tokens;
                        req.sampling = c.sampling;
                        tr.prompt_tokens = req.prompt.size();
                        tr.t = timed_generate(engine, req, clock);
                        if (tr.t.error.empty() && tr.t.result.finish == runtime::FinishReason::Error) {
                            tr.t.error = std::format("engine reported error: {}", tr.t.result.error);
                        }
                        context = req.prompt;
                        context.insert(context.end(), tr.t.result.tokens.begin(), tr.t.result.tokens.end());
                        const bool failed = !tr.t.error.empty();
                        per_agent[a].push_back(std::move(tr));
                        if (failed) return;
                        if (script.think_time_ms > 0 && turn + 1 < script.turns.size()) {
                            if (c.think) {
                                c.think(script.think_time_ms);
                            } else {
                                std::this_thread::sleep_for(std::chrono::milliseconds(script.think_time_ms));
                            }
                        }
                    }
                });
            }
        }
        const std::int64_t t1 = clock.now_ns();

        std::size_t generated = 0, prompt_total = 0, cached_total = 0;
        std::vector<double> ttfts, ttfts_reuse;
        bool failed = false;
        for (const auto& agent : per_agent) {
            for (const auto& tr : agent) {
                if (tr.prefix_unstable && !warned_unstable) {
                    warned_unstable = true;
                    b.note("system: encoder output for turn k does not extend turn k-1; turns were sent "
                           "without conversation history");
                }
                if (!tr.t.error.empty()) {
                    b.failure(std::format("system: turn {} failed: {}", tr.turn, tr.t.error));
                    failed = true;
                    continue;
                }
                generated += tr.t.tokens;
                prompt_total += tr.prompt_tokens;
                cached_total += tr.t.result.cached_prompt_tokens;
                if (const auto v = ttft_ms(tr.t)) {
                    ttfts.push_back(*v);
                    if (tr.turn >= 1) ttfts_reuse.push_back(*v);
                }
            }
        }
        if (failed) break;
        BenchmarkRecord r = base_record(c.identity, "system", "agents");
        r.phase = i < c.warmup ? Phase::Cold : Phase::Steady;
        r.repetition = i < c.warmup ? i : i - c.warmup;
        r.concurrency = c.agents;
        r.batch = c.agents;
        if (t1 > t0) r.decode_tps = static_cast<double>(generated) * 1e9 / static_cast<double>(t1 - t0);
        if (!ttfts.empty()) r.ttft_ms = compute_sample_stats(ttfts).p95;
        if (!ttfts_reuse.empty()) r.ttft_cached_ms = compute_sample_stats(ttfts_reuse).mean;
        if (prompt_total > 0) r.prefix_cache_hit_rate = static_cast<double>(cached_total) / static_cast<double>(prompt_total);
        r.extra["workload"] = c.workload.id;
        r.extra["generated_tokens"] = generated;
        r.extra["prompt_tokens"] = prompt_total;
        r.extra["wall_ms"] = ns_to_ms(t1 - t0);
        b.add(std::move(r));
    }
    return b.finish();
}

SuiteArtifact run_suite(const SuiteRequest& req, const Clock& clock) {
    switch (req.kind) {
        case SuiteKind::Micro:
            HALO_CHECK(req.micro.has_value(), ErrorCode::Config, "run_suite: micro config missing");
            return run_micro_suite(*req.micro, clock);
        case SuiteKind::Model:
            HALO_CHECK(req.model.has_value() && req.engine != nullptr, ErrorCode::Config,
                       "run_suite: model config or engine missing");
            return run_model_suite(*req.engine, *req.model, clock);
        case SuiteKind::System:
            HALO_CHECK(req.system.has_value() && req.engine != nullptr, ErrorCode::Config,
                       "run_suite: system config or engine missing");
            return run_system_suite(*req.engine, *req.system, clock);
    }
    throw_error(ErrorCode::Config, "run_suite: unknown suite kind");
}

}  // namespace halo::profiling
