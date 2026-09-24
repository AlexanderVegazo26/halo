// Suites driven through runtime::Engine with a fake engine whose timing is controlled by a
// ManualClock. Single-threaded paths (micro, load, prompt, decode, prefix replay) assert exact
// numbers; concurrent paths (concurrency, storm, system) share one clock across threads, so
// they assert only invariants (counts, cancellation, conformance), never exact timings.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "halo/profiling/suite.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;
namespace rt = halo::runtime;

namespace {

constexpr std::int64_t kPrefillNsPerToken = 100'000;  // 10k tok/s prompt processing
constexpr std::int64_t kDecodeNsPerToken = 10'000'000;  // 100 tok/s decode

struct FakeOptions {
    std::size_t context_length = 8192;
    bool has_mtp = false;
    bool throw_on_generate = false;
    bool report_error = false;
    bool ignore_cancel = false;
    int fail_from_call = 0;         ///< >0: the N-th generate() call and later report Error
    bool prefix_cache = false;      ///< longest-common-prefix reuse against earlier prompts
    bool speculate_in_plain_decode = false;
};

class FakeEngine final : public rt::Engine {
public:
    FakeEngine(ManualClock& clock, FakeOptions o) : clock_(clock), o_(o) {
        info_.id = "fake";
        info_.architecture = "qwen35";
        info_.context_length = o.context_length;
        info_.vocab_size = 1000;
        info_.has_mtp = o.has_mtp;
    }

    const rt::ModelInfo& model() const override { return info_; }
    const halo::tokenizer::Tokenizer& tokenizer() const override { throw std::logic_error("fake: no tokenizer"); }
    const halo::chat::ChatTemplate& chat_template() const override { throw std::logic_error("fake: no template"); }

    rt::GenerateResult generate(const rt::GenerateRequest& req, const rt::TokenCallback& cb) override {
        const int call = calls_.fetch_add(1) + 1;
        if (o_.throw_on_generate) throw std::runtime_error("fake: boom");
        rt::GenerateResult r;
        r.prompt_tokens = req.prompt.size();
        if (o_.report_error || (o_.fail_from_call > 0 && call >= o_.fail_from_call)) {
            r.finish = rt::FinishReason::Error;
            r.error = "fake: engine error";
            return r;
        }
        std::size_t cached = 0;
        if (o_.prefix_cache) {
            const std::scoped_lock lock(mu_);
            for (const auto& p : seen_) {
                std::size_t k = 0;
                while (k < p.size() && k < req.prompt.size() && p[k] == req.prompt[k]) ++k;
                cached = std::max(cached, k);
            }
            seen_.push_back(req.prompt);
        }
        // Never report the whole prompt as cached: the last token is always recomputed.
        cached = std::min(cached, req.prompt.empty() ? 0 : req.prompt.size() - 1);
        r.cached_prompt_tokens = cached;
        clock_.advance(kPrefillNsPerToken * static_cast<std::int64_t>(req.prompt.size() - cached));
        r.finish = rt::FinishReason::Length;
        for (std::size_t i = 0; i < req.max_tokens; ++i) {
            if (i > 0) clock_.advance(kDecodeNsPerToken);
            const std::int32_t tok = static_cast<std::int32_t>((req.prompt.size() + i) % 1000);
            r.tokens.push_back(tok);
            if (!cb({tok, "x", false}) && !o_.ignore_cancel) {
                r.finish = rt::FinishReason::Cancelled;
                break;
            }
        }
        if (o_.has_mtp || o_.speculate_in_plain_decode) {
            // Half the emitted tokens came from accepted drafts: steps = gen - accepted,
            // tau = gen / steps = 2; two drafts proposed per step.
            r.accepted_draft_tokens = r.tokens.size() / 2;
            r.draft_tokens = 2 * (r.tokens.size() - r.accepted_draft_tokens);
        }
        return r;
    }

    rt::EngineStats stats() const override { return {}; }
    int calls() const { return calls_.load(); }

private:
    ManualClock& clock_;
    FakeOptions o_;
    rt::ModelInfo info_;
    std::mutex mu_;
    std::vector<std::vector<std::int32_t>> seen_;
    std::atomic<int> calls_{0};
};

RecordIdentity identity() {
    RecordIdentity id;
    id.model = "Qwen/Qwen3.8-27B";
    id.pack = "fake";
    id.backend = "cpu";
    id.host_label = "unit-test";
    return id;
}

EnvironmentOptions env_pinned() {
    EnvironmentOptions e;
    e.discovery.root = test::sysfs_fixture("evo_x2");
    e.discovery.use_process_env = false;
    e.discovery.probe_runtime_cpuid = false;
    e.platform_power_mode = "performance";
    return e;
}

ModelSuiteConfig model_config(std::set<ModelMode> modes) {
    ModelSuiteConfig c;
    c.identity = identity();
    c.modes = std::move(modes);
    c.contexts = {256, 1024};
    c.concurrency = {1, 2, 4};
    c.concurrency_context = 256;
    c.prompt_tokens = 64;
    c.decode_tokens = 16;
    c.prefix_replay_context = 512;
    c.prefix_replay_suffix = 64;
    c.storm_requests = 4;
    c.storm_cancel_after = 3;
    c.environment = env_pinned();
    return c;
}

const RunSummary* find_summary(const SuiteArtifact& a, std::string_view mode, Phase phase,
                               std::uint64_t context = 0, std::uint32_t concurrency = 0) {
    for (const auto& s : a.summaries) {
        if (s.mode == mode && s.phase == phase && (context == 0 || s.context == context) &&
            (concurrency == 0 || s.concurrency == concurrency)) {
            return &s;
        }
    }
    return nullptr;
}

bool has_note(const SuiteArtifact& a, std::string_view needle) {
    return std::any_of(a.notes.begin(), a.notes.end(),
                       [&](const std::string& n) { return n.find(needle) != std::string::npos; });
}

}  // namespace

// ---- micro -------------------------------------------------------------------------------

TEST(MicroSuite, WarmupAndMeasuredRecordsWithInjectedClock) {
    ManualClock clock;
    MicroSuiteConfig c;
    c.identity = identity();
    c.environment = env_pinned();
    int calls = 0;
    c.benchmarks.push_back({"COPY/1MiB", [&] { ++calls; clock.advance(1000); }, 1'000'000});
    const SuiteArtifact a = run_micro_suite(c, clock);
    EXPECT_EQ(calls, 25);
    ASSERT_EQ(a.records.size(), 25u);
    EXPECT_EQ(std::count_if(a.records.begin(), a.records.end(), [](const auto& r) { return r.phase == Phase::Cold; }),
              5);
    const RunSummary* s = find_summary(a, "COPY/1MiB", Phase::Steady);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->measured_count, 20u);
    EXPECT_EQ(s->warmup_count, 5u);
    EXPECT_TRUE(s->conformant);
    EXPECT_DOUBLE_EQ(s->metrics.at("extra.latency_ns").stats.mean, 1000.0);
    EXPECT_DOUBLE_EQ(s->metrics.at("extra.gbps").stats.mean, 1000.0);  // 1e6 B / 1000 ns
    EXPECT_EQ(s->metrics.at("extra.latency_ns").stability, Stability::Stable);
    EXPECT_TRUE(a.conformant);
    EXPECT_TRUE(a.thermal.valid);
    EXPECT_TRUE(a.power_mode_pinned.comparable) << a.power_mode_pinned.reason;
    EXPECT_TRUE(a.valid);
    EXPECT_EQ(a.records[0].power_mode, "performance|auto/BOOTUP_DEFAULT");
    EXPECT_EQ(a.records[0].gpu, "gfx1151");
    EXPECT_EQ(a.records[0].host_label, "unit-test");
    ASSERT_TRUE(a.records[0].hardware_before && a.records[0].hardware_after);
}

TEST(MicroSuite, RefusesBelowMethodologyMinimumUnlessAllowed) {
    ManualClock clock;
    MicroSuiteConfig c;
    c.identity = identity();
    c.environment.enabled = false;
    c.benchmarks.push_back({"NOP", [] {}, 0});
    c.warmup = 4;
    test::expect_error(ErrorCode::Config, [&] { (void)run_micro_suite(c, clock); });
    c.warmup = 5;
    c.iterations = 19;
    test::expect_error(ErrorCode::Config, [&] { (void)run_micro_suite(c, clock); });
    c.allow_nonconformant = true;
    const SuiteArtifact a = run_micro_suite(c, clock);
    EXPECT_FALSE(a.conformant);
    EXPECT_FALSE(a.valid);
    c.identity.host_label.clear();  // D-001: every record must say where it ran
    test::expect_error(ErrorCode::Config, [&] { (void)run_micro_suite(c, clock); });
}

TEST(MicroSuite, BeforeSnapshotIsTakenBeforeTheRun) {
    ManualClock clock;
    MicroSuiteConfig c;
    c.identity = identity();
    c.environment = env_pinned();
    std::atomic<int> phase{0};
    c.environment.utc_now = [&] { return phase.load() == 0 ? std::string("before-run") : std::string("after-run"); };
    c.benchmarks.push_back({"NOP", [&] { phase.store(1); }, 0});
    const SuiteArtifact a = run_micro_suite(c, clock);
    ASSERT_TRUE(a.hardware_before && a.hardware_after);
    EXPECT_EQ(a.hardware_before->captured_at, "before-run");
    EXPECT_EQ(a.hardware_after->captured_at, "after-run");
}

// ---- environment verdicts --------------------------------------------------------------------

TEST(Environment, UnpinnedPowerModeDriftAndDisabledCaptureInvalidate) {
    ManualClock clock;
    MicroSuiteConfig c;
    c.identity = identity();
    c.benchmarks.push_back({"NOP", [] {}, 0});

    c.environment = env_pinned();
    c.environment.platform_power_mode.clear();  // BIOS/EC mode not supplied
    SuiteArtifact a = run_micro_suite(c, clock);
    EXPECT_TRUE(a.conformant);
    EXPECT_FALSE(a.power_mode_pinned.comparable);
    EXPECT_FALSE(a.valid);

    c.environment = env_pinned();
    c.environment.patch_states = [](HardwareState&, HardwareState& after) { after.temperature_c = 70.0; };
    a = run_micro_suite(c, clock);
    EXPECT_FALSE(a.thermal.valid);
    EXPECT_DOUBLE_EQ(*a.thermal.drift_c, 25.0);
    EXPECT_FALSE(a.valid);

    c.environment = env_pinned();
    c.environment.patch_states = [](HardwareState&, HardwareState& after) {
        after.power_mode = make_power_mode("quiet", std::string("auto"), std::string("BOOTUP_DEFAULT"));
    };
    a = run_micro_suite(c, clock);
    EXPECT_FALSE(a.power_mode_pinned.comparable);
    EXPECT_FALSE(a.valid);

    c.environment = EnvironmentOptions{};
    c.environment.enabled = false;
    a = run_micro_suite(c, clock);
    EXPECT_FALSE(a.hardware_before);
    EXPECT_FALSE(a.valid);
    EXPECT_EQ(a.records[0].power_mode, "unknown|unknown/unknown");
    const nlohmann::json j = a;
    EXPECT_EQ(j.at("schema"), "halo.bench.artifact/1");
    EXPECT_FALSE(j.at("valid").get<bool>());
}

// ---- model -------------------------------------------------------------------------------

TEST(ModelSuite, PromptAndDecodeTimingsAreExact) {
    ManualClock clock;
    FakeEngine e(clock, {});
    auto c = model_config({ModelMode::PromptProcessing, ModelMode::Decode});
    c.contexts = {256, 1024, 16, 16384};  // 16 <= decode_tokens, 16384 > context_length
    c.efficiency = EfficiencyInputs{};
    c.efficiency->byte_model.w_trunk_gb = 16.48;
    c.efficiency->byte_model.w_mtp_gb = 0.5;   // must be ignored outside mtp_decode
    c.efficiency->byte_model.draft_tokens = 2;  // idem
    c.efficiency->measured_bandwidth_gbps = 256.0;
    const SuiteArtifact a = run_model_suite(e, c, clock);
    EXPECT_TRUE(a.valid) << nlohmann::json(a.notes).dump();

    const RunSummary* p = find_summary(a, "prompt", Phase::Steady);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->measured_count, 3u);
    EXPECT_EQ(p->warmup_count, 1u);
    EXPECT_DOUBLE_EQ(p->metrics.at("ttft_ms").stats.mean, 64 * 0.1);  // 64 tokens * 0.1 ms
    EXPECT_NEAR(p->metrics.at("prompt_tps").stats.mean, 10000.0, 1e-6);

    for (const std::uint64_t ctx : {256u, 1024u}) {
        const RunSummary* d = find_summary(a, "decode", Phase::Steady, ctx);
        ASSERT_NE(d, nullptr) << ctx;
        EXPECT_NEAR(d->metrics.at("decode_tps").stats.mean, 100.0, 1e-9);
        EXPECT_DOUBLE_EQ(d->metrics.at("ttft_ms").stats.mean, static_cast<double>(ctx - 16) * 0.1);
        DecodeByteModel m;
        m.w_trunk_gb = 16.48;
        m.context_tokens = ctx;  // n = 0, no MTP weights
        EXPECT_NEAR(d->metrics.at("efficiency").stats.mean, 100.0 * m.bytes_per_step() / 256e9, 1e-12);
    }
    EXPECT_EQ(find_summary(a, "decode", Phase::Steady, 16), nullptr);
    EXPECT_EQ(find_summary(a, "decode", Phase::Steady, 16384), nullptr);
    EXPECT_TRUE(has_note(a, "context 16 skipped"));
    EXPECT_TRUE(has_note(a, "context 16384 skipped"));
    for (const auto& r : a.records) {
        EXPECT_TRUE(r.extra.contains("engine_decode_tps"));  // engine self-report kept separately
        EXPECT_FALSE(r.mtp_acceptance);
    }
}

TEST(ModelSuite, MtpDecodeDerivesTauForEfficiency) {
    ManualClock clock;
    FakeEngine e(clock, {.has_mtp = true});
    auto c = model_config({ModelMode::MtpDecode});
    c.contexts = {256};
    c.efficiency = EfficiencyInputs{};
    c.efficiency->byte_model.w_trunk_gb = 16.48;
    c.efficiency->byte_model.w_mtp_gb = 0.5;
    c.efficiency->byte_model.w_head_gb = 1.04;
    c.efficiency->byte_model.draft_tokens = 2;
    c.efficiency->measured_bandwidth_gbps = 256.0;
    const SuiteArtifact a = run_model_suite(e, c, clock);
    const RunSummary* s = find_summary(a, "mtp_decode", Phase::Steady, 256);
    ASSERT_NE(s, nullptr);
    EXPECT_FALSE(s->metrics.contains("decode_tps"));
    EXPECT_NEAR(s->metrics.at("decode_effective_tps").stats.mean, 100.0, 1e-9);
    // 16 generated, 8 accepted, 16 drafted -> acceptance 0.5, tau = 16 / 8 = 2
    EXPECT_DOUBLE_EQ(s->metrics.at("mtp_acceptance").stats.mean, 0.5);
    EXPECT_DOUBLE_EQ(s->metrics.at("extra.tau_observed").stats.mean, 2.0);
    DecodeByteModel m = c.efficiency->byte_model;
    m.context_tokens = 256;
    EXPECT_NEAR(s->metrics.at("efficiency").stats.mean, 100.0 * m.bytes_per_token(2.0) / 256e9, 1e-12);
}

TEST(ModelSuite, MtpDecodeSkippedWithoutMtpAndSpeculationInPlainDecodeNoted) {
    ManualClock clock;
    FakeEngine e(clock, {.speculate_in_plain_decode = true});
    auto c = model_config({ModelMode::Decode, ModelMode::MtpDecode});
    c.contexts = {256};
    const SuiteArtifact a = run_model_suite(e, c, clock);
    EXPECT_TRUE(has_note(a, "mtp_decode skipped"));
    EXPECT_TRUE(has_note(a, "draft tokens during plain decode"));
}

TEST(ModelSuite, LoadTimeUsesFactory) {
    ManualClock clock;
    FakeEngine e(clock, {});
    auto c = model_config({ModelMode::LoadTime});
    c.engine_factory = [&]() -> std::unique_ptr<rt::Engine> {
        clock.advance(2'000'000'000);
        return std::make_unique<FakeEngine>(clock, FakeOptions{});
    };
    SuiteArtifact a = run_model_suite(e, c, clock);
    const RunSummary* s = find_summary(a, "load_time", Phase::Steady);
    ASSERT_NE(s, nullptr);
    EXPECT_DOUBLE_EQ(s->metrics.at("load_time_s").stats.mean, 2.0);
    EXPECT_TRUE(a.valid);

    c.engine_factory = []() -> std::unique_ptr<rt::Engine> { return nullptr; };
    a = run_model_suite(e, c, clock);
    EXPECT_TRUE(has_note(a, "FAILED: load_time: engine_factory returned null"));
    EXPECT_FALSE(a.valid);

    c.engine_factory = nullptr;
    a = run_model_suite(e, c, clock);
    EXPECT_TRUE(has_note(a, "load_time skipped"));
    EXPECT_FALSE(a.conformant);  // nothing measured is not a conformant artifact
}

TEST(ModelSuite, PrefixReplayMeasuresCachedTtft) {
    ManualClock clock;
    FakeEngine e(clock, {.prefix_cache = true});
    auto c = model_config({ModelMode::PrefixReplay});
    const SuiteArtifact a = run_model_suite(e, c, clock);
    const RunSummary* s = find_summary(a, "prefix_replay", Phase::Steady, 512);
    ASSERT_NE(s, nullptr);
    EXPECT_DOUBLE_EQ(s->metrics.at("ttft_ms").stats.mean, 512 * 0.1);     // cold: whole prefix
    EXPECT_DOUBLE_EQ(s->metrics.at("ttft_cached_ms").stats.mean, 64 * 0.1);  // replay: suffix only
    EXPECT_DOUBLE_EQ(s->metrics.at("prefix_cache_hit_rate").stats.mean, 512.0 / 576.0);
    EXPECT_TRUE(a.valid);
}

TEST(ModelSuite, ConcurrencyAndStormInvariants) {
    ManualClock clock;
    FakeEngine e(clock, {});
    auto c = model_config({ModelMode::Concurrency, ModelMode::CancellationStorm});
    c.efficiency = EfficiencyInputs{};
    c.efficiency->byte_model.w_trunk_gb = 16.48;
    c.efficiency->measured_bandwidth_gbps = 256.0;
    const SuiteArtifact a = run_model_suite(e, c, clock);
    for (const auto& r : a.records) {
        if (r.mode == "concurrency") EXPECT_FALSE(r.efficiency) << "prefill-inclusive rate must not yield eta";
    }
    EXPECT_TRUE(has_note(a, "efficiency not computed"));
    EXPECT_TRUE(a.valid) << nlohmann::json(a.notes).dump();
    for (const std::uint32_t n : {1u, 2u, 4u}) {
        const RunSummary* s = find_summary(a, "concurrency", Phase::Steady, 256, n);
        ASSERT_NE(s, nullptr) << n;
        EXPECT_EQ(s->measured_count, 3u);
        EXPECT_TRUE(s->metrics.contains("decode_tps"));
        EXPECT_DOUBLE_EQ(s->metrics.at("extra.generated_tokens").stats.mean, 16.0 * n);
    }
    const RunSummary* st = find_summary(a, "cancellation_storm", Phase::Steady);
    ASSERT_NE(st, nullptr);
    EXPECT_DOUBLE_EQ(st->metrics.at("extra.cancelled").stats.mean, 4.0);
    EXPECT_DOUBLE_EQ(st->metrics.at("extra.tokens_after_cancel").stats.max, 0.0);
    for (const auto& r : a.records) {
        if (r.mode == "cancellation_storm") EXPECT_TRUE(r.extra.at("probe_ok").get<bool>());
    }
}

TEST(ModelSuite, StormDetectsIgnoredCancellationAndBrokenEngine) {
    ManualClock clock;
    auto c = model_config({ModelMode::CancellationStorm});
    {
        FakeEngine e(clock, {.ignore_cancel = true});
        const SuiteArtifact a = run_model_suite(e, c, clock);
        EXPECT_TRUE(has_note(a, "did not finish as Cancelled"));
        EXPECT_FALSE(a.valid);
        EXPECT_GT(a.records.back().extra.at("tokens_after_cancel").get<int>(), 0);
    }
    {
        // 4 storm requests of the warm-up run are calls 1-4; the probe is call 5.
        FakeEngine e(clock, {.fail_from_call = 5});
        const SuiteArtifact a = run_model_suite(e, c, clock);
        EXPECT_TRUE(has_note(a, "post-storm probe"));
        EXPECT_FALSE(a.valid);
    }
}

TEST(ModelSuite, EngineFailuresNeverProduceAValidArtifact) {
    ManualClock clock;
    for (const FakeOptions o : {FakeOptions{.throw_on_generate = true}, FakeOptions{.report_error = true}}) {
        FakeEngine e(clock, o);
        const SuiteArtifact a = run_model_suite(e, model_config({ModelMode::PromptProcessing}), clock);
        EXPECT_TRUE(has_note(a, "FAILED: prompt"));
        EXPECT_FALSE(a.conformant);
        EXPECT_FALSE(a.valid);
        EXPECT_EQ(e.calls(), 1);  // stops at the first failure
    }
}

TEST(ModelSuite, RefusesTooFewRepetitions) {
    ManualClock clock;
    FakeEngine e(clock, {});
    auto c = model_config({ModelMode::PromptProcessing});
    c.repetitions = 2;
    test::expect_error(ErrorCode::Config, [&] { (void)run_model_suite(e, c, clock); });
    c.allow_nonconformant = true;
    const SuiteArtifact a = run_model_suite(e, c, clock);
    EXPECT_FALSE(a.conformant);
    EXPECT_FALSE(a.valid);
}

// ---- system ------------------------------------------------------------------------------

TEST(SystemSuite, FourAgentReplayReusesPrefix) {
    ManualClock clock;
    FakeEngine e(clock, {.prefix_cache = true});
    SystemSuiteConfig c;
    c.identity = identity();
    c.environment = env_pinned();
    c.workload = load_workload(test::source_dir() / "bench" / "workloads" / "concurrency_4agents.json");
    c.encoder = synthetic_encoder(1000);
    std::atomic<int> thinks{0};
    c.think = [&](std::uint64_t ms) {
        thinks.fetch_add(1);
        clock.advance(static_cast<std::int64_t>(ms) * 1'000'000);
    };
    const SuiteArtifact a = run_system_suite(e, c, clock);
    EXPECT_TRUE(a.valid) << nlohmann::json(a.notes).dump();
    EXPECT_EQ(a.records.size(), 4u);                  // 1 warm-up + 3 measured
    EXPECT_EQ(e.calls(), 4 * 4 * 3);                  // runs * agents * turns
    EXPECT_EQ(thinks.load(), 4 * 4 * 2);              // between turns only
    std::size_t expected_generated = 0;
    for (const auto& ag : c.workload.agents) expected_generated += ag.max_tokens * ag.turns.size();
    for (const auto& r : a.records) {
        EXPECT_EQ(r.concurrency, 4u);
        EXPECT_EQ(r.extra.at("generated_tokens").get<std::size_t>(), expected_generated);
        ASSERT_TRUE(r.prefix_cache_hit_rate);
        EXPECT_GT(*r.prefix_cache_hit_rate, 0.5);  // turns 1-2 resend the whole conversation
        ASSERT_TRUE(r.ttft_cached_ms && r.ttft_ms);
    }
    EXPECT_FALSE(has_note(a, "does not extend"));
}

TEST(SystemSuite, ConversationHistoryIsResentInOrder) {
    // Turn k's prompt must start with turn k-1's prompt followed by its generated tokens.
    ManualClock clock;
    struct Recorder final : rt::Engine {
        explicit Recorder(ManualClock& c) : inner(c, {}) {}
        const rt::ModelInfo& model() const override { return inner.model(); }
        const halo::tokenizer::Tokenizer& tokenizer() const override { return inner.tokenizer(); }
        const halo::chat::ChatTemplate& chat_template() const override { return inner.chat_template(); }
        rt::GenerateResult generate(const rt::GenerateRequest& r, const rt::TokenCallback& cb) override {
            auto res = inner.generate(r, cb);
            prompts.push_back(r.prompt);
            outputs.push_back(res.tokens);
            return res;
        }
        rt::EngineStats stats() const override { return {}; }
        FakeEngine inner;
        std::vector<std::vector<std::int32_t>> prompts, outputs;
    } e(clock);
    SystemSuiteConfig c;
    c.identity = identity();
    c.environment.enabled = false;
    c.agents = 1;
    c.warmup = 0;
    c.repetitions = 1;
    c.allow_nonconformant = true;
    c.workload = load_workload(test::source_dir() / "bench" / "workloads" / "agent_replay.json");
    c.encoder = synthetic_encoder(1000);
    (void)run_system_suite(e, c, clock);
    const auto enc = synthetic_encoder(1000);
    ASSERT_EQ(e.prompts.size(), c.workload.agents[0].turns.size());
    for (std::size_t k = 1; k < e.prompts.size(); ++k) {
        std::vector<std::int32_t> expect = e.prompts[k - 1];
        expect.insert(expect.end(), e.outputs[k - 1].begin(), e.outputs[k - 1].end());
        const auto prev = enc(c.workload, c.workload.agents[0], k - 1);
        const auto full = enc(c.workload, c.workload.agents[0], k);
        expect.insert(expect.end(), full.begin() + static_cast<std::ptrdiff_t>(prev.size()), full.end());
        EXPECT_EQ(e.prompts[k], expect) << "turn " << k;
    }
}

TEST(SystemSuite, FailuresAndDispatch) {
    ManualClock clock;
    FakeEngine bad(clock, {.report_error = true});
    SystemSuiteConfig c;
    c.identity = identity();
    c.environment.enabled = false;
    c.workload = load_workload(test::source_dir() / "bench" / "workloads" / "concurrency_4agents.json");
    c.encoder = synthetic_encoder(1000);
    c.think = [](std::uint64_t) {};
    SuiteRequest req;
    req.kind = SuiteKind::System;
    req.system = c;
    test::expect_error(ErrorCode::Config, [&] { (void)run_suite(req, clock); });  // no engine
    req.engine = &bad;
    const SuiteArtifact a = run_suite(req, clock);
    EXPECT_TRUE(has_note(a, "FAILED: system"));
    EXPECT_FALSE(a.valid);
    EXPECT_TRUE(a.records.empty());

    req.system->encoder = nullptr;
    test::expect_error(ErrorCode::Config, [&] { (void)run_suite(req, clock); });
    SuiteRequest micro;
    micro.kind = SuiteKind::Micro;
    test::expect_error(ErrorCode::Config, [&] { (void)run_suite(micro, clock); });
}
