#include <gtest/gtest.h>

#include <algorithm>

#include <nlohmann/json.hpp>

#include "halo/profiling/record.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;

namespace {

BenchmarkRecord rec(std::string suite, std::string mode, Phase phase, std::optional<double> decode) {
    BenchmarkRecord r;
    r.model = "Qwen/Qwen3.8-27B";
    r.suite = std::move(suite);
    r.mode = std::move(mode);
    r.engine = "halo";
    r.context = 4096;
    r.phase = phase;
    r.decode_tps = decode;
    r.power_mode = "performance|manual/COMPUTE";
    return r;
}

}  // namespace

// ---- D-011 byte model ----------------------------------------------------------------------

TEST(ByteModel, ReproducesD011WorkedNumbers) {
    DecodeByteModel m;
    m.w_trunk_gb = 16.48;  // UD-Q4_K_XL trunk incl. LM head
    // S=1, n=0 -> K=1: B = 16.48 + 2*0.157 = 16.794 GB (ctx ~ 0 in the llama-cli run)
    EXPECT_NEAR(m.bytes_per_step(), 16.794e9, 1.0);
    EXPECT_NEAR(m.ceiling_tps(256.0), 15.2434, 1e-3);  // D-011: "about 15.2 tok/s"
    EXPECT_NEAR(decode_efficiency(12.0, m.bytes_per_token(), 256.0), 0.78722, 1e-4);  // "eta ~ 0.79"

    m.context_tokens = 4096;  // + 4096 * 64 KiB = 0.268435456 GB
    EXPECT_NEAR(m.bytes_per_step(), 16.794e9 + 4096.0 * 65536.0, 1.0);
    EXPECT_NEAR(m.ceiling_tps(256.0), 256e9 / (16.794e9 + 268435456.0), 1e-9);
}

TEST(ByteModel, MtpAndSequencesTerms) {
    DecodeByteModel m;
    m.w_trunk_gb = 16.48;
    m.w_mtp_gb = 0.5;
    m.w_head_gb = 1.04;
    m.draft_tokens = 2;  // K defaults to 3
    m.sequences = 2;
    m.context_tokens = 1000;
    const double weights = (16.48 + 3 * 0.5 + 2 * 1.04) * 1e9;
    const double per_seq = 4 * 0.157e9 + 1000.0 * 65536.0;
    EXPECT_NEAR(m.bytes_per_step(), weights + 2 * per_seq, 1.0);
    EXPECT_NEAR(m.bytes_per_token(2.5), (weights + 2 * per_seq) / 5.0, 1.0);
    EXPECT_NEAR(m.ceiling_tps(200.0, 0.8, 2.5), 0.8 * 200e9 * 5.0 / (weights + 2 * per_seq), 1e-9);
    m.state_copies = 0;
    EXPECT_NEAR(m.bytes_per_step(), weights + 2 * (0.157e9 + 1000.0 * 65536.0), 1.0);
}

TEST(ByteModel, RejectsInvalidInputs) {
    DecodeByteModel m;
    m.w_trunk_gb = 16;
    m.sequences = 0;
    test::expect_error(ErrorCode::Config, [&] { (void)m.bytes_per_step(); });
    m.sequences = 1;
    m.w_mtp_gb = -1;
    test::expect_error(ErrorCode::Config, [&] { (void)m.bytes_per_step(); });
    m.w_mtp_gb = 0;
    test::expect_error(ErrorCode::Config, [&] { (void)m.bytes_per_token(0.5); });
    test::expect_error(ErrorCode::Config, [&] { (void)m.ceiling_tps(0.0); });
    test::expect_error(ErrorCode::Config, [] { (void)decode_efficiency(1, 1, -5); });
}

TEST(ByteModel, ApplyEfficiencyPrefersEffectiveTps) {
    DecodeByteModel m;
    m.w_trunk_gb = 16.48;
    BenchmarkRecord r;
    apply_efficiency(r, m, 256.0);
    EXPECT_DOUBLE_EQ(*r.measured_bandwidth_gbps, 256.0);
    EXPECT_NEAR(*r.predicted_bytes_per_token, 16.794e9, 1.0);
    EXPECT_FALSE(r.efficiency);  // no decode measured
    r.decode_tps = 12.0;
    apply_efficiency(r, m, 256.0);
    EXPECT_NEAR(*r.efficiency, 0.78722, 1e-4);
    r.decode_effective_tps = 24.0;
    apply_efficiency(r, m, 256.0);
    EXPECT_NEAR(*r.efficiency, 2 * 0.78722, 2e-4);
}

// ---- record JSON ---------------------------------------------------------------------------

TEST(Record, JsonHasEveryTrd28FieldAndNullsForUnmeasured) {
    BenchmarkRecord r = rec("model", "decode", Phase::Steady, 12.5);
    r.memory_by_tier_gb["VRAM"] = 17.2;
    r.clocks_mhz["sclk"] = 2900;
    const nlohmann::json j = r;
    for (const char* k : {"model", "model_hash", "pack", "backend", "driver", "gpu", "quantization", "lm_head",
                          "context", "batch", "concurrency", "ttft_ms", "ttft_cached_ms", "prompt_tps",
                          "decode_tps", "decode_effective_tps", "mtp_acceptance", "prefix_cache_hit_rate",
                          "memory_by_tier_gb", "load_time_s", "temperature_c", "clocks_mhz", "power_mode",
                          "measured_bandwidth_gbps", "predicted_bytes_per_token", "efficiency", "host_label",
                          "phase", "hardware_before", "hardware_after", "invocation"}) {
        EXPECT_TRUE(j.contains(k)) << k;
    }
    EXPECT_EQ(j.at("schema"), kRecordSchema);
    EXPECT_DOUBLE_EQ(j.at("decode_tps").get<double>(), 12.5);
    EXPECT_TRUE(j.at("ttft_ms").is_null());  // not measured != 0
    EXPECT_TRUE(j.at("mtp_acceptance").is_null());
    EXPECT_EQ(j.at("phase"), "steady");
}

TEST(Record, RoundTrip) {
    BenchmarkRecord r = rec("baseline", "tg128", Phase::Cold, 20.0);
    r.ttft_ms = 0.0;  // a measured zero survives as 0, not null
    r.invocation = Invocation{"/root/llama.cpp/build/bin/llama-bench", "0.5.0-dev", "bd4f514",
                              {"llama-bench", "-m", "x.gguf", "-o", "json"}, {{"RADV_PERFTEST", "nogttspill"}}};
    HardwareState hw;
    hw.power_mode = "a|b/c";
    hw.temperature_c = 44.0;
    r.hardware_before = hw;
    r.extra["n_prompt"] = 512;
    const nlohmann::json j = r;
    const BenchmarkRecord back = j.get<BenchmarkRecord>();
    EXPECT_EQ(nlohmann::json(back), j);
    EXPECT_EQ(back.phase, Phase::Cold);
    ASSERT_TRUE(back.ttft_ms);
    EXPECT_EQ(*back.ttft_ms, 0.0);
    EXPECT_EQ(back.invocation->argv.size(), 5u);
}

TEST(Record, ParseRejectsForeignOrMalformed) {
    const nlohmann::json j = rec("model", "decode", Phase::Steady, 1.0);
    nlohmann::json foreign = j;
    foreign["schema"] = "halo.bench.record/2";
    test::expect_error(ErrorCode::Config, [&] { (void)foreign.get<BenchmarkRecord>(); });
    nlohmann::json missing = j;
    missing.erase("power_mode");
    test::expect_error(ErrorCode::Config, [&] { (void)missing.get<BenchmarkRecord>(); });
    nlohmann::json badtype = j;
    badtype["decode_tps"] = "fast";
    test::expect_error(ErrorCode::Config, [&] { (void)badtype.get<BenchmarkRecord>(); });
    nlohmann::json negative = j;
    negative["context"] = -1;
    test::expect_error(ErrorCode::Config, [&] { (void)negative.get<BenchmarkRecord>(); });
    nlohmann::json fractional = j;
    fractional["batch"] = 1.5;
    test::expect_error(ErrorCode::Config, [&] { (void)fractional.get<BenchmarkRecord>(); });
    nlohmann::json phase = j;
    phase["phase"] = "warm";
    test::expect_error(ErrorCode::Config, [&] { (void)phase.get<BenchmarkRecord>(); });
    test::expect_error(ErrorCode::Config, [] { (void)nlohmann::json::array().get<BenchmarkRecord>(); });
}

// ---- roll-up -------------------------------------------------------------------------------

TEST(Summary, ColdIsSeparateFromSteadyAndCountedAsWarmup) {
    std::vector<BenchmarkRecord> rs;
    rs.push_back(rec("model", "decode", Phase::Cold, 1.0));  // slow first run
    for (double v : {10.0, 11.0, 12.0}) rs.push_back(rec("model", "decode", Phase::Steady, v));
    const auto sums = summarize(rs);
    ASSERT_EQ(sums.size(), 2u);
    const RunSummary* steady = nullptr;
    const RunSummary* cold = nullptr;
    for (const auto& s : sums) (s.phase == Phase::Steady ? steady : cold) = &s;
    ASSERT_TRUE(steady && cold);
    EXPECT_EQ(steady->measured_count, 3u);
    EXPECT_EQ(steady->warmup_count, 1u);
    EXPECT_TRUE(steady->conformant);
    const auto& d = steady->metrics.at("decode_tps").stats;
    EXPECT_DOUBLE_EQ(d.min, 10.0);  // the cold 1.0 must not leak in
    EXPECT_DOUBLE_EQ(d.mean, 11.0);
    EXPECT_EQ(cold->measured_count, 1u);
    EXPECT_FALSE(steady->metrics.contains("ttft_ms"));  // never measured -> absent
    EXPECT_EQ(steady->metrics.at("decode_tps").stability, Stability::Unstable);  // cv 0.09 > 0.05
}

TEST(Summary, ConformanceMinimums) {
    std::vector<BenchmarkRecord> e2e;
    for (double v : {10.0, 10.0}) e2e.push_back(rec("model", "decode", Phase::Steady, v));
    auto s = summarize(e2e);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(s[0].required_measured, kEndToEndMinRepetitions);
    EXPECT_FALSE(s[0].conformant);  // 2 < 3
    EXPECT_EQ(s[0].metrics.at("decode_tps").stability, Stability::TooFewSamples);

    std::vector<BenchmarkRecord> micro;
    for (int i = 0; i < 4; ++i) micro.push_back(rec("micro", "MATMUL", Phase::Cold, std::nullopt));
    for (int i = 0; i < 20; ++i) {
        auto r = rec("micro", "MATMUL", Phase::Steady, std::nullopt);
        r.extra["latency_ns"] = 1000 + i;
        micro.push_back(r);
    }
    s = summarize(micro);
    const auto steady = std::find_if(s.begin(), s.end(), [](const RunSummary& x) { return x.phase == Phase::Steady; });
    ASSERT_NE(steady, s.end());
    EXPECT_EQ(steady->required_measured, kMicroMinMeasured);
    EXPECT_EQ(steady->warmup_count, 4u);
    EXPECT_FALSE(steady->conformant);  // 4 < 5 warm-ups
    EXPECT_EQ(steady->metrics.at("extra.latency_ns").stats.n, 20u);
    EXPECT_EQ(steady->metrics.at("extra.latency_ns").stability, Stability::Stable);

    micro.push_back(rec("micro", "MATMUL", Phase::Cold, std::nullopt));
    s = summarize(micro);
    for (const auto& x : s) {
        if (x.phase == Phase::Steady) EXPECT_TRUE(x.conformant);
    }
}

TEST(Summary, GroupsByContextAndConcurrency) {
    std::vector<BenchmarkRecord> rs;
    for (std::uint32_t n : {1u, 4u}) {
        for (int i = 0; i < 3; ++i) {
            auto r = rec("model", "concurrency", Phase::Steady, 10.0 * n);
            r.concurrency = n;
            rs.push_back(r);
        }
    }
    const auto s = summarize(rs);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_DOUBLE_EQ(s[0].metrics.at("decode_tps").stats.mean, 10.0);
    EXPECT_DOUBLE_EQ(s[1].metrics.at("decode_tps").stats.mean, 40.0);
    const nlohmann::json j = s[1];
    EXPECT_EQ(j.at("schema"), kSummarySchema);
    EXPECT_EQ(j.at("metrics").at("decode_tps").at("stability"), "stable");
}
