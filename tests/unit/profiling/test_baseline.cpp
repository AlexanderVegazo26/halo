#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "fake_http.h"
#include "halo/profiling/baseline.h"
#include "halo/profiling/sha256.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;
using halo::profiling::test::FakeHttpServer;
using namespace std::chrono_literals;

namespace {

std::filesystem::path fixture(const char* name) {
    return test::source_dir() / "tests" / "unit" / "profiling" / "fixtures" / name;
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json load(const char* name) { return nlohmann::json::parse(slurp(fixture(name))); }

BaselineParseContext ctx(std::string engine, std::string backend) {
    return {std::move(engine), std::move(backend), "tiny", "tiny-q8_0", "abc", "unit-test", std::nullopt};
}

BaselineCommon common() {
    BaselineCommon c;
    c.model_name = "tiny";
    c.pack = "tiny-q8_0";
    c.model_hash = "precomputed";
    c.host_label = "unit-test";
    c.environment.enabled = false;
    return c;
}

}  // namespace

// ---- version / backend names ---------------------------------------------------------------

TEST(Baseline, ParsesLlamaVersion) {
    const BuildInfo bi = parse_llama_version(slurp(fixture("llama_bench_version.txt")));
    EXPECT_EQ(bi.version, "0.5.0-dev (build 1, commit bd4f514)");
    EXPECT_EQ(bi.build, "1");
    EXPECT_EQ(bi.commit, "bd4f514");
    test::expect_error(ErrorCode::Api, [] { (void)parse_llama_version("usage: ..."); });
    test::expect_error(ErrorCode::Api, [] { (void)parse_llama_version("version: 1.0 (build 2)"); });
    EXPECT_EQ(ggml_backend_name("vulkan"), "Vulkan");
    EXPECT_EQ(ggml_backend_name("hip"), "ROCm");
    EXPECT_EQ(ggml_backend_name("cpu"), "CPU");
    test::expect_error(ErrorCode::Config, [] { (void)ggml_backend_name("cuda"); });
}

// ---- llama-bench -----------------------------------------------------------------------------

TEST(LlamaBench, ArgvForVulkanAndHipConfigs) {
    LlamaBenchConfig v;
    v.binary = "/opt/llama-vulkan/bin/llama-bench";
    v.model = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf";
    v.n_prompt = {512};
    v.n_gen = {128};
    v.n_depth = {0, 32768};
    v.threads = 16;
    EXPECT_EQ(llama_bench_argv(v),
              (std::vector<std::string>{"/opt/llama-vulkan/bin/llama-bench", "-m", "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                                        "-p", "512", "-n", "128", "-d", "0,32768", "-r", "5", "-ngl", "999", "-fa", "on",
                                        "-o", "json", "-t", "16"}));
    LlamaBenchConfig h = v;
    h.backend = "hip";
    h.binary = "/opt/llama-hip/bin/llama-bench";
    h.flash_attn = "off";
    h.no_warmup = true;
    h.extra_args = {"-ub", "256"};
    const auto a = llama_bench_argv(h);
    EXPECT_EQ(a.front(), "/opt/llama-hip/bin/llama-bench");
    EXPECT_EQ(std::vector<std::string>(a.end() - 3, a.end()), (std::vector<std::string>{"--no-warmup", "-ub", "256"}));
    h.flash_attn = "1";  // this commit takes on|off|auto
    test::expect_error(ErrorCode::Config, [&] { (void)llama_bench_argv(h); });
    h.flash_attn = "on";
    h.backend = "metal";
    test::expect_error(ErrorCode::Config, [&] { (void)llama_bench_argv(h); });
}

TEST(LlamaBench, ParsesCapturedFixture) {
    auto c = ctx("llama-bench", "cpu");
    c.expected_commit = "bd4f514";
    const BaselineParse p = parse_llama_bench_json(load("llama_bench_tiny_q8_0.json"), c);
    EXPECT_TRUE(p.failures.empty()) << p.failures.front();
    EXPECT_TRUE(p.notes.empty());
    ASSERT_EQ(p.records.size(), 4u);  // 2 tests x 2 samples
    const auto& pp = p.records[0];
    EXPECT_EQ(pp.mode, "pp16");
    EXPECT_EQ(pp.context, 16u);
    EXPECT_DOUBLE_EQ(*pp.prompt_tps, 1578.73);
    EXPECT_FALSE(pp.decode_tps);
    EXPECT_EQ(pp.quantization, "Q8_0");
    EXPECT_EQ(pp.engine, "llama-bench");
    EXPECT_EQ(pp.suite, "baseline");
    EXPECT_EQ(pp.extra.at("backends"), "CPU");
    EXPECT_EQ(pp.extra.at("n_threads"), 2);
    EXPECT_DOUBLE_EQ(pp.extra.at("sample_ns").get<double>(), 10134731.0);
    EXPECT_EQ(p.records[1].repetition, 1u);
    const auto& tg = p.records[3];
    EXPECT_EQ(tg.mode, "tg8");
    EXPECT_EQ(tg.context, 8u);
    EXPECT_DOUBLE_EQ(*tg.decode_tps, 144.574);
    EXPECT_FALSE(tg.prompt_tps);
}

TEST(LlamaBench, BackendAndCommitMismatchesFail) {
    const auto j = load("llama_bench_tiny_q8_0.json");
    auto c = ctx("llama-bench", "vulkan");  // the captured binary is CPU-only
    BaselineParse p = parse_llama_bench_json(j, c);
    ASSERT_EQ(p.failures.size(), 2u);
    EXPECT_NE(p.failures[0].find("config expects vulkan ('Vulkan')"), std::string::npos) << p.failures[0];
    c = ctx("llama-bench", "cpu");
    c.expected_commit = "deadbee";
    p = parse_llama_bench_json(j, c);
    ASSERT_EQ(p.failures.size(), 2u);
    EXPECT_NE(p.failures[0].find("differs from --version commit"), std::string::npos);
    // A Vulkan build reports "Vulkan" (not the CPU registry name).
    auto vk = j;
    for (auto& t : vk) t["backends"] = "Vulkan";
    EXPECT_TRUE(parse_llama_bench_json(vk, ctx("llama-bench", "vulkan")).failures.empty());
    for (auto& t : vk) t["backends"] = "ROCm,RPC";
    EXPECT_TRUE(parse_llama_bench_json(vk, ctx("llama-bench", "hip")).failures.empty());
}

TEST(LlamaBench, ModesDepthAndCrossChecks) {
    auto j = load("llama_bench_tiny_q8_0.json");
    j[0]["n_depth"] = 4096;
    j[1]["n_prompt"] = 16;  // pg test
    j[0]["avg_ts"] = 2000.0;  // disagrees with mean(samples_ts) by > 1 %
    const auto p = parse_llama_bench_json(j, ctx("llama-bench", "cpu"));
    EXPECT_EQ(p.records[0].mode, "pp16@d4096");
    EXPECT_EQ(p.records[0].context, 4096u + 16u);
    EXPECT_EQ(p.records[2].mode, "pp16+tg8");
    EXPECT_EQ(p.records[2].context, 24u);
    EXPECT_FALSE(p.records[2].decode_tps);
    EXPECT_TRUE(p.records[2].extra.contains("combined_tps"));
    ASSERT_EQ(p.notes.size(), 1u);
    EXPECT_NE(p.notes[0].find("differs from avg_ts"), std::string::npos);
}

TEST(LlamaBench, RejectsMalformedOutput) {
    const auto j = load("llama_bench_tiny_q8_0.json");
    const auto bad = [&](const std::function<void(nlohmann::json&)>& m) {
        nlohmann::json x = j;
        m(x);
        test::expect_error(ErrorCode::Api, [&] { (void)parse_llama_bench_json(x, ctx("llama-bench", "cpu")); });
    };
    bad([](auto& x) { x = nlohmann::json::object(); });
    bad([](auto& x) { x[0].erase("samples_ts"); });
    bad([](auto& x) { x[0]["samples_ts"] = nlohmann::json::array(); });
    bad([](auto& x) { x[0]["samples_ts"][0] = -1.0; });
    bad([](auto& x) { x[0]["samples_ts"][0] = "fast"; });
    bad([](auto& x) { x[0]["samples_ns"] = nlohmann::json::array({1}); });  // length mismatch
    bad([](auto& x) { x[0]["n_prompt"] = -16; });
    bad([](auto& x) { x[0]["n_prompt"] = 0; x[0]["n_gen"] = 0; });
    bad([](auto& x) { x[0]["backends"] = 3; });
    bad([](auto& x) { x[0]["avg_ts"] = nullptr; });
}

TEST(LlamaBench, RunsFakeBinaryEndToEnd) {
    LlamaBenchConfig c;
    c.binary = HALO_CHILD_HELPER;
    c.model = fixture("llama_bench_version.txt");  // any regular file (S-36)
    c.backend = "cpu";
    c.common = common();
    c.common.process.env = {{"HALO_FAKE_MODE", "llama_bench"},
                            {"HALO_FAKE_JSON", fixture("llama_bench_tiny_q8_0.json").string()}};
    SuiteArtifact a = run_llama_bench(c);
    ASSERT_EQ(a.records.size(), 4u) << nlohmann::json(a.notes).dump();
    const auto& r = a.records[0];
    ASSERT_TRUE(r.invocation);
    EXPECT_EQ(r.invocation->version, "0.5.0-dev (build 1, commit bd4f514)");
    EXPECT_EQ(r.invocation->commit, "bd4f514");
    EXPECT_EQ(r.invocation->argv, llama_bench_argv(c));
    EXPECT_EQ(r.invocation->environment.at("HALO_FAKE_MODE"), kRedacted);  // name kept, value not (S-31)
    EXPECT_EQ(r.model_hash, "precomputed");
    EXPECT_TRUE(r.pack_hash.empty());  // a placeholder hash yields no PACK_ID
    EXPECT_EQ(r.power_mode, "unknown|unknown/unknown");  // capture disabled: says so
    EXPECT_EQ(r.driver, "unknown (hardware state not captured)");
    EXPECT_FALSE(a.valid);  // 2 samples < 3 repetitions, and no hardware state
    EXPECT_FALSE(a.conformant);

    // Model hash is streamed from the file when not supplied.
    c.common.model_hash.reset();
    c.model = fixture("llama_bench_version.txt");
    a = run_llama_bench(c);
    ASSERT_FALSE(a.records.empty());
    EXPECT_EQ(a.records[0].model_hash, sha256_file(c.model));
    EXPECT_EQ(a.records[0].pack_hash, make_pack_id(a.records[0].model_hash, std::nullopt));
}

TEST(LlamaBench, ToolFailuresBecomeInvalidArtifacts) {
    LlamaBenchConfig c;
    c.binary = HALO_CHILD_HELPER;
    c.model = fixture("llama_bench_version.txt");  // any regular file (S-36)
    c.backend = "cpu";
    c.common = common();
    const auto run = [&](std::map<std::string, std::string> env) {
        c.common.process.env = std::move(env);
        c.common.process.env["HALO_FAKE_MODE"] = "llama_bench";
        return run_llama_bench(c);
    };
    const auto has = [](const SuiteArtifact& a, std::string_view s) {
        for (const auto& n : a.notes) {
            if (n.find(s) != std::string::npos) return true;
        }
        return false;
    };
    auto a = run({{"HALO_FAKE_JSON", fixture("llama_bench_tiny_q8_0.json").string()}, {"HALO_FAKE_CODE", "3"}});
    EXPECT_TRUE(has(a, "FAILED: llama-bench: failed (exit 3"));
    EXPECT_TRUE(a.records.empty());
    EXPECT_FALSE(a.valid);
    a = run({{"HALO_FAKE_JSON", fixture("llama_bench_version.txt").string()}});  // not JSON
    EXPECT_TRUE(has(a, "FAILED: llama-bench: unparsable output"));
    a = run({{"HALO_FAKE_JSON", fixture("llama_bench_tiny_q8_0.json").string()}, {"HALO_FAKE_VERSION", "garbage"}});
    EXPECT_TRUE(has(a, "--version"));
    c.backend = "vulkan";
    a = run({{"HALO_FAKE_JSON", fixture("llama_bench_tiny_q8_0.json").string()}});
    EXPECT_TRUE(has(a, "config expects vulkan"));
    EXPECT_FALSE(a.valid);
    c.common.host_label.clear();
    test::expect_error(ErrorCode::Config, [&] { (void)run({}); });
    c.common.host_label = "unit-test";
    c.binary = "relative/llama-bench";
    test::expect_error(ErrorCode::Config, [&] { (void)run({}); });
}

// ---- llama-server ----------------------------------------------------------------------------

TEST(LlamaServer, ArgvFollowsEngineReportMtpFlags) {
    LlamaServerConfig c;
    c.binary = "/opt/llama-vulkan/bin/llama-server";
    c.model = "/models/m.gguf";
    c.port = 18084;
    c.ctx_size = 131072;
    c.parallel = 8;
    c.extra_args = {"--kv-unified", "-b", "512", "-ub", "256"};
    EXPECT_EQ(llama_server_argv(c),
              (std::vector<std::string>{"/opt/llama-vulkan/bin/llama-server", "-m", "/models/m.gguf", "-ngl", "999",
                                        "--ctx-size", "131072", "--parallel", "8", "--flash-attn", "on", "--spec-type",
                                        "draft-mtp", "--spec-draft-n-max", "2", "--host", "127.0.0.1", "--port",
                                        "18084", "--no-webui", "--kv-unified", "-b", "512", "-ub", "256"}));
    c.mtp = false;
    const auto a = llama_server_argv(c);
    EXPECT_EQ(std::find(a.begin(), a.end(), "--spec-type"), a.end());
    c.port = 0;
    test::expect_error(ErrorCode::Config, [&] { (void)llama_server_argv(c); });
}

TEST(LlamaServer, ParsesCapturedCompletions) {
    auto p = parse_llama_server_completion(load("llama_server_completion_tiny.json"), ctx("llama-server", "cpu"), false);
    ASSERT_EQ(p.records.size(), 1u);
    EXPECT_TRUE(p.failures.empty());
    const auto& r = p.records[0];
    EXPECT_EQ(r.mode, "completion");
    EXPECT_DOUBLE_EQ(*r.prompt_tps, 136.5161089008503);
    EXPECT_DOUBLE_EQ(*r.decode_tps, 48.21169064624327);
    EXPECT_FALSE(r.decode_effective_tps);
    EXPECT_FALSE(r.mtp_acceptance);
    EXPECT_FALSE(r.ttft_ms);  // not exposed by non-streaming /completion; never substituted
    EXPECT_EQ(r.context, 15u);

    p = parse_llama_server_completion(load("llama_server_completion_mtp_tiny.json"), ctx("llama-server", "cpu"), true);
    EXPECT_TRUE(p.failures.empty());
    const auto& m = p.records[0];
    EXPECT_EQ(m.mode, "mtp_completion");
    EXPECT_FALSE(m.decode_tps);
    EXPECT_DOUBLE_EQ(*m.decode_effective_tps, 32.02630427124144);
    ASSERT_TRUE(m.mtp_acceptance);
    EXPECT_EQ(*m.mtp_acceptance, 0.0);  // 0 of 27 accepted: a MEASURED zero, not null
    EXPECT_EQ(m.extra.at("draft_n"), 27);

    // MTP requested but the server did not speculate.
    p = parse_llama_server_completion(load("llama_server_completion_tiny.json"), ctx("llama-server", "cpu"), true);
    ASSERT_EQ(p.failures.size(), 1u);
    // draft_n == 0: acceptance undefined -> null, with a note.
    auto z = load("llama_server_completion_mtp_tiny.json");
    z["timings"]["draft_n"] = 0;
    z["timings"]["draft_n_accepted"] = 0;
    p = parse_llama_server_completion(z, ctx("llama-server", "cpu"), true);
    EXPECT_FALSE(p.records[0].mtp_acceptance);
    EXPECT_EQ(p.notes.size(), 1u);
    z["timings"]["draft_n_accepted"] = 5;  // more accepted than drafted
    test::expect_error(ErrorCode::Api, [&] { (void)parse_llama_server_completion(z, ctx("llama-server", "cpu"), true); });
    auto bad = load("llama_server_completion_tiny.json");
    bad["timings"]["predicted_per_second"] = "fast";
    test::expect_error(ErrorCode::Api, [&] { (void)parse_llama_server_completion(bad, ctx("llama-server", "cpu"), false); });
    bad.erase("timings");
    test::expect_error(ErrorCode::Api, [&] { (void)parse_llama_server_completion(bad, ctx("llama-server", "cpu"), false); });
}

// ---- Ollama ----------------------------------------------------------------------------------

TEST(Ollama, ParsesDocumentedResponseInNanoseconds) {
    const auto p = parse_ollama_generate(load("ollama_generate.json"), ctx("ollama", "ollama"));
    ASSERT_EQ(p.records.size(), 1u);
    const auto& r = p.records[0];
    EXPECT_DOUBLE_EQ(*r.decode_tps, 50.0);   // 100 tokens / 2e9 ns
    EXPECT_DOUBLE_EQ(*r.prompt_tps, 200.0);  // 26 tokens / 1.3e8 ns
    EXPECT_DOUBLE_EQ(r.extra.at("load_duration_s").get<double>(), 0.012);
    EXPECT_EQ(r.context, 126u);
    EXPECT_EQ(r.mode, "generate");
    auto j = load("ollama_generate.json");
    j["done"] = false;
    test::expect_error(ErrorCode::Api, [&] { (void)parse_ollama_generate(j, ctx("ollama", "ollama")); });
    j = load("ollama_generate.json");
    j["eval_duration"] = -5;
    test::expect_error(ErrorCode::Api, [&] { (void)parse_ollama_generate(j, ctx("ollama", "ollama")); });
}

TEST(Ollama, RunsAgainstAFakeServer) {
    const std::string gen = slurp(fixture("ollama_generate.json"));
    const std::string ver = slurp(fixture("ollama_version.json"));
    const std::string tags = slurp(fixture("ollama_tags.json"));
    FakeHttpServer srv([&](const std::string& m, const std::string& p, const std::string&) {
        if (m == "GET" && p == "/api/version") return FakeHttpServer::json(200, ver);
        if (m == "GET" && p == "/api/tags") return FakeHttpServer::json(200, tags);
        if (m == "POST" && p == "/api/generate") return FakeHttpServer::json(200, gen);
        return FakeHttpServer::json(404, "{}");
    });
    OllamaConfig c;
    c.port = srv.port();
    c.model = "llama3.2:latest";
    c.common = common();
    c.common.model_hash.reset();
    const SuiteArtifact a = run_ollama(c);
    EXPECT_EQ(a.records.size(), 4u) << nlohmann::json(a.notes).dump();  // 1 warm-up + 3
    EXPECT_EQ(a.records[0].phase, Phase::Cold);
    EXPECT_EQ(a.records[0].model_hash,
              "ollama-digest:0000000000000000000000000000000000000000000000000000000000000000");
    EXPECT_EQ(a.records[0].invocation->version, "0.17.6");
    const auto reqs = srv.requests();
    ASSERT_EQ(reqs.size(), 6u);
    const auto body = nlohmann::json::parse(reqs[2].substr(std::string("POST /api/generate ").size()));
    EXPECT_EQ(body.at("stream"), false);
    EXPECT_EQ(body.at("options").at("temperature"), 0.0);
    EXPECT_TRUE(a.conformant);  // 3 steady repetitions

    c.model = "not-pulled";
    const SuiteArtifact missing = run_ollama(c);
    EXPECT_TRUE(missing.records.empty());
    EXPECT_FALSE(missing.valid);
    c.host = "192.168.1.10";  // loopback only
    EXPECT_FALSE(run_ollama(c).notes.empty());
}

// ---- aggregates ------------------------------------------------------------------------------

TEST(Baseline, AggregateIsTheMedianPerConfiguration) {
    SuiteArtifact a;
    a.suite = "baseline";
    for (double v : {10.0, 20.0, 30.0}) {  // first = 10, last = 30, median = 20
        BenchmarkRecord r;
        r.suite = "baseline";
        r.engine = "llama-bench";
        r.mode = "tg128";
        r.context = 128;
        r.decode_tps = v;
        r.repetition = static_cast<std::uint32_t>(a.records.size());
        a.records.push_back(r);
    }
    BenchmarkRecord other = a.records[0];
    other.mode = "pp512";
    other.decode_tps.reset();
    other.prompt_tps = 500.0;
    a.records.push_back(other);
    a.summaries = summarize(a.records);
    const auto agg = aggregate_records(a);
    ASSERT_EQ(agg.size(), 2u);
    const auto& tg = agg[0].mode == "tg128" ? agg[0] : agg[1];
    EXPECT_DOUBLE_EQ(*tg.decode_tps, 20.0);  // the median: neither the first nor the last sample
    EXPECT_EQ(tg.repetition, 0u);
    EXPECT_EQ(tg.extra.at("aggregate").at("method"), "median");
    EXPECT_EQ(tg.extra.at("aggregate").at("n"), 3);
    EXPECT_DOUBLE_EQ(tg.extra.at("aggregate").at("metrics").at("decode_tps").at("stats").at("max").get<double>(), 30.0);
}
