// Milestone 4: integration of the separately tested halves on the tiny model.
//   (1) `halo run` end to end (tools/halo run_cli in-process) == transformers golden
//   (2) ApiServer over the real Engine on 127.0.0.1: completions / chat / messages,
//       streaming == non-streaming, disconnect cancels, 4 concurrent == 4 sequential
//   (3) `halo bench model` over the real Engine -> a valid-schema artifact (D-001: no
//       performance claim is made or checked)
//   (4) TRD §64: tune into a temp profile DB, next launch applies the persisted winner
//   (5) json_schema structured output through the API, validated independently
//
// Kept small for ASan: short generations, one shared engine + server for the API tests.

#include <gtest/gtest.h>
#include <httplib.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "../models/tiny_golden.h"
#include "cli.h"
#include "halo/api/server.h"
#include "halo/autotune/cpu_ops.h"
#include "halo/autotune/db.h"
#include "halo/autotune/lookup.h"
#include "halo/autotune/tuner.h"
#include "halo/hardware/hardware.h"
#include "halo/models/qwen35.h"
#include "halo/profiling/hw_state.h"
#include "halo/profiling/record.h"
#include "halo/runtime/cpu_engine.h"
#include "halo/runtime/profile.h"
#include "halo/tokenizer/tokenizer.h"

using halo::runtime::CpuEngineOptions;
using halo::runtime::Engine;
using halo::runtime::EngineConfig;
using halo::runtime::TickInfo;
using halo::test::Golden;
using Json = nlohmann::json;
using Toks = std::vector<std::int32_t>;
namespace fs = std::filesystem;

namespace {

fs::path tiny() { return halo::test::tiny_dir() / "tiny-f32.gguf"; }
/// The profile-key tests hash and load the model several times; the 102 MB Q4_0 file keeps
/// them well under the ASan budget (the 540 MB F32 file took ~2400 s under ctest -j). They
/// assert tuning plumbing, not golden numerics.
fs::path tiny_q4() { return halo::test::tiny_dir() / "tiny-q4_0.gguf"; }

std::unique_ptr<halo::tokenizer::Tokenizer> tiny_tokenizer() {
    const auto nm = halo::model::NormalizedModel::load(tiny(), halo::model::GgufMode::HeaderOnly);
    return std::make_unique<halo::tokenizer::Tokenizer>(
        halo::tokenizer::Tokenizer::from_spec(halo::models::vocab_spec(nm.tokenizer())));
}

Toks prefix(const Toks& t, std::size_t n) { return Toks(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(std::min(n, t.size()))); }

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& tag) {
        path = fs::temp_directory_path() / ("halo-wsg-" + tag + "-" + std::to_string(::getpid()));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

/// Shared fixture data: golden + tokenizer (skip if the reference data is missing).
class Fx : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        golden_ = Golden::load();
        if (golden_ && fs::exists(tiny())) tok_ = tiny_tokenizer();
    }
    static void TearDownTestSuite() { tok_.reset(); }
    void SetUp() override {
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        if (!tok_) GTEST_SKIP() << "tiny-f32.gguf missing under " << halo::test::tiny_dir();
    }
    static Toks prompt(const char* p) { return golden_->i32(std::string(p) + ".tokens"); }
    static Toks gold(const char* p) { return golden_->i32(std::string(p) + ".decode_tokens"); }
    /// The prompt as text, checked to tokenize back to exactly the golden ids.
    static std::string prompt_text(const Toks& t) {
        std::string s = tok_->decode(t, false);
        EXPECT_EQ(tok_->encode(s, true), t) << "golden prompt does not round-trip through text";
        return s;
    }
    static inline std::optional<Golden> golden_;
    static inline std::unique_ptr<halo::tokenizer::Tokenizer> tok_;
};

EngineConfig tiny_cfg() {
    EngineConfig c;
    c.model_path = tiny().string();
    c.max_context = 1024;
    c.max_sequences = 4;
    c.mtp_max_draft = 2;
    c.prefix_cache = false;
    return c;
}

}  // namespace

// ---- SHA-256 known answers (the profile key hashes the model with it) -------------------------

TEST(RuntimeProfile, Sha256KnownAnswers) {
    using halo::runtime::sha256_hex;
    EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    EXPECT_EQ(sha256_hex(std::string(1000000, 'a')), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    TempDir d("sha");
    const auto f = d.path / "x.bin";
    std::ofstream(f, std::ios::binary) << std::string(5u << 20, 'a');  // crosses the 4 MiB read buffer
    EXPECT_EQ(halo::runtime::sha256_file_hex(f), sha256_hex(std::string(5u << 20, 'a')));
    EXPECT_EQ(halo::runtime::pack_id("t", std::nullopt), sha256_hex("halo.pack/1\ntrunk=t\nmtp=none\n"));
    EXPECT_THROW((void)halo::runtime::sha256_file_hex(d.path / "missing"), halo::Error);
}

// ---- (1) halo run -------------------------------------------------------------------------

TEST_F(Fx, HaloRunGreedyOutputEqualsGolden) {
    const Toks p = prompt("p1");
    const Toks g = gold("p1");
    const std::string text = prompt_text(p);
    std::ostringstream out, err;
    std::istringstream in;
    halo::cli::Context ctx;
    ctx.out = &out;
    ctx.err = &err;
    ctx.in = &in;
    ctx.engine_factory = halo::cli::default_engine_factory();
    ASSERT_TRUE(ctx.engine_factory) << "this CLI build does not link halo_runtime";
    const int rc = halo::cli::run_cli({"run", tiny().string(), "--raw", "-p", text, "--temperature", "0", "-n",
                                       std::to_string(g.size()), "--ctx", "512"},
                                      ctx);
    ASSERT_EQ(rc, 0) << err.str();
    EXPECT_EQ(out.str(), tok_->decode(g, false) + "\n");
    EXPECT_NE(err.str().find("generated " + std::to_string(g.size()) + " tokens"), std::string::npos) << err.str();
    EXPECT_NE(err.str().find("finish length"), std::string::npos) << err.str();
}

// ---- (2) + (5) ApiServer over the real engine ----------------------------------------------------

namespace {

int ApiPort();

class Api : public Fx {
protected:
    static void SetUpTestSuite() {
        Fx::SetUpTestSuite();
        if (!golden_ || !tok_) return;
        CpuEngineOptions o;
        o.gate_mode = halo::speculative::GateMode::Always;  // the MTP path runs under the API too
        o.on_tick = [](const TickInfo& t) {
            if (t.sequences > max_tick_seqs_.load()) max_tick_seqs_ = t.sequences;
        };
        engine_ = halo::runtime::create_cpu_engine(tiny_cfg(), o);
        halo::api::ServerConfig sc;
        sc.port = 0;
        sc.max_concurrent = 4;
        sc.request_timeout = std::chrono::seconds(0);
        sc.read_timeout = std::chrono::seconds(600);
        sc.write_timeout = std::chrono::seconds(600);
        server_ = std::make_unique<halo::api::ApiServer>(*engine_, sc);
        server_->start();
        port_ = server_->port();
    }
    static void TearDownTestSuite() {
        if (server_) server_->stop();
        server_.reset();
        engine_.reset();
        Fx::TearDownTestSuite();
    }
    static std::unique_ptr<httplib::Client> client() {
        auto c = std::make_unique<httplib::Client>("127.0.0.1", port_);
        c->set_connection_timeout(std::chrono::seconds(10));
        c->set_read_timeout(std::chrono::seconds(900));  // ASan: a 12-token generation can take minutes
        c->set_write_timeout(std::chrono::seconds(60));
        c->set_keep_alive(false);
        return c;
    }
    static Json post(const std::string& path, const Json& body) {
        auto r = client()->Post(path, body.dump(), "application/json");
        EXPECT_TRUE(r) << "no HTTP response for " << path;
        if (!r) return Json();
        EXPECT_EQ(r->status, 200) << r->body;
        return Json::parse(r->body);
    }
    /// SSE body -> the JSON of every data event except [DONE].
    static std::vector<Json> post_stream(const std::string& path, Json body) {
        body["stream"] = true;
        auto r = client()->Post(path, body.dump(), "application/json");
        EXPECT_TRUE(r) << "no HTTP response for " << path;
        std::vector<Json> out;
        if (!r) return out;
        EXPECT_EQ(r->status, 200) << r->body;
        std::istringstream s(r->body);
        for (std::string line; std::getline(s, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.starts_with("data: ")) continue;
            const std::string d = line.substr(6);
            if (d == "[DONE]") continue;
            out.push_back(Json::parse(d));
        }
        return out;
    }
    static Json completion_body(const std::string& text, std::size_t n) {
        return Json{{"prompt", text}, {"max_tokens", n}, {"temperature", 0}};
    }
    static std::string complete(const std::string& text, std::size_t n) {
        const Json j = post("/v1/completions", completion_body(text, n));
        return j.is_null() ? std::string() : j.at("choices").at(0).at("text").get<std::string>();
    }
    static Json chat_body(std::size_t n) {
        return Json{{"messages", Json::array({{{"role", "user"}, {"content", "hello there"}}})},
                    {"max_tokens", n},
                    {"temperature", 0}};
    }

    static inline std::unique_ptr<Engine> engine_;
    static inline std::unique_ptr<halo::api::ApiServer> server_;
    static inline int port_ = 0;
    static inline std::atomic<std::size_t> max_tick_seqs_{0};
    friend int ApiPort();
};

int ApiPort() { return Api::port_; }

}  // namespace

TEST_F(Api, CompletionsGreedyEqualsGoldenStreamingAndNot) {
    const Toks g = gold("p1");
    const std::string text = prompt_text(prompt("p1"));
    const std::string want = tok_->decode(prefix(g, 10), false);
    EXPECT_EQ(complete(text, 10), want);
    std::string streamed;
    for (const Json& ev : post_stream("/v1/completions", completion_body(text, 10))) {
        if (ev.contains("choices") && !ev["choices"].empty()) streamed += ev["choices"][0].value("text", "");
    }
    EXPECT_EQ(streamed, want);
}

TEST_F(Api, ChatCompletionsStreamingEqualsNonStreaming) {
    const Json j = post("/v1/chat/completions", chat_body(12));
    ASSERT_FALSE(j.is_null());
    const Json& msg = j.at("choices").at(0).at("message");
    const std::string content = msg.value("content", Json("")).is_string() ? msg.value("content", "") : "";
    const std::string reasoning = msg.value("reasoning_content", "");
    EXPECT_FALSE(content.empty() && reasoning.empty()) << j.dump();
    EXPECT_EQ(j.at("choices").at(0).at("finish_reason"), "length");
    EXPECT_EQ(j.at("usage").at("completion_tokens"), 12);
    std::string sc, sr, finish;
    for (const Json& ev : post_stream("/v1/chat/completions", chat_body(12))) {
        if (!ev.contains("choices") || ev["choices"].empty()) continue;
        const Json& d = ev["choices"][0].value("delta", Json::object());
        if (d.contains("content") && d["content"].is_string()) sc += d["content"].get<std::string>();
        if (d.contains("reasoning_content") && d["reasoning_content"].is_string()) sr += d["reasoning_content"].get<std::string>();
        if (ev["choices"][0].contains("finish_reason") && ev["choices"][0]["finish_reason"].is_string()) {
            finish = ev["choices"][0]["finish_reason"];
        }
    }
    EXPECT_EQ(sc, content);
    EXPECT_EQ(sr, reasoning);
    EXPECT_EQ(finish, "length");
}

TEST_F(Api, AnthropicMessagesStreamingEqualsNonStreaming) {
    Json body = chat_body(12);
    body["model"] = engine_->model().id;
    const Json j = post("/v1/messages", body);
    ASSERT_FALSE(j.is_null());
    std::string text, thinking;
    for (const Json& b : j.at("content")) {
        if (b.at("type") == "text") text += b.at("text").get<std::string>();
        if (b.at("type") == "thinking") thinking += b.at("thinking").get<std::string>();
    }
    EXPECT_FALSE(text.empty() && thinking.empty()) << j.dump();
    EXPECT_EQ(j.at("stop_reason"), "max_tokens");
    std::string st, sth;
    for (const Json& ev : post_stream("/v1/messages", body)) {
        if (ev.value("type", "") != "content_block_delta") continue;
        const Json& d = ev.at("delta");
        if (d.value("type", "") == "text_delta") st += d.at("text").get<std::string>();
        if (d.value("type", "") == "thinking_delta") sth += d.at("thinking").get<std::string>();
    }
    EXPECT_EQ(st, text);
    EXPECT_EQ(sth, thinking);
}

TEST_F(Api, ClientDisconnectMidStreamCancelsTheSequence) {
    const std::string body =
        Json{{"prompt", prompt_text(prompt("p1"))}, {"max_tokens", 400}, {"temperature", 0}, {"stream", true}}.dump();
    const auto gen0 = engine_->stats().tokens_generated;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(fd, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<std::uint16_t>(port_));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)), 0);
    const std::string req = "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                            "Connection: close\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body;
    ASSERT_EQ(::send(fd, req.data(), req.size(), 0), static_cast<ssize_t>(req.size()));
    std::string got;
    char buf[4096];
    while (got.find("data: {") == std::string::npos) {  // the first streamed token
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        ASSERT_GT(n, 0) << "stream ended before the first token: " << got;
        got.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);  // the client goes away mid-stream
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(900);
    while (engine_->stats().active_sequences != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_EQ(engine_->stats().active_sequences, 0u) << "the sequence was not retired";
    const auto generated = engine_->stats().tokens_generated - gen0;
    EXPECT_LT(generated, 200u) << "the engine kept decoding after the client disconnected";
    std::printf("disconnect: %llu tokens generated before retirement (max_tokens 400)\n",
                static_cast<unsigned long long>(generated));
    // The server still serves.
    EXPECT_EQ(complete(prompt_text(prompt("p1")), 3), tok_->decode(prefix(gold("p1"), 3), false));
}

TEST_F(Api, FourConcurrentHttpClientsGetTheSequentialGreedyTokens) {
    const Toks p0 = prompt("p0");
    const std::vector<std::string> prompts = {
        prompt_text(Toks(p0.begin(), p0.begin() + 40)), prompt_text(prompt("p1")), prompt_text(prompt("p2")),
        prompt_text(Toks(p0.end() - 30, p0.end()))};
    std::vector<std::string> seq;
    for (const auto& p : prompts) seq.push_back(complete(p, 8));
    max_tick_seqs_ = 0;
    std::vector<std::string> con(4);
    std::barrier start(4);
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < 4; ++i) {
        th.emplace_back([&, i] {
            start.arrive_and_wait();
            con[i] = complete(prompts[i], 8);
        });
    }
    for (auto& t : th) t.join();
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FALSE(seq[i].empty()) << i;
        EXPECT_EQ(con[i], seq[i]) << i;
    }
    EXPECT_GE(max_tick_seqs_.load(), 2u) << "the HTTP requests never shared an engine tick";
}

namespace {

/// Independent validation of {"color": "red"|"green"|"blue"} with no other keys.
std::string schema_violation(const std::string& text) {
    Json j;
    try {
        j = Json::parse(text);
    } catch (const std::exception& e) {
        return std::string("not JSON: ") + e.what();
    }
    if (!j.is_object()) return "not an object";
    if (j.size() != 1 || !j.contains("color")) return "keys must be exactly {color}";
    if (!j["color"].is_string()) return "color must be a string";
    static const std::set<std::string> ok = {"red", "green", "blue"};
    if (!ok.contains(j["color"].get<std::string>())) return "color not in the enum";
    return {};
}

}  // namespace

/// One json_schema chat request; returns (content, full response).
std::pair<std::string, Json> schema_chat(bool thinking_off) {
    const Json schema = {{"type", "object"},
                         {"properties", {{"color", {{"type", "string"}, {"enum", {"red", "green", "blue"}}}}}},
                         {"required", {"color"}},
                         {"additionalProperties", false}};
    Json body = {{"messages", Json::array({{{"role", "user"}, {"content", "hello there"}}})},
                 {"max_tokens", 160},
                 {"temperature", 0},
                 {"response_format", {{"type", "json_schema"}, {"json_schema", {{"name", "pick"}, {"schema", schema}}}}}};
    if (thinking_off) body["chat_template_kwargs"] = {{"enable_thinking", false}};
    auto c = std::make_unique<httplib::Client>("127.0.0.1", ApiPort());
    c->set_read_timeout(std::chrono::seconds(900));
    auto r = c->Post("/v1/chat/completions", body.dump(), "application/json");
    if (!r) return {"", Json()};
    EXPECT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    const Json& msg = j.at("choices").at(0).at("message");
    const std::string content = msg.value("content", Json("")).is_string() ? msg.value("content", "") : "";
    std::printf("[%s] finish %s, content %s, reasoning %s\n", thinking_off ? "no-think" : "think",
                j.at("choices").at(0).at("finish_reason").dump().c_str(), Json(content).dump().c_str(),
                Json(msg.value("reasoning_content", "")).dump().c_str());
    return {content, j};
}

TEST_F(Api, JsonSchemaStructuredOutputValidatesAgainstTheSchema) {
    const auto [content, j] = schema_chat(true);
    ASSERT_FALSE(j.is_null());
    EXPECT_EQ(schema_violation(content), "") << j.dump();
    EXPECT_EQ(j.at("choices").at(0).at("finish_reason"), "stop") << "the grammar completed, then EOS";
}

// Finding F-1 (src/api, WS-I): with thinking on (the Qwen template default) the generation
// prompt opens <think>, the API's output parser starts in reasoning mode, and the grammar
// forbids the UserDefined </think> token, so the schema-valid JSON must still be returned as
// `content`, not reasoning_content. Fixed (see the WS-G M4 report; commit e6ce83e).
TEST_F(Api, JsonSchemaWithThinkingOnReturnsTheJsonAsContent) {
    const auto [content, j] = schema_chat(false);
    ASSERT_FALSE(j.is_null());
    EXPECT_EQ(schema_violation(content), "") << j.dump();
}

// ---- (3) halo bench model ------------------------------------------------------------------------

TEST_F(Fx, HaloBenchModelWritesAValidSchemaArtifact) {
    TempDir d("bench");
    const auto out_path = d.path / "artifact.json";
    std::ostringstream out, err;
    std::istringstream in;
    halo::cli::Context ctx;
    ctx.out = &out;
    ctx.err = &err;
    ctx.in = &in;
    ctx.engine_factory = halo::cli::default_engine_factory();
    const int rc = halo::cli::run_cli({"bench", "model", tiny().string(), "--host-label", "dev-host", "--power-mode", "dev-host",
                                       "--out", out_path.string(), "--modes", "prompt,decode,concurrency", "--contexts", "64",
                                       "--concurrency", "1,2", "--prompt-tokens", "16", "--decode-tokens", "4",
                                       "--repetitions", "3", "--ctx", "512", "--mtp-draft", "0"},
                                      ctx);
    std::printf("%s", err.str().c_str());
    ASSERT_EQ(rc, 0) << err.str();
    std::ifstream f(out_path);
    ASSERT_TRUE(f.good());
    const Json a = Json::parse(f);
    EXPECT_EQ(a.at("schema"), "halo.bench.artifact/1");
    EXPECT_EQ(a.at("suite"), "model");
    ASSERT_TRUE(a.at("records").is_array());
    ASSERT_FALSE(a.at("records").empty());
    std::set<std::string> modes;
    for (const Json& r : a.at("records")) {
        halo::profiling::BenchmarkRecord rec;
        EXPECT_NO_THROW(halo::profiling::from_json(r, rec)) << r.dump();  // record schema check
        modes.insert(r.at("mode").get<std::string>());
    }
    EXPECT_TRUE(modes.contains("prompt") && modes.contains("decode")) << Json(modes).dump();
    // Concurrency runs at ModelSuiteConfig::concurrency_context (4096), which `halo bench
    // model` cannot set (finding F-2): on a 512-token engine the harness must say it skipped
    // the mode instead of producing nothing silently.
    bool concurrency_noted = modes.contains("concurrency");
    for (const Json& n : a.at("notes")) {
        if (n.get<std::string>().starts_with("concurrency:")) concurrency_noted = true;
    }
    EXPECT_TRUE(concurrency_noted) << a.at("notes").dump();
    for (const char* k : {"hardware_before", "hardware_after", "thermal", "power_mode_pinned", "conformant", "valid", "notes",
                          "summaries"}) {
        EXPECT_TRUE(a.contains(k)) << k;
    }
    // D-001: the dev-host artifact is reported as it is; no number is asserted.
    std::printf("bench artifact: %zu records, conformant %s, valid %s\n", a.at("records").size(),
                a.at("conformant").dump().c_str(), a.at("valid").dump().c_str());
}

// ---- (4) TRD §64: tune -> persist -> next launch -----------------------------------------------

namespace {

halo::autotune::TuneOptions tune_options() {
    halo::autotune::TuneOptions o;
    o.strategy = halo::autotune::Strategy::Exhaustive;
    o.stability.max_cv = 1e9;  // tests the chain, not stability (ctest -j loads the host)
    return o;
}

}  // namespace

TEST_F(Fx, ProfileDbWinnerIsAppliedOnTheNextLaunch) {
    TempDir d("profile");
    const auto db_path = d.path / "profiles.db";
    if (!fs::exists(tiny_q4())) GTEST_SKIP() << "tiny-q4_0.gguf missing under " << halo::test::tiny_dir();
    EngineConfig cfg = tiny_cfg();
    cfg.model_path = tiny_q4().string();
    cfg.max_sequences = 1;
    cfg.mtp_enabled = false;  // ASan budget: no draft head passes (MTP is not what this tests)
    cfg.profile_db = db_path.string();
    cfg.platform_power_mode = "dev-host";
    const auto nm = halo::model::NormalizedModel::load(cfg.model_path, halo::model::GgufMode::HeaderOnly);
    const auto& hp = nm.hparams();
    const std::size_t default_threads = halo::cpu::ThreadPool::default_threads();
    // Single-candidate tunes: the winner is deterministic and differs from the defaults
    // (64-token GDN chunk; default thread count), so the test cannot pass by coincidence.
    const unsigned tuned_threads = default_threads == 1 ? 2u : 1u;
    const unsigned tuned_chunk = 16;
    // Before any tune: a missing database is "no profiles", reported, defaults kept.
    {
        auto e = halo::runtime::create_cpu_engine(cfg);
        const auto s = e->stats();
        EXPECT_EQ(s.threads, default_threads);
        EXPECT_EQ(s.gdn_chunk, 64u);
        EXPECT_NE(s.tuning.find("none"), std::string::npos) << s.tuning;
        EXPECT_NE(s.tuning.find("no profile database"), std::string::npos) << s.tuning;
    }
    // Offline tune (the `halo tune` side) with the SAME key and op keys the engine builds.
    const halo::autotune::ProfileKey key = halo::runtime::engine_profile_key(cfg);
    // The key is re-derived independently from its documented recipe (profile.h), so a
    // change on the engine side alone cannot pass by also changing the tuner's key.
    {
        halo::hardware::DiscoveryOptions dopt;
        dopt.root = "/";
        const auto hw = halo::profiling::capture_hardware_state(dopt, "dev-host");
        const std::string trunk = halo::runtime::sha256_file_hex(cfg.model_path);  // KAT-verified above
        const std::string pack = halo::runtime::sha256_hex("halo.pack/1\ntrunk=" + trunk + "\nmtp=none\n");
        const auto simd = halo::hardware::runtime_simd_flags();
        const std::string isa = simd && simd->avx512f ? "x86-64-avx512" : simd && simd->avx2 ? "x86-64-avx2" : "x86-64";
        halo::autotune::ProfileKey want = halo::autotune::make_profile_key(hw, trunk, pack, isa);
        EXPECT_EQ(halo::autotune::match_keys(want, key).kind, halo::autotune::MatchKind::Exact)
            << "engine key deviates from the documented recipe";
    }
    halo::autotune::CpuMatmulTunable mm(1, hp.n_embd, hp.n_ff, {tuned_threads});
    const halo::cpu::GdnDims dims{hp.gdn_n_k_heads, hp.gdn_n_v_heads, hp.gdn_head_k_dim, hp.gdn_head_v_dim,
                                  halo::cpu::GdnHeadMapping::Tiled};
    halo::autotune::CpuGdnChunkedTunable gdn(dims, halo::runtime::kGdnTuneTokens, {tuned_chunk}, {tuned_threads});
    ASSERT_EQ(mm.key(), halo::runtime::matmul_op_key(hp)) << "tuner and runtime disagree on the MATMUL op key";
    ASSERT_EQ(gdn.key(), halo::runtime::gdn_op_key(hp)) << "tuner and runtime disagree on the GDN op key";
    {
        halo::autotune::ProfileDb db = halo::autotune::ProfileDb::open(db_path);
        halo::autotune::TunableOp* ops[] = {&mm, &gdn};
        const auto report = halo::autotune::tune(db, key, ops, tune_options());
        ASSERT_EQ(report.ops.size(), 2u);
        for (const auto& r : report.ops) ASSERT_TRUE(r.persisted) << r.key.family << ": " << r.note;
    }
    // Next launch: the persisted winners are applied.
    {
        auto e = halo::runtime::create_cpu_engine(cfg);
        const auto s = e->stats();
        EXPECT_EQ(s.threads, tuned_threads) << s.tuning;
        EXPECT_EQ(s.gdn_chunk, tuned_chunk) << s.tuning;
        EXPECT_NE(s.tuning.find("MATMUL=exact"), std::string::npos) << s.tuning;
        EXPECT_NE(s.tuning.find("GATED_DELTANET=exact"), std::string::npos) << s.tuning;
        // The tuned engine still computes the same function as an untuned one on this file:
        // p0's 150-token prefill crosses nine 16-token GDN chunks (chunk 16 vs 64: fp32-close,
        // same greedy tokens).
        halo::runtime::GenerateRequest r;
        r.prompt = prompt("p0");
        r.sampling.temperature = 0.0f;
        r.max_tokens = 2;
        EngineConfig plain = cfg;
        plain.profile_db.reset();
        EXPECT_EQ(e->generate(r, {}).tokens, halo::runtime::create_cpu_engine(plain)->generate(r, {}).tokens);
    }
}

TEST_F(Fx, ProfileMismatchesAndOverridesKeepDefaultsVisibly) {
    TempDir d("profile2");
    const auto db_path = d.path / "profiles.db";
    if (!fs::exists(tiny_q4())) GTEST_SKIP() << "tiny-q4_0.gguf missing under " << halo::test::tiny_dir();
    EngineConfig cfg = tiny_cfg();
    cfg.model_path = tiny_q4().string();
    cfg.max_sequences = 1;
    cfg.mtp_enabled = false;
    cfg.profile_db = db_path.string();
    cfg.platform_power_mode = "dev-host";
    const auto nm = halo::model::NormalizedModel::load(cfg.model_path, halo::model::GgufMode::HeaderOnly);
    const auto& hp = nm.hparams();
    const std::size_t default_threads = halo::cpu::ThreadPool::default_threads();
    const unsigned tuned_threads = default_threads == 1 ? 2u : 1u;
    const unsigned tuned_chunk = 16;
    {
        halo::autotune::CpuMatmulTunable mm(1, hp.n_embd, hp.n_ff, {tuned_threads});
        const halo::cpu::GdnDims dims{hp.gdn_n_k_heads, hp.gdn_n_v_heads, hp.gdn_head_k_dim, hp.gdn_head_v_dim,
                                      halo::cpu::GdnHeadMapping::Tiled};
        halo::autotune::CpuGdnChunkedTunable gdn(dims, halo::runtime::kGdnTuneTokens, {tuned_chunk}, {tuned_threads});
        halo::autotune::ProfileDb db = halo::autotune::ProfileDb::open(db_path);
        halo::autotune::TunableOp* ops[] = {&mm, &gdn};
        for (const auto& r : halo::autotune::tune(db, halo::runtime::engine_profile_key(cfg), ops, tune_options()).ops) {
            ASSERT_TRUE(r.persisted) << r.note;
        }
    }
    // Another power mode: the must-match key field differs, nothing applies.
    {
        EngineConfig other = cfg;
        other.platform_power_mode = "other-mode";
        auto e = halo::runtime::create_cpu_engine(other);
        EXPECT_EQ(e->stats().gdn_chunk, 64u) << e->stats().tuning;
        EXPECT_EQ(e->stats().threads, default_threads);
        EXPECT_NE(e->stats().tuning.find("MATMUL=none"), std::string::npos) << e->stats().tuning;
    }
    // An explicit thread count overrides the profile; the GDN chunk still applies.
    {
        EngineConfig expl = cfg;
        expl.threads = 3;
        auto e = halo::runtime::create_cpu_engine(expl);
        EXPECT_EQ(e->stats().threads, 3u);
        EXPECT_EQ(e->stats().gdn_chunk, tuned_chunk);
        EXPECT_NE(e->stats().tuning.find("explicit threads"), std::string::npos) << e->stats().tuning;
    }
    // Without a power-mode label the profile key cannot be built: a typed error, not a silent default.
    {
        EngineConfig bad = cfg;
        bad.platform_power_mode.clear();
        try {
            (void)halo::runtime::create_cpu_engine(bad);
            ADD_FAILURE() << "expected Error(Config)";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Config) << e.what();
        }
    }
}

TEST_F(Fx, ProfileLookupIsOffByDefault) {
    auto e = halo::runtime::create_cpu_engine(tiny_cfg());
    EXPECT_EQ(e->stats().tuning, "off");
    EXPECT_EQ(e->stats().gdn_chunk, 64u);
}
