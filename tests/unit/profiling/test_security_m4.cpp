// Regression tests for the security review of the bench runner
// (docs/reviews/2026-09-25-security-review-bench.md, S-29..S-37). Each test was written
// before its fix and observed failing against the reviewed code.

#include <gtest/gtest.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "fake_http.h"
#include "halo/profiling/baseline.h"
#include "halo/profiling/http_client.h"
#include "halo/profiling/sha256.h"
#include "halo/profiling/subprocess.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;
using halo::profiling::test::FakeHttpServer;
using namespace std::chrono_literals;

namespace {

const std::string kHelper = HALO_CHILD_HELPER;

std::filesystem::path tmp(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("halo_wsj_m4_" + std::to_string(::getpid()) + "_" + name);
}

std::filesystem::path fixture(const char* name) {
    return test::source_dir() / "tests" / "unit" / "profiling" / "fixtures" / name;
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool has_note(const SuiteArtifact& a, std::string_view s) {
    for (const auto& n : a.notes) {
        if (n.find(s) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

// ---- S-31: secrets never recorded ---------------------------------------------------------

TEST(SecurityS31, SecretValuesAreRedactedAndOnlyAllowlistedValuesKept) {
    ::setenv("LLAMA_API_KEY", "sk-inherited-secret", 1);
    ::setenv("LLAMA_ARG_HOST", "0.0.0.0", 1);
    ::setenv("RADV_PERFTEST", "nogttspill", 1);
    ::setenv("HALO_WSJ_UNRELATED", "x", 1);
    ProcessOptions o;
    o.inherit_env = true;
    o.env = {{"HF_TOKEN", "hf_explicit_secret"},
             {"MY_PASSWORD", "hunter2"},
             {"GGML_VK_SECRET_THING", "s3cret"},
             {"GGML_VK_VISIBLE_DEVICES", "0"},
             {"HALO_FAKE_MODE", "llama_bench"}};
    const auto rec = recorded_environment(o);
    ::unsetenv("LLAMA_API_KEY");
    ::unsetenv("LLAMA_ARG_HOST");
    ::unsetenv("RADV_PERFTEST");
    ::unsetenv("HALO_WSJ_UNRELATED");
    // Names are recorded, secret values never.
    EXPECT_EQ(rec.at("LLAMA_API_KEY"), kRedacted);
    EXPECT_EQ(rec.at("HF_TOKEN"), kRedacted);
    EXPECT_EQ(rec.at("MY_PASSWORD"), kRedacted);
    EXPECT_EQ(rec.at("GGML_VK_SECRET_THING"), kRedacted);  // pattern beats the allowlist
    // Values only for allowlisted, non-secret names.
    EXPECT_EQ(rec.at("RADV_PERFTEST"), "nogttspill");
    EXPECT_EQ(rec.at("GGML_VK_VISIBLE_DEVICES"), "0");
    EXPECT_EQ(rec.at("LLAMA_ARG_HOST"), kRedacted);   // LLAMA_* is recorded by name only
    EXPECT_EQ(rec.at("HALO_FAKE_MODE"), kRedacted);   // explicit but not allowlisted
    EXPECT_FALSE(rec.contains("HALO_WSJ_UNRELATED"));  // inherited, unrelated: not recorded
    for (const auto& [k, v] : rec) {
        EXPECT_EQ(v.find("secret"), std::string::npos) << k;
        EXPECT_EQ(v.find("hunter2"), std::string::npos) << k;
    }
}

TEST(SecurityS31, BaselineInvocationCarriesNoSecret) {
    LlamaBenchConfig c;
    c.binary = HALO_CHILD_HELPER;
    c.model = fixture("llama_bench_version.txt");
    c.backend = "cpu";
    c.common.model_name = "tiny";
    c.common.host_label = "unit-test";
    c.common.model_hash = "precomputed";
    c.common.environment.enabled = false;
    c.common.process.env = {{"HALO_FAKE_MODE", "llama_bench"},
                            {"HALO_FAKE_JSON", fixture("llama_bench_tiny_q8_0.json").string()},
                            {"LLAMA_API_KEY", "sk-do-not-store"}};
    const SuiteArtifact a = run_llama_bench(c);
    ASSERT_FALSE(a.records.empty());
    const std::string dumped = nlohmann::json(a).dump();
    EXPECT_EQ(dumped.find("sk-do-not-store"), std::string::npos);
    EXPECT_NE(dumped.find("LLAMA_API_KEY"), std::string::npos);  // the name is kept
}

// ---- S-30: waitpid failure is never ok() --------------------------------------------------

TEST(SecurityS30, LostExitStatusIsNotOk) {
    struct sigaction old{};
    struct sigaction ign{};
    ign.sa_handler = SIG_IGN;
    sigemptyset(&ign.sa_mask);
    ::sigaction(SIGCHLD, &ign, &old);  // children are auto-reaped: waitpid -> ECHILD
    ProcessOptions o;
    o.env = {{"HALO_FAKE_MODE", "exit"}, {"HALO_FAKE_CODE", "7"}};
    o.timeout = 10s;
    const ProcessResult r = run_process({kHelper}, o);
    ::sigaction(SIGCHLD, &old, nullptr);
    EXPECT_FALSE(r.ok());
    EXPECT_TRUE(r.status_unknown);
    EXPECT_NE(r.exit_code, 0);
}

// ---- S-32: never signal after reap --------------------------------------------------------

TEST(SecurityS32, NoSignalIsSentToAReapedChild) {
    const auto before = detail::signals_refused_after_reap();
    ProcessOptions o;
    o.env = {{"HALO_FAKE_MODE", "exit"}, {"HALO_FAKE_CODE", "0"}};
    EXPECT_TRUE(run_process({kHelper}, o).ok());
    ProcessOptions t;
    t.timeout = 100ms;
    t.kill_grace = 100ms;
    EXPECT_TRUE(run_process({"/usr/bin/sleep", "5"}, t).timed_out);
    {
        ChildProcess c({"/usr/bin/true"}, ProcessOptions{});
        for (int i = 0; i < 200 && c.running(); ++i) ::usleep(5000);
        c.terminate();
        c.terminate();
    }
    EXPECT_EQ(detail::signals_refused_after_reap(), before);
}

// ---- S-33: no fd leaks into children; own fds are CLOEXEC -----------------------------------

TEST(SecurityS33, ParentFdsWithoutCloexecAreNotInherited) {
    const int leak = ::open("/etc/passwd", O_RDONLY);  // deliberately without O_CLOEXEC
    ASSERT_GE(leak, 3);
    ProcessOptions o;
    o.env = {{"HALO_FAKE_MODE", "fds"}};
    const auto r = run_process({kHelper}, o);
    ::close(leak);
    ASSERT_TRUE(r.ok());
    std::set<int> fds;
    std::istringstream in(r.out);
    for (int fd; in >> fd;) fds.insert(fd);
    EXPECT_EQ(fds, (std::set<int>{0, 1, 2}));
}

TEST(SecurityS33, Sha256FileOpensWithCloexec) {
    const auto p = tmp("cloexec");
    { std::ofstream(p) << std::string(10000, 'z'); }
    bool checked = false;
    bool cloexec = false;
    (void)sha256_file(p, 1000, [&](std::uint64_t) {
        if (checked) return;
        for (const auto& e : std::filesystem::directory_iterator("/proc/self/fd")) {
            std::error_code ec;
            const auto target = std::filesystem::read_symlink(e.path(), ec);
            if (ec || target != p) continue;
            const int fd = std::stoi(e.path().filename().string());
            cloexec = (::fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0;
            checked = true;
        }
    });
    std::filesystem::remove(p);
    ASSERT_TRUE(checked) << "the hashed file's fd was not found in /proc/self/fd";
    EXPECT_TRUE(cloexec);
}

// ---- S-35: header injection ---------------------------------------------------------------

TEST(SecurityS35, CrLfInHeaderValuesIsRejected) {
    FakeHttpServer srv([](const auto&, const auto&, const auto&) { return FakeHttpServer::json(200, "{}"); });
    for (const std::string& bad : {std::string("application/json\r\nX-Injected: 1"), std::string("a\nb"),
                                  std::string("a\0b", 3)}) {
        HttpRequest r;
        r.method = "POST";
        r.port = srv.port();
        r.path = "/x";
        r.content_type = bad;
        test::expect_error(ErrorCode::Config, [&] { (void)http_request(r); });
    }
    EXPECT_TRUE(srv.requests().empty());  // nothing was sent
}

// ---- S-36: the hashed model is the benchmarked model ------------------------------------------

TEST(SecurityS36, ModelAndArgvCannotBeRedirected) {
    LlamaBenchConfig b;
    b.binary = "/opt/llama/bin/llama-bench";
    b.model = "/models/a.gguf,/models/b.gguf";  // llama-bench splits -m on ','
    test::expect_error(ErrorCode::Config, [&] { (void)llama_bench_argv(b); });
    b.model = "/models/a.gguf";
    for (const std::vector<std::string>& extra : {std::vector<std::string>{"-m", "/other.gguf"},
                                                 std::vector<std::string>{"--model", "/other.gguf"},
                                                 std::vector<std::string>{"--model=/other.gguf"}}) {
        b.extra_args = extra;
        test::expect_error(ErrorCode::Config, [&] { (void)llama_bench_argv(b); });
    }
    LlamaServerConfig s;
    s.binary = "/opt/llama/bin/llama-server";
    s.model = "/models/a.gguf";
    for (const std::vector<std::string>& extra :
         {std::vector<std::string>{"--host", "0.0.0.0"}, std::vector<std::string>{"--host=0.0.0.0"},
          std::vector<std::string>{"--port", "9999"}, std::vector<std::string>{"-m", "/x.gguf"},
          std::vector<std::string>{"--model", "/x.gguf"}, std::vector<std::string>{"-hf", "user/repo"},
          std::vector<std::string>{"--model-url", "http://x"}}) {
        s.extra_args = extra;
        test::expect_error(ErrorCode::Config, [&] { (void)llama_server_argv(s); });
    }
    s.extra_args = {"--kv-unified", "-b", "512"};
    EXPECT_NO_THROW((void)llama_server_argv(s));
    // The model must be a regular file (a FIFO / device would block the hash forever).
    b.extra_args.clear();
    b.binary = HALO_CHILD_HELPER;
    b.model = "/dev/zero";
    b.common.host_label = "unit-test";
    b.common.environment.enabled = false;
    test::expect_error(ErrorCode::Config, [&] { (void)run_llama_bench(b); });
}

// ---- S-37: log file safety ----------------------------------------------------------------

TEST(SecurityS37, LogFileIsPrivateAndNeverFollowsSymlinks) {
    const auto log = tmp("log");
    std::filesystem::remove(log);
    ProcessOptions o;
    o.kill_grace = 100ms;
    {
        ChildProcess c({"/usr/bin/true"}, o, log);
        for (int i = 0; i < 200 && c.running(); ++i) ::usleep(5000);
    }
    struct stat st{};
    ASSERT_EQ(::stat(log.c_str(), &st), 0);
    EXPECT_EQ(st.st_mode & 0777, 0600u);
    std::filesystem::remove(log);

    const auto target = tmp("victim");
    { std::ofstream(target) << "keep me"; }
    std::filesystem::create_symlink(target, log);
    test::expect_error(ErrorCode::Io, [&] { ChildProcess c({"/usr/bin/true"}, o, log); });
    EXPECT_EQ(slurp(target), "keep me");  // not truncated through the link
    std::filesystem::remove(log);
    std::filesystem::remove(target);
}

// ---- S-29: the answering server is the spawned child ------------------------------------------

TEST(SecurityS29, PortAlreadyInUseIsRefusedBeforeSpawn) {
    const std::string forged = slurp(fixture("llama_server_completion_mtp_tiny.json"));
    FakeHttpServer impostor([&](const std::string&, const std::string& p, const std::string&) {
        if (p == "/health") return FakeHttpServer::json(200, "{\"status\":\"ok\"}");
        return FakeHttpServer::json(200, forged);
    });
    LlamaServerConfig c;
    c.binary = HALO_CHILD_HELPER;  // stays alive; never opens the port
    c.model = fixture("llama_bench_version.txt");
    c.backend = "cpu";
    c.port = impostor.port();
    c.mtp = true;
    c.startup_timeout = 3s;
    c.common.model_name = "tiny";
    c.common.host_label = "unit-test";
    c.common.model_hash = "precomputed";
    c.common.environment.enabled = false;
    c.common.process.env = {{"HALO_FAKE_MODE", "llama_bench"}};  // --version works, then exits
    const SuiteArtifact a = run_llama_server(c);
    EXPECT_TRUE(a.records.empty()) << "forged numbers were recorded";
    EXPECT_TRUE(has_note(a, "FAILED")) << nlohmann::json(a.notes).dump();
    EXPECT_TRUE(has_note(a, "already in use"));
    EXPECT_FALSE(a.valid);
    // Nothing but the version probe reached the impostor... and not even that: it is HTTP.
    for (const auto& r : impostor.requests()) EXPECT_EQ(r.find("/completion"), std::string::npos) << r;
}

TEST(SecurityS29, ListenerOwnershipIsCheckedAgainstProc) {
    FakeHttpServer mine([](const auto&, const auto&, const auto&) { return FakeHttpServer::json(200, "{}"); });
    EXPECT_EQ(port_listener_is(::getpid(), mine.port()), true);  // this process owns it
    ChildProcess other({"/usr/bin/sleep", "5"}, ProcessOptions{});
    EXPECT_EQ(port_listener_is(other.pid(), mine.port()), false);
    other.terminate();
}

TEST(SecurityS29, AnswerFromAProcessOtherThanTheSpawnedChildIsRejected) {
    // The port is free when HALO probes it; the spawned "server" then has a grandchild answer
    // on the port with forged numbers. Only the /proc ownership check can tell.
    std::uint16_t port = 0;
    {
        FakeHttpServer probe([](const auto&, const auto&, const auto&) { return std::string(); });
        port = probe.port();
    }
    LlamaServerConfig c;
    c.binary = HALO_CHILD_HELPER;
    c.model = fixture("llama_bench_version.txt");
    c.backend = "cpu";
    c.port = port;
    c.mtp = true;
    c.startup_timeout = 10s;
    c.common.model_name = "tiny";
    c.common.host_label = "unit-test";
    c.common.model_hash = "precomputed";
    c.common.environment.enabled = false;
    c.common.process.kill_grace = 200ms;
    c.common.process.env = {{"HALO_FAKE_MODE", "fake_server"},
                            {"HALO_FAKE_PORT", std::to_string(port)},
                            {"HALO_FAKE_JSON", fixture("llama_server_completion_mtp_tiny.json").string()}};
    const SuiteArtifact a = run_llama_server(c);
    EXPECT_TRUE(a.records.empty()) << "a non-child answer was recorded";
    EXPECT_TRUE(has_note(a, "is not the spawned server")) << nlohmann::json(a.notes).dump();
    EXPECT_FALSE(a.valid);
    EXPECT_TRUE(loopback_port_free(port));  // the whole group (incl. the grandchild) was torn down
}
