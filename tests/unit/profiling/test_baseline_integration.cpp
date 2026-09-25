// Integration tests against the real llama.cpp binaries (commit bd4f514, the target
// machine's commit) on the tiny random model. The dev host's llama.cpp is a CPU-only build,
// so these run with backend "cpu"; the Vulkan/HIP configurations are covered by the argv and
// parser tests only. Timings are harness smoke tests (D-001).
//
// Binary directory: env HALO_LLAMA_BIN_DIR, else the compile definition (default
// $HOME/llama-build-wsj/bin; /root/llama.cpp/build/bin has llama-server but no llama-bench).

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "halo/profiling/baseline.h"
#include "halo/profiling/http_client.h"
#include "halo/profiling/sha256.h"
#include "test_util.h"

using namespace halo::profiling;
using namespace std::chrono_literals;

namespace {

std::filesystem::path bin_dir() {
    const char* e = std::getenv("HALO_LLAMA_BIN_DIR");
    return e != nullptr ? std::filesystem::path(e) : std::filesystem::path(HALO_LLAMA_BIN_DIR);
}

std::filesystem::path tiny_model() { return std::filesystem::path(HALO_REF_DIR) / "tiny" / "tiny-q8_0.gguf"; }

std::uint16_t free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    socklen_t len = sizeof(a);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    ::close(fd);
    return ntohs(a.sin_port);
}

BaselineCommon dev_host_common() {
    BaselineCommon c;
    c.model_name = "halo tiny random qwen35";
    c.pack = "tiny-q8_0";
    c.host_label = "dev-host";
    c.environment.platform_power_mode = "dev-host";
    c.environment.discovery.use_process_env = false;
    c.process.timeout = 5min;
    return c;
}

#define REQUIRE_FILE(p)                                                                          \
    do {                                                                                         \
        if (!std::filesystem::exists(p)) GTEST_SKIP() << "missing " << (p) << " (see header)"; \
    } while (0)

}  // namespace

TEST(BaselineIntegration, LlamaBenchOnTinyModel) {
    const auto bench = bin_dir() / "llama-bench";
    REQUIRE_FILE(bench);
    REQUIRE_FILE(tiny_model());
    LlamaBenchConfig c;
    c.binary = bench;
    c.model = tiny_model();
    c.backend = "cpu";
    c.n_prompt = {16};
    c.n_gen = {8};
    c.repetitions = 3;
    c.threads = 2;
    c.flash_attn = "auto";
    c.common = dev_host_common();
    const SuiteArtifact a = run_llama_bench(c);
    for (const auto& n : a.notes) EXPECT_EQ(n.find("FAILED"), std::string::npos) << n;
    ASSERT_EQ(a.records.size(), 6u);  // pp16 + tg8, 3 samples each
    EXPECT_TRUE(a.conformant);         // 3 repetitions per configuration
    EXPECT_TRUE(a.thermal.valid || !a.thermal.drift_c) << a.thermal.reason;
    // TRD §51 fields.
    const auto& r = a.records.front();
    ASSERT_TRUE(r.invocation);
    EXPECT_EQ(r.invocation->commit, "bd4f514");
    EXPECT_EQ(r.invocation->binary, bench.string());
    EXPECT_EQ(r.invocation->argv, llama_bench_argv(c));
    EXPECT_EQ(r.backend, "cpu");
    EXPECT_EQ(r.driver, "cpu");
    EXPECT_EQ(r.extra.at("backends"), "CPU");
    EXPECT_EQ(r.power_mode, a.hardware_before->power_mode);
    ASSERT_TRUE(r.hardware_before && r.hardware_after);
    // Model hash: independent oracle = coreutils sha256sum, run through the same runner.
    const auto sum = run_process({"/usr/bin/sha256sum", tiny_model().string()});
    ASSERT_TRUE(sum.ok());
    EXPECT_EQ(r.model_hash, sum.out.substr(0, 64));
    bool pp = false, tg = false;
    for (const auto& x : a.records) {
        pp = pp || (x.mode == "pp16" && x.prompt_tps && *x.prompt_tps > 0);
        tg = tg || (x.mode == "tg8" && x.decode_tps && *x.decode_tps > 0);
    }
    EXPECT_TRUE(pp && tg);
    // The dev host exposes no GPU power state: the run is correctly not comparable.
    EXPECT_FALSE(a.power_mode_pinned.comparable);
    EXPECT_FALSE(a.valid);
}

TEST(BaselineIntegration, LlamaServerMtpOnTinyModel) {
    const auto server = bin_dir() / "llama-server";
    REQUIRE_FILE(server);
    REQUIRE_FILE(tiny_model());
    LlamaServerConfig c;
    c.binary = server;
    c.model = tiny_model();
    c.backend = "cpu";
    c.port = free_port();
    c.ctx_size = 512;
    c.n_gpu_layers = 0;
    c.flash_attn = "auto";
    c.mtp = true;
    c.extra_args = {"-t", "2"};
    c.prompt = "Hello world, this is a test";
    c.n_predict = 16;
    c.startup_timeout = 120s;
    c.request_timeout = 120s;
    c.common = dev_host_common();
    c.common.environment.enabled = false;
    const SuiteArtifact a = run_llama_server(c);
    for (const auto& n : a.notes) EXPECT_EQ(n.find("FAILED"), std::string::npos) << n;
    ASSERT_EQ(a.records.size(), 4u);  // 1 warm-up + 3
    for (const auto& r : a.records) {
        EXPECT_EQ(r.mode, "mtp_completion");
        ASSERT_TRUE(r.decode_effective_tps);
        EXPECT_GT(*r.decode_effective_tps, 0);
        EXPECT_TRUE(r.extra.contains("draft_n"));  // speculation was active
        EXPECT_FALSE(r.ttft_ms);
        EXPECT_EQ(r.invocation->commit, "bd4f514");
    }
    // The server is gone: nothing listens on its port any more.
    HttpRequest h;
    h.port = c.port;
    h.path = "/health";
    h.timeout = 1s;
    EXPECT_THROW((void)http_request(h), halo::Error);
}
