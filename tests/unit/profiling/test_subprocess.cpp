#include <gtest/gtest.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "halo/profiling/subprocess.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;
using namespace std::chrono_literals;

namespace {

const std::string kHelper = HALO_CHILD_HELPER;

ProcessOptions mode(const std::string& m, std::map<std::string, std::string> extra = {}) {
    ProcessOptions o;
    o.env = std::move(extra);
    o.env["HALO_FAKE_MODE"] = m;
    o.timeout = 20s;
    return o;
}

bool alive(pid_t p) { return ::kill(p, 0) == 0; }

}  // namespace

TEST(Subprocess, ExitCodes) {
    EXPECT_TRUE(run_process({"/usr/bin/true"}).ok());
    const auto f = run_process({"/usr/bin/false"});
    EXPECT_FALSE(f.ok());
    EXPECT_EQ(f.exit_code, 1);
    const auto e = run_process({kHelper}, mode("exit", {{"HALO_FAKE_CODE", "7"}}));
    EXPECT_EQ(e.exit_code, 7);
    EXPECT_FALSE(e.timed_out);
    EXPECT_FALSE(e.signaled);
}

TEST(Subprocess, ArgumentsAreNotShellParsed) {
    const std::vector<std::string> args{"a b", "$(echo pwned)", "*", ";", "`id`", "'q'", "\"dq\"", "x&&y", ""};
    std::vector<std::string> argv{kHelper};
    argv.insert(argv.end(), args.begin(), args.end());
    const auto r = run_process(argv, mode("argv"));
    ASSERT_TRUE(r.ok()) << r.err;
    std::string expect;
    for (const auto& a : args) expect += a + "\n";
    EXPECT_EQ(r.out, expect);  // byte-for-byte: no expansion, globbing or splitting
}

TEST(Subprocess, ExplicitEnvironmentOnly) {
    ProcessOptions o;
    o.env = {{"HALO_A", "1"}, {"RADV_PERFTEST", "nogttspill"}};
    const auto r = run_process({"/usr/bin/env"}, o);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.out, "HALO_A=1\nRADV_PERFTEST=nogttspill\n");
    // Recorded environment: only what was passed (+ prefixed inherited vars when inheriting).
    EXPECT_EQ(recorded_environment(o), o.env);
    ::setenv("HALO_WSJ_SECRET_TOKEN", "hunter2", 1);
    ::setenv("GGML_VK_VISIBLE_DEVICES", "0", 1);
    o.inherit_env = true;
    const auto rec = recorded_environment(o);
    EXPECT_FALSE(rec.contains("HALO_WSJ_SECRET_TOKEN"));
    EXPECT_EQ(rec.at("GGML_VK_VISIBLE_DEVICES"), "0");
    const auto inh = run_process({"/usr/bin/env"}, o);
    EXPECT_NE(inh.out.find("HALO_WSJ_SECRET_TOKEN=hunter2"), std::string::npos);  // passed, not recorded
    ::unsetenv("HALO_WSJ_SECRET_TOKEN");
    ::unsetenv("GGML_VK_VISIBLE_DEVICES");
    o.env = {{"BAD=KEY", "x"}};
    test::expect_error(ErrorCode::Config, [&] { (void)run_process({"/usr/bin/true"}, o); });
}

TEST(Subprocess, BothPipesDrainedConcurrently) {
    // 4 MiB on each stream, interleaved: sequential reads would block once a pipe fills.
    const auto r = run_process({kHelper}, mode("spam", {{"HALO_FAKE_BYTES", "4194304"}}));
    ASSERT_TRUE(r.ok());
    EXPECT_FALSE(r.timed_out);
    EXPECT_EQ(r.out.size(), 4194304u);
    EXPECT_EQ(r.err.size(), 4194304u);
}

TEST(Subprocess, OutputIsCapped) {
    ProcessOptions o = mode("spam", {{"HALO_FAKE_BYTES", "1048576"}});
    o.max_output_bytes = 1000;
    const auto r = run_process({kHelper}, o);
    EXPECT_TRUE(r.ok());  // the child still ran to completion (rest drained)
    EXPECT_EQ(r.out.size(), 1000u);
    EXPECT_TRUE(r.out_truncated);
    EXPECT_TRUE(r.err_truncated);
}

TEST(Subprocess, TimeoutTerminatesAndEscalatesToSigkill) {
    ProcessOptions o;
    o.timeout = 200ms;
    o.kill_grace = 300ms;
    auto r = run_process({"/usr/bin/sleep", "30"}, o);
    EXPECT_TRUE(r.timed_out);
    EXPECT_TRUE(r.signaled);
    EXPECT_EQ(r.term_signal, SIGTERM);
    EXPECT_LT(r.wall_s, 5.0);

    ProcessOptions stubborn = mode("ignore_term");
    stubborn.timeout = 200ms;
    stubborn.kill_grace = 300ms;
    r = run_process({kHelper}, stubborn);
    EXPECT_TRUE(r.timed_out);
    EXPECT_EQ(r.term_signal, SIGKILL);  // SIGTERM ignored -> escalated
    EXPECT_LT(r.wall_s, 5.0);

    ProcessOptions closer = mode("close_sleep");  // closes its pipes, keeps running
    closer.timeout = 300ms;
    closer.kill_grace = 200ms;
    r = run_process({kHelper}, closer);
    EXPECT_TRUE(r.timed_out);
    EXPECT_LT(r.wall_s, 5.0);
}

TEST(Subprocess, TimeoutKillsTheWholeProcessGroup) {
    const auto pidfile = std::filesystem::temp_directory_path() / ("halo_wsj_gc_" + std::to_string(::getpid()));
    ProcessOptions o = mode("grandchild", {{"HALO_FAKE_PIDFILE", pidfile.string()}});
    o.timeout = 500ms;
    o.kill_grace = 200ms;
    const auto r = run_process({kHelper}, o);
    EXPECT_TRUE(r.timed_out);
    pid_t g = 0;
    std::ifstream(pidfile) >> g;
    std::filesystem::remove(pidfile);
    ASSERT_GT(g, 0);
    // The grandchild was in the child's group: it must be gone (allow reaping by init).
    bool gone = false;
    for (int i = 0; i < 100 && !gone; ++i) {
        gone = !alive(g);
        if (!gone) ::usleep(20000);
    }
    EXPECT_TRUE(gone) << "grandchild " << g << " survived the timeout";
}

TEST(Subprocess, InvalidProgramsAreTypedErrors) {
    test::expect_error(ErrorCode::Config, [] { (void)run_process({}); });
    test::expect_error(ErrorCode::Config, [] { (void)run_process({"sleep", "1"}); });  // no PATH search
    test::expect_error(ErrorCode::Io, [] { (void)run_process({"/nonexistent/halo-wsj-binary"}); });
    test::expect_error(ErrorCode::Io, [] { (void)run_process({"/etc/passwd"}); });  // not executable
    test::expect_error(ErrorCode::Config, [] { (void)run_process({"/usr/bin/echo", std::string("a\0b", 3)}); });
}

TEST(Subprocess, ChildProcessIsTornDownByItsDestructor) {
    pid_t pid = 0;
    {
        ProcessOptions o = mode("ignore_term");
        o.kill_grace = 200ms;
        ChildProcess c({kHelper}, o);
        pid = c.pid();
        EXPECT_TRUE(c.running());
    }  // destructor: SIGTERM ignored -> SIGKILL, reaped
    EXPECT_FALSE(alive(pid));
    ChildProcess quick({"/usr/bin/true"}, ProcessOptions{});
    for (int i = 0; i < 200 && quick.running(); ++i) ::usleep(5000);
    EXPECT_FALSE(quick.running());
    EXPECT_EQ(quick.exit_code(), 0);
    quick.terminate();  // idempotent after exit
}
