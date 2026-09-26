// HALO_LOG_LEVEL / HALO_LOG_FORMAT / HALO_LOG_FILE (include/halo/core/log.h).
//
// The environment is read once per process, so each case runs HALO_LOG_ENV_HELPER
// (log_env_helper.cpp) in a fresh process with an explicit environment, and checks its
// stdout, stderr, exit status and the log file.
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef HALO_LOG_ENV_HELPER
#error "HALO_LOG_ENV_HELPER must name the helper binary"
#endif

namespace {

namespace fs = std::filesystem;

struct Child {
    bool exited = false;
    int code = -1;
    int signal = 0;
    bool timed_out = false;
    std::string out;
    std::string err;
};

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

class LogEnv : public ::testing::Test {
protected:
    void SetUp() override {
        std::string tmpl = (fs::path(::testing::TempDir()) / "halo-log-env-XXXXXX").string();
        ASSERT_NE(::mkdtemp(tmpl.data()), nullptr);
        dir_ = tmpl;
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    // Runs the helper with PATH plus `vars` (and the sanitizer options, if set) as its whole
    // environment. A child still running after 20 s is killed and reported as timed out.
    Child run(const std::string& mode, const std::vector<std::string>& vars = {}) const {
        std::vector<std::string> env = {"PATH=/usr/bin:/bin"};
        for (const char* keep : {"ASAN_OPTIONS", "UBSAN_OPTIONS", "LSAN_OPTIONS"}) {
            if (const char* v = std::getenv(keep)) env.push_back(std::string(keep) + "=" + v);
        }
        env.insert(env.end(), vars.begin(), vars.end());
        std::vector<char*> envp;
        for (auto& e : env) envp.push_back(e.data());
        envp.push_back(nullptr);
        std::string exe = HALO_LOG_ENV_HELPER;
        std::string m = mode;
        char* argv[] = {exe.data(), m.data(), nullptr};
        const std::string out_path = (dir_ / "child.out").string();
        const std::string err_path = (dir_ / "child.err").string();

        Child c;
        const pid_t pid = ::fork();
        if (pid == 0) {  // only async-signal-safe calls until exec
            const int o = ::open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            const int e = ::open(err_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (o < 0 || e < 0 || ::dup2(o, 1) < 0 || ::dup2(e, 2) < 0) ::_exit(126);
            ::close(o);
            ::close(e);
            ::execve(exe.c_str(), argv, envp.data());
            ::_exit(127);
        }
        if (pid < 0) {
            ADD_FAILURE() << "fork failed";
            return c;
        }
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (::waitpid(pid, &status, WNOHANG) == 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                c.timed_out = true;
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &status, 0);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        c.exited = WIFEXITED(status);
        c.code = c.exited ? WEXITSTATUS(status) : -1;
        c.signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
        c.out = slurp(out_path);
        c.err = slurp(err_path);
        return c;
    }

    static bool has_line(const std::string& text, const std::string& line) {
        std::istringstream in(text);
        for (std::string l; std::getline(in, l);) {
            if (l == line) return true;
        }
        return false;
    }
    static int count(const std::string& text, const std::string& needle) {
        int n = 0;
        for (auto pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1)) ++n;
        return n;
    }
    static unsigned mode_of(const fs::path& p) {
        struct stat st{};
        return ::stat(p.c_str(), &st) == 0 ? (st.st_mode & 07777U) : 0U;
    }

    fs::path dir_;
};

TEST_F(LogEnv, DefaultIsInfoAsTextOnStderr) {
    const Child c = run("emit");
    ASSERT_TRUE(c.exited) << c.err;
    EXPECT_EQ(c.code, 0);
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
    EXPECT_EQ(c.err.find("TRACE"), std::string::npos) << c.err;
    EXPECT_EQ(c.err.find("DEBUG"), std::string::npos) << c.err;
}

TEST_F(LogEnv, LevelTraceShowsTraceAndDebug) {
    const Child c = run("emit", {"HALO_LOG_LEVEL=trace"});
    EXPECT_TRUE(has_line(c.err, "[TRACE] t: trace-line 1")) << c.err;
    EXPECT_TRUE(has_line(c.err, "[DEBUG] d: debug-line")) << c.err;
}

TEST_F(LogEnv, FormatJsonWritesOneObjectPerLineWithEpochMilliseconds) {
    const Child c = run("emit", {"HALO_LOG_LEVEL=TRACE", "HALO_LOG_FORMAT=json"});
    std::istringstream in(c.err);
    std::vector<nlohmann::json> lines;
    for (std::string l; std::getline(in, l);) {
        if (!l.empty()) lines.push_back(nlohmann::json::parse(l));  // throws (test fails) on a non-JSON line
    }
    ASSERT_EQ(lines.size(), 4U) << c.err;
    EXPECT_EQ(lines[0]["level"], "TRACE");
    EXPECT_EQ(lines[0]["component"], "t");
    EXPECT_EQ(lines[0]["msg"], "trace-line 1");
    ASSERT_TRUE(lines[0]["ts_ms"].is_number_integer());
    EXPECT_GT(lines[0]["ts_ms"].get<long long>(), 1'700'000'000'000LL);  // epoch ms, not seconds
    EXPECT_EQ(lines[3]["msg"], "warn-line \"quoted\"");
}

TEST_F(LogEnv, FileReceivesLinesWithMode0600AndNothingOnStderr) {
    const fs::path log = dir_ / "halo.log";
    const Child c = run("emit", {"HALO_LOG_FILE=" + log.string()});
    EXPECT_EQ(c.err, "");
    EXPECT_TRUE(has_line(slurp(log), "[INFO] i: info-line"));
    EXPECT_EQ(mode_of(log), 0600U);
}

TEST_F(LogEnv, FileIsAppendedNotTruncated) {
    const fs::path log = dir_ / "halo.log";
    run("emit", {"HALO_LOG_FILE=" + log.string()});
    run("emit", {"HALO_LOG_FILE=" + log.string()});
    EXPECT_EQ(count(slurp(log), "info-line"), 2);
}

TEST_F(LogEnv, PreExistingFileIsTightenedTo0600) {
    const fs::path log = dir_ / "halo.log";
    std::ofstream(log).put('\n');
    ::chmod(log.c_str(), 0644);
    run("emit", {"HALO_LOG_FILE=" + log.string()});
    EXPECT_EQ(mode_of(log), 0600U);
}

TEST_F(LogEnv, SymlinkIsRefusedAndLoggingStaysOnStderr) {
    const fs::path target = dir_ / "target";
    const fs::path link = dir_ / "link";
    fs::create_symlink(target, link);
    const Child c = run("emit", {"HALO_LOG_FILE=" + link.string()});
    EXPECT_NE(c.err.find("cannot open HALO_LOG_FILE"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
    EXPECT_FALSE(fs::exists(target));
}

// A symlink to an EXISTING file of ours would pass every other check (regular, owned, one link):
// only O_NOFOLLOW refuses it. The target must stay untouched.
TEST_F(LogEnv, SymlinkToExistingFileIsRefusedAndTargetUntouched) {
    const fs::path target = dir_ / "target";
    const fs::path link = dir_ / "link";
    std::ofstream(target) << "precious\n";
    ::chmod(target.c_str(), 0600);
    fs::create_symlink(target, link);
    const Child c = run("emit", {"HALO_LOG_FILE=" + link.string()});
    EXPECT_NE(c.err.find("cannot open HALO_LOG_FILE"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
    EXPECT_EQ(slurp(target), "precious\n");
}

TEST_F(LogEnv, FifoDoesNotBlockAndFallsBackToStderr) {
    const fs::path fifo = dir_ / "fifo";
    ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);
    const Child c = run("emit", {"HALO_LOG_FILE=" + fifo.string()});
    EXPECT_FALSE(c.timed_out);
    EXPECT_EQ(c.code, 0);
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
}

// Review S-43: an existing non-regular path is refused BEFORE open(), so a driver never sees an
// open. A FIFO shows this: a reader blocked in open(O_RDONLY) returns only once a writer opens.
TEST_F(LogEnv, FifoWithReaderIsRefusedWithoutBeingOpened) {
    const fs::path fifo = dir_ / "fifo";
    ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);
    std::atomic<bool> reader_open{false};
    std::thread reader([&] {
        const int fd = ::open(fifo.c_str(), O_RDONLY | O_CLOEXEC);  // blocks until a writer opens
        reader_open.store(true);
        if (fd >= 0) ::close(fd);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // let the reader block in open()
    const Child c = run("emit", {"HALO_LOG_FILE=" + fifo.string()});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool opened_by_child = reader_open.load();
    const int unblock = ::open(fifo.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);  // release the reader
    reader.join();
    if (unblock >= 0) ::close(unblock);
    EXPECT_FALSE(opened_by_child) << "the child opened the FIFO before refusing it";
    EXPECT_NE(c.err.find("not a regular file"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
}

// Review S-42: a file with a second hard link is refused and left untouched.
TEST_F(LogEnv, HardLinkedFileIsRefusedAndLeftUntouched) {
    const fs::path orig = dir_ / "orig";
    const fs::path link = dir_ / "hl";
    std::ofstream(orig) << "precious\n";
    ::chmod(orig.c_str(), 0644);
    fs::create_hard_link(orig, link);
    const Child c = run("emit", {"HALO_LOG_FILE=" + link.string()});
    EXPECT_NE(c.err.find("more than one hard link"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
    EXPECT_EQ(slurp(orig), "precious\n");
    EXPECT_EQ(mode_of(orig), 0644U);
}

// Review S-42: a file owned by another user is refused and left untouched (no chmod, no append),
// which matters when the kit runs as root. Creating such a file needs root.
TEST_F(LogEnv, ForeignOwnedFileIsRefusedAndLeftUntouched) {
    if (::geteuid() != 0) GTEST_SKIP() << "needs root to create a file owned by another user (chown)";
    const fs::path f = dir_ / "shared.txt";
    std::ofstream(f) << "precious\n";
    ::chmod(f.c_str(), 0644);
    ASSERT_EQ(::chown(f.c_str(), 65534, 65534), 0);  // nobody
    const Child c = run("emit", {"HALO_LOG_FILE=" + f.string()});
    EXPECT_NE(c.err.find("not owned by this user"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
    EXPECT_EQ(slurp(f), "precious\n");
    EXPECT_EQ(mode_of(f), 0644U);
}

TEST_F(LogEnv, NonRegularFileIsRefused) {
    const Child c = run("emit", {"HALO_LOG_FILE=/dev/null"});
    EXPECT_NE(c.err.find("not a regular file"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
}

TEST_F(LogEnv, UnknownLevelWarnsAndKeepsInfo) {
    const Child c = run("emit", {"HALO_LOG_LEVEL=loud"});
    EXPECT_EQ(c.code, 0);
    EXPECT_NE(c.err.find("HALO_LOG_LEVEL='loud' is not a level"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
}

TEST_F(LogEnv, UnknownFormatWarnsAndKeepsText) {
    const Child c = run("emit", {"HALO_LOG_FORMAT=yaml"});
    EXPECT_NE(c.err.find("HALO_LOG_FORMAT='yaml' is not text or json"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
}

TEST_F(LogEnv, ExplicitSetLevelBeforeFirstLineBeatsEnvironment) {
    const Child c = run("explicit", {"HALO_LOG_LEVEL=trace"});
    EXPECT_EQ(c.err.find("TRACE"), std::string::npos) << c.err;
    EXPECT_EQ(c.err.find("INFO"), std::string::npos) << c.err;
    EXPECT_TRUE(has_line(c.err, "[WARN] w: warn-line \"quoted\"")) << c.err;
}

TEST_F(LogEnv, ExplicitSetJsonBeforeFirstLineBeatsEnvironment) {
    const Child c = run("explicit-json", {"HALO_LOG_FORMAT=json"});
    EXPECT_TRUE(has_line(c.err, "[INFO] i: info-line")) << c.err;
}

TEST_F(LogEnv, SetLevelAfterLoggingStillApplies) {
    const Child c = run("late", {"HALO_LOG_LEVEL=trace"});
    EXPECT_NE(c.err.find("trace-line"), std::string::npos) << c.err;
    EXPECT_EQ(c.err.find("late-info-line"), std::string::npos) << c.err;
    EXPECT_NE(c.err.find("late-error-line"), std::string::npos) << c.err;
}

TEST_F(LogEnv, EmptyValuesCountAsUnsetWithoutWarning) {
    const Child c = run("emit", {"HALO_LOG_LEVEL=", "HALO_LOG_FORMAT=", "HALO_LOG_FILE="});
    EXPECT_EQ(c.err, "[INFO] i: info-line\n[WARN] w: warn-line \"quoted\"\n");
}

TEST_F(LogEnv, LogFileDescriptorIsNotInheritedAcrossExec) {
    const fs::path log = dir_ / "halo.log";
    const Child c = run("fds", {"HALO_LOG_FILE=" + log.string()});
    ASSERT_EQ(c.code, 0) << c.err;
    ASSERT_NE(c.out.find(" -> "), std::string::npos) << "listfds printed nothing: " << c.out;
    EXPECT_EQ(c.out.find(log.string()), std::string::npos) << c.out;
}

TEST_F(LogEnv, LinesSurviveAbortBecauseEachLineIsFlushed) {
    const fs::path log = dir_ / "halo.log";
    const Child c = run("abort", {"HALO_LOG_FILE=" + log.string()});
    EXPECT_EQ(c.signal, SIGABRT);
    EXPECT_TRUE(has_line(slurp(log), "[INFO] i: info-line"));
}

}  // namespace
