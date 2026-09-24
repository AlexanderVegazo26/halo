#pragma once
// Helpers for the autotune tests: unique temp DB paths (ctest runs test processes in
// parallel), typed-error expectations, a fake TunableOp driven by a ManualClock.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <string>

#include "halo/autotune/db.h"
#include "halo/autotune/tuner.h"
#include "halo/profiling/timer.h"

namespace halo::autotune::test {

/// A unique path on the Linux FS; removes the DB and its -wal/-shm files on destruction.
class TempDb {
public:
    explicit TempDb(const std::string& name) {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                ("halo_wsj_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + "_" + name + ".db");
        remove();
    }
    ~TempDb() { remove(); }
    TempDb(const TempDb&) = delete;
    TempDb& operator=(const TempDb&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }
    void remove() const {
        std::error_code ec;
        for (const char* suffix : {"", "-wal", "-shm", "-journal"}) {
            std::filesystem::remove(path_.string() + suffix, ec);
        }
    }

private:
    std::filesystem::path path_;
};

inline void expect_db_error(DbErrorKind kind, const std::function<void()>& fn) {
    try {
        fn();
        ADD_FAILURE() << "expected ProfileDbError(" << to_string(kind) << "), nothing was thrown";
    } catch (const ProfileDbError& e) {
        EXPECT_EQ(e.kind(), kind) << e.what();
    } catch (const std::exception& e) {
        ADD_FAILURE() << "expected ProfileDbError(" << to_string(kind) << "), got: " << e.what();
    }
}

inline void expect_error(ErrorCode code, const std::function<void()>& fn) {
    try {
        fn();
        ADD_FAILURE() << "expected halo::Error(" << halo::to_string(code) << "), nothing was thrown";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), code) << e.what();
    } catch (const std::exception& e) {
        ADD_FAILURE() << "expected halo::Error, got: " << e.what();
    }
}

inline ProfileKey sample_key() {
    ProfileKey k;
    k.halo_version = "0.2.0";
    k.model_hash = "a1b2";
    k.pack_id = "pack-1";
    k.gpu_device = "1002:1586";
    k.gpu_arch = "gfx1151";
    k.driver_version = "radv 25.2";
    k.rocm_version = "7.15";
    k.vulkan_version = "1.4.318";
    k.kernel_version = "7.0.0";
    k.os = "Ubuntu 26.04";
    k.power_mode = "performance|manual/COMPUTE";
    k.isa_target = "gfx1151";
    return k;
}

/// Fake op: candidate "v=i" costs base_ns[i] per run (advancing the ManualClock), with an
/// optional alternating jitter (fraction of base) to make a candidate unstable, an optional
/// validation failure, and predicted cost = predicted_ns[i] (flops at 1 GFLOP/s).
class FakeOp final : public TunableOp {
public:
    struct Spec {
        std::int64_t base_ns = 1000;
        double jitter = 0.0;
        bool valid = true;
        double predicted_ns = 1000;
    };
    FakeOp(profiling::ManualClock& clock, std::map<std::int64_t, Spec> specs, std::string shape = "S",
           std::string backend = "cpu")
        : clock_(clock), specs_(std::move(specs)), shape_(std::move(shape)), backend_(std::move(backend)) {}

    [[nodiscard]] OpKey key() const override { return {"FAKE", shape_}; }
    [[nodiscard]] std::string backend() const override { return backend_; }
    [[nodiscard]] std::vector<Candidate> candidates() const override {
        std::vector<Candidate> out;
        for (const auto& [v, s] : specs_) out.push_back(Candidate{{{"v", v}}});
        return out;
    }
    Measurement run(const Candidate& c) override {
        const Spec& s = specs_.at(c.get("v"));
        const double sign = (runs_[c.get("v")]++ % 2 == 0) ? 1.0 : -1.0;
        clock_.advance(static_cast<std::int64_t>(static_cast<double>(s.base_ns) * (1.0 + sign * s.jitter)));
        ++total_runs;
        return {};
    }
    bool validate(const Candidate& c, std::string& why) override {
        if (specs_.at(c.get("v")).valid) return true;
        why = "fake mismatch";
        return false;
    }
    [[nodiscard]] std::optional<CostInputs> cost(const Candidate& c) const override {
        CostInputs in;
        in.flops = specs_.at(c.get("v")).predicted_ns;  // at 1 GFLOP/s/thread: ns == flops
        in.bytes = 0;
        in.parallelism = 1;
        in.launches = 0;
        return in;
    }
    int total_runs = 0;

private:
    profiling::ManualClock& clock_;
    std::map<std::int64_t, Spec> specs_;
    std::map<std::int64_t, int> runs_;
    std::string shape_;
    std::string backend_;
};

/// Cost model under which FakeOp::cost predictions are exactly predicted_ns.
inline CostModel unit_cost_model() {
    CostModel m;
    m.bandwidth_gbps[hardware::MemoryTier::Host] = 100.0;
    m.gflops_per_thread = 1.0;
    return m;
}

inline TuneOptions fast_options(Strategy s) {
    TuneOptions o;
    o.strategy = s;
    o.cost_model = unit_cost_model();
    return o;  // TRD §50 defaults: 5 warm-up + 20 measured
}

}  // namespace halo::autotune::test
