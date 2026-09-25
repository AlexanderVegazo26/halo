#pragma once
// Internal to halo_profiling: builds a SuiteArtifact around a run (before/after hardware
// snapshots, thermal + power-mode verdicts, summaries, conformance). Shared by the suites
// (suite.cpp) and the baseline adapters (baseline.cpp).

#include <optional>
#include <string>

#include "halo/core/error.h"
#include "halo/core/log.h"
#include "halo/profiling/suite.h"

namespace halo::profiling {

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

}  // namespace halo::profiling
