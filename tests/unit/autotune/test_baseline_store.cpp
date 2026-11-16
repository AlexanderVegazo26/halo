#include <gtest/gtest.h>

#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "autotune_test_util.h"
#include "halo/autotune/baseline_store.h"
#include "halo/profiling/baseline.h"

using namespace halo::autotune;
using halo::ErrorCode;
using test::TempDb;

namespace {

nlohmann::json fixture(const char* name) {
    std::ifstream in(std::filesystem::path(HALO_SOURCE_DIR) / "tests" / "unit" / "profiling" / "fixtures" / name);
    std::stringstream ss;
    ss << in.rdbuf();
    return nlohmann::json::parse(ss.str());
}

/// A valid-looking baseline artifact from the captured llama-bench output, with 3 samples
/// per test (the fixture has 2; a third is appended) and pinned hardware verdicts.
halo::profiling::SuiteArtifact artifact(const std::string& backend, double tg_scale = 1.0) {
    auto j = fixture("llama_bench_tiny_q8_0.json");
    for (auto& t : j) {
        t["samples_ts"].push_back(t["samples_ts"][0]);
        t["samples_ns"].push_back(t["samples_ns"][0]);
        for (auto& v : t["samples_ts"]) v = v.get<double>() * tg_scale;
        if (backend == "vulkan") t["backends"] = "Vulkan";
        if (backend == "hip") t["backends"] = "ROCm";
    }
    halo::profiling::BaselineParseContext ctx{"llama-bench", backend, "tiny", "tiny-q8_0", "sha", "evo-x2", "bd4f514"};
    auto p = halo::profiling::parse_llama_bench_json(j, ctx);
    EXPECT_TRUE(p.failures.empty());
    halo::profiling::SuiteArtifact a;
    a.suite = "baseline";
    a.records = std::move(p.records);
    for (auto& r : a.records) r.power_mode = "performance|manual/COMPUTE";
    a.summaries = halo::profiling::summarize(a.records);
    a.thermal.valid = true;
    a.power_mode_pinned.comparable = true;
    a.conformant = true;
    a.valid = true;
    return a;
}

/// A missing metric reads as NaN, so EXPECT_DOUBLE_EQ fails readably instead of dereferencing
/// an empty optional.
double value(const std::optional<double>& v) { return v.value_or(std::numeric_limits<double>::quiet_NaN()); }

/// decode_tps of the latest tiny-q8_0 tg8 baseline for `backend`; NaN when none is stored.
double latest_tg8_decode_tps(const ProfileDb& db, const std::string& backend) {
    const auto r = db.latest_run("baseline", backend, "tiny-q8_0", 8, "tg8");
    return r ? value(r->decode_tps) : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

TEST(BaselineStore, StoresMedianAggregatesQueryableAsLatest) {
    TempDb t("baseline");
    ProfileDb db = ProfileDb::open(t.path());
    EXPECT_EQ(store_baseline(db, artifact("vulkan")), 2u);  // pp16 + tg8 aggregates
    EXPECT_EQ(db.count("benchmark_run"), 2);
    const auto tg = db.latest_run("baseline", "vulkan", "tiny-q8_0", 8, "tg8");
    if (!tg) GTEST_FAIL() << "no vulkan tiny-q8_0 tg8 baseline stored";
    // Samples 167.092, 144.574, 167.092 -> median 167.092 (the last sample would be too).
    EXPECT_DOUBLE_EQ(value(tg->decode_tps), 167.092);
    EXPECT_EQ(tg->extra.at("aggregate").at("n"), 3);
    EXPECT_EQ(tg->engine, "llama-bench");
    const auto pp = db.latest_run("baseline", "vulkan", "tiny-q8_0", 16, "pp16");
    if (!pp) GTEST_FAIL() << "no vulkan tiny-q8_0 pp16 baseline stored";
    EXPECT_DOUBLE_EQ(value(pp->prompt_tps), 1578.73);

    // A newer run of the same configuration becomes the latest; other backends are separate.
    EXPECT_EQ(store_baseline(db, artifact("vulkan", 2.0)), 2u);
    EXPECT_DOUBLE_EQ(latest_tg8_decode_tps(db, "vulkan"), 2 * 167.092) << "the newer vulkan run should be the latest";
    EXPECT_EQ(store_baseline(db, artifact("hip")), 2u);
    EXPECT_DOUBLE_EQ(latest_tg8_decode_tps(db, "hip"), 167.092) << "hip is a separate configuration";
    EXPECT_DOUBLE_EQ(latest_tg8_decode_tps(db, "vulkan"), 2 * 167.092) << "storing hip must not replace vulkan";
    EXPECT_FALSE(db.latest_run("baseline", "vulkan", "tiny-q8_0", 4096, "tg8"));
}

TEST(BaselineStore, RefusesInvalidArtifactsUnlessAllowed) {
    TempDb t("baseline_invalid");
    ProfileDb db = ProfileDb::open(t.path());
    auto a = artifact("vulkan");
    a.valid = false;
    a.power_mode_pinned = {false, "power modes differ"};
    try {
        (void)store_baseline(db, a);
        ADD_FAILURE() << "an invalid artifact was stored";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Config);
        EXPECT_NE(std::string(e.what()).find("power modes differ"), std::string::npos);
    }
    EXPECT_EQ(db.count("benchmark_run"), 0);
    EXPECT_EQ(store_baseline(db, a, /*allow_invalid=*/true), 2u);
    const auto r = db.latest_run("baseline", "vulkan", "tiny-q8_0", 8, "tg8");
    if (!r) GTEST_FAIL() << "the invalid artifact stored with allow_invalid has no tg8 row";
    EXPECT_FALSE(r->extra.at("aggregate").at("artifact_valid").get<bool>());  // marked, not hidden
    a.suite = "micro";
    test::expect_error(ErrorCode::Config, [&] { (void)store_baseline(db, a, true); });
}
