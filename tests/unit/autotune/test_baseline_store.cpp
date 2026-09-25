#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

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

}  // namespace

TEST(BaselineStore, StoresMedianAggregatesQueryableAsLatest) {
    TempDb t("baseline");
    ProfileDb db = ProfileDb::open(t.path());
    EXPECT_EQ(store_baseline(db, artifact("vulkan")), 2u);  // pp16 + tg8 aggregates
    EXPECT_EQ(db.count("benchmark_run"), 2);
    auto tg = db.latest_run("baseline", "vulkan", "tiny-q8_0", 8, "tg8");
    ASSERT_TRUE(tg);
    // Samples 167.092, 144.574, 167.092 -> median 167.092 (the last sample would be too).
    EXPECT_DOUBLE_EQ(*tg->decode_tps, 167.092);
    EXPECT_EQ(tg->extra.at("aggregate").at("n"), 3);
    EXPECT_EQ(tg->engine, "llama-bench");
    const auto pp = db.latest_run("baseline", "vulkan", "tiny-q8_0", 16, "pp16");
    ASSERT_TRUE(pp);
    EXPECT_DOUBLE_EQ(*pp->prompt_tps, 1578.73);

    // A newer run of the same configuration becomes the latest; other backends are separate.
    EXPECT_EQ(store_baseline(db, artifact("vulkan", 2.0)), 2u);
    tg = db.latest_run("baseline", "vulkan", "tiny-q8_0", 8, "tg8");
    EXPECT_DOUBLE_EQ(*tg->decode_tps, 2 * 167.092);
    EXPECT_EQ(store_baseline(db, artifact("hip")), 2u);
    EXPECT_DOUBLE_EQ(*db.latest_run("baseline", "hip", "tiny-q8_0", 8, "tg8")->decode_tps, 167.092);
    EXPECT_DOUBLE_EQ(*db.latest_run("baseline", "vulkan", "tiny-q8_0", 8, "tg8")->decode_tps, 2 * 167.092);
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
    ASSERT_TRUE(r);
    EXPECT_FALSE(r->extra.at("aggregate").at("artifact_valid").get<bool>());  // marked, not hidden
    a.suite = "micro";
    test::expect_error(ErrorCode::Config, [&] { (void)store_baseline(db, a, true); });
}
