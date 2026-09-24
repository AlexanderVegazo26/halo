#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include "halo/profiling/workload.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;

namespace {

nlohmann::json minimal() {
    return {{"schema", kWorkloadSchema},
            {"id", "w"},
            {"kind", "agent_replay"},
            {"provenance", "test"},
            {"agents", nlohmann::json::array({{{"id", "a"},
                                               {"preamble", {{{"role", "system"}, {"text", "sys"}}}},
                                               {"turns", {{{"role", "user"}, {"text", "hello"}}}}}})}};
}

std::filesystem::path temp_file(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("halo_wsj_" + name);
}

}  // namespace

TEST(Workload, FrozenReferenceWorkloadsParse) {
    const auto dir = test::source_dir() / "bench" / "workloads";
    ASSERT_TRUE(std::filesystem::is_directory(dir)) << dir;
    std::set<std::string> ids;
    std::set<std::string> kinds;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() != ".json") continue;
        SCOPED_TRACE(e.path().string());
        const Workload w = load_workload(e.path());
        EXPECT_TRUE(ids.insert(w.id).second) << "duplicate workload id " << w.id;
        kinds.insert(w.kind);
        EXPECT_FALSE(w.provenance.empty());
        // Every frozen workload must encode without a tokenizer.
        const auto enc = synthetic_encoder(248320);
        for (const auto& a : w.agents) {
            for (std::size_t t = 0; t < a.turns.size(); ++t) EXPECT_FALSE(enc(w, a, t).empty());
        }
    }
    EXPECT_EQ(kinds, (std::set<std::string>{"agent_replay", "long_context", "concurrency"}));

    const Workload conc = load_workload(dir / "concurrency_4agents.json");
    EXPECT_EQ(conc.agents.size(), 4u);
    const Workload replay = load_workload(dir / "agent_replay.json");
    ASSERT_EQ(replay.agents.size(), 1u);
    EXPECT_GE(replay.agents[0].turns.size(), 3u);
    EXPECT_EQ(replay.tools.size(), 4u);

    // Long-context agents: prompt + max_tokens fits each PRD section 18 depth.
    const Workload lc = load_workload(dir / "long_context.json");
    const auto enc = synthetic_encoder(248320);
    const std::uint64_t depths[] = {4096, 32768, 131072, 262144};
    ASSERT_EQ(lc.agents.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto n = enc(lc, lc.agents[i], 0).size() + lc.agents[i].max_tokens;
        EXPECT_LE(n, depths[i]) << lc.agents[i].id;
        EXPECT_GE(n, depths[i] - 256) << lc.agents[i].id;
    }
}

TEST(Workload, SyntheticTokensDeterministicAndInRange) {
    const auto a = synthetic_tokens(7, 1000, 100, 5);
    const auto b = synthetic_tokens(7, 1000, 100, 5);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, synthetic_tokens(8, 1000, 100, 5));
    std::set<std::int32_t> seen;
    for (const auto t : a) {
        ASSERT_GE(t, 5);
        ASSERT_LT(t, 105);
        seen.insert(t);
    }
    EXPECT_GT(seen.size(), 90u);  // covers the range, not a constant
    // Pinned values: SplitMix64(seed 0) first outputs reduced by mulhi into [0, 1000).
    // 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F
    EXPECT_EQ(synthetic_tokens(0, 3, 1000), (std::vector<std::int32_t>{883, 431, 26}));
    EXPECT_TRUE(synthetic_tokens(1, 0, 10).empty());
    test::expect_error(ErrorCode::Config, [] { (void)synthetic_tokens(1, 1, 0); });
    test::expect_error(ErrorCode::Config, [] { (void)synthetic_tokens(1, kMaxSyntheticTokens + 1, 10); });
}

TEST(Workload, SyntheticEncoderGrowsWithTurns) {
    nlohmann::json j = minimal();
    j["agents"][0]["turns"].push_back({{"role", "user"}, {"synthetic_tokens", 10}, {"seed", 3}});
    const Workload w = parse_workload(j);
    const auto enc = synthetic_encoder(1000);
    const auto t0 = enc(w, w.agents[0], 0);
    const auto t1 = enc(w, w.agents[0], 1);
    EXPECT_EQ(t0.size(), 1u + 2u);          // "sys" -> 1, "hello" -> 2 (ceil(bytes/4))
    EXPECT_EQ(t1.size(), t0.size() + 10u);  // prefix + synthetic segment
    EXPECT_TRUE(std::equal(t0.begin(), t0.end(), t1.begin()));  // turn 1 extends turn 0
    test::expect_error(ErrorCode::Config, [&] { (void)enc(w, w.agents[0], 2); });
    test::expect_error(ErrorCode::Config, [] { (void)synthetic_encoder(0); });
}

TEST(Workload, RejectsMalformed) {
    EXPECT_NO_THROW((void)parse_workload(minimal()));
    const auto bad = [](const std::function<void(nlohmann::json&)>& mutate) {
        nlohmann::json j = minimal();
        mutate(j);
        test::expect_error(ErrorCode::Config, [&] { (void)parse_workload(j); });
    };
    bad([](auto& j) { j["schema"] = "halo.bench.workload/2"; });
    bad([](auto& j) { j.erase("provenance"); });
    bad([](auto& j) { j["id"] = 5; });
    bad([](auto& j) { j["kind"] = "chat"; });
    bad([](auto& j) { j["tools"] = "none"; });
    bad([](auto& j) { j["agents"] = nlohmann::json::array(); });
    bad([](auto& j) {
        nlohmann::json many = nlohmann::json::array();
        for (std::size_t i = 0; i <= kMaxAgents; ++i) many.push_back(j["agents"][0]);
        j["agents"] = many;
    });
    bad([](auto& j) { j["agents"][0]["turns"] = nlohmann::json::array(); });
    bad([](auto& j) { j["agents"][0]["turns"][0]["role"] = "robot"; });
    bad([](auto& j) { j["agents"][0]["turns"][0]["synthetic_tokens"] = 4; });  // both
    bad([](auto& j) { j["agents"][0]["turns"][0].erase("text"); });            // neither
    bad([](auto& j) { j["agents"][0]["turns"][0]["text"] = 1; });
    bad([](auto& j) { j["agents"][0]["turns"][0] = {{"role", "user"}, {"synthetic_tokens", 0}}; });
    bad([](auto& j) {
        j["agents"][0]["turns"][0] = {{"role", "user"}, {"synthetic_tokens", kMaxSyntheticTokens + 1}};
    });
    bad([](auto& j) { j["agents"][0]["turns"][0] = {{"role", "user"}, {"synthetic_tokens", -3}}; });
    bad([](auto& j) { j["agents"][0]["turns"][0] = {{"role", "user"}, {"synthetic_tokens", 3}, {"seed", -1}}; });
    bad([](auto& j) { j["agents"][0]["max_tokens"] = 0; });
    bad([](auto& j) { j["agents"][0]["max_tokens"] = 70000; });
    bad([](auto& j) { j["agents"][0]["think_time_ms"] = 600001; });
    bad([](auto& j) { j["agents"][0]["preamble"] = "x"; });
    test::expect_error(ErrorCode::Config, [] { (void)parse_workload(nlohmann::json::array()); });
}

TEST(Workload, LoadErrorsAreTyped) {
    test::expect_error(ErrorCode::Io, [] { (void)load_workload(temp_file("does_not_exist.json")); });

    const auto invalid = temp_file("invalid.json");
    { std::ofstream(invalid) << "{ not json"; }
    test::expect_error(ErrorCode::Config, [&] { (void)load_workload(invalid); });

    // A *valid* workload padded past kMaxWorkloadBytes, so only the size bound can reject it.
    const auto big = temp_file("big.json");
    {
        nlohmann::json j = minimal();
        j["description"] = std::string(17ULL << 20, 'x');
        std::ofstream(big, std::ios::binary) << j.dump();
    }
    test::expect_error(ErrorCode::Config, [&] { (void)load_workload(big); });
    const auto small = temp_file("small.json");
    { std::ofstream(small, std::ios::binary) << minimal().dump(); }
    EXPECT_NO_THROW((void)load_workload(small));
    std::filesystem::remove(small);
    std::filesystem::remove(invalid);
    std::filesystem::remove(big);
}
