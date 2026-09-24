#include <gtest/gtest.h>
#include <sqlite3.h>

#include <fstream>

#include <nlohmann/json.hpp>

#include "autotune_test_util.h"
#include "halo/autotune/db.h"
#include "halo/autotune/lookup.h"

using namespace halo::autotune;
using halo::ErrorCode;
using test::TempDb;

namespace {

/// Raw SQLite access for building foreign fixtures (not through ProfileDb).
void raw_exec(const std::filesystem::path& p, const std::string& sql) {
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(p.string().c_str(), &db), SQLITE_OK);
    char* err = nullptr;
    const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    EXPECT_EQ(rc, SQLITE_OK) << (err != nullptr ? err : "");
    sqlite3_free(err);
    sqlite3_close(db);
}

OpTuneResult result_with(const std::string& winner, double median, std::vector<std::string> measured) {
    OpTuneResult r;
    r.key = {"MATMUL", "T=1,K=4,N=4,w=f32"};
    r.backend = "cpu";
    r.strategy = Strategy::Exhaustive;
    for (const auto& m : measured) {
        CandidateResult cr;
        cr.candidate = Candidate::parse(m);
        cr.measured = true;
        cr.stats.n = 20;
        cr.stats.median = m == winner ? median : median * 2;
        cr.stability = halo::profiling::Stability::Stable;
        r.candidates.push_back(cr);
    }
    if (!winner.empty()) {
        r.winner = Candidate::parse(winner);
        r.winner_median_ns = median;
    }
    return r;
}

}  // namespace

TEST(ProfileDb, CreatesExactlyTheTrd58Schema) {
    TempDb t("schema");
    ProfileDb db = ProfileDb::open(t.path());
    EXPECT_EQ(db.table_names(),
              (std::vector<std::string>{"benchmark_run", "hardware_profile", "kernel_candidate", "model_profile",
                                        "operator_profile", "schema_version", "tier_bandwidth",
                                        "winning_configuration"}));
    EXPECT_EQ(db.schema_version(), 1);
    EXPECT_EQ(db.journal_mode(), "wal");
    for (const char* tbl : {"hardware_profile", "winning_configuration", "benchmark_run"}) {
        EXPECT_EQ(db.count(tbl), 0) << tbl;
    }
    test::expect_error(ErrorCode::Config, [&] { (void)db.count("sqlite_master; DROP TABLE x"); });
}

TEST(ProfileDb, PersistTuneRoundTripAndWinnerReplacement) {
    TempDb t("roundtrip");
    const ProfileKey key = test::sample_key();
    {
        ProfileDb db = ProfileDb::open(t.path());
        EXPECT_TRUE(db.persist_tune(key, result_with("threads=4", 100.0, {"threads=1", "threads=4"})));
        EXPECT_EQ(db.count("hardware_profile"), 1);
        EXPECT_EQ(db.count("model_profile"), 1);
        EXPECT_EQ(db.count("operator_profile"), 1);
        EXPECT_EQ(db.count("kernel_candidate"), 2);
        EXPECT_EQ(db.count("benchmark_run"), 2);
        EXPECT_EQ(db.count("winning_configuration"), 1);
    }
    {
        const auto lk = ProfileLookup::open(t.path());
        const auto r = lk.find(key, {"MATMUL", "T=1,K=4,N=4,w=f32"}, "cpu");
        EXPECT_EQ(r.source, SelectionSource::Exact);
        ASSERT_TRUE(r.candidate);
        EXPECT_EQ(r.candidate->to_string(), "threads=4");
        EXPECT_DOUBLE_EQ(r.value, 100.0);
        EXPECT_EQ(r.strategy, "exhaustive");
        EXPECT_EQ(r.created_at.size(), 20u);  // UTC "YYYY-MM-DDTHH:MM:SSZ"
        EXPECT_EQ(r.created_at.back(), 'Z');
    }
    {
        ProfileDb db = ProfileDb::open(t.path());
        // A re-tune with a new winner replaces the old one (same hardware/model/operator).
        EXPECT_TRUE(db.persist_tune(key, result_with("threads=1", 50.0, {"threads=1", "threads=4"})));
        EXPECT_EQ(db.count("winning_configuration"), 1);
        EXPECT_EQ(db.count("kernel_candidate"), 2);  // candidates deduplicated
        EXPECT_EQ(db.count("benchmark_run"), 4);
        // A tune without a winner records its runs but leaves the winner in place.
        EXPECT_FALSE(db.persist_tune(key, result_with("", 10.0, {"threads=4"})));
        EXPECT_EQ(db.count("benchmark_run"), 5);
    }
    const auto r = ProfileLookup::open(t.path()).find(key, {"MATMUL", "T=1,K=4,N=4,w=f32"}, "cpu");
    ASSERT_TRUE(r.candidate);
    EXPECT_EQ(r.candidate->to_string(), "threads=1");
    EXPECT_DOUBLE_EQ(r.value, 50.0);
}

TEST(ProfileDb, WinnerMustBeAMeasuredCandidateAndWritesAreAtomic) {
    TempDb t("atomic");
    ProfileDb db = ProfileDb::open(t.path());
    OpTuneResult bad = result_with("threads=4", 1.0, {"threads=1"});  // winner never measured
    test::expect_error(ErrorCode::Config, [&] { (void)db.persist_tune(test::sample_key(), bad); });
    // The failed transaction rolled back everything it had inserted.
    EXPECT_EQ(db.count("hardware_profile"), 0);
    EXPECT_EQ(db.count("benchmark_run"), 0);
}

TEST(ProfileDb, ValuesAreBoundNotFormatted) {
    TempDb t("bind");
    ProfileKey key = test::sample_key();
    key.os = "x'); DROP TABLE winning_configuration; --";
    key.model_hash = std::string("nul\0byte", 8);
    {
        ProfileDb db = ProfileDb::open(t.path());
        EXPECT_TRUE(db.persist_tune(key, result_with("threads=2", 7.0, {"threads=2"})));
        EXPECT_EQ(db.count("winning_configuration"), 1);
    }
    const auto r = ProfileLookup::open(t.path()).find(key, {"MATMUL", "T=1,K=4,N=4,w=f32"}, "cpu");
    EXPECT_EQ(r.source, SelectionSource::Exact);  // exact bytes survived, incl. the NUL
}

TEST(ProfileDb, TierBandwidthAndBenchmarkRecords) {
    TempDb t("records");
    ProfileDb db = ProfileDb::open(t.path());
    halo::hardware::TierBandwidth bw;
    bw.tier = halo::hardware::MemoryTier::Host;
    bw.label = "dev-host";
    bw.threads = 4;
    bw.read_gbps.median = 40.0;
    bw.write_gbps.median = 20.0;
    bw.copy_gbps.median = 30.0;
    db.record_tier_bandwidth(test::sample_key(), bw);
    const auto rows = db.tier_bandwidth(test::sample_key());
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].tier, "HOST");
    EXPECT_DOUBLE_EQ(rows[0].read_gbps, 40.0);
    EXPECT_TRUE(db.tier_bandwidth(ProfileKey{}).empty());

    halo::profiling::BenchmarkRecord rec;
    rec.model = "m";
    rec.backend = "vulkan";
    rec.pack = "UD-Q4_K_XL";
    rec.context = 4096;
    rec.mode = "tg128";
    rec.engine = "llama-bench";
    rec.power_mode = "p|a/b";
    rec.decode_tps = 11.0;
    (void)db.add_benchmark_record(rec, "baseline");
    rec.decode_tps = 12.0;
    (void)db.add_benchmark_record(rec, "baseline");  // same second: the higher id is latest
    rec.context = 32768;
    rec.decode_tps = 9.0;
    (void)db.add_benchmark_record(rec, "baseline");
    const auto latest = db.latest_run("baseline", "vulkan", "UD-Q4_K_XL", 4096, "tg128");
    ASSERT_TRUE(latest);
    EXPECT_DOUBLE_EQ(*latest->decode_tps, 12.0);
    EXPECT_FALSE(db.latest_run("baseline", "hip", "UD-Q4_K_XL", 4096, "tg128"));
    EXPECT_FALSE(db.latest_run("bench", "vulkan", "UD-Q4_K_XL", 4096, "tg128"));
    test::expect_error(ErrorCode::Config, [&] { (void)db.add_benchmark_record(rec, "tune"); });
}

// ---- typed errors on bad files --------------------------------------------------------------

TEST(ProfileDbErrors, RandomBytesAreCorrupt) {
    TempDb t("garbage");
    {
        std::ofstream out(t.path(), std::ios::binary);
        for (int i = 0; i < 8192; ++i) out.put(static_cast<char>((i * 131 + 7) & 0xff));
    }
    test::expect_db_error(DbErrorKind::Corrupt, [&] { (void)ProfileDb::open(t.path()); });
    test::expect_db_error(DbErrorKind::Corrupt, [&] { (void)ProfileLookup::open(t.path()); });
}

TEST(ProfileDbErrors, ForeignSqliteDbIsNotHalo) {
    TempDb t("foreign");
    raw_exec(t.path(), "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT); INSERT INTO users VALUES (1, 'x');");
    test::expect_db_error(DbErrorKind::NotHaloDb, [&] { (void)ProfileDb::open(t.path()); });
    test::expect_db_error(DbErrorKind::NotHaloDb, [&] { (void)ProfileLookup::open(t.path()); });
    // HALO application id but no schema_version table.
    TempDb t2("noversion");
    raw_exec(t2.path(), "PRAGMA application_id = 1212238927; CREATE TABLE x(a);");
    test::expect_db_error(DbErrorKind::NotHaloDb, [&] { (void)ProfileDb::open(t2.path()); });
    // HALO application id, schema_version from another app.
    TempDb t3("otherapp");
    raw_exec(t3.path(),
             "PRAGMA application_id = 1212238927; CREATE TABLE schema_version(id INTEGER PRIMARY KEY, app TEXT, "
             "version INTEGER); INSERT INTO schema_version VALUES (1, 'other', 1);");
    test::expect_db_error(DbErrorKind::NotHaloDb, [&] { (void)ProfileDb::open(t3.path()); });
    // A HALO-shaped schema_version row but no HALO application id (e.g. a copied table).
    TempDb t4("noappid");
    raw_exec(t4.path(),
             "CREATE TABLE schema_version(id INTEGER PRIMARY KEY, app TEXT, version INTEGER); "
             "INSERT INTO schema_version VALUES (1, 'halo.profile', 1);");
    test::expect_db_error(DbErrorKind::NotHaloDb, [&] { (void)ProfileDb::open(t4.path()); });
}

TEST(ProfileDbErrors, NewerSchemaVersionIsForeign) {
    TempDb t("v99");
    { (void)ProfileDb::open(t.path()); }
    raw_exec(t.path(), "UPDATE schema_version SET version = 99;");
    test::expect_db_error(DbErrorKind::ForeignVersion, [&] { (void)ProfileDb::open(t.path()); });
    test::expect_db_error(DbErrorKind::ForeignVersion, [&] { (void)ProfileLookup::open(t.path()); });
    try {
        (void)ProfileDb::open(t.path());
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Unsupported);  // still a halo::Error for generic handlers
    }
}

TEST(ProfileDbErrors, OverwrittenPageIsCorrupt) {
    TempDb t("pages");
    {
        ProfileDb db = ProfileDb::open(t.path());
        halo::profiling::BenchmarkRecord rec;
        rec.backend = "cpu";
        rec.power_mode = "p|a/b";
        rec.extra["padding"] = std::string(2000, 'z');
        for (int i = 0; i < 64; ++i) {
            rec.context = static_cast<std::uint64_t>(i);
            (void)db.add_benchmark_record(rec, "bench");
        }
    }  // closed: WAL checkpointed into the main file
    const auto size = std::filesystem::file_size(t.path());
    ASSERT_GT(size, 12u * 4096u);
    {
        std::fstream f(t.path(), std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(4096 * 2);
        for (int i = 0; i < 4096 * 8; ++i) f.put(static_cast<char>(0xA5));
    }
    test::expect_db_error(DbErrorKind::Corrupt, [&] { (void)ProfileDb::open(t.path()); });
}

TEST(ProfileDbErrors, EmptyAndMissingFiles) {
    TempDb t("empty");
    { std::ofstream(t.path(), std::ios::binary); }  // 0 bytes
    test::expect_db_error(DbErrorKind::NotHaloDb, [&] { (void)ProfileLookup::open(t.path()); });
    {
        ProfileDb db = ProfileDb::open(t.path());  // read-write: initialises schema v1
        EXPECT_EQ(db.schema_version(), 1);
    }
    EXPECT_EQ(ProfileLookup::open(t.path()).size(), 0u);
    TempDb missing("missing");
    test::expect_db_error(DbErrorKind::Io, [&] { (void)ProfileLookup::open(missing.path()); });
    EXPECT_FALSE(std::filesystem::exists(missing.path()));  // a read-only open never creates
}

TEST(ProfileDbErrors, ReadOnlyOpenRefusesWrites) {
    TempDb t("ro");
    { (void)ProfileDb::open(t.path()); }
    OpenOptions ro;
    ro.read_only = true;
    ProfileDb db = ProfileDb::open(t.path(), ro);
    test::expect_db_error(DbErrorKind::Io,
                          [&] { (void)db.persist_tune(test::sample_key(), result_with("threads=1", 1, {"threads=1"})); });
}

TEST(ProfileDbErrors, LockedDatabaseIsBusyAfterTimeout) {
    TempDb t("busy");
    { (void)ProfileDb::open(t.path()); }
    sqlite3* other = nullptr;  // a second writer (another process in production)
    ASSERT_EQ(sqlite3_open(t.path().string().c_str(), &other), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(other, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr), SQLITE_OK);
    OpenOptions o;
    o.busy_timeout_ms = 100;
    // The lock may surface at open or at the write; either way it must be typed Busy.
    test::expect_db_error(DbErrorKind::Busy, [&] {
        ProfileDb db = ProfileDb::open(t.path(), o);
        (void)db.persist_tune(test::sample_key(), result_with("threads=1", 1, {"threads=1"}));
    });
    sqlite3_exec(other, "COMMIT", nullptr, nullptr, nullptr);
    sqlite3_close(other);
    ProfileDb db = ProfileDb::open(t.path(), o);  // lock released: the write goes through
    EXPECT_TRUE(db.persist_tune(test::sample_key(), result_with("threads=1", 1, {"threads=1"})));
}

// ---- forward migration hook -----------------------------------------------------------------

TEST(ProfileDbMigration, AppliesInOrderAndFailedMigrationKeepsVersion) {
    TempDb t("migrate");
    { (void)ProfileDb::open(t.path()); }
    OpenOptions failing;
    failing.target_version = 2;
    failing.migrations.push_back({1, [](const auto& exec) {
                                      exec("ALTER TABLE operator_profile ADD COLUMN note TEXT");
                                      throw std::runtime_error("migration step failed");
                                  }});
    EXPECT_THROW((void)ProfileDb::open(t.path(), failing), std::runtime_error);
    {
        ProfileDb db = ProfileDb::open(t.path());
        EXPECT_EQ(db.schema_version(), 1);  // rolled back, column not added either
    }
    OpenOptions missing;
    missing.target_version = 2;  // no migration registered from 1
    test::expect_db_error(DbErrorKind::ForeignVersion, [&] { (void)ProfileDb::open(t.path(), missing); });

    OpenOptions v2;
    v2.target_version = 2;
    v2.migrations.push_back({1, [](const auto& exec) { exec("ALTER TABLE operator_profile ADD COLUMN note TEXT"); }});
    {
        ProfileDb db = ProfileDb::open(t.path(), v2);
        EXPECT_EQ(db.schema_version(), 2);
    }
    // This build (v1) must now refuse the v2 file, read-write and read-only.
    test::expect_db_error(DbErrorKind::ForeignVersion, [&] { (void)ProfileDb::open(t.path()); });
    test::expect_db_error(DbErrorKind::ForeignVersion, [&] { (void)ProfileLookup::open(t.path()); });
}
