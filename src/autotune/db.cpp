#include "halo/autotune/db.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>

#include <nlohmann/json.hpp>

#include "halo/core/log.h"
#include "sqlite.h"

namespace halo::autotune {

// ---- errors -------------------------------------------------------------------------------

std::string_view to_string(DbErrorKind k) noexcept {
    switch (k) {
        case DbErrorKind::Corrupt: return "corrupt";
        case DbErrorKind::NotHaloDb: return "not_halo_db";
        case DbErrorKind::ForeignVersion: return "foreign_version";
        case DbErrorKind::Busy: return "busy";
        case DbErrorKind::Io: return "io";
        case DbErrorKind::Constraint: return "constraint";
        case DbErrorKind::Internal: return "internal";
    }
    return "?";
}

namespace {

ErrorCode code_for(DbErrorKind k) noexcept {
    switch (k) {
        case DbErrorKind::Corrupt:
        case DbErrorKind::Busy:
        case DbErrorKind::Io: return ErrorCode::Io;
        case DbErrorKind::ForeignVersion: return ErrorCode::Unsupported;
        default: return ErrorCode::Config;
    }
}

}  // namespace

ProfileDbError::ProfileDbError(DbErrorKind kind, const std::string& message)
    : Error(code_for(kind), std::format("profile db ({}): {}", to_string(kind), message)), kind_(kind) {}

// ---- schema -------------------------------------------------------------------------------

namespace {

constexpr std::array<std::string_view, 8> kTables{
    "benchmark_run",    "hardware_profile", "kernel_candidate", "model_profile",
    "operator_profile", "schema_version",   "tier_bandwidth",   "winning_configuration"};

constexpr const char* kSchemaV1 = R"SQL(
CREATE TABLE schema_version(
  id INTEGER PRIMARY KEY CHECK (id = 1),
  app TEXT NOT NULL,
  version INTEGER NOT NULL);
CREATE TABLE hardware_profile(
  id INTEGER PRIMARY KEY,
  halo_version TEXT NOT NULL, gpu_device TEXT NOT NULL, gpu_arch TEXT NOT NULL,
  driver_version TEXT NOT NULL, rocm_version TEXT NOT NULL, vulkan_version TEXT NOT NULL,
  kernel_version TEXT NOT NULL, os TEXT NOT NULL, power_mode TEXT NOT NULL, isa_target TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(halo_version, gpu_device, gpu_arch, driver_version, rocm_version, vulkan_version,
         kernel_version, os, power_mode, isa_target));
CREATE TABLE tier_bandwidth(
  id INTEGER PRIMARY KEY,
  hardware_profile_id INTEGER NOT NULL REFERENCES hardware_profile(id) ON DELETE CASCADE,
  tier TEXT NOT NULL, processor TEXT NOT NULL, label TEXT NOT NULL,
  threads INTEGER NOT NULL, buffer_bytes INTEGER NOT NULL,
  read_gbps REAL NOT NULL, write_gbps REAL NOT NULL, copy_gbps REAL NOT NULL,
  stats_json TEXT NOT NULL, measured_at TEXT NOT NULL);
CREATE TABLE model_profile(
  id INTEGER PRIMARY KEY,
  model_hash TEXT NOT NULL, pack_id TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(model_hash, pack_id));
CREATE TABLE operator_profile(
  id INTEGER PRIMARY KEY,
  family TEXT NOT NULL, shape TEXT NOT NULL,
  UNIQUE(family, shape));
CREATE TABLE kernel_candidate(
  id INTEGER PRIMARY KEY,
  operator_profile_id INTEGER NOT NULL REFERENCES operator_profile(id) ON DELETE CASCADE,
  backend TEXT NOT NULL, params TEXT NOT NULL,
  UNIQUE(operator_profile_id, backend, params));
CREATE TABLE benchmark_run(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('tune', 'baseline', 'bench')),
  hardware_profile_id INTEGER REFERENCES hardware_profile(id),
  model_profile_id INTEGER REFERENCES model_profile(id),
  kernel_candidate_id INTEGER REFERENCES kernel_candidate(id),
  engine TEXT NOT NULL, backend TEXT NOT NULL, pack TEXT NOT NULL,
  context INTEGER NOT NULL, mode TEXT NOT NULL, power_mode TEXT NOT NULL,
  strategy TEXT, stability TEXT, median REAL, rejected TEXT,
  stats_json TEXT, record_json TEXT,
  created_at TEXT NOT NULL);
CREATE INDEX benchmark_run_latest ON benchmark_run(kind, backend, pack, context, mode, created_at);
CREATE TABLE winning_configuration(
  id INTEGER PRIMARY KEY,
  hardware_profile_id INTEGER NOT NULL REFERENCES hardware_profile(id),
  model_profile_id INTEGER NOT NULL REFERENCES model_profile(id),
  operator_profile_id INTEGER NOT NULL REFERENCES operator_profile(id),
  backend TEXT NOT NULL,
  kernel_candidate_id INTEGER NOT NULL REFERENCES kernel_candidate(id),
  benchmark_run_id INTEGER REFERENCES benchmark_run(id),
  strategy TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(hardware_profile_id, model_profile_id, operator_profile_id, backend));
)SQL";

bool is_known_table(std::string_view t) {
    return std::find(kTables.begin(), kTables.end(), t) != kTables.end();
}

}  // namespace

struct ProfileDb::Impl {
    sql::Connection conn;
    bool read_only;
    Impl(const std::string& path, bool ro, int busy) : conn(path, ro, busy), read_only(ro) {}
};

namespace {

int read_version(sql::Connection& c) {
    sql::Statement s(c, "SELECT app, version FROM schema_version WHERE id = 1");
    if (!s.step()) throw ProfileDbError(DbErrorKind::NotHaloDb, "schema_version has no row");
    const std::string app = s.text(0);
    if (app != kDbApp) throw ProfileDbError(DbErrorKind::NotHaloDb, std::format("schema_version.app is '{}'", app));
    return static_cast<int>(s.int64(1));
}

void create_v1(sql::Connection& c) {
    sql::Transaction tx(c);
    c.exec(kSchemaV1);
    sql::Statement ins(c, "INSERT INTO schema_version(id, app, version) VALUES (1, ?1, ?2)");
    ins.bind(1, kDbApp).bind(2, std::int64_t{1});
    ins.step();
    c.exec(std::format("PRAGMA application_id = {}", kApplicationId));
    tx.commit();
}

void migrate(sql::Connection& c, int from, const OpenOptions& o) {
    for (int v = from; v < o.target_version; ++v) {
        const auto it = std::find_if(o.migrations.begin(), o.migrations.end(),
                                     [v](const Migration& m) { return m.from_version == v; });
        if (it == o.migrations.end() || !it->apply) {
            throw ProfileDbError(DbErrorKind::ForeignVersion,
                                 std::format("schema version {} is older than {} and no migration from {} exists", from,
                                             o.target_version, v));
        }
        sql::Transaction tx(c);
        it->apply([&c](const std::string& statement) { c.exec(statement); });
        sql::Statement up(c, "UPDATE schema_version SET version = ?1 WHERE id = 1");
        up.bind(1, static_cast<std::int64_t>(v + 1));
        up.step();
        tx.commit();
        HALO_INFO("autotune", "profile db migrated from schema {} to {}", v, v + 1);
    }
}

}  // namespace

ProfileDb ProfileDb::open(const std::filesystem::path& path, const OpenOptions& o) {
    HALO_CHECK(o.target_version >= 1, ErrorCode::Config, "profile db: target_version {} < 1", o.target_version);
    std::error_code ec;
    if (o.read_only && !std::filesystem::exists(path, ec)) {
        throw ProfileDbError(DbErrorKind::Io, std::format("open {}: no such file", path.string()));
    }
    auto impl = std::make_unique<Impl>(path.string(), o.read_only, o.busy_timeout_ms);
    auto& c = impl->conn;
    // The first real read: surfaces SQLITE_NOTADB / SQLITE_CORRUPT for a foreign file.
    const std::int64_t app_id = c.query_int("PRAGMA application_id");
    const std::int64_t n_objects = c.query_int("SELECT count(*) FROM sqlite_master");
    if (n_objects == 0 && app_id == 0) {
        if (o.read_only) {
            throw ProfileDbError(DbErrorKind::NotHaloDb,
                                 std::format("{} is empty (not initialised by `halo tune`)", path.string()));
        }
        create_v1(c);
    } else if (app_id != kApplicationId) {
        throw ProfileDbError(DbErrorKind::NotHaloDb,
                             std::format("{} has application_id {:#x}, not HALO ({:#x})", path.string(), app_id,
                                         kApplicationId));
    }
    int version = 0;
    try {
        version = read_version(c);
    } catch (const ProfileDbError& e) {
        if (e.kind() == DbErrorKind::Internal) {  // e.g. "no such table: schema_version"
            throw ProfileDbError(DbErrorKind::NotHaloDb, e.what());
        }
        throw;
    }
    if (version > o.target_version) {
        throw ProfileDbError(DbErrorKind::ForeignVersion,
                             std::format("{} has schema version {}; this build supports {}", path.string(), version,
                                         o.target_version));
    }
    if (version < o.target_version) {
        if (o.read_only) {
            throw ProfileDbError(DbErrorKind::ForeignVersion,
                                 std::format("{} has schema version {} < {}; open it read-write (halo tune) to migrate",
                                             path.string(), version, o.target_version));
        }
        migrate(c, version, o);
    }
    const std::string qc = c.query_text("PRAGMA quick_check");
    if (qc != "ok") throw ProfileDbError(DbErrorKind::Corrupt, std::format("quick_check: {}", qc));
    c.exec("PRAGMA foreign_keys = ON");
    if (!o.read_only) {
        const std::string mode = c.query_text("PRAGMA journal_mode = WAL");
        if (mode != "wal") throw ProfileDbError(DbErrorKind::Io, std::format("journal_mode is '{}', not wal", mode));
    }
    return ProfileDb(std::move(impl));
}

ProfileDb::ProfileDb(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ProfileDb::ProfileDb(ProfileDb&&) noexcept = default;
ProfileDb& ProfileDb::operator=(ProfileDb&&) noexcept = default;
ProfileDb::~ProfileDb() = default;

int ProfileDb::schema_version() const { return read_version(impl_->conn); }

std::string ProfileDb::journal_mode() const { return impl_->conn.query_text("PRAGMA journal_mode"); }

std::vector<std::string> ProfileDb::table_names() const {
    std::vector<std::string> out;
    sql::Statement s(impl_->conn, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name");
    while (s.step()) out.push_back(s.text(0));
    return out;
}

std::int64_t ProfileDb::count(std::string_view table) const {
    HALO_CHECK(is_known_table(table), ErrorCode::Config, "profile db: '{}' is not a schema table", table);
    // The name is one of the fixed identifiers above (checked), never caller text.
    return impl_->conn.query_int(std::format("SELECT count(*) FROM {}", table));
}

// ---- helpers: ensure rows -----------------------------------------------------------------

namespace {

void require_writable(const ProfileDb::Impl& impl) {
    if (impl.read_only) throw ProfileDbError(DbErrorKind::Io, "database was opened read-only");
}

std::int64_t ensure_hardware(sql::Connection& c, const ProfileKey& k) {
    sql::Statement ins(c,
                       "INSERT INTO hardware_profile(halo_version, gpu_device, gpu_arch, driver_version, rocm_version, "
                       "vulkan_version, kernel_version, os, power_mode, isa_target, created_at) "
                       "VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11) ON CONFLICT DO NOTHING");
    ins.bind(1, k.halo_version).bind(2, k.gpu_device).bind(3, k.gpu_arch).bind(4, k.driver_version);
    ins.bind(5, k.rocm_version).bind(6, k.vulkan_version).bind(7, k.kernel_version).bind(8, k.os);
    ins.bind(9, k.power_mode).bind(10, k.isa_target).bind(11, sql::utc_now());
    ins.step();
    sql::Statement sel(c,
                       "SELECT id FROM hardware_profile WHERE halo_version=?1 AND gpu_device=?2 AND gpu_arch=?3 AND "
                       "driver_version=?4 AND rocm_version=?5 AND vulkan_version=?6 AND kernel_version=?7 AND os=?8 "
                       "AND power_mode=?9 AND isa_target=?10");
    sel.bind(1, k.halo_version).bind(2, k.gpu_device).bind(3, k.gpu_arch).bind(4, k.driver_version);
    sel.bind(5, k.rocm_version).bind(6, k.vulkan_version).bind(7, k.kernel_version).bind(8, k.os);
    sel.bind(9, k.power_mode).bind(10, k.isa_target);
    if (!sel.step()) throw ProfileDbError(DbErrorKind::Internal, "hardware_profile row vanished");
    return sel.int64(0);
}

std::optional<std::int64_t> find_hardware(sql::Connection& c, const ProfileKey& k) {
    sql::Statement sel(c,
                       "SELECT id FROM hardware_profile WHERE halo_version=?1 AND gpu_device=?2 AND gpu_arch=?3 AND "
                       "driver_version=?4 AND rocm_version=?5 AND vulkan_version=?6 AND kernel_version=?7 AND os=?8 "
                       "AND power_mode=?9 AND isa_target=?10");
    sel.bind(1, k.halo_version).bind(2, k.gpu_device).bind(3, k.gpu_arch).bind(4, k.driver_version);
    sel.bind(5, k.rocm_version).bind(6, k.vulkan_version).bind(7, k.kernel_version).bind(8, k.os);
    sel.bind(9, k.power_mode).bind(10, k.isa_target);
    if (!sel.step()) return std::nullopt;
    return sel.int64(0);
}

std::int64_t ensure_model(sql::Connection& c, const ProfileKey& k) {
    sql::Statement ins(c,
                       "INSERT INTO model_profile(model_hash, pack_id, created_at) VALUES (?1, ?2, ?3) "
                       "ON CONFLICT DO NOTHING");
    ins.bind(1, k.model_hash).bind(2, k.pack_id).bind(3, sql::utc_now());
    ins.step();
    sql::Statement sel(c, "SELECT id FROM model_profile WHERE model_hash = ?1 AND pack_id = ?2");
    sel.bind(1, k.model_hash).bind(2, k.pack_id);
    if (!sel.step()) throw ProfileDbError(DbErrorKind::Internal, "model_profile row vanished");
    return sel.int64(0);
}

std::int64_t ensure_operator(sql::Connection& c, const OpKey& op) {
    sql::Statement ins(c, "INSERT INTO operator_profile(family, shape) VALUES (?1, ?2) ON CONFLICT DO NOTHING");
    ins.bind(1, op.family).bind(2, op.shape);
    ins.step();
    sql::Statement sel(c, "SELECT id FROM operator_profile WHERE family = ?1 AND shape = ?2");
    sel.bind(1, op.family).bind(2, op.shape);
    if (!sel.step()) throw ProfileDbError(DbErrorKind::Internal, "operator_profile row vanished");
    return sel.int64(0);
}

std::int64_t ensure_candidate(sql::Connection& c, std::int64_t op_id, const std::string& backend,
                              const std::string& params) {
    sql::Statement ins(c,
                       "INSERT INTO kernel_candidate(operator_profile_id, backend, params) VALUES (?1, ?2, ?3) "
                       "ON CONFLICT DO NOTHING");
    ins.bind(1, op_id).bind(2, backend).bind(3, params);
    ins.step();
    sql::Statement sel(c,
                       "SELECT id FROM kernel_candidate WHERE operator_profile_id = ?1 AND backend = ?2 AND "
                       "params = ?3");
    sel.bind(1, op_id).bind(2, backend).bind(3, params);
    if (!sel.step()) throw ProfileDbError(DbErrorKind::Internal, "kernel_candidate row vanished");
    return sel.int64(0);
}

}  // namespace

void ProfileDb::record_tier_bandwidth(const ProfileKey& key, const hardware::TierBandwidth& bw) {
    require_writable(*impl_);
    auto& c = impl_->conn;
    sql::Transaction tx(c);
    const std::int64_t hw = ensure_hardware(c, key);
    sql::Statement ins(c,
                       "INSERT INTO tier_bandwidth(hardware_profile_id, tier, processor, label, threads, buffer_bytes, "
                       "read_gbps, write_gbps, copy_gbps, stats_json, measured_at) "
                       "VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)");
    ins.bind(1, hw).bind(2, hardware::to_string(bw.tier)).bind(3, hardware::to_string(bw.processor));
    ins.bind(4, bw.label).bind(5, static_cast<std::int64_t>(bw.threads));
    ins.bind(6, static_cast<std::int64_t>(bw.buffer_bytes));
    ins.bind(7, bw.read_gbps.median).bind(8, bw.write_gbps.median).bind(9, bw.copy_gbps.median);
    ins.bind(10, nlohmann::json(bw).dump()).bind(11, sql::utc_now());
    ins.step();
    tx.commit();
}

std::vector<TierBandwidthRow> ProfileDb::tier_bandwidth(const ProfileKey& key) const {
    std::vector<TierBandwidthRow> out;
    const auto hw = find_hardware(impl_->conn, key);
    if (!hw) return out;
    sql::Statement s(impl_->conn,
                     "SELECT tier, processor, label, read_gbps, write_gbps, copy_gbps, measured_at FROM tier_bandwidth "
                     "WHERE hardware_profile_id = ?1 ORDER BY id");
    s.bind(1, *hw);
    while (s.step()) {
        out.push_back({s.text(0), s.text(1), s.text(2), s.real(3), s.real(4), s.real(5), s.text(6)});
    }
    return out;
}

bool ProfileDb::persist_tune(const ProfileKey& key, const OpTuneResult& r) {
    require_writable(*impl_);
    auto& c = impl_->conn;
    sql::Transaction tx(c);
    const std::int64_t hw = ensure_hardware(c, key);
    const std::int64_t model = ensure_model(c, key);
    const std::int64_t op = ensure_operator(c, r.key);
    const std::string now = sql::utc_now();
    std::optional<std::int64_t> winner_run;
    std::optional<std::int64_t> winner_cand;
    const std::string winner_text = r.winner ? r.winner->to_string() : std::string();
    for (const auto& cr : r.candidates) {
        const std::string params = cr.candidate.to_string();
        const std::int64_t cand = ensure_candidate(c, op, r.backend, params);
        if (!cr.measured) continue;
        sql::Statement ins(c,
                           "INSERT INTO benchmark_run(kind, hardware_profile_id, model_profile_id, kernel_candidate_id, "
                           "engine, backend, pack, context, mode, power_mode, strategy, stability, median, rejected, "
                           "stats_json, record_json, created_at) "
                           "VALUES ('tune', ?1, ?2, ?3, 'halo', ?4, ?5, 0, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)");
        ins.bind(1, hw).bind(2, model).bind(3, cand).bind(4, r.backend).bind(5, key.pack_id);
        ins.bind(6, r.key.family + "/" + r.key.shape).bind(7, key.power_mode).bind(8, to_string(r.strategy));
        ins.bind(9, profiling::to_string(cr.stability)).bind(10, cr.stats.median).bind(11, cr.rejected);
        ins.bind(12, nlohmann::json(cr.stats).dump()).bind(13, nlohmann::json(cr).dump()).bind(14, now);
        ins.step();
        if (r.winner && params == winner_text) {
            winner_run = c.last_insert_rowid();
            winner_cand = cand;
        }
    }
    bool wrote = false;
    if (r.winner) {
        HALO_CHECK(winner_cand.has_value(), ErrorCode::Config,
                   "persist_tune: winner {} is not a measured candidate of the result", winner_text);
        sql::Statement up(c,
                          "INSERT INTO winning_configuration(hardware_profile_id, model_profile_id, "
                          "operator_profile_id, kernel_candidate_id, benchmark_run_id, strategy, metric, value, "
                          "created_at, backend) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10) "
                          "ON CONFLICT(hardware_profile_id, model_profile_id, operator_profile_id, backend) DO UPDATE SET "
                          "kernel_candidate_id = excluded.kernel_candidate_id, "
                          "benchmark_run_id = excluded.benchmark_run_id, strategy = excluded.strategy, "
                          "metric = excluded.metric, value = excluded.value, created_at = excluded.created_at");
        up.bind(1, hw).bind(2, model).bind(3, op).bind(4, *winner_cand).bind(5, *winner_run);
        up.bind(6, to_string(r.strategy)).bind(7, kWinnerMetric).bind(8, r.winner_median_ns).bind(9, now);
        up.bind(10, r.backend);
        up.step();
        wrote = true;
    }
    tx.commit();
    return wrote;
}

std::int64_t ProfileDb::add_benchmark_record(const profiling::BenchmarkRecord& rec, std::string_view kind) {
    require_writable(*impl_);
    HALO_CHECK(kind == "baseline" || kind == "bench", ErrorCode::Config,
               "add_benchmark_record: kind '{}' is not baseline/bench", kind);
    HALO_CHECK(rec.context <= static_cast<std::uint64_t>(INT64_MAX), ErrorCode::Config,
               "add_benchmark_record: context too large");
    auto& c = impl_->conn;
    sql::Transaction tx(c);
    sql::Statement ins(c,
                       "INSERT INTO benchmark_run(kind, engine, backend, pack, context, mode, power_mode, median, "
                       "record_json, created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)");
    ins.bind(1, kind).bind(2, rec.engine).bind(3, rec.backend).bind(4, rec.pack);
    ins.bind(5, static_cast<std::int64_t>(rec.context)).bind(6, rec.mode).bind(7, rec.power_mode);
    ins.bind(8, rec.decode_tps).bind(9, nlohmann::json(rec).dump()).bind(10, sql::utc_now());
    ins.step();
    const std::int64_t id = c.last_insert_rowid();
    tx.commit();
    return id;
}

std::optional<profiling::BenchmarkRecord> ProfileDb::latest_run(std::string_view kind, std::string_view backend,
                                                                std::string_view pack, std::uint64_t context,
                                                                std::string_view mode) const {
    HALO_CHECK(context <= static_cast<std::uint64_t>(INT64_MAX), ErrorCode::Config, "latest_run: context too large");
    sql::Statement s(impl_->conn,
                     "SELECT record_json FROM benchmark_run WHERE kind = ?1 AND backend = ?2 AND pack = ?3 AND "
                     "context = ?4 AND mode = ?5 AND record_json IS NOT NULL ORDER BY created_at DESC, id DESC LIMIT 1");
    s.bind(1, kind).bind(2, backend).bind(3, pack).bind(4, static_cast<std::int64_t>(context)).bind(5, mode);
    if (!s.step()) return std::nullopt;
    const auto j = nlohmann::json::parse(s.text(0), nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) throw ProfileDbError(DbErrorKind::Corrupt, "benchmark_run.record_json is not JSON");
    return j.get<profiling::BenchmarkRecord>();  // Error(Config) on a foreign/malformed record
}

}  // namespace halo::autotune
