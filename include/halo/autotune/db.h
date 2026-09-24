#pragma once
// Persistent profile database (TRD §58): SQLite, schema v1.
//
// Tables (exactly the TRD §58 set plus the version table):
//   schema_version, hardware_profile, tier_bandwidth, model_profile, operator_profile,
//   kernel_candidate, benchmark_run, winning_configuration
// The profile key (TRD §57) is split across hardware_profile (the 10 hardware/software
// fields) and model_profile (MODEL_HASH, PACK_ID).
//
// Access rules: prepared statements only (values are always bound, never formatted into
// SQL); writes happen inside BEGIN IMMEDIATE transactions; WAL journal mode; a busy
// timeout; foreign keys on. The file header carries application_id 0x48414C4F ("HALO").
//
// Opening validates the file: a non-SQLite or corrupt file, a SQLite file that is not a
// HALO profile DB, and a newer schema version all fail with ProfileDbError (never UB).
// `PRAGMA quick_check` runs at open, so page-level corruption is reported at open time.
// A read-write open of a missing or empty (0-byte) file creates schema v1; a read-only
// open of one fails with NotHaloDb.
//
// Thread-safety: a ProfileDb is NOT thread-safe (one connection, no internal locking);
// use one per thread. Several processes may share the file (SQLite locking + busy timeout).
// Timestamps (created_at / measured_at) are UTC ISO-8601 with second resolution.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "halo/autotune/profile_key.h"
#include "halo/autotune/types.h"
#include "halo/core/error.h"
#include "halo/hardware/bandwidth.h"
#include "halo/profiling/record.h"

namespace halo::autotune {

inline constexpr int kSchemaVersion = 1;
inline constexpr std::string_view kDbApp = "halo.profile";
inline constexpr std::int32_t kApplicationId = 0x48414C4F;

enum class DbErrorKind : std::uint8_t {
    Corrupt,         ///< not a SQLite file, or a damaged one
    NotHaloDb,       ///< valid SQLite, but not a HALO profile DB (or empty on read-only open)
    ForeignVersion,  ///< schema version this build cannot use (newer, or older on read-only)
    Busy,            ///< locked by another connection past the busy timeout
    Io,              ///< cannot open / read / write
    Constraint,      ///< a constraint rejected the write
    Internal,        ///< any other SQLite error
};
[[nodiscard]] std::string_view to_string(DbErrorKind k) noexcept;

/// Typed database error. Derives from halo::Error, so generic catch sites keep working;
/// code(): Io for Corrupt/Busy/Io, Unsupported for ForeignVersion, Config otherwise.
class ProfileDbError : public Error {
public:
    ProfileDbError(DbErrorKind kind, const std::string& message);
    [[nodiscard]] DbErrorKind kind() const noexcept { return kind_; }

private:
    DbErrorKind kind_;
};

/// Forward-migration hook: brings a DB at `from_version` to `from_version + 1` by running
/// SQL through `exec`. Migrations run in one transaction together with the version bump;
/// if one throws, the DB is left at its old version.
struct Migration {
    int from_version = 0;
    std::function<void(const std::function<void(const std::string& sql)>& exec)> apply;
};

struct OpenOptions {
    bool read_only = false;
    int busy_timeout_ms = 5000;
    /// Version this build expects. Tests raise it together with `migrations` to exercise
    /// the forward-migration path; production uses the default.
    int target_version = kSchemaVersion;
    std::vector<Migration> migrations;  ///< v1 has no predecessors, so none are built in
};

/// Stored tier bandwidth row.
struct TierBandwidthRow {
    std::string tier;
    std::string processor;
    std::string label;
    double read_gbps = 0;
    double write_gbps = 0;
    double copy_gbps = 0;
    std::string measured_at;
};

class ProfileDb {
public:
    /// Throws ProfileDbError (see header comment).
    [[nodiscard]] static ProfileDb open(const std::filesystem::path& path, const OpenOptions& options = {});

    ProfileDb(ProfileDb&&) noexcept;
    ProfileDb& operator=(ProfileDb&&) noexcept;
    ~ProfileDb();

    [[nodiscard]] int schema_version() const;
    [[nodiscard]] std::string journal_mode() const;
    /// Names of the tables in sqlite_master, sorted.
    [[nodiscard]] std::vector<std::string> table_names() const;

    /// Records one measured tier bandwidth under the key's hardware profile.
    void record_tier_bandwidth(const ProfileKey& key, const hardware::TierBandwidth& bw);
    [[nodiscard]] std::vector<TierBandwidthRow> tier_bandwidth(const ProfileKey& key) const;

    /// Persists one tuning result in a single transaction: the profile rows, one
    /// kernel_candidate per evaluated candidate, one benchmark_run (kind "tune") per
    /// measured candidate, and — only when result.winner is set — the winning
    /// configuration for (hardware profile, model profile, operator, backend), replacing an
    /// older one for the same backend only. Without a winner an existing winning_configuration is left untouched.
    /// Returns true when a winner was written.
    bool persist_tune(const ProfileKey& key, const OpTuneResult& result);

    /// Stores a benchmark record (kind "baseline" or "bench") with its JSON; indexed for
    /// latest_run(). Returns the row id.
    std::int64_t add_benchmark_record(const profiling::BenchmarkRecord& record, std::string_view kind);
    /// Most recent record of `kind` for (backend, pack, context, mode); ties on created_at
    /// go to the highest row id.
    [[nodiscard]] std::optional<profiling::BenchmarkRecord> latest_run(std::string_view kind,
                                                                       std::string_view backend,
                                                                       std::string_view pack,
                                                                       std::uint64_t context,
                                                                       std::string_view mode) const;

    /// Number of rows in a §58 table (tests / diagnostics). Throws Error(Config) for a name
    /// that is not one of the schema's tables.
    [[nodiscard]] std::int64_t count(std::string_view table) const;

    struct Impl;

private:
    explicit ProfileDb(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::autotune
