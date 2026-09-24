#pragma once
// Minimal RAII wrapper over the SQLite C API (internal to halo_autotune).
// Every call's return code is checked and mapped to a typed ProfileDbError.
// Text is always bound with SQLITE_TRANSIENT (SQLite copies it), so temporaries are safe.

#include <sqlite3.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "halo/autotune/db.h"

namespace halo::autotune::sql {

/// Maps a SQLite result code to the error kind (primary code, extended bits stripped).
[[nodiscard]] DbErrorKind kind_of(int rc) noexcept;

/// Throws ProfileDbError when rc is not OK/ROW/DONE.
void check(int rc, sqlite3* db, std::string_view what);

class Connection;

class Statement {
public:
    Statement(Connection& c, std::string_view sql);
    ~Statement();
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    Statement(Statement&& o) noexcept;
    Statement& operator=(Statement&&) = delete;

    Statement& bind(int idx, std::int64_t v);
    Statement& bind(int idx, double v);
    Statement& bind(int idx, std::string_view v);
    Statement& bind(int idx, const char* v) { return bind(idx, std::string_view(v)); }
    Statement& bind_null(int idx);
    Statement& bind(int idx, const std::optional<double>& v) { return v ? bind(idx, *v) : bind_null(idx); }

    /// true = a row is available; false = done.
    bool step();
    void reset();

    [[nodiscard]] bool is_null(int col) const;
    [[nodiscard]] std::int64_t int64(int col) const;
    [[nodiscard]] double real(int col) const;
    /// NULL -> "" (never constructs a std::string from a null pointer).
    [[nodiscard]] std::string text(int col) const;

private:
    Connection* c_;
    sqlite3_stmt* s_ = nullptr;
    std::string sql_;
};

class Connection {
public:
    Connection(const std::string& path, bool read_only, int busy_timeout_ms);
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    [[nodiscard]] sqlite3* handle() const noexcept { return db_; }
    /// For DDL / PRAGMA without values only (never used with untrusted text).
    void exec(const std::string& sql);
    [[nodiscard]] std::int64_t last_insert_rowid() const noexcept;
    /// First column of the first row as text ("" when no row).
    [[nodiscard]] std::string query_text(std::string_view sql);
    [[nodiscard]] std::int64_t query_int(std::string_view sql);

private:
    sqlite3* db_ = nullptr;
};

/// BEGIN IMMEDIATE ... COMMIT; rolls back on destruction unless committed.
class Transaction {
public:
    explicit Transaction(Connection& c);
    ~Transaction();
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    void commit();

private:
    Connection& c_;
    bool done_ = false;
};

/// UTC now, "YYYY-MM-DDTHH:MM:SSZ".
[[nodiscard]] std::string utc_now();

}  // namespace halo::autotune::sql
