#include "sqlite.h"

#include <chrono>
#include <format>

#include "halo/core/log.h"

namespace halo::autotune::sql {

DbErrorKind kind_of(int rc) noexcept {
    switch (rc & 0xff) {
        case SQLITE_CORRUPT:
        case SQLITE_NOTADB: return DbErrorKind::Corrupt;
        case SQLITE_BUSY:
        case SQLITE_LOCKED: return DbErrorKind::Busy;
        case SQLITE_IOERR:
        case SQLITE_CANTOPEN:
        case SQLITE_FULL:
        case SQLITE_READONLY:
        case SQLITE_PERM:
        case SQLITE_NOMEM: return DbErrorKind::Io;
        case SQLITE_CONSTRAINT: return DbErrorKind::Constraint;
        default: return DbErrorKind::Internal;
    }
}

void check(int rc, sqlite3* db, std::string_view what) {
    if (rc == SQLITE_OK || rc == SQLITE_ROW || rc == SQLITE_DONE) return;
    const char* msg = db != nullptr ? sqlite3_errmsg(db) : sqlite3_errstr(rc);
    throw ProfileDbError(kind_of(rc), std::format("{}: {} (sqlite rc {})", what, msg != nullptr ? msg : "?", rc));
}

// ---- Statement ----------------------------------------------------------------------------

Statement::Statement(Connection& c, std::string_view sql) : c_(&c), sql_(sql) {
    const int rc = sqlite3_prepare_v2(c.handle(), sql_.c_str(), static_cast<int>(sql_.size()), &s_, nullptr);
    check(rc, c.handle(), "prepare");
}

Statement::~Statement() {
    if (s_ != nullptr) sqlite3_finalize(s_);
}

Statement::Statement(Statement&& o) noexcept : c_(o.c_), s_(o.s_), sql_(std::move(o.sql_)) { o.s_ = nullptr; }

Statement& Statement::bind(int idx, std::int64_t v) {
    check(sqlite3_bind_int64(s_, idx, v), c_->handle(), "bind int");
    return *this;
}

Statement& Statement::bind(int idx, double v) {
    check(sqlite3_bind_double(s_, idx, v), c_->handle(), "bind real");
    return *this;
}

Statement& Statement::bind(int idx, std::string_view v) {
    check(sqlite3_bind_text64(s_, idx, v.data(), v.size(), SQLITE_TRANSIENT, SQLITE_UTF8), c_->handle(),
          "bind text");
    return *this;
}

Statement& Statement::bind_null(int idx) {
    check(sqlite3_bind_null(s_, idx), c_->handle(), "bind null");
    return *this;
}

bool Statement::step() {
    const int rc = sqlite3_step(s_);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    check(rc, c_->handle(), "step");
    return false;
}

void Statement::reset() {
    sqlite3_reset(s_);
    sqlite3_clear_bindings(s_);
}

bool Statement::is_null(int col) const { return sqlite3_column_type(s_, col) == SQLITE_NULL; }
std::int64_t Statement::int64(int col) const { return sqlite3_column_int64(s_, col); }
double Statement::real(int col) const { return sqlite3_column_double(s_, col); }

std::string Statement::text(int col) const {
    const unsigned char* p = sqlite3_column_text(s_, col);
    if (p == nullptr) return {};
    const int n = sqlite3_column_bytes(s_, col);
    return {reinterpret_cast<const char*>(p), static_cast<std::size_t>(n)};
}

// ---- Connection ---------------------------------------------------------------------------

Connection::Connection(const std::string& path, bool read_only, int busy_timeout_ms) {
    const int flags = (read_only ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)) |
                      SQLITE_OPEN_NOMUTEX | SQLITE_OPEN_EXRESCODE;
    const int rc = sqlite3_open_v2(path.c_str(), &db_, flags, nullptr);
    if (rc != SQLITE_OK) {
        const std::string msg = db_ != nullptr ? sqlite3_errmsg(db_) : sqlite3_errstr(rc);
        if (db_ != nullptr) sqlite3_close(db_);
        db_ = nullptr;
        throw ProfileDbError(kind_of(rc), std::format("open {}: {}", path, msg));
    }
    check(sqlite3_busy_timeout(db_, busy_timeout_ms), db_, "busy_timeout");
}

Connection::~Connection() {
    if (db_ != nullptr) {
        const int rc = sqlite3_close(db_);
        if (rc != SQLITE_OK) HALO_ERROR("autotune", "sqlite3_close failed: {}", sqlite3_errstr(rc));
    }
}

void Connection::exec(const std::string& sql) {
    char* err = nullptr;
    const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        const std::string msg = err != nullptr ? err : sqlite3_errstr(rc);
        sqlite3_free(err);
        throw ProfileDbError(kind_of(rc), std::format("exec: {}", msg));
    }
}

std::int64_t Connection::last_insert_rowid() const noexcept { return sqlite3_last_insert_rowid(db_); }

std::string Connection::query_text(std::string_view sql) {
    Statement s(*this, sql);
    return s.step() ? s.text(0) : std::string();
}

std::int64_t Connection::query_int(std::string_view sql) {
    Statement s(*this, sql);
    return s.step() ? s.int64(0) : 0;
}

// ---- Transaction --------------------------------------------------------------------------

Transaction::Transaction(Connection& c) : c_(c) { c_.exec("BEGIN IMMEDIATE"); }

Transaction::~Transaction() {
    if (!done_) {
        // Never throws from a destructor: a failed rollback is logged (SQLite rolls back
        // an open transaction on close anyway).
        char* err = nullptr;
        if (sqlite3_exec(c_.handle(), "ROLLBACK", nullptr, nullptr, &err) != SQLITE_OK) {
            HALO_ERROR("autotune", "rollback failed: {}", err != nullptr ? err : "?");
        }
        sqlite3_free(err);
    }
}

void Transaction::commit() {
    c_.exec("COMMIT");
    done_ = true;
}

std::string utc_now() {
    const auto now = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now());
    return std::format("{:%FT%TZ}", now);
}

}  // namespace halo::autotune::sql
