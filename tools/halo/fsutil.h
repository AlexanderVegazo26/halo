#pragma once
// Output-file helpers for the CLI (security review S-21).

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace halo::cli {

/// Writes `content` to `path` without following a symlink at `path`: the data goes to a
/// fresh temporary file in the same directory (O_CREAT | O_EXCL | O_NOFOLLOW, mode 0644)
/// and is renamed into place, so an existing file is replaced atomically and never
/// truncated in place. Throws Error(Io) when `path` is a symlink or any step fails (a
/// failed write is never reported as success).
void write_file_no_follow(const std::filesystem::path& path, std::string_view content);

/// Throws Error(Io) when `path` exists and is a symlink (used for paths opened by other
/// libraries, e.g. the SQLite profile database).
void refuse_symlink(const std::filesystem::path& path, std::string_view what);

/// Identity of a file (device, inode, size, mtime), to detect that a file was replaced
/// between two opens.
struct FileIdentity {
    std::uint64_t dev = 0, ino = 0, size = 0;
    std::int64_t mtime_ns = 0;
    friend bool operator==(const FileIdentity&, const FileIdentity&) = default;
};
/// Throws Error(Io) when the file cannot be stat'ed.
[[nodiscard]] FileIdentity file_identity(const std::filesystem::path& path);

}  // namespace halo::cli
