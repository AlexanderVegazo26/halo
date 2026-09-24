#pragma once
// Root-relative, size-capped, non-throwing filesystem helpers for discovery.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace halo::hardware::detail {

/// procfs/sysfs text files HALO reads are small; anything larger is treated as corrupt.
inline constexpr std::size_t kMaxTextFileBytes = 4U << 20;

/// `root` joined with an absolute-looking system path ("/proc/meminfo").
[[nodiscard]] std::filesystem::path under_root(const std::filesystem::path& root, std::string_view sys_path);

/// Read a whole text file; nullopt if missing, unreadable, or larger than `max_bytes`.
[[nodiscard]] std::optional<std::string> read_text(const std::filesystem::path& p,
                                                   std::size_t max_bytes = kMaxTextFileBytes);

/// Directory entry names (not paths), sorted; empty if the directory is missing.
[[nodiscard]] std::vector<std::string> list_dir(const std::filesystem::path& dir);

[[nodiscard]] bool is_dir(const std::filesystem::path& p) noexcept;

}  // namespace halo::hardware::detail
