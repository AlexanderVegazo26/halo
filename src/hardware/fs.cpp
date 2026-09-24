#include "hardware/fs.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <system_error>

namespace halo::hardware::detail {

std::filesystem::path under_root(const std::filesystem::path& root, std::string_view sys_path) {
    std::filesystem::path rel(sys_path);
    return root / rel.relative_path();
}

std::optional<std::string> read_text(const std::filesystem::path& p, std::size_t max_bytes) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec) || ec) return std::nullopt;
    std::ifstream in(p, std::ios::binary);
    if (!in) return std::nullopt;
    // sysfs reports st_size 4096 regardless of content, so read until EOF with a cap.
    std::string out;
    std::array<char, 4096> buf{};
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = static_cast<std::size_t>(in.gcount());
        if (got == 0) break;
        if (out.size() + got > max_bytes) return std::nullopt;
        out.append(buf.data(), got);
    }
    if (in.bad()) return std::nullopt;
    return out;
}

std::vector<std::string> list_dir(const std::filesystem::path& dir) {
    std::vector<std::string> names;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) return names;
    for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (ec) break;
        names.push_back(it->path().filename().string());
    }
    std::ranges::sort(names);
    return names;
}

bool is_dir(const std::filesystem::path& p) noexcept {
    std::error_code ec;
    return std::filesystem::is_directory(p, ec) && !ec;
}

}  // namespace halo::hardware::detail
