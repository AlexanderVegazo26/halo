#pragma once
// Helpers for tests that read reference data from HALO_REF_DIR (see tests/CMakeLists.txt).

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace halo::test {

inline std::filesystem::path ref_dir() { return std::filesystem::path(HALO_REF_DIR); }

inline std::vector<std::byte> read_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::byte> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) out[i] = static_cast<std::byte>(raw[i]);
    return out;
}

inline std::optional<nlohmann::json> read_json(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) return std::nullopt;
    return nlohmann::json::parse(f);
}

}  // namespace halo::test
