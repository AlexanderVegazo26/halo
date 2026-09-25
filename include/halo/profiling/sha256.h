#pragma once
// Streaming SHA-256 (FIPS 180-4) for model hashes (TRD §51: "model hash"; profile key
// MODEL_HASH / PACK_ID, TRD §57). Files are read in fixed-size chunks, so a ~17 GB GGUF is
// hashed in constant memory.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace halo::profiling {

class Sha256 {
public:
    Sha256() noexcept { reset(); }
    void reset() noexcept;
    void update(std::span<const std::uint8_t> data) noexcept;
    void update(std::string_view text) noexcept;
    /// Final digest; the object must be reset() before reuse.
    [[nodiscard]] std::array<std::uint8_t, 32> finish() noexcept;
    /// finish() as lowercase hex.
    [[nodiscard]] std::string finish_hex();

private:
    void block(const std::uint8_t* p) noexcept;
    std::array<std::uint32_t, 8> h_{};
    std::array<std::uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_ = 0;  ///< bytes
};

[[nodiscard]] std::string to_hex(std::span<const std::uint8_t> bytes);

/// SHA-256 hex of `text`.
[[nodiscard]] std::string sha256_hex(std::string_view text);

/// Streams a file through SHA-256 in `chunk_bytes` reads (default 4 MiB). `progress`
/// (optional) receives the bytes hashed so far. Throws Error(Io) when the file cannot be
/// opened or a read fails.
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path, std::size_t chunk_bytes = 4U << 20,
                                      const std::function<void(std::uint64_t)>& progress = {});

/// PACK_ID (profile key, TRD §57) = SHA-256 hex of the ASCII text
///   "halo.pack/1\ntrunk=<trunk sha256 hex>\nmtp=<mtp sha256 hex or 'none'>\n".
/// A pack with an MTP block embedded in the trunk GGUF has mtp = "none" (the trunk hash
/// already covers it). Throws Error(Config) unless the hashes are 64 lowercase hex chars.
[[nodiscard]] std::string make_pack_id(std::string_view trunk_sha256,
                                       const std::optional<std::string>& mtp_sha256);

}  // namespace halo::profiling
