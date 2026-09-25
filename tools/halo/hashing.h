#pragma once
// SHA-256 (FIPS 180-4) of files and the PACK_ID of the profile key (TRD §57), for
// `halo tune`. Streams in fixed-size chunks, so a ~17 GB GGUF hashes in constant memory.
//
// Known debt: WS-J has an equivalent (profiling::sha256_file / make_pack_id in
// include/halo/profiling/sha256.h) that was not committed when this was written. The PACK_ID
// text format below is the one that header documents, so keys stay identical; switch to
// the shared implementation once it is committed.

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace halo::cli {

[[nodiscard]] std::string sha256_hex(std::string_view data);
/// Throws Error(Io) when the file cannot be read.
[[nodiscard]] std::string sha256_file_hex(const std::filesystem::path& path);
/// SHA-256 hex of "halo.pack/1\ntrunk=<trunk>\nmtp=<mtp or 'none'>\n".
[[nodiscard]] std::string pack_id(const std::string& trunk_sha256, const std::optional<std::string>& mtp_sha256);

}  // namespace halo::cli
