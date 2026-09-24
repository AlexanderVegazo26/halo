#pragma once
/// \file
/// Embedded SPIR-V registry (TRD §41). Every compute shader under
/// `backends/vulkan/shaders/` is compiled by glslc at build time and linked into
/// `halo_backend_vulkan` together with the SHA-256 of its SPIR-V bytes. The hashes are
/// the cache-validation key: a pipeline cache produced against one shader set must not be
/// trusted against another (see `shader_set_hash()`).

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace halo::vulkan {

/// One embedded SPIR-V module. Storage has static lifetime.
struct EmbeddedShader {
    std::string_view name;         ///< lookup key, e.g. "rms_norm"
    std::string_view source_path;  ///< relative to backends/vulkan/shaders/, e.g. "norm/rms_norm.comp"
    std::span<const std::uint32_t> spirv;
    std::string_view sha256;  ///< lowercase hex SHA-256 of the SPIR-V bytes (little-endian words)
};

/// All embedded shaders, in build order.
[[nodiscard]] std::span<const EmbeddedShader* const> embedded_shaders() noexcept;

/// Throws Error(Kernel) if no shader with that name is embedded.
[[nodiscard]] const EmbeddedShader& find_shader(std::string_view name);

/// Recomputes the SHA-256 of `shader.spirv` and compares it with the recorded hash.
[[nodiscard]] bool verify_shader(const EmbeddedShader& shader);

/// SHA-256 over "name:sha256\n" of every embedded shader, in registry order. Changes
/// whenever any shader's SPIR-V changes; used to key persisted pipeline caches.
[[nodiscard]] std::string shader_set_hash();

/// Lowercase hex SHA-256 of `bytes` (FIPS 180-4). Exposed for tests and cache keys.
[[nodiscard]] std::string sha256_hex(std::span<const std::byte> bytes);

}  // namespace halo::vulkan
