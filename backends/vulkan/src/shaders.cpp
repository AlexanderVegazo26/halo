#include <bit>
#include <cstddef>
#include <span>
#include <string>

#include "halo/backends/vulkan/shaders.h"
#include "halo/core/error.h"
#include "shader_registry.h"

namespace halo::vulkan {

static_assert(std::endian::native == std::endian::little,
              "embedded SPIR-V words are hashed as little-endian bytes");

std::span<const EmbeddedShader* const> embedded_shaders() noexcept { return detail::k_shaders; }

const EmbeddedShader& find_shader(std::string_view name) {
    for (const EmbeddedShader* s : detail::k_shaders) {
        if (s->name == name) return *s;
    }
    throw_error(ErrorCode::Kernel, "no embedded Vulkan shader named '{}'", name);
}

bool verify_shader(const EmbeddedShader& shader) {
    return sha256_hex(std::as_bytes(shader.spirv)) == shader.sha256;
}

std::string shader_set_hash() {
    std::string manifest;
    for (const EmbeddedShader* s : detail::k_shaders) {
        manifest += s->name;
        manifest += ':';
        manifest += s->sha256;
        manifest += '\n';
    }
    return sha256_hex(std::as_bytes(std::span(manifest.data(), manifest.size())));
}

}  // namespace halo::vulkan
