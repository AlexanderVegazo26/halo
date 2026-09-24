#pragma once
// Internal: the table populated by the build-generated sources (see CMakeLists.txt and
// cmake/embed_spirv.cmake).

#include <cstdint>
#include <span>

#include "halo/backends/vulkan/shaders.h"

namespace halo::vulkan::detail {

extern const std::span<const EmbeddedShader* const> k_shaders;

}  // namespace halo::vulkan::detail
