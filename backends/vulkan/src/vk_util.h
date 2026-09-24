#pragma once
// Internal helpers for the Vulkan backend.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include <vulkan/vulkan.h>

#include "halo/core/error.h"

namespace halo::vulkan::detail {

[[nodiscard]] std::string_view result_string(VkResult r) noexcept;

/// Throws Error(Memory) for out-of-memory results, Error(Backend) otherwise.
[[noreturn]] void throw_vk(VkResult r, std::string_view what);

inline void vk_check(VkResult r, std::string_view what) {
    if (r != VK_SUCCESS) [[unlikely]] throw_vk(r, what);
}

[[nodiscard]] std::optional<std::string> getenv_str(const char* name);

/// a * b, throwing Error(Kernel) on uint64 overflow.
[[nodiscard]] std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view what);
[[nodiscard]] std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view what);

/// Narrow to uint32, throwing Error(Kernel) if the value does not fit (shader indices
/// are 32-bit).
[[nodiscard]] std::uint32_t to_u32(std::uint64_t v, std::string_view what);

[[nodiscard]] std::string version_string(std::uint32_t v);

}  // namespace halo::vulkan::detail
