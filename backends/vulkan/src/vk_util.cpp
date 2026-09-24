#include "vk_util.h"

#include <cstdlib>
#include <format>
#include <limits>

namespace halo::vulkan::detail {

std::string_view result_string(VkResult r) noexcept {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        default: return "VK_RESULT_UNKNOWN";
    }
}

void throw_vk(VkResult r, std::string_view what) {
    const bool oom = r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                     r == VK_ERROR_OUT_OF_POOL_MEMORY;
    throw_error(oom ? ErrorCode::Memory : ErrorCode::Backend, "{} failed: {} ({})", what,
                result_string(r), static_cast<int>(r));
}

std::optional<std::string> getenv_str(const char* name) {
    const char* v = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read at init only
    if (v == nullptr || *v == '\0') return std::nullopt;
    return std::string(v);
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view what) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw_error(ErrorCode::Kernel, "{}: size overflow ({} * {})", what, a, b);
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view what) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw_error(ErrorCode::Kernel, "{}: size overflow ({} + {})", what, a, b);
    }
    return a + b;
}

std::uint32_t to_u32(std::uint64_t v, std::string_view what) {
    if (v > std::numeric_limits<std::uint32_t>::max()) {
        throw_error(ErrorCode::Kernel, "{}: {} does not fit the 32-bit shader index range", what, v);
    }
    return static_cast<std::uint32_t>(v);
}

std::string version_string(std::uint32_t v) {
    return std::format("{}.{}.{}", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v),
                       VK_API_VERSION_PATCH(v));
}

}  // namespace halo::vulkan::detail
