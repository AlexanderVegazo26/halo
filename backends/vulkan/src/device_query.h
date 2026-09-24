#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "halo/backends/vulkan/device.h"

namespace halo::vulkan::detail {

/// Query everything HALO records about a physical device (properties, driver, subgroup,
/// limits, memory heaps/types, compute queue family).
[[nodiscard]] DeviceInfo query_device(VkPhysicalDevice pd, std::uint32_t index);

}  // namespace halo::vulkan::detail
