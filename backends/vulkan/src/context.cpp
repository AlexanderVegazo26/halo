#include "halo/backends/vulkan/context.h"

#include <cstring>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "device_query.h"
#include "halo/backends/vulkan/shaders.h"
#include "halo/core/error.h"
#include "halo/core/log.h"
#include "vk_util.h"

namespace halo::vulkan {

namespace {

constexpr const char* k_validation_layer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* user) {
    const char* msg = (data != nullptr && data->pMessage != nullptr) ? data->pMessage : "";
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        static_cast<std::atomic<std::uint64_t>*>(user)->fetch_add(1, std::memory_order_relaxed);
        HALO_ERROR("vulkan", "validation: {}", msg);
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        HALO_WARN("vulkan", "validation: {}", msg);
    }
    return VK_FALSE;
}

bool layer_available(const char* name) {
    std::uint32_t n = 0;
    if (vkEnumerateInstanceLayerProperties(&n, nullptr) != VK_SUCCESS) return false;
    std::vector<VkLayerProperties> layers(n);
    if (vkEnumerateInstanceLayerProperties(&n, layers.data()) != VK_SUCCESS) return false;
    for (const VkLayerProperties& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) return true;
    }
    return false;
}

bool extension_available(const char* name) {
    std::uint32_t n = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data()) != VK_SUCCESS) return false;
    for (const VkExtensionProperties& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------- Instance

std::shared_ptr<Instance> Instance::create(const Options& options) {
    auto inst = std::make_shared<Instance>(Key{});
    inst->validation_errors_ = std::make_unique<std::atomic<std::uint64_t>>(0);

    std::uint32_t loader_version = VK_API_VERSION_1_0;
    // vkEnumerateInstanceVersion exists in every 1.1+ loader; a 1.0 loader is too old anyway.
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) loader_version = VK_API_VERSION_1_0;
    HALO_CHECK(loader_version >= k_min_api_version, ErrorCode::Device,
               "Vulkan loader/instance version {} < required 1.3",
               detail::version_string(loader_version));

    bool want_validation = false;
    if (options.validation.has_value()) {
        want_validation = *options.validation;
    } else if (auto env = detail::getenv_str("HALO_VK_VALIDATION")) {
        want_validation = *env != "0";
    }
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    if (want_validation) {
        if (layer_available(k_validation_layer) &&
            extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            layers.push_back(k_validation_layer);
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            inst->validation_ = true;
        } else {
            HALO_WARN("vulkan", "validation requested but {} (or {}) is not installed; continuing "
                                "without API validation",
                      k_validation_layer, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "halo";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 2, 0);
    app.pEngineName = "halo";
    app.engineVersion = VK_MAKE_API_VERSION(0, 0, 2, 0);
    app.apiVersion = k_min_api_version;

    VkDebugUtilsMessengerCreateInfoEXT dbg{};
    dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg.pfnUserCallback = debug_callback;
    dbg.pUserData = inst->validation_errors_.get();

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    ci.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    if (inst->validation_) ci.pNext = &dbg;

    const VkResult r = vkCreateInstance(&ci, nullptr, &inst->instance_);
    if (r != VK_SUCCESS) {
        inst->instance_ = VK_NULL_HANDLE;
        throw_error(ErrorCode::Device, "vkCreateInstance failed: {} — no usable Vulkan ICD?",
                    detail::result_string(r));
    }
    inst->api_version_ = loader_version;

    if (inst->validation_) {
        auto create_fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(inst->instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create_fn != nullptr) {
            detail::vk_check(create_fn(inst->instance_, &dbg, nullptr, &inst->messenger_),
                             "vkCreateDebugUtilsMessengerEXT");
        }
    }

    std::uint32_t n = 0;
    detail::vk_check(vkEnumeratePhysicalDevices(inst->instance_, &n, nullptr),
                     "vkEnumeratePhysicalDevices");
    inst->physical_.resize(n);
    if (n > 0) {
        detail::vk_check(vkEnumeratePhysicalDevices(inst->instance_, &n, inst->physical_.data()),
                         "vkEnumeratePhysicalDevices");
    }
    inst->physical_.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        inst->infos_.push_back(detail::query_device(inst->physical_[i], i));
    }
    return inst;
}

Instance::~Instance() {
    if (messenger_ != VK_NULL_HANDLE) {
        auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_fn != nullptr) destroy_fn(instance_, messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

std::uint64_t Instance::validation_errors() const noexcept {
    return validation_errors_ ? validation_errors_->load(std::memory_order_relaxed) : 0;
}

VkPhysicalDevice Instance::physical_device(std::uint32_t index) const {
    HALO_CHECK(index < physical_.size(), ErrorCode::Device, "physical device index {} out of range",
               index);
    return physical_[index];
}

// ---------------------------------------------------------------- Context

std::shared_ptr<Context> Context::create(const ContextOptions& options) {
    auto ctx = std::make_shared<Context>(Key{});
    ctx->instance_ = Instance::create(options.instance);
    ctx->force_staging_ = options.force_staging;
    const auto& devices = ctx->instance_->devices();
    HALO_CHECK(!devices.empty(), ErrorCode::Device, "no Vulkan physical devices enumerated");

    std::optional<std::string> ov = options.device;
    if (!ov) ov = detail::getenv_str("HALO_VK_DEVICE");
    ctx->selection_ = select_device(devices, ov ? std::optional<std::string_view>(*ov) : std::nullopt);
    ctx->info_ = devices[ctx->selection_.index];
    ctx->physical_ = ctx->instance_->physical_device(ctx->info_.index);
    for (const std::string& w : ctx->selection_.warnings) HALO_WARN("vulkan", "{}", w);
    HALO_INFO("vulkan", "device '{}' ({}, {}; API {}) — {}", ctx->info_.name,
              to_string(ctx->info_.driver), ctx->info_.driver_info,
              detail::version_string(ctx->info_.api_version), ctx->selection_.reason);

    ctx->queue_family_ = *ctx->info_.compute_queue_family;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = ctx->queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    detail::vk_check(vkCreateDevice(ctx->physical_, &dci, nullptr, &ctx->device_), "vkCreateDevice");
    vkGetDeviceQueue(ctx->device_, ctx->queue_family_, 0, &ctx->queue_);

    VkPipelineCacheCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pci.initialDataSize = options.pipeline_cache_data.size();
    pci.pInitialData = options.pipeline_cache_data.empty() ? nullptr : options.pipeline_cache_data.data();
    VkResult r = vkCreatePipelineCache(ctx->device_, &pci, nullptr, &ctx->pipeline_cache_);
    if (r != VK_SUCCESS && pci.initialDataSize > 0) {
        HALO_WARN("vulkan", "pipeline cache data rejected ({}); starting empty",
                  detail::result_string(r));
        pci.initialDataSize = 0;
        pci.pInitialData = nullptr;
        r = vkCreatePipelineCache(ctx->device_, &pci, nullptr, &ctx->pipeline_cache_);
    }
    detail::vk_check(r, "vkCreatePipelineCache");

    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = ctx->queue_family_;
    detail::vk_check(vkCreateCommandPool(ctx->device_, &cpi, nullptr, &ctx->transfer_pool_),
                     "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = ctx->transfer_pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    detail::vk_check(vkAllocateCommandBuffers(ctx->device_, &cai, &ctx->transfer_cmd_),
                     "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    detail::vk_check(vkCreateFence(ctx->device_, &fci, nullptr, &ctx->transfer_fence_), "vkCreateFence");
    return ctx;
}

Context::~Context() {
    if (device_ == VK_NULL_HANDLE) return;
    // Teardown only: all child objects hold a shared_ptr to this Context, so they are
    // gone; wait for anything still queued before destroying the device.
    vkDeviceWaitIdle(device_);
    if (transfer_fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, transfer_fence_, nullptr);
    if (transfer_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, transfer_pool_, nullptr);
    if (pipeline_cache_ != VK_NULL_HANDLE) vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
    vkDestroyDevice(device_, nullptr);
}

void Context::submit(VkCommandBuffer cmd, VkFence fence) {
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    const std::scoped_lock lock(queue_mutex_);
    detail::vk_check(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit");
}

void Context::copy_buffer_sync(VkBuffer src, VkBuffer dst, VkDeviceSize src_offset,
                               VkDeviceSize dst_offset, VkDeviceSize size) {
    const std::scoped_lock lock(transfer_mutex_);
    detail::vk_check(vkResetCommandBuffer(transfer_cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    detail::vk_check(vkBeginCommandBuffer(transfer_cmd_, &bi), "vkBeginCommandBuffer");
    VkBufferCopy region{src_offset, dst_offset, size};
    vkCmdCopyBuffer(transfer_cmd_, src, dst, 1, &region);
    // Make the copy visible to later host reads (download path) and to any later device
    // work (queue submission order + this barrier).
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(transfer_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb,
                         0, nullptr, 0, nullptr);
    detail::vk_check(vkEndCommandBuffer(transfer_cmd_), "vkEndCommandBuffer");
    detail::vk_check(vkResetFences(device_, 1, &transfer_fence_), "vkResetFences");
    submit(transfer_cmd_, transfer_fence_);
    const VkResult r = vkWaitForFences(device_, 1, &transfer_fence_, VK_TRUE, 60'000'000'000ull);
    detail::vk_check(r, "vkWaitForFences(transfer)");
}

std::vector<std::byte> Context::pipeline_cache_data() const {
    std::size_t n = 0;
    detail::vk_check(vkGetPipelineCacheData(device_, pipeline_cache_, &n, nullptr),
                     "vkGetPipelineCacheData");
    std::vector<std::byte> out(n);
    if (n > 0) {
        detail::vk_check(vkGetPipelineCacheData(device_, pipeline_cache_, &n, out.data()),
                         "vkGetPipelineCacheData");
        out.resize(n);
    }
    return out;
}

nlohmann::json Context::describe() const {
    nlohmann::json j;
    j["available"] = true;
    j["backend"] = "vulkan";
    j["instance_api_version"] = detail::version_string(instance_->api_version());
    j["validation_enabled"] = instance_->validation_enabled();
    j["selected"] = selection_.index;
    j["selection_reason"] = selection_.reason;
    j["warnings"] = selection_.warnings;
    j["correctness_only"] = info_.correctness_only;
    j["shader_set_sha256"] = shader_set_hash();
    j["devices"] = nlohmann::json::array();
    for (const DeviceInfo& d : instance_->devices()) j["devices"].push_back(to_json(d));
    return j;
}

nlohmann::json describe_devices(std::optional<std::string_view> override) {
    nlohmann::json j;
    j["backend"] = "vulkan";
    std::shared_ptr<Instance> inst;
    try {
        inst = Instance::create();
    } catch (const Error& e) {
        j["available"] = false;
        j["error"] = e.what();
        return j;
    }
    j["available"] = true;
    j["instance_api_version"] = detail::version_string(inst->api_version());
    j["devices"] = nlohmann::json::array();
    for (const DeviceInfo& d : inst->devices()) j["devices"].push_back(to_json(d));
    std::optional<std::string> env;
    if (!override) env = detail::getenv_str("HALO_VK_DEVICE");
    try {
        const DeviceSelection sel = select_device(
            inst->devices(), override ? override : (env ? std::optional<std::string_view>(*env) : std::nullopt));
        j["selected"] = sel.index;
        j["selection_reason"] = sel.reason;
        j["warnings"] = sel.warnings;
    } catch (const Error& e) {
        j["selected"] = nullptr;
        j["selection_error"] = e.what();
    }
    return j;
}

}  // namespace halo::vulkan
