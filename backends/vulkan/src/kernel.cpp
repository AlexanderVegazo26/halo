#include "halo/backends/vulkan/kernel.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "halo/backends/vulkan/context.h"
#include "halo/core/error.h"
#include "vk_util.h"

namespace halo::vulkan {

Kernel::Kernel(std::shared_ptr<Context> ctx, KernelDesc desc) : ctx_(std::move(ctx)), desc_(std::move(desc)) {
    HALO_CHECK(ctx_ != nullptr, ErrorCode::Kernel, "Kernel: null context");
    HALO_CHECK(desc_.shader != nullptr && !desc_.shader->spirv.empty(), ErrorCode::Kernel,
               "Kernel: missing SPIR-V");
    const DeviceInfo& info = ctx_->info();
    const std::string_view name = desc_.shader->name;
    HALO_CHECK(desc_.push_constant_bytes % 4 == 0 && desc_.push_constant_bytes <= info.max_push_constants,
               ErrorCode::Kernel, "kernel {}: push constant block of {} bytes (limit {}, multiple of 4)",
               name, desc_.push_constant_bytes, info.max_push_constants);
    HALO_CHECK(desc_.num_buffers <= info.max_storage_buffers_per_stage, ErrorCode::Kernel,
               "kernel {}: {} storage buffers > device limit {}", name, desc_.num_buffers,
               info.max_storage_buffers_per_stage);
    std::uint64_t invocations = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        const std::uint32_t s = desc_.local_size[i];
        HALO_CHECK(s >= 1 && s <= info.max_workgroup_size[i], ErrorCode::Kernel,
                   "kernel {}: local_size[{}]={} outside [1, {}]", name, i, s, info.max_workgroup_size[i]);
        invocations *= s;
    }
    HALO_CHECK(invocations <= info.max_workgroup_invocations, ErrorCode::Kernel,
               "kernel {}: {} invocations per workgroup > limit {}", name, invocations,
               info.max_workgroup_invocations);

    try {
        const VkDevice dev = ctx_->device();
        VkShaderModuleCreateInfo smi{};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = desc_.shader->spirv.size_bytes();
        smi.pCode = desc_.shader->spirv.data();
        detail::vk_check(vkCreateShaderModule(dev, &smi, nullptr, &module_), "vkCreateShaderModule");

        std::vector<VkDescriptorSetLayoutBinding> bindings(desc_.num_buffers);
        for (std::uint32_t i = 0; i < desc_.num_buffers; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{};
        dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = desc_.num_buffers;
        dli.pBindings = bindings.data();
        detail::vk_check(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &set_layout_),
                         "vkCreateDescriptorSetLayout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = desc_.push_constant_bytes;
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout_;
        pli.pushConstantRangeCount = desc_.push_constant_bytes > 0 ? 1 : 0;
        pli.pPushConstantRanges = &pcr;
        detail::vk_check(vkCreatePipelineLayout(dev, &pli, nullptr, &layout_), "vkCreatePipelineLayout");

        std::vector<VkSpecializationMapEntry> entries;
        std::vector<std::uint32_t> values;
        for (const SpecConstant& sc : desc_.spec_constants) {
            entries.push_back({sc.id, static_cast<std::uint32_t>(values.size() * 4), 4});
            values.push_back(sc.value);
        }
        VkSpecializationInfo spec{};
        spec.mapEntryCount = static_cast<std::uint32_t>(entries.size());
        spec.pMapEntries = entries.data();
        spec.dataSize = values.size() * 4;
        spec.pData = values.data();

        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = module_;
        cpi.stage.pName = "main";
        cpi.stage.pSpecializationInfo = entries.empty() ? nullptr : &spec;
        cpi.layout = layout_;
        detail::vk_check(vkCreateComputePipelines(dev, ctx_->pipeline_cache(), 1, &cpi, nullptr, &pipeline_),
                         "vkCreateComputePipelines");
    } catch (...) {
        release();
        throw;
    }
}

Kernel::~Kernel() { release(); }

Kernel::Kernel(Kernel&& o) noexcept
    : ctx_(std::move(o.ctx_)),
      desc_(std::move(o.desc_)),
      module_(std::exchange(o.module_, VK_NULL_HANDLE)),
      set_layout_(std::exchange(o.set_layout_, VK_NULL_HANDLE)),
      layout_(std::exchange(o.layout_, VK_NULL_HANDLE)),
      pipeline_(std::exchange(o.pipeline_, VK_NULL_HANDLE)) {}

Kernel& Kernel::operator=(Kernel&& o) noexcept {
    if (this != &o) {
        release();
        ctx_ = std::move(o.ctx_);
        desc_ = std::move(o.desc_);
        module_ = std::exchange(o.module_, VK_NULL_HANDLE);
        set_layout_ = std::exchange(o.set_layout_, VK_NULL_HANDLE);
        layout_ = std::exchange(o.layout_, VK_NULL_HANDLE);
        pipeline_ = std::exchange(o.pipeline_, VK_NULL_HANDLE);
    }
    return *this;
}

void Kernel::release() noexcept {
    if (!ctx_) return;
    const VkDevice dev = ctx_->device();
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, layout_, nullptr);
    if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, set_layout_, nullptr);
    if (module_ != VK_NULL_HANDLE) vkDestroyShaderModule(dev, module_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    module_ = VK_NULL_HANDLE;
    ctx_.reset();
}

GroupCount grid_1d(std::uint64_t n, std::uint32_t max_x, std::uint32_t max_y) {
    HALO_CHECK(n > 0, ErrorCode::Kernel, "grid_1d: zero workgroups");
    HALO_CHECK(max_x > 0 && max_y > 0, ErrorCode::Kernel, "grid_1d: zero limits");
    const std::uint64_t x = std::min<std::uint64_t>(n, max_x);
    const std::uint64_t y = (n + x - 1) / x;
    HALO_CHECK(y <= max_y, ErrorCode::Kernel, "grid_1d: {} workgroups exceed the device grid ({} x {})",
               n, max_x, max_y);
    return {static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), 1};
}

}  // namespace halo::vulkan
