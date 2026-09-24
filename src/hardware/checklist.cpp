#include "halo/hardware/checklist.h"

#include <algorithm>
#include <format>

#include <nlohmann/json.hpp>

namespace halo::hardware {
namespace {

std::string gib(std::uint64_t b) { return std::format("{:.2f} GiB", static_cast<double>(b) / (1ULL << 30)); }

CheckItem item(std::string check, CheckStatus s, std::string detail) {
    return CheckItem{std::move(check), s, std::move(detail)};
}

}  // namespace

std::string_view to_string(CheckStatus s) noexcept {
    switch (s) {
        case CheckStatus::Ok: return "ok";
        case CheckStatus::Warn: return "warn";
        case CheckStatus::Fail: return "fail";
        case CheckStatus::Unknown: return "unknown";
    }
    return "unknown";
}

CheckStatus overall_status(const std::vector<CheckItem>& items) noexcept {
    auto rank = [](CheckStatus s) {
        switch (s) {
            case CheckStatus::Ok: return 0;
            case CheckStatus::Unknown: return 1;
            case CheckStatus::Warn: return 2;
            case CheckStatus::Fail: return 3;
        }
        return 3;
    };
    CheckStatus worst = CheckStatus::Ok;
    for (const auto& i : items)
        if (rank(i.status) > rank(worst)) worst = i.status;
    return worst;
}

std::vector<CheckItem> deployment_checklist(const HardwareInfo& info) {
    std::vector<CheckItem> out;
    const GpuInfo* gpu = info.gpus.empty() ? nullptr : &info.gpus.front();
    const auto& t = info.tiers;

    // amd_gpu
    if (!gpu) out.push_back(item("amd_gpu", CheckStatus::Fail, "no AMD GPU (PCI vendor 0x1002) under /sys/class/drm"));
    else
        out.push_back(item("amd_gpu", CheckStatus::Ok,
                           std::format("{} ({}, device 0x{:04x})", gpu->drm_card, gpu->pci_slot.value_or("pci slot unknown"),
                                       gpu->device_id)));

    // gfx_target
    if (!gpu || !gpu->gfx_target)
        out.push_back(item("gfx_target", CheckStatus::Unknown, "no KFD topology node associated with the GPU"));
    else if (*gpu->gfx_target == "gfx1151")
        out.push_back(item("gfx_target", CheckStatus::Ok,
                           std::format("gfx1151 (gfx_target_version {}), {} CUs", gpu->gfx_target_version.value_or(0),
                                       gpu->compute_units ? std::to_string(*gpu->compute_units) : "?")));
    else
        out.push_back(item("gfx_target", CheckStatus::Warn,
                           std::format("{} is not the gfx1151 target HALO is specialized for", *gpu->gfx_target)));

    // carveout (D-002: both the large-carveout and the small-carveout layouts are valid)
    if (!t.vram_total) {
        out.push_back(item("carveout", CheckStatus::Unknown, "mem_info_vram_total not readable"));
    } else if (*t.vram_total >= kLargeCarveoutBytes) {
        out.push_back(item("carveout", CheckStatus::Ok,
                           std::format("{} VRAM carveout: large-carveout layout (D-002, the measured EVO-X2 "
                                       "configuration); GPU-accessible total {}",
                                       gib(*t.vram_total), gib(t.gpu_accessible_bytes))));
    } else if (*t.vram_total <= kSmallCarveoutBytes) {
        out.push_back(item("carveout", CheckStatus::Ok,
                           std::format("{} VRAM carveout: small-carveout layout (TRD §59); GTT must carry the model",
                                       gib(*t.vram_total))));
    } else {
        out.push_back(item("carveout", CheckStatus::Warn,
                           std::format("{} VRAM carveout is neither the >= 64 GiB layout (D-002) nor the <= 2 GiB "
                                       "layout (TRD §59); GPU-accessible total {}",
                                       gib(*t.vram_total), gib(t.gpu_accessible_bytes))));
    }

    // gtt
    if (!t.gtt_total) {
        out.push_back(item("gtt", CheckStatus::Unknown, "mem_info_gtt_total not readable"));
    } else {
        std::string params;
        if (info.cmdline.amdgpu_gttsize_mib) params += std::format(" amdgpu.gttsize={}", *info.cmdline.amdgpu_gttsize_mib);
        if (info.cmdline.ttm_pages_limit) params += std::format(" ttm.pages_limit={}", *info.cmdline.ttm_pages_limit);
        if (params.empty()) params = " (no GTT kernel parameters set)";
        if (t.topology == MemoryTopology::CarveoutPrimary) {
            out.push_back(item("gtt", CheckStatus::Ok,
                               std::format("{} GTT; not the workhorse tier in the carveout-primary layout;{}",
                                           gib(*t.gtt_total), params)));
        } else if (*t.gtt_total >= kGttWorkhorseBytes) {
            out.push_back(item("gtt", CheckStatus::Ok, std::format("{} GTT;{}", gib(*t.gtt_total), params)));
        } else {
            out.push_back(item("gtt", CheckStatus::Warn,
                               std::format("{} GTT with a small carveout; TRD §59 expects the ~120 GB class via "
                                           "amdgpu.gttsize / ttm.pages_limit;{}",
                                           gib(*t.gtt_total), params)));
        }
    }

    // iommu
    switch (info.iommu.state) {
        case IommuState::Off: out.push_back(item("iommu", CheckStatus::Ok, "disabled: " + info.iommu.evidence)); break;
        case IommuState::On:
            out.push_back(item("iommu", CheckStatus::Warn,
                               "enabled: " + info.iommu.evidence + "; disable it (e.g. amd_iommu=off), TRD §59"));
            break;
        case IommuState::Passthrough:
            out.push_back(item("iommu", CheckStatus::Warn,
                               "passthrough: " + info.iommu.evidence + "; TRD §59 asks for disabled"));
            break;
        case IommuState::Unknown: out.push_back(item("iommu", CheckStatus::Unknown, info.iommu.evidence)); break;
    }

    // vulkan_driver
    {
        const bool radv = info.vulkan.has(VulkanDriverKind::Radv);
        const bool amdvlk = info.vulkan.has(VulkanDriverKind::Amdvlk);
        std::string inventory;
        for (const auto& i : info.vulkan.icds)
            inventory += std::format("{}{}={}", inventory.empty() ? "" : ", ", i.manifest_path, to_string(i.kind));
        if (inventory.empty()) inventory = "none";
        if (amdvlk)
            out.push_back(item("vulkan_driver", CheckStatus::Warn,
                               "AMDVLK ICD present — it can hijack ICD selection; RADV is the default (TRD §3.2). ICDs: " +
                                   inventory));
        else if (!radv)
            out.push_back(item("vulkan_driver", CheckStatus::Warn, "RADV ICD not found. ICDs: " + inventory));
        else
            out.push_back(item("vulkan_driver", CheckStatus::Ok,
                               std::string(info.vulkan.override_active ? "RADV (loader override active). ICDs: "
                                                                       : "RADV, no AMDVLK. ICDs: ") +
                                   inventory));
    }

    // kernel
    if (!info.os.kernel_version)
        out.push_back(item("kernel", CheckStatus::Unknown, "kernel release not readable"));
    else if (*info.os.kernel_version < kMinRocmKernel)
        out.push_back(item("kernel", CheckStatus::Warn,
                           std::format("{} < 6.19: ROCm/HIP path unsupported on gfx1151; Vulkan path unaffected",
                                       info.os.kernel_release.value_or("?"))));
    else
        out.push_back(item("kernel", CheckStatus::Ok, std::format("{} (>= 6.19)", info.os.kernel_release.value_or("?"))));

    // rocm
    if (!info.rocm.installed)
        out.push_back(item("rocm", CheckStatus::Warn, "ROCm not installed (/opt/rocm absent): HIP path unavailable"));
    else
        out.push_back(item("rocm", CheckStatus::Ok,
                           std::format("{} at {}; confirm it is the Phase-0-pinned build (TRD §3.3), not verified here",
                                       info.rocm.version.value_or("version unknown"), info.rocm.path.value_or("?"))));

    // performance_mode (TRD §49 — the BIOS/EC cTDP mode is not visible through sysfs)
    if (!gpu || !gpu->telemetry.performance_level) {
        out.push_back(item("performance_mode", CheckStatus::Unknown, "power_dpm_force_performance_level not readable"));
    } else {
        const auto& lvl = *gpu->telemetry.performance_level;
        const std::string profile = gpu->telemetry.power_profile.value_or("unknown");
        const bool pinned = lvl == "high" || lvl == "manual" || lvl.starts_with("profile_");
        out.push_back(item("performance_mode", pinned ? CheckStatus::Ok : CheckStatus::Warn,
                           std::format("power_dpm_force_performance_level={}, power profile={}{}; cTDP mode not "
                                       "observable via sysfs",
                                       lvl, profile, pinned ? "" : " (clocks float: pin for benchmarks, TRD §49)")));
    }
    return out;
}

void to_json(nlohmann::json& j, const CheckItem& v) {
    j = {{"check", v.check}, {"status", std::string(to_string(v.status))}, {"detail", v.detail}};
}

}  // namespace halo::hardware
