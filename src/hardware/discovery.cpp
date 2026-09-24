#include <algorithm>
#include <cstdlib>
#include <format>
#include <limits>
#include <set>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"
#include "halo/hardware/hardware.h"
#include "hardware/fs.h"

#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace halo::hardware {
namespace {

namespace fs = std::filesystem;
using detail::is_dir;
using detail::list_dir;
using detail::read_text;
using detail::under_root;

std::optional<std::uint64_t> read_u64(const fs::path& p) {
    const auto t = read_text(p);
    return t ? parse_u64(*t) : std::nullopt;
}

std::optional<std::string> read_trimmed(const fs::path& p) {
    const auto t = read_text(p);
    if (!t) return std::nullopt;
    const auto s = trim(*t);
    if (s.empty()) return std::nullopt;
    return std::string(s);
}

// Numeric suffix of "card12" / "12"; nullopt if not all digits after the prefix.
std::optional<std::uint64_t> index_after(std::string_view name, std::string_view prefix) {
    if (!name.starts_with(prefix)) return std::nullopt;
    return parse_u64(name.substr(prefix.size()));
}

std::optional<std::uint32_t> narrow_u32(std::optional<std::uint64_t> v) {
    if (!v || *v > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
    return static_cast<std::uint32_t>(*v);
}

// ---- CPU ------------------------------------------------------------------------------

CpuInfo discover_cpu(const fs::path& root, bool probe_runtime) {
    CpuInfo cpu;
    if (const auto text = read_text(under_root(root, "/proc/cpuinfo"))) {
        const auto parsed = parse_cpuinfo(*text);
        cpu.model_name = parsed.model_name;
        cpu.vendor_id = parsed.vendor_id;
        cpu.logical_threads = parsed.logical_threads;
        cpu.physical_cores = parsed.physical_cores != 0 ? parsed.physical_cores : parsed.logical_threads;
        cpu.sockets = parsed.sockets;
        cpu.simd = parsed.simd;
    }
    // L3: sum sizes over distinct shared_cpu_list values of level-3 caches.
    const auto cpu_dir = under_root(root, "/sys/devices/system/cpu");
    std::set<std::string> seen;
    std::uint64_t total = 0;
    bool any = false;
    for (const auto& name : list_dir(cpu_dir)) {
        if (!index_after(name, "cpu")) continue;
        const auto cache_dir = cpu_dir / name / "cache";
        for (const auto& idx : list_dir(cache_dir)) {
            if (!idx.starts_with("index")) continue;
            const auto d = cache_dir / idx;
            if (read_u64(d / "level") != 3u) continue;
            const auto shared = read_trimmed(d / "shared_cpu_list");
            const auto size_text = read_text(d / "size");
            const auto size = size_text ? parse_cache_size(*size_text) : std::nullopt;
            if (!shared || !size) continue;
            if (!seen.insert(*shared).second) continue;
            if (__builtin_add_overflow(total, *size, &total)) return cpu;  // corrupt input
            any = true;
        }
    }
    if (any) {
        cpu.l3_bytes = total;
        cpu.l3_instances = static_cast<std::uint32_t>(seen.size());
    }
    if (probe_runtime) cpu.runtime_simd = runtime_simd_flags();
    return cpu;
}

// ---- GPU ------------------------------------------------------------------------------

struct KfdNode {
    std::uint32_t id = 0;
    std::map<std::string, std::uint64_t> props;
    bool used = false;
};

std::vector<KfdNode> discover_kfd_gpu_nodes(const fs::path& root) {
    std::vector<KfdNode> nodes;
    const auto dir = under_root(root, "/sys/class/kfd/kfd/topology/nodes");
    for (const auto& name : list_dir(dir)) {
        const auto id = narrow_u32(parse_u64(name));
        if (!id) continue;
        const auto text = read_text(dir / name / "properties");
        if (!text) continue;
        auto props = parse_kfd_properties(*text);
        const auto it = props.find("gfx_target_version");
        if (it == props.end() || it->second == 0) continue;  // CPU node
        nodes.push_back(KfdNode{*id, std::move(props), false});
    }
    std::ranges::sort(nodes, {}, &KfdNode::id);
    return nodes;
}

void apply_kfd(GpuInfo& gpu, const KfdNode& node, std::string association) {
    auto get = [&](const char* key) -> std::optional<std::uint64_t> {
        const auto it = node.props.find(key);
        return it == node.props.end() ? std::nullopt : std::optional(it->second);
    };
    gpu.kfd_node = node.id;
    gpu.kfd_association = std::move(association);
    gpu.gfx_target_version = get("gfx_target_version");
    if (gpu.gfx_target_version) gpu.gfx_target = decode_gfx_target_version(*gpu.gfx_target_version);
    gpu.simd_count = narrow_u32(get("simd_count"));
    gpu.simd_per_cu = narrow_u32(get("simd_per_cu"));
    if (gpu.simd_count && gpu.simd_per_cu && *gpu.simd_per_cu != 0 && *gpu.simd_count % *gpu.simd_per_cu == 0)
        gpu.compute_units = *gpu.simd_count / *gpu.simd_per_cu;
    gpu.wavefront_size = narrow_u32(get("wave_front_size"));
    gpu.max_engine_clock_mhz = narrow_u32(get("max_engine_clk_fcompute"));
}

std::optional<std::uint32_t> active_mhz(const std::vector<DpmLevel>& levels) {
    for (const auto& l : levels)
        if (l.active) return l.mhz;
    return std::nullopt;
}

GpuTelemetry discover_telemetry(const fs::path& dev) {
    GpuTelemetry t;
    for (const auto& h : list_dir(dev / "hwmon")) {
        const auto d = dev / "hwmon" / h;
        if (!t.temperature_c) {
            if (const auto v = read_text(d / "temp1_input")) {
                if (const auto mc = parse_i64(*v)) t.temperature_c = static_cast<double>(*mc) / 1000.0;
            }
        }
        if (!t.power_w) {
            auto uw = read_u64(d / "power1_average");
            if (!uw) uw = read_u64(d / "power1_input");
            if (uw) t.power_w = static_cast<double>(*uw) / 1e6;
        }
    }
    if (const auto s = read_text(dev / "pp_dpm_sclk")) t.sclk_levels = parse_dpm_levels(*s);
    if (const auto s = read_text(dev / "pp_dpm_mclk")) t.mclk_levels = parse_dpm_levels(*s);
    t.sclk_mhz = active_mhz(t.sclk_levels);
    t.mclk_mhz = active_mhz(t.mclk_levels);
    t.performance_level = read_trimmed(dev / "power_dpm_force_performance_level");
    if (const auto s = read_text(dev / "pp_power_profile_mode")) t.power_profile = parse_active_power_profile(*s);
    return t;
}

std::vector<GpuInfo> discover_gpus(const fs::path& root) {
    const auto drm = under_root(root, "/sys/class/drm");
    std::vector<std::pair<std::uint64_t, std::string>> cards;
    for (const auto& name : list_dir(drm)) {
        if (const auto idx = index_after(name, "card")) cards.emplace_back(*idx, name);
    }
    std::ranges::sort(cards);

    std::vector<GpuInfo> gpus;
    for (const auto& [idx, name] : cards) {
        const auto dev = drm / name / "device";
        const auto vendor_text = read_text(dev / "vendor");
        const auto vendor = vendor_text ? parse_hex_u32(*vendor_text) : std::nullopt;
        if (vendor != kAmdPciVendor) continue;
        GpuInfo g;
        g.drm_card = name;
        g.vendor_id = *vendor;
        if (const auto t = read_text(dev / "device")) g.device_id = parse_hex_u32(*t).value_or(0);
        if (const auto ue = read_text(dev / "uevent")) {
            for (const auto line : split_lines(*ue)) {
                if (line.starts_with("PCI_SLOT_NAME=")) g.pci_slot = std::string(trim(line.substr(14)));
            }
        }
        g.vram_total = read_u64(dev / "mem_info_vram_total");
        g.vram_used = read_u64(dev / "mem_info_vram_used");
        g.vis_vram_total = read_u64(dev / "mem_info_vis_vram_total");
        g.gtt_total = read_u64(dev / "mem_info_gtt_total");
        g.gtt_used = read_u64(dev / "mem_info_gtt_used");
        g.telemetry = discover_telemetry(dev);
        gpus.push_back(std::move(g));
    }

    auto nodes = discover_kfd_gpu_nodes(root);
    for (auto& g : gpus) {
        if (!g.pci_slot) continue;
        const auto addr = parse_pci_slot_name(*g.pci_slot);
        if (!addr) continue;
        for (auto& n : nodes) {
            if (n.used) continue;
            const auto loc = n.props.find("location_id");
            const auto dom = n.props.find("domain");
            const std::uint64_t domain = dom == n.props.end() ? 0 : dom->second;
            if (loc != n.props.end() && loc->second == addr->location_id && domain == addr->domain) {
                apply_kfd(g, n, "pci");
                n.used = true;
                break;
            }
        }
    }
    // Fallback: exactly one unmatched card and one unmatched GPU node.
    const auto unmatched_cards = std::ranges::count_if(gpus, [](const GpuInfo& g) { return !g.kfd_node; });
    const auto unmatched_nodes = std::ranges::count_if(nodes, [](const KfdNode& n) { return !n.used; });
    if (unmatched_cards == 1 && unmatched_nodes == 1) {
        auto g = std::ranges::find_if(gpus, [](const GpuInfo& x) { return !x.kfd_node; });
        auto n = std::ranges::find_if(nodes, [](const KfdNode& x) { return !x.used; });
        apply_kfd(*g, *n, "single-device");
        n->used = true;
    }
    return gpus;
}

// ---- OS / drivers ----------------------------------------------------------------------

OsInfo discover_os(const fs::path& root) {
    OsInfo os;
    os.kernel_release = read_trimmed(under_root(root, "/proc/sys/kernel/osrelease"));
    if (os.kernel_release) os.kernel_version = parse_kernel_version(*os.kernel_release);
    auto rel = read_text(under_root(root, "/etc/os-release"));
    if (!rel) rel = read_text(under_root(root, "/usr/lib/os-release"));
    if (rel) {
        const auto kv = parse_os_release(*rel);
        auto get = [&](const char* k) -> std::optional<std::string> {
            const auto it = kv.find(k);
            return it == kv.end() ? std::nullopt : std::optional(it->second);
        };
        os.distro_pretty_name = get("PRETTY_NAME");
        os.distro_id = get("ID");
        os.distro_version_id = get("VERSION_ID");
    }
    return os;
}

RocmInfo discover_rocm(const fs::path& root) {
    RocmInfo r;
    std::vector<std::string> candidates{"/opt/rocm"};
    for (const auto& name : list_dir(under_root(root, "/opt"))) {
        if (name.starts_with("rocm-")) candidates.push_back("/opt/" + name);
    }
    for (const auto& c : candidates) {
        const auto base = under_root(root, c);
        if (!is_dir(base)) continue;
        r.installed = true;
        r.path = c;
        if (const auto v = read_text(base / ".info" / "version")) {
            const auto lines = split_lines(*v);
            if (!lines.empty() && !trim(lines.front()).empty()) {
                r.version = std::string(trim(lines.front()));
                return r;
            }
        }
    }
    return r;
}

std::optional<std::string> env_lookup(const DiscoveryOptions& o, const char* key) {
    if (const auto it = o.env.find(key); it != o.env.end()) return it->second;
    if (o.use_process_env) {
        if (const char* v = std::getenv(key)) return std::string(v);
    }
    return std::nullopt;
}

void add_icd_manifest(VulkanInfo& vk, const fs::path& root, const std::string& sys_path,
                      const std::string& source) {
    VulkanIcd icd;
    icd.manifest_path = sys_path;
    icd.source = source;
    const auto text = read_text(under_root(root, sys_path), 1U << 20);
    if (!text) {
        icd.error = "unreadable or larger than 1 MiB";
    } else {
        const auto j = nlohmann::json::parse(*text, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) {
            icd.error = "malformed JSON";
        } else if (const auto it = j.find("ICD"); it == j.end() || !it->is_object()) {
            icd.error = "missing ICD object";
        } else {
            if (const auto lp = it->find("library_path"); lp != it->end() && lp->is_string())
                icd.library_path = lp->get<std::string>();
            if (const auto av = it->find("api_version"); av != it->end() && av->is_string())
                icd.api_version = av->get<std::string>();
            if (!icd.library_path) icd.error = "missing ICD.library_path";
        }
    }
    const auto fname = fs::path(sys_path).filename().string();
    icd.kind = classify_vulkan_icd(fname, icd.library_path.value_or(""));
    vk.icds.push_back(std::move(icd));
}

void add_icd_path(VulkanInfo& vk, const fs::path& root, const std::string& sys_path,
                  const std::string& source) {
    const auto p = under_root(root, sys_path);
    if (is_dir(p)) {
        std::string base = sys_path;
        if (!base.empty() && base.back() == '/') base.pop_back();
        for (const auto& name : list_dir(p)) {
            if (name.ends_with(".json")) add_icd_manifest(vk, root, base + "/" + name, source);
        }
    } else {
        add_icd_manifest(vk, root, sys_path, source);
    }
}

std::vector<std::string> split_colon(std::string_view s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        auto end = s.find(':', start);
        if (end == std::string_view::npos) end = s.size();
        const auto part = trim(s.substr(start, end - start));
        if (!part.empty()) out.emplace_back(part);
        start = end + 1;
    }
    return out;
}

VulkanInfo discover_vulkan(const fs::path& root, const DiscoveryOptions& o) {
    VulkanInfo vk;
    // Loader precedence: VK_DRIVER_FILES, then the legacy VK_ICD_FILENAMES; either one
    // replaces the search directories. VK_ADD_DRIVER_FILES is added on top.
    std::optional<std::string> override_list = env_lookup(o, "VK_DRIVER_FILES");
    std::string override_src = "VK_DRIVER_FILES";
    if (!override_list || trim(*override_list).empty()) {
        override_list = env_lookup(o, "VK_ICD_FILENAMES");
        override_src = "VK_ICD_FILENAMES";
    }
    if (override_list && !trim(*override_list).empty()) {
        vk.override_active = true;
        for (const auto& p : split_colon(*override_list)) add_icd_path(vk, root, p, override_src);
    } else {
        for (const char* dir : {"/etc/xdg/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d",
                                "/usr/share/vulkan/icd.d"}) {
            if (is_dir(under_root(root, dir))) add_icd_path(vk, root, dir, "search-dir");
        }
    }
    if (const auto add = env_lookup(o, "VK_ADD_DRIVER_FILES")) {
        for (const auto& p : split_colon(*add)) add_icd_path(vk, root, p, "VK_ADD_DRIVER_FILES");
    }
    return vk;
}

KernelCmdline discover_cmdline(const fs::path& root) {
    KernelCmdline c;
    c.raw = read_trimmed(under_root(root, "/proc/cmdline"));
    if (!c.raw) return c;
    for (const auto& [k, v] : parse_cmdline(*c.raw)) {
        if (k == "amdgpu.gttsize") c.amdgpu_gttsize_mib = parse_i64(v);
        else if (k == "ttm.pages_limit") {
            c.ttm_pages_limit = parse_u64(v);
            std::uint64_t b = 0;
            if (c.ttm_pages_limit && !__builtin_mul_overflow(*c.ttm_pages_limit, std::uint64_t{4096}, &b))
                c.ttm_pages_limit_bytes_4k = b;
        } else if (k == "amd_iommu") c.amd_iommu = v;
        else if (k == "iommu") c.iommu = v;
    }
    return c;
}

IommuInfo discover_iommu(const fs::path& root, const KernelCmdline& cmd) {
    IommuInfo info;
    if (cmd.amd_iommu == "off" || cmd.iommu == "off") {
        info.state = IommuState::Off;
        info.evidence = cmd.amd_iommu == "off" ? "kernel cmdline amd_iommu=off" : "kernel cmdline iommu=off";
        return info;
    }
    const auto dir = under_root(root, "/sys/class/iommu");
    const auto entries = list_dir(dir);
    if (cmd.iommu == "pt") {
        info.state = IommuState::Passthrough;
        info.evidence = std::format("kernel cmdline iommu=pt ({} IOMMU device(s) registered)", entries.size());
        return info;
    }
    if (!entries.empty()) {
        info.state = IommuState::On;
        info.evidence = std::format("/sys/class/iommu lists {} device(s) (first: {})", entries.size(), entries.front());
    } else if (is_dir(dir)) {
        info.state = IommuState::Off;
        info.evidence = "/sys/class/iommu is empty";
    } else {
        info.state = IommuState::Unknown;
        info.evidence = "/sys/class/iommu not present";
    }
    return info;
}

std::uint64_t sat_add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    return __builtin_add_overflow(a, b, &r) ? std::numeric_limits<std::uint64_t>::max() : r;
}

std::string gib(std::uint64_t b) { return std::format("{:.2f} GiB", static_cast<double>(b) / (1ULL << 30)); }

}  // namespace

std::string_view to_string(MemoryTopology t) noexcept {
    switch (t) {
        case MemoryTopology::NoGpu: return "no_gpu";
        case MemoryTopology::CarveoutPrimary: return "carveout_primary";
        case MemoryTopology::GttPrimary: return "gtt_primary";
    }
    return "?";
}

std::string_view to_string(IommuState s) noexcept {
    switch (s) {
        case IommuState::Off: return "off";
        case IommuState::On: return "on";
        case IommuState::Passthrough: return "passthrough";
        case IommuState::Unknown: return "unknown";
    }
    return "unknown";
}

bool VulkanInfo::has(VulkanDriverKind k) const noexcept {
    return std::ranges::any_of(icds, [k](const VulkanIcd& i) { return i.kind == k; });
}

MemoryTiers make_memory_tiers(const std::vector<GpuInfo>& gpus, const HostMemory& host) {
    MemoryTiers t;
    t.host_total = host.total_bytes;
    t.host_available = host.available_bytes;
    const auto primary = std::ranges::find_if(gpus, [](const GpuInfo& g) { return g.vram_total || g.gtt_total; });
    if (primary == gpus.end()) {
        t.topology = MemoryTopology::NoGpu;
        return t;
    }
    t.vram_total = primary->vram_total;
    t.vram_used = primary->vram_used;
    t.vis_vram_total = primary->vis_vram_total;
    t.gtt_total = primary->gtt_total;
    t.gtt_used = primary->gtt_used;
    const std::uint64_t vram = t.vram_total.value_or(0);
    const std::uint64_t gtt = t.gtt_total.value_or(0);
    t.topology = vram >= gtt ? MemoryTopology::CarveoutPrimary : MemoryTopology::GttPrimary;
    // GTT is backed by OS RAM: never count more GTT than the OS actually has.
    const std::uint64_t gtt_effective = host.total_bytes ? std::min(gtt, *host.total_bytes) : gtt;
    t.gpu_accessible_bytes = sat_add(vram, gtt_effective);
    return t;
}

std::vector<std::string> collect_warnings(const HardwareInfo& info) {
    std::vector<std::string> w;
    if (info.gpus.empty()) w.emplace_back("no AMD GPU (PCI vendor 0x1002) found under /sys/class/drm");
    for (const auto& icd : info.vulkan.icds) {
        if (icd.kind == VulkanDriverKind::Amdvlk)
            w.push_back(std::format(
                "AMDVLK Vulkan ICD present ({}): it can silently take over ICD selection and has regressed "
                "prefill in field reports; HALO's default and reference driver is RADV (TRD §3.2)",
                icd.manifest_path));
    }
    if (!info.gpus.empty() && !info.vulkan.has(VulkanDriverKind::Radv))
        w.emplace_back("no RADV Vulkan ICD (libvulkan_radeon.so) found; the Vulkan backend's reference driver is RADV");
    if (info.os.kernel_version && *info.os.kernel_version < kMinRocmKernel)
        w.push_back(std::format("kernel {} is older than 6.19: the ROCm/HIP path on gfx1151 requires >= 6.19 "
                                "(the Vulkan path is unaffected)",
                                info.os.kernel_release.value_or("?")));
    if (info.iommu.state == IommuState::On || info.iommu.state == IommuState::Passthrough)
        w.push_back(std::format("IOMMU is {} ({}); TRD §59 expects it disabled (~6% memory-read cost in field reports)",
                                to_string(info.iommu.state), info.iommu.evidence));
    return w;
}

std::optional<SimdFlags> runtime_simd_flags() noexcept {
#if defined(__x86_64__)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid(1, &a, &b, &c, &d) == 0) return std::nullopt;
    const bool osxsave = (c & (1U << 27)) != 0;
    const bool fma = (c & (1U << 12)) != 0;
    std::uint64_t xcr0 = 0;
    if (osxsave) {
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = (static_cast<std::uint64_t>(hi) << 32) | lo;
    }
    const bool os_avx = (xcr0 & 0x6) == 0x6;       // XMM + YMM state
    const bool os_avx512 = (xcr0 & 0xe6) == 0xe6;  // + opmask, ZMM_Hi256, Hi16_ZMM
    SimdFlags f;
    f.fma = fma && os_avx;
    unsigned max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf >= 7) {
        unsigned a7 = 0, b7 = 0, c7 = 0, d7 = 0;
        __cpuid_count(7, 0, a7, b7, c7, d7);
        f.avx2 = os_avx && (b7 & (1U << 5)) != 0;
        f.avx512f = os_avx512 && (b7 & (1U << 16)) != 0;
        f.avx512dq = os_avx512 && (b7 & (1U << 17)) != 0;
        f.avx512bw = os_avx512 && (b7 & (1U << 30)) != 0;
        f.avx512vl = os_avx512 && (b7 & (1U << 31)) != 0;
        f.avx512_vnni = os_avx512 && (c7 & (1U << 11)) != 0;
        unsigned a71 = 0, b71 = 0, c71 = 0, d71 = 0;
        __cpuid_count(7, 1, a71, b71, c71, d71);
        f.avx_vnni = os_avx && (a71 & (1U << 4)) != 0;
        f.avx512_bf16 = os_avx512 && (a71 & (1U << 5)) != 0;
    }
    return f;
#else
    return std::nullopt;
#endif
}

HardwareInfo discover(const DiscoveryOptions& options) {
    HALO_CHECK(is_dir(options.root), ErrorCode::Device, "discovery root '{}' is not a directory",
               options.root.string());
    HardwareInfo info;
    info.root = options.root.string();
    const auto& root = options.root;
    info.cpu = discover_cpu(root, options.probe_runtime_cpuid);
    if (const auto text = read_text(under_root(root, "/proc/meminfo"))) {
        const auto m = parse_meminfo(*text);
        info.host_memory.total_bytes = m.total_bytes;
        info.host_memory.available_bytes = m.available_bytes;
    }
    info.gpus = discover_gpus(root);
    info.tiers = make_memory_tiers(info.gpus, info.host_memory);
    info.os = discover_os(root);
    info.rocm = discover_rocm(root);
    info.vulkan = discover_vulkan(root, options);
    info.cmdline = discover_cmdline(root);
    info.iommu = discover_iommu(root, info.cmdline);
    info.warnings = collect_warnings(info);
    return info;
}

// ---- JSON ---------------------------------------------------------------------------------

namespace {
template <typename T>
nlohmann::json opt(const std::optional<T>& v) {
    return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
}
nlohmann::json levels_json(const std::vector<DpmLevel>& levels) {
    auto a = nlohmann::json::array();
    for (const auto& l : levels) a.push_back({{"index", l.index}, {"mhz", l.mhz}, {"active", l.active}});
    return a;
}
}  // namespace

void to_json(nlohmann::json& j, const SimdFlags& v) {
    j = {{"avx2", v.avx2},         {"fma", v.fma},           {"avx512f", v.avx512f},
         {"avx512bw", v.avx512bw}, {"avx512vl", v.avx512vl}, {"avx512dq", v.avx512dq},
         {"avx512_vnni", v.avx512_vnni}, {"avx512_bf16", v.avx512_bf16}, {"avx_vnni", v.avx_vnni}};
}

void to_json(nlohmann::json& j, const MemoryTiers& v) {
    j = {{"topology", std::string(to_string(v.topology))},
         {"vram_total", opt(v.vram_total)},
         {"vram_used", opt(v.vram_used)},
         {"vis_vram_total", opt(v.vis_vram_total)},
         {"gtt_total", opt(v.gtt_total)},
         {"gtt_used", opt(v.gtt_used)},
         {"host_total", opt(v.host_total)},
         {"host_available", opt(v.host_available)},
         {"gpu_accessible_bytes", v.gpu_accessible_bytes},
         {"gpu_accessible_human", gib(v.gpu_accessible_bytes)}};
}

void to_json(nlohmann::json& j, const HardwareInfo& v) {
    nlohmann::json cpu = {{"model_name", v.cpu.model_name},
                          {"vendor_id", v.cpu.vendor_id},
                          {"logical_threads", v.cpu.logical_threads},
                          {"physical_cores", v.cpu.physical_cores},
                          {"sockets", v.cpu.sockets},
                          {"l3_bytes", opt(v.cpu.l3_bytes)},
                          {"l3_instances", v.cpu.l3_instances},
                          {"simd", v.cpu.simd},
                          {"runtime_simd", v.cpu.runtime_simd ? nlohmann::json(*v.cpu.runtime_simd) : nlohmann::json(nullptr)},
                          {"runtime_simd_note", "CPUID of the process running discovery, not of the inspected root"}};
    auto gpus = nlohmann::json::array();
    for (const auto& g : v.gpus) {
        const auto& t = g.telemetry;
        gpus.push_back({{"drm_card", g.drm_card},
                        {"pci_slot", opt(g.pci_slot)},
                        {"vendor_id", std::format("0x{:04x}", g.vendor_id)},
                        {"device_id", std::format("0x{:04x}", g.device_id)},
                        {"vram_total", opt(g.vram_total)},
                        {"vram_used", opt(g.vram_used)},
                        {"vis_vram_total", opt(g.vis_vram_total)},
                        {"gtt_total", opt(g.gtt_total)},
                        {"gtt_used", opt(g.gtt_used)},
                        {"kfd_node", opt(g.kfd_node)},
                        {"kfd_association", g.kfd_association},
                        {"gfx_target_version", opt(g.gfx_target_version)},
                        {"gfx_target", opt(g.gfx_target)},
                        {"compute_units", opt(g.compute_units)},
                        {"simd_count", opt(g.simd_count)},
                        {"simd_per_cu", opt(g.simd_per_cu)},
                        {"wavefront_size", opt(g.wavefront_size)},
                        {"max_engine_clock_mhz", opt(g.max_engine_clock_mhz)},
                        {"telemetry",
                         {{"temperature_c", opt(t.temperature_c)},
                          {"power_w", opt(t.power_w)},
                          {"sclk_mhz", opt(t.sclk_mhz)},
                          {"mclk_mhz", opt(t.mclk_mhz)},
                          {"sclk_levels", levels_json(t.sclk_levels)},
                          {"mclk_levels", levels_json(t.mclk_levels)},
                          {"performance_level", opt(t.performance_level)},
                          {"power_profile", opt(t.power_profile)}}}});
    }
    auto icds = nlohmann::json::array();
    for (const auto& i : v.vulkan.icds)
        icds.push_back({{"manifest_path", i.manifest_path},
                        {"source", i.source},
                        {"library_path", opt(i.library_path)},
                        {"api_version", opt(i.api_version)},
                        {"driver", std::string(to_string(i.kind))},
                        {"error", opt(i.error)}});
    nlohmann::json kv = nullptr;
    if (v.os.kernel_version)
        kv = {v.os.kernel_version->major, v.os.kernel_version->minor, v.os.kernel_version->patch};
    j = {{"root", v.root},
         {"cpu", cpu},
         {"host_memory", {{"total_bytes", opt(v.host_memory.total_bytes)}, {"available_bytes", opt(v.host_memory.available_bytes)}}},
         {"gpus", gpus},
         {"memory_tiers", v.tiers},
         {"os",
          {{"kernel_release", opt(v.os.kernel_release)},
           {"kernel_version", kv},
           {"distro_pretty_name", opt(v.os.distro_pretty_name)},
           {"distro_id", opt(v.os.distro_id)},
           {"distro_version_id", opt(v.os.distro_version_id)}}},
         {"rocm", {{"installed", v.rocm.installed}, {"version", opt(v.rocm.version)}, {"path", opt(v.rocm.path)}}},
         {"vulkan", {{"override_active", v.vulkan.override_active}, {"icds", icds}}},
         {"iommu", {{"state", std::string(to_string(v.iommu.state))}, {"evidence", v.iommu.evidence}}},
         {"cmdline",
          {{"raw", opt(v.cmdline.raw)},
           {"amdgpu_gttsize_mib", opt(v.cmdline.amdgpu_gttsize_mib)},
           {"ttm_pages_limit", opt(v.cmdline.ttm_pages_limit)},
           {"ttm_pages_limit_bytes_assuming_4k_pages", opt(v.cmdline.ttm_pages_limit_bytes_4k)},
           {"amd_iommu", opt(v.cmdline.amd_iommu)},
           {"iommu", opt(v.cmdline.iommu)}}},
         {"warnings", v.warnings}};
}

}  // namespace halo::hardware
