#include "halo/hardware/parse.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <limits>

namespace halo::hardware {
namespace {

constexpr bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Split on runs of whitespace.
std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_space(s[i])) ++i;
        const std::size_t start = i;
        while (i < s.size() && !is_space(s[i])) ++i;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

bool mul_ok(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    return !__builtin_mul_overflow(a, b, &out);
}

// "key<sep>value" -> trimmed pair; nullopt if no separator.
std::optional<std::pair<std::string_view, std::string_view>> split_kv(std::string_view line,
                                                                      char sep) {
    const auto pos = line.find(sep);
    if (pos == std::string_view::npos) return std::nullopt;
    return std::pair{trim(line.substr(0, pos)), trim(line.substr(pos + 1))};
}

bool name_char(char c) noexcept {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

}  // namespace

std::string_view trim(std::string_view s) noexcept {
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
    return s;
}

std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        auto end = text.find('\n', start);
        if (end == std::string_view::npos) end = text.size();
        std::string_view line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (end == text.size()) {
            if (!line.empty()) out.push_back(line);
            break;
        }
        out.push_back(line);
        start = end + 1;
    }
    return out;
}

std::optional<std::uint64_t> parse_u64(std::string_view s) noexcept {
    s = trim(s);
    if (s.empty()) return std::nullopt;
    std::uint64_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v, 10);
    if (ec != std::errc{} || ptr != s.data() + s.size()) return std::nullopt;
    return v;
}

std::optional<std::int64_t> parse_i64(std::string_view s) noexcept {
    s = trim(s);
    if (s.empty()) return std::nullopt;
    std::int64_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v, 10);
    if (ec != std::errc{} || ptr != s.data() + s.size()) return std::nullopt;
    return v;
}

std::optional<std::uint32_t> parse_hex_u32(std::string_view s) noexcept {
    s = trim(s);
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s.remove_prefix(2);
    if (s.empty()) return std::nullopt;
    std::uint32_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v, 16);
    if (ec != std::errc{} || ptr != s.data() + s.size()) return std::nullopt;
    return v;
}

MemInfo parse_meminfo(std::string_view text) {
    MemInfo out;
    for (const auto line : split_lines(text)) {
        const auto kv = split_kv(line, ':');
        if (!kv) continue;
        auto [key, value] = *kv;
        std::optional<std::uint64_t>* dst = nullptr;
        if (key == "MemTotal") dst = &out.total_bytes;
        else if (key == "MemAvailable") dst = &out.available_bytes;
        else continue;
        const auto parts = split_ws(value);
        if (parts.empty() || parts.size() > 2) continue;
        std::uint64_t mult = 1;
        if (parts.size() == 2) {
            if (parts[1] == "kB") mult = 1024;
            else continue;  // unknown unit: refuse rather than guess
        }
        const auto n = parse_u64(parts[0]);
        std::uint64_t bytes = 0;
        if (n && mul_ok(*n, mult, bytes)) *dst = bytes;
    }
    return out;
}

SimdFlags simd_from_flags(std::string_view flags) {
    SimdFlags f;
    for (const auto tok : split_ws(flags)) {
        if (tok == "avx2") f.avx2 = true;
        else if (tok == "fma") f.fma = true;
        else if (tok == "avx512f") f.avx512f = true;
        else if (tok == "avx512bw") f.avx512bw = true;
        else if (tok == "avx512vl") f.avx512vl = true;
        else if (tok == "avx512dq") f.avx512dq = true;
        else if (tok == "avx512_vnni") f.avx512_vnni = true;
        else if (tok == "avx512_bf16") f.avx512_bf16 = true;
        else if (tok == "avx_vnni") f.avx_vnni = true;
    }
    return f;
}

CpuInfoText parse_cpuinfo(std::string_view text) {
    CpuInfoText out;
    std::map<std::string, std::uint32_t> cores_by_package;
    std::string current_package;
    std::optional<std::uint32_t> current_cores;
    bool have_flags = false;
    auto flush = [&] {
        if (current_cores) {
            auto& slot = cores_by_package[current_package];
            slot = std::max(slot, *current_cores);
        }
        current_package.clear();
        current_cores.reset();
    };
    for (const auto line : split_lines(text)) {
        if (trim(line).empty()) {
            flush();
            continue;
        }
        const auto kv = split_kv(line, ':');
        if (!kv) continue;
        const auto [key, value] = *kv;
        if (key == "processor") {
            if (parse_u64(value)) ++out.logical_threads;
        } else if (key == "model name" && out.model_name.empty()) {
            out.model_name = std::string(value);
        } else if (key == "vendor_id" && out.vendor_id.empty()) {
            out.vendor_id = std::string(value);
        } else if (key == "physical id") {
            current_package = std::string(value);
        } else if (key == "cpu cores") {
            if (const auto n = parse_u64(value); n && *n <= std::numeric_limits<std::uint32_t>::max())
                current_cores = static_cast<std::uint32_t>(*n);
        } else if (key == "flags" && !have_flags) {
            out.simd = simd_from_flags(value);
            have_flags = true;
        }
    }
    flush();
    out.sockets = static_cast<std::uint32_t>(cores_by_package.size());
    std::uint64_t cores = 0;
    for (const auto& [pkg, n] : cores_by_package) cores += n;
    out.physical_cores = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(cores, std::numeric_limits<std::uint32_t>::max()));
    return out;
}

std::optional<std::uint64_t> parse_cache_size(std::string_view s) noexcept {
    s = trim(s);
    if (s.empty()) return std::nullopt;
    std::uint64_t mult = 1;
    const char last = s.back();
    if (last == 'K' || last == 'k') mult = 1024;
    else if (last == 'M' || last == 'm') mult = 1024ULL * 1024;
    else if (last == 'G' || last == 'g') mult = 1024ULL * 1024 * 1024;
    if (mult != 1) s.remove_suffix(1);
    const auto n = parse_u64(s);
    std::uint64_t bytes = 0;
    if (!n || !mul_ok(*n, mult, bytes)) return std::nullopt;
    return bytes;
}

std::optional<KernelVersion> parse_kernel_version(std::string_view release) noexcept {
    release = trim(release);
    KernelVersion v;
    std::uint32_t* fields[] = {&v.major, &v.minor, &v.patch};
    std::size_t i = 0;
    int parsed = 0;
    for (auto* field : fields) {
        const char* begin = release.data() + i;
        const char* end = release.data() + release.size();
        if (begin == end || !std::isdigit(static_cast<unsigned char>(*begin))) break;
        const auto [ptr, ec] = std::from_chars(begin, end, *field, 10);
        if (ec != std::errc{}) return std::nullopt;
        ++parsed;
        i = static_cast<std::size_t>(ptr - release.data());
        if (i < release.size() && release[i] == '.') ++i;
        else break;
    }
    if (parsed < 2) return std::nullopt;  // need at least major.minor
    return v;
}

std::map<std::string, std::string> parse_os_release(std::string_view text) {
    std::map<std::string, std::string> out;
    for (const auto raw : split_lines(text)) {
        const auto line = trim(raw);
        if (line.empty() || line.front() == '#') continue;
        const auto kv = split_kv(line, '=');
        if (!kv || kv->first.empty()) continue;
        auto value = kv->second;
        if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
            value.back() == value.front()) {
            value = value.substr(1, value.size() - 2);
        }
        out[std::string(kv->first)] = std::string(value);
    }
    return out;
}

std::vector<std::pair<std::string, std::string>> parse_cmdline(std::string_view text) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto tok : split_ws(text)) {
        const auto pos = tok.find('=');
        if (pos == std::string_view::npos) out.emplace_back(std::string(tok), std::string());
        else out.emplace_back(std::string(tok.substr(0, pos)), std::string(tok.substr(pos + 1)));
    }
    return out;
}

std::optional<std::string> decode_gfx_target_version(std::uint64_t v) {
    if (v == 0) return std::nullopt;
    const std::uint64_t major = v / 10000;
    const std::uint64_t minor = (v / 100) % 100;
    const std::uint64_t stepping = v % 100;
    if (major == 0 || minor >= 16 || stepping >= 16) return std::nullopt;
    constexpr char hex[] = "0123456789abcdef";
    std::string out = "gfx" + std::to_string(major);
    out += hex[minor];
    out += hex[stepping];
    return out;
}

std::map<std::string, std::uint64_t> parse_kfd_properties(std::string_view text) {
    std::map<std::string, std::uint64_t> out;
    for (const auto line : split_lines(text)) {
        const auto parts = split_ws(line);
        if (parts.size() != 2) continue;
        if (const auto n = parse_u64(parts[1])) out[std::string(parts[0])] = *n;
    }
    return out;
}

std::vector<DpmLevel> parse_dpm_levels(std::string_view text) {
    std::vector<DpmLevel> out;
    for (const auto line : split_lines(text)) {
        const auto kv = split_kv(line, ':');
        if (!kv || kv->first.empty()) continue;
        auto value = kv->second;
        DpmLevel lvl;
        lvl.index = std::string(kv->first);
        if (!value.empty() && value.back() == '*') {
            lvl.active = true;
            value = trim(value.substr(0, value.size() - 1));
        }
        // Expect "<digits>Mhz" (case-insensitive unit).
        std::size_t digits = 0;
        while (digits < value.size() && std::isdigit(static_cast<unsigned char>(value[digits])))
            ++digits;
        if (digits == 0) continue;
        std::string unit(value.substr(digits));
        std::ranges::transform(unit, unit.begin(),
                               [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
        if (trim(unit) != "mhz") continue;
        const auto n = parse_u64(value.substr(0, digits));
        if (!n || *n > std::numeric_limits<std::uint32_t>::max()) continue;
        lvl.mhz = static_cast<std::uint32_t>(*n);
        out.push_back(std::move(lvl));
    }
    return out;
}

std::optional<std::string> parse_active_power_profile(std::string_view text) {
    for (const auto line : split_lines(text)) {
        const auto toks = split_ws(line);
        for (std::size_t i = 0; i < toks.size(); ++i) {
            auto tok = toks[i];
            if (tok.find('*') == std::string_view::npos) continue;
            // Strip '*' and ':' decorations.
            std::string name;
            for (const char c : tok)
                if (c != '*' && c != ':') name += c;
            // Bare "*" / "*:" marker: the name is the previous token.
            if (name.empty() && i > 0) {
                for (const char c : toks[i - 1])
                    if (c != ':') name += c;
            }
            // Drop a "(...)" suffix, e.g. "BOOTUP_DEFAULT(0)".
            if (const auto p = name.find('('); p != std::string::npos) name.resize(p);
            if (!name.empty() && std::ranges::all_of(name, name_char) &&
                !std::ranges::all_of(name, [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }))
                return name;
        }
    }
    return std::nullopt;
}

std::optional<PciAddress> parse_pci_slot_name(std::string_view s) noexcept {
    // DDDD:BB:DD.F
    s = trim(s);
    const auto c1 = s.find(':');
    const auto c2 = s.find(':', c1 == std::string_view::npos ? 0 : c1 + 1);
    const auto dot = s.find('.', c2 == std::string_view::npos ? 0 : c2 + 1);
    if (c1 == std::string_view::npos || c2 == std::string_view::npos || dot == std::string_view::npos)
        return std::nullopt;
    const auto dom = parse_hex_u32(s.substr(0, c1));
    const auto bus = parse_hex_u32(s.substr(c1 + 1, c2 - c1 - 1));
    const auto dev = parse_hex_u32(s.substr(c2 + 1, dot - c2 - 1));
    const auto fn = parse_hex_u32(s.substr(dot + 1));
    if (!dom || !bus || !dev || !fn || *bus > 0xff || *dev > 0x1f || *fn > 0x7) return std::nullopt;
    return PciAddress{*dom, (*bus << 8) | (*dev << 3) | *fn};
}

std::string_view to_string(VulkanDriverKind k) noexcept {
    switch (k) {
        case VulkanDriverKind::Radv: return "radv";
        case VulkanDriverKind::Amdvlk: return "amdvlk";
        case VulkanDriverKind::Lavapipe: return "lavapipe";
        case VulkanDriverKind::Other: return "other";
        case VulkanDriverKind::Unknown: return "unknown";
    }
    return "unknown";
}

VulkanDriverKind classify_vulkan_icd(std::string_view manifest_filename, std::string_view library_path) {
    auto lower = [](std::string_view s) {
        std::string r(s);
        std::ranges::transform(r, r.begin(),
                               [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
        return r;
    };
    auto basename = [](std::string_view p) {
        const auto pos = p.find_last_of('/');
        return pos == std::string_view::npos ? p : p.substr(pos + 1);
    };
    const std::string lib = lower(basename(library_path));
    if (!lib.empty()) {
        if (lib.find("vulkan_radeon") != std::string::npos) return VulkanDriverKind::Radv;
        if (lib.find("amdvlk") != std::string::npos) return VulkanDriverKind::Amdvlk;
        if (lib.find("vulkan_lvp") != std::string::npos) return VulkanDriverKind::Lavapipe;
        return VulkanDriverKind::Other;
    }
    const std::string file = lower(basename(manifest_filename));
    if (file.starts_with("radeon_icd")) return VulkanDriverKind::Radv;
    if (file.starts_with("amd_icd") || file.find("amdvlk") != std::string::npos) return VulkanDriverKind::Amdvlk;
    if (file.starts_with("lvp_icd")) return VulkanDriverKind::Lavapipe;
    return file.empty() ? VulkanDriverKind::Unknown : VulkanDriverKind::Other;
}

}  // namespace halo::hardware
