#include "args.h"

#include <charconv>
#include <cmath>
#include <format>
#include <limits>

namespace halo::cli {

namespace {

const OptionSpec* find_long(const std::vector<OptionSpec>& specs, const std::string& name) {
    for (const auto& s : specs) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

const OptionSpec* find_short(const std::vector<OptionSpec>& specs, char c) {
    for (const auto& s : specs) {
        if (s.short_name != 0 && s.short_name == c) return &s;
    }
    return nullptr;
}

}  // namespace

ParsedArgs parse_args(const std::vector<std::string>& args, const std::vector<OptionSpec>& specs) {
    ParsedArgs p;
    bool only_positionals = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (only_positionals || a == "-" || !a.starts_with("-")) {
            p.positionals.push_back(a);
            continue;
        }
        if (a == "--") {
            only_positionals = true;
            continue;
        }
        const OptionSpec* spec = nullptr;
        std::optional<std::string> inline_value;
        std::string shown;
        if (a.starts_with("--")) {
            std::string name = a.substr(2);
            if (const auto eq = name.find('='); eq != std::string::npos) {
                inline_value = name.substr(eq + 1);
                name = name.substr(0, eq);
            }
            spec = find_long(specs, name);
            shown = "--" + name;
        } else if (a.size() == 2) {
            spec = find_short(specs, a[1]);
            shown = a;
        }
        if (spec == nullptr) throw UsageError("unknown option " + (shown.empty() ? a : shown));
        if (!spec->takes_value) {
            if (inline_value) throw UsageError("option --" + spec->name + " takes no value");
            p.flags_[spec->name] = true;
            continue;
        }
        std::string v;
        if (inline_value) {
            v = *inline_value;
        } else {
            if (i + 1 >= args.size()) throw UsageError("option --" + spec->name + " needs a value");
            v = args[++i];
        }
        auto& slot = p.values_[spec->name];
        if (!slot.empty() && !spec->repeatable) throw UsageError("option --" + spec->name + " given more than once");
        slot.push_back(std::move(v));
    }
    return p;
}

ParsedArgs parse_args(const std::vector<std::string>& args, const std::vector<OptionSpec>& specs,
                      const std::string& usage) {
    try {
        return parse_args(args, specs);
    } catch (const UsageError& e) {
        throw UsageError(std::string(e.what()) + "\n" + usage);
    }
}

std::uint64_t parse_u64(const std::string& s, const std::string& what, std::uint64_t lo, std::uint64_t hi) {
    std::uint64_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc() || ptr != s.data() + s.size() || v < lo || v > hi) {
        throw UsageError(std::format("{} must be an integer in [{}, {}] (got '{}')", what, lo, hi, s));
    }
    return v;
}

double parse_number(const std::string& s, const std::string& what, double lo, double hi) {
    double v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc() || ptr != s.data() + s.size() || !std::isfinite(v) || v < lo || v > hi) {
        throw UsageError(std::format("{} must be a number in [{}, {}] (got '{}')", what, lo, hi, s));
    }
    return v;
}

std::optional<std::uint64_t> ParsedArgs::u64(const std::string& name, std::uint64_t lo, std::uint64_t hi) const {
    const auto v = value(name);
    if (!v) return std::nullopt;
    return parse_u64(*v, "--" + name, lo, hi);
}

std::optional<double> ParsedArgs::number(const std::string& name, double lo, double hi) const {
    const auto v = value(name);
    if (!v) return std::nullopt;
    return parse_number(*v, "--" + name, lo, hi);
}

std::vector<std::string> split_list(const std::string& s, const std::string& what) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        const auto comma = s.find(',', start);
        std::string item = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) throw UsageError(what + ": empty item in list '" + s + "'");
        out.push_back(std::move(item));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

std::string options_help(const std::vector<OptionSpec>& specs) {
    std::string out;
    for (const auto& s : specs) {
        if (s.hidden) continue;
        std::string left = "  ";
        if (s.short_name != 0) left += std::string("-") + s.short_name + ", ";
        left += "--" + s.name + (s.takes_value ? " VALUE" : "");
        if (left.size() < 30) left.resize(30, ' ');
        out += left + " " + s.help + "\n";
    }
    return out;
}

}  // namespace halo::cli
