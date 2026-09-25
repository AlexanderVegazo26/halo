#pragma once
// Minimal, strict argument parser for the halo subcommands: every option must be declared;
// "--name value", "--name=value" and declared short aliases ("-p value") are accepted;
// unknown options, missing values and repeated single-valued options are usage errors.

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace halo::cli {

/// A usage error: printed with the command's usage text, exit code 2.
class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct OptionSpec {
    std::string name;         ///< long name without "--"
    bool takes_value = true;  ///< false = boolean flag
    char short_name = 0;      ///< e.g. 'p' for -p
    bool repeatable = false;  ///< value options only: collected in order
    bool hidden = false;      ///< not shown in usage (e.g. --root)
    std::string help;
};

class ParsedArgs {
public:
    std::vector<std::string> positionals;

    [[nodiscard]] bool flag(const std::string& name) const { return flags_.contains(name); }
    [[nodiscard]] bool has(const std::string& name) const { return values_.contains(name); }
    [[nodiscard]] std::optional<std::string> value(const std::string& name) const {
        const auto it = values_.find(name);
        if (it == values_.end() || it->second.empty()) return std::nullopt;
        return it->second.back();
    }
    [[nodiscard]] std::vector<std::string> values(const std::string& name) const {
        const auto it = values_.find(name);
        return it == values_.end() ? std::vector<std::string>{} : it->second;
    }
    /// The flag's value parsed as an unsigned integer in [lo, hi]; UsageError otherwise.
    [[nodiscard]] std::optional<std::uint64_t> u64(const std::string& name, std::uint64_t lo, std::uint64_t hi) const;
    [[nodiscard]] std::optional<double> number(const std::string& name, double lo, double hi) const;

    std::map<std::string, std::vector<std::string>> values_;
    std::map<std::string, bool> flags_;
};

[[nodiscard]] ParsedArgs parse_args(const std::vector<std::string>& args, const std::vector<OptionSpec>& specs);
/// parse_args, with `usage` appended to any UsageError (so --help can show it).
[[nodiscard]] ParsedArgs parse_args(const std::vector<std::string>& args, const std::vector<OptionSpec>& specs,
                                    const std::string& usage);

/// "a,b,c" -> {"a","b","c"} (empty items rejected).
[[nodiscard]] std::vector<std::string> split_list(const std::string& s, const std::string& what);
[[nodiscard]] std::uint64_t parse_u64(const std::string& s, const std::string& what, std::uint64_t lo, std::uint64_t hi);
[[nodiscard]] double parse_number(const std::string& s, const std::string& what, double lo, double hi);

/// Usage lines for `specs` ("  --name VALUE   help").
[[nodiscard]] std::string options_help(const std::vector<OptionSpec>& specs);

}  // namespace halo::cli
