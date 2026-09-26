#pragma once
// Configuration resolution for `halo` (TRD §44): command line > environment (HALO_*) >
// config file > built-in default. The config file is JSON (YAML is not supported in v0.2),
// named by --config or HALO_CONFIG, with sections mirroring PRD §15:
//
//   {"model":   {"path": "...", "mtp": "..."},
//    "runtime": {"backend": "auto", "threads": 0, "ctx": 32768, "parallel": 4,
//                "mtp_draft": 2, "prefix_cache": true},
//    "server":  {"host": "127.0.0.1", "port": 8080, "api_key": "...", "cors_origins": [],
//                "allowed_hosts": [], "served_model_name": "", "max_concurrent": 4,
//                "max_queue": 16, "max_body_bytes": 8388608, "max_tokens_cap": 32768,
//                "default_max_tokens": 8192, "request_timeout_s": 600,
//                "allow_unauthenticated_remote": false}}
//
// Unknown sections/keys and wrongly typed values are errors (Error(Config)), as are
// unparsable environment values; a typo never silently falls back to a default. The API key
// is never printed (the resolved-config dump shows "<redacted>").

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "args.h"
#include "halo/api/server.h"
#include "halo/runtime/engine.h"

namespace halo::cli {

enum class KeyType { String, U64, Bool, List };

struct ConfigKey {
    std::string key;   ///< "section.name"
    std::string flag;  ///< CLI long option
    std::string env;   ///< environment variable
    KeyType type = KeyType::String;
    nlohmann::json default_value;  ///< null = unset
    std::uint64_t lo = 0, hi = 0;  ///< U64 range
    bool secret = false;
    std::string help;
};

[[nodiscard]] const std::vector<ConfigKey>& config_keys();

/// OptionSpecs for the given keys (plus --config).
[[nodiscard]] std::vector<OptionSpec> config_options(const std::vector<std::string>& keys);

class ResolvedConfig {
public:
    struct Entry {
        nlohmann::json value;  ///< null = unset
        std::string source;    ///< "cli", "env:HALO_X", "file:<path>", "default"
    };

    [[nodiscard]] std::optional<std::string> str(const std::string& key) const;
    [[nodiscard]] std::uint64_t u64(const std::string& key) const;  ///< requires a value
    [[nodiscard]] std::optional<std::uint64_t> opt_u64(const std::string& key) const;
    [[nodiscard]] bool boolean(const std::string& key) const;
    [[nodiscard]] std::vector<std::string> list(const std::string& key) const;
    [[nodiscard]] const std::string& source(const std::string& key) const;
    [[nodiscard]] bool has(const std::string& key) const;

    /// {"key": {"value": ..., "source": ...}}, secrets redacted.
    [[nodiscard]] nlohmann::json to_json() const;

    std::map<std::string, Entry> entries;
    std::optional<std::string> config_file;
};

/// Resolves `keys` from args (the command's parsed options), `env` and the config file.
[[nodiscard]] ResolvedConfig resolve_config(const ParsedArgs& args, const std::map<std::string, std::string>& env,
                                            const std::vector<std::string>& keys);

/// `default_profile_db` is used when a power mode is set and no profile DB is given.
[[nodiscard]] runtime::EngineConfig engine_config(const ResolvedConfig& c,
                                                  const std::filesystem::path& default_profile_db = {});
[[nodiscard]] api::ServerConfig server_config(const ResolvedConfig& c);

/// Keys used by serve / run / bench model.
[[nodiscard]] const std::vector<std::string>& runtime_keys();
[[nodiscard]] const std::vector<std::string>& server_keys();

}  // namespace halo::cli
