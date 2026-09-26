#include "config.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "halo/core/error.h"

namespace halo::cli {

using nlohmann::json;

namespace {

constexpr std::uintmax_t kMaxConfigBytes = 1u << 20;

std::vector<ConfigKey> make_keys() {
    const auto s = [](std::string k, std::string f, std::string e, json d, std::string h, bool secret = false) {
        ConfigKey c;
        c.key = std::move(k);
        c.flag = std::move(f);
        c.env = std::move(e);
        c.type = KeyType::String;
        c.default_value = std::move(d);
        c.secret = secret;
        c.help = std::move(h);
        return c;
    };
    const auto u = [](std::string k, std::string f, std::string e, json d, std::uint64_t lo, std::uint64_t hi,
                      std::string h) {
        ConfigKey c;
        c.key = std::move(k);
        c.flag = std::move(f);
        c.env = std::move(e);
        c.type = KeyType::U64;
        c.default_value = std::move(d);
        c.lo = lo;
        c.hi = hi;
        c.help = std::move(h);
        return c;
    };
    const auto b = [](std::string k, std::string f, std::string e, bool d, std::string h) {
        ConfigKey c;
        c.key = std::move(k);
        c.flag = std::move(f);
        c.env = std::move(e);
        c.type = KeyType::Bool;
        c.default_value = d;
        c.help = std::move(h);
        return c;
    };
    const auto l = [](std::string k, std::string f, std::string e, std::string h) {
        ConfigKey c;
        c.key = std::move(k);
        c.flag = std::move(f);
        c.env = std::move(e);
        c.type = KeyType::List;
        c.default_value = json::array();
        c.help = std::move(h);
        return c;
    };
    const api::ServerConfig sd;
    return {
        s("model.path", "model", "HALO_MODEL", nullptr, "trunk GGUF"),
        s("model.mtp", "mtp", "HALO_MTP", nullptr, "separate MTP GGUF (D-006)"),
        s("runtime.backend", "backend", "HALO_BACKEND", "auto", "cpu | vulkan | hip | auto"),
        u("runtime.threads", "threads", "HALO_THREADS", 0, 0, 1024, "CPU threads (0 = auto)"),
        u("runtime.ctx", "ctx", "HALO_CTX", 32768, 1, 1u << 20, "maximum context per sequence"),
        u("runtime.parallel", "parallel", "HALO_PARALLEL", 4, 1, 256, "concurrent sequences"),
        u("runtime.mtp_draft", "mtp-draft", "HALO_MTP_DRAFT", 2, 0, 16, "MTP draft tokens (0 = MTP off)"),
        b("runtime.prefix_cache", "no-prefix-cache", "HALO_PREFIX_CACHE", true, "disable the prefix cache"),
        s("runtime.power_mode", "power-mode", "HALO_POWER_MODE", nullptr,
          "platform (BIOS/EC) power mode; with it the engine applies tuned profiles (TRD §64)"),
        s("runtime.profile_db", "profile-db", "HALO_PROFILE_DB", nullptr,
          "autotune profile DB to apply (default with --power-mode: halo tune's default DB)"),
        s("runtime.isa_target", "isa-target", "HALO_ISA_TARGET", nullptr,
          "ISA label of the profile key (default: CPU ISA of this process; must match halo tune)"),
        s("server.host", "host", "HALO_HOST", sd.host, "bind address"),
        u("server.port", "port", "HALO_PORT", sd.port, 0, 65535, "TCP port (0 = ephemeral)"),
        s("server.api_key", "api-key", "HALO_API_KEY", nullptr,
          "require this key (Bearer / x-api-key); prefer HALO_API_KEY (argv is visible to other users)", true),
        l("server.cors_origins", "cors-origins", "HALO_CORS_ORIGINS", "comma-separated allowed browser origins"),
        l("server.allowed_hosts", "allowed-hosts", "HALO_ALLOWED_HOSTS", "extra accepted Host header names"),
        s("server.served_model_name", "served-model-name", "HALO_SERVED_MODEL_NAME", nullptr,
          "model id reported by the API"),
        u("server.max_concurrent", "max-concurrent", "HALO_MAX_CONCURRENT", nullptr, 1, 1024,
          "generations at once (default: --parallel)"),
        u("server.max_queue", "max-queue", "HALO_MAX_QUEUE", sd.max_queue, 0, 4096, "queued generations"),
        u("server.utility_concurrency", "utility-concurrency", "HALO_UTILITY_CONCURRENCY", sd.utility_concurrency, 1,
          256, "/tokenize and /apply-template running at once"),
        u("server.utility_queue", "utility-queue", "HALO_UTILITY_QUEUE", sd.utility_queue, 0, 4096,
          "/tokenize and /apply-template waiting"),
        u("server.header_timeout_s", "header-timeout", "HALO_HEADER_TIMEOUT", sd.header_timeout.count() / 1000, 0, 3600,
          "seconds to receive a request's headers (0 = unlimited)"),
        u("server.body_timeout_s", "body-timeout", "HALO_BODY_TIMEOUT", sd.body_timeout.count() / 1000, 0, 3600,
          "seconds to receive a request's body (0 = unlimited)"),
        b("server.completions_parse_special", "no-completions-parse-special", "HALO_COMPLETIONS_PARSE_SPECIAL",
          sd.completions_parse_special, "treat special-token text in /v1/completions prompts as plain text"),
        u("server.max_body_bytes", "max-body-bytes", "HALO_MAX_BODY_BYTES", sd.max_body_bytes, 1024, 1ULL << 30,
          "request body limit"),
        u("server.max_tokens_cap", "max-tokens-cap", "HALO_MAX_TOKENS_CAP", sd.max_tokens_cap, 1, 1u << 20,
          "per-request max_tokens limit"),
        u("server.default_max_tokens", "default-max-tokens", "HALO_DEFAULT_MAX_TOKENS", sd.default_max_tokens, 1,
          1u << 20, "max_tokens when a request gives none"),
        u("server.request_timeout_s", "request-timeout", "HALO_REQUEST_TIMEOUT", sd.request_timeout.count(), 0,
          86400, "wall-clock limit per generation, seconds (0 = none)"),
        b("server.allow_remote", "allow-remote", "HALO_ALLOW_REMOTE", false,
          "allow a non-loopback --host (D-017: only behind a reverse proxy with connection and header timeouts)"),
        b("server.allow_unauthenticated_remote", "allow-unauthenticated-remote", "HALO_ALLOW_UNAUTHENTICATED_REMOTE",
          false, "allow a non-loopback bind without an API key"),
    };
}

const ConfigKey& key_spec(const std::string& key) {
    for (const auto& k : config_keys()) {
        if (k.key == key) return k;
    }
    throw_error(ErrorCode::Config, "internal: unknown config key {}", key);
}

bool parse_bool(const std::string& v, const std::string& what) {
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw_error(ErrorCode::Config, "{} must be true/false (got '{}')", what, v);
}

/// Value from a string. A bad command-line value is a UsageError (exit 2); a bad
/// environment value is Error(Config).
json from_text(const ConfigKey& k, const std::string& v, const std::string& what, bool from_cli) {
    try {
        switch (k.type) {
            case KeyType::String: return v;
            case KeyType::U64: return parse_u64(v, what, k.lo, k.hi);
            case KeyType::Bool: return parse_bool(v, what);
            case KeyType::List: return v.empty() ? json::array() : json(split_list(v, what));
        }
    } catch (const UsageError& e) {
        if (from_cli) throw;
        throw_error(ErrorCode::Config, "{}", e.what());
    }
    return nullptr;
}

/// Value from the config file (type-checked).
json from_file(const ConfigKey& k, const json& v, const std::string& what) {
    switch (k.type) {
        case KeyType::String:
            HALO_CHECK(v.is_string(), ErrorCode::Config, "{} must be a string", what);
            return v;
        case KeyType::U64: {
            HALO_CHECK(v.is_number_unsigned() || (v.is_number_integer() && v.get<std::int64_t>() >= 0), ErrorCode::Config,
                       "{} must be a non-negative integer", what);
            const auto x = v.get<std::uint64_t>();
            HALO_CHECK(x >= k.lo && x <= k.hi, ErrorCode::Config, "{} must be in [{}, {}] (got {})", what, k.lo, k.hi, x);
            return x;
        }
        case KeyType::Bool:
            HALO_CHECK(v.is_boolean(), ErrorCode::Config, "{} must be a boolean", what);
            return v;
        case KeyType::List:
            HALO_CHECK(v.is_array(), ErrorCode::Config, "{} must be an array of strings", what);
            for (const auto& e : v) HALO_CHECK(e.is_string() && !e.get<std::string>().empty(), ErrorCode::Config,
                                               "{} must be an array of non-empty strings", what);
            return v;
    }
    return nullptr;
}

json read_config_file(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    HALO_CHECK(!ec, ErrorCode::Config, "cannot read config file {}: {}", path, ec.message());
    HALO_CHECK(size <= kMaxConfigBytes, ErrorCode::Config, "config file {} is larger than 1 MiB", path);
    std::ifstream f(path, std::ios::binary);
    HALO_CHECK(f.good(), ErrorCode::Config, "cannot open config file {}", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    json j = json::parse(ss.str(), nullptr, /*allow_exceptions=*/false, /*ignore_comments=*/false);
    HALO_CHECK(!j.is_discarded(), ErrorCode::Config, "config file {} is not valid JSON (YAML is not supported in v0.2)",
               path);
    HALO_CHECK(j.is_object(), ErrorCode::Config, "config file {} must hold a JSON object", path);
    // Strict: every section/key must be known.
    for (auto it = j.begin(); it != j.end(); ++it) {
        HALO_CHECK(it.value().is_object(), ErrorCode::Config, "{}: section '{}' must be an object", path, it.key());
        for (auto kt = it.value().begin(); kt != it.value().end(); ++kt) {
            const std::string full = it.key() + "." + kt.key();
            bool known = false;
            for (const auto& k : config_keys()) known = known || k.key == full;
            HALO_CHECK(known, ErrorCode::Config, "{}: unknown setting '{}'", path, full);
        }
    }
    return j;
}

}  // namespace

const std::vector<ConfigKey>& config_keys() {
    static const std::vector<ConfigKey> keys = make_keys();
    return keys;
}

std::vector<OptionSpec> config_options(const std::vector<std::string>& keys) {
    std::vector<OptionSpec> out;
    out.push_back({"config", true, 0, false, false, "JSON config file (also HALO_CONFIG)"});
    for (const auto& key : keys) {
        const auto& k = key_spec(key);
        out.push_back({k.flag, k.type != KeyType::Bool, 0, false, false, k.help + " [" + k.env + "]"});
    }
    return out;
}

ResolvedConfig resolve_config(const ParsedArgs& args, const std::map<std::string, std::string>& env,
                              const std::vector<std::string>& keys) {
    ResolvedConfig rc;
    if (auto c = args.value("config")) {
        rc.config_file = *c;
    } else if (const auto it = env.find("HALO_CONFIG"); it != env.end() && !it->second.empty()) {
        rc.config_file = it->second;
    }
    const json file = rc.config_file ? read_config_file(*rc.config_file) : json::object();
    for (const auto& key : keys) {
        const auto& k = key_spec(key);
        ResolvedConfig::Entry e{k.default_value, "default"};
        const auto dot = k.key.find('.');
        const std::string section = k.key.substr(0, dot), name = k.key.substr(dot + 1);
        if (file.contains(section) && file[section].contains(name)) {
            e = {from_file(k, file[section][name], *rc.config_file + ": " + k.key), "file:" + *rc.config_file};
        }
        // An empty environment value counts as unset: `HALO_API_KEY=` passed through by a
        // service unit must not silently replace a configured key (security review S-18).
        if (const auto it = env.find(k.env); it != env.end() && !it->second.empty()) {
            e = {from_text(k, it->second, k.env, false), "env:" + k.env};
        }
        if (k.type == KeyType::Bool) {
            // Boolean CLI flags state the non-default: --no-prefix-cache / --allow-...
            if (args.flag(k.flag)) e = {k.flag.starts_with("no-") ? json(false) : json(true), "cli"};
        } else if (args.has(k.flag)) {
            const std::string v = args.values(k.flag).back();
            if (k.secret && v.empty()) throw UsageError("--" + k.flag + " must not be empty");
            e = {from_text(k, v, "--" + k.flag, true), "cli"};
        }
        rc.entries[k.key] = std::move(e);
    }
    return rc;
}

bool ResolvedConfig::has(const std::string& key) const {
    const auto it = entries.find(key);
    return it != entries.end() && !it->second.value.is_null();
}

std::optional<std::string> ResolvedConfig::str(const std::string& key) const {
    if (!has(key)) return std::nullopt;
    return entries.at(key).value.get<std::string>();
}

std::uint64_t ResolvedConfig::u64(const std::string& key) const {
    HALO_CHECK(has(key), ErrorCode::Config, "setting {} is required", key);
    return entries.at(key).value.get<std::uint64_t>();
}

std::optional<std::uint64_t> ResolvedConfig::opt_u64(const std::string& key) const {
    if (!has(key)) return std::nullopt;
    return entries.at(key).value.get<std::uint64_t>();
}

bool ResolvedConfig::boolean(const std::string& key) const { return has(key) && entries.at(key).value.get<bool>(); }

std::vector<std::string> ResolvedConfig::list(const std::string& key) const {
    if (!has(key)) return {};
    return entries.at(key).value.get<std::vector<std::string>>();
}

const std::string& ResolvedConfig::source(const std::string& key) const {
    static const std::string none = "unset";
    const auto it = entries.find(key);
    return it == entries.end() ? none : it->second.source;
}

json ResolvedConfig::to_json() const {
    json j = json::object();
    for (const auto& [k, e] : entries) {
        // Truthful about secrets without revealing them (S-18): "<redacted>" only for a
        // non-empty value.
        json shown = e.value;
        if (key_spec(k).secret && e.value.is_string()) {
            shown = e.value.get_ref<const std::string&>().empty() ? json("<empty>") : json("<redacted>");
        }
        j[k] = {{"value", shown}, {"source", e.source}};
    }
    if (config_file) j["config_file"] = *config_file;
    return j;
}

const std::vector<std::string>& runtime_keys() {
    static const std::vector<std::string> k = {"model.path",       "model.mtp",       "runtime.backend",
                                               "runtime.threads",  "runtime.ctx",     "runtime.parallel",
                                               "runtime.mtp_draft", "runtime.prefix_cache", "runtime.power_mode",
                                               "runtime.profile_db", "runtime.isa_target"};
    return k;
}

const std::vector<std::string>& server_keys() {
    static const std::vector<std::string> k = {
        "server.host",           "server.port",          "server.api_key",         "server.cors_origins",
        "server.allowed_hosts",  "server.served_model_name", "server.max_concurrent", "server.max_queue",
        "server.max_body_bytes", "server.max_tokens_cap", "server.default_max_tokens", "server.request_timeout_s",
        "server.allow_unauthenticated_remote", "server.allow_remote", "server.utility_concurrency", "server.utility_queue",
        "server.header_timeout_s", "server.body_timeout_s", "server.completions_parse_special"};
    return k;
}

runtime::EngineConfig engine_config(const ResolvedConfig& c, const std::filesystem::path& default_profile_db) {
    runtime::EngineConfig e;
    const auto model = c.str("model.path");
    HALO_CHECK(model.has_value() && !model->empty(), ErrorCode::Config,
               "no model given (pass <model.gguf>, --model, HALO_MODEL or model.path in the config file)");
    e.model_path = *model;
    e.mtp_path = c.str("model.mtp");
    e.backend = c.str("runtime.backend").value_or("auto");
    HALO_CHECK(e.backend == "auto" || e.backend == "cpu" || e.backend == "vulkan" || e.backend == "hip",
               ErrorCode::Config, "runtime.backend must be auto, cpu, vulkan or hip (got '{}')", e.backend);
    e.threads = static_cast<int>(c.u64("runtime.threads"));
    e.max_context = c.u64("runtime.ctx");
    e.max_sequences = c.u64("runtime.parallel");
    const auto draft = c.u64("runtime.mtp_draft");
    e.mtp_enabled = draft > 0;
    e.mtp_max_draft = static_cast<int>(draft == 0 ? 1 : draft);
    e.prefix_cache = c.boolean("runtime.prefix_cache");
    // TRD §64: the engine applies tuned winners when it knows the profile DB and the power
    // mode (a must-match field of the profile key).
    e.isa_target = c.str("runtime.isa_target");
    if (auto pm = c.str("runtime.power_mode")) {
        e.platform_power_mode = *pm;
        if (auto db = c.str("runtime.profile_db")) {
            e.profile_db = *db;
        } else if (!default_profile_db.empty()) {
            e.profile_db = default_profile_db.string();
        }
    } else {
        HALO_CHECK(!c.has("runtime.profile_db"), ErrorCode::Config,
                   "runtime.profile_db needs runtime.power_mode (--power-mode): the power mode is part of the profile key");
    }
    return e;
}

api::ServerConfig server_config(const ResolvedConfig& c) {
    api::ServerConfig s;
    s.host = c.str("server.host").value_or(s.host);
    s.port = static_cast<int>(c.u64("server.port"));
    if (auto k = c.str("server.api_key")) {
        HALO_CHECK(!k->empty(), ErrorCode::Config, "server.api_key is empty ({}); remove it or set a key",
                   c.source("server.api_key"));
        s.api_key = *k;
    }
    s.cors_origins = c.list("server.cors_origins");
    s.allowed_hosts = c.list("server.allowed_hosts");
    s.served_model_name = c.str("server.served_model_name").value_or("");
    s.max_concurrent = c.opt_u64("server.max_concurrent").value_or(c.has("runtime.parallel") ? c.u64("runtime.parallel") : 4);
    s.max_queue = c.u64("server.max_queue");
    s.max_body_bytes = c.u64("server.max_body_bytes");
    s.max_tokens_cap = c.u64("server.max_tokens_cap");
    s.default_max_tokens = c.u64("server.default_max_tokens");
    s.request_timeout = std::chrono::seconds(c.u64("server.request_timeout_s"));
    s.allow_unauthenticated_remote = c.boolean("server.allow_unauthenticated_remote");
    s.allow_remote = c.boolean("server.allow_remote");
    s.utility_concurrency = c.u64("server.utility_concurrency");
    s.utility_queue = c.u64("server.utility_queue");
    s.header_timeout = std::chrono::seconds(c.u64("server.header_timeout_s"));
    s.body_timeout = std::chrono::seconds(c.u64("server.body_timeout_s"));
    s.completions_parse_special = c.boolean("server.completions_parse_special");
    return s;
}

}  // namespace halo::cli
