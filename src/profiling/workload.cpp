#include "halo/profiling/workload.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <format>
#include <optional>

#include "halo/core/error.h"

namespace halo::profiling {

namespace {

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// High 64 bits of a 64x64 product (portable; no __int128 under -Wpedantic).
std::uint64_t mulhi64(std::uint64_t a, std::uint64_t b) noexcept {
    const std::uint64_t a_lo = a & 0xffffffffULL, a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xffffffffULL, b_hi = b >> 32;
    const std::uint64_t lo_lo = a_lo * b_lo;
    const std::uint64_t hi_lo = a_hi * b_lo;
    const std::uint64_t lo_hi = a_lo * b_hi;
    const std::uint64_t hi_hi = a_hi * b_hi;
    const std::uint64_t cross = (lo_lo >> 32) + (hi_lo & 0xffffffffULL) + lo_hi;
    return hi_hi + (hi_lo >> 32) + (cross >> 32);
}

std::uint64_t fnv1a(std::string_view s) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 0x100000001b3ULL;
    }
    return h;
}

/// Non-negative JSON integer (parsed text yields number_unsigned, values built in C++ from
/// an int yield number_integer; both are accepted, negatives and floats are not).
std::optional<std::uint64_t> as_uint(const nlohmann::json& j) {
    if (j.is_number_unsigned()) return j.get<std::uint64_t>();
    if (j.is_number_integer() && j.get<std::int64_t>() >= 0) return static_cast<std::uint64_t>(j.get<std::int64_t>());
    return std::nullopt;
}

WorkloadMessage parse_message(const nlohmann::json& j, const std::string& where) {
    HALO_CHECK(j.is_object(), ErrorCode::Config, "{}: message is not an object", where);
    WorkloadMessage m;
    m.role = j.value("role", "");
    HALO_CHECK(m.role == "system" || m.role == "user" || m.role == "assistant" || m.role == "tool",
               ErrorCode::Config, "{}: role '{}' is not system/user/assistant/tool", where, m.role);
    const bool has_text = j.contains("text");
    const bool has_syn = j.contains("synthetic_tokens");
    HALO_CHECK(has_text != has_syn, ErrorCode::Config, "{}: exactly one of text / synthetic_tokens is required",
               where);
    if (has_text) {
        HALO_CHECK(j.at("text").is_string(), ErrorCode::Config, "{}: text is not a string", where);
        m.text = j.at("text").get<std::string>();
    } else {
        const auto n = as_uint(j.at("synthetic_tokens"));
        HALO_CHECK(n && *n > 0 && *n <= kMaxSyntheticTokens, ErrorCode::Config,
                   "{}: synthetic_tokens must be an integer in [1, {}]", where, kMaxSyntheticTokens);
        m.synthetic_tokens = *n;
        if (j.contains("seed")) {
            const auto seed = as_uint(j.at("seed"));
            HALO_CHECK(seed.has_value(), ErrorCode::Config, "{}: seed must be a non-negative integer", where);
            m.seed = *seed;
        }
    }
    return m;
}

std::vector<WorkloadMessage> parse_messages(const nlohmann::json& j, const std::string& where) {
    std::vector<WorkloadMessage> out;
    if (j.is_null()) return out;
    HALO_CHECK(j.is_array(), ErrorCode::Config, "{}: not an array", where);
    HALO_CHECK(j.size() <= kMaxTurns, ErrorCode::Config, "{}: {} messages exceed {}", where, j.size(), kMaxTurns);
    for (std::size_t i = 0; i < j.size(); ++i) out.push_back(parse_message(j[i], std::format("{}[{}]", where, i)));
    return out;
}

}  // namespace

Workload parse_workload(const nlohmann::json& j) {
    try {
        HALO_CHECK(j.is_object(), ErrorCode::Config, "workload: not a JSON object");
        const std::string schema = j.value("schema", "");
        HALO_CHECK(schema == kWorkloadSchema, ErrorCode::Config, "workload: schema '{}' is not '{}'", schema,
                   kWorkloadSchema);
        Workload w;
        w.id = j.value("id", "");
        HALO_CHECK(!w.id.empty(), ErrorCode::Config, "workload: missing id");
        w.kind = j.value("kind", "");
        HALO_CHECK(w.kind == "agent_replay" || w.kind == "long_context" || w.kind == "concurrency",
                   ErrorCode::Config, "workload {}: kind '{}' is not agent_replay/long_context/concurrency", w.id,
                   w.kind);
        w.description = j.value("description", "");
        w.provenance = j.value("provenance", "");
        HALO_CHECK(!w.provenance.empty(), ErrorCode::Config, "workload {}: provenance is required", w.id);
        if (j.contains("tools")) {
            HALO_CHECK(j.at("tools").is_array(), ErrorCode::Config, "workload {}: tools is not an array", w.id);
            w.tools = j.at("tools");
        }
        const auto& agents = j.at("agents");
        HALO_CHECK(agents.is_array() && !agents.empty() && agents.size() <= kMaxAgents, ErrorCode::Config,
                   "workload {}: agents must be an array of 1..{}", w.id, kMaxAgents);
        for (std::size_t a = 0; a < agents.size(); ++a) {
            const auto& aj = agents[a];
            const std::string where = std::format("workload {}: agents[{}]", w.id, a);
            HALO_CHECK(aj.is_object(), ErrorCode::Config, "{}: not an object", where);
            AgentScript s;
            s.id = aj.value("id", std::format("agent{}", a));
            s.preamble = parse_messages(aj.value("preamble", nlohmann::json()), where + ".preamble");
            s.turns = parse_messages(aj.value("turns", nlohmann::json()), where + ".turns");
            HALO_CHECK(!s.turns.empty(), ErrorCode::Config, "{}: at least one turn is required", where);
            const auto mt = aj.contains("max_tokens") ? as_uint(aj.at("max_tokens")) : std::optional<std::uint64_t>(128);
            HALO_CHECK(mt && *mt >= 1 && *mt <= 65536, ErrorCode::Config, "{}: max_tokens must be in [1, 65536]",
                       where);
            s.max_tokens = static_cast<std::size_t>(*mt);
            const auto tt =
                aj.contains("think_time_ms") ? as_uint(aj.at("think_time_ms")) : std::optional<std::uint64_t>(0);
            HALO_CHECK(tt && *tt <= 600000, ErrorCode::Config, "{}: think_time_ms must be in [0, 600000]", where);
            s.think_time_ms = *tt;
            w.agents.push_back(std::move(s));
        }
        return w;
    } catch (const nlohmann::json::exception& e) {
        throw_error(ErrorCode::Config, "malformed workload: {}", e.what());
    }
}

Workload load_workload(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    HALO_CHECK(!ec, ErrorCode::Io, "workload {}: {}", path.string(), ec.message());
    HALO_CHECK(size <= kMaxWorkloadBytes, ErrorCode::Config, "workload {}: {} bytes exceed {}", path.string(), size,
               kMaxWorkloadBytes);
    // open(O_CLOEXEC) + bounded read (S-33: no descriptor without close-on-exec).
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    HALO_CHECK(fd >= 0, ErrorCode::Io, "workload {}: cannot open: {}", path.string(), std::strerror(errno));
    struct Closer {
        int fd;
        ~Closer() { ::close(fd); }
    } closer{fd};
    std::string text;
    std::array<char, 65536> buf{};
    while (true) {
        const ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n < 0 && errno == EINTR) continue;
        HALO_CHECK(n >= 0, ErrorCode::Io, "workload {}: read error: {}", path.string(), std::strerror(errno));
        if (n == 0) break;
        text.append(buf.data(), static_cast<std::size_t>(n));
        HALO_CHECK(text.size() <= kMaxWorkloadBytes, ErrorCode::Config, "workload {}: exceeds {} bytes", path.string(),
                   kMaxWorkloadBytes);
    }
    const auto j = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    HALO_CHECK(!j.is_discarded(), ErrorCode::Config, "workload {}: invalid JSON", path.string());
    return parse_workload(j);
}

std::vector<std::int32_t> synthetic_tokens(std::uint64_t seed, std::size_t n, std::int32_t id_range,
                                           std::int32_t first_id) {
    HALO_CHECK(id_range > 0, ErrorCode::Config, "synthetic_tokens: id_range must be > 0");
    HALO_CHECK(first_id >= 0, ErrorCode::Config, "synthetic_tokens: first_id must be >= 0");
    HALO_CHECK(n <= kMaxSyntheticTokens, ErrorCode::Config, "synthetic_tokens: {} > {}", n, kMaxSyntheticTokens);
    std::vector<std::int32_t> out(n);
    std::uint64_t state = seed;
    const auto range = static_cast<std::uint64_t>(id_range);
    for (auto& t : out) {
        t = first_id + static_cast<std::int32_t>(mulhi64(splitmix64(state), range));
    }
    return out;
}

WorkloadEncoder synthetic_encoder(std::int32_t vocab_size) {
    HALO_CHECK(vocab_size > 0, ErrorCode::Config, "synthetic_encoder: vocab_size must be > 0");
    return [vocab_size](const Workload&, const AgentScript& agent, std::size_t turn) {
        HALO_CHECK(turn < agent.turns.size(), ErrorCode::Config, "synthetic_encoder: turn {} >= {}", turn,
                   agent.turns.size());
        std::vector<std::int32_t> out;
        const auto append = [&](const WorkloadMessage& m) {
            std::vector<std::int32_t> t;
            if (m.synthetic_tokens) {
                t = synthetic_tokens(m.seed, *m.synthetic_tokens, vocab_size);
            } else {
                const std::size_t n = (m.text->size() + 3) / 4;
                t = synthetic_tokens(fnv1a(*m.text), n, vocab_size);
            }
            out.insert(out.end(), t.begin(), t.end());
        };
        for (const auto& m : agent.preamble) append(m);
        for (std::size_t i = 0; i <= turn; ++i) append(agent.turns[i]);
        return out;
    };
}

}  // namespace halo::profiling
