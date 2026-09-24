#pragma once
// Runtime contract shared by the scheduler, API server and CLI (docs/dev/ARCHITECTURE.md).
// The implementation (qwen35 forward, KV + GDN state, MTP) lives in src/runtime and
// src/models; consumers depend only on this header plus sampling.h.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "halo/sampling/sampling.h"

namespace halo::tokenizer { class Tokenizer; }
namespace halo::chat { class ChatTemplate; }

namespace halo::runtime {

struct EngineConfig {
    std::string model_path;                 // GGUF (trunk; may embed MTP)
    std::optional<std::string> mtp_path;    // separate MTP GGUF (D-006)
    std::string backend = "cpu";            // cpu | vulkan | hip | auto
    std::size_t max_context = 32768;
    std::size_t max_sequences = 4;
    int threads = 0;                        // 0 = auto
    bool mtp_enabled = true;                // profit-gated at runtime (FR-009)
    int mtp_max_draft = 2;
    bool prefix_cache = true;
    std::optional<std::uint64_t> max_memory_bytes;
};

struct ModelInfo {
    std::string id;                          // served model name
    std::string architecture;                // "qwen35"
    std::size_t context_length = 0;
    std::size_t vocab_size = 0;
    bool has_mtp = false;
};

// Per-token callback during generation. Return false to cancel.
struct TokenEvent {
    std::int32_t token = -1;
    std::string piece;                       // complete UTF-8 only (StreamDecoder)
    bool is_eos = false;
};
using TokenCallback = std::function<bool(const TokenEvent&)>;

enum class FinishReason { Stop, Length, Cancelled, Error };

struct GenerateRequest {
    std::vector<std::int32_t> prompt;        // already templated + tokenized
    SamplingParams sampling;
    std::size_t max_tokens = 256;
    std::vector<std::vector<std::int32_t>> stop_token_seqs;  // besides EOS
    std::vector<std::string> stop_strings;
};

struct GenerateResult {
    std::vector<std::int32_t> tokens;
    FinishReason finish = FinishReason::Stop;
    std::size_t prompt_tokens = 0;
    std::size_t cached_prompt_tokens = 0;    // prefix-cache hit length
    double ttft_ms = 0.0;
    double decode_tps = 0.0;
    std::size_t draft_tokens = 0;
    std::size_t accepted_draft_tokens = 0;
    std::string error;
};

struct EngineStats {
    std::size_t active_sequences = 0;
    std::size_t queued_requests = 0;
    std::uint64_t tokens_generated = 0;
    std::uint64_t prefix_cache_hits = 0;
    std::uint64_t prefix_cache_misses = 0;
    std::uint64_t speculative_attempts = 0;
    std::uint64_t speculative_accepts = 0;
};

// Thread-safe. generate() may be called concurrently from several API threads; the
// implementation schedules them (batched decode, prefix cache) internally.
class Engine {
public:
    virtual ~Engine() = default;
    virtual const ModelInfo& model() const = 0;
    virtual const tokenizer::Tokenizer& tokenizer() const = 0;
    virtual const chat::ChatTemplate& chat_template() const = 0;
    virtual GenerateResult generate(const GenerateRequest&, const TokenCallback&) = 0;
    virtual EngineStats stats() const = 0;
};

std::unique_ptr<Engine> create_engine(const EngineConfig&);

}  // namespace halo::runtime
