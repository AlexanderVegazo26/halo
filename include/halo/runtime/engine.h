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
    /// Additive (WS-G M3, D-013): prompt token offsets where a GDN prefix checkpoint is
    /// worth taking (e.g. the template's preamble end and user-message starts converted to
    /// token positions). Optional; the engine also checkpoints at prompt end - k and at a
    /// fixed spacing. Out-of-range or too-closely-spaced hints are ignored.
    std::vector<std::size_t> checkpoint_hints;
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
    // Additive (WS-G M3): scheduler and cost-model observability.
    std::uint64_t ticks = 0;                       ///< batched forward ticks run
    std::uint64_t prefix_cache_reused_tokens = 0;  ///< prompt tokens restored instead of recomputed
    std::uint64_t mtp_fallbacks = 0;               ///< ticks retried with k = 0 after MTP-KV exhaustion
    std::uint32_t last_tick_weight_passes = 0;     ///< trunk + MTP weight passes of the last tick
    std::uint64_t last_tick_predicted_bytes = 0;   ///< cost-model weight bytes of the last tick (D-011)
};

// Thread-safe. generate() may be called concurrently from several API threads; the
// implementation schedules them (batched decode, prefix cache) internally.
//
// Callback threading guarantee (WS-G M3, relied on by src/api):
//   * The TokenCallback is invoked ONLY on the thread that called generate(), never on an
//     engine-internal thread. The engine's worker produces tokens into a per-request queue;
//     generate() drains that queue and calls the callback. A slow callback (e.g. an SSE
//     write to a slow client) therefore delays only its own request's delivery — the
//     batch keeps decoding, and that request's tokens buffer (at most max_tokens of them).
//   * Calls for one request are sequential and in token order.
//   * Returning false (or throwing) cancels the request: no further callbacks are made,
//     the worker retires the sequence at its next tick, and generate() returns with
//     FinishReason::Cancelled and `tokens` = exactly the tokens passed to the callback
//     (an exception from the callback is rethrown after the sequence is retired).
//   * EOS is delivered as a TokenEvent with is_eos = true and an empty piece, and is part
//     of `tokens` (FinishReason::Stop).
//   * A stop string ends generation after the token whose decoded text completes it; that
//     token's piece is delivered unchanged (callers that must hide the stop string strip it).
// Invalid requests (empty prompt, token ids outside the vocabulary, prompt >= context,
// max_tokens == 0, invalid SamplingParams / JSON schema) throw halo::Error(Api) from
// generate() before anything is scheduled (Error(Unsupported) for a JSON-schema construct
// the grammar compiler does not support). Failures during generation are reported as
// FinishReason::Error with `error` set.
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
