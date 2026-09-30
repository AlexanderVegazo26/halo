#pragma once
// Runtime contract shared by the scheduler, API server and CLI (docs/dev/ARCHITECTURE.md).
// The implementation (qwen35 forward, KV + GDN state, MTP) lives in src/runtime and
// src/models; consumers depend only on this header plus sampling.h.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
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
    /// True when the caller asked for max_context (--ctx / HALO_CTX / config file), false for
    /// the built-in default. An explicit value that cannot be honoured (larger than the model's
    /// trained context, or than the KV pool the backend / free VRAM allows) is Error(Config) at
    /// creation; a defaulted one is reduced with a warning.
    bool max_context_explicit = false;
    std::size_t max_sequences = 4;
    int threads = 0;                        // 0 = auto
    bool mtp_enabled = true;                // profit-gated at runtime (FR-009)
    int mtp_max_draft = 2;
    bool prefix_cache = true;
    std::optional<std::uint64_t> max_memory_bytes;
    // Additive (WS-G M4, TRD §64): consume an autotune profile database at creation.
    /// Path of the profile DB (opened read-only; a missing file = no profiles). Absent = off.
    std::optional<std::string> profile_db;
    /// Platform (BIOS/EC) power-mode label of the profile key; required with profile_db.
    std::string platform_power_mode;
    /// ISA label of the profile key; default = the CPU ISA of this process.
    std::optional<std::string> isa_target;
};

struct ModelInfo {
    std::string id;                          // served model name
    std::string architecture;                // "qwen35"
    std::size_t context_length = 0;
    std::size_t vocab_size = 0;
    bool has_mtp = false;
    std::uint64_t kv_pool_bytes = 0;  // trunk + MTP KV pools actually allocated (0 = not reported)
    std::size_t kv_segments = 0;      // backend buffers of the trunk KV pool (1 = contiguous)
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
    /// Additive (WS-G M5, review R-3): cancellation that needs no token.
    ///
    /// The engine checks both fields at every scheduler tick for this request, whatever its
    /// state: QUEUED (waiting for a sequence slot), PREFILLING or DECODING. The worst-case
    /// latency from the event to the request ending is one tick (one batched forward, i.e.
    /// at most one prefill chunk plus one decode step), plus the time to drain already
    /// produced tokens.
    ///   * `cancel.stop_requested()`             -> FinishReason::Cancelled
    ///   * `steady_clock::now() >= *deadline`     -> FinishReason::Length with
    ///                                              GenerateResult::deadline_expired = true
    /// If both hold at the same check, Cancelled wins. generate() also checks both before
    /// every callback, so tokens the worker produced ahead of a slow callback are dropped,
    /// not delivered: once either fires, no further callback is made. generate() then
    /// returns normally (it does not throw); `tokens` holds exactly the tokens already passed
    /// to the callback, which may be none for a queued or prefilling request.
    /// The KV/GDN state of a cancelled sequence stays consistent and is kept for the prefix
    /// cache like any finished sequence.
    ///
    /// `cancel` is copied into the engine; std::stop_source::request_stop() is thread-safe
    /// and may be called from any thread at any time, including before generate() (the
    /// request then ends at its first tick without running a forward). A default-constructed
    /// std::stop_token never requests a stop. Returning false from the TokenCallback remains
    /// an equivalent way to cancel once tokens flow.
    std::stop_token cancel;
    std::optional<std::chrono::steady_clock::time_point> deadline;
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
    /// Additive (WS-G M5): the request ended because GenerateRequest::deadline passed
    /// (finish == Length); distinguishes a wall-clock limit from max_tokens.
    bool deadline_expired = false;
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
    // Additive (WS-G M4): kernel configuration in effect and where it came from (TRD §56).
    std::uint32_t threads = 0;                     ///< worker thread-pool size
    std::uint32_t gdn_chunk = 0;                   ///< chunked-GDN prefill chunk
    std::string tuning;                            ///< e.g. "MATMUL=Exact(threads=2); GATED_DELTANET=Exact(chunk=32)"
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
