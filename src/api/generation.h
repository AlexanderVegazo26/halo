#pragma once
// Generation driver shared by every API route: runs Engine::generate, splits output into
// reasoning / content / tool calls (OutputParser), applies stop sequences, enforces the
// reasoning budget (TRD §25), and reports cancellation (client gone, shutdown).
//
// Reasoning budget: when thinking is on, reasoning may use `reasoning_budget` tokens. When
// it is reached while the model is still inside <think>, the driver cancels the engine
// call, appends "\n</think>\n\n" to the context (prompt + every token observed so far) and
// continues in a second Engine::generate call with the remaining max_tokens, so the answer
// always gets headroom. Reasoning and output tokens are counted separately.
//
// Stop sequences are owned here, not by the engine: they are matched on *content* text
// (not reasoning), excluded from the output, and end generation by cancelling the engine
// call. GenerateRequest::stop_strings is left empty.
//
// Limits (security review A-3, A-11): an optional wall-clock deadline ends generation like
// max_tokens (StopCause::Deadline); with parse_output, output whose running count of
// unmatched '[' / '{' exceeds `max_output_nesting` stops generation with an error *before*
// the offending piece reaches the OutputParser, whose tool-argument JSON handling recurses
// (S-9).
//
// Threading: the TokenCallback runs on whatever thread the Engine invokes it on, and the
// EventSink (an SSE writer) may block in it for up to the server's write timeout when the
// client reads slowly. That is harmless when the Engine calls each request's callback on the
// request's own thread; an Engine that invokes callbacks from a shared decode thread would be
// stalled by one slow reader (assumption stated for WS-G).

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "halo/runtime/engine.h"
#include "halo/template/output_parser.h"
#include "json_util.h"

namespace halo::api {

/// Receives output as it is produced. Every method returns false when the client is gone.
class EventSink {
public:
    virtual ~EventSink() = default;
    virtual bool reasoning(std::string_view text) = 0;
    virtual bool content(std::string_view text) = 0;
    virtual bool tool_call(std::size_t index, const chat::ToolCall& call) = 0;
    /// Polled once per token; false = client disconnected.
    virtual bool alive() = 0;
};

enum class StopCause {
    Eos,           // model ended its turn
    Length,        // max_tokens reached
    StopSequence,  // a client stop sequence matched
    Client,        // client disconnected
    Shutdown,      // server stopping
    Deadline,      // wall-clock limit reached (reported like Length)
    Error,         // engine failure or an output limit
};

struct GenerationSpec {
    runtime::GenerateRequest request;  // prompt, sampling, max_tokens
    bool parse_output = true;          // false: raw text (/v1/completions)
    chat::ParserOptions parser;
    bool thinking = false;             // prompt opened a think block / thinking allowed
    std::optional<std::size_t> reasoning_budget;
    std::vector<std::string> stop;
    std::vector<std::int32_t> close_reasoning_tokens;  // tokenization of "\n</think>\n\n"
    std::optional<std::int32_t> think_start, think_end;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    std::size_t max_output_nesting = 256;  // 0 = unchecked
    /// Model context (tokens); 0 = unknown. The reasoning-budget continuation is skipped
    /// when prompt + observed + close would not fit (review N-3).
    std::size_t context_length = 0;
};

struct GenerationOutcome {
    chat::ParsedMessage message;  // content = exactly what was emitted (stop-truncated)
    StopCause cause = StopCause::Eos;
    std::optional<std::string> stop_sequence;
    std::size_t prompt_tokens = 0;
    std::size_t cached_prompt_tokens = 0;
    std::size_t completion_tokens = 0;  // every token the engine produced (both calls)
    std::size_t reasoning_tokens = 0;   // of which inside the think block
    bool reasoning_closed_early = false;
    std::size_t engine_calls = 0;
    std::vector<std::string> warnings;
    double ttft_ms = 0.0;               // measured by the API (includes engine queueing)
    double decode_tps = 0.0;
    std::size_t draft_tokens = 0, accepted_draft_tokens = 0;
    std::string error;         // internal detail (logs only)
    std::string public_error;  // safe to show the client (set with cause == Error)
    /// Set when the engine rejected the request itself before producing any token
    /// (halo::Error Api / Unsupported from generate(), e.g. an invalid or unsupported JSON
    /// schema): the client's fault, reported as 400 in the calling API's format (review R-4).
    std::optional<ApiErrorInfo> client_error;
};

/// Runs a generation. Never throws for engine-side failures: they are reported as
/// StopCause::Error with `error` set (the engine may also throw halo::Error; that is caught
/// and reported the same way).
[[nodiscard]] GenerationOutcome run_generation(runtime::Engine& engine, const GenerationSpec& spec, EventSink& sink,
                                               const std::atomic<bool>& stopping);

/// Incremental stop-sequence matcher with hold-back: text that could still become the start
/// of a stop sequence is withheld until disambiguated. UTF-8 characters are never split.
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops);
    /// Returns the text that is safe to emit now. After a match, matched() is set and
    /// further pushes return "".
    [[nodiscard]] std::string push(std::string_view text);
    /// Emits the held-back tail (end of stream without a match).
    [[nodiscard]] std::string flush();
    [[nodiscard]] const std::optional<std::string>& matched() const noexcept { return matched_; }

private:
    std::vector<std::string> stops_;
    std::size_t max_len_ = 0;
    std::string pending_;
    std::optional<std::string> matched_;
};

}  // namespace halo::api
