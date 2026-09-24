#pragma once
// HALO chat-template engine (TRD §10): renders the model's own Jinja chat template
// (GGUF `tokenizer.chat_template`) with minja, matching HF `apply_chat_template(...,
// tokenize=False)` byte-for-byte on the Qwen3.8 templates (golden-tested).
//
// Differences from calling minja's chat_template class directly: no capability probing and
// no polyfills (HF applies none), a Python-compatible `trim`, `raise_exception` surfaced
// as a typed halo::Error(Api), a capped `range`, and a deterministic `strftime_now`.
//
// Untrusted-template limits (minja 021c229): there is no {% include %}/{% import %}, so a
// template cannot read files; there is no recursion depth, loop-iteration or output-size
// limit inside minja. HALO bounds what it can from outside: template source <= 1 MiB,
// `range()` <= 100000 elements. A template with unbounded macro recursion can still
// exhaust the stack — only templates from model files the operator chose are rendered.
//
// minja semantic gaps handled at the boundary (each is golden- or unit-tested):
//   - no `is undefined` test: the source is rewritten to `is not defined` before parsing;
//   - a null value counts as undefined and `is true/false` test truthiness: render() rejects
//     null template variables and non-boolean enable_thinking / preserve_thinking /
//     add_vision_id (Api);
//   - undefined attributes concatenate silently where Jinja raises: tool calls without a
//     function name are rejected up front (Api), as both Qwen templates would raise.
//
// Thread safety: a ChatTemplate is immutable after construction; render() may be called
// concurrently (each call builds its own minja context).

#include <chrono>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace halo::chat {

using OrderedJson = nlohmann::ordered_json;

struct RenderOptions {
    bool add_generation_prompt = true;
    /// Extra template variables, e.g. {"enable_thinking": false, "reasoning_effort": "low",
    /// "preserve_thinking": false}. Must be an object or null.
    OrderedJson extra_context = OrderedJson::object();
    /// OpenAI clients send tool_calls[].function.arguments as a JSON *string*; the Qwen
    /// templates need an object (`|items`). When true, string arguments that parse to a JSON
    /// object are replaced by that object before rendering. Off by default so render() is
    /// exactly HF's behavior on the same input.
    bool parse_string_tool_arguments = false;
    /// Time for `strftime_now` (the Qwen templates do not use it); default: now.
    std::optional<std::chrono::system_clock::time_point> now;
};

/// A byte offset in the rendered prompt where a message's turn header begins.
struct MessageOffset {
    std::size_t message_index;  ///< index into the messages array
    std::string role;           ///< the role as rendered in the header (tool runs render "user")
    std::size_t offset;         ///< position of the header's `<|im_start|>`
};

struct RenderResult {
    std::string prompt;
    /// True when the ChatML turn headers (`<|im_start|>ROLE\n`) in `prompt` line up exactly
    /// with the messages. False (and the fields below empty) if the structure could not be
    /// verified, e.g. a message *content* contains a literal `<|im_start|>`; callers (the
    /// prefix cache) must then not place checkpoints from these offsets.
    bool offsets_valid = false;
    /// One entry per message that opens a turn: every user and assistant message, the first
    /// tool message of a consecutive run, and the leading system/developer message(s) when
    /// the template rendered them. Continuation tool messages share their run's header and
    /// have no entry.
    std::vector<MessageOffset> message_starts;
    /// End of the system/tools preamble (0 when the template rendered none).
    std::optional<std::size_t> preamble_end;
    /// Start of the generation prompt (`<|im_start|>assistant\n...`) when requested.
    std::optional<std::size_t> generation_prompt_start;
};

class ChatTemplate {
public:
    /// Parses `source`. Throws halo::Error(Config) if it is empty, larger than 1 MiB, or
    /// minja cannot parse it.
    explicit ChatTemplate(std::string source, std::string bos_token = {}, std::string eos_token = {});
    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;
    ChatTemplate(const ChatTemplate&) = delete;
    ChatTemplate& operator=(const ChatTemplate&) = delete;
    ~ChatTemplate();

    /// Renders `messages` (array of objects) with optional `tools` (array, or null for
    /// none). Deterministic for identical inputs. Throws halo::Error(Api) when the template
    /// calls raise_exception (message = the template's text) or rendering fails on the
    /// given input, and for malformed arguments (messages not an array, etc.).
    [[nodiscard]] RenderResult render(const OrderedJson& messages, const OrderedJson& tools,
                                      const RenderOptions& options) const;

    /// render(...).prompt
    [[nodiscard]] std::string apply(const OrderedJson& messages, const OrderedJson& tools,
                                    const RenderOptions& options) const;

    [[nodiscard]] const std::string& source() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::chat
