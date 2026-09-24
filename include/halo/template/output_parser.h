#pragma once
// Model-output parser for the Qwen3.8 chat format: splits `<think>…</think>` reasoning from
// content and turns the template's tool-call markup
//
//   <tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n…</function>\n</tool_call>
//
// into OpenAI-style tool calls. Works incrementally on decoded text pieces (e.g. from
// tokenizer::StreamDecoder) and reports reasoning/content deltas and completed tool calls.
//
// Whitespace: the template renders history as `reasoning_content|trim` and `content|trim`,
// so the parser trims each reasoning/content segment (Python str.strip() whitespace) —
// whitespace next to `</think>` and around tool-call blocks is dropped, interior
// whitespace kept. Streaming deltas hold back trailing whitespace until more text arrives,
// so the concatenated deltas always equal the final fields.
//
// Argument typing: a parameter whose schema type (from `tools`) is "string" stays the raw
// text; any other parameter is parsed as JSON, falling back to the raw string when it is not
// valid JSON. For a type list containing "string" (e.g. ["string","null"]) the JSON parse is
// kept only if it has one of the other listed types.
//
// Content segments separated by a tool-call block are joined with a blank line ("\n\n").
//
// Malformed markup (e.g. a `<tool_call>` block without `<function=…>`, or one still open at
// finish()) is not dropped: it is appended to content verbatim and recorded in `warnings`.

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace halo::chat {

struct ToolCall {
    std::string name;
    nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
};

struct ParsedMessage {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::vector<std::string> warnings;

    /// {"role":"assistant","content":…,"reasoning_content":…,"tool_calls":[{"id":
    /// "<prefix><i>","type":"function","function":{"name":…,"arguments":"<json string>"}}]}.
    /// content is null when empty and tool calls exist; empty fields are omitted.
    [[nodiscard]] nlohmann::ordered_json to_openai_json(std::string_view id_prefix = "call_") const;
};

struct ParserOptions {
    /// True when the prompt ended inside a think block (the generation prompt ends with
    /// "<think>\n", i.e. thinking enabled): output starts as reasoning until `</think>`.
    /// When false, a `<think>` at the very start of the output still opens reasoning.
    bool starts_in_reasoning = true;
    /// The request's tools array (OpenAI format); used only for argument types.
    nlohmann::ordered_json tools = nullptr;
};

struct OutputEvent {
    enum class Kind { ReasoningDelta, ContentDelta, ToolCall };
    Kind kind;
    std::string text;            ///< delta text (Reasoning/ContentDelta)
    std::size_t tool_index = 0;  ///< index into message().tool_calls (ToolCall)
};

class OutputParser {
public:
    explicit OutputParser(ParserOptions options = {});

    /// Feeds the next piece of decoded output (any split, including mid-tag).
    [[nodiscard]] std::vector<OutputEvent> feed(std::string_view piece);
    /// Ends the stream: flushes held text and resolves an unterminated block.
    [[nodiscard]] std::vector<OutputEvent> finish();
    /// Everything parsed so far (complete after finish()).
    [[nodiscard]] const ParsedMessage& message() const noexcept { return msg_; }

    /// One-shot convenience: feed(text) + finish().
    [[nodiscard]] static ParsedMessage parse(std::string_view text, const ParserOptions& options = {});

private:
    enum class State { Reasoning, Content, ToolCall };
    void emit(OutputEvent::Kind kind, std::string_view text, std::vector<OutputEvent>& out);
    void end_segment();
    void finish_tool_block(std::string_view block, std::vector<OutputEvent>& out);
    void process(std::vector<OutputEvent>& out, bool final);

    ParserOptions opt_;
    ParsedMessage msg_;
    State state_;
    std::string buf_;
    std::string utf8_tail_;  ///< incomplete UTF-8 sequence from the last piece
    std::string held_ws_;
    bool segment_start_ = true;
    bool output_start_ = true;            ///< nothing but whitespace seen yet
    bool field_needs_separator_ = false;  ///< next content segment follows a tool call
};

/// True when a rendered prompt leaves the model inside a think block (ends "<think>\n").
[[nodiscard]] bool prompt_ends_in_reasoning(std::string_view prompt) noexcept;

}  // namespace halo::chat
