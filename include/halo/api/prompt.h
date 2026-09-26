#pragma once
// Chat prompt construction with special-token injection protection (PRD §12).
//
// Problem: the chat template renders client strings (message content, tool results, tool
// definitions) into one prompt string, and that string must be tokenized with
// parse_special=true so the template's own `<|im_start|>` / `<|im_end|>` become control
// tokens. Tokenizing the whole string that way would also turn a literal `<|im_end|>` inside
// a user message or a tool result into the real control token, letting the client (or a web
// page quoted in a tool result) close the turn and forge a system message.
//
// Approach ("escape, render, segment"):
//   1. Every client string is scanned for added-token literals. Occurrences are replaced by
//      an ASCII-alphanumeric placeholder carrying a per-call random nonce (it survives the
//      template's `trim` and `tojson`; the call is rejected if the nonce already occurs in
//      the input). Which literals are escaped:
//        - Control tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, vision/audio
//          markers, ...): in every client string of every role and in the tools array.
//        - UserDefined markup tokens (`<think>`, `<tool_call>`, `<tool_response>`, FIM
//          markers, ...): in system/user/tool message strings and in the tools array, but
//          not in assistant history — the template itself parses `</think>` out of
//          assistant content, and assistant history is the model's own earlier output.
//   2. The escaped messages are rendered by the model's template.
//   3. The rendered text is split at the placeholders. Template text is tokenized with
//      parse_special=true; each placeholder becomes the tokenization of its literal *as
//      text* (encode_as_text: no added token can be produced).
//   When no client string contains any literal (the normal case) the result is exactly
//   tokenizer.encode(rendered, parse_special=true), i.e. HF apply_chat_template + tokenize.
//
// Residual assumption (documented in docs/api.md): template text adjacent to a client
// string does not complete a *partial* literal the client string ends or begins with. The
// Qwen3.8 templates follow content with `<|im_end|>`, `\n</tool_response>` or `\n`, none of
// which can complete one.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "halo/template/chat_template.h"

namespace halo::tokenizer { class Tokenizer; }

namespace halo::api {

/// Added-token literals of a tokenizer, split by type. Built once per tokenizer (it scans
/// the vocabulary). Immutable; safe to share across threads.
class SpecialTokens {
public:
    explicit SpecialTokens(const tokenizer::Tokenizer& tok);

    [[nodiscard]] const std::vector<std::string>& control() const noexcept { return control_; }
    [[nodiscard]] const std::vector<std::string>& user_defined() const noexcept { return user_defined_; }
    [[nodiscard]] std::optional<std::int32_t> think_start() const noexcept { return think_start_; }
    [[nodiscard]] std::optional<std::int32_t> think_end() const noexcept { return think_end_; }

private:
    std::vector<std::string> control_;
    std::vector<std::string> user_defined_;
    std::optional<std::int32_t> think_start_, think_end_;
};

struct ChatPrompt {
    std::string text;                  ///< rendered prompt (client literals restored)
    std::vector<std::int32_t> tokens;  ///< what the model is fed
    bool starts_in_reasoning = false;  ///< the template left the model inside <think>
    std::size_t neutralized_literals = 0;  ///< client literals tokenized as plain text
    /// M10 (D-013): token offsets of the last few message boundaries, for
    /// GenerateRequest::checkpoint_hints. Bounded to a small, fixed number of the most
    /// recent messages regardless of conversation length (an O(messages) re-render per
    /// hint would be an O(N^2) cost for a long-running conversation).
    std::vector<std::size_t> checkpoint_hints;
};

/// Renders `messages`/`tools` with `tmpl` and tokenizes them as described above. Throws
/// halo::Error(Api) for template errors (raise_exception, malformed input) and for a
/// placeholder-nonce collision that persists after several retries.
[[nodiscard]] ChatPrompt build_chat_prompt(const tokenizer::Tokenizer& tok, const SpecialTokens& specials,
                                           const chat::ChatTemplate& tmpl, const chat::OrderedJson& messages,
                                           const chat::OrderedJson& tools, const chat::RenderOptions& options);

/// Tokenizes `text` so that no added token (Control or UserDefined) is produced: each
/// occurrence of an added-token literal becomes its plain-text tokenization (or, if that
/// would still contain an added token, its first byte + the rest). The text between
/// literals is tokenized with encode(segment, false).
[[nodiscard]] std::vector<std::int32_t> encode_as_text(const tokenizer::Tokenizer& tok, const SpecialTokens& specials,
                                                       std::string_view text);

}  // namespace halo::api
