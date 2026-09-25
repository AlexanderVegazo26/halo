#pragma once
// HALO chat-template engine (TRD §10): renders the model's own Jinja chat template
// (GGUF `tokenizer.chat_template`) with minja, matching HF `apply_chat_template(...,
// tokenize=False)` byte-for-byte on the Qwen3.8 templates (golden-tested).
//
// Differences from calling minja's chat_template class directly: no capability probing and
// no polyfills (HF applies none), a Python-compatible `trim`, `raise_exception` surfaced
// as a typed halo::Error(Api), a capped `range`, and a deterministic `strftime_now`.
//
// Untrusted templates (security review S-3, S-8, S-10; docs/security-hardening.md): chat
// templates come from model files downloaded from Hugging Face and are treated as hostile.
// minja 021c229 has no {% include %}/{% import %}, so a template cannot read files. HALO
// builds minja with an in-tree patch (cmake/patches/minja-halo-limits.patch) that counts
// the work of every parse and render against TemplateLimits:
//   - parse (constructor): source size, parser nesting depth; the parse runs on a dedicated
//     thread with a large, known stack (std::regex in minja's lexer recurses per character);
//     a violated limit throws halo::Error(Config);
//   - render: render depth, steps, loop iterations, output bytes, largest single value and
//     cumulative value bytes; a violated limit throws halo::Error(Api) whose message starts
//     with "chat template limit exceeded: ". Render runs on the caller's thread; at the
//     default limits it needs at most kMaxRenderStackBytes of stack (measured, see below).
//   - `range()` is capped at 100000 elements (a range of exactly 100000 is allowed).
//
// Defaults are multiples of what the real Qwen3.8 templates need on large conversations;
// they are documented per field. A limit of 0 disables it (tests only).
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
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace halo::chat {

using OrderedJson = nlohmann::ordered_json;

/// Stack a render() needs at the default TemplateLimits: at the render-depth cap (1024) the
/// peak measured by stack painting is ~1.2 MiB (RelWithDebInfo) and ~2.9 MiB (ASan + Debug)
/// (test RenderAtDepthCapFitsTheDocumentedStack). Render on a thread with at least this.
inline constexpr std::size_t kMaxRenderStackBytes = std::size_t{4} << 20;

/// Resource limits applied to one template (see the file comment). 0 disables a limit
/// (tests only). "Measured" figures are the Qwen3.8 ggml/unsloth templates rendering a
/// 4002-message agentic conversation with 64 tools (2.2 MB prompt, about twice a 262K-token
/// context; test DefaultsLeaveHeadroomOverRealTemplates, which asserts >= 4x headroom).
struct TemplateLimits {
    /// Template source size. Real Qwen templates are ~9-10 KB. (Was 1 MiB before S-3.)
    std::size_t max_source_bytes = std::size_t{64} << 10;
    /// Parser recursion depth: each nesting level of brackets, filters, `~`, `not`, if-else
    /// expressions and blocks counts. Violations throw Error(Config).
    std::size_t max_parse_depth = 256;
    /// Stack of the dedicated parse thread (virtual; touched lazily); 0 parses on the
    /// caller's thread, which a 64 KiB single token can overflow (tested).
    std::size_t parse_stack_bytes = std::size_t{256} << 20;
    /// Nesting of node renders + expression evaluations + value traversals. Measured: 20.
    std::size_t max_render_depth = 1024;
    /// Node renders + expression evaluations + charged builtin work (substring searches,
    /// container scans). Measured: 0.86M. A hostile template reaches the cap in ~4 s on the
    /// dev host (D-001), the slowest bound; steps cost ~1 us each.
    std::uint64_t max_steps = 4'000'000;
    /// For-loop iterations, filtering passes included. Measured: 36K.
    std::uint64_t max_loop_iterations = 1'000'000;
    /// Bytes written by output nodes, summed over nested buffers (macro bodies, {% set %}
    /// blocks, filters); an upper bound of the prompt size. Measured: 3.6 MB.
    std::uint64_t max_output_bytes = std::uint64_t{64} << 20;
    /// Largest single value one operation may build: string bytes, or container elements x
    /// sizeof(minja::Value) (~80 B; about 400K elements). Checked before the known
    /// amplifiers (repeat, replace, join, indent, split, tojson indent) and on every result.
    std::uint64_t max_string_bytes = std::uint64_t{32} << 20;
    /// Cumulative bytes of string values produced by expressions or stored into containers,
    /// plus one slot per container element stored. Measured: 21 MB.
    std::uint64_t max_alloc_bytes = std::uint64_t{1} << 30;
};

/// Work one render used (for observability and for sizing TemplateLimits).
struct RenderStats {
    std::size_t peak_depth = 0;
    std::uint64_t steps = 0;
    std::uint64_t loop_iterations = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t alloc_bytes = 0;
};

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
    /// Resources the render used, measured against TemplateLimits.
    RenderStats stats;
};

class ChatTemplate {
public:
    /// Parses `source` with the default TemplateLimits. Throws halo::Error(Config) if it is
    /// empty, larger than TemplateLimits::max_source_bytes (64 KiB), nested deeper than
    /// max_parse_depth, or minja cannot parse it.
    explicit ChatTemplate(std::string source, std::string bos_token = {}, std::string eos_token = {});
    /// As above with explicit limits (also applied to every render()).
    ChatTemplate(std::string source, std::string bos_token, std::string eos_token, const TemplateLimits& limits);
    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;
    ChatTemplate(const ChatTemplate&) = delete;
    ChatTemplate& operator=(const ChatTemplate&) = delete;
    ~ChatTemplate();

    /// Renders `messages` (array of objects) with optional `tools` (array, or null for
    /// none). Deterministic for identical inputs. Throws halo::Error(Api) when the template
    /// calls raise_exception (message = the template's text), rendering fails on the given
    /// input, a TemplateLimits bound is exceeded ("chat template limit exceeded: ..."), and
    /// for malformed arguments (messages not an array, extra_context overriding a reserved
    /// variable — messages, tools, bos_token, eos_token, add_generation_prompt — etc.).
    [[nodiscard]] RenderResult render(const OrderedJson& messages, const OrderedJson& tools,
                                      const RenderOptions& options) const;

    /// render(...).prompt
    [[nodiscard]] std::string apply(const OrderedJson& messages, const OrderedJson& tools,
                                    const RenderOptions& options) const;

    [[nodiscard]] const std::string& source() const noexcept;
    [[nodiscard]] const TemplateLimits& limits() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::chat
