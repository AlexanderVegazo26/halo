#pragma once
// Request parsing and validation for the OpenAI and Anthropic routes. Every function
// validates types and ranges and throws RequestError naming the offending field; the output
// is a normalized, template-ready job (only known fields are forwarded to the template).

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "halo/api/server.h"
#include "halo/sampling/sampling.h"
#include "halo/template/chat_template.h"
#include "json_util.h"

namespace halo::api {

struct ChatJob {
    Json messages = Json::array();  // template form: role/content/reasoning_content/tool_calls
    Json tools = nullptr;           // OpenAI-form tools array, or null
    chat::RenderOptions render;     // extra_context: enable_thinking / reasoning_effort / preserve_thinking
    SamplingParams sampling;
    std::optional<std::size_t> max_tokens;
    std::optional<std::size_t> reasoning_budget;  // explicit (Anthropic thinking.budget_tokens)
    bool thinking_disabled = false;
    std::vector<std::string> stop;
    bool stream = false;
    bool include_usage = false;
    bool add_generation_prompt = true;  // /apply-template only
    bool tokenize = false;              // /apply-template only
    std::vector<std::string> warnings;  // request-level notes for the response
};

struct CompletionJob {
    std::string prompt;
    SamplingParams sampling;
    std::optional<std::size_t> max_tokens;
    std::vector<std::string> stop;
    bool stream = false;
    bool include_usage = false;
};

/// POST /v1/chat/completions
[[nodiscard]] ChatJob parse_openai_chat(const Json& body, const ServerConfig& cfg);
/// POST /v1/messages
[[nodiscard]] ChatJob parse_anthropic_messages(const Json& body, const ServerConfig& cfg);
/// POST /v1/completions
[[nodiscard]] CompletionJob parse_openai_completion(const Json& body, const ServerConfig& cfg);
/// POST /apply-template: OpenAI-form messages/tools plus add_generation_prompt, tokenize,
/// reasoning_effort and chat_template_kwargs.
[[nodiscard]] ChatJob parse_apply_template(const Json& body, const ServerConfig& cfg);

/// OpenAI reasoning_effort / Anthropic output_config.effort -> template value, or nullopt for
/// "none" (thinking disabled). Throws RequestError for unknown values.
[[nodiscard]] std::optional<std::string> normalize_effort(const std::string& v, const std::string& where);

/// Throws RequestError(TooLarge) when `n` exceeds cfg.max_content_bytes.
void check_content_bytes(std::size_t n, const ServerConfig& cfg, const std::string& where);

}  // namespace halo::api
