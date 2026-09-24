#include "requests.h"

#include <algorithm>
#include <format>
#include <limits>

namespace halo::api {

namespace {

constexpr std::size_t kMaxStopBytes = 256;

[[noreturn]] void bad(const std::string& where, const std::string& msg) {
    throw RequestError(ErrorKind::InvalidRequest, msg, where);
}
[[noreturn]] void unsupported(const std::string& where, const std::string& msg) {
    throw RequestError(ErrorKind::Unsupported, msg, where);
}

const Json& require_object(const Json& v, const std::string& where) {
    if (!v.is_object()) bad(where, "must be an object");
    return v;
}

const Json& require_array(const Json& obj, std::string_view key, const std::string& where, bool non_empty) {
    const Json* v = field(obj, key);
    if (v == nullptr) bad(where, "is required");
    if (!v->is_array()) bad(where, "must be an array");
    if (non_empty && v->empty()) bad(where, "must not be empty");
    return *v;
}

/// Tool names end up inside the template's `<function=NAME>` markup; a restricted alphabet
/// keeps a name from breaking that markup.
void check_tool_name(const std::string& name, const std::string& where) {
    if (name.empty() || name.size() > 128) bad(where, "must be 1-128 characters");
    const bool ok = std::ranges::all_of(name, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
               c == '.' || c == ':';
    });
    if (!ok) bad(where, "may contain only letters, digits, '_', '-', '.' and ':'");
}

/// Total bytes of every string (values and keys) in `v`. Iterative: `v` is depth-capped
/// already, but this keeps the walk independent of that.
std::size_t string_bytes(const Json& v) {
    std::size_t total = 0;
    std::vector<const Json*> stack{&v};
    while (!stack.empty()) {
        const Json* j = stack.back();
        stack.pop_back();
        if (j->is_string()) {
            total += j->get_ref<const std::string&>().size();
        } else if (j->is_array()) {
            for (const auto& e : *j) stack.push_back(&e);
        } else if (j->is_object()) {
            for (auto it = j->begin(); it != j->end(); ++it) {
                total += it.key().size();
                stack.push_back(&it.value());
            }
        }
    }
    return total;
}

void check_bytes(std::size_t n, const ServerConfig& cfg, const std::string& where) {
    if (n > cfg.max_content_bytes) {
        throw RequestError(ErrorKind::TooLarge,
                           std::format("{} bytes of text exceed this server's limit of {} bytes", n,
                                       cfg.max_content_bytes),
                           where);
    }
}

void check_schema_size(const Json& schema, const ServerConfig& cfg, const std::string& where) {
    const std::size_t n = dump(schema).size();
    if (n > cfg.max_tool_schema_bytes) {
        bad(where, std::format("is {} bytes; this server accepts at most {} bytes per tool schema", n,
                               cfg.max_tool_schema_bytes));
    }
}

std::optional<std::size_t> parse_max_tokens(const Json& body, const ServerConfig& cfg, const char* primary,
                                            const char* alias) {
    const auto hi = static_cast<std::int64_t>(std::min<std::size_t>(cfg.max_tokens_cap, std::numeric_limits<std::int32_t>::max()));
    std::optional<std::int64_t> v;
    std::string where = primary;
    if (field(body, primary) != nullptr) {
        v = opt_int(body, primary, primary, 1, std::numeric_limits<std::int64_t>::max());
    } else if (alias != nullptr && field(body, alias) != nullptr) {
        where = alias;
        v = opt_int(body, alias, alias, 1, std::numeric_limits<std::int64_t>::max());
    }
    if (!v) return std::nullopt;
    if (*v > hi) {
        bad(where, std::format("exceeds this server's per-request cap of {} tokens", cfg.max_tokens_cap));
    }
    return static_cast<std::size_t>(*v);
}

std::vector<std::string> parse_stop(const Json& body, std::string_view key, const ServerConfig& cfg, bool allow_string) {
    std::vector<std::string> out;
    const Json* v = field(body, key);
    const std::string where(key);
    if (v == nullptr) return out;
    if (v->is_string() && allow_string) {
        out.push_back(v->get<std::string>());
    } else if (v->is_array()) {
        for (std::size_t i = 0; i < v->size(); ++i) {
            if (!(*v)[i].is_string()) bad(join_path(where, i), "must be a string");
            out.push_back((*v)[i].get<std::string>());
        }
    } else {
        bad(where, allow_string ? "must be a string or an array of strings" : "must be an array of strings");
    }
    if (out.size() > cfg.max_stop_sequences) {
        bad(where, std::format("at most {} stop sequences are allowed", cfg.max_stop_sequences));
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i].empty()) bad(join_path(where, i), "must not be empty");
        if (out[i].size() > kMaxStopBytes) bad(join_path(where, i), "must be at most 256 bytes");
    }
    return out;
}

/// Sampling fields shared by the OpenAI routes (plus llama.cpp-style extensions).
void parse_openai_sampling(const Json& b, SamplingParams& s) {
    if (auto v = opt_number(b, "temperature", "temperature", 0.0, 2.0)) s.temperature = static_cast<float>(*v);
    if (auto v = opt_number(b, "top_p", "top_p", 0.0, 1.0)) s.top_p = static_cast<float>(*v);
    if (auto v = opt_int(b, "top_k", "top_k", 0, 1 << 20)) s.top_k = static_cast<int>(*v);
    if (auto v = opt_number(b, "min_p", "min_p", 0.0, 1.0)) s.min_p = static_cast<float>(*v);
    if (auto v = opt_number(b, "typical_p", "typical_p", 0.0, 1.0)) s.typical_p = static_cast<float>(*v);
    if (auto v = opt_number(b, "repetition_penalty", "repetition_penalty", 0.0, 10.0)) {
        if (*v <= 0.0) bad("repetition_penalty", "must be > 0");
        s.repetition_penalty = static_cast<float>(*v);
    }
    if (auto v = opt_number(b, "presence_penalty", "presence_penalty", -2.0, 2.0)) s.presence_penalty = static_cast<float>(*v);
    if (auto v = opt_number(b, "frequency_penalty", "frequency_penalty", -2.0, 2.0)) {
        s.frequency_penalty = static_cast<float>(*v);
    }
    if (auto v = opt_int(b, "seed", "seed", std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max())) {
        s.seed = static_cast<std::uint64_t>(*v);
    }
    if (auto v = opt_int(b, "n", "n", 1, 1024); v && *v != 1) unsupported("n", "only n=1 is supported");
    if (auto v = opt_bool(b, "logprobs", "logprobs"); v && *v) unsupported("logprobs", "logprobs are not supported");
    if (field(b, "top_logprobs") != nullptr) unsupported("top_logprobs", "logprobs are not supported");
    if (field(b, "logit_bias") != nullptr && !b["logit_bias"].empty()) unsupported("logit_bias", "logit_bias is not supported");
}

void parse_stream(const Json& b, bool& stream, bool& include_usage) {
    stream = opt_bool(b, "stream", "stream").value_or(false);
    if (const Json* so = field(b, "stream_options")) {
        require_object(*so, "stream_options");
        include_usage = opt_bool(*so, "include_usage", "stream_options.include_usage").value_or(false);
    }
}

void parse_response_format(const Json& b, SamplingParams& s) {
    const Json* rf = field(b, "response_format");
    if (rf == nullptr) return;
    require_object(*rf, "response_format");
    const auto type = opt_string(*rf, "type", "response_format.type");
    if (!type) bad("response_format.type", "is required");
    if (*type == "text") return;
    if (*type == "json_object") {
        s.json_object = true;
        return;
    }
    if (*type == "json_schema") {
        const Json* js = field(*rf, "json_schema");
        if (js == nullptr) bad("response_format.json_schema", "is required for type json_schema");
        require_object(*js, "response_format.json_schema");
        const Json* schema = field(*js, "schema");
        if (schema == nullptr) bad("response_format.json_schema.schema", "is required");
        if (!schema->is_object() && !schema->is_boolean()) {
            bad("response_format.json_schema.schema", "must be a JSON Schema object");
        }
        s.json_schema = dump(*schema);
        return;
    }
    bad("response_format.type", "must be one of text, json_object, json_schema");
}

/// OpenAI message content: string | null | array of {type:"text", text}.
std::string openai_text(const Json& m, const std::string& where) {
    const Json* c = field(m, "content");
    if (c == nullptr) return {};
    if (c->is_string()) return c->get<std::string>();
    if (!c->is_array()) bad(where, "must be a string, an array of content parts, or null");
    std::string out;
    for (std::size_t i = 0; i < c->size(); ++i) {
        const Json& part = (*c)[i];
        const std::string pw = join_path(where, i);
        require_object(part, pw);
        const auto type = opt_string(part, "type", join_path(pw, "type"));
        if (!type) bad(join_path(pw, "type"), "is required");
        if (*type == "text" || *type == "refusal") {
            const auto t = opt_string(part, *type == "text" ? "text" : "refusal", join_path(pw, *type));
            out += t.value_or("");
        } else if (*type == "image_url" || *type == "input_audio" || *type == "file" || *type == "image") {
            unsupported(join_path(pw, "type"), "only text input is supported by this server (" + *type + ")");
        } else {
            bad(join_path(pw, "type"), "unknown content part type '" + *type + "'");
        }
    }
    return out;
}

Json parse_openai_messages(const Json& body, const ServerConfig& cfg) {
    const Json& msgs = require_array(body, "messages", "messages", true);
    if (msgs.size() > cfg.max_messages) bad("messages", std::format("at most {} messages are allowed", cfg.max_messages));
    Json out = Json::array();
    for (std::size_t i = 0; i < msgs.size(); ++i) {
        const std::string w = join_path("messages", i);
        const Json& m = require_object(msgs[i], w);
        const auto role = opt_string(m, "role", join_path(w, "role"));
        if (!role) bad(join_path(w, "role"), "is required");
        Json o = Json::object();
        if (*role == "system" || *role == "developer") {
            o["role"] = "system";  // the ggml-org template has no developer role
            o["content"] = openai_text(m, join_path(w, "content"));
        } else if (*role == "user") {
            o["role"] = "user";
            o["content"] = openai_text(m, join_path(w, "content"));
        } else if (*role == "assistant") {
            o["role"] = "assistant";
            o["content"] = openai_text(m, join_path(w, "content"));
            if (auto rc = opt_string(m, "reasoning_content", join_path(w, "reasoning_content"))) o["reasoning_content"] = *rc;
            if (const Json* tcs = field(m, "tool_calls")) {
                const std::string tw = join_path(w, "tool_calls");
                if (!tcs->is_array()) bad(tw, "must be an array");
                Json calls = Json::array();
                for (std::size_t j = 0; j < tcs->size(); ++j) {
                    const std::string cw = join_path(tw, j);
                    const Json& tc = require_object((*tcs)[j], cw);
                    const Json* fn = field(tc, "function");
                    if (fn == nullptr) bad(join_path(cw, "function"), "is required");
                    require_object(*fn, join_path(cw, "function"));
                    const auto name = opt_string(*fn, "name", join_path(cw, "function.name"));
                    if (!name) bad(join_path(cw, "function.name"), "is required");
                    check_tool_name(*name, join_path(cw, "function.name"));
                    Json args = Json::object();
                    if (const Json* a = field(*fn, "arguments")) {
                        const std::string aw = join_path(cw, "function.arguments");
                        if (a->is_string()) {
                            // Same depth cap as the body (security review S-2): the parsed
                            // value is copied, dumped and rendered recursively later.
                            args = parse_embedded_json(a->get_ref<const std::string&>(), cfg.max_json_depth, aw);
                        } else {
                            args = *a;
                        }
                        if (!args.is_object()) bad(aw, "must be a JSON object");
                    }
                    Json call = {{"type", "function"}, {"function", {{"name", *name}, {"arguments", std::move(args)}}}};
                    if (auto id = opt_string(tc, "id", join_path(cw, "id"))) call["id"] = *id;
                    calls.push_back(std::move(call));
                }
                if (!calls.empty()) o["tool_calls"] = std::move(calls);
            }
        } else if (*role == "tool") {
            o["role"] = "tool";
            o["content"] = openai_text(m, join_path(w, "content"));
            if (auto id = opt_string(m, "tool_call_id", join_path(w, "tool_call_id"))) o["tool_call_id"] = *id;
        } else {
            bad(join_path(w, "role"), "must be one of system, developer, user, assistant, tool");
        }
        out.push_back(std::move(o));
    }
    check_bytes(string_bytes(out), cfg, "messages");
    return out;
}

Json parse_openai_tools(const Json& body, const ServerConfig& cfg) {
    const Json* t = field(body, "tools");
    if (t == nullptr) return nullptr;
    if (!t->is_array()) bad("tools", "must be an array");
    if (t->size() > cfg.max_tools) bad("tools", std::format("at most {} tools are allowed", cfg.max_tools));
    if (t->empty()) return nullptr;
    Json out = Json::array();
    for (std::size_t i = 0; i < t->size(); ++i) {
        const std::string w = join_path("tools", i);
        const Json& tool = require_object((*t)[i], w);
        const auto type = opt_string(tool, "type", join_path(w, "type")).value_or("function");
        if (type != "function") unsupported(join_path(w, "type"), "only function tools are supported");
        const Json* fn = field(tool, "function");
        if (fn == nullptr) bad(join_path(w, "function"), "is required");
        require_object(*fn, join_path(w, "function"));
        const auto name = opt_string(*fn, "name", join_path(w, "function.name"));
        if (!name) bad(join_path(w, "function.name"), "is required");
        check_tool_name(*name, join_path(w, "function.name"));
        Json f = {{"name", *name}};
        if (auto d = opt_string(*fn, "description", join_path(w, "function.description"))) f["description"] = *d;
        if (const Json* p = field(*fn, "parameters")) {
            if (!p->is_object()) bad(join_path(w, "function.parameters"), "must be a JSON Schema object");
            check_schema_size(*p, cfg, join_path(w, "function.parameters"));
            f["parameters"] = *p;
        }
        out.push_back(Json{{"type", "function"}, {"function", std::move(f)}});
    }
    return out;
}

/// Returns false when tool_choice disables tools.
bool parse_openai_tool_choice(const Json& body) {
    const Json* tc = field(body, "tool_choice");
    if (tc == nullptr) return true;
    if (tc->is_string()) {
        const auto s = tc->get<std::string>();
        if (s == "auto") return true;
        if (s == "none") return false;
        if (s == "required") unsupported("tool_choice", "tool_choice \"required\" is not supported (no forced tool calls)");
        bad("tool_choice", "must be auto, none, required, or a function object");
    }
    if (tc->is_object()) unsupported("tool_choice", "forcing a specific tool is not supported");
    bad("tool_choice", "must be a string or an object");
}

/// OpenAI-side thinking options: reasoning_effort + chat_template_kwargs.
void parse_openai_thinking(const Json& body, ChatJob& job) {
    std::optional<std::string> effort_raw = opt_string(body, "reasoning_effort", "reasoning_effort");
    std::string effort_where = "reasoning_effort";
    std::optional<bool> enable, preserve;
    if (const Json* kw = field(body, "chat_template_kwargs")) {
        require_object(*kw, "chat_template_kwargs");
        for (auto it = kw->begin(); it != kw->end(); ++it) {
            const std::string k = it.key();
            if (k != "enable_thinking" && k != "preserve_thinking" && k != "reasoning_effort") {
                unsupported("chat_template_kwargs." + k,
                            "only enable_thinking, preserve_thinking and reasoning_effort are accepted");
            }
        }
        enable = opt_bool(*kw, "enable_thinking", "chat_template_kwargs.enable_thinking");
        preserve = opt_bool(*kw, "preserve_thinking", "chat_template_kwargs.preserve_thinking");
        if (auto e = opt_string(*kw, "reasoning_effort", "chat_template_kwargs.reasoning_effort")) {
            effort_raw = e;
            effort_where = "chat_template_kwargs.reasoning_effort";
        }
    }
    std::optional<std::string> effort;
    if (effort_raw) {
        effort = normalize_effort(*effort_raw, effort_where);
        if (!effort) job.thinking_disabled = true;
    }
    if (enable && !*enable) job.thinking_disabled = true;
    Json& x = job.render.extra_context;
    if (job.thinking_disabled) {
        x["enable_thinking"] = false;
    } else {
        if (enable) x["enable_thinking"] = true;
        if (effort) x["reasoning_effort"] = *effort;
    }
    if (preserve) x["preserve_thinking"] = *preserve;
}

// ---- Anthropic -----------------------------------------------------------------------------

/// Anthropic text-ish content: string | array of {type:"text", text}.
std::string anthropic_text_blocks(const Json& v, const std::string& where) {
    if (v.is_string()) return v.get<std::string>();
    if (!v.is_array()) bad(where, "must be a string or an array of text blocks");
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const std::string bw = join_path(where, i);
        const Json& blk = require_object(v[i], bw);
        const auto type = opt_string(blk, "type", join_path(bw, "type"));
        if (type == "text") {
            if (!out.empty()) out += "\n";
            out += opt_string(blk, "text", join_path(bw, "text")).value_or("");
        } else if (type == "image" || type == "document") {
            unsupported(join_path(bw, "type"), "only text input is supported by this server (" + *type + ")");
        } else {
            bad(join_path(bw, "type"), "expected a text block");
        }
    }
    return out;
}

void append_anthropic_message(const Json& m, const std::string& w, Json& out) {
    const auto role = opt_string(m, "role", join_path(w, "role"));
    if (!role) bad(join_path(w, "role"), "is required");
    if (*role != "user" && *role != "assistant") bad(join_path(w, "role"), "must be user or assistant");
    const Json* c = field(m, "content");
    if (c == nullptr) bad(join_path(w, "content"), "is required");
    const std::string cw = join_path(w, "content");
    if (c->is_string()) {
        out.push_back(Json{{"role", *role}, {"content", c->get<std::string>()}});
        return;
    }
    if (!c->is_array()) bad(cw, "must be a string or an array of content blocks");

    if (*role == "assistant") {
        std::string text, thinking;
        Json calls = Json::array();
        for (std::size_t i = 0; i < c->size(); ++i) {
            const std::string bw = join_path(cw, i);
            const Json& blk = require_object((*c)[i], bw);
            const auto type = opt_string(blk, "type", join_path(bw, "type")).value_or("");
            if (type == "text") {
                text += opt_string(blk, "text", join_path(bw, "text")).value_or("");
            } else if (type == "thinking") {
                thinking += opt_string(blk, "thinking", join_path(bw, "thinking")).value_or("");
            } else if (type == "redacted_thinking") {
                // opaque; nothing to render
            } else if (type == "tool_use") {
                const auto name = opt_string(blk, "name", join_path(bw, "name"));
                if (!name) bad(join_path(bw, "name"), "is required");
                check_tool_name(*name, join_path(bw, "name"));
                Json input = Json::object();
                if (const Json* in = field(blk, "input")) {
                    if (!in->is_object()) bad(join_path(bw, "input"), "must be an object");
                    input = *in;
                }
                Json call = {{"type", "function"}, {"function", {{"name", *name}, {"arguments", std::move(input)}}}};
                if (auto id = opt_string(blk, "id", join_path(bw, "id"))) call["id"] = *id;
                calls.push_back(std::move(call));
            } else {
                bad(join_path(bw, "type"), "unsupported assistant content block type '" + type + "'");
            }
        }
        Json o = {{"role", "assistant"}, {"content", text}};
        if (!thinking.empty()) o["reasoning_content"] = thinking;
        if (!calls.empty()) o["tool_calls"] = std::move(calls);
        out.push_back(std::move(o));
        return;
    }

    // user: text blocks become a user message, tool_result blocks become tool messages, in
    // block order.
    std::string text;
    bool have_text = false;
    const auto flush_text = [&] {
        if (have_text) out.push_back(Json{{"role", "user"}, {"content", text}});
        text.clear();
        have_text = false;
    };
    for (std::size_t i = 0; i < c->size(); ++i) {
        const std::string bw = join_path(cw, i);
        const Json& blk = require_object((*c)[i], bw);
        const auto type = opt_string(blk, "type", join_path(bw, "type")).value_or("");
        if (type == "text") {
            if (have_text) text += "\n";
            text += opt_string(blk, "text", join_path(bw, "text")).value_or("");
            have_text = true;
        } else if (type == "tool_result") {
            flush_text();
            std::string result;
            if (const Json* rc = field(blk, "content")) result = anthropic_text_blocks(*rc, join_path(bw, "content"));
            if (opt_bool(blk, "is_error", join_path(bw, "is_error")).value_or(false)) result = "Error: " + result;
            Json o = {{"role", "tool"}, {"content", std::move(result)}};
            if (auto id = opt_string(blk, "tool_use_id", join_path(bw, "tool_use_id"))) o["tool_call_id"] = *id;
            out.push_back(std::move(o));
        } else if (type == "image" || type == "document") {
            unsupported(join_path(bw, "type"), "only text input is supported by this server (" + type + ")");
        } else {
            bad(join_path(bw, "type"), "unsupported user content block type '" + type + "'");
        }
    }
    flush_text();
}

void parse_anthropic_structured(const Json& body, SamplingParams& s) {
    // Canonical: output_config.format; deprecated alias: output_format. Same shape:
    // {"type":"json_schema","schema":{...}}.
    const Json* fmt = nullptr;
    std::string where;
    if (const Json* oc = field(body, "output_config")) {
        require_object(*oc, "output_config");
        if ((fmt = field(*oc, "format")) != nullptr) where = "output_config.format";
    }
    if (fmt == nullptr) {
        if ((fmt = field(body, "output_format")) != nullptr) where = "output_format";
    }
    if (fmt == nullptr) return;
    require_object(*fmt, where);
    const auto type = opt_string(*fmt, "type", join_path(where, "type"));
    if (type != "json_schema") bad(join_path(where, "type"), "must be json_schema");
    const Json* schema = field(*fmt, "schema");
    if (schema == nullptr || !schema->is_object()) bad(join_path(where, "schema"), "must be a JSON Schema object");
    s.json_schema = dump(*schema);
}

}  // namespace

void check_content_bytes(std::size_t n, const ServerConfig& cfg, const std::string& where) { check_bytes(n, cfg, where); }

std::optional<std::string> normalize_effort(const std::string& v, const std::string& where) {
    if (v == "none") return std::nullopt;
    if (v == "minimal" || v == "low") return "low";
    if (v == "medium") return "medium";
    // The Qwen3.8 templates know xhigh/medium/low; the ggml-org one rejects "high".
    if (v == "high" || v == "xhigh" || v == "max") return "xhigh";
    bad(where, "must be one of none, minimal, low, medium, high, xhigh");
}

ChatJob parse_openai_chat(const Json& body, const ServerConfig& cfg) {
    ChatJob job;
    (void)opt_string(body, "model", "model");
    job.messages = parse_openai_messages(body, cfg);
    job.tools = parse_openai_tools(body, cfg);
    if (!parse_openai_tool_choice(body)) job.tools = nullptr;
    (void)opt_bool(body, "parallel_tool_calls", "parallel_tool_calls");  // accepted, not enforced (docs/api.md)
    parse_openai_thinking(body, job);
    parse_openai_sampling(body, job.sampling);
    parse_response_format(body, job.sampling);
    job.max_tokens = parse_max_tokens(body, cfg, "max_completion_tokens", "max_tokens");
    job.stop = parse_stop(body, "stop", cfg, true);
    parse_stream(body, job.stream, job.include_usage);
    return job;
}

ChatJob parse_apply_template(const Json& body, const ServerConfig& cfg) {
    ChatJob job;
    job.messages = parse_openai_messages(body, cfg);
    job.tools = parse_openai_tools(body, cfg);
    parse_openai_thinking(body, job);
    job.add_generation_prompt = opt_bool(body, "add_generation_prompt", "add_generation_prompt").value_or(true);
    job.tokenize = opt_bool(body, "tokenize", "tokenize").value_or(false);
    return job;
}

ChatJob parse_anthropic_messages(const Json& body, const ServerConfig& cfg) {
    ChatJob job;
    (void)opt_string(body, "model", "model");
    job.max_tokens = parse_max_tokens(body, cfg, "max_tokens", nullptr);
    if (!job.max_tokens) bad("max_tokens", "is required");

    if (const Json* sys = field(body, "system")) {
        job.messages.push_back(Json{{"role", "system"}, {"content", anthropic_text_blocks(*sys, "system")}});
    }
    const Json& msgs = require_array(body, "messages", "messages", true);
    if (msgs.size() > cfg.max_messages) bad("messages", std::format("at most {} messages are allowed", cfg.max_messages));
    for (std::size_t i = 0; i < msgs.size(); ++i) {
        const std::string w = join_path("messages", i);
        append_anthropic_message(require_object(msgs[i], w), w, job.messages);
    }
    check_bytes(string_bytes(job.messages), cfg, "messages");
    if (msgs.back().is_object() && msgs.back().value("role", "") == "assistant") {
        unsupported("messages", "a final assistant message (response prefill) is not supported");
    }

    // tools
    bool tools_enabled = true;
    if (const Json* tc = field(body, "tool_choice")) {
        require_object(*tc, "tool_choice");
        const auto type = opt_string(*tc, "type", "tool_choice.type").value_or("auto");
        if (type == "none") {
            tools_enabled = false;
        } else if (type == "any" || type == "tool") {
            unsupported("tool_choice.type", "forced tool use is not supported");
        } else if (type != "auto") {
            bad("tool_choice.type", "must be auto, any, tool or none");
        }
        (void)opt_bool(*tc, "disable_parallel_tool_use", "tool_choice.disable_parallel_tool_use");  // not enforced
    }
    if (const Json* t = field(body, "tools")) {
        if (!t->is_array()) bad("tools", "must be an array");
        if (t->size() > cfg.max_tools) bad("tools", std::format("at most {} tools are allowed", cfg.max_tools));
        Json out = Json::array();
        for (std::size_t i = 0; i < t->size(); ++i) {
            const std::string w = join_path("tools", i);
            const Json& tool = require_object((*t)[i], w);
            if (auto type = opt_string(tool, "type", join_path(w, "type")); type && *type != "custom") {
                unsupported(join_path(w, "type"), "server tools are not supported ('" + *type + "')");
            }
            const auto name = opt_string(tool, "name", join_path(w, "name"));
            if (!name) bad(join_path(w, "name"), "is required");
            check_tool_name(*name, join_path(w, "name"));
            const Json* schema = field(tool, "input_schema");
            if (schema == nullptr || !schema->is_object()) bad(join_path(w, "input_schema"), "must be a JSON Schema object");
            check_schema_size(*schema, cfg, join_path(w, "input_schema"));
            Json f = {{"name", *name}};
            if (auto d = opt_string(tool, "description", join_path(w, "description"))) f["description"] = *d;
            f["parameters"] = *schema;
            out.push_back(Json{{"type", "function"}, {"function", std::move(f)}});
        }
        if (tools_enabled && !out.empty()) job.tools = std::move(out);
    }

    // thinking + effort
    std::optional<std::string> effort;
    if (const Json* oc = field(body, "output_config")) {
        require_object(*oc, "output_config");
        if (auto e = opt_string(*oc, "effort", "output_config.effort")) {
            if (*e == "none") bad("output_config.effort", "must be one of low, medium, high, xhigh, max");
            effort = normalize_effort(*e, "output_config.effort");
        }
    }
    if (const Json* th = field(body, "thinking")) {
        require_object(*th, "thinking");
        const auto type = opt_string(*th, "type", "thinking.type");
        if (!type) bad("thinking.type", "is required");
        if (*type == "disabled") {
            job.thinking_disabled = true;
        } else if (*type == "enabled") {
            const auto b = opt_int(*th, "budget_tokens", "thinking.budget_tokens", 1, std::numeric_limits<std::int64_t>::max());
            if (!b) bad("thinking.budget_tokens", "is required when thinking is enabled");
            if (static_cast<std::size_t>(*b) >= *job.max_tokens) bad("thinking.budget_tokens", "must be less than max_tokens");
            job.reasoning_budget = static_cast<std::size_t>(*b);
        } else if (*type != "adaptive") {
            bad("thinking.type", "must be enabled, disabled or adaptive");
        }
    }
    Json& x = job.render.extra_context;
    if (job.thinking_disabled) {
        x["enable_thinking"] = false;
    } else if (effort) {
        x["reasoning_effort"] = *effort;
    }

    // sampling (Anthropic ranges)
    if (auto v = opt_number(body, "temperature", "temperature", 0.0, 1.0)) job.sampling.temperature = static_cast<float>(*v);
    if (auto v = opt_number(body, "top_p", "top_p", 0.0, 1.0)) job.sampling.top_p = static_cast<float>(*v);
    if (auto v = opt_int(body, "top_k", "top_k", 0, 1 << 20)) job.sampling.top_k = static_cast<int>(*v);
    parse_anthropic_structured(body, job.sampling);
    job.stop = parse_stop(body, "stop_sequences", cfg, false);
    job.stream = opt_bool(body, "stream", "stream").value_or(false);
    job.include_usage = true;
    return job;
}

CompletionJob parse_openai_completion(const Json& body, const ServerConfig& cfg) {
    CompletionJob job;
    (void)opt_string(body, "model", "model");
    const Json* p = field(body, "prompt");
    if (p == nullptr) bad("prompt", "is required");
    if (p->is_string()) {
        job.prompt = p->get<std::string>();
    } else if (p->is_array() && p->size() == 1 && (*p)[0].is_string()) {
        job.prompt = (*p)[0].get<std::string>();
    } else if (p->is_array()) {
        unsupported("prompt", "only a single string prompt is supported (no batches, no token arrays)");
    } else {
        bad("prompt", "must be a string");
    }
    if (job.prompt.empty()) bad("prompt", "must not be empty");
    check_bytes(job.prompt.size(), cfg, "prompt");
    if (auto v = opt_bool(body, "echo", "echo"); v && *v) unsupported("echo", "echo is not supported");
    if (field(body, "suffix") != nullptr) unsupported("suffix", "suffix is not supported");
    if (auto v = opt_int(body, "best_of", "best_of", 1, 1024); v && *v != 1) unsupported("best_of", "only best_of=1 is supported");
    if (const Json* lp = field(body, "logprobs"); lp != nullptr && !(lp->is_boolean() && !lp->get<bool>())) {
        unsupported("logprobs", "logprobs are not supported");
    }
    Json b = body;
    b.erase("logprobs");  // validated above (completions uses an integer here)
    parse_openai_sampling(b, job.sampling);
    job.max_tokens = parse_max_tokens(body, cfg, "max_tokens", nullptr);
    job.stop = parse_stop(body, "stop", cfg, true);
    parse_stream(body, job.stream, job.include_usage);
    return job;
}

}  // namespace halo::api
