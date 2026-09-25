// HTTP API server: routing, auth/CORS/limits, OpenAI + Anthropic response shapes, SSE.

#include "halo/api/server.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <thread>

#include "admission.h"
#include "generation.h"
#include "halo/api/prompt.h"
#include "halo/core/error.h"
#include "halo/core/log.h"
#include "halo/runtime/engine.h"
#include "halo/template/chat_template.h"
#include "halo/tokenizer/tokenizer.h"
#include "json_util.h"
#include "metrics.h"
#include "requests.h"

namespace halo::api {

namespace {

constexpr const char* kLog = "api";

ApiFamily family_of(const std::string& path) {
    return path.starts_with("/v1/messages") ? ApiFamily::Anthropic : ApiFamily::OpenAI;
}

std::string route_label(const std::string& path) {
    for (const char* r : {"/health", "/v1/models", "/metrics", "/v1/chat/completions", "/v1/completions",
                          "/v1/messages", "/tokenize", "/apply-template"}) {
        if (path == r) return r;
    }
    return "other";
}

std::int64_t unix_now() { return static_cast<std::int64_t>(std::time(nullptr)); }

std::string lower(std::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return s;
}

/// Host header -> lower-case host name without the port ("[::1]:8080" -> "[::1]").
std::string host_name(std::string_view h) {
    h = trim(h);
    if (h.starts_with('[')) {
        const auto close = h.find(']');
        return lower(std::string(close == std::string_view::npos ? h : h.substr(0, close + 1)));
    }
    return lower(std::string(h.substr(0, h.find(':'))));
}

/// Bind address as it appears in a Host header.
std::string host_form(const std::string& bind) {
    return lower(bind.find(':') != std::string::npos && !bind.starts_with('[') ? "[" + bind + "]" : bind);
}

std::string openai_finish(const GenerationOutcome& o) {
    switch (o.cause) {
        case StopCause::Length:
        case StopCause::Deadline: return "length";
        case StopCause::Eos: return o.message.tool_calls.empty() ? "stop" : "tool_calls";
        default: return "stop";
    }
}

std::string anthropic_stop_reason(const GenerationOutcome& o) {
    switch (o.cause) {
        case StopCause::Length:
        case StopCause::Deadline: return "max_tokens";
        case StopCause::StopSequence: return "stop_sequence";
        case StopCause::Eos: return o.message.tool_calls.empty() ? "end_turn" : "tool_use";
        default: return "end_turn";
    }
}

Json openai_usage(const GenerationOutcome& o) {
    return Json{{"prompt_tokens", o.prompt_tokens},
                {"completion_tokens", o.completion_tokens},
                {"total_tokens", o.prompt_tokens + o.completion_tokens},
                {"prompt_tokens_details", {{"cached_tokens", o.cached_prompt_tokens}}},
                {"completion_tokens_details", {{"reasoning_tokens", o.reasoning_tokens}}}};
}

Json anthropic_usage(const GenerationOutcome& o) {
    const std::size_t cached = std::min(o.cached_prompt_tokens, o.prompt_tokens);
    return Json{{"input_tokens", o.prompt_tokens - cached},
                {"output_tokens", o.completion_tokens},
                {"cache_read_input_tokens", cached},
                {"cache_creation_input_tokens", 0}};
}

Json openai_tool_call(const std::string& id, std::size_t index, const chat::ToolCall& c, bool with_index) {
    Json j = Json::object();
    if (with_index) j["index"] = index;
    j["id"] = id;
    j["type"] = "function";
    j["function"] = {{"name", c.name}, {"arguments", dump(c.arguments)}};
    return j;
}

/// Everything a finished generation needs to be rendered, fixed at request time.
struct ResponseMeta {
    std::string id;       // chatcmpl-/msg-/cmpl- id
    std::string call_id;  // prefix for tool-call ids
    std::string model;
    std::int64_t created = 0;
    bool include_usage = false;
    std::vector<std::string> warnings;  // request-level (e.g. clamped max_tokens)
};

Json warnings_json(const ResponseMeta& m, const GenerationOutcome& o) {
    Json w = Json::array();
    for (const auto& s : m.warnings) w.push_back(s);
    for (const auto& s : o.warnings) w.push_back(s);
    return w;
}

// ---- sinks ---------------------------------------------------------------------------------

class CollectSink final : public EventSink {
public:
    explicit CollectSink(const httplib::Request& req) : req_(req) {}
    bool reasoning(std::string_view) override { return true; }
    bool content(std::string_view) override { return true; }
    bool tool_call(std::size_t, const chat::ToolCall&) override { return true; }
    bool alive() override { return !req_.is_connection_closed(); }

private:
    const httplib::Request& req_;
};

class SseWriter {
public:
    explicit SseWriter(httplib::DataSink& ds) : ds_(ds) {}
    bool data(const Json& j) { return raw("data: " + dump(j) + "\n\n"); }
    bool event(std::string_view name, const Json& j) {
        return raw(std::format("event: {}\ndata: {}\n\n", name, dump(j)));
    }
    bool raw(const std::string& s) {
        if (!ok_) return false;
        ok_ = ds_.write(s.data(), s.size());
        return ok_;
    }
    bool alive() { return ok_ && ds_.is_writable(); }

private:
    httplib::DataSink& ds_;
    bool ok_ = true;
};

class OpenAIChatStream final : public EventSink {
public:
    OpenAIChatStream(SseWriter& w, const ResponseMeta& m) : w_(w), m_(m) {}

    bool start() { return w_.data(chunk(Json{{"role", "assistant"}, {"content", ""}})); }
    bool reasoning(std::string_view t) override { return w_.data(chunk(Json{{"reasoning_content", t}})); }
    bool content(std::string_view t) override { return w_.data(chunk(Json{{"content", t}})); }
    bool tool_call(std::size_t i, const chat::ToolCall& c) override {
        Json calls = Json::array({openai_tool_call(m_.call_id + std::to_string(i), i, c, true)});
        return w_.data(chunk(Json{{"tool_calls", std::move(calls)}}));
    }
    bool alive() override { return w_.alive(); }

    void finish(const GenerationOutcome& o) {
        Json last = chunk(Json::object(), openai_finish(o));
        if (const Json w = warnings_json(m_, o); !w.empty()) last["warnings"] = w;
        if (!w_.data(last)) return;
        if (m_.include_usage) {
            Json u = base();
            u["choices"] = Json::array();
            u["usage"] = openai_usage(o);
            if (!w_.data(u)) return;
        }
        (void)w_.raw("data: [DONE]\n\n");
    }
    void error(const ApiErrorInfo& e) {
        if (w_.data(openai_error_body(e))) (void)w_.raw("data: [DONE]\n\n");
    }

private:
    Json base() const {
        return Json{{"id", m_.id}, {"object", "chat.completion.chunk"}, {"created", m_.created}, {"model", m_.model}};
    }
    Json chunk(Json delta, const std::optional<std::string>& finish = std::nullopt) const {
        Json j = base();
        j["choices"] = Json::array({Json{{"index", 0},
                                         {"delta", std::move(delta)},
                                         {"logprobs", nullptr},
                                         {"finish_reason", finish ? Json(*finish) : Json(nullptr)}}});
        return j;
    }
    SseWriter& w_;
    const ResponseMeta& m_;
};

class CompletionStream final : public EventSink {
public:
    CompletionStream(SseWriter& w, const ResponseMeta& m) : w_(w), m_(m) {}
    bool reasoning(std::string_view t) override { return content(t); }
    bool content(std::string_view t) override { return w_.data(chunk(std::string(t), std::nullopt)); }
    bool tool_call(std::size_t, const chat::ToolCall&) override { return true; }
    bool alive() override { return w_.alive(); }
    void finish(const GenerationOutcome& o) {
        Json last = chunk("", openai_finish(o));
        if (const Json w = warnings_json(m_, o); !w.empty()) last["warnings"] = w;
        if (!w_.data(last)) return;
        if (m_.include_usage) {
            Json u = {{"id", m_.id}, {"object", "text_completion"}, {"created", m_.created}, {"model", m_.model}};
            u["choices"] = Json::array();
            u["usage"] = openai_usage(o);
            if (!w_.data(u)) return;
        }
        (void)w_.raw("data: [DONE]\n\n");
    }
    void error(const ApiErrorInfo& e) {
        if (w_.data(openai_error_body(e))) (void)w_.raw("data: [DONE]\n\n");
    }

private:
    Json chunk(std::string text, const std::optional<std::string>& finish) const {
        return Json{{"id", m_.id},
                    {"object", "text_completion"},
                    {"created", m_.created},
                    {"model", m_.model},
                    {"choices", Json::array({Json{{"text", std::move(text)},
                                                  {"index", 0},
                                                  {"logprobs", nullptr},
                                                  {"finish_reason", finish ? Json(*finish) : Json(nullptr)}}})}};
    }
    SseWriter& w_;
    const ResponseMeta& m_;
};

class AnthropicStream final : public EventSink {
public:
    AnthropicStream(SseWriter& w, const ResponseMeta& m) : w_(w), m_(m) {}

    bool start(std::size_t input_tokens) {
        Json msg = {{"id", m_.id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", m_.model},
                    {"content", Json::array()},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}}}};
        return w_.event("message_start", Json{{"type", "message_start"}, {"message", std::move(msg)}});
    }
    bool reasoning(std::string_view t) override {
        if (!open(Block::Thinking, Json{{"type", "thinking"}, {"thinking", ""}})) return false;
        return delta(Json{{"type", "thinking_delta"}, {"thinking", t}});
    }
    bool content(std::string_view t) override {
        if (!open(Block::Text, Json{{"type", "text"}, {"text", ""}})) return false;
        return delta(Json{{"type", "text_delta"}, {"text", t}});
    }
    bool tool_call(std::size_t i, const chat::ToolCall& c) override {
        if (!close()) return false;
        const Json start = {{"type", "tool_use"}, {"id", m_.call_id + std::to_string(i)}, {"name", c.name}, {"input", Json::object()}};
        if (!open(Block::ToolUse, start)) return false;
        if (!delta(Json{{"type", "input_json_delta"}, {"partial_json", dump(c.arguments)}})) return false;
        return close();
    }
    bool alive() override { return w_.alive(); }

    void finish(const GenerationOutcome& o) {
        if (!close()) return;
        Json md = {{"type", "message_delta"},
                   {"delta", {{"stop_reason", anthropic_stop_reason(o)},
                              {"stop_sequence", o.stop_sequence ? Json(*o.stop_sequence) : Json(nullptr)}}},
                   {"usage", {{"output_tokens", o.completion_tokens}}}};
        if (const Json w = warnings_json(m_, o); !w.empty()) md["warnings"] = w;
        if (!w_.event("message_delta", md)) return;
        (void)w_.event("message_stop", Json{{"type", "message_stop"}});
    }
    void error(const ApiErrorInfo& e) { (void)w_.event("error", anthropic_error_body(e)); }

private:
    enum class Block { None, Thinking, Text, ToolUse };
    bool open(Block b, const Json& block) {
        if (cur_ == b && b != Block::ToolUse) return true;
        if (!close()) return false;
        cur_ = b;
        return w_.event("content_block_start",
                        Json{{"type", "content_block_start"}, {"index", index_}, {"content_block", block}});
    }
    bool delta(const Json& d) {
        return w_.event("content_block_delta", Json{{"type", "content_block_delta"}, {"index", index_}, {"delta", d}});
    }
    bool close() {
        if (cur_ == Block::None) return true;
        cur_ = Block::None;
        const bool ok = w_.event("content_block_stop", Json{{"type", "content_block_stop"}, {"index", index_}});
        ++index_;
        return ok;
    }
    SseWriter& w_;
    const ResponseMeta& m_;
    Block cur_ = Block::None;
    std::size_t index_ = 0;
};

}  // namespace

// ---- Impl ----------------------------------------------------------------------------------

struct ApiServer::Impl {
    Impl(runtime::Engine& e, ServerConfig c)
        : engine(e),
          cfg(std::move(c)),
          specials(e.tokenizer()),
          admission(cfg.max_concurrent, cfg.max_queue) {
        model_name = cfg.served_model_name.empty() ? engine.model().id : cfg.served_model_name;
        if (model_name.empty()) model_name = "halo";
        close_reasoning = engine.tokenizer().encode("\n</think>\n\n", true);
        if (is_loopback_host(cfg.host) || !cfg.allowed_hosts.empty()) {
            check_host = true;
            if (is_loopback_host(cfg.host)) allowed_hosts = {"localhost", "127.0.0.1", "[::1]"};
            allowed_hosts.push_back(host_form(cfg.host));
            for (const auto& h : cfg.allowed_hosts) allowed_hosts.push_back(host_name(h));
        }
        setup();
    }

    bool check_host = false;
    std::vector<std::string> allowed_hosts;

    runtime::Engine& engine;
    ServerConfig cfg;
    SpecialTokens specials;
    std::vector<std::int32_t> close_reasoning;
    std::string model_name;
    httplib::Server svr;
    Admission admission;
    Metrics metrics;
    std::atomic<bool> stopping{false};
    std::thread thread;
    std::mutex ctl;
    int port = -1;
    bool stopped = false;

    // -- helpers ----------------------------------------------------------------------------

    void send_error(httplib::Response& res, ApiFamily fam, const ApiErrorInfo& e) {
        res.status = http_status(e.kind);
        if (e.kind == ErrorKind::RateLimited || e.kind == ErrorKind::Overloaded) res.set_header("Retry-After", "1");
        res.set_content(dump(error_body(fam, e)), "application/json");
    }

    template <typename F>
    httplib::Server::Handler guarded(F f) {
        return [this, f](const httplib::Request& req, httplib::Response& res) {
            const ApiFamily fam = family_of(req.path);
            try {
                f(req, res);
            } catch (const std::exception& e) {
                const ApiErrorInfo info = classify_exception(e);
                if (info.kind == ErrorKind::Server || info.kind == ErrorKind::Overloaded) {
                    HALO_ERROR(kLog, "{} {}: {}", req.method, sanitize_for_log(req.path, 256), sanitize_for_log(e.what()));
                }
                send_error(res, fam, info);
            } catch (...) {
                send_error(res, fam, {ErrorKind::Server, "unknown internal error", std::nullopt, std::nullopt});
            }
        };
    }

    bool origin_allowed(const std::string& origin) const {
        for (const auto& o : cfg.cors_origins) {
            if (o == "*" || o == origin) return true;
        }
        return false;
    }

    bool authorized(const httplib::Request& req) const {
        const std::string& key = *cfg.api_key;
        bool ok = false;
        const std::string auth = req.get_header_value("Authorization");
        if (auth.size() > 7) {
            std::string scheme = auth.substr(0, 7);
            for (auto& ch : scheme) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (scheme == "bearer ") ok = constant_time_equal(key, std::string_view(auth).substr(7)) || ok;
        }
        if (req.has_header("x-api-key")) ok = constant_time_equal(key, req.get_header_value("x-api-key")) || ok;
        return ok;
    }

    std::shared_ptr<AdmissionSlot> admit() {
        switch (admission.acquire(cfg.queue_timeout)) {
            case Admission::Result::Admitted: return std::make_shared<AdmissionSlot>(admission);
            case Admission::Result::QueueFull:
                metrics.count_rejection("queue_full");
                throw RequestError(ErrorKind::RateLimited,
                                   std::format("too many requests: {} running and {} queued", cfg.max_concurrent,
                                               cfg.max_queue));
            case Admission::Result::Timeout:
                metrics.count_rejection("queue_timeout");
                throw RequestError(ErrorKind::Overloaded, "timed out waiting for a generation slot");
            case Admission::Result::Stopped: break;
        }
        throw RequestError(ErrorKind::Overloaded, "server is shutting down");
    }

    ResponseMeta meta(std::string_view id_prefix, std::string_view call_prefix, bool include_usage) const {
        ResponseMeta m;
        const std::string rid = random_hex(24);
        m.id = std::string(id_prefix) + rid;
        m.call_id = std::string(call_prefix) + rid.substr(0, 12) + "_";
        m.model = model_name;
        m.created = unix_now();
        m.include_usage = include_usage;
        return m;
    }

    /// Context check, max_tokens default/clamp, reasoning budget.
    GenerationSpec make_spec(std::vector<std::int32_t> tokens, bool starts_in_reasoning, const Json& tools,
                             const SamplingParams& sampling, std::optional<std::size_t> max_tokens,
                             std::optional<std::size_t> explicit_budget, std::vector<std::string> stop,
                             bool parse_output, ResponseMeta& m) const {
        const std::size_t ctx = engine.model().context_length;
        const std::size_t n = tokens.size();
        if (n == 0) throw RequestError(ErrorKind::InvalidRequest, "the prompt is empty");
        if (ctx != 0 && n >= ctx) {
            throw RequestError(ErrorKind::InvalidRequest,
                               std::format("the prompt is {} tokens; the model context is {}", n, ctx), "messages",
                               "context_length_exceeded");
        }
        std::size_t max = max_tokens.value_or(std::min(cfg.default_max_tokens, cfg.max_tokens_cap));
        if (ctx != 0 && n + max > ctx) {
            if (max_tokens) {
                m.warnings.push_back(std::format("max_tokens reduced from {} to {} to fit the {}-token context", max,
                                                 ctx - n, ctx));
            }
            max = ctx - n;
        }
        GenerationSpec s;
        s.request.prompt = std::move(tokens);
        s.request.sampling = sampling;
        s.request.max_tokens = max;
        s.parse_output = parse_output;
        s.parser.starts_in_reasoning = starts_in_reasoning;
        s.parser.tools = tools;
        s.thinking = starts_in_reasoning;
        s.stop = std::move(stop);
        s.close_reasoning_tokens = close_reasoning;
        s.think_start = specials.think_start();
        s.think_end = specials.think_end();
        if (cfg.request_timeout.count() > 0) s.deadline = std::chrono::steady_clock::now() + cfg.request_timeout;
        s.max_output_nesting = cfg.max_output_nesting;
        s.context_length = ctx;
        if (parse_output) {
            if (explicit_budget) {
                s.reasoning_budget = std::min(*explicit_budget, max);
            } else {
                const std::size_t reserve = std::min(cfg.reasoning_output_reserve, max / 2);
                s.reasoning_budget = max - reserve;
            }
        }
        return s;
    }

    GenerationOutcome run(const GenerationSpec& spec, EventSink& sink) {
        metrics.active.fetch_add(1);
        const auto t0 = std::chrono::steady_clock::now();
        GenerationOutcome o = run_generation(engine, spec, sink, stopping);
        metrics.active.fetch_sub(1);
        metrics.duration.observe(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (o.completion_tokens > 0) metrics.ttft.observe(o.ttft_ms / 1000.0);
        metrics.prompt_tokens += o.prompt_tokens;
        metrics.completion_tokens += o.completion_tokens;
        metrics.reasoning_tokens += o.reasoning_tokens;
        if (o.decode_tps > 0) metrics.last_decode_tps.store(o.decode_tps);
        if (o.cause == StopCause::Client || o.cause == StopCause::Shutdown) metrics.cancelled += 1;
        if (o.reasoning_closed_early) metrics.budget_closes += 1;
        for (const auto& w : o.warnings) HALO_WARN(kLog, "{}", sanitize_for_log(w));
        if (o.cause == StopCause::Error && o.client_error) {
            HALO_INFO(kLog, "request rejected by the engine: {}", sanitize_for_log(o.error));
        } else if (o.cause == StopCause::Error) {
            HALO_ERROR(kLog, "generation failed: {}", sanitize_for_log(o.error));
        }
        return o;
    }

    /// Client-facing error for a failed generation (never the engine's internal text, A-9).
    static ApiErrorInfo stream_error(const GenerationOutcome& o) {
        if (o.cause == StopCause::Shutdown) return {ErrorKind::Overloaded, "server is shutting down", std::nullopt, std::nullopt};
        if (o.client_error) return *o.client_error;
        return {ErrorKind::Server, o.public_error.empty() ? "generation failed" : o.public_error, std::nullopt,
                std::nullopt};
    }

    // -- routes -----------------------------------------------------------------------------

    void chat(const httplib::Request& req, httplib::Response& res, ApiFamily fam) {
        const Json body = parse_request_body(req.body, cfg.max_json_depth);
        ChatJob job = fam == ApiFamily::OpenAI ? parse_openai_chat(body, cfg) : parse_anthropic_messages(body, cfg);
        job.render.add_generation_prompt = true;
        const ChatPrompt prompt = build_chat_prompt(engine.tokenizer(), specials, engine.chat_template(), job.messages,
                                                    job.tools, job.render);
        ResponseMeta m = fam == ApiFamily::OpenAI ? meta("chatcmpl-", "call_", job.include_usage)
                                                  : meta("msg_", "toolu_", true);
        if (prompt.neutralized_literals > 0) {
            HALO_INFO(kLog, "neutralized {} special-token literal(s) in client strings", prompt.neutralized_literals);
        }
        auto spec = std::make_shared<GenerationSpec>(make_spec(prompt.tokens, prompt.starts_in_reasoning, job.tools,
                                                               job.sampling, job.max_tokens, job.reasoning_budget,
                                                               job.stop, true, m));
        auto slot = admit();

        if (!job.stream) {
            CollectSink sink(req);
            const GenerationOutcome o = run(*spec, sink);
            if (o.cause == StopCause::Error || o.cause == StopCause::Shutdown) {
                const ApiErrorInfo e = stream_error(o);
                throw RequestError(e.kind, e.message, e.param, e.code);
            }
            if (o.cause == StopCause::Client) return;  // nobody to answer
            res.set_content(dump(fam == ApiFamily::OpenAI ? openai_chat_json(o, m) : anthropic_json(o, m)),
                            "application/json");
            return;
        }

        auto meta_ptr = std::make_shared<ResponseMeta>(std::move(m));
        set_sse_headers(res);
        res.set_chunked_content_provider(
            "text/event-stream", [this, spec, slot, meta_ptr, fam](std::size_t, httplib::DataSink& ds) {
                SseWriter w(ds);
                if (fam == ApiFamily::OpenAI) {
                    OpenAIChatStream s(w, *meta_ptr);
                    if (!s.start()) return false;
                    const GenerationOutcome o = run(*spec, s);
                    if (o.cause == StopCause::Client) return false;
                    if (o.cause == StopCause::Error || o.cause == StopCause::Shutdown) s.error(stream_error(o));
                    else s.finish(o);
                } else {
                    AnthropicStream s(w, *meta_ptr);
                    if (!s.start(spec->request.prompt.size())) return false;
                    const GenerationOutcome o = run(*spec, s);
                    if (o.cause == StopCause::Client) return false;
                    if (o.cause == StopCause::Error || o.cause == StopCause::Shutdown) s.error(stream_error(o));
                    else s.finish(o);
                }
                ds.done();
                return true;
            });
    }

    void completion(const httplib::Request& req, httplib::Response& res) {
        const Json body = parse_request_body(req.body, cfg.max_json_depth);
        const CompletionJob job = parse_openai_completion(body, cfg);
        ResponseMeta m = meta("cmpl-", "call_", job.include_usage);
        auto tokens = engine.tokenizer().encode(job.prompt, cfg.completions_parse_special);
        auto spec = std::make_shared<GenerationSpec>(
            make_spec(std::move(tokens), false, nullptr, job.sampling, job.max_tokens, std::nullopt, job.stop, false, m));
        auto slot = admit();
        if (!job.stream) {
            CollectSink sink(req);
            const GenerationOutcome o = run(*spec, sink);
            if (o.cause == StopCause::Error || o.cause == StopCause::Shutdown) {
                const ApiErrorInfo e = stream_error(o);
                throw RequestError(e.kind, e.message, e.param, e.code);
            }
            if (o.cause == StopCause::Client) return;
            Json j = {{"id", m.id},
                      {"object", "text_completion"},
                      {"created", m.created},
                      {"model", m.model},
                      {"choices", Json::array({Json{{"text", o.message.content},
                                                    {"index", 0},
                                                    {"logprobs", nullptr},
                                                    {"finish_reason", openai_finish(o)}}})},
                      {"usage", openai_usage(o)}};
            if (const Json w = warnings_json(m, o); !w.empty()) j["warnings"] = w;
            res.set_content(dump(j), "application/json");
            return;
        }
        auto meta_ptr = std::make_shared<ResponseMeta>(std::move(m));
        set_sse_headers(res);
        res.set_chunked_content_provider("text/event-stream",
                                         [this, spec, slot, meta_ptr](std::size_t, httplib::DataSink& ds) {
                                             SseWriter w(ds);
                                             CompletionStream s(w, *meta_ptr);
                                             const GenerationOutcome o = run(*spec, s);
                                             if (o.cause == StopCause::Client) return false;
                                             if (o.cause == StopCause::Error || o.cause == StopCause::Shutdown) {
                                                 s.error(stream_error(o));
                                             } else {
                                                 s.finish(o);
                                             }
                                             ds.done();
                                             return true;
                                         });
    }

    static void set_sse_headers(httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
    }

    Json openai_chat_json(const GenerationOutcome& o, const ResponseMeta& m) const {
        Json msg = {{"role", "assistant"}};
        const auto& pm = o.message;
        msg["content"] = (pm.content.empty() && !pm.tool_calls.empty()) ? Json(nullptr) : Json(pm.content);
        if (!pm.reasoning_content.empty()) msg["reasoning_content"] = pm.reasoning_content;
        if (!pm.tool_calls.empty()) {
            Json calls = Json::array();
            for (std::size_t i = 0; i < pm.tool_calls.size(); ++i) {
                calls.push_back(openai_tool_call(m.call_id + std::to_string(i), i, pm.tool_calls[i], false));
            }
            msg["tool_calls"] = std::move(calls);
        }
        Json j = {{"id", m.id},
                  {"object", "chat.completion"},
                  {"created", m.created},
                  {"model", m.model},
                  {"choices", Json::array({Json{{"index", 0},
                                                {"message", std::move(msg)},
                                                {"logprobs", nullptr},
                                                {"finish_reason", openai_finish(o)}}})},
                  {"usage", openai_usage(o)}};
        if (const Json w = warnings_json(m, o); !w.empty()) j["warnings"] = w;
        return j;
    }

    Json anthropic_json(const GenerationOutcome& o, const ResponseMeta& m) const {
        Json content = Json::array();
        const auto& pm = o.message;
        if (!pm.reasoning_content.empty()) {
            content.push_back(Json{{"type", "thinking"}, {"thinking", pm.reasoning_content}, {"signature", ""}});
        }
        if (!pm.content.empty()) content.push_back(Json{{"type", "text"}, {"text", pm.content}});
        for (std::size_t i = 0; i < pm.tool_calls.size(); ++i) {
            content.push_back(Json{{"type", "tool_use"},
                                   {"id", m.call_id + std::to_string(i)},
                                   {"name", pm.tool_calls[i].name},
                                   {"input", pm.tool_calls[i].arguments}});
        }
        Json j = {{"id", m.id},
                  {"type", "message"},
                  {"role", "assistant"},
                  {"model", m.model},
                  {"content", std::move(content)},
                  {"stop_reason", anthropic_stop_reason(o)},
                  {"stop_sequence", o.stop_sequence ? Json(*o.stop_sequence) : Json(nullptr)},
                  {"usage", anthropic_usage(o)}};
        if (const Json w = warnings_json(m, o); !w.empty()) j["warnings"] = w;
        return j;
    }

    void tokenize(const httplib::Request& req, httplib::Response& res) const {
        const Json body = parse_request_body(req.body, cfg.max_json_depth);
        const auto content = opt_string(body, "content", "content");
        if (!content) throw RequestError(ErrorKind::InvalidRequest, "is required", "content");
        check_content_bytes(content->size(), cfg, "content");
        const bool parse_special = opt_bool(body, "parse_special", "parse_special").value_or(false);
        const bool with_pieces = opt_bool(body, "with_pieces", "with_pieces").value_or(false);
        const auto& tok = engine.tokenizer();
        const auto ids = tok.encode(*content, parse_special);
        Json arr = Json::array();
        for (const auto id : ids) {
            if (with_pieces) arr.push_back(Json{{"id", id}, {"piece", tok.token_to_piece(id)}});
            else arr.push_back(id);
        }
        res.set_content(dump(Json{{"tokens", std::move(arr)}}), "application/json");
    }

    void apply_template(const httplib::Request& req, httplib::Response& res) const {
        const Json body = parse_request_body(req.body, cfg.max_json_depth);
        ChatJob job = parse_apply_template(body, cfg);
        job.render.add_generation_prompt = job.add_generation_prompt;
        const ChatPrompt p = build_chat_prompt(engine.tokenizer(), specials, engine.chat_template(), job.messages,
                                               job.tools, job.render);
        Json j = {{"prompt", p.text}};
        if (job.tokenize) j["tokens"] = p.tokens;
        j["neutralized_literals"] = p.neutralized_literals;
        res.set_content(dump(j), "application/json");
    }

    void models(httplib::Response& res) const {
        const auto& mi = engine.model();
        const Json entry = {{"id", model_name},
                            {"object", "model"},
                            {"type", "model"},
                            {"created", 0},
                            {"created_at", "1970-01-01T00:00:00Z"},
                            {"owned_by", "halo"},
                            {"display_name", model_name},
                            {"architecture", mi.architecture},
                            {"context_length", mi.context_length},
                            {"has_mtp", mi.has_mtp}};
        const Json j = {{"object", "list"},
                        {"data", Json::array({entry})},
                        {"has_more", false},
                        {"first_id", model_name},
                        {"last_id", model_name}};
        res.set_content(dump(j), "application/json");
    }

    // -- wiring -----------------------------------------------------------------------------

    void setup() {
        const std::size_t threads = cfg.http_threads != 0 ? cfg.http_threads : cfg.max_concurrent + cfg.max_queue + 4;
        // httplib takes ownership of the returned queue (library contract: raw pointer).
        svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads, threads, 64); };
        svr.set_payload_max_length(cfg.max_body_bytes);
        svr.set_read_timeout(cfg.read_timeout);
        svr.set_write_timeout(cfg.write_timeout);
        svr.set_keep_alive_max_count(100);
        svr.set_keep_alive_timeout(cfg.keep_alive_timeout);

        svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
            const ApiFamily fam = family_of(req.path);
            // 1. Host (DNS rebinding, A-6): a page on attacker.example that rebinds its name to
            //    127.0.0.1 still sends "Host: attacker.example".
            if (check_host) {
                const std::string h = host_name(req.get_header_value("Host"));
                if (h.empty() || std::ranges::find(allowed_hosts, h) == allowed_hosts.end()) {
                    metrics.count_rejection("host");
                    send_error(res, fam, {ErrorKind::Forbidden, "Host header is not one of this server's names",
                                          std::nullopt, std::nullopt});
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            // 2. Origin: browsers send it on cross-origin requests, including "simple" POSTs
            //    that skip preflight; an Origin that is not configured is refused on every
            //    route so the request never runs.
            const std::string origin = req.get_header_value("Origin");
            const bool cors = !origin.empty() && origin_allowed(origin);
            if (!origin.empty() && !cors) {
                metrics.count_rejection("origin");
                send_error(res, fam,
                           {ErrorKind::Forbidden, "cross-origin requests are not allowed", std::nullopt, std::nullopt});
                return httplib::Server::HandlerResponse::Handled;
            }
            if (cors) {
                res.set_header("Access-Control-Allow-Origin", origin);
                res.set_header("Vary", "Origin");
            }
            if (req.method == "OPTIONS") {
                if (!cors) {
                    send_error(res, fam,
                               {ErrorKind::Forbidden, "cross-origin requests are not allowed", std::nullopt, std::nullopt});
                    return httplib::Server::HandlerResponse::Handled;
                }
                res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                res.set_header("Access-Control-Allow-Headers",
                               "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta");
                res.set_header("Access-Control-Max-Age", "600");
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
            if (cfg.api_key && !cfg.api_key->empty() && req.path != "/health" && !authorized(req)) {
                metrics.count_rejection("unauthorized");
                res.set_header("WWW-Authenticate", "Bearer");
                send_error(res, fam,
                           {ErrorKind::Authentication, "missing or invalid API key", std::nullopt, std::nullopt});
                return httplib::Server::HandlerResponse::Handled;
            }
            // 3. POST bodies must be JSON: blocks HTML-form CSRF (text/plain, form-urlencoded).
            if (req.method == "POST") {
                const std::string ct = req.get_header_value("Content-Type");
                const std::string media = lower(std::string(trim(std::string_view(ct).substr(0, ct.find(';')))));
                if (media != "application/json") {
                    metrics.count_rejection("content_type");
                    send_error(res, fam, {ErrorKind::MediaType, "Content-Type must be application/json", std::nullopt,
                                          std::nullopt});
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        svr.set_error_handler([this](const httplib::Request& req, httplib::Response& res) {
            if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
            ApiErrorInfo e{ErrorKind::Server, "request failed", std::nullopt, std::nullopt};
            switch (res.status) {
                case 404:
                    e = {ErrorKind::NotFound, "no route for " + req.method + " " + sanitize_for_log(req.path, 256),
                         std::nullopt, std::nullopt};
                    break;
                case 405: e = {ErrorKind::NotFound, "method not allowed", std::nullopt, std::nullopt}; break;
                case 413:
                    metrics.count_rejection("too_large");
                    e = {ErrorKind::TooLarge, std::format("request body exceeds {} bytes", cfg.max_body_bytes), std::nullopt,
                         std::nullopt};
                    break;
                case 400: e = {ErrorKind::InvalidRequest, "malformed HTTP request", std::nullopt, std::nullopt}; break;
                case 415:
                    e = {ErrorKind::MediaType, "unsupported Content-Encoding (compressed request bodies are not accepted)",
                         std::nullopt, std::nullopt};
                    break;
                case 414:
                case 431: e = {ErrorKind::TooLarge, "request line or headers too large", std::nullopt, std::nullopt}; break;
                default: break;
            }
            const int status = res.status;
            send_error(res, family_of(req.path), e);
            res.status = status;
            return httplib::Server::HandlerResponse::Handled;
        });

        svr.set_exception_handler([this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            std::string what = "unknown";
            try {
                if (ep) std::rethrow_exception(ep);
            } catch (const std::exception& e) {
                what = e.what();
            } catch (...) {
            }
            HALO_ERROR(kLog, "unhandled exception on {} {}: {}", req.method, sanitize_for_log(req.path, 256),
                       sanitize_for_log(what));
            send_error(res, family_of(req.path), {ErrorKind::Server, "internal error", std::nullopt, std::nullopt});
        });

        svr.set_logger([this](const httplib::Request& req, const httplib::Response& res) {
            metrics.count_request(route_label(req.path), res.status);
        });

        svr.Get("/health", guarded([this](const httplib::Request&, httplib::Response& res) {
            if (stopping.load()) {
                res.status = 503;
                res.set_content(R"({"status":"stopping"})", "application/json");
            } else {
                res.set_content(R"({"status":"ok"})", "application/json");
            }
        }));
        svr.Get("/v1/models", guarded([this](const httplib::Request&, httplib::Response& res) { models(res); }));
        svr.Get("/metrics", guarded([this](const httplib::Request&, httplib::Response& res) {
            res.set_content(metrics.render(engine.stats(), admission.queued(), admission.active()),
                            "text/plain; version=0.0.4; charset=utf-8");
        }));
        svr.Post("/v1/chat/completions", guarded([this](const httplib::Request& q, httplib::Response& r) {
            chat(q, r, ApiFamily::OpenAI);
        }));
        svr.Post("/v1/messages", guarded([this](const httplib::Request& q, httplib::Response& r) {
            chat(q, r, ApiFamily::Anthropic);
        }));
        svr.Post("/v1/completions", guarded([this](const httplib::Request& q, httplib::Response& r) { completion(q, r); }));
        svr.Post("/tokenize", guarded([this](const httplib::Request& q, httplib::Response& r) { tokenize(q, r); }));
        svr.Post("/apply-template",
                 guarded([this](const httplib::Request& q, httplib::Response& r) { apply_template(q, r); }));
    }
};

// ---- ApiServer -----------------------------------------------------------------------------

namespace {
ServerConfig validated(ServerConfig c) {
    const bool no_key = !c.api_key || c.api_key->empty();
    HALO_CHECK(is_loopback_host(c.host) || !no_key || c.allow_unauthenticated_remote, ErrorCode::Config,
               "server.host {} is not a loopback address: an api_key is required (or set "
               "allow_unauthenticated_remote explicitly)",
               c.host);
    HALO_CHECK(c.max_concurrent > 0, ErrorCode::Config, "server.max_concurrent must be > 0");
    HALO_CHECK(c.max_body_bytes > 0, ErrorCode::Config, "server.max_body_bytes must be > 0");
    HALO_CHECK(c.max_tokens_cap > 0, ErrorCode::Config, "server.max_tokens_cap must be > 0");
    HALO_CHECK(c.default_max_tokens > 0, ErrorCode::Config, "server.default_max_tokens must be > 0");
    HALO_CHECK(c.max_json_depth >= 4, ErrorCode::Config, "server.max_json_depth must be >= 4");
    HALO_CHECK(c.port >= 0 && c.port <= 65535, ErrorCode::Config, "server.port must be 0-65535");
    HALO_CHECK(!c.host.empty(), ErrorCode::Config, "server.host must not be empty");
    return c;
}
}  // namespace

bool is_loopback_host(std::string_view host) noexcept {
    if (host.starts_with('[') && host.ends_with(']')) host = host.substr(1, host.size() - 2);
    if (host == "localhost" || host == "::1") return true;
    // Dotted quad 127.a.b.c: exactly four groups of 1-3 digits, each <= 255.
    if (!host.starts_with("127.")) return false;
    int groups = 0;
    std::size_t i = 0;
    while (i <= host.size()) {
        std::size_t j = i;
        int value = 0;
        while (j < host.size() && host[j] >= '0' && host[j] <= '9' && j - i < 3) value = value * 10 + (host[j++] - '0');
        if (j == i || value > 255) return false;
        ++groups;
        if (j == host.size()) break;
        if (host[j] != '.') return false;
        i = j + 1;
    }
    return groups == 4;
}

ApiServer::ApiServer(runtime::Engine& engine, ServerConfig config)
    : impl_(std::make_unique<Impl>(engine, validated(std::move(config)))) {
    if (!is_loopback_host(impl_->cfg.host) && (!impl_->cfg.api_key || impl_->cfg.api_key->empty())) {
        HALO_WARN(kLog, "binding {} without an API key: anyone who can reach this address can use the model",
                  impl_->cfg.host);
    }
}

ApiServer::~ApiServer() { stop(); }

int ApiServer::bind() {
    std::lock_guard lk(impl_->ctl);
    if (impl_->port >= 0) return impl_->port;
    auto& c = impl_->cfg;
    if (c.port == 0) {
        impl_->port = impl_->svr.bind_to_any_port(c.host);
    } else {
        impl_->port = impl_->svr.bind_to_port(c.host, c.port) ? c.port : -1;
    }
    HALO_CHECK(impl_->port > 0, ErrorCode::Io, "cannot bind {}:{}", c.host, c.port);
    HALO_INFO(kLog, "listening on http://{}:{}", c.host, impl_->port);
    return impl_->port;
}

void ApiServer::listen() {
    (void)bind();
    impl_->svr.listen_after_bind();
}

void ApiServer::start() {
    (void)bind();
    std::lock_guard lk(impl_->ctl);
    if (impl_->thread.joinable()) return;
    impl_->thread = std::thread([this] { impl_->svr.listen_after_bind(); });
    impl_->svr.wait_until_ready();
}

void ApiServer::stop() {
    std::lock_guard lk(impl_->ctl);
    if (impl_->stopped) return;
    impl_->stopped = true;
    impl_->stopping.store(true);
    impl_->admission.shutdown();
    impl_->svr.stop();
    if (impl_->thread.joinable()) impl_->thread.join();
}

int ApiServer::port() const noexcept { return impl_->port; }
const ServerConfig& ApiServer::config() const noexcept { return impl_->cfg; }

}  // namespace halo::api
