// Functional tests of the HTTP API over the fake Engine: every endpoint, non-streaming and
// SSE streaming for both API families, tool calls, structured-output passthrough, the
// reasoning budget and stop sequences. Response shapes are checked against the hand-written
// fixtures in fixtures/ (see fixtures/README.md).

#include <thread>

#include "api_test_util.h"

namespace halo::test {
namespace {

Script sc(std::string text, std::chrono::milliseconds delay = std::chrono::milliseconds(0), bool eos = true) {
    Script s;
    s.text = std::move(text);
    s.delay = delay;
    s.eos = eos;
    return s;
}

Json user_chat(const std::string& content) {
    return Json{{"model", "m"}, {"messages", Json::array({Json{{"role", "user"}, {"content", content}}})}};
}

Json user_messages(const std::string& content, int max_tokens = 256) {
    return Json{{"model", "m"},
                {"max_tokens", max_tokens},
                {"messages", Json::array({Json{{"role", "user"}, {"content", content}}})}};
}

const char* kToolOutput =
    "I should call the tool.</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>";

Json weather_tools_openai() {
    return Json::parse(R"([{"type":"function","function":{"name":"get_weather","description":"Weather for a city",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])");
}

Json weather_tools_anthropic() {
    return Json::parse(R"([{"name":"get_weather","description":"Weather for a city",
        "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}])");
}

/// POSTs and collects a streamed body.
struct Streamed {
    int status = 0;
    std::string body;
    httplib::Headers headers;
};
Streamed post_stream(const TestServer& ts, const std::string& path, const Json& body, httplib::Headers h = {}) {
    Streamed s;
    auto c = ts.client();
    auto r = c->Post(path, h, body.dump(), "application/json", [&](const char* d, std::size_t n) {
        s.body.append(d, n);
        return true;
    });
    if (r) {
        s.status = r->status;
        s.headers = r->headers;
    }
    return s;
}

std::vector<Json> data_events(const std::string& body, bool& saw_done) {
    std::vector<Json> out;
    saw_done = false;
    for (const auto& e : parse_sse(body)) {
        EXPECT_FALSE(saw_done) << "event after [DONE]";
        if (e.data == "[DONE]") {
            saw_done = true;
            continue;
        }
        out.push_back(Json::parse(e.data));
    }
    return out;
}

// ---- utility routes ------------------------------------------------------------------------

TEST(ApiServer, HealthModelsMetrics) {
    TestServer ts;
    auto h = ts.get("/health");
    ASSERT_TRUE(h) << httplib::to_string(h.error());
    EXPECT_EQ(h->status, 200);
    EXPECT_EQ(Json::parse(h->body).at("status"), "ok");

    auto m = ts.get("/v1/models");
    ASSERT_TRUE(m);
    EXPECT_EQ(m->status, 200);
    const Json mj = Json::parse(m->body);
    EXPECT_EQ(shape_mismatch(load_fixture("openai_models.json"), mj), "");
    EXPECT_EQ(mj.at("data").at(0).at("id"), "fake-model");

    ASSERT_EQ(ts.post("/v1/chat/completions", user_chat("hi"))->status, 200);
    // The request counter is updated by httplib's logger after the response is written, so
    // it can trail the client by a moment.
    httplib::Result x;
    for (int i = 0; i < 200; ++i) {
        x = ts.get("/metrics");
        ASSERT_TRUE(x);
        if (x->body.find("route=\"/v1/chat/completions\"") != std::string::npos) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(x->status, 200);
    EXPECT_NE(x->get_header_value("Content-Type").find("text/plain"), std::string::npos);
    for (const char* needle :
         {"# TYPE halo_api_requests_total counter", "halo_api_requests_total{route=\"/v1/chat/completions\",status=\"200\"} 1",
          "halo_api_time_to_first_token_seconds_count 1", "halo_engine_prefix_cache_hit_ratio 0.75",
          "halo_engine_speculative_acceptance_ratio 0.7", "halo_api_completion_tokens_total ",
          "halo_api_queued_requests 0"}) {
        EXPECT_NE(x->body.find(needle), std::string::npos) << needle << "\n" << x->body;
    }
}

TEST(ApiServer, ServedModelNameOverride) {
    api::ServerConfig cfg;
    cfg.served_model_name = "custom";
    TestServer ts(cfg);
    EXPECT_EQ(Json::parse(ts.get("/v1/models")->body).at("data").at(0).at("id"), "custom");
    EXPECT_EQ(Json::parse(ts.post("/v1/chat/completions", user_chat("hi"))->body).at("model"), "custom");
}

TEST(ApiServer, TokenizeAndApplyTemplate) {
    TestServer ts;
    const auto& tok = ts.engine->tokenizer();
    const auto im_end = tok.piece_to_id("<|im_end|>").value();
    auto r = ts.post("/tokenize", Json{{"content", "a<|im_end|>"}});
    ASSERT_EQ(r->status, 200);
    Json j = Json::parse(r->body);
    EXPECT_EQ(std::ranges::count(j.at("tokens"), Json(im_end)), 0) << "parse_special defaults to false";
    r = ts.post("/tokenize", Json{{"content", "a<|im_end|>"}, {"parse_special", true}, {"with_pieces", true}});
    j = Json::parse(r->body);
    ASSERT_EQ(j.at("tokens").size(), 2u);
    EXPECT_EQ(j.at("tokens").at(1).at("id"), im_end);
    EXPECT_EQ(j.at("tokens").at(1).at("piece"), "<|im_end|>");

    Json at = user_chat("hello");
    at["tokenize"] = true;
    at["add_generation_prompt"] = false;
    r = ts.post("/apply-template", at);
    ASSERT_EQ(r->status, 200) << r->body;
    j = Json::parse(r->body);
    EXPECT_EQ(j.at("prompt"), "<|im_start|>user\nhello<|im_end|>\n");
    EXPECT_EQ(j.at("tokens").get<std::vector<std::int32_t>>(), tok.encode("<|im_start|>user\nhello<|im_end|>\n", true));
    EXPECT_EQ(j.at("neutralized_literals"), 0);
}

// ---- OpenAI chat ---------------------------------------------------------------------------

TEST(ApiOpenAI, ChatNonStreamingSplitsReasoningAndContent) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("Let me think.</think>\n\nHello."); };
    auto r = ts.post("/v1/chat/completions", user_chat("hi"));
    ASSERT_TRUE(r);
    ASSERT_EQ(r->status, 200) << r->body;
    EXPECT_NE(r->get_header_value("Content-Type").find("application/json"), std::string::npos);
    const Json j = Json::parse(r->body);
    EXPECT_EQ(shape_mismatch(load_fixture("openai_chat_completion.json"), j), "");
    EXPECT_TRUE(j.at("id").get<std::string>().starts_with("chatcmpl-"));
    const Json& msg = j.at("choices").at(0).at("message");
    EXPECT_EQ(msg.at("content"), "Hello.");
    EXPECT_EQ(msg.at("reasoning_content"), "Let me think.");
    EXPECT_FALSE(msg.contains("tool_calls"));
    EXPECT_EQ(j.at("choices").at(0).at("finish_reason"), "stop");
    // Byte-level tokenizer: "Let me think." (13) + </think> = 14 reasoning tokens;
    // + "\n\nHello." (8) + EOS = 23 generated.
    EXPECT_EQ(j.at("usage").at("completion_tokens_details").at("reasoning_tokens"), 14);
    EXPECT_EQ(j.at("usage").at("completion_tokens"), 23);
    const auto calls = ts.engine->calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(j.at("usage").at("prompt_tokens"), calls.at(0).request.prompt.size());
    EXPECT_EQ(ts.engine->prompt_text(0), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
    EXPECT_TRUE(calls.at(0).request.stop_strings.empty()) << "stop sequences are applied by the API layer";
}

TEST(ApiOpenAI, SamplingParametersReachTheEngine) {
    TestServer ts;
    Json b = user_chat("hi");
    b["temperature"] = 0.25;
    b["top_p"] = 0.5;
    b["top_k"] = 7;
    b["min_p"] = 0.1;
    b["seed"] = 99;
    b["presence_penalty"] = 1.5;
    b["max_completion_tokens"] = 33;
    ASSERT_EQ(ts.post("/v1/chat/completions", b)->status, 200);
    const auto calls = ts.engine->calls();
    const auto& s = calls.at(0).request;
    EXPECT_FLOAT_EQ(s.sampling.temperature, 0.25f);
    EXPECT_FLOAT_EQ(s.sampling.top_p, 0.5f);
    EXPECT_EQ(s.sampling.top_k, 7);
    EXPECT_FLOAT_EQ(s.sampling.min_p, 0.1f);
    EXPECT_EQ(s.sampling.seed, 99u);
    EXPECT_FLOAT_EQ(s.sampling.presence_penalty, 1.5f);
    EXPECT_EQ(s.max_tokens, 33u);
}

TEST(ApiOpenAI, ChatStreamingEventSequence) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("Hmm.</think>\n\nHello there."); };
    Json b = user_chat("hi");
    b["stream"] = true;
    b["stream_options"] = {{"include_usage", true}};
    const auto s = post_stream(ts, "/v1/chat/completions", b);
    ASSERT_EQ(s.status, 200) << s.body;
    bool done = false;
    const auto ev = data_events(s.body, done);
    EXPECT_TRUE(done) << "stream must end with data: [DONE]";
    ASSERT_GE(ev.size(), 4u);
    const Json chunk_shape = load_fixture("openai_chat_chunk.json");
    std::string reasoning, content;
    const std::string id = ev.at(0).at("id");
    EXPECT_EQ(ev.at(0).at("choices").at(0).at("delta").at("role"), "assistant");
    for (std::size_t i = 0; i + 1 < ev.size(); ++i) {
        EXPECT_EQ(shape_mismatch(chunk_shape, ev[i]), "") << ev[i].dump();
        EXPECT_EQ(ev[i].at("id"), id);
        const Json& d = ev[i].at("choices").at(0).at("delta");
        if (d.contains("reasoning_content")) reasoning += d.at("reasoning_content").get<std::string>();
        if (d.contains("content")) content += d.at("content").get<std::string>();
        if (i + 2 < ev.size()) EXPECT_TRUE(ev[i].at("choices").at(0).at("finish_reason").is_null());
    }
    EXPECT_EQ(reasoning, "Hmm.");
    EXPECT_EQ(content, "Hello there.");
    EXPECT_EQ(ev[ev.size() - 2].at("choices").at(0).at("finish_reason"), "stop");
    // include_usage: a final chunk with empty choices and the usage object.
    const Json& u = ev.back();
    EXPECT_TRUE(u.at("choices").empty());
    EXPECT_EQ(u.at("usage").at("completion_tokens_details").at("reasoning_tokens"), 5);
    EXPECT_EQ(s.headers.find("Content-Type")->second.find("text/event-stream"), 0u);
}

TEST(ApiOpenAI, ToolCallNonStreamingAndStreaming) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc(kToolOutput); };
    Json b = user_chat("weather in Paris?");
    b["tools"] = weather_tools_openai();
    auto r = ts.post("/v1/chat/completions", b);
    ASSERT_EQ(r->status, 200) << r->body;
    Json j = Json::parse(r->body);
    const Json& msg = j.at("choices").at(0).at("message");
    EXPECT_EQ(j.at("choices").at(0).at("finish_reason"), "tool_calls");
    EXPECT_TRUE(msg.at("content").is_null()) << "content is null when only tool calls were produced";
    ASSERT_EQ(msg.at("tool_calls").size(), 1u);
    EXPECT_EQ(shape_mismatch(load_fixture("openai_tool_call.json"), msg.at("tool_calls").at(0)), "");
    EXPECT_EQ(msg.at("tool_calls").at(0).at("function").at("name"), "get_weather");
    EXPECT_EQ(Json::parse(msg.at("tool_calls").at(0).at("function").at("arguments").get<std::string>()), Json({{"city", "Paris"}}));
    EXPECT_TRUE(msg.at("tool_calls").at(0).at("id").get<std::string>().starts_with("call_"));
    // The tools reached the template.
    EXPECT_NE(ts.engine->prompt_text(0).find("\"get_weather\""), std::string::npos);

    b["stream"] = true;
    const auto s = post_stream(ts, "/v1/chat/completions", b);
    bool done = false;
    const auto ev = data_events(s.body, done);
    EXPECT_TRUE(done);
    std::vector<Json> deltas;
    for (const auto& e : ev) {
        const Json& d = e.at("choices").at(0).at("delta");
        if (d.contains("tool_calls")) deltas.push_back(d.at("tool_calls").at(0));
    }
    ASSERT_EQ(deltas.size(), 1u);
    EXPECT_EQ(deltas.at(0).at("index"), 0);
    EXPECT_EQ(deltas.at(0).at("function").at("name"), "get_weather");
    EXPECT_EQ(Json::parse(deltas.at(0).at("function").at("arguments").get<std::string>()), Json({{"city", "Paris"}}));
    EXPECT_EQ(ev.back().at("choices").at(0).at("finish_reason"), "tool_calls");
}

TEST(ApiOpenAI, ToolRoundTripRendersHistory) {
    TestServer ts;
    Json b = user_chat("weather in Paris?");
    b["tools"] = weather_tools_openai();
    b.at("messages").push_back(Json::parse(R"({"role":"assistant","content":null,"tool_calls":[{"id":"call_1",
        "type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]})"));
    b.at("messages").push_back(Json::parse(R"({"role":"tool","tool_call_id":"call_1","content":"Sunny, 21C"})"));
    auto r = ts.post("/v1/chat/completions", b);
    ASSERT_EQ(r->status, 200) << r->body;
    const std::string p = ts.engine->prompt_text(0);
    EXPECT_NE(p.find("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>"), std::string::npos) << p;
    EXPECT_NE(p.find("<tool_response>\nSunny, 21C\n</tool_response>"), std::string::npos) << p;
}

TEST(ApiOpenAI, StructuredOutputPassthrough) {
    TestServer ts;
    Json b = user_chat("json please");
    const Json schema = Json::parse(R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})");
    b["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", schema}, {"strict", true}}}};
    ASSERT_EQ(ts.post("/v1/chat/completions", b)->status, 200);
    b["response_format"] = {{"type", "json_object"}};
    ASSERT_EQ(ts.post("/v1/chat/completions", b)->status, 200);
    b["response_format"] = {{"type", "text"}};
    ASSERT_EQ(ts.post("/v1/chat/completions", b)->status, 200);
    b["response_format"] = {{"type", "xml"}};
    EXPECT_EQ(ts.post("/v1/chat/completions", b)->status, 400);
    const auto calls = ts.engine->calls();
    ASSERT_EQ(calls.size(), 3u);
    EXPECT_EQ(Json::parse(calls.at(0).request.sampling.json_schema.value()), schema);
    EXPECT_FALSE(calls.at(0).request.sampling.json_object);
    EXPECT_TRUE(calls.at(1).request.sampling.json_object);
    EXPECT_FALSE(calls.at(1).request.sampling.json_schema.has_value());
    EXPECT_FALSE(calls.at(2).request.sampling.json_object);
    EXPECT_FALSE(calls.at(2).request.sampling.json_schema.has_value());
}

TEST(ApiOpenAI, StopSequencesTruncateContent) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("r</think>\n\nHello. World."); };
    Json b = user_chat("hi");
    b["stop"] = Json::array({"lo.", "zzz"});
    Json j = Json::parse(ts.post("/v1/chat/completions", b)->body);
    EXPECT_EQ(j.at("choices").at(0).at("message").at("content"), "Hel");
    EXPECT_EQ(j.at("choices").at(0).at("finish_reason"), "stop");
    EXPECT_TRUE(ts.engine->calls().at(0).cancelled) << "a matched stop sequence ends the engine call";
    // Streaming never leaks a partial stop sequence.
    b["stream"] = true;
    const auto s = post_stream(ts, "/v1/chat/completions", b);
    bool done = false;
    std::string content;
    for (const auto& e : data_events(s.body, done)) {
        const Json& d = e.at("choices").at(0).at("delta");
        if (d.contains("content")) content += d.at("content").get<std::string>();
    }
    EXPECT_EQ(content, "Hel");
}

TEST(ApiOpenAI, ReasoningEffortNoneDisablesThinking) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("Hello."); };
    Json b = user_chat("hi");
    b["reasoning_effort"] = "none";
    Json j = Json::parse(ts.post("/v1/chat/completions", b)->body);
    EXPECT_EQ(j.at("choices").at(0).at("message").at("content"), "Hello.");
    EXPECT_FALSE(j.at("choices").at(0).at("message").contains("reasoning_content"));
    EXPECT_EQ(j.at("usage").at("completion_tokens_details").at("reasoning_tokens"), 0);
    EXPECT_TRUE(ts.engine->prompt_text(0).ends_with("<think>\n\n</think>\n\n"));
}

TEST(ApiOpenAI, ReasoningBudgetClosesThinkBlockAndLeavesHeadroom) {
    // TRD §25: with max_tokens 40 and no explicit budget, reasoning may use 40 - min(512, 20)
    // = 20 tokens; then HALO closes the think block and continues with the rest.
    TestServer ts;
    ts.engine->script = [](const auto&, int call) {
        return call == 0 ? sc(std::string(100, 'r'), std::chrono::milliseconds(0), false) : sc("Answer.");
    };
    Json b = user_chat("hi");
    b["max_tokens"] = 40;
    auto r = ts.post("/v1/chat/completions", b);
    ASSERT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    const auto calls = ts.engine->calls();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_TRUE(calls.at(0).cancelled);
    EXPECT_EQ(calls.at(0).request.max_tokens, 40u);
    EXPECT_EQ(calls.at(1).request.max_tokens, 20u);
    // The continuation context is prompt + the 20 reasoning tokens + "\n</think>\n\n".
    const auto& tok = ts.engine->tokenizer();
    auto expect_prompt = calls.at(0).request.prompt;
    for (int i = 0; i < 20; ++i) expect_prompt.push_back(tok.encode("r", false).at(0));
    for (const auto t : tok.encode("\n</think>\n\n", true)) expect_prompt.push_back(t);
    EXPECT_EQ(calls.at(1).request.prompt, expect_prompt);
    EXPECT_EQ(j.at("choices").at(0).at("message").at("content"), "Answer.");
    EXPECT_EQ(j.at("choices").at(0).at("message").at("reasoning_content"), std::string(20, 'r'));
    EXPECT_EQ(j.at("usage").at("completion_tokens_details").at("reasoning_tokens"), 20);
    ASSERT_TRUE(j.contains("warnings"));
    EXPECT_NE(j.value("warnings", Json::array()).dump().find("reasoning budget"), std::string::npos);
    EXPECT_NE(ts.get("/metrics")->body.find("halo_api_reasoning_budget_closes_total 1"), std::string::npos);
}

TEST(ApiOpenAI, EmptyContentDiagnostic) {
    // Reasoning consumed the budget and nothing was left for the answer: the response says
    // so instead of returning an unexplained empty string.
    TestServer ts;
    ts.engine->script = [](const auto&, int call) {
        return call == 0 ? sc(std::string(50, 'r'), std::chrono::milliseconds(0), false)
                         : sc("", std::chrono::milliseconds(0), false);
    };
    Json b = user_chat("hi");
    b["max_tokens"] = 10;
    const Json j = Json::parse(ts.post("/v1/chat/completions", b)->body);
    EXPECT_EQ(j.at("choices").at(0).at("message").at("content"), "");
    EXPECT_EQ(j.at("choices").at(0).at("finish_reason"), "length");
    EXPECT_NE(j.value("warnings", Json::array()).dump().find("no content was produced"), std::string::npos) << j.dump();
}

TEST(ApiOpenAI, MaxTokensDefaultsAndContextClamp) {
    api::ServerConfig cfg;
    cfg.default_max_tokens = 100;
    TestServer ts(cfg, FakeEngine::synthetic(/*context_length=*/64));
    ASSERT_EQ(ts.post("/v1/chat/completions", user_chat("hi"))->status, 200);
    const auto n = ts.engine->calls().at(0).request.prompt.size();
    EXPECT_EQ(ts.engine->calls().at(0).request.max_tokens, 64 - n) << "clamped to the context";
    Json b = user_chat("hi");
    b["max_tokens"] = 60;
    const Json j = Json::parse(ts.post("/v1/chat/completions", b)->body);
    EXPECT_NE(j.value("warnings", Json::array()).dump().find("max_tokens reduced"), std::string::npos);
    auto r = ts.post("/v1/chat/completions", user_chat(std::string(80, 'x')));
    EXPECT_EQ(r->status, 400);
    EXPECT_EQ(Json::parse(r->body).at("error").at("code"), "context_length_exceeded");
}

// ---- OpenAI completions --------------------------------------------------------------------

TEST(ApiOpenAI, CompletionsRawTextBothModes) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("</think> raw <tool_call>"); };
    auto r = ts.post("/v1/completions", Json{{"model", "m"}, {"prompt", "Once upon"}, {"max_tokens", 50}});
    ASSERT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    EXPECT_EQ(shape_mismatch(load_fixture("openai_completion.json"), j), "");
    EXPECT_EQ(j.at("choices").at(0).at("text"), "</think> raw <tool_call>") << "no output parsing on /v1/completions";
    EXPECT_EQ(ts.engine->prompt_text(0), "Once upon");

    const auto s = post_stream(ts, "/v1/completions", Json{{"prompt", "x"}, {"stream", true}});
    bool done = false;
    std::string text;
    const auto ev = data_events(s.body, done);
    EXPECT_TRUE(done);
    for (const auto& e : ev) {
        EXPECT_EQ(e.at("object"), "text_completion");
        text += e.at("choices").at(0).at("text").get<std::string>();
    }
    EXPECT_EQ(text, "</think> raw <tool_call>");
    EXPECT_EQ(ev.back().at("choices").at(0).at("finish_reason"), "stop");
}

// ---- Anthropic -----------------------------------------------------------------------------

TEST(ApiAnthropic, MessagesNonStreaming) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("Thinking.</think>\n\nHi!"); };
    Json b = user_messages("hello");
    b["system"] = "Be kind.";
    auto r = ts.post("/v1/messages", b, {{"anthropic-version", "2023-06-01"}});
    ASSERT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    EXPECT_EQ(shape_mismatch(load_fixture("anthropic_message.json"), j), "");
    const Json blocks = load_fixture("anthropic_blocks.json");
    ASSERT_EQ(j.at("content").size(), 2u);
    EXPECT_EQ(shape_mismatch(blocks.at("thinking"), j.at("content").at(0)), "");
    EXPECT_EQ(shape_mismatch(blocks.at("text"), j.at("content").at(1)), "");
    EXPECT_EQ(j.at("content").at(0).at("thinking"), "Thinking.");
    EXPECT_EQ(j.at("content").at(1).at("text"), "Hi!");
    EXPECT_EQ(j.at("stop_reason"), "end_turn");
    EXPECT_TRUE(j.at("id").get<std::string>().starts_with("msg_"));
    EXPECT_EQ(j.at("usage").at("input_tokens"), ts.engine->calls().at(0).request.prompt.size());
    EXPECT_TRUE(ts.engine->prompt_text(0).starts_with("<|im_start|>system\nBe kind.<|im_end|>\n"));
}

TEST(ApiAnthropic, MessagesStreamingEventSequence) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc("Plan.</think>\n\nDone here."); };
    Json b = user_messages("hello");
    b["stream"] = true;
    const auto s = post_stream(ts, "/v1/messages", b);
    ASSERT_EQ(s.status, 200) << s.body;
    const auto ev = parse_sse(s.body);
    const Json fx = load_fixture("anthropic_stream.json");
    std::vector<std::string> names;
    std::string thinking, text;
    for (const auto& e : ev) {
        names.push_back(e.event);
        const Json d = Json::parse(e.data);
        EXPECT_EQ(d.at("type"), e.event) << "event name and data.type agree";
        if (fx.contains(e.event)) EXPECT_EQ(shape_mismatch(fx[e.event], d), "") << e.data;
        if (e.event == "content_block_delta") {
            if (d.at("delta").at("type") == "thinking_delta") thinking += d.at("delta").at("thinking").get<std::string>();
            if (d.at("delta").at("type") == "text_delta") text += d.at("delta").at("text").get<std::string>();
        }
    }
    ASSERT_GE(names.size(), 8u);
    EXPECT_EQ(names.front(), "message_start");
    EXPECT_EQ(names.at(1), "content_block_start");
    EXPECT_EQ(names[names.size() - 2], "message_delta");
    EXPECT_EQ(names.back(), "message_stop");
    EXPECT_EQ(thinking, "Plan.");
    EXPECT_EQ(text, "Done here.");
    // Block indices: 0 = thinking, 1 = text; every start has a matching stop.
    std::vector<int> starts, stops;
    for (const auto& e : ev) {
        const Json d = Json::parse(e.data);
        if (e.event == "content_block_start") starts.push_back(d.at("index"));
        if (e.event == "content_block_stop") stops.push_back(d.at("index"));
    }
    EXPECT_EQ(starts, (std::vector<int>{0, 1}));
    EXPECT_EQ(stops, (std::vector<int>{0, 1}));
    EXPECT_EQ(Json::parse(ev[ev.size() - 2].data).at("delta").at("stop_reason"), "end_turn");
}

TEST(ApiAnthropic, ToolUseRoundTrip) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc(kToolOutput); };
    Json b = user_messages("weather?");
    b["tools"] = weather_tools_anthropic();
    auto r = ts.post("/v1/messages", b);
    ASSERT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    EXPECT_EQ(j.at("stop_reason"), "tool_use");
    const Json& last = j.at("content").back();
    EXPECT_EQ(shape_mismatch(load_fixture("anthropic_blocks.json").at("tool_use"), last), "");
    EXPECT_EQ(last.at("input"), Json({{"city", "Paris"}}));
    const std::string id = last.at("id");
    EXPECT_TRUE(id.starts_with("toolu_"));

    // Streaming: a tool_use block with an input_json_delta.
    b["stream"] = true;
    const auto s = post_stream(ts, "/v1/messages", b);
    std::string partial;
    bool saw_tool_start = false;
    for (const auto& e : parse_sse(s.body)) {
        const Json d = Json::parse(e.data);
        if (e.event == "content_block_start" && d.at("content_block").at("type") == "tool_use") {
            saw_tool_start = true;
            EXPECT_EQ(d.at("content_block").at("name"), "get_weather");
        }
        if (e.event == "content_block_delta" && d.at("delta").at("type") == "input_json_delta") {
            partial += d.at("delta").at("partial_json").get<std::string>();
        }
        if (e.event == "message_delta") EXPECT_EQ(d.at("delta").at("stop_reason"), "tool_use");
    }
    EXPECT_TRUE(saw_tool_start);
    EXPECT_EQ(Json::parse(partial), Json({{"city", "Paris"}}));

    // Round trip: tool_use in assistant history, tool_result from the user.
    Json b2 = user_messages("weather?");
    b2["tools"] = weather_tools_anthropic();
    b2.at("messages").push_back(Json{{"role", "assistant"}, {"content", Json::array({last})}});
    b2.at("messages").push_back(Json::parse(R"({"role":"user","content":[{"type":"tool_result","tool_use_id":")" + id +
                                         R"(","content":"Sunny"}]})"));
    ts.engine->script = [](const auto&, int) { return sc("It is sunny."); };
    ASSERT_EQ(ts.post("/v1/messages", b2)->status, 200);
    const std::string p = ts.engine->prompt_text(ts.engine->calls().size() - 1);
    EXPECT_NE(p.find("<function=get_weather>\n<parameter=city>\nParis"), std::string::npos) << p;
    EXPECT_NE(p.find("<tool_response>\nSunny\n</tool_response>"), std::string::npos) << p;
}

TEST(ApiAnthropic, ThinkingBudgetEffortAndStructuredOutput) {
    TestServer ts;
    ts.engine->script = [](const auto&, int call) {
        return call == 0 ? sc(std::string(30, 't'), std::chrono::milliseconds(0), false) : sc("ok");
    };
    Json b = user_messages("x", 64);
    b["thinking"] = {{"type", "enabled"}, {"budget_tokens", 8}};
    const Json schema = Json::parse(R"({"type":"object"})");
    b["output_config"] = {{"format", {{"type", "json_schema"}, {"schema", schema}}}};
    auto r = ts.post("/v1/messages", b);
    ASSERT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    EXPECT_EQ(j.at("content").at(0).at("thinking"), std::string(8, 't')) << "explicit budget_tokens is the reasoning budget";
    EXPECT_EQ(j.at("content").at(1).at("text"), "ok");
    EXPECT_EQ(Json::parse(ts.engine->calls().at(0).request.sampling.json_schema.value()), schema);

    b = user_messages("x");
    b["thinking"] = {{"type", "disabled"}};
    ts.engine->script = [](const auto&, int) { return sc("plain"); };
    ASSERT_EQ(ts.post("/v1/messages", b)->status, 200);
    EXPECT_TRUE(ts.engine->prompt_text(ts.engine->calls().size() - 1).ends_with("<think>\n\n</think>\n\n"));

    b = user_messages("x");
    b["stop_sequences"] = Json::array({"ai"});
    ts.engine->script = [](const auto&, int) { return sc("</think>\n\nplain"); };
    const Json s = Json::parse(ts.post("/v1/messages", b)->body);
    EXPECT_EQ(s.at("stop_reason"), "stop_sequence");
    EXPECT_EQ(s.at("stop_sequence"), "ai");
    EXPECT_EQ(s.at("content").at(0).at("text"), "pl");

    b = user_messages("x", 16);
    b["thinking"] = {{"type", "enabled"}, {"budget_tokens", 16}};
    EXPECT_EQ(ts.post("/v1/messages", b)->status, 400) << "budget_tokens must be below max_tokens";
    Json nomax = user_messages("x");
    nomax.erase("max_tokens");
    EXPECT_EQ(ts.post("/v1/messages", nomax)->status, 400) << "max_tokens is required by the Anthropic API";
}

TEST(ApiAnthropic, MaxTokensStopReason) {
    TestServer ts;
    ts.engine->script = [](const auto&, int) { return sc(std::string(100, 'a')); };
    Json b = user_messages("x", 5);
    b["thinking"] = {{"type", "disabled"}};
    const Json j = Json::parse(ts.post("/v1/messages", b)->body);
    EXPECT_EQ(j.at("stop_reason"), "max_tokens");
    EXPECT_EQ(j.at("usage").at("output_tokens"), 5);
}

TEST(ApiAnthropic, BudgetContinuationSkippedWhenContextIsFull) {
    // Review N-3. 64-token context, a 22-token prompt, so max_tokens is clamped to 42. The
    // explicit budget of 40 closes reasoning at 40 tokens; the continuation prompt
    // (22 + 40 + close) would not fit, so generation ends like max_tokens instead of sending
    // the engine an over-long prompt (which it rejects, and which used to become a 500).
    TestServer ts({}, FakeEngine::synthetic(/*context_length=*/64));
    ts.engine->script = [](const auto&, int) { return sc(std::string(100, 't'), std::chrono::milliseconds(0), false); };
    Json b = user_messages("x", 60);
    b["thinking"] = {{"type", "enabled"}, {"budget_tokens", 40}};
    auto r = ts.post("/v1/messages", b);
    ASSERT_EQ(r->status, 200) << r->body;
    const Json j = Json::parse(r->body);
    EXPECT_EQ(j.at("stop_reason"), "max_tokens");
    EXPECT_EQ(j.at("content").at(0).at("thinking"), std::string(40, 't'));
    EXPECT_NE(j.value("warnings", Json::array()).dump().find("context is full"), std::string::npos) << j.dump();
    const auto calls = ts.engine->calls();
    ASSERT_EQ(calls.size(), 1u) << "no continuation call";
    EXPECT_EQ(calls.at(0).request.prompt.size(), 22u);
    EXPECT_EQ(calls.at(0).request.max_tokens, 42u);
    // With room left, the continuation still happens.
    b["thinking"] = {{"type", "enabled"}, {"budget_tokens", 20}};
    ts.engine->script = [](const auto&, int call) {  // call index counts across requests
        return call == 1 ? sc(std::string(100, 't'), std::chrono::milliseconds(0), false) : sc("ok");
    };
    r = ts.post("/v1/messages", b);
    ASSERT_EQ(r->status, 200) << r->body;
    EXPECT_EQ(ts.engine->calls().size(), 3u);
    EXPECT_EQ(Json::parse(r->body).at("content").at(1).at("text"), "ok");
}

}  // namespace
}  // namespace halo::test
