// Fuzz target: an HTTP request body through parse -> OpenAI/Anthropic validation -> chat
// template render -> tokenize, against the synthetic FakeEngine (no model, no network).
// Invariant: every input ends as a prompt with tokens, an api::RequestError, or a
// halo::Error with code Api; anything else (other exception, crash, sanitizer report) is a
// bug. Seeds: tests/fuzz/corpus/api_request (both APIs, tools, tool history, control tokens).

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>

#include "api/generation.h"
#include "api/json_util.h"
#include "api/requests.h"
#include "fake_engine.h"
#include "fuzz_check.h"
#include "halo/api/prompt.h"
#include "halo/core/error.h"

namespace {

struct Env {
    std::unique_ptr<halo::test::FakeEngine> eng = halo::test::FakeEngine::synthetic();
    halo::api::SpecialTokens sp{eng->tokenizer()};
    halo::api::ServerConfig cfg = [] {
        halo::api::ServerConfig c;
        c.max_content_bytes = 1u << 20;
        return c;
    }();
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static Env env;  // built once per process: the engine and tokenizer are read-only here
    const std::string body_text(reinterpret_cast<const char*>(data), size);
    try {
        const auto body = halo::api::parse_request_body(body_text, env.cfg.max_json_depth);
        const bool anthropic = body.is_object() && (body.contains("system") || body.contains("thinking"));
        const halo::api::ChatJob job = anthropic ? halo::api::parse_anthropic_messages(body, env.cfg)
                                                 : halo::api::parse_openai_chat(body, env.cfg);
        const auto p = halo::api::build_chat_prompt(env.eng->tokenizer(), env.sp, env.eng->chat_template(),
                                                    job.messages, job.tools, job.render);
        if (p.tokens.empty()) HALO_FUZZ_FAIL("accepted request rendered an empty prompt");
    } catch (const halo::api::RequestError&) {
        // client error (400-class): expected
    } catch (const halo::Error& e) {
        if (e.code() != halo::ErrorCode::Api) HALO_FUZZ_FAIL("halo::Error with a non-Api code: %s", e.what());
    } catch (const std::exception& e) {
        HALO_FUZZ_FAIL("untyped exception escaped the request path: %s", e.what());
    }
    return 0;
}
