#pragma once
// Test double for runtime::Engine (the real one is WS-G's). Deterministic: every generate()
// call asks `script` what to emit, tokenizes that text with parse_special=true, and feeds the
// tokens to the callback one by one (pieces built with StreamDecoder, like the real engine),
// optionally sleeping between tokens. It records every request and whether the callback
// asked it to stop, so tests can assert on what the API layer actually sent the model.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "halo/core/error.h"
#include "halo/runtime/engine.h"
#include "halo/template/chat_template.h"
#include "halo/tokenizer/tokenizer.h"

namespace halo::test {

struct Script {
    std::string text;                        ///< emitted output (special tokens parsed)
    std::chrono::milliseconds delay{0};      ///< sleep before each token
    /// Simulated queue wait + prefill before the first token; the engine's cancel token and
    /// deadline are honoured during it (engine.h, checked every "tick" of 10 ms).
    std::chrono::milliseconds prefill{0};
    bool eos = true;                         ///< end with an EOS event after `text`
    std::optional<std::string> throw_error;  ///< throw halo::Error(throw_code, ...) instead
    ErrorCode throw_code = ErrorCode::Backend;  ///< Api / Unsupported mimic an engine-side request rejection
    std::optional<runtime::FinishReason> finish;  ///< report this instead of Stop/Length
    std::string error_text;                  ///< GenerateResult::error with finish == Error
};

struct RecordedCall {
    runtime::GenerateRequest request;
    bool cancelled = false;          ///< the callback returned false
    std::size_t tokens_emitted = 0;  ///< callback invocations
    bool stop_token_cancelled = false;  ///< ended through GenerateRequest::cancel
    bool deadline_expired = false;      ///< ended through GenerateRequest::deadline
    std::chrono::milliseconds duration{0};
};

class FakeEngine final : public runtime::Engine {
public:
    FakeEngine(std::unique_ptr<tokenizer::Tokenizer> tok, std::unique_ptr<chat::ChatTemplate> tmpl,
               runtime::ModelInfo info);

    /// Byte-level synthetic tokenizer (256 byte tokens, no merges) with the Qwen added tokens
    /// (<|im_start|>, <|im_end|>, <|endoftext|> Control; <think>, </think>, <tool_call>,
    /// </tool_call>, <tool_response>, </tool_response> UserDefined; one Unused) and a small
    /// ChatML template. Always available. Like the real engine (engine.h), generate() throws
    /// Error(Api) for an empty prompt, a prompt >= the context, or max_tokens == 0.
    static std::unique_ptr<FakeEngine> synthetic(std::size_t context_length = 4096);
    /// The real Qwen3.8 tokenizer.json + chat_template.jinja from HALO_REF_DIR; nullptr (with
    /// `why` set) when the files are missing.
    static std::unique_ptr<FakeEngine> real(std::string& why, std::size_t context_length = 262144);

    const runtime::ModelInfo& model() const override { return info_; }
    const tokenizer::Tokenizer& tokenizer() const override { return *tok_; }
    const chat::ChatTemplate& chat_template() const override { return *tmpl_; }
    runtime::GenerateResult generate(const runtime::GenerateRequest& req, const runtime::TokenCallback& cb) override;
    runtime::EngineStats stats() const override;

    /// (request, zero-based call index) -> what to emit. Default: "Hello." then EOS.
    std::function<Script(const runtime::GenerateRequest&, int)> script;

    [[nodiscard]] std::vector<RecordedCall> calls() const;
    [[nodiscard]] int active() const { return active_.load(); }
    [[nodiscard]] int cancellations() const { return cancellations_.load(); }
    /// Waits until at least `n` generate() calls are running (true) or `timeout` passes.
    bool wait_active(int n, std::chrono::milliseconds timeout) const;
    /// Waits until at least `n` calls have been cancelled by their callback.
    bool wait_cancellations(int n, std::chrono::milliseconds timeout) const;
    /// Decodes the prompt of call `i` (for readable assertions).
    [[nodiscard]] std::string prompt_text(std::size_t i) const;

private:
    std::unique_ptr<tokenizer::Tokenizer> tok_;
    std::unique_ptr<chat::ChatTemplate> tmpl_;
    runtime::ModelInfo info_;
    mutable std::mutex mu_;
    mutable std::condition_variable cv_;
    std::vector<RecordedCall> calls_;
    std::atomic<int> active_{0};
    std::atomic<int> cancellations_{0};
    std::atomic<int> call_index_{0};
    std::atomic<std::uint64_t> generated_{0};
};

/// The byte-level synthetic tokenizer spec used by FakeEngine::synthetic().
[[nodiscard]] tokenizer::VocabSpec synthetic_vocab();
/// The synthetic ChatML template source.
[[nodiscard]] const std::string& synthetic_template();

[[nodiscard]] std::string read_file(const std::filesystem::path& p);
[[nodiscard]] std::filesystem::path ref_dir();

/// Number of occurrences of token `id` in `ids`.
[[nodiscard]] std::size_t count_id(const std::vector<std::int32_t>& ids, std::int32_t id);

}  // namespace halo::test
