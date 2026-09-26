#include "generation.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <thread>
#include <exception>
#include <format>

#include "halo/core/error.h"

namespace halo::api {

// ---- StopMatcher ---------------------------------------------------------------------------

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    std::erase_if(stops_, [](const std::string& s) { return s.empty(); });
    for (const auto& s : stops_) max_len_ = std::max(max_len_, s.size());
}

std::string StopMatcher::push(std::string_view text) {
    if (matched_) return {};
    pending_ += text;
    if (stops_.empty()) return std::exchange(pending_, {});

    std::size_t best = std::string::npos;
    const std::string* which = nullptr;
    for (const auto& s : stops_) {
        const auto p = pending_.find(s);
        if (p != std::string::npos && (p < best || (p == best && s.size() > which->size()))) {
            best = p;
            which = &s;
        }
    }
    if (which != nullptr) {
        matched_ = *which;
        std::string out = pending_.substr(0, best);
        pending_.clear();
        return out;
    }
    // Hold back the longest suffix that is a proper prefix of some stop sequence.
    std::size_t keep = 0;
    for (std::size_t k = std::min(pending_.size(), max_len_ - 1); k > 0; --k) {
        const std::string_view suffix(pending_.data() + pending_.size() - k, k);
        if (std::ranges::any_of(stops_, [&](const std::string& s) { return s.starts_with(suffix); })) {
            keep = k;
            break;
        }
    }
    std::size_t cut = pending_.size() - keep;
    while (cut > 0 && cut < pending_.size() && (static_cast<unsigned char>(pending_[cut]) & 0xC0u) == 0x80u) --cut;
    std::string out = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return out;
}

std::string StopMatcher::flush() {
    if (matched_) return {};
    return std::exchange(pending_, {});
}

// ---- driver --------------------------------------------------------------------------------

namespace {

enum class Cancel { None, Budget, StopSequence, Client, Shutdown, Deadline, Nesting };

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

GenerationOutcome run_generation(runtime::Engine& engine, const GenerationSpec& spec, EventSink& sink,
                                 const std::atomic<bool>& stopping) {
    GenerationOutcome out;
    chat::OutputParser parser(spec.parser);
    StopMatcher stopper(spec.stop);
    Cancel cancel = Cancel::None;
    bool in_reasoning = spec.parse_output && spec.thinking && spec.parser.starts_in_reasoning;
    bool seen_output = false;
    std::optional<std::size_t> budget = spec.parse_output ? spec.reasoning_budget : std::nullopt;
    std::vector<std::int32_t> observed;
    std::string content_out;
    std::vector<chat::ToolCall> emitted_tools;
    const auto t0 = std::chrono::steady_clock::now();
    bool first = true;
    std::size_t nesting = 0;
    const bool guard_nesting = spec.parse_output && spec.max_output_nesting > 0;

    const auto emit_content = [&](std::string_view text) -> bool {
        if (stopper.matched()) return true;  // everything after a stop sequence is dropped
        const std::string safe = stopper.push(text);
        if (!safe.empty()) {
            content_out += safe;
            if (!sink.content(safe)) {
                cancel = Cancel::Client;
                return false;
            }
        }
        if (stopper.matched()) {
            cancel = Cancel::StopSequence;
            return false;
        }
        return true;
    };
    const auto deliver = [&](const std::vector<chat::OutputEvent>& events) -> bool {
        for (const auto& e : events) {
            switch (e.kind) {
                case chat::OutputEvent::Kind::ReasoningDelta:
                    if (!sink.reasoning(e.text)) {
                        cancel = Cancel::Client;
                        return false;
                    }
                    break;
                case chat::OutputEvent::Kind::ContentDelta:
                    in_reasoning = false;
                    seen_output = true;
                    if (!emit_content(e.text)) return false;
                    break;
                case chat::OutputEvent::Kind::ToolCall: {
                    in_reasoning = false;
                    seen_output = true;
                    if (stopper.matched()) break;
                    const auto& call = parser.message().tool_calls.at(e.tool_index);
                    emitted_tools.push_back(call);
                    if (!sink.tool_call(emitted_tools.size() - 1, call)) {
                        cancel = Cancel::Client;
                        return false;
                    }
                    break;
                }
            }
        }
        return true;
    };
    const auto feed = [&](std::string_view piece) -> bool {
        if (spec.parse_output) return deliver(parser.feed(piece));
        return emit_content(piece);
    };

    const runtime::TokenCallback cb = [&](const runtime::TokenEvent& ev) -> bool {
        if (stopping.load(std::memory_order_relaxed)) {
            cancel = Cancel::Shutdown;
            return false;
        }
        if (spec.deadline && std::chrono::steady_clock::now() >= *spec.deadline) {
            cancel = Cancel::Deadline;
            return false;
        }
        if (first) {
            out.ttft_ms = ms_since(t0);
            first = false;
        }
        ++out.completion_tokens;
        if (ev.is_eos) return true;
        observed.push_back(ev.token);
        if (spec.parse_output) {
            if (in_reasoning) {
                ++out.reasoning_tokens;
                if (spec.think_end && ev.token == *spec.think_end) in_reasoning = false;
            } else if (!seen_output && spec.think_start && ev.token == *spec.think_start) {
                in_reasoning = true;
                ++out.reasoning_tokens;
            }
        }
        if (guard_nesting) {
            for (const char c : ev.piece) {
                if (c == '[' || c == '{') {
                    ++nesting;
                } else if ((c == ']' || c == '}') && nesting > 0) {
                    --nesting;
                }
            }
            if (nesting > spec.max_output_nesting) {  // checked before the parser sees it
                cancel = Cancel::Nesting;
                return false;
            }
        }
        if (!feed(ev.piece)) return false;
        if (!sink.alive()) {
            cancel = Cancel::Client;
            return false;
        }
        if (in_reasoning && budget && out.reasoning_tokens >= *budget) {
            cancel = Cancel::Budget;
            return false;
        }
        return true;
    };

    runtime::GenerateRequest req = spec.request;
    req.stop_strings.clear();  // owned by the API (see generation.h)
    out.prompt_tokens = req.prompt.size();

    // Engine-side cancel + deadline (R-3 / S-15): reach queued and prefilling requests.
    std::stop_source cancel_source;
    enum class Why { None, Client, Shutdown };
    std::atomic<Why> why{Why::None};
    req.cancel = cancel_source.get_token();
    req.deadline = spec.deadline;
    struct Watcher {
        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
        std::thread thread;
        ~Watcher() {
            {
                std::lock_guard lk(mu);
                done = true;
            }
            cv.notify_all();
            if (thread.joinable()) thread.join();
        }
    } watcher;
    watcher.thread = std::thread([&] {
        std::unique_lock lk(watcher.mu);
        while (!watcher.done) {
            watcher.cv.wait_for(lk, std::chrono::milliseconds(100), [&] { return watcher.done; });
            if (watcher.done) break;
            if (stopping.load(std::memory_order_relaxed)) {
                why = Why::Shutdown;
                cancel_source.request_stop();
                break;
            }
            if (!sink.peer_connected()) {
                why = Why::Client;
                cancel_source.request_stop();
                break;
            }
        }
    });

    std::optional<runtime::GenerateResult> result;
    const auto call = [&](const runtime::GenerateRequest& r) {
        ++out.engine_calls;
        const std::size_t tokens_before = out.completion_tokens;
        try {
            result = engine.generate(r, cb);
        } catch (const std::exception& e) {
            result.reset();
            out.cause = StopCause::Error;
            out.error = e.what();
            out.public_error = "generation failed";
            // The Engine contract (engine.h) throws Error(Api) / Error(Unsupported) for an
            // invalid request *before* anything is scheduled: keep that typed error so the
            // client gets a 400 instead of a retry-inviting 500 (review R-4).
            const auto* he = dynamic_cast<const halo::Error*>(&e);
            if (he != nullptr && out.completion_tokens == tokens_before &&
                (he->code() == ErrorCode::Api || he->code() == ErrorCode::Unsupported)) {
                out.client_error = classify_exception(e);
                out.public_error = out.client_error->message;
            }
        }
    };

    // An engine call that ended through the stop token or the deadline (no token needed).
    const auto classify_engine_end = [&] {
        if (!result || cancel != Cancel::None) return;
        if (result->deadline_expired) {
            cancel = Cancel::Deadline;
        } else if (result->finish == runtime::FinishReason::Cancelled && cancel_source.stop_requested()) {
            cancel = why.load() == Why::Shutdown ? Cancel::Shutdown : Cancel::Client;
        }
    };

    call(req);
    classify_engine_end();
    if (result) out.cached_prompt_tokens = result->cached_prompt_tokens;

    bool length_after_budget = false;
    if (result && cancel == Cancel::Budget) {
        out.reasoning_closed_early = true;
        out.warnings.push_back(std::format(
            "reasoning budget of {} tokens reached; the think block was closed early to leave room for the answer",
            *budget));
        cancel = Cancel::None;
        budget.reset();  // one forced close per request
        in_reasoning = false;
        if (!deliver(parser.feed("\n</think>\n\n"))) {
            // client went away while closing
        } else if (out.completion_tokens >= req.max_tokens) {
            length_after_budget = true;
        } else if (spec.context_length != 0 && req.prompt.size() + observed.size() + spec.close_reasoning_tokens.size() + 1 >
                                                    spec.context_length) {
            // The continuation prompt (prompt + reasoning + "\n</think>\n\n") would not leave
            // room for one answer token in the context: end like max_tokens (review N-3).
            length_after_budget = true;
            out.warnings.push_back("the reasoning budget was reached but the context is full; no room was left for the answer");
        } else {
            runtime::GenerateRequest r2 = req;
            r2.prompt.insert(r2.prompt.end(), observed.begin(), observed.end());
            r2.prompt.insert(r2.prompt.end(), spec.close_reasoning_tokens.begin(), spec.close_reasoning_tokens.end());
            r2.max_tokens = req.max_tokens - out.completion_tokens;
            call(r2);  // r2 carries the same stop token and deadline
            classify_engine_end();
        }
    }

    if (out.cause != StopCause::Error) {
        switch (cancel) {
            case Cancel::StopSequence: out.cause = StopCause::StopSequence; break;
            case Cancel::Client: out.cause = StopCause::Client; break;
            case Cancel::Shutdown: out.cause = StopCause::Shutdown; break;
            case Cancel::Deadline:
                out.cause = StopCause::Deadline;
                out.warnings.push_back("generation stopped: the request reached this server's time limit");
                break;
            case Cancel::Nesting:
                out.cause = StopCause::Error;
                out.error = out.public_error =
                    std::format("generation stopped: model output nests '[' / '{{' deeper than {} levels",
                                spec.max_output_nesting);
                break;
            case Cancel::Budget:  // unreachable: consumed above
            case Cancel::None:
                if (length_after_budget) {
                    out.cause = StopCause::Length;
                } else if (result) {
                    switch (result->finish) {
                        case runtime::FinishReason::Stop: out.cause = StopCause::Eos; break;
                        case runtime::FinishReason::Length: out.cause = StopCause::Length; break;
                        case runtime::FinishReason::Cancelled:
                            out.cause = StopCause::Error;
                            out.error = "engine cancelled the request";
                            out.public_error = "generation was cancelled by the engine";
                            break;
                        case runtime::FinishReason::Error:
                            out.cause = StopCause::Error;
                            out.error = result->error.empty() ? "engine error" : result->error;
                            out.public_error = "generation failed";
                            break;
                    }
                }
                break;
        }
    }

    if (out.cause != StopCause::Client && out.cause != StopCause::Shutdown && out.cause != StopCause::Error) {
        if (spec.parse_output) (void)deliver(parser.finish());
        const std::string tail = stopper.flush();
        if (!tail.empty()) {
            content_out += tail;
            (void)sink.content(tail);
        }
    }

    if (spec.parse_output) {
        out.message = parser.message();
        out.message.tool_calls = std::move(emitted_tools);
    }
    out.message.content = std::move(content_out);
    out.stop_sequence = stopper.matched();
    if (result) {
        out.decode_tps = result->decode_tps;
        out.draft_tokens = result->draft_tokens;
        out.accepted_draft_tokens = result->accepted_draft_tokens;
    }
    if ((out.cause == StopCause::Length || out.cause == StopCause::Deadline) && out.message.content.empty() &&
        out.message.tool_calls.empty() && out.reasoning_tokens > 0) {
        out.warnings.push_back(std::format(
            "no content was produced: reasoning used {} of the {} max_tokens; increase max_tokens, lower "
            "reasoning_effort, or disable thinking (reasoning_effort \"none\")",
            out.reasoning_tokens, req.max_tokens));
    }
    for (const auto& w : out.message.warnings) out.warnings.push_back("output parser: " + w);
    return out;
}

}  // namespace halo::api
