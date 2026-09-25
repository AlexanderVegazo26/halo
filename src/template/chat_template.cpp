#include "halo/template/chat_template.h"

#include <minja/minja.hpp>

#include <pthread.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <ctime>
#include <exception>
#include <limits>

#include "halo/core/error.h"

// Fail closed: without the in-tree patch minja has no limits at all (security review S-3).
// A FETCHCONTENT_SOURCE_DIR_MINJA override or a stale _deps directory would otherwise build
// silently against unlimited upstream minja.
#if !defined(MINJA_HALO_LIMITS) || MINJA_HALO_LIMITS < 1
#error "minja is missing cmake/patches/minja-halo-limits.patch (see cmake/HaloDeps.cmake)"
#endif

namespace halo::chat {
namespace {

constexpr std::uint64_t kMaxRangeLength = 100000;
constexpr std::string_view kLimitPrefix = "chat template limit exceeded: ";
constexpr std::string_view kTurnHeader = "<|im_start|>";

// ---- Python str.strip() for Jinja's `trim` ------------------------------------------------

// Characters for which Python 3.12 str.isspace() is true.
bool py_isspace(char32_t c) noexcept {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// Decodes one codepoint of (assumed valid) UTF-8; invalid bytes decode as themselves.
char32_t decode_at(std::string_view s, std::size_t i, std::size_t& len) noexcept {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const auto cont = [&](std::size_t k) { return static_cast<char32_t>(static_cast<unsigned char>(s[i + k]) & 0x3Fu); };
    if (b0 >= 0xF0 && i + 3 < s.size()) {
        len = 4;
        return (static_cast<char32_t>(b0 & 0x07u) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
    }
    if (b0 >= 0xE0 && i + 2 < s.size()) {
        len = 3;
        return (static_cast<char32_t>(b0 & 0x0Fu) << 12) | (cont(1) << 6) | cont(2);
    }
    if (b0 >= 0xC0 && i + 1 < s.size()) {
        len = 2;
        return (static_cast<char32_t>(b0 & 0x1Fu) << 6) | cont(1);
    }
    len = 1;
    return b0;
}

std::string py_strip(std::string_view s, const std::vector<char32_t>* chars) {
    const auto strip_char = [&](char32_t c) {
        return chars == nullptr ? py_isspace(c) : std::find(chars->begin(), chars->end(), c) != chars->end();
    };
    std::size_t begin = 0;
    while (begin < s.size()) {
        std::size_t len = 1;
        if (!strip_char(decode_at(s, begin, len))) break;
        begin += len;
    }
    std::size_t end = begin;  // one past the last non-stripped codepoint
    for (std::size_t i = begin; i < s.size();) {
        std::size_t len = 1;
        const char32_t c = decode_at(s, i, len);
        i += len;
        if (!strip_char(c)) end = i;
    }
    return std::string(s.substr(begin, end - begin));
}

std::vector<char32_t> codepoints(std::string_view s) {
    std::vector<char32_t> out;
    for (std::size_t i = 0; i < s.size();) {
        std::size_t len = 1;
        out.push_back(decode_at(s, i, len));
        i += len;
    }
    return out;
}

// ---- source compatibility rewrite ---------------------------------------------------------
//
// minja 021c229 has no `undefined` test (`x is undefined` throws "Unknown type for 'is'
// operator"), while its `defined` test is exactly `not undefined` for our inputs. Inside
// Jinja tags, and outside string literals, rewrite
//   `is not undefined` -> `is defined`   and   `is undefined` -> `is not defined`.
// This is the smallest fix that keeps minja unforked; everything else is parsed as written.

bool is_ident(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

// If `s[i..]` starts with `word` followed by a non-identifier char, returns the end.
std::size_t match_word(std::string_view s, std::size_t i, std::string_view word) {
    if (s.substr(i, word.size()) != word) return std::string_view::npos;
    const std::size_t e = i + word.size();
    if (e < s.size() && is_ident(s[e])) return std::string_view::npos;
    return e;
}

std::size_t skip_spaces(std::string_view s, std::size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    return i;
}

std::string rewrite_undefined_tests(std::string_view src) {
    std::string out;
    out.reserve(src.size());
    std::size_t i = 0;
    while (i < src.size()) {
        if (src.substr(i, 2) == "{#") {  // comment: copy verbatim
            const std::size_t e = src.find("#}", i + 2);
            const std::size_t end = e == std::string_view::npos ? src.size() : e + 2;
            out.append(src.substr(i, end - i));
            i = end;
            continue;
        }
        if (src.substr(i, 2) != "{{" && src.substr(i, 2) != "{%") {
            out.push_back(src[i++]);
            continue;
        }
        const std::string_view close = src[i + 1] == '{' ? "}}" : "%}";
        out.append(src.substr(i, 2));
        i += 2;
        char quote = 0;
        while (i < src.size()) {
            const char c = src[i];
            if (quote != 0) {
                out.push_back(c);
                if (c == '\\' && i + 1 < src.size()) {
                    out.push_back(src[i + 1]);
                    i += 2;
                    continue;
                }
                if (c == quote) quote = 0;
                ++i;
                continue;
            }
            if (c == '\'' || c == '"') {
                quote = c;
                out.push_back(c);
                ++i;
                continue;
            }
            if (src.substr(i, 2) == close) {
                out.append(close);
                i += 2;
                break;
            }
            if ((i == 0 || !is_ident(src[i - 1])) && match_word(src, i, "is") != std::string_view::npos) {
                const std::size_t a = skip_spaces(src, i + 2);
                const std::size_t n = match_word(src, a, "not");
                if (n != std::string_view::npos) {
                    const std::size_t b = skip_spaces(src, n);
                    const std::size_t u = match_word(src, b, "undefined");
                    if (b > n && u != std::string_view::npos) {
                        out.append("is defined");
                        i = u;
                        continue;
                    }
                } else if (a > i + 2) {
                    const std::size_t u = match_word(src, a, "undefined");
                    if (u != std::string_view::npos) {
                        out.append("is not defined");
                        i = u;
                        continue;
                    }
                }
            }
            out.push_back(c);
            ++i;
        }
    }
    return out;
}

// ---- render-time state shared with the overridden globals ---------------------------------

// A `{% macro %}` is stored in the context it is defined in and captures that context, so
// every render that defines a macro (both Qwen templates do) forms a shared_ptr cycle that
// leaks the whole context chain (~17 KB per render; found by LeakSanitizer); `{% call %}`
// does the same with `caller`. Nulling the variables of those contexts after rendering
// breaks the cycles: the top-level context, plus every context the patched minja recorded
// in the render budget (macros defined inside loops or other macros, call blocks; S-10).
struct ContextCycleBreaker {
    std::shared_ptr<minja::Context> ctx;
    minja::RenderBudget& budget;
    ContextCycleBreaker(std::shared_ptr<minja::Context> c, minja::RenderBudget& b) : ctx(std::move(c)), budget(b) {}
    ContextCycleBreaker(const ContextCycleBreaker&) = delete;
    ContextCycleBreaker& operator=(const ContextCycleBreaker&) = delete;
    ~ContextCycleBreaker() {
        minja::RenderBudget* const installed = minja::current_budget;
        minja::current_budget = nullptr;  // clearing must not be charged (or throw)
        const auto clear = [](const std::shared_ptr<minja::Context>& c) {
            try {
                for (const auto& key : c->keys()) c->set(key, minja::Value());
            } catch (...) {  // allocation failure while clearing: the cycle leaks, nothing worse
            }
        };
        clear(ctx);
        for (const auto& weak : budget.contexts) {
            if (const auto c = weak.lock()) clear(c);
        }
        minja::current_budget = installed;
    }
};

struct RaiseState {
    bool raised = false;
    std::string message;
};

// Number of elements range(start, end, step) produces: ceil((end - start) / step), or 0.
// Exact in unsigned 64-bit arithmetic (the difference of two int64 values fits).
std::uint64_t range_length(std::int64_t start, std::int64_t end, std::int64_t step) noexcept {
    if (step == 0) return 0;  // the builtin raises
    const bool up = step > 0;
    if (up ? end <= start : end >= start) return 0;
    const std::uint64_t span = up ? static_cast<std::uint64_t>(end) - static_cast<std::uint64_t>(start)
                                  : static_cast<std::uint64_t>(start) - static_cast<std::uint64_t>(end);
    const std::uint64_t stride = up ? static_cast<std::uint64_t>(step) : std::uint64_t{0} - static_cast<std::uint64_t>(step);
    return span / stride + (span % stride != 0 ? 1 : 0);
}

std::uint64_t range_length(minja::ArgumentsValue& args) {
    std::array<std::int64_t, 3> v{0, 0, 1};  // start, end, step
    if (args.args.size() == 1) {
        v[1] = args.args[0].get<std::int64_t>();
    } else {
        for (std::size_t i = 0; i < args.args.size() && i < 3; ++i) v[i] = args.args[i].get<std::int64_t>();
    }
    for (auto& [name, value] : args.kwargs) {
        if (name == "start") v[0] = value.get<std::int64_t>();
        if (name == "end") v[1] = value.get<std::int64_t>();
        if (name == "step") v[2] = value.get<std::int64_t>();
    }
    return range_length(v[0], v[1], v[2]);
}

void install_globals(const std::shared_ptr<minja::Context>& ctx, const std::shared_ptr<RaiseState>& raise,
                     std::chrono::system_clock::time_point now) {
    ctx->set("raise_exception",
             minja::Value::callable([raise](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
                 raise->raised = true;
                 raise->message = args.args.empty() ? std::string("raise_exception") : args.args[0].to_str();
                 throw std::runtime_error(raise->message);
             }));
    ctx->set("trim", minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        if (args.args.empty()) throw std::runtime_error("trim expects a value");
        const minja::Value& v = args.args[0];
        if (v.is_null()) return v;
        const std::string text = v.to_str();
        const minja::Value* chars = args.args.size() > 1 ? &args.args[1] : nullptr;
        for (auto& [name, value] : args.kwargs) {
            if (name == "chars") chars = &value;
        }
        if (chars != nullptr && !chars->is_null()) {
            const auto set = codepoints(chars->to_str());
            return minja::Value(py_strip(text, &set));
        }
        return minja::Value(py_strip(text, nullptr));
    }));
    const minja::Value builtin_range = minja::Context::builtins()->get("range");
    ctx->set("range", minja::Value::callable([builtin_range](const std::shared_ptr<minja::Context>& c,
                                                           minja::ArgumentsValue& args) -> minja::Value {
        if (range_length(args) > kMaxRangeLength) {  // exactly kMaxRangeLength is allowed (S-10)
            throw std::runtime_error("range() longer than " + std::to_string(kMaxRangeLength) + " elements");
        }
        return builtin_range.call(c, args);
    }));
    ctx->set("strftime_now", minja::Value::callable([now](const std::shared_ptr<minja::Context>&,
                                                         minja::ArgumentsValue& args) -> minja::Value {
        if (args.args.size() != 1) throw std::runtime_error("strftime_now expects one argument");
        const std::string fmt = args.args[0].to_str();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        localtime_r(&t, &tm);
        std::array<char, 256> buf{};
        const std::size_t n = std::strftime(buf.data(), buf.size(), fmt.c_str(), &tm);
        return minja::Value(std::string(buf.data(), n));
    }));
}

OrderedJson with_parsed_arguments(const OrderedJson& messages) {
    OrderedJson out = messages;
    for (auto& m : out) {
        if (!m.is_object() || !m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
        for (auto& tc : m["tool_calls"]) {
            if (!tc.is_object()) continue;
            OrderedJson& fn = tc.contains("function") && tc["function"].is_object() ? tc["function"] : tc;
            if (!fn.contains("arguments") || !fn["arguments"].is_string()) continue;
            OrderedJson parsed = OrderedJson::parse(fn["arguments"].get<std::string>(), nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object()) fn["arguments"] = std::move(parsed);
        }
    }
    return out;
}

// Aligns the ChatML turn headers in `prompt` with `messages` (see RenderResult).
void compute_offsets(const OrderedJson& messages, bool add_generation_prompt, RenderResult& r) {
    struct Header {
        std::size_t pos;
        std::string role;
    };
    std::vector<Header> headers;
    const std::string& p = r.prompt;
    for (std::size_t at = p.find(kTurnHeader); at != std::string::npos; at = p.find(kTurnHeader, at + 1)) {
        const std::size_t role_begin = at + kTurnHeader.size();
        const std::size_t nl = p.find('\n', role_begin);
        headers.push_back({at, p.substr(role_begin, (nl == std::string::npos ? p.size() : nl) - role_begin)});
    }
    struct Expect {
        std::size_t index;  // messages index, or npos for the generation prompt
        std::string role;
    };
    std::size_t lead = 0;  // leading system/developer run
    const auto role_of = [&](std::size_t i) -> std::string {
        const auto& m = messages[i];
        return m.contains("role") && m["role"].is_string() ? m["role"].get<std::string>() : std::string();
    };
    while (lead < messages.size() && (role_of(lead) == "system" || role_of(lead) == "developer")) ++lead;
    std::vector<Expect> expect;
    for (std::size_t i = lead; i < messages.size(); ++i) {
        const std::string role = role_of(i);
        if (role == "user" || role == "assistant") {
            expect.push_back({i, role});
        } else if (role == "tool") {
            if (i == lead || role_of(i - 1) != "tool") expect.push_back({i, "user"});
        } else {
            return;  // unknown role: structure cannot be verified
        }
    }
    if (add_generation_prompt) expect.push_back({std::string::npos, "assistant"});
    const bool has_preamble = !headers.empty() && headers[0].role == "system" &&
                              headers.size() == expect.size() + 1;
    const std::size_t first = has_preamble ? 1 : 0;
    if (headers.size() - first != expect.size()) return;
    for (std::size_t k = 0; k < expect.size(); ++k) {
        if (headers[first + k].role != expect[k].role) return;
    }
    r.offsets_valid = true;
    if (has_preamble) {
        for (std::size_t i = 0; i < lead; ++i) r.message_starts.push_back({i, "system", headers[0].pos});
        r.preamble_end = expect.empty() ? p.size() : headers[1].pos;
    } else {
        r.preamble_end = 0;
    }
    for (std::size_t k = 0; k < expect.size(); ++k) {
        if (expect[k].index == std::string::npos) {
            r.generation_prompt_start = headers[first + k].pos;
        } else {
            r.message_starts.push_back({expect[k].index, expect[k].role, headers[first + k].pos});
        }
    }
}

// Runs `fn` to completion on a new thread with a `stack_bytes` stack (0: on this thread)
// and rethrows whatever it threw. minja's lexer uses std::regex, whose libstdc++ executor
// recurses once per matched character; the source-size cap bounds that recursion and this
// stack makes the bound independent of the caller's thread.
template <typename Fn>
void run_on_stack(std::size_t stack_bytes, Fn&& fn) {
    if (stack_bytes == 0) {
        fn();
        return;
    }
    struct Job {
        Fn* fn;
        std::exception_ptr error;
    };
    Job job{&fn, nullptr};
    pthread_attr_t attr;
    HALO_CHECK(pthread_attr_init(&attr) == 0, ErrorCode::Memory, "pthread_attr_init failed");
    const int set = pthread_attr_setstacksize(&attr, stack_bytes);
    pthread_t thread{};
    const int created = set == 0 ? pthread_create(
                                       &thread, &attr,
                                       [](void* p) -> void* {
                                           auto* j = static_cast<Job*>(p);
                                           try {
                                               (*j->fn)();
                                           } catch (...) {
                                               j->error = std::current_exception();
                                           }
                                           return nullptr;
                                       },
                                       &job)
                                 : set;
    pthread_attr_destroy(&attr);
    HALO_CHECK(created == 0, ErrorCode::Memory, "cannot start the template parser thread ({} byte stack): {}",
               stack_bytes, std::strerror(created));
    pthread_join(thread, nullptr);
    if (job.error) std::rethrow_exception(job.error);
}

minja::Limits to_minja(const TemplateLimits& l) {
    minja::Limits m;
    m.max_depth = l.max_render_depth;
    m.max_steps = l.max_steps;
    m.max_loop_iterations = l.max_loop_iterations;
    m.max_output_bytes = l.max_output_bytes;
    m.max_string_bytes = l.max_string_bytes;
    m.max_alloc_bytes = l.max_alloc_bytes;
    return m;
}

// Installs a render budget on this thread for the scope (restoring any outer one).
class ScopedBudget {
public:
    explicit ScopedBudget(minja::RenderBudget& b) noexcept : prev_(minja::current_budget) { minja::current_budget = &b; }
    ~ScopedBudget() { minja::current_budget = prev_; }
    ScopedBudget(const ScopedBudget&) = delete;
    ScopedBudget& operator=(const ScopedBudget&) = delete;

private:
    minja::RenderBudget* prev_;
};

constexpr std::array<std::string_view, 5> kReservedVariables = {"messages", "tools", "bos_token", "eos_token",
                                                               "add_generation_prompt"};

}  // namespace

struct ChatTemplate::Impl {
    std::string source;
    std::string bos;
    std::string eos;
    TemplateLimits limits;
    std::shared_ptr<minja::TemplateNode> root;
};

ChatTemplate::ChatTemplate(std::string source, std::string bos_token, std::string eos_token)
    : ChatTemplate(std::move(source), std::move(bos_token), std::move(eos_token), TemplateLimits{}) {}

ChatTemplate::ChatTemplate(std::string source, std::string bos_token, std::string eos_token,
                           const TemplateLimits& limits)
    : impl_(std::make_unique<Impl>()) {
    HALO_CHECK(!source.empty(), ErrorCode::Config, "chat template is empty");
    HALO_CHECK(limits.max_source_bytes == 0 || source.size() <= limits.max_source_bytes, ErrorCode::Config,
               "chat template is {} bytes (limit {})", source.size(), limits.max_source_bytes);
    impl_->source = std::move(source);
    impl_->bos = std::move(bos_token);
    impl_->eos = std::move(eos_token);
    impl_->limits = limits;
    try {
        const std::string rewritten = rewrite_undefined_tests(impl_->source);
        const minja::Options options{/*trim_blocks=*/true, /*lstrip_blocks=*/true, /*keep_trailing_newline=*/false,
                                     /*max_depth=*/limits.max_parse_depth};
        run_on_stack(limits.parse_stack_bytes, [&] { impl_->root = minja::Parser::parse(rewritten, options); });
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        throw_error(ErrorCode::Config, "cannot parse chat template: {}", e.what());
    }
}

ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;
ChatTemplate::~ChatTemplate() = default;

const std::string& ChatTemplate::source() const noexcept { return impl_->source; }

RenderResult ChatTemplate::render(const OrderedJson& messages, const OrderedJson& tools,
                                  const RenderOptions& options) const {
    HALO_CHECK(messages.is_array(), ErrorCode::Api, "messages must be an array");
    for (const auto& m : messages) {
        HALO_CHECK(m.is_object(), ErrorCode::Api, "each message must be an object");
        // HF's Jinja raises on `'<function=' + tool_call.name` when the name is missing;
        // minja would silently concatenate, so the check is made here.
        if (!m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
        for (const auto& tc : m["tool_calls"]) {
            const OrderedJson* fn = tc.is_object() && tc.contains("function") ? &tc["function"] : &tc;
            HALO_CHECK(fn->is_object() && fn->contains("name") && (*fn)["name"].is_string(), ErrorCode::Api,
                       "tool call is missing a function name");
        }
    }
    HALO_CHECK(tools.is_null() || tools.is_array(), ErrorCode::Api, "tools must be an array or null");
    HALO_CHECK(options.extra_context.is_null() || options.extra_context.is_object(), ErrorCode::Api,
               "extra template context must be an object");
    if (options.extra_context.is_object()) {
        for (const auto& [k, v] : options.extra_context.items()) {
            // minja treats a null value as undefined (Jinja does not), so a null would render
            // differently from HF; omit the key instead.
            HALO_CHECK(!v.is_null(), ErrorCode::Api, "template variable '{}' is null; omit it instead", k);
            // `x is true` / `x is false` are identity tests in Jinja but truthiness in minja;
            // they agree only for real booleans.
            if (k == "enable_thinking" || k == "preserve_thinking" || k == "add_vision_id") {
                HALO_CHECK(v.is_boolean(), ErrorCode::Api, "template variable '{}' must be a boolean", k);
            }
            // S-8: extra_context must not replace the conversation, the tools or the tokens
            // (and so desynchronise the message offsets computed from `messages`).
            HALO_CHECK(std::find(kReservedVariables.begin(), kReservedVariables.end(), k) == kReservedVariables.end(),
                       ErrorCode::Api, "template variable '{}' is reserved and cannot be set in extra_context", k);
        }
    }

    const OrderedJson msgs = options.parse_string_tool_arguments ? with_parsed_arguments(messages) : messages;
    OrderedJson vars = OrderedJson::object();
    vars["messages"] = msgs;
    vars["add_generation_prompt"] = options.add_generation_prompt;
    vars["bos_token"] = impl_->bos;
    vars["eos_token"] = impl_->eos;
    if (!tools.is_null()) vars["tools"] = tools;
    if (options.extra_context.is_object()) {
        for (const auto& [k, v] : options.extra_context.items()) vars[k] = v;
    }
    auto raise = std::make_shared<RaiseState>();
    RenderResult r;
    minja::RenderBudget budget;
    budget.limits = to_minja(impl_->limits);
    try {
        const ScopedBudget scope{budget};
        auto ctx = minja::Context::make(minja::Value(vars));
        const ContextCycleBreaker breaker{ctx, budget};
        install_globals(ctx, raise, options.now.value_or(std::chrono::system_clock::now()));
        r.prompt = impl_->root->render(ctx);
    } catch (const minja::LimitExceeded& e) {
        throw_error(ErrorCode::Api, "{}{}", kLimitPrefix, e.what());
    } catch (const std::exception& e) {
        if (raise->raised) throw_error(ErrorCode::Api, "chat template raised: {}", raise->message);
        throw_error(ErrorCode::Api, "chat template rendering failed: {}", e.what());
    }
    r.stats = RenderStats{budget.peak_depth, budget.steps, budget.loop_iterations, budget.output_bytes,
                          budget.alloc_bytes};
    compute_offsets(msgs, options.add_generation_prompt, r);
    return r;
}

const TemplateLimits& ChatTemplate::limits() const noexcept { return impl_->limits; }

std::string ChatTemplate::apply(const OrderedJson& messages, const OrderedJson& tools,
                                const RenderOptions& options) const {
    return render(messages, tools, options).prompt;
}

}  // namespace halo::chat
