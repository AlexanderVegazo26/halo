#include "fake_engine.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>

#include "halo/core/error.h"

namespace halo::test {

namespace fs = std::filesystem;
using tokenizer::TokenType;

namespace {

void append_utf8(std::string& s, char32_t c) {
    if (c < 0x80) {
        s.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (c >> 6)));
        s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xE0 | (c >> 12)));
        s.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
}

// GPT-2 bytes_to_unicode.
std::vector<std::string> byte_map() {
    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
    for (int c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
    for (int c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);
    std::vector<int> cs = bs;
    int n = 0;
    for (int c = 0; c < 256; ++c) {
        if (std::ranges::find(bs, c) == bs.end()) {
            bs.push_back(c);
            cs.push_back(256 + n++);
        }
    }
    std::vector<std::string> out(256);
    for (std::size_t i = 0; i < bs.size(); ++i) {
        std::string s;
        append_utf8(s, static_cast<char32_t>(cs[i]));
        out[static_cast<std::size_t>(bs[i])] = s;
    }
    return out;
}

}  // namespace

tokenizer::VocabSpec synthetic_vocab() {
    tokenizer::VocabSpec s;
    s.tokens = byte_map();
    s.types.assign(256, TokenType::Normal);
    const auto add = [&](std::string t, TokenType type) {
        s.tokens.push_back(std::move(t));
        s.types.push_back(type);
        return static_cast<std::int32_t>(s.tokens.size() - 1);
    };
    add("<|endoftext|>", TokenType::Control);
    add("<|im_start|>", TokenType::Control);
    s.eos = add("<|im_end|>", TokenType::Control);
    for (const char* t : {"<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"}) {
        add(t, TokenType::UserDefined);
    }
    add("[PAD300]", TokenType::Unused);
    return s;
}

const std::string& synthetic_template() {
    static const std::string t = R"TPL({%- if tools %}
{{- '<|im_start|>system\n# Tools\n' }}
{%- for t in tools %}
{{- t | tojson }}
{{- '\n' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- endif %}
{%- for m in messages %}
{%- if m.role == 'assistant' %}
{{- '<|im_start|>assistant\n' }}
{%- if m.reasoning_content is defined and m.reasoning_content %}
{{- '<think>\n' + m.reasoning_content + '\n</think>\n\n' }}
{%- endif %}
{{- m.content }}
{%- if m.tool_calls is defined %}
{%- for tc in m.tool_calls %}
{{- '\n<tool_call>\n<function=' + tc.function.name + '>\n' }}
{%- for k, v in tc.function.arguments | items %}
{{- '<parameter=' + k + '>\n' }}
{%- if v is string %}{{- v }}{%- else %}{{- v | tojson }}{%- endif %}
{{- '\n</parameter>\n' }}
{%- endfor %}
{{- '</function>\n</tool_call>' }}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- elif m.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + m.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- else %}
{{- '<think>\n' }}
{%- endif %}
{%- endif %})TPL";
    return t;
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

fs::path ref_dir() { return HALO_REF_DIR; }

std::size_t count_id(const std::vector<std::int32_t>& ids, std::int32_t id) {
    return static_cast<std::size_t>(std::ranges::count(ids, id));
}

FakeEngine::FakeEngine(std::unique_ptr<tokenizer::Tokenizer> tok, std::unique_ptr<chat::ChatTemplate> tmpl,
                       runtime::ModelInfo info)
    : tok_(std::move(tok)), tmpl_(std::move(tmpl)), info_(std::move(info)) {
    script = [](const runtime::GenerateRequest&, int) {
        Script s;
        s.text = "Hello.";
        return s;
    };
}

std::unique_ptr<FakeEngine> FakeEngine::synthetic(std::size_t context_length) {
    auto tok = std::make_unique<tokenizer::Tokenizer>(tokenizer::Tokenizer::from_spec(synthetic_vocab()));
    auto tmpl = std::make_unique<chat::ChatTemplate>(synthetic_template(), "", "<|im_end|>");
    runtime::ModelInfo info{.id = "fake-model",
                            .architecture = "qwen35",
                            .context_length = context_length,
                            .vocab_size = tok->vocab_size(),
                            .has_mtp = false};
    return std::make_unique<FakeEngine>(std::move(tok), std::move(tmpl), std::move(info));
}

std::unique_ptr<FakeEngine> FakeEngine::real(std::string& why, std::size_t context_length) {
    const fs::path d = ref_dir();
    for (const char* f : {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}) {
        if (!fs::exists(d / f)) {
            why = "reference file missing: " + (d / f).string();
            return nullptr;
        }
    }
    auto tok = std::make_unique<tokenizer::Tokenizer>(
        tokenizer::Tokenizer::from_hf_json(read_file(d / "tokenizer.json"), read_file(d / "tokenizer_config.json")));
    auto tmpl = std::make_unique<chat::ChatTemplate>(read_file(d / "chat_template.jinja"), "", "<|im_end|>");
    runtime::ModelInfo info{.id = "qwen3.8-27b",
                            .architecture = "qwen35",
                            .context_length = context_length,
                            .vocab_size = tok->vocab_size(),
                            .has_mtp = true};
    return std::make_unique<FakeEngine>(std::move(tok), std::move(tmpl), std::move(info));
}

runtime::GenerateResult FakeEngine::generate(const runtime::GenerateRequest& req, const runtime::TokenCallback& cb) {
    const int index = call_index_.fetch_add(1);
    std::size_t slot = 0;
    {
        std::lock_guard lk(mu_);
        calls_.push_back(RecordedCall{req, false, 0});
        slot = calls_.size() - 1;
        active_.fetch_add(1);  // under mu_: no lost wake-up for wait_active
    }
    cv_.notify_all();
    struct Leave {
        FakeEngine* e;
        ~Leave() {
            {
                std::lock_guard lk(e->mu_);
                e->active_.fetch_sub(1);
            }
            e->cv_.notify_all();
        }
    } leave{this};

    const Script s = script(req, index);
    if (s.throw_error) throw_error(ErrorCode::Backend, "{}", *s.throw_error);

    runtime::GenerateResult r;
    r.prompt_tokens = req.prompt.size();
    const auto ids = tok_->encode(s.text, true);
    tokenizer::StreamDecoder dec(*tok_);
    bool cancelled = false;
    const auto emit = [&](const runtime::TokenEvent& ev) {
        {
            std::lock_guard lk(mu_);
            ++calls_[slot].tokens_emitted;
        }
        if (!cb(ev)) {
            cancelled = true;
            {
                std::lock_guard lk(mu_);
                calls_[slot].cancelled = true;
                cancellations_.fetch_add(1);
            }
            cv_.notify_all();
        }
        return !cancelled;
    };
    r.finish = runtime::FinishReason::Stop;
    bool hit_length = false;
    for (const auto id : ids) {
        if (r.tokens.size() >= req.max_tokens) {
            hit_length = true;
            break;
        }
        if (s.delay.count() > 0) std::this_thread::sleep_for(s.delay);
        r.tokens.push_back(id);
        generated_.fetch_add(1);
        runtime::TokenEvent ev{.token = id, .piece = dec.push(id), .is_eos = false};
        if (!emit(ev)) break;
    }
    if (cancelled) {
        r.finish = runtime::FinishReason::Cancelled;
        return r;
    }
    if (hit_length) {
        r.finish = runtime::FinishReason::Length;
    } else if (s.eos && r.tokens.size() < req.max_tokens) {
        const std::int32_t eos = tok_->eos().value_or(0);
        r.tokens.push_back(eos);
        if (!emit(runtime::TokenEvent{.token = eos, .piece = "", .is_eos = true})) {
            r.finish = runtime::FinishReason::Cancelled;
            return r;
        }
    } else if (!s.eos) {
        r.finish = runtime::FinishReason::Length;
    }
    if (s.finish) {
        r.finish = *s.finish;
        r.error = s.error_text;
    }
    r.decode_tps = 42.0;
    return r;
}

runtime::EngineStats FakeEngine::stats() const {
    runtime::EngineStats st;
    st.active_sequences = static_cast<std::size_t>(active_.load());
    st.tokens_generated = generated_.load();
    st.prefix_cache_hits = 3;
    st.prefix_cache_misses = 1;
    st.speculative_attempts = 10;
    st.speculative_accepts = 7;
    return st;
}

std::vector<RecordedCall> FakeEngine::calls() const {
    std::lock_guard lk(mu_);
    return calls_;
}

bool FakeEngine::wait_active(int n, std::chrono::milliseconds timeout) const {
    std::unique_lock lk(mu_);
    return cv_.wait_for(lk, timeout, [&] { return active_.load() >= n; });
}

bool FakeEngine::wait_cancellations(int n, std::chrono::milliseconds timeout) const {
    std::unique_lock lk(mu_);
    return cv_.wait_for(lk, timeout, [&] { return cancellations_.load() >= n; });
}

std::string FakeEngine::prompt_text(std::size_t i) const {
    const auto c = calls();
    return tok_->decode(c.at(i).request.prompt, false);
}

}  // namespace halo::test
