#include "halo/template/output_parser.h"

#include <algorithm>
#include <utility>

namespace halo::chat {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kToolOpen = "<tool_call>";
constexpr std::string_view kToolClose = "</tool_call>";
constexpr std::string_view kFunctionOpen = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen = "<parameter=";
constexpr std::string_view kParamClose = "</parameter>";
constexpr std::string_view kSegmentSeparator = "\n\n";

// Python str.isspace() set (matches Jinja's trim used by the template).
bool py_isspace(char32_t c) noexcept {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// Decodes a codepoint at s[i] if it is a complete, valid sequence; otherwise reports the
// single byte as a non-space "codepoint" (so partial bytes pass through untouched).
char32_t cp_at(std::string_view s, std::size_t i, std::size_t& len) noexcept {
    constexpr char32_t kNotSpace = 0xFFFFFFFF;
    const auto b0 = static_cast<unsigned char>(s[i]);
    len = 1;
    if (b0 < 0x80) return b0;
    const std::size_t need = b0 >= 0xF0 ? 3 : b0 >= 0xE0 ? 2 : b0 >= 0xC0 ? 1 : 0;
    if (need == 0 || i + need >= s.size()) return kNotSpace;
    char32_t cp = b0 & (need == 3 ? 0x07u : need == 2 ? 0x0Fu : 0x1Fu);
    for (std::size_t k = 1; k <= need; ++k) {
        const auto b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0u) != 0x80u) return kNotSpace;
        cp = (cp << 6) | (b & 0x3Fu);
    }
    len = need + 1;
    return cp;
}

// Bytes at the end of `s` that begin a UTF-8 sequence not yet complete.
std::size_t incomplete_utf8_tail(std::string_view s) noexcept {
    for (std::size_t back = 1; back <= 3 && back <= s.size(); ++back) {
        const auto b = static_cast<unsigned char>(s[s.size() - back]);
        if ((b & 0xC0u) == 0x80u) continue;  // continuation byte: keep looking for the lead
        const std::size_t need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
        return need > back ? back : 0;
    }
    return 0;
}

std::size_t skip_ws(std::string_view s, std::size_t i) {
    while (i < s.size()) {
        std::size_t len = 1;
        if (!py_isspace(cp_at(s, i, len))) break;
        i += len;
    }
    return i;
}

// End of the text once trailing whitespace is removed.
std::size_t trim_end(std::string_view s) {
    std::size_t end = 0;
    for (std::size_t i = 0; i < s.size();) {
        std::size_t len = 1;
        const char32_t c = cp_at(s, i, len);
        i += len;
        if (!py_isspace(c)) end = i;
    }
    return end;
}

std::string_view trim(std::string_view s) {
    const std::size_t b = skip_ws(s, 0);
    s.remove_prefix(b);
    return s.substr(0, trim_end(s));
}

// Length of the longest suffix of `s` that is a proper prefix of `tag`.
std::size_t partial_tag_suffix(std::string_view s, std::string_view tag) {
    const std::size_t max = std::min(s.size(), tag.size() - 1);
    for (std::size_t n = max; n > 0; --n) {
        if (s.substr(s.size() - n) == tag.substr(0, n)) return n;
    }
    return 0;
}

const OrderedJson* param_schema(const OrderedJson& tools, std::string_view fn, std::string_view param) {
    if (!tools.is_array()) return nullptr;
    for (const auto& t : tools) {
        if (!t.is_object()) continue;
        const OrderedJson& f = t.contains("function") && t["function"].is_object() ? t["function"] : t;
        if (!f.contains("name") || !f["name"].is_string() || f["name"].get<std::string>() != fn) continue;
        if (!f.contains("parameters") || !f["parameters"].is_object()) return nullptr;
        const auto& params = f["parameters"];
        if (!params.contains("properties") || !params["properties"].is_object()) return nullptr;
        const auto it = params["properties"].find(std::string(param));
        return it == params["properties"].end() ? nullptr : &*it;
    }
    return nullptr;
}

bool json_has_type(const OrderedJson& v, std::string_view type) {
    if (type == "null") return v.is_null();
    if (type == "boolean") return v.is_boolean();
    if (type == "integer") return v.is_number_integer();
    if (type == "number") return v.is_number();
    if (type == "object") return v.is_object();
    if (type == "array") return v.is_array();
    return false;
}

// Schema-typed conversion of a raw parameter value (see header). With a type list that
// includes "string", a JSON parse is kept only if it has one of the other listed types.
OrderedJson convert_value(const OrderedJson* schema, std::string_view value) {
    std::vector<std::string> types;
    if (schema != nullptr && schema->is_object() && schema->contains("type")) {
        const auto& t = (*schema)["type"];
        if (t.is_string()) types.push_back(t.get<std::string>());
        if (t.is_array()) {
            for (const auto& x : t) {
                if (x.is_string()) types.push_back(x.get<std::string>());
            }
        }
    }
    const bool allows_string = std::find(types.begin(), types.end(), "string") != types.end();
    if (allows_string && types.size() == 1) return std::string(value);
    OrderedJson parsed = parse_json_bounded(value);  // S-9: depth capped while parsing
    if (parsed.is_discarded()) return std::string(value);
    if (allows_string) {
        const bool typed = std::any_of(types.begin(), types.end(),
                                       [&](const std::string& t) { return t != "string" && json_has_type(parsed, t); });
        if (!typed) return std::string(value);
    }
    return parsed;
}

struct TooDeep {};

}  // namespace

OrderedJson parse_json_bounded(std::string_view text, std::size_t max_depth) {
    // nlohmann's parser is iterative; the callback runs for every element before it is
    // attached, so an over-deep document is abandoned before any deep value exists.
    const auto cb = [max_depth](int depth, OrderedJson::parse_event_t, OrderedJson&) -> bool {
        if (depth < 0 || static_cast<std::size_t>(depth) > max_depth) throw TooDeep{};
        return true;
    };
    try {
        return OrderedJson::parse(text, cb, /*allow_exceptions=*/false);
    } catch (const TooDeep&) {
        return OrderedJson(OrderedJson::value_t::discarded);
    }
}

std::size_t json_nesting_depth(const OrderedJson& v, std::size_t limit) {
    std::size_t deepest = 0;
    std::vector<std::pair<const OrderedJson*, std::size_t>> todo{{&v, 0}};
    while (!todo.empty()) {
        const auto [node, enclosing] = todo.back();  // enclosing = containers around node
        todo.pop_back();
        if (enclosing > limit) return limit + 1;
        deepest = std::max(deepest, enclosing);
        if (!node->is_structured()) continue;
        for (const auto& child : *node) todo.emplace_back(&child, enclosing + 1);
    }
    return deepest;
}

OrderedJson ParsedMessage::to_openai_json(std::string_view id_prefix) const {
    OrderedJson j = OrderedJson::object();
    j["role"] = "assistant";
    j["content"] = content.empty() && !tool_calls.empty() ? OrderedJson(nullptr) : OrderedJson(content);
    if (!reasoning_content.empty()) j["reasoning_content"] = reasoning_content;
    if (!tool_calls.empty()) {
        OrderedJson calls = OrderedJson::array();
        for (std::size_t i = 0; i < tool_calls.size(); ++i) {
            calls.push_back({{"id", std::string(id_prefix) + std::to_string(i)},
                             {"type", "function"},
                             {"function", {{"name", tool_calls[i].name}, {"arguments", tool_calls[i].arguments.dump()}}}});
        }
        j["tool_calls"] = std::move(calls);
    }
    return j;
}

OutputParser::OutputParser(ParserOptions options)
    : opt_(std::move(options)),
      state_(opt_.starts_in_reasoning ? State::Reasoning : State::Content),
      output_start_(!opt_.starts_in_reasoning) {}

void OutputParser::emit(OutputEvent::Kind kind, std::string_view text, std::vector<OutputEvent>& out) {
    if (text.empty()) return;
    if (segment_start_) {
        text.remove_prefix(skip_ws(text, 0));
        if (text.empty()) return;
        segment_start_ = false;
    }
    const std::size_t body = trim_end(text);
    if (body == 0) {
        held_ws_.append(text);
        return;
    }
    std::string delta = std::move(held_ws_);
    held_ws_.assign(text.substr(body));
    std::string& field = kind == OutputEvent::Kind::ReasoningDelta ? msg_.reasoning_content : msg_.content;
    if (kind == OutputEvent::Kind::ContentDelta && field_needs_separator_) {
        if (!msg_.content.empty()) delta.insert(0, kSegmentSeparator);
        field_needs_separator_ = false;
    }
    delta.append(text.substr(0, body));
    field += delta;
    if (!out.empty() && out.back().kind == kind) {
        out.back().text += delta;
    } else {
        out.push_back({kind, std::move(delta), 0});
    }
}

void OutputParser::end_segment() {
    held_ws_.clear();
    segment_start_ = true;
}

void OutputParser::finish_tool_block(std::string_view block, std::vector<OutputEvent>& out) {
    const auto fail = [&](std::string_view why) {
        msg_.warnings.push_back("malformed tool call (" + std::string(why) + "); kept as content");
        end_segment();
        field_needs_separator_ = true;
        std::string raw;
        raw.append(kToolOpen).append(block).append(kToolClose);
        emit(OutputEvent::Kind::ContentDelta, raw, out);
        end_segment();
        field_needs_separator_ = true;  // content after the block is a new segment
    };
    std::size_t i = skip_ws(block, 0);
    if (block.substr(i, kFunctionOpen.size()) != kFunctionOpen) return fail("missing <function=");
    i += kFunctionOpen.size();
    const std::size_t name_end = block.find('>', i);
    if (name_end == std::string_view::npos) return fail("unterminated function name");
    ToolCall tc;
    tc.name = std::string(trim(block.substr(i, name_end - i)));
    if (tc.name.empty() || tc.name.find('\n') != std::string::npos) return fail("bad function name");
    i = name_end + 1;
    while (true) {
        i = skip_ws(block, i);
        if (i >= block.size()) break;  // tolerate a missing </function>
        if (block.substr(i, kFunctionClose.size()) == kFunctionClose) {
            i = skip_ws(block, i + kFunctionClose.size());
            if (i != block.size()) return fail("text after </function>");
            break;
        }
        if (block.substr(i, kParamOpen.size()) != kParamOpen) return fail("expected <parameter=");
        i += kParamOpen.size();
        const std::size_t pe = block.find('>', i);
        if (pe == std::string_view::npos) return fail("unterminated parameter name");
        const std::string pname(trim(block.substr(i, pe - i)));
        if (pname.empty()) return fail("empty parameter name");
        i = pe + 1;
        if (i < block.size() && block[i] == '\n') ++i;
        const std::size_t ve = block.find(kParamClose, i);
        if (ve == std::string_view::npos) return fail("unterminated parameter value");
        std::string_view value = block.substr(i, ve - i);
        if (!value.empty() && value.back() == '\n') value.remove_suffix(1);
        i = ve + kParamClose.size();
        tc.arguments[pname] = convert_value(param_schema(opt_.tools, tc.name, pname), value);
    }
    msg_.tool_calls.push_back(std::move(tc));
    out.push_back({OutputEvent::Kind::ToolCall, {}, msg_.tool_calls.size() - 1});
    field_needs_separator_ = true;
}

void OutputParser::process(std::vector<OutputEvent>& out, bool final) {
    while (true) {
        switch (state_) {
            case State::Reasoning: {
                const std::size_t p = buf_.find(kThinkClose);
                if (p != std::string::npos) {
                    emit(OutputEvent::Kind::ReasoningDelta, std::string_view(buf_).substr(0, p), out);
                    end_segment();
                    buf_.erase(0, p + kThinkClose.size());
                    state_ = State::Content;
                    continue;
                }
                const std::size_t keep = final ? 0 : partial_tag_suffix(buf_, kThinkClose);
                emit(OutputEvent::Kind::ReasoningDelta, std::string_view(buf_).substr(0, buf_.size() - keep), out);
                buf_.erase(0, buf_.size() - keep);
                return;
            }
            case State::Content: {
                if (output_start_) {
                    const std::size_t w = skip_ws(buf_, 0);
                    const std::string_view rest = std::string_view(buf_).substr(w);
                    if (rest.substr(0, kThinkOpen.size()) == kThinkOpen) {
                        buf_.erase(0, w + kThinkOpen.size());
                        output_start_ = false;
                        end_segment();
                        state_ = State::Reasoning;
                        continue;
                    }
                    if (!final && kThinkOpen.substr(0, rest.size()) == rest) return;  // undecided yet
                    output_start_ = false;
                }
                const std::size_t p = buf_.find(kToolOpen);
                if (p != std::string::npos) {
                    emit(OutputEvent::Kind::ContentDelta, std::string_view(buf_).substr(0, p), out);
                    end_segment();
                    buf_.erase(0, p + kToolOpen.size());
                    state_ = State::ToolCall;
                    continue;
                }
                const std::size_t keep = final ? 0 : partial_tag_suffix(buf_, kToolOpen);
                emit(OutputEvent::Kind::ContentDelta, std::string_view(buf_).substr(0, buf_.size() - keep), out);
                buf_.erase(0, buf_.size() - keep);
                return;
            }
            case State::ToolCall: {
                const std::size_t p = buf_.find(kToolClose);
                if (p != std::string::npos) {
                    const std::string block = buf_.substr(0, p);
                    buf_.erase(0, p + kToolClose.size());
                    finish_tool_block(block, out);
                    end_segment();
                    state_ = State::Content;
                    continue;
                }
                if (final) {
                    msg_.warnings.push_back("unterminated <tool_call> at end of output; kept as content");
                    field_needs_separator_ = true;
                    std::string raw;
                    raw.append(kToolOpen).append(buf_);
                    buf_.clear();
                    emit(OutputEvent::Kind::ContentDelta, raw, out);
                    state_ = State::Content;
                }
                return;
            }
        }
    }
}

std::vector<OutputEvent> OutputParser::feed(std::string_view piece) {
    std::vector<OutputEvent> out;
    // Only whole UTF-8 sequences enter the state machine, so whitespace classification never
    // sees half a codepoint; an incomplete tail waits for the next piece (or finish()).
    utf8_tail_.append(piece);
    const std::size_t hold = incomplete_utf8_tail(utf8_tail_);
    buf_.append(utf8_tail_, 0, utf8_tail_.size() - hold);
    utf8_tail_.erase(0, utf8_tail_.size() - hold);
    process(out, false);
    return out;
}

std::vector<OutputEvent> OutputParser::finish() {
    std::vector<OutputEvent> out;
    buf_.append(utf8_tail_);
    utf8_tail_.clear();
    process(out, true);
    end_segment();
    return out;
}

ParsedMessage OutputParser::parse(std::string_view text, const ParserOptions& options) {
    OutputParser p(options);
    (void)p.feed(text);
    (void)p.finish();
    return p.msg_;
}

bool prompt_ends_in_reasoning(std::string_view prompt) noexcept {
    return prompt.ends_with("<think>\n");
}

}  // namespace halo::chat
