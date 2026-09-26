// Chat prompt construction with special-token injection protection. See prompt.h.

#include "halo/api/prompt.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <random>

#include "halo/core/error.h"
#include "halo/template/output_parser.h"
#include "halo/tokenizer/tokenizer.h"

namespace halo::api {

namespace {

using chat::OrderedJson;
using tokenizer::TokenType;

// Placeholder: "HLX" + 16 hex nonce + "Q" + decimal index + "Z". ASCII alphanumerics only, so
// the template's `trim` and `tojson` leave it untouched and no BPE merge can involve an
// added token.
constexpr std::string_view kPlaceholderHead = "HLX";

struct Literals {
    std::vector<const std::string*> sorted;  // longest first (leftmost-longest matching)
};

Literals make_literals(const std::vector<std::string>& a, const std::vector<std::string>* b) {
    Literals l;
    for (const auto& s : a) l.sorted.push_back(&s);
    if (b != nullptr) {
        for (const auto& s : *b) l.sorted.push_back(&s);
    }
    std::ranges::stable_sort(l.sorted, [](const std::string* x, const std::string* y) { return x->size() > y->size(); });
    return l;
}

/// Longest literal matching `s` at `pos`, or nullptr.
const std::string* match_at(std::string_view s, std::size_t pos, const Literals& lits) {
    for (const std::string* lit : lits.sorted) {
        if (!lit->empty() && s.substr(pos).starts_with(*lit)) return lit;
    }
    return nullptr;
}

class Escaper {
public:
    Escaper(std::string nonce, std::vector<std::string>& table) : nonce_(std::move(nonce)), table_(table) {}

    std::string escape(std::string_view s, const Literals& lits) {
        std::string out;
        out.reserve(s.size());
        std::size_t i = 0;
        while (i < s.size()) {
            if (const std::string* lit = match_at(s, i, lits)) {
                out += placeholder(*lit);
                i += lit->size();
            } else {
                out.push_back(s[i]);
                ++i;
            }
        }
        return out;
    }

    /// Escapes every string (and object key) in `v`, recursively.
    void escape_json(OrderedJson& v, const Literals& lits) {
        if (v.is_string()) {
            v = escape(v.get_ref<const std::string&>(), lits);
        } else if (v.is_array()) {
            for (auto& e : v) escape_json(e, lits);
        } else if (v.is_object()) {
            OrderedJson rebuilt = OrderedJson::object();
            for (auto it = v.begin(); it != v.end(); ++it) {
                OrderedJson child = it.value();
                escape_json(child, lits);
                rebuilt[escape(it.key(), lits)] = std::move(child);
            }
            v = std::move(rebuilt);
        }
    }

    [[nodiscard]] std::size_t replaced() const noexcept { return replaced_; }

private:
    std::string placeholder(const std::string& lit) {
        std::size_t idx = 0;
        const auto it = std::ranges::find(table_, lit);
        if (it == table_.end()) {
            table_.push_back(lit);
            idx = table_.size() - 1;
        } else {
            idx = static_cast<std::size_t>(it - table_.begin());
        }
        ++replaced_;
        return std::string(kPlaceholderHead) + nonce_ + "Q" + std::to_string(idx) + "Z";
    }

    std::string nonce_;
    std::vector<std::string>& table_;
    std::size_t replaced_ = 0;
};

std::string random_nonce() {
    std::random_device rd;
    const std::uint64_t v = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    std::array<char, 17> buf{};
    constexpr std::string_view hex = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) buf[static_cast<std::size_t>(i)] = hex[(v >> (4 * i)) & 0xF];
    return {buf.data(), 16};
}

bool is_added(const tokenizer::Tokenizer& tok, std::int32_t id) {
    const auto t = tok.token_type(id);
    return t == TokenType::Control || t == TokenType::UserDefined;
}

void append(std::vector<std::int32_t>& out, const std::vector<std::int32_t>& v) {
    out.insert(out.end(), v.begin(), v.end());
}

/// Tokenizes one literal as text: its natural text tokenization when that contains no added
/// token, otherwise first byte + the rest (recursively).
void literal_as_text(const tokenizer::Tokenizer& tok, const SpecialTokens& sp, std::string_view lit,
                     std::vector<std::int32_t>& out) {
    auto ids = tok.encode(lit, false);
    if (std::ranges::none_of(ids, [&](std::int32_t id) { return is_added(tok, id); })) {
        append(out, ids);
        return;
    }
    append(out, tok.encode(lit.substr(0, 1), false));
    append(out, encode_as_text(tok, sp, lit.substr(1)));
}

}  // namespace

SpecialTokens::SpecialTokens(const tokenizer::Tokenizer& tok) {
    const std::size_t n = tok.vocab_size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto id = static_cast<std::int32_t>(i);
        const auto t = tok.token_type(id);
        if (t == TokenType::Control) {
            if (!tok.token_to_piece(id).empty()) control_.push_back(tok.token_to_piece(id));
        } else if (t == TokenType::UserDefined) {
            if (!tok.token_to_piece(id).empty()) user_defined_.push_back(tok.token_to_piece(id));
        }
    }
    think_start_ = tok.piece_to_id("<think>");
    think_end_ = tok.piece_to_id("</think>");
}

std::vector<std::int32_t> encode_as_text(const tokenizer::Tokenizer& tok, const SpecialTokens& sp,
                                         std::string_view text) {
    const Literals all = make_literals(sp.control(), &sp.user_defined());
    std::vector<std::int32_t> out;
    std::size_t seg = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        if (const std::string* lit = match_at(text, i, all)) {
            if (i > seg) append(out, tok.encode(text.substr(seg, i - seg), false));
            literal_as_text(tok, sp, *lit, out);
            i += lit->size();
            seg = i;
        } else {
            ++i;
        }
    }
    if (seg < text.size()) append(out, tok.encode(text.substr(seg), false));
    return out;
}

ChatPrompt build_chat_prompt(const tokenizer::Tokenizer& tok, const SpecialTokens& sp,
                             const chat::ChatTemplate& tmpl, const OrderedJson& messages,
                             const OrderedJson& tools, const chat::RenderOptions& options) {
    HALO_CHECK(messages.is_array(), ErrorCode::Api, "messages must be an array");
    // M9: escape_json below recursively walks/copies messages and tools before render() ever
    // runs its own depth check (chat_template.cpp), and this function is also a public library
    // entry point that non-HTTP embedders can call without the API's body-parse depth cap. Guard
    // here too so the "defense in depth" claim is actually true for the first recursive pass.
    HALO_CHECK(chat::json_nesting_depth(messages) <= chat::kMaxJsonDepth, ErrorCode::Api,
               "messages nests deeper than {} levels", chat::kMaxJsonDepth);
    HALO_CHECK(chat::json_nesting_depth(tools) <= chat::kMaxJsonDepth, ErrorCode::Api,
               "tools nests deeper than {} levels", chat::kMaxJsonDepth);
    const Literals control_only = make_literals(sp.control(), nullptr);
    const Literals control_and_markup = make_literals(sp.control(), &sp.user_defined());

    // Pick a nonce that does not occur anywhere in the input.
    const std::string input_dump = messages.dump(-1, ' ', false, OrderedJson::error_handler_t::replace) +
                                   tools.dump(-1, ' ', false, OrderedJson::error_handler_t::replace);
    std::string nonce;
    for (int attempt = 0; attempt < 8; ++attempt) {
        std::string candidate = random_nonce();
        if (input_dump.find(std::string(kPlaceholderHead) + candidate) == std::string::npos) {
            nonce = std::move(candidate);
            break;
        }
    }
    HALO_CHECK(!nonce.empty(), ErrorCode::Api, "could not choose an escape nonce absent from the request");

    std::vector<std::string> table;
    Escaper esc(nonce, table);
    OrderedJson msgs = messages;
    for (auto& m : msgs) {
        if (!m.is_object()) continue;  // the template reports malformed messages
        const bool assistant = m.contains("role") && m["role"] == "assistant";
        esc.escape_json(m, assistant ? control_only : control_and_markup);
    }
    OrderedJson tl = tools;
    esc.escape_json(tl, control_and_markup);

    ChatPrompt out;
    const std::string rendered = tmpl.apply(msgs, tl, options);
    out.starts_in_reasoning = chat::prompt_ends_in_reasoning(rendered);

    if (esc.replaced() == 0) {  // fast path: identical to HF apply_chat_template + tokenize
        out.tokens = tok.encode(rendered, true);
        out.text = rendered;
        return out;
    }

    const std::string head = std::string(kPlaceholderHead) + nonce + "Q";
    std::size_t seg = 0;
    std::size_t pos = 0;
    while ((pos = rendered.find(head, pos)) != std::string::npos) {
        std::size_t j = pos + head.size();
        std::size_t idx = 0;
        const auto [ptr, ec] = std::from_chars(rendered.data() + j, rendered.data() + rendered.size(), idx);
        const auto end = static_cast<std::size_t>(ptr - rendered.data());
        if (ec != std::errc() || end >= rendered.size() || rendered[end] != 'Z' || idx >= table.size()) {
            ++pos;  // not one of ours (cannot happen: nonce is absent from the input)
            continue;
        }
        j = end + 1;
        if (pos > seg) {
            const std::string_view part(rendered.data() + seg, pos - seg);
            append(out.tokens, tok.encode(part, true));
            out.text += part;
        }
        literal_as_text(tok, sp, table[idx], out.tokens);
        out.text += table[idx];
        ++out.neutralized_literals;
        seg = pos = j;
    }
    if (seg < rendered.size()) {
        const std::string_view part(rendered.data() + seg, rendered.size() - seg);
        append(out.tokens, tok.encode(part, true));
        out.text += part;
    }
    return out;
}

}  // namespace halo::api
