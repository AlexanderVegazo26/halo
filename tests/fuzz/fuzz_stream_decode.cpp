// Fuzz target: the streaming detokenizer (tokenizer::StreamDecoder) and the SSE `data:` frame
// each streamed piece goes out in (api::dump, as SseWriter writes it).
// Input layout: byte 0 = options (bit 0 skip_special), the rest = token ids: a byte b < 0xFF is
// the byte token b; 0xFF n is the extra token n (0xFF 0xFF = the byte token 0xFF). The vocab is
// FakeEngine's synthetic one plus tokens that split or break UTF-8 (a partial U+20AC, emoji
// halves, a lone continuation byte, an overlong and a surrogate encoding) and tokens holding the
// SSE delimiters ("\n\n", "\r\n", "data: ", a Control token ending in "\n\n").
// Invariants, for any id sequence:
//   - push()/flush() never throw;
//   - every push() and the flush() return valid UTF-8 (no character split across two SSE
//     events, so no U+FFFD a client did not get from the model);
//   - at most 3 bytes (an incomplete UTF-8 prefix) are ever held back, none after flush();
//   - the concatenated pieces equal Tokenizer::decode of all ids (same skip_special);
//   - each piece's frame "data: <json>\n\n" has no CR or LF before its final "\n\n" (one event
//     per piece, whatever the model wrote), and its JSON parses back to exactly the piece.
// Seeds: tests/fuzz/corpus/stream_decode.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "api/json_util.h"
#include "fake_engine.h"
#include "fuzz_check.h"
#include "halo/tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace {

using halo::tokenizer::StreamDecoder;
using halo::tokenizer::Tokenizer;
using halo::tokenizer::TokenType;
using halo::tokenizer::VocabSpec;

std::size_t g_extras = 0;  // set once with the vocab

Tokenizer make_tokenizer() {
    VocabSpec s = halo::test::synthetic_vocab();
    const std::vector<std::string> bytes(s.tokens.begin(), s.tokens.begin() + 256);
    const auto normal = [&](std::string_view raw) {  // raw bytes -> GPT-2 mapped token
        std::string t;
        for (const char c : raw) t += bytes[static_cast<unsigned char>(c)];
        s.tokens.push_back(std::move(t));
        s.types.push_back(TokenType::Normal);
    };
    normal("\xe2\x82\xac");          // U+20AC whole
    normal("\xe2\x82");              // its first two bytes
    normal("\xac" "a");              // a lone continuation byte, then ASCII
    normal("\xf0\x9f");              // U+1F600 first half
    normal("\x98\x80");              // ... second half
    normal("\xf0\x9f\x98\x80 ok");   // whole, then ASCII
    normal("\xc0\xaf");              // overlong '/'
    normal("\xed\xa0\x80");          // UTF-16 surrogate D800
    normal("\n\n");                  // the SSE event delimiter inside content
    normal("\r\n");
    normal("data: [DONE]\n\n");      // a fake terminator inside content
    normal("\"\\\x7f");              // JSON-escaped characters + DEL
    s.tokens.emplace_back("<|nl|>\n\n");
    s.types.push_back(TokenType::Control);
    s.tokens.emplace_back("<u>\r");
    s.types.push_back(TokenType::UserDefined);
    g_extras = s.tokens.size() - 256;
    return Tokenizer::from_spec(s);
}

bool valid_utf8(const std::string& s) { return halo::tokenizer::unicode::to_valid_utf8(s) == s; }

void check_frame(const std::string& piece, std::size_t index) {
    const std::string json = halo::api::dump(halo::api::Json{{"content", piece}});
    const std::string frame = "data: " + json + "\n\n";
    const auto cut = frame.find_first_of("\r\n");
    if (cut != frame.size() - 2) {
        HALO_FUZZ_FAIL("piece %zu (%zu bytes): SSE frame has a CR/LF at byte %zu of %zu, before its final \\n\\n",
                       index, piece.size(), cut, frame.size());
    }
    const auto back = halo::api::Json::parse(json);
    if (back.at("content").get<std::string>() != piece) {
        HALO_FUZZ_FAIL("piece %zu (%zu bytes): SSE frame JSON does not parse back to the piece", index, piece.size());
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static const Tokenizer tok = make_tokenizer();
    if (size < 1) return 0;
    const bool skip_special = (data[0] & 1u) != 0;
    std::vector<std::int32_t> ids;
    for (std::size_t i = 1; i < size; ++i) {
        if (data[i] != 0xFF) {
            ids.push_back(data[i]);
        } else if (i + 1 < size) {
            const std::uint8_t n = data[++i];
            ids.push_back(n == 0xFF ? 0xFF : static_cast<std::int32_t>(256 + n % g_extras));
        }
    }
    try {
        StreamDecoder dec(tok, skip_special);
        std::string streamed;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const std::string piece = dec.push(ids[i]);
            if (!valid_utf8(piece)) {
                HALO_FUZZ_FAIL("push of id %d (token %zu of %zu) returned invalid UTF-8 (%zu bytes)", ids[i], i,
                               ids.size(), piece.size());
            }
            if (dec.pending_bytes() > 3) {
                HALO_FUZZ_FAIL("after id %d (token %zu) %zu bytes are held back, expected <= 3 (an incomplete prefix)",
                               ids[i], i, dec.pending_bytes());
            }
            check_frame(piece, i);
            streamed += piece;
        }
        const std::string tail = dec.flush();
        if (!valid_utf8(tail)) HALO_FUZZ_FAIL("flush returned invalid UTF-8 (%zu bytes)", tail.size());
        if (dec.pending_bytes() != 0) HALO_FUZZ_FAIL("%zu bytes still held after flush", dec.pending_bytes());
        check_frame(tail, ids.size());
        streamed += tail;
        const std::string whole = tok.decode(ids, skip_special);
        if (streamed != whole) {
            HALO_FUZZ_FAIL("streamed pieces (%zu bytes) differ from decode (%zu bytes) for %zu ids, skip_special=%d",
                           streamed.size(), whole.size(), ids.size(), skip_special ? 1 : 0);
        }
    } catch (const std::exception& e) {
        HALO_FUZZ_FAIL("stream decode threw: %s", e.what());
    }
    return 0;
}
