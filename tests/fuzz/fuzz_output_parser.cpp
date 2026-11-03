// Fuzz target: the streaming model-output parser (reasoning / content / <tool_call> blocks).
// Input layout: byte 0 = options (bit 0 starts_in_reasoning, bit 1 typed tools schema),
// byte 1 = chunking seed, the rest = model output. Invariants:
//   - feed()/finish() never throw, for any bytes and any split;
//   - streaming is split-invariant: feeding the output in the seed's chunks gives exactly the
//     message (to_openai_json + warnings) that the one-shot OutputParser::parse gives;
//   - every ToolCall event indexes an existing tool call.
// Seeds: tests/fuzz/corpus/output_parser.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>

#include "fuzz_check.h"
#include "halo/template/output_parser.h"

namespace {

using halo::chat::OutputEvent;
using halo::chat::OutputParser;
using halo::chat::ParsedMessage;
using halo::chat::ParserOptions;

nlohmann::ordered_json typed_tools() {
    return nlohmann::ordered_json::parse(R"([{"type":"function","function":{"name":"f","parameters":{
        "type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"},"o":{"type":["string","null"]},
        "a":{"type":"array"}}}}}])");
}

std::string fingerprint(const ParsedMessage& m) {
    // replace: model output may end in a cut UTF-8 sequence; the server dumps the same way.
    std::string s = m.to_openai_json().dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
    for (const auto& w : m.warnings) s += "\nwarning: " + w;
    return s;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 2) return 0;
    ParserOptions opt;
    opt.starts_in_reasoning = (data[0] & 1u) != 0;
    if ((data[0] & 2u) != 0) {
        static const auto tools = typed_tools();
        opt.tools = tools;
    }
    std::uint32_t rng = 0x9E3779B9u ^ data[1];
    const std::string_view text(reinterpret_cast<const char*>(data + 2), size - 2);
    try {
        const ParsedMessage whole = OutputParser::parse(text, opt);
        OutputParser p(opt);
        std::size_t pos = 0;
        const auto check_events = [&](const std::vector<OutputEvent>& evs) {
            for (const auto& ev : evs) {
                if (ev.kind == OutputEvent::Kind::ToolCall && ev.tool_index >= p.message().tool_calls.size()) {
                    HALO_FUZZ_FAIL("ToolCall event index %zu but only %zu tool calls", ev.tool_index,
                                   p.message().tool_calls.size());
                }
            }
        };
        while (pos < text.size()) {
            rng = rng * 1664525u + 1013904223u;  // LCG: chunk sizes 0..16 from the seed byte
            const std::size_t n = std::min<std::size_t>((rng >> 24) % 17, text.size() - pos);
            check_events(p.feed(text.substr(pos, n)));
            pos += n;
        }
        check_events(p.finish());
        const std::string a = fingerprint(whole), b = fingerprint(p.message());
        if (a != b) HALO_FUZZ_FAIL("chunked parse differs from one-shot parse\none-shot: %s\nchunked:  %s", a.c_str(), b.c_str());
    } catch (const std::exception& e) {
        HALO_FUZZ_FAIL("output parser threw: %s", e.what());
    }
    return 0;
}
