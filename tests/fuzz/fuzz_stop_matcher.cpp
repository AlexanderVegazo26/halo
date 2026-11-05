// Fuzz target: the streaming stop-sequence matcher (api::StopMatcher, client `stop` strings).
// Input layout: byte 0 = chunking seed, byte 1 = number of stop strings (low 3 bits, 0..7),
// then each stop as a length byte (low 4 bits, 0..15) followed by that many bytes, the rest =
// model content. Invariants, for any bytes and any split:
//   - push()/flush() never throw;
//   - split-invariant: the seed's chunks emit exactly what a byte-at-a-time stream emits, and
//     match the same stop;
//   - no match: everything pushed comes back out, unchanged and in order;
//   - match: the emitted text + the matched stop is a prefix of the content, the matched stop is
//     one of the non-empty client stops, and no stop occurs that ends earlier (ties: the longer);
//   - after a match, push() and flush() return "".
// Seeds: tests/fuzz/corpus/stop_matcher.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "api/generation.h"
#include "fuzz_check.h"

namespace {

using halo::api::StopMatcher;

struct Result {
    std::string emitted;
    std::optional<std::string> matched;
};

template <typename NextChunk>
Result run(const std::vector<std::string>& stops, std::string_view text, NextChunk next) {
    StopMatcher m(stops);
    Result r;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t n = std::min(next(), text.size() - pos);
        const bool was_matched = m.matched().has_value();
        const std::string out = m.push(text.substr(pos, n));
        if (was_matched && !out.empty()) HALO_FUZZ_FAIL("push after a match returned %zu bytes", out.size());
        r.emitted += out;
        pos += n;
    }
    const std::string tail = m.flush();
    if (m.matched() && !tail.empty()) HALO_FUZZ_FAIL("flush after a match returned %zu bytes", tail.size());
    r.emitted += tail;
    r.matched = m.matched();
    return r;
}

std::string show(const std::optional<std::string>& s) { return s ? "\"" + *s + "\"" : std::string("<none>"); }

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 2) return 0;
    std::uint32_t rng = 0x9E3779B9u ^ data[0];
    const std::size_t n_stops = data[1] & 7u;
    std::size_t pos = 2;
    std::vector<std::string> stops;
    for (std::size_t i = 0; i < n_stops && pos < size; ++i) {
        const std::size_t want = data[pos++] & 15u;
        const std::size_t len = std::min(want, size - pos);
        stops.emplace_back(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
    }
    const std::string_view text(reinterpret_cast<const char*>(data + pos), size - pos);
    try {
        const Result bytewise = run(stops, text, [] { return std::size_t{1}; });
        const Result chunked = run(stops, text, [&] {
            rng = rng * 1664525u + 1013904223u;  // LCG: chunk sizes 0..16 from the seed byte
            return static_cast<std::size_t>((rng >> 24) % 17);
        });
        if (chunked.emitted != bytewise.emitted || chunked.matched != bytewise.matched) {
            HALO_FUZZ_FAIL("chunked stream differs from byte-at-a-time\nbyte-at-a-time: %zu bytes, stop %s\n"
                           "chunked:        %zu bytes, stop %s",
                           bytewise.emitted.size(), show(bytewise.matched).c_str(), chunked.emitted.size(),
                           show(chunked.matched).c_str());
        }
        if (!chunked.matched) {
            if (chunked.emitted != text) {
                HALO_FUZZ_FAIL("no stop matched but emitted %zu of %zu bytes differently", chunked.emitted.size(),
                               text.size());
            }
            for (const auto& s : stops) {
                if (!s.empty() && text.find(s) != std::string_view::npos) {
                    HALO_FUZZ_FAIL("stop %s occurs in the content but was not matched", show(s).c_str());
                }
            }
            return 0;
        }
        const std::string& hit = *chunked.matched;
        const std::size_t end = chunked.emitted.size() + hit.size();
        if (hit.empty() || std::find(stops.begin(), stops.end(), hit) == stops.end()) {
            HALO_FUZZ_FAIL("matched %s is not one of the non-empty client stops", show(hit).c_str());
        }
        if (text.substr(0, end) != chunked.emitted + hit) {
            HALO_FUZZ_FAIL("emitted text + matched stop %s is not a prefix of the content", show(hit).c_str());
        }
        for (const auto& s : stops) {
            if (s.empty()) continue;
            const auto p = text.find(s);
            if (p == std::string_view::npos) continue;
            const std::size_t e = p + s.size();
            if (e < end || (e == end && s.size() > hit.size())) {
                HALO_FUZZ_FAIL("stop %s ends at %zu, before (or as long as and longer than) the matched %s at %zu",
                               show(s).c_str(), e, show(hit).c_str(), end);
            }
        }
    } catch (const std::exception& e) {
        HALO_FUZZ_FAIL("stop matcher threw: %s", e.what());
    }
    return 0;
}
