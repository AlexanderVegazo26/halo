#pragma once
// Test helper: shortest completion over the byte matcher runtime (white-box).

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>

#include "sampling/grammar_impl.h"

namespace halo::sampling::test {

/// Shortest byte string leading from `from` to an accepting state (breadth-first over the
/// interned matcher states, bytes tried in ascending order). nullopt if none is found
/// within `max_states` visited states — for these grammars that means a live state that
/// can never complete, which is itself a defect.
inline std::optional<std::string> shortest_completion(detail::Runtime& rt, std::int32_t from,
                                                      std::size_t max_states = 200000) {
    struct Prev {
        std::int32_t state;
        std::uint8_t byte;
    };
    std::unordered_map<std::int32_t, Prev> prev{{from, {from, 0}}};
    std::deque<std::int32_t> queue{from};
    while (!queue.empty() && prev.size() <= max_states) {
        const std::int32_t s = queue.front();
        queue.pop_front();
        if (rt.accepting(s)) {
            std::string path;
            for (std::int32_t t = s; t != from; t = prev.at(t).state) path.push_back(static_cast<char>(prev.at(t).byte));
            return std::string(path.rbegin(), path.rend());
        }
        const auto next = rt.next_bytes(s);
        for (unsigned b = 0; b < 256; ++b) {
            if (!detail::byteset_has(next, static_cast<std::uint8_t>(b))) continue;
            const std::int32_t t = rt.step(s, static_cast<std::uint8_t>(b));
            if (t == detail::kDead || prev.contains(t)) continue;
            prev.emplace(t, Prev{s, static_cast<std::uint8_t>(b)});
            queue.push_back(t);
        }
    }
    return std::nullopt;
}

}  // namespace halo::sampling::test
