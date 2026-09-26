#pragma once
// Sampling contract (FR-007, TRD §23). Implementation: src/sampling (WS-H).

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace halo {

struct SamplingParams {
    float temperature = 1.0f;       // 0 => greedy
    int top_k = 0;                  // 0 = disabled
    float top_p = 1.0f;
    float min_p = 0.0f;
    float typical_p = 1.0f;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int penalty_last_n = 64;
    std::optional<std::uint64_t> seed;
    // JSON-Schema structured output (constrained decoding via token masks, TRD §25).
    std::optional<std::string> json_schema;
    bool json_object = false;       // any valid JSON object

    [[nodiscard]] bool greedy() const noexcept { return temperature == 0.0f; }
};

}  // namespace halo
