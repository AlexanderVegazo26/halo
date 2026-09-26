#pragma once
// Token sampler (FR-007, TRD §19 / §23) — the per-step LogitsProcessor the runtime calls.
// Implementation: src/sampling/sampler.cpp (WS-H).
//
// Input per step: either the full logits row (vocab_size floats) or a GPU top-k
// *prefiltered* candidate set (ids + logits, typically k <= 1024, TRD §19).
//
// Chain (fixed order; every step is a no-op at its neutral value):
//   1. penalties   repetition / presence / frequency over the last `penalty_last_n` tokens
//                  of `history` (llama.cpp `llama_sampler_penalties_apply` semantics, read
//                  from /root/llama.cpp/src/llama-sampler.cpp @ bd4f514db): for a token seen
//                  c > 0 times in the window,
//                      logit = logit <= 0 ? logit * repetition_penalty
//                                         : logit / repetition_penalty      (once, not per c)
//                      logit -= c * frequency_penalty + presence_penalty
//                  penalty_last_n: 0 = disabled, -1 = the whole history, n = last n entries.
//                  `history` is whatever the caller passes (the runtime decides whether the
//                  prompt is included; llama-server includes it).
//   2. grammar     structured output (json_schema / json_object): tokens the TokenMatcher
//                  disallows are removed (EOS only once the JSON is complete).
//   3. greedy      temperature <= 0: exact argmax of the penalized, masked logits, ties to
//                  the lowest id. Steps 4-8 are skipped and no random number is consumed.
//   4. temperature logit / temperature. NOTE: llama.cpp's default chain applies temperature
//                  *last*; this chain applies it before the truncation filters as TRD §23 /
//                  WS-H specify (HF order), so min_p / top_p / typical_p act on the
//                  temperature-scaled distribution.
//   5. top_k       keep the k largest (ties to the lower id); k <= 0 or k >= n: no-op.
//   6. typical_p   locally typical sampling (llama.cpp): with H the entropy, sort by
//                  |-log p - H| ascending (ties to lower id) and keep the shortest prefix
//                  whose cumulative probability is  > typical_p  (at least one). p >= 1: no-op.
//   7. top_p       sort by probability descending (ties to lower id) and keep the shortest
//                  prefix whose cumulative probability is  >= top_p  (at least one). p >= 1: no-op.
//   8. min_p       keep logit >= max_logit + log(min_p)  (p_i >= min_p * p_max). p <= 0: no-op.
//   9. draw        u = 53 random bits of std::mt19937_64 * 2^-53 in [0,1); candidates in id
//                  order, w_i = exp(l_i - max) (double); the first id whose cumulative
//                  weight exceeds u * sum(w). One engine draw per sampled token.
// Probabilities are always computed in double with the max subtracted (log-sum-exp), and
// every sum runs in a canonical order, so the result depends only on (params, seed, inputs)
// — never on thread count or input order — and a prefiltered call returns exactly the
// full-row token whenever exact_prefilter_ok() holds.
//
// Invalid input is a typed error, never a silent pick: NaN or +Inf logits -> Error(Kernel)
// (an upstream bug, as for ARGMAX, D-016); -Inf is a legal "impossible" logit; a row whose
// every logit is -Inf (or masked away) -> Error(Api); bad params -> Error(Api).
//
// Thread safety: a Sampler is per-sequence mutable state (RNG, grammar matcher): one sampler
// per sequence, not shared between threads. Distinct samplers may run concurrently; they may
// share one TokenVocab (immutable).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "halo/sampling/sampling.h"
#include "halo/sampling/structured.h"

namespace halo::tokenizer { class Tokenizer; }

namespace halo::sampling {

/// A GPU top-k prefiltered candidate set: ids[i] has logit logits[i]. Ids must be unique
/// and inside the vocabulary; order does not matter.
struct CandidateSet {
    std::span<const std::int32_t> ids;
    std::span<const float> logits;
};

struct SamplerConfig {
    /// Logits row length. 0 = tokenizer->vocab_size() (required when tokenizer is null).
    std::size_t vocab_size = 0;
    /// Real (tokenizer) vocabulary size for unstructured sampling (M2): rows at/after this
    /// index are GGUF LM-head padding and are never emitted. 0 = tokenizer->vocab_size(), or
    /// vocab_size (no clamp) when tokenizer is null.
    std::size_t emit_vocab_size = 0;
    /// End-of-generation ids for structured output (allowed only once the JSON is
    /// complete). Empty = tokenizer->eos(). Must be non-Normal tokens.
    std::vector<std::int32_t> eos_ids;
    /// Shared token trie for structured output. Building it for the 248K vocabulary costs
    /// a noticeable fraction of a second, so the runtime should build it once per tokenizer
    /// (TokenVocab::build) and pass it here. Null = built from the tokenizer on demand.
    std::shared_ptr<const TokenVocab> vocab;
    SchemaOptions schema;
};

/// Throws Error(Api) for NaN / out-of-range parameters: temperature must be finite;
/// top_k >= 0; top_p, min_p, typical_p in [0, 1]; repetition_penalty finite and > 0;
/// presence/frequency penalties finite; penalty_last_n >= -1; json_schema and json_object
/// are mutually exclusive.
void validate(const SamplingParams& p);

/// Any of the three penalties would change a logit (and penalty_last_n != 0).
[[nodiscard]] bool penalties_active(const SamplingParams& p) noexcept;
/// json_schema or json_object is set.
[[nodiscard]] bool is_structured(const SamplingParams& p) noexcept;

/// True when sampling from the top-k prefiltered candidates (k = prefilter size, ties to the
/// lowest id — the rule of cpu::top_k, which GPU prefilters must match) gives exactly the
/// full-row result for every logits row and seed:
///   * no active penalty — even a penalty that only demotes a candidate can let a token
///     outside the prefilter into the true top set, and negative presence / frequency or
///     repetition_penalty < 1 promote tokens the prefilter never saw;
///   * no structured output — the mask may remove candidates the prefilter kept;
///   * greedy (k >= 1), or 1 <= top_k <= k (top_k runs before every probability-dependent
///     filter, and temperature is monotone, so both paths keep the same set).
[[nodiscard]] bool exact_prefilter_ok(const SamplingParams& p, std::size_t k) noexcept;

/// The chain's individual steps, exposed so tests can check each against a direct
/// reference. They operate on an unordered working set; results are order-independent.
namespace chain {

struct Cand {
    std::int32_t id = 0;
    double logit = 0.0;
};

/// (id, count) for each distinct id in the last `last_n` entries of history (-1 = all,
/// 0 = none), sorted by id.
[[nodiscard]] std::vector<std::pair<std::int32_t, std::int32_t>> window_counts(
    std::span<const std::int32_t> history, int last_n);
/// Step 1. `cands` must be sorted by id.
void apply_penalties(std::vector<Cand>& cands, const SamplingParams& p,
                     std::span<const std::pair<std::int32_t, std::int32_t>> counts);
void apply_temperature(std::vector<Cand>& cands, float temperature);
void apply_top_k(std::vector<Cand>& cands, int k);
void apply_typical(std::vector<Cand>& cands, float p);
void apply_top_p(std::vector<Cand>& cands, float p);
void apply_min_p(std::vector<Cand>& cands, float p);
/// Softmax probabilities of `cands` in their current order (log-sum-exp, double).
[[nodiscard]] std::vector<double> probabilities(const std::vector<Cand>& cands);
/// Argmax, ties to the lowest id. Requires a non-empty set.
[[nodiscard]] std::int32_t greedy(const std::vector<Cand>& cands);
/// Step 9 with a given u in [0, 1). Sorts `cands` by id.
[[nodiscard]] std::int32_t draw(std::vector<Cand>& cands, double u);
/// 53-bit uniform in [0, 1) from one engine output.
[[nodiscard]] double uniform01(std::mt19937_64& rng);

}  // namespace chain

class Sampler {
public:
    /// Validates params (see validate()). With json_schema / json_object set, compiles the
    /// grammar (Error Api / Unsupported for bad or unsupported schemas) and requires a
    /// tokenizer (or cfg.vocab) and EOS ids. Seed: params.seed, else std::random_device.
    static Sampler create(const SamplingParams& params, const tokenizer::Tokenizer* tokenizer,
                          SamplerConfig cfg = {});

    Sampler(Sampler&&) noexcept;
    Sampler& operator=(Sampler&&) noexcept;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    ~Sampler();

    /// Samples from the full logits row (size vocab_size()). Does not advance the grammar:
    /// call accept() with the token the sequence actually keeps.
    [[nodiscard]] std::int32_t sample(std::span<const float> logits, std::span<const std::int32_t> history);
    /// Samples from a prefiltered candidate set. Throws Error(Api) if the grammar mask
    /// removes every candidate; use try_sample to fall back to the full row instead.
    [[nodiscard]] std::int32_t sample(const CandidateSet& candidates, std::span<const std::int32_t> history);
    /// As sample(candidates), but returns nullopt (consuming no randomness) when structured
    /// output allows none of the candidates — the caller must then sample the full row.
    [[nodiscard]] std::optional<std::int32_t> try_sample(const CandidateSet& candidates,
                                                         std::span<const std::int32_t> history);

    /// Advances the grammar by the token kept for this sequence (no-op without structured
    /// output). Throws Error(Api) for a token the grammar does not allow (state unchanged).
    void accept(std::int32_t token);
    /// Back to the initial grammar state and the initial RNG state (same seed).
    void reset();

    [[nodiscard]] bool structured() const noexcept { return matcher_ != nullptr; }
    /// Structured: the text so far is a complete JSON document (EOS allowed).
    [[nodiscard]] bool grammar_complete() const;
    /// Structured: an EOS id was accepted.
    [[nodiscard]] bool grammar_finished() const noexcept;
    /// Null without structured output. For diagnostics / speculative rollback.
    [[nodiscard]] TokenMatcher* matcher() noexcept { return matcher_.get(); }

    [[nodiscard]] const SamplingParams& params() const noexcept { return params_; }
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return vocab_size_; }

private:
    Sampler(const SamplingParams& params, std::size_t vocab_size, std::size_t emit_vocab_size, std::uint64_t seed,
            std::unique_ptr<TokenMatcher> matcher);
    [[nodiscard]] std::int32_t run_chain(std::span<const std::int32_t> history);

    SamplingParams params_;
    std::size_t vocab_size_;
    /// Real (tokenizer) vocabulary size <= vocab_size_. Rows in [emit_vocab_size_, vocab_size_)
    /// are GGUF LM-head padding (M2): never valid to emit in unstructured generation, since
    /// TokenVocab-based structured output already excludes them via mask_has.
    std::size_t emit_vocab_size_;
    std::uint64_t seed_;
    std::mt19937_64 rng_;
    std::unique_ptr<TokenMatcher> matcher_;
    std::vector<chain::Cand> work_;
};

}  // namespace halo::sampling
