#pragma once
// Qwen3.8 ("qwen35") CPU reference forward pass (DECISIONS.md D-004, D-005, D-012, D-016).
//
// Built from a NormalizedModel: every projection is a cpu::WeightMatrix whose rows are
// dequantized on demand from the GGUF bytes (F32 tensors are viewed in place), norms and
// other small vectors are dequantized once. The forward is composed from halo::cpu ops.
//
// Batch-shaped entry point (ARCHITECTURE "Backend::forward", review H-1/H-2): one call
// runs S sequences, each with its own token rows, KV cache and GDN state; every matmul
// runs once over the rows of all sequences (one weight pass per call). Per-sequence ops
// (conv, GDN, attention) loop over sequences. Every per-row computation is independent of
// which other rows share the call, so a sequence's results are bit-identical whether it
// runs alone or batched (tested).
//
// GDN path: prefill uses the chunked form (chunk 64), decode and speculative verification
// the recurrent form. Recurrent calls are bit-identical however the rows are split, which
// is what makes "verify + rollback to r" == "decode r tokens" hold exactly (D-012).
//
// Failure atomicity: all inputs are validated and every KV reservation is made before any
// state is modified, so a thrown Error (bad token, KV pool exhausted) leaves every
// sequence's KV and GDN state as they were.
//
// Not implemented here (see the WS-G report): the generic Backend virtual interface with
// variants()/run_op() and the "forward == composed run_op" differential test.
//
// Thread-safety: a Qwen35 is immutable after construction; forward() may be called from
// several threads only with disjoint sequences (it shares the ThreadPool, which serializes).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/thread_pool.h"
#include "halo/backends/cpu/weight_matrix.h"
#include "halo/kv_cache/paged_kv.h"
#include "halo/model/model.h"
#include "halo/state/gdn_state.h"
#include "halo/tokenizer/tokenizer.h"

namespace halo::models {

// ---------------------------------------------------------------------------------------
// Adapters (code review S-4)
// ---------------------------------------------------------------------------------------

/// A 2-D GGUF tensor (ggml ne = {K, N}) as an N x K WeightMatrix over rows
/// [first_row, first_row + n_rows) (n_rows = all remaining when nullopt). F32 tensors are
/// viewed in place (alignment checked); other types dequantize rows on demand with
/// tensor::dequantize_row over row_bytes(type, K). The model must outlive the matrix.
/// Throws Error(Model) for a non-2-D tensor or a row range outside it, Error(Unsupported)
/// for a type without dequantization.
[[nodiscard]] cpu::WeightMatrix weight_matrix(const model::WeightRef& w, std::size_t first_row = 0,
                                              std::optional<std::size_t> n_rows = std::nullopt);

/// Every element of a (small) tensor as fp32, in storage order (F32/F16/BF16/quantized).
[[nodiscard]] std::vector<float> weight_vector(const model::WeightRef& w);

/// Row `token` of an embedding tensor {n_embd, n_vocab} into `out` (n_embd floats).
/// Throws Error(Api) for a token outside [0, n_vocab).
void embedding_row(const model::WeightRef& embd, std::int32_t token, std::span<float> out);

/// GGUF tokenizer metadata -> tokenizer spec (D-008, D-015). Throws Error(Unsupported) for
/// a tokenizer model other than "gpt2" or a pre-tokenizer other than "qwen35", and
/// Error(Model) for missing tokens, out-of-range token types or special ids.
[[nodiscard]] tokenizer::VocabSpec vocab_spec(const model::TokenizerMetadata& meta);

// ---------------------------------------------------------------------------------------
// Batch-shaped step
// ---------------------------------------------------------------------------------------

enum class GdnPath : std::uint8_t {
    Auto,       ///< Chunked when rows > 1 and no state slots are requested, else Recurrent
    Chunked,    ///< prefill (transformers torch_chunk_gated_delta_rule semantics, chunk 64)
    Recurrent,  ///< decode / speculative verification
};

enum class LogitsMode : std::uint8_t {
    None,
    Argmax,  ///< greedy fast path (FR-007): only (index, value) of the best logit per row
    Full,    ///< the full vocab row (plus its argmax)
};

/// One sequence of a trunk step. Rows are appended at KV row kv->length() and RoPE
/// position kv->length() + i (text-only 1-D RoPE, D-004).
struct SeqStep {
    std::span<const std::int32_t> tokens;
    /// Optional; if given it must equal kv->length() + i (M-RoPE / gaps are rejected).
    std::span<const std::int32_t> positions;
    kv_cache::SequenceKv* kv = nullptr;  ///< trunk attention layers (Qwen35::kv_layout())
    state::GdnState* gdn = nullptr;      ///< Qwen35::gdn_shape()
    std::span<const std::size_t> logit_rows;  ///< row indices (ascending not required)
    LogitsMode logits = LogitsMode::Argmax;
    bool want_hidden = false;          ///< final output-normed hidden of every row (for MTP)
    std::size_t n_state_slots = 0;     ///< D-012 rollback slots to write (<= gdn->max_slots())
    GdnPath gdn_path = GdnPath::Auto;
};

/// One sequence of an MTP step (D-005): row i = MTP(embed(tokens[i]), hidden[i]) at RoPE
/// position first_position + i, appended at MTP-KV row kv->length() + i.
struct MtpStep {
    std::span<const std::int32_t> tokens;  ///< the embedded tokens x_{p+1}
    std::span<const float> hidden;         ///< rows x n_embd: trunk (or MTP) hidden h_p
    kv_cache::SequenceKv* kv = nullptr;    ///< MTP layer (Qwen35::mtp_kv_layout())
    std::int32_t first_position = 1;
    std::span<const std::size_t> logit_rows;
    LogitsMode logits = LogitsMode::Argmax;
    bool want_hidden = false;              ///< post shared_head_norm hidden of every row
};

struct SeqOutput {
    std::vector<cpu::TopKEntry> argmax;  ///< one per logit row (Argmax and Full)
    std::vector<float> logits;           ///< logit_rows.size() x n_vocab (Full only)
    std::vector<float> hidden;           ///< rows x n_embd (want_hidden only)
};

/// Compulsory-traffic cost of one call (cpu/traffic.h model, review §3.3 / D-011):
/// weights are counted once per call because every matmul runs once over all rows.
struct StepCost {
    std::uint64_t weight_bytes = 0;      ///< stored bytes of every matrix read (each once per call) + LM head
    std::uint32_t weight_passes = 0;     ///< full weight passes (1 per forward/mtp_forward call)
    std::uint64_t embedding_bytes = 0;   ///< gathered embedding rows (one stored row per input row)
    std::uint64_t activation_bytes = 0;  ///< activations read + written by every op
    std::uint64_t state_bytes = 0;       ///< GDN recurrent/conv state read + written, + written slots
    std::uint64_t kv_bytes = 0;          ///< KV rows read by attention + rows written
    [[nodiscard]] std::uint64_t total() const noexcept {
        return weight_bytes + embedding_bytes + activation_bytes + state_bytes + kv_bytes;
    }
    StepCost& operator+=(const StepCost& o) noexcept;
};

struct StepResult {
    std::vector<SeqOutput> seqs;  ///< one per input step, same order
    StepCost cost;
    /// The GDN path each trunk step actually ran (Auto resolved); empty for mtp_forward.
    std::vector<GdnPath> gdn_paths;
    /// With ForwardOptions::capture_layer_inputs: [layer][rows x n_embd] over the
    /// concatenated rows of all sequences (layer 0 = token embeddings).
    std::vector<std::vector<float>> layer_inputs;
};

struct ForwardOptions {
    bool capture_layer_inputs = false;
};

class Qwen35 {
public:
    /// `model` and `pool` must outlive this object. pool may be null (single-threaded).
    /// Throws Error(Unsupported) for weight types without dequantization.
    Qwen35(const model::NormalizedModel& model, cpu::ThreadPool* pool);
    ~Qwen35();
    Qwen35(const Qwen35&) = delete;
    Qwen35& operator=(const Qwen35&) = delete;

    [[nodiscard]] const model::Qwen35HParams& hparams() const noexcept;
    [[nodiscard]] const model::NormalizedModel& model() const noexcept { return *model_; }
    [[nodiscard]] bool has_mtp() const noexcept;
    [[nodiscard]] std::size_t n_vocab() const noexcept;
    [[nodiscard]] std::size_t n_embd() const noexcept;

    [[nodiscard]] kv_cache::KvLayout kv_layout(std::size_t block_tokens = 16) const;
    [[nodiscard]] kv_cache::KvLayout mtp_kv_layout(std::size_t block_tokens = 16) const;
    [[nodiscard]] state::GdnShape gdn_shape() const;

    /// Trunk step over all sequences. On success every sequence's KV length grows by its
    /// row count and its GDN state has advanced (slots written as requested).
    void forward(std::span<const SeqStep> steps, StepResult& out, const ForwardOptions& opts = {}) const;

    /// MTP block step (D-005). Error(Unsupported) if the model has no MTP block.
    void mtp_forward(std::span<const MtpStep> steps, StepResult& out) const;

    // ---- cost model inputs (stored bytes) ------------------------------------------------
    [[nodiscard]] std::uint64_t trunk_weight_bytes() const noexcept;  ///< all trunk layers + output_norm, no head/embedding
    [[nodiscard]] std::uint64_t lm_head_bytes() const noexcept;
    [[nodiscard]] std::uint64_t mtp_block_bytes() const noexcept;     ///< 0 without MTP
    [[nodiscard]] std::uint64_t mtp_head_bytes() const noexcept;      ///< LM head the MTP uses (0 without MTP)
    [[nodiscard]] std::uint64_t embedding_row_bytes() const noexcept;

    struct Impl;

private:
    const model::NormalizedModel* model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::models
