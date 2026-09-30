#pragma once
// Qwen3.8 ("qwen35") forward pass (DECISIONS.md D-004, D-005, D-012, D-016, D-017).
//
// One forward for every backend (ADR-001 §4): the forward is composed from the ops of a
// halo::backend::Backend. The GGUF tensor bytes are imported into the backend read-only
// (on the CPU backend zero-copy; F32 matrices are then viewed in place and other types
// dequantize rows on demand, exactly as models::weight_matrix does); norms and other small
// vectors are dequantized once and uploaded. On the CPU backend every op is the halo::cpu
// function the pre-backend forward called, on the same values: results are bit-identical
// (tests/unit/models/test_backend_gate.cpp). The pool constructors build an internal CPU
// backend, so existing callers are unchanged.
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
// GDN/conv state is the ADR-001 §5.3 ring and the KV pools are State-arena buffers
// (ADR-001 §5.2), both attached to the backend once (WS-BI-2: device-resident on GPU
// backends; zero-copy host memory on the CPU backend). The GDN commit is the `live` integer
// advance after the LM head succeeds (mark_slots_written); a failure leaves every sequence
// bit-for-bit at its pre-step state (KV rows are written past length(), committed after the
// status check).
//
// Not implemented yet (ADR-001 §9): the KernelPlan (WS-BI-4).
//
// Thread-safety: a Qwen35 is immutable after construction. On the CPU backend, forward()
// may be called from several threads with disjoint sequences (it shares the ThreadPool,
// which serializes). On a GPU backend the model and its backend are driven by one thread.

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

namespace halo::backend {
class Backend;
class Buffer;
}

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
    /// Tree verification (opt-in, HALO_MTP_TREE): parents of the rows of `tokens` (cpu::tree_*
    /// semantics; empty = an ordinary linear step). The one accepted shape is
    /// cpu::is_chain_plus_root_leaf: tokens = [x, d1 .. dk, d1'] with parents [-1, 0, 1 .. k-1, 0],
    /// i.e. the draft chain plus ONE alternate depth-1 token d1' (a sibling of d1). Requirements
    /// (Error(Api) otherwise): k >= 1, gdn_path Recurrent (or Auto), n_state_slots == k + 1,
    /// gdn->fits_tree_leaf(k + 1) (ring >= k + 3 states), `positions` empty, fp32 KV.
    /// Rows are appended in the given order (KV rows L .. L+k+1; the leaf's KV row is L+k+1)
    /// but row i sits at RoPE position L + depth(i) and attends its root path only.
    /// The GDN state is begun/committed as a (k+1)-row step: after the forward the sequence is in
    /// the "pending verify" state of the k+1 chain rows, and the leaf's state waits in the ring
    /// (GdnState::commit_tree_leaf). KV length grows by ALL k+2 rows; the caller must then either
    /// truncate to L + m (chain rows kept, commit_rows_kept(k+1, m)) or, to keep [x, d1'], call
    /// Qwen35::kv_move_row(kv, L+k+1, L+1), truncate to L+2 and GdnState::commit_tree_leaf(k+1).
    std::span<const std::int32_t> tree_parents{};
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
    /// Device-resident hidden for a ONE-row step: a Buffer of >= n_embd fp32 that an earlier
    /// mtp_forward on this same backend returned in SeqOutput::hidden_device. When set, `hidden`
    /// must be empty and nothing is uploaded (the row is copied device-side, in stream order).
    /// Additive: null (the default) is the host path, unchanged; the CPU backend accepts it too.
    std::shared_ptr<const backend::Buffer> hidden_device{};
    /// Keep the LAST row of the post-head-norm hidden in a fresh device Buffer
    /// (SeqOutput::hidden_device); independent of want_hidden (the host download).
    bool want_hidden_device = false;
};

struct SeqOutput {
    std::vector<cpu::TopKEntry> argmax;  ///< one per logit row (Argmax and Full)
    std::vector<float> logits;           ///< logit_rows.size() x n_vocab (Full only)
    std::vector<float> hidden;           ///< rows x n_embd (want_hidden only)
    /// Last hidden row as an n_embd-float device Buffer (MtpStep::want_hidden_device only).
    std::shared_ptr<backend::Buffer> hidden_device{};
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

/// Construction-time kernel choices (the CPU tunables the autotuner can set, TRD §64).
struct Qwen35Options {
    /// Chunk length of the chunked GDN prefill (transformers uses 64). Any value >= 1 is the
    /// same function within fp32 tolerance; results are bitwise reproducible per value.
    std::size_t gdn_chunk = 64;
    /// M2: the tokenizer's real vocabulary size, when smaller than the (possibly padded)
    /// GGUF LM-head row count. 0 = no clamp (the head's full row count is the vocabulary).
    /// Excludes padded rows from the fused greedy argmax; never affects full logits output.
    std::size_t valid_vocab = 0;
};

class Qwen35 {
public:
    /// `model` and `pool` must outlive this object. pool may be null (single-threaded).
    /// Throws Error(Unsupported) for weight types without dequantization.
    Qwen35(const model::NormalizedModel& model, cpu::ThreadPool* pool);
    /// As above with explicit kernel options. Error(Config) for gdn_chunk outside [1, 4096];
    /// Error(Unsupported) for a gdn_chunk above the backend's limit (the CPU backend's chunked
    /// GDN accepts at most 1024; before ADR-001 such a chunk failed at the first chunked
    /// forward with Error(Kernel) instead).
    Qwen35(const model::NormalizedModel& model, cpu::ThreadPool* pool, const Qwen35Options& options);
    /// Over any backend (ADR-001 §6.1); `backend` must outlive this object. Error(Unsupported)
    /// when a model dimension or gdn_chunk exceeds the backend's Limits.
    Qwen35(const model::NormalizedModel& model, backend::Backend& backend, const Qwen35Options& options = {});
    ~Qwen35();
    Qwen35(const Qwen35&) = delete;
    Qwen35& operator=(const Qwen35&) = delete;

    [[nodiscard]] const model::Qwen35HParams& hparams() const noexcept;
    [[nodiscard]] const model::NormalizedModel& model() const noexcept { return *model_; }
    [[nodiscard]] bool has_mtp() const noexcept;
    [[nodiscard]] std::size_t n_vocab() const noexcept;
    [[nodiscard]] std::size_t n_embd() const noexcept;
    [[nodiscard]] std::size_t gdn_chunk() const noexcept;
    /// The backend the forward runs on (the internal CPU backend for the pool constructors).
    [[nodiscard]] backend::Backend& backend() const noexcept;

    [[nodiscard]] kv_cache::KvLayout kv_layout(std::size_t block_tokens = 16) const;
    [[nodiscard]] kv_cache::KvLayout mtp_kv_layout(std::size_t block_tokens = 16) const;
    [[nodiscard]] state::GdnShape gdn_shape() const;

    /// Trunk step over all sequences. On success every sequence's KV length grows by its
    /// row count and its GDN state has advanced (slots written as requested).
    void forward(std::span<const SeqStep> steps, StepResult& out, const ForwardOptions& opts = {}) const;

    /// MTP block step (D-005). Error(Unsupported) if the model has no MTP block.
    void mtp_forward(std::span<const MtpStep> steps, StepResult& out) const;

    /// Chained MTP drafting (HALO_MTP_CHAIN): the depth-1 step of every sequence (`steps`, as for
    /// mtp_forward: teacher-forced rows, host `hidden` or one-row `hidden_device`) followed by
    /// k[i] - 1 further one-row draft depths, recorded into ONE command stream with ONE submit / wait.
    /// The winning token of depth d feeds depth d + 1's embedding lookup straight from device memory
    /// (LmHeadArgs::ids -> GET_ROWS) and its hidden row stays on the device, so nothing crosses to the
    /// host until the end. k[i] == 0: the sequence only runs its depth-1 rows (teacher-forced flush,
    /// no logits). out.seqs[i].argmax holds the k[i] drafts in depth order (empty for k[i] == 0);
    /// out.cost sums every depth (weight_passes = depths run). steps[i].logits / logit_rows /
    /// want_hidden* are ignored. Requires Limits::lm_head_ids (Error(Unsupported) otherwise).
    /// Greedy drafts are identical to the equivalent mtp_forward sequence (same ops, same order).
    /// KV: on success each MTP KV grew by tokens.size() + k[i] - 1 rows (as the unchained sequence
    /// leaves it); on Error every touched MTP KV is truncated back to its length on entry. A poisoned
    /// (NaN) draft is reported as Error(Api) after the sync, like the unchained path's next-depth
    /// token check.
    void mtp_draft_chain(std::span<const MtpStep> steps, std::span<const std::size_t> k, StepResult& out) const;

    /// Copies KV row `from` to row `to` (both < kv.length()) of every trunk attention layer, K and
    /// V, on the model's backend (synchronous). The tree-verify commit of the alternate branch
    /// (SeqStep::tree_parents). fp32 KV pools only (Error(Unsupported) otherwise); the destination
    /// block must be exclusively owned (rows written by a forward always are). `kv` must be a
    /// trunk sequence KV.
    void kv_move_row(kv_cache::SequenceKv& kv, std::size_t from, std::size_t to) const;

    // ---- cost model inputs (stored bytes) ------------------------------------------------
    [[nodiscard]] std::uint64_t trunk_weight_bytes() const noexcept;  ///< all trunk layers + output_norm, no head/embedding
    [[nodiscard]] std::uint64_t lm_head_bytes() const noexcept;
    [[nodiscard]] std::uint64_t mtp_block_bytes() const noexcept;     ///< 0 without MTP
    [[nodiscard]] std::uint64_t mtp_head_bytes() const noexcept;      ///< LM head the MTP uses (0 without MTP)
    [[nodiscard]] std::uint64_t embedding_row_bytes() const noexcept;

    struct Impl;

private:
    Qwen35(const model::NormalizedModel& model, std::unique_ptr<backend::Backend> owned, const Qwen35Options& options,
           backend::Backend* borrowed = nullptr);

    const model::NormalizedModel* model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::models
