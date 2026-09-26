#pragma once
// MTP speculative decoding for the qwen35 CPU reference (D-005, D-012, FR-009).
//
// One decode tick over S sequences (batch-shaped, review H-1):
//   1. draft   — per draft depth one batched mtp_forward over every drafting sequence.
//                Depth 1 also flushes each sequence's MTP catch-up queue plus the pair
//                (x, h_{L-1}) for the pending token x, all teacher-forced (D-005). Depth
//                i >= 2 feeds the MTP's own post-shared_head_norm hidden of depth i-1
//                (D-005 amendment). Drafts are the MTP argmax (greedy drafting).
//   2. verify  — ONE trunk forward over every sequence of the tick: decode sequences feed
//                [x, d_1..d_k] with k+1 GDN state slots (recurrent path), prefill sequences
//                feed their chunk (chunked path). One weight pass serves every stream.
//   3. commit  — greedy acceptance: n_acc = longest prefix with d_i == argmax(row i-1);
//                the emitted tokens are argmax(rows 0..n_keep-1), n_keep = n_acc + 1
//                (capped by max_emit and at the first stop token). KV is truncated to
//                L + n_keep, the GDN state selects slot (k+1) - n_keep (D-012, no replay
//                forward), MTP KV is truncated to L (dropping the draft-chain rows, which
//                used MTP hiddens) and the accepted rows are queued for teacher-forced
//                catch-up. pending_h is trunk hidden row n_keep-1 (D-005 amendment).
//
// Every emitted token is a trunk argmax, and the recurrent GDN path and the row-wise
// matmuls are bit-identical however rows are split, so greedy output with MTP on equals
// greedy output with MTP off token for token (tested) — whatever the drafts are.
//
// Cancellation / failure (RR-006):
//   - after draft():  abort_draft() restores the MTP KV exactly (the queue is only
//                     consumed at commit), leaving the pre-tick state bit for bit;
//   - after verify(): the fed rows are genuine (x was already emitted, accepted drafts equal
//                     trunk argmaxes), so commit(n_keep = r) for any 1 <= r <= the accepted
//                     count is a consistent state: bit-identical to plain-decoding those r
//                     rows (tested for every r). Cancelling right after verify = commit(1).
//   - a verify that throws (validation, KV exhaustion, or an Error(Kernel)/Error(Memory)
//     from the LM head such as a NaN logit or an allocation failure) aborts the drafts and
//     leaves every sequence at its pre-tick state: qwen35 forward commits trunk KV/GDN only
//     after the head section succeeds (H1/M1, ADR-001 WS-BI-2 step 1), so an exception from
//     the head section never leaves trunk state partially advanced.
//
// Not implemented (see the WS-G report): stochastic (non-greedy) speculative acceptance
// (non-greedy sequences decode with k = 0), llama.cpp's p_min draft confidence gating and
// top-k=10 draft sampler, fp16 KV.
//
// Thread-safety: a Speculator is used by one thread (the engine worker).

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "halo/models/qwen35.h"
#include "halo/state/sequence.h"

namespace halo::speculative {

// ---------------------------------------------------------------------------------------
// Cost model and profit gate (FR-009, D-011, D-014)
// ---------------------------------------------------------------------------------------

/// Predicted weight bytes of one batch-1 step (D-011's dominant terms; KV, state and
/// activations are ignored, which favours MTP slightly at long context).
struct CostModel {
    std::uint64_t trunk = 0;      ///< trunk weights (all layers + output norm)
    std::uint64_t head = 0;       ///< trunk LM head
    std::uint64_t mtp_block = 0;  ///< MTP block + eh_proj + norms
    std::uint64_t mtp_head = 0;   ///< LM head the MTP drafts with

    [[nodiscard]] static CostModel from(const models::Qwen35& m) noexcept {
        return {m.trunk_weight_bytes(), m.lm_head_bytes(), m.mtp_block_bytes(), m.mtp_head_bytes()};
    }
    /// One plain decode step: one trunk pass + head.
    [[nodiscard]] std::uint64_t plain_step() const noexcept { return trunk + head; }
    /// A step with k drafts: the verify pass (trunk + head, once for all k+1 rows) plus k
    /// MTP calls, each one block pass + one head pass. D-011 counts (n+1)·W_mtp because
    /// llama.cpp runs the catch-up as a separate call; here the catch-up rides in the
    /// depth-1 draft call (one weight pass for catch-up rows + draft row).
    [[nodiscard]] std::uint64_t spec_step(std::size_t k) const noexcept {
        return trunk + head + k * (mtp_block + mtp_head);
    }
};

enum class GateMode : std::uint8_t {
    Off,     ///< never draft
    Auto,    ///< draft while the measured speedup stays >= min_speedup, re-probe when off
    Always,  ///< always draft (tests, benchmarks)
};

struct GateConfig {
    GateMode mode = GateMode::Auto;
    double min_speedup = 1.05;           ///< D-014: auto-disable below 1.05x
    std::size_t window = 16;             ///< speculative sequence-steps per evaluation
    std::size_t probe_interval = 256;    ///< plain sequence-steps before a re-probe
};

/// Decides whether to draft from measured acceptance x predicted cost:
///   speedup = (tokens emitted / predicted spec bytes) * plain bytes per token
/// evaluated over each window of speculative sequence-steps. Below min_speedup it turns
/// off; after probe_interval plain sequence-steps it turns on again for one window.
class ProfitGate {
public:
    ProfitGate(GateConfig cfg, std::uint64_t plain_step_bytes) : cfg_(cfg), plain_bytes_(plain_step_bytes) {}

    [[nodiscard]] bool allow() const noexcept {
        return cfg_.mode == GateMode::Always || (cfg_.mode == GateMode::Auto && enabled_);
    }
    /// One sequence ran a speculative step: `tokens` emitted (accepted drafts + 1, capped by
    /// max_emit / a stop token -- L6: recording the uncapped count biased GateMode::Auto
    /// toward keeping speculation on for workloads that mostly hit their token cap) for
    /// `bytes` predicted.
    void record_spec(std::size_t tokens, std::uint64_t bytes) noexcept;
    /// One sequence ran a plain decode step (while the gate was off, or it declined).
    void record_plain() noexcept;

    [[nodiscard]] bool enabled() const noexcept { return allow(); }
    [[nodiscard]] double last_speedup() const noexcept { return last_speedup_; }
    [[nodiscard]] std::uint64_t disables() const noexcept { return disables_; }
    [[nodiscard]] std::uint64_t probes() const noexcept { return probes_; }
    [[nodiscard]] const GateConfig& config() const noexcept { return cfg_; }

private:
    GateConfig cfg_;
    std::uint64_t plain_bytes_;
    bool enabled_ = true;
    std::size_t win_steps_ = 0;
    std::uint64_t win_tokens_ = 0;
    std::uint64_t win_bytes_ = 0;
    std::size_t plain_since_off_ = 0;
    double last_speedup_ = 0.0;
    std::uint64_t disables_ = 0;
    std::uint64_t probes_ = 0;
};

// ---------------------------------------------------------------------------------------
// Tick
// ---------------------------------------------------------------------------------------

struct SpecConfig {
    std::size_t max_draft = 2;        ///< k cap (also capped by gdn.max_slots() - 1)
    GateConfig gate;
    std::size_t mtp_flush_rows = 64;  ///< flush a sequence's MTP queue at this many rows
};

/// One sequence of a tick: either a decode step (feed the pending token `token`) or a
/// prefill chunk (`prefill` non-empty).
struct StepRequest {
    state::SequenceState* seq = nullptr;
    std::int32_t token = -1;                      ///< decode: the pending token x (already emitted)
    std::span<const std::int32_t> prefill;        ///< prefill chunk (then `token` is ignored)
    bool want_output = true;                      ///< prefill: produce the last row's argmax/logits
    bool greedy = true;                           ///< false: k = 0 and full logits of the last row
    std::size_t max_draft = 0;                    ///< requested drafts (capped: config, gate, slots, MTP)
    std::size_t max_emit = std::numeric_limits<std::size_t>::max();  ///< >= 1
    std::span<const std::int32_t> stop_tokens;    ///< acceptance ends after the first of these
    /// Test / external-drafter seam: use these drafts instead of the MTP's (the MTP still
    /// runs its depth-1 catch-up so the MTP state evolves exactly as with real drafting).
    std::span<const std::int32_t> forced_drafts;
};

struct StepOutput {
    std::vector<std::int32_t> drafts;   ///< the k drafts that were verified
    std::vector<std::int32_t> targets;  ///< greedy: trunk argmax of every fed row (prefill: last row)
    std::vector<std::int32_t> tokens;   ///< greedy: emitted tokens (decode: targets[0..n_keep))
    std::vector<float> logits;          ///< non-greedy: full logits of the last row
    std::size_t accepted = 0;           ///< drafts accepted (before max_emit / stop caps)
    std::size_t n_keep = 0;             ///< rows committed (decode) — set by verify, used by commit
};

struct Metrics {
    std::uint64_t ticks = 0;
    std::uint64_t decode_steps = 0;     ///< sequence-steps
    std::uint64_t spec_steps = 0;       ///< sequence-steps with k > 0
    std::uint64_t drafted = 0;
    std::uint64_t accepted = 0;
    std::vector<std::uint64_t> accepted_at;  ///< [i]: steps where draft i+1 was accepted
    std::uint64_t emitted = 0;
    std::uint64_t mtp_flush_rows = 0;
    std::uint64_t mtp_dropped = 0;      ///< sequences whose MTP was dropped after a failed flush (N-1)
    std::uint64_t predicted_bytes = 0;  ///< sum of CostModel predictions over sequence-steps
    models::StepCost last_cost;         ///< measured traffic of the last tick (all forwards)
    std::uint32_t last_weight_passes = 0;  ///< trunk + MTP weight passes of the last tick
    std::uint32_t last_trunk_passes = 0;   ///< trunk weight passes of the last tick (1 per verify)
    std::uint64_t trunk_passes = 0;        ///< cumulative trunk weight passes
    std::uint64_t weight_passes = 0;       ///< cumulative trunk + MTP weight passes
};

/// Per-tick working state (one entry per StepRequest, same order).
struct Tick {
    std::vector<StepOutput> out;
    // internals, valid between draft() / verify() / commit()
    struct Seq {
        std::size_t k = 0;             ///< drafts in this tick
        bool mtp_flushed = false;      ///< depth-1 MTP ran (queue + (x, h) now in MTP KV)
        std::size_t mtp_len0 = 0;      ///< MTP KV length before the tick
        std::size_t len0 = 0;          ///< trunk length before verify
        std::vector<float> hidden;     ///< verify rows' trunk hidden
        std::vector<float> mtp_hidden; ///< last MTP hidden (draft chain)
        std::vector<std::int32_t> fed; ///< rows fed to the trunk
    };
    std::vector<Seq> seqs;
    enum class Phase : std::uint8_t { Empty, Drafted, Verified, Committed } phase = Phase::Empty;
};

class Speculator {
public:
    /// The model must outlive the speculator.
    Speculator(const models::Qwen35& model, SpecConfig cfg);

    /// draft + verify + commit, gate and metrics. On an exception every sequence is at its
    /// pre-tick state (see the header comment for the Error(Kernel) caveat).
    void step(std::span<const StepRequest> reqs, Tick& tick);

    // ---- phases (step() = draft, verify, commit; exposed for cancellation) ---------------
    void draft(std::span<const StepRequest> reqs, Tick& tick);
    void abort_draft(std::span<const StepRequest> reqs, Tick& tick) noexcept;
    void verify(std::span<const StepRequest> reqs, Tick& tick);
    /// n_keep: optional per-request override, 1 <= n_keep[i] <= tick.out[i].n_keep (a
    /// rollback to fewer rows; Error(Api) otherwise, nothing changed). Ignored for prefill.
    void commit(std::span<const StepRequest> reqs, Tick& tick, std::span<const std::size_t> n_keep = {});

    /// Run every queued MTP catch-up pair of these sequences (one batched mtp_forward).
    void flush_mtp(std::span<state::SequenceState* const> seqs);

    [[nodiscard]] const Metrics& metrics() const noexcept { return metrics_; }
    [[nodiscard]] const ProfitGate& gate() const noexcept { return gate_; }
    [[nodiscard]] const CostModel& cost_model() const noexcept { return cost_; }
    [[nodiscard]] const SpecConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] const models::Qwen35& model() const noexcept { return *model_; }

private:
    [[nodiscard]] std::size_t effective_k(const StepRequest& r) const;

    const models::Qwen35* model_;
    SpecConfig cfg_;
    CostModel cost_;
    ProfitGate gate_;
    Metrics metrics_;
};

}  // namespace halo::speculative
