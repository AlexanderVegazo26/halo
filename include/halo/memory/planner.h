#pragma once
// Memory planner (PRD FR-004, TRD §14, §13.3, §48; D-002, D-003).
//
// Given a model shape, a runtime request and the discovered memory tiers, estimate every
// allocation class per tier *before* anything is allocated, and either accept the plan or
// refuse it with a typed MEMORY_ERROR naming the overflowing tier and the shortfall.
//
// Policy (D-002 — tier-discovery-driven, never assumption-driven):
//   * budget(tier) = floor(available(tier) × safety_factor); the rest is the reserve.
//     safety_factor is validated to (0, 0.98] — HALO never plans all of a tier.
//   * GPU-resident classes prefer VRAM (carveout) and fall back to GTT; host-only classes
//     go to HOST. With no GPU, everything goes to HOST (CPU reference path).
//   * Placement is greedy in Component declaration order: GDN recurrent/conv state and its
//     MTP rollback copies first (read and written every token, so they get the fastest
//     tier that fits), then KV, then small per-step buffers, then weights, and last the
//     prefix-cache GDN checkpoints (D-013, on by default; GPU pool only, never HOST; the derived
//     per-slot count shrinks to fit unless configured explicitly).
//   * GTT pages come from OS RAM, so GTT placements also count against the HOST budget
//     (constraint "HOST": planned(HOST) + planned(GTT) <= budget(HOST)).
//   * A user tier override pins a class to one tier (no fallback) but never bypasses a
//     budget; `max_memory` caps the planned total across all tiers.
//   * A class is never split across tiers (weights are already split by tensor class).
//
// Estimates for activations/workspace/runtime overhead are formulas, not measurements,
// until a backend reports measured sizes through the *_override fields.
//
// Thread-safety: all functions are pure; no global state.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "halo/hardware/hardware.h"  // MemoryTiers (plain struct; no link dependency)
#include "halo/hardware/tier.h"

namespace halo::memory {

using hardware::MemoryTier;

// ---- checked size arithmetic --------------------------------------------------------------

[[nodiscard]] std::optional<std::uint64_t> checked_add(std::uint64_t a, std::uint64_t b) noexcept;
[[nodiscard]] std::optional<std::uint64_t> checked_mul(std::uint64_t a, std::uint64_t b) noexcept;
/// Throw Error(Config, "size arithmetic overflow computing <what>") on overflow.
[[nodiscard]] std::uint64_t add_or_throw(std::uint64_t a, std::uint64_t b, std::string_view what);
[[nodiscard]] std::uint64_t mul_or_throw(std::uint64_t a, std::uint64_t b, std::string_view what);

/// Storage cost of a tensor dtype as ggml-style blocks: `block_bytes` per `block_elems`.
struct DtypeSize {
    std::string_view name;
    std::uint32_t block_elems = 1;
    std::uint32_t block_bytes = 4;

    static constexpr DtypeSize f32() noexcept { return {"f32", 1, 4}; }
    static constexpr DtypeSize f16() noexcept { return {"f16", 1, 2}; }
    static constexpr DtypeSize bf16() noexcept { return {"bf16", 1, 2}; }
    static constexpr DtypeSize q8_0() noexcept { return {"q8_0", 32, 34}; }  ///< 1.0625 B/elem
    static constexpr DtypeSize q4_0() noexcept { return {"q4_0", 32, 18}; }  ///< 0.5625 B/elem

    /// Bytes for one row of `elems` elements; throws Error(Config) if `elems` is not a
    /// multiple of block_elems or on overflow.
    [[nodiscard]] std::uint64_t row_bytes(std::uint64_t elems) const;
};

/// Per-tensor-class weight bytes (as stored, i.e. quantized).
struct WeightBytes {
    std::uint64_t embeddings = 0;  ///< token_embd
    std::uint64_t lm_head = 0;     ///< output
    std::uint64_t attention = 0;   ///< full-attention projections
    std::uint64_t gdn = 0;         ///< Gated DeltaNet projections, conv, A/dt, ssm_norm
    std::uint64_t ffn = 0;
    std::uint64_t norms = 0;
    std::uint64_t mtp = 0;         ///< blk.64 (NextN) block, 0 if absent
    [[nodiscard]] std::uint64_t total() const;  ///< throws on overflow
};

/// Model shape as the planner needs it (independent of the model module).
struct ModelShape {
    WeightBytes weights;
    // Full-attention layers (KV cache).
    std::uint32_t n_attn_layers = 0;
    std::uint32_t n_head = 0;
    std::uint32_t n_head_kv = 0;
    std::uint32_t key_dim = 0;
    std::uint32_t value_dim = 0;
    // Gated DeltaNet layers (recurrent + conv state).
    std::uint32_t n_gdn_layers = 0;
    std::uint32_t n_v_heads = 0;
    std::uint32_t d_k = 0;
    std::uint32_t d_v = 0;
    std::uint32_t conv_kernel = 0;
    std::uint32_t conv_channels = 0;
    std::uint32_t gdn_chunk = 64;  ///< chunked delta-rule tile (workspace estimate only)
    // Trunk.
    std::uint32_t hidden = 0;
    std::uint32_t ffn_intermediate = 0;
    std::uint32_t vocab = 0;
    std::uint64_t max_trained_context = 0;  ///< upper bound for max_context_for_budget
    /// Attention layers in the MTP block (each adds one layer of KV when MTP is enabled).
    std::uint32_t mtp_attn_layers = 1;
};

/// Qwen3.8-27B shape constants (DECISIONS D-003/D-004/D-005, GGUF header of the unsloth
/// UD-Q4_K_XL file). Weight bytes differ per quantization and are supplied by the caller.
[[nodiscard]] ModelShape qwen38_27b_shape(const WeightBytes& weights);

enum class KvLayout : std::uint8_t {
    PerSequence,  ///< every sequence can hold max_context tokens (cells = ctx × seqs)
    Unified,      ///< one pool of max_context cells shared by all sequences (llama.cpp --kv-unified)
};

enum class Component : std::uint8_t {
    GdnRecurrentState,
    GdnConvState,
    GdnRollbackState,  ///< MTP verification rollback copies (recurrent + conv)
    KvCache,
    MtpKvCache,
    Logits,
    Activations,
    Workspace,
    GpuRuntimeOverhead,
    Norms,
    AttentionWeights,
    GdnWeights,
    FfnWeights,
    LmHead,
    Embeddings,
    MtpWeights,
    PrefixCheckpoints,  ///< prefix-cache GDN state checkpoints; GPU tiers only, lowest priority
    HostRuntimeOverhead,
};
[[nodiscard]] std::string_view to_string(Component c) noexcept;

inline constexpr std::uint64_t kDefaultGpuRuntimeOverhead = 256ULL << 20;   ///< unmeasured placeholder
inline constexpr std::uint64_t kDefaultHostRuntimeOverhead = 1ULL << 30;    ///< unmeasured placeholder
inline constexpr double kMaxSafetyFactor = 0.98;

/// Prefix-cache GDN state checkpoints — committed by DECISIONS D-013 (supersedes the TRD
/// §12.2 "R&D spike" status); enabled by default with a planner-owned budget.
/// Each checkpoint is one full recurrent + conv state copy. D-013 puts them in the GPU pool,
/// never the OS RAM pool: VRAM then GTT in the gtt-primary layout, VRAM only in the
/// carveout-primary layout (there GTT pages are the OS pool). With no eligible GPU tier a
/// *derived* count degrades to 0 (D-013's always-recompute fallback, noted in the plan); an
/// explicit `max_per_slot` > 0 is then a Config error.
struct PrefixCheckpoints {
    bool enabled = true;
    std::uint32_t spacing_tokens = 8192;
    /// Explicit per-slot count. When unset the count is max_context / spacing + 1
    /// (17 at 131072 / 8192).
    std::optional<std::uint32_t> max_per_slot;
    /// When the plan does not fit, reduce the *derived* per-slot count to the largest that
    /// fits (and say so in the plan notes). An explicit `max_per_slot` is never shrunk:
    /// explicit configuration is honoured or refused.
    bool shrink_to_fit = true;
};

struct PlanRequest {
    std::uint64_t max_context = 32768;
    std::uint32_t max_sequences = 1;
    KvLayout kv_layout = KvLayout::PerSequence;
    DtypeSize kv_dtype = DtypeSize::f16();
    DtypeSize state_dtype = DtypeSize::f32();  ///< D-003: fp32 reference default
    bool mtp_enabled = false;
    std::uint32_t mtp_draft_depth = 0;  ///< must be >= 1 when MTP is enabled
    /// Extra full GDN state copies (recurrent + conv) per sequence kept for rolling back
    /// rejected draft tokens. Default: draft_depth + 1 when MTP is enabled, else 1 — the
    /// ADR-001 §5.3 state ring needs P = K + 1 >= 2 physical states even without MTP
    /// (failure atomicity of every decode step).
    std::optional<std::uint32_t> gdn_rollback_copies;
    PrefixCheckpoints prefix_checkpoints;
    std::uint32_t batch = 512;
    std::uint32_t ubatch = 256;
    bool flash_attention = true;  ///< false adds an ubatch × n_head × ctx fp32 score buffer
    std::optional<std::uint64_t> logits_rows;          ///< default seqs × (1 + draft depth)
    std::optional<std::uint64_t> activations_override; ///< measured value replaces the estimate
    std::optional<std::uint64_t> workspace_override;   ///< measured value replaces the estimate
    std::uint64_t gpu_runtime_overhead = kDefaultGpuRuntimeOverhead;
    std::uint64_t host_runtime_overhead = kDefaultHostRuntimeOverhead;
    double safety_factor = 0.9;
    std::optional<std::uint64_t> max_memory;  ///< user cap on the planned total (all tiers)
    std::map<Component, MemoryTier> tier_overrides;
};

/// Sizes of every class for a request, independent of placement. Throws Error(Config) on
/// invalid input or arithmetic overflow.
struct SizeBreakdown {
    std::uint64_t kv_bytes_per_token = 0;  ///< trunk attention layers
    std::uint64_t kv_cells = 0;            ///< tokens of KV storage (layout-dependent)
    std::uint64_t kv = 0;
    std::uint64_t mtp_kv = 0;
    std::uint64_t gdn_recurrent = 0;          ///< live state, all sequences
    std::uint64_t gdn_conv = 0;               ///< live state, all sequences
    std::uint64_t gdn_state_per_copy = 0;     ///< recurrent + conv, one sequence
    std::uint32_t gdn_rollback_copies = 0;    ///< per sequence
    std::uint64_t gdn_rollback = 0;
    std::uint32_t prefix_checkpoints_per_slot = 0;
    std::uint64_t prefix_checkpoints = 0;
    std::uint64_t logits = 0;
    std::uint64_t activations = 0;
    std::uint64_t workspace = 0;
    std::uint64_t gpu_runtime_overhead = 0;
    std::uint64_t host_runtime_overhead = 0;
    WeightBytes weights;
    [[nodiscard]] std::uint64_t bytes(Component c) const noexcept;
    [[nodiscard]] std::uint64_t total() const;
};
[[nodiscard]] SizeBreakdown compute_sizes(const ModelShape& shape, const PlanRequest& request);

struct Allocation {
    Component component{};
    std::uint64_t bytes = 0;
    MemoryTier tier = MemoryTier::Host;
    bool overridden = false;     ///< placed by a user tier override
    bool overcommitted = false;  ///< fit no candidate tier; placed on its last candidate anyway
};

struct TierBudget {
    MemoryTier tier = MemoryTier::Host;
    std::uint64_t total = 0;
    std::uint64_t available = 0;  ///< total − used (VRAM/GTT) or MemAvailable (HOST)
    std::uint64_t budget = 0;     ///< floor(available × safety_factor)
    std::uint64_t reserve = 0;    ///< available − budget
    std::uint64_t planned = 0;    ///< bytes placed on this tier
};

/// A violated constraint. `constraint` is "VRAM", "GTT", "HOST" (system RAM incl. GTT
/// pages) or "MAX_MEMORY".
struct Overflow {
    std::string constraint;
    std::uint64_t demand = 0;
    std::uint64_t budget = 0;
    std::uint64_t excess = 0;
};

struct MemoryPlan {
    bool ok = false;
    SizeBreakdown sizes;
    std::vector<Allocation> allocations;  ///< in placement-priority order
    std::vector<TierBudget> tiers;        ///< present tiers only
    std::vector<Overflow> overflows;      ///< sorted by excess, largest first
    std::vector<std::string> notes;
    std::uint64_t planned_total = 0;
    /// Checkpoints per slot before shrink-to-fit (== sizes.prefix_checkpoints_per_slot
    /// unless the planner shrank it).
    std::uint32_t prefix_checkpoints_requested = 0;
    double safety_factor = 0;
    std::string message;  ///< one-line verdict ("fits ..." or the refusal reason)

    [[nodiscard]] std::uint64_t placed_on(MemoryTier t) const noexcept;
    [[nodiscard]] std::optional<MemoryTier> tier_of(Component c) const noexcept;
    /// Readable budget table (components, tiers, reserve, verdict).
    [[nodiscard]] std::string table() const;
    /// Throw Error(Memory, message + table) if the plan was refused.
    void require_ok() const;
};

/// Build the plan. Never throws for "does not fit" (that is `ok == false`); throws
/// Error(Config) for invalid requests/overrides and arithmetic overflow.
[[nodiscard]] MemoryPlan plan_memory(const ModelShape& shape, const PlanRequest& request,
                                     const hardware::MemoryTiers& tiers);

/// plan_memory(...).require_ok(), returning the accepted plan.
[[nodiscard]] MemoryPlan plan_memory_or_throw(const ModelShape& shape, const PlanRequest& request,
                                              const hardware::MemoryTiers& tiers);

/// Largest max_context in [1, upper] whose plan fits (other request fields unchanged);
/// 0 if even 1 token does not fit. `upper` defaults to shape.max_trained_context.
/// Requests whose size arithmetic overflows count as "does not fit". A plan only counts
/// as fitting with the full requested prefix-checkpoint count (no shrink-to-fit).
[[nodiscard]] std::uint64_t max_context_for_budget(const ModelShape& shape, const PlanRequest& request,
                                                   const hardware::MemoryTiers& tiers,
                                                   std::optional<std::uint64_t> upper = std::nullopt);

/// Largest max_sequences in [1, upper] whose plan fits; 0 if one sequence does not fit.
[[nodiscard]] std::uint32_t max_sequences_for_budget(const ModelShape& shape, const PlanRequest& request,
                                                     const hardware::MemoryTiers& tiers, std::uint32_t upper = 256);

void to_json(nlohmann::json& j, const MemoryPlan& plan);

}  // namespace halo::memory
