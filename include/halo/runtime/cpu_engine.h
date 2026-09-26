#pragma once
// CPU reference Engine (WS-G M3): options and observability beyond the engine.h contract.
// create_engine(cfg) == create_cpu_engine(cfg, {}) for backend "cpu" / "auto".
//
// Scheduling (ARCHITECTURE "Scheduler", review H-1): one worker thread. Each tick issues
// ONE batched trunk forward over every active sequence — decode rows (with MTP drafts for
// greedy requests) and at most one prefill chunk per prefilling sequence — so one weight
// pass serves every stream. Tokens are handed to each request's own queue; the request's
// generate() thread invokes the callback (engine.h threading guarantee).
//
// Prefix cache (D-013): a finished (or cancelled) sequence keeps its KV blocks as a cache
// entry (refcounted, shared with later sequences by copy-on-write) plus GDN checkpoints
// taken during its prefill (every `checkpoint_spacing` tokens and at prompt end - tail) and
// at retirement (all fed tokens). A new prompt restores the longest checkpoint whose owner
// entry is still cached and whose length is <= min(LCP, prompt - 1), shares the owner's KV
// blocks up to it, and recomputes only the tail. Always-recompute is the fallback. Rows
// restored this way are bit-identical to the owner's; they equal a fresh computation of
// the same tokens within fp32 tolerance (a different chunking), not bitwise.
//
// Memory: the model is planned with memory::plan_memory over the discovered tiers
// (hardware::discover, root "/"); a plan that does not fit is Error(Memory) at creation.
// D-013 puts checkpoints in the GPU pool; with no GPU tier (the CPU reference on the dev
// host) the planner assigns them none, and the CPU backend then budgets the planner's
// requested count on the host instead (logged at creation). The real allocation (KV pools
// incl. the prefix-cache share, checkpoint budget) is checked against max_memory_bytes.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "halo/runtime/engine.h"
#include "halo/speculative/speculative.h"

namespace halo::runtime {

/// One scheduler tick, reported on the worker thread (tests / profiling).
struct TickInfo {
    std::size_t sequences = 0;
    std::size_t prefill_rows = 0;
    std::size_t decode_rows = 0;         ///< fed decode rows (pending token + drafts)
    std::size_t max_draft = 0;           ///< deepest draft chain in the tick
    std::size_t drafted = 0;
    std::size_t accepted = 0;
    /// Additive (WS-G M6): the draft tokens of every decode sequence, in tick order
    /// (drafted == drafts.size()). Lets tests compare what two MTP sources proposed.
    std::vector<std::int32_t> drafts;
    /// Pass counters count forward / mtp_forward *calls* (review N-4). Each call is one
    /// weight pass by construction (every matmul runs once over all rows of the call), so a
    /// count of 1 proves one batched forward, not a measured byte count; the byte figure is
    /// measured_weight_bytes.
    std::uint32_t weight_passes = 0;     ///< trunk + MTP forward calls
    std::uint32_t trunk_passes = 0;      ///< trunk forward calls; must be 1 (one batched forward per tick)
    std::uint64_t predicted_bytes = 0;   ///< CostModel: trunk + head + max_draft * (MTP block + head)
    /// Weight bytes the tick's forwards accounted (models::StepCost): every matrix read,
    /// counted once per call, plus the LM head. Excludes norm/conv vectors, so it sits just
    /// below predicted_bytes.
    std::uint64_t measured_weight_bytes = 0;
    bool mtp_fallback = false;           ///< retried with k = 0 after MTP-KV exhaustion
};

/// Failure injection for tests (review R-5(a)). Each hook runs on the worker thread and
/// may throw to simulate a failure at that point; null hooks cost nothing.
struct FaultInjection {
    /// Before the tick's batched step (a throw here is a tick-wide failure; Error(Memory)
    /// exercises the evict / MTP-fallback retry loop).
    std::function<void(std::span<const speculative::StepRequest>)> before_step;
    /// In the per-sequence output phase, after the step committed (R-1 isolation).
    std::function<void(std::span<const std::int32_t> prompt)> output;
    /// Inside a best-effort GDN prefix checkpoint (D-013).
    std::function<void()> checkpoint;
    /// Inside retirement's prefix-cache part (R-2: must never escape the worker).
    std::function<void()> retire_cache;
};

struct CpuEngineOptions {
    std::size_t prefill_chunk = 256;     ///< rows per prefill chunk (a multiple of 64 keeps GDN chunks aligned)
    std::size_t kv_block_tokens = 16;
    std::optional<std::size_t> kv_blocks;       ///< trunk KV pool size override (default: planned)
    std::optional<std::size_t> mtp_kv_blocks;   ///< MTP KV pool size override
    speculative::GateMode gate_mode = speculative::GateMode::Auto;
    std::size_t gate_window = 16;
    std::size_t gate_probe_interval = 256;
    std::optional<std::uint64_t> checkpoint_budget_bytes;  ///< default: from the memory plan
    std::size_t checkpoint_spacing = 0;         ///< 0 = the planner's spacing (8192)
    std::size_t checkpoint_tail = 1;            ///< checkpoint at prompt end - tail (D-013 "N - k")
    std::size_t checkpoint_min_spacing = 64;    ///< minimum distance between two prompt checkpoints
    std::size_t prefix_cache_entries = 0;       ///< cached finished sequences; 0 = max_sequences
    std::string hardware_root = "/";
    /// Test seam: treat this id as EOS instead of the tokenizer's.
    std::optional<std::int32_t> eos_token;
    /// Test / external-drafter seam, worker thread: forced drafts for a greedy decode step
    /// given (prompt, generated so far; pending = generated.back()). Empty = MTP drafting.
    /// At most mtp_max_draft are used.
    std::function<std::vector<std::int32_t>(std::span<const std::int32_t>, std::span<const std::int32_t>)> draft_hook;
    /// Called on the worker thread after every tick. Must not block or call the engine.
    std::function<void(const TickInfo&)> on_tick;
    FaultInjection faults;  ///< tests only
};

/// Throws Error(Unsupported) for a backend other than cpu/auto, Error(Model/Io) for model
/// files, Error(Memory) if the memory plan does not fit.
[[nodiscard]] std::unique_ptr<Engine> create_cpu_engine(const EngineConfig& cfg, const CpuEngineOptions& opts = {});

}  // namespace halo::runtime
