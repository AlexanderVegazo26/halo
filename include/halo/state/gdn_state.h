#pragma once
// Per-sequence Gated DeltaNet state (D-003, D-012, D-013).
//
// For every GDN layer: the recurrent state [n_v_heads, d_k, d_v] fp32 (d_v fastest, the
// layout halo::cpu::gated_delta_rule_* uses) and the conv state [conv_kernel - 1,
// conv_channels] fp32 (oldest row first, the layout halo::cpu::causal_conv1d_silu uses).
//
// Rollback slots (D-012): the state also owns `max_slots` per-row snapshot slots per
// layer. A forward over T rows that is asked for n slots writes slot s = the state after
// row T-1-s (s < min(T, n)); the forward reports how many it wrote (mark_slots_written).
// Keeping the first m of those T rows is commit_rows_kept(T, m): slot T-m becomes the live
// state (slot 0 is the live state already, so accepting everything copies nothing).
//
// Checkpoints (D-013): snapshot() copies the full live state of every layer; restore()
// copies one back. CheckpointStore (checkpoint.h) keeps snapshots under a byte budget.
//
// Thread-safety: a GdnState is not thread-safe (one sequence, one owner).

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "halo/backends/cpu/views.h"

namespace halo::state {

struct GdnShape {
    std::size_t n_layers = 0;       ///< GDN layers
    std::size_t n_v_heads = 0;
    std::size_t d_k = 0;
    std::size_t d_v = 0;
    std::size_t conv_kernel = 0;    ///< >= 1
    std::size_t conv_channels = 0;

    [[nodiscard]] std::size_t recurrent_floats() const;  ///< per layer
    [[nodiscard]] std::size_t conv_floats() const;       ///< per layer
    [[nodiscard]] std::size_t layer_floats() const { return recurrent_floats() + conv_floats(); }
    [[nodiscard]] std::size_t total_floats() const;      ///< all layers, one copy
    [[nodiscard]] std::size_t total_bytes() const { return total_floats() * sizeof(float); }
};

/// A full copy of the live state of every layer.
struct GdnSnapshot {
    std::vector<float> data;  ///< layer-major: [layer][recurrent | conv]
    [[nodiscard]] std::size_t bytes() const noexcept { return data.size() * sizeof(float); }
};

class GdnState {
public:
    /// Zero state, `max_slots` rollback slots per layer. Throws Error(Config) on a zero
    /// dimension, Error(Memory) if the storage cannot be allocated.
    GdnState(GdnShape shape, std::size_t max_slots);

    [[nodiscard]] const GdnShape& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t max_slots() const noexcept { return max_slots_; }

    /// Live recurrent state of GDN layer `layer` (n_v_heads * d_k * d_v floats).
    [[nodiscard]] std::span<float> recurrent(std::size_t layer);
    /// Live conv state of `layer` as [conv_kernel - 1, conv_channels].
    [[nodiscard]] cpu::Rows conv(std::size_t layer);
    /// The first n (<= max_slots) recurrent / conv slots of `layer`, contiguous, in the
    /// layout the cpu ops take. n == 0 gives an empty span.
    [[nodiscard]] std::span<float> recurrent_slots(std::size_t layer, std::size_t n);
    [[nodiscard]] std::span<float> conv_slots(std::size_t layer, std::size_t n);

    /// Called by the forward after it ran T rows with n slots requested: records that
    /// slots [0, min(T, n)) hold the states after rows T-1 ... T-min(T,n) of that call.
    void mark_slots_written(std::size_t rows, std::size_t n_slots) noexcept;
    /// Rows of the last slot-writing call (0 if none since the last commit).
    [[nodiscard]] std::size_t slot_rows() const noexcept { return slot_rows_; }
    [[nodiscard]] std::size_t slots_valid() const noexcept { return slots_valid_; }

    /// Keep the first `kept` of the `rows` rows of the last slot-writing forward: selects
    /// slot rows - kept (a copy into the live state unless it is slot 0) for every layer.
    /// Requires rows == slot_rows() and 1 <= kept <= rows with rows - kept < slots_valid()
    /// (Error(Api) otherwise; nothing changes). Invalidates the slots.
    void commit_rows_kept(std::size_t rows, std::size_t kept);
    /// Forget the slots (the last call is accepted as a whole).
    void drop_slots() noexcept { slot_rows_ = 0; slots_valid_ = 0; }

    void reset() noexcept;  ///< zero live state, drop slots
    [[nodiscard]] GdnSnapshot snapshot() const;
    /// Error(Api) if the snapshot has the wrong size.
    void restore(const GdnSnapshot& s);
    void copy_from(const GdnState& other);  ///< live state only; shapes must match

    /// Bytes moved by commit_rows_kept(rows, kept) (read + write), for the cost model.
    [[nodiscard]] std::uint64_t commit_bytes(std::size_t rows, std::size_t kept) const noexcept;

private:
    [[nodiscard]] float* live(std::size_t layer) noexcept { return live_.data() + layer * shape_.layer_floats(); }
    void check_layer(std::size_t layer) const;

    GdnShape shape_;
    std::size_t max_slots_;
    std::vector<float> live_;   // [layer][recurrent | conv]
    std::vector<float> slots_;  // [layer][recurrent slots (max_slots) | conv slots (max_slots)]
    std::size_t slot_rows_ = 0;
    std::size_t slots_valid_ = 0;
};

}  // namespace halo::state
