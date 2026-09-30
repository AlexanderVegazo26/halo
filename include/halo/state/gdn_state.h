#pragma once
// Per-sequence Gated DeltaNet state (D-003, D-012, D-013) as a state ring (ADR-001 §5.3,
// WS-BI-2).
//
// For every GDN layer: the recurrent state [n_v_heads, d_k, d_v] fp32 (d_v fastest, the
// layout halo::cpu::gated_delta_rule_* uses) and the conv state [conv_kernel - 1,
// conv_channels] fp32 (oldest row first, the layout halo::cpu::causal_conv1d_silu uses).
//
// The ring (ADR-001 §5.3): the state owns P = max(2, max_slots + 1) physical states per
// layer in one slab, plus one integer `live` shared by all layers. A forward over T rows
// reads slab[live] (never written), always writes the final state (logical slot 0) to
// slab[(live+1) mod P], and writes rollback slot s to slab[(live+1+s) mod P] for
// s < min(T, n_slots). The forward records base = live at the start (begin_step) and, after
// a clean status, commits with live = (base+1) mod P (mark_slots_written — the whole commit
// for decode and prefill). Keeping the first m of a verify's T rows is
// commit_rows_kept(T, m): live = (base + 1 + (T - m)) mod P, from the recorded base — one
// integer, no copy. A failure leaves live unchanged: the sequence is bit-for-bit at its
// pre-step state. A forward on a sequence with an uncommitted verify is Error(Api)
// (begin_step).
//
// Placement (ADR-001 §5.2): the slab lives in host memory until attach(backend) gives it a
// backend-owned image. On a zero-copy backend (CPU, HIP emulation) the image wraps the host
// slab (host accessors stay valid). On a GPU backend (Vulkan) the image is device-resident:
// host accessors then raise Error(Api) (the content is a stale mirror); pull() refreshes
// the mirror for tests, and snapshot()/restore() go through a stream.
//
// Checkpoints (D-013): snapshot() copies the live state of every layer; restore() copies
// one back into slab[live]. CheckpointStore (checkpoint.h) keeps snapshots under a byte
// budget.
//
// Thread-safety: a GdnState is not thread-safe (one sequence, one owner).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/backends/cpu/ops.h"
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
    /// Zero state (live = slot 0), P = max(2, max_slots + 1) physical states per layer
    /// (ADR-001 §5.3: live + the final-state slot + rollback slots). Throws Error(Config) on
    /// a zero dimension, Error(Memory) if the storage cannot be allocated.
    GdnState(GdnShape shape, std::size_t max_slots);

    [[nodiscard]] const GdnShape& shape() const noexcept { return shape_; }
    /// Rollback slots one forward may request: P - 1 (>= 1 even without MTP: the ring needs
    /// a second physical state for failure atomicity, ADR-001 §5.3 line 371).
    [[nodiscard]] std::size_t max_slots() const noexcept { return p_ - 1; }
    /// Physical states per layer (P).
    [[nodiscard]] std::size_t ring_size() const noexcept { return p_; }
    /// The slot every layer's next forward reads.
    [[nodiscard]] std::uint32_t live() const noexcept { return live_; }

    // ---- backend attachment (ADR-001 §5.2: the slab is a State-arena buffer) -------------
    /// Gives the slab a backend-owned image (idempotent for the same backend; Error(Api) for
    /// another). CPU / HIP emulation: zero-copy over the host slab. GPU backends: a
    /// device-resident buffer (zero-filled; a previously written host state is uploaded).
    /// Error(Memory) on allocation failure.
    void attach(backend::Backend& be);
    [[nodiscard]] bool attached() const noexcept { return be_ != nullptr; }
    /// True when the host slab is the storage (unattached, or a zero-copy attachment), so
    /// the host accessors are current. False on a device-resident backend until pull().
    [[nodiscard]] bool host_coherent() const noexcept { return coherent_; }
    /// The ring slab of `layer` as a backend operand (attached only): the recurrent ring is
    /// P dense recurrent states, the conv ring P dense conv states.
    [[nodiscard]] backend::TensorRef recurrent_ref(std::size_t layer) const;
    [[nodiscard]] backend::TensorRef conv_ref(std::size_t layer) const;
    /// Refreshes the host mirror from the device (device-resident states; a no-op when
    /// host-coherent). Tests use this before the host accessors.
    void pull();

    // ---- host views (host-coherent states only; Error(Api) otherwise) --------------------
    /// Live recurrent state of GDN layer `layer` (n_v_heads * d_k * d_v floats).
    [[nodiscard]] std::span<float> recurrent(std::size_t layer);
    /// Live conv state of `layer` as [conv_kernel - 1, conv_channels].
    [[nodiscard]] cpu::Rows conv(std::size_t layer);
    /// The ring of `layer` as the cpu ops' StateRingView (recurrent: P dense recurrent
    /// states; conv: P dense conv states).
    [[nodiscard]] cpu::StateRingView recurrent_ring(std::size_t layer, std::size_t n_slots);
    [[nodiscard]] cpu::StateRingView conv_ring(std::size_t layer, std::size_t n_slots);
    /// The whole host slab (tests; layout: recurrent rings [layer][P][rec] then conv rings
    /// [layer][P][conv]).
    [[nodiscard]] std::span<const float> slab() const;

    // ---- the step protocol (ADR-001 §5.3 "who advances live") -----------------------------
    /// Records base = live for a forward over `rows` rows asking for `n_slots` rollback
    /// slots. Error(Api) if the previous slot-writing forward was never committed
    /// (commit_rows_kept / drop_slots) — no forward may run between a verify and its commit.
    void begin_step(std::size_t rows, std::size_t n_slots);
    /// The commit after a clean status: live = (base + 1) mod P (keeps all rows). With
    /// n_slots > 0 the call is a verify: the pending slots stay referenced until
    /// commit_rows_kept / drop_slots. Called by the forward once the step's status is clean.
    void mark_slots_written(std::size_t rows, std::size_t n_slots) noexcept;
    /// Rows of the last slot-writing call (0 if none since the last commit).
    [[nodiscard]] std::size_t slot_rows() const noexcept { return pend_rows_; }
    [[nodiscard]] std::size_t slots_valid() const noexcept { return pend_slots_; }

    /// Keep the first `kept` of the `rows` rows of the last slot-writing forward:
    /// live = (base + 1 + (rows - kept)) mod P — one integer, no copy. Requires a pending
    /// verify with rows == slot_rows() and 1 <= kept <= rows with rows - kept <
    /// slots_valid() (Error(Api) otherwise; nothing changes). Ends the pending verify.
    void commit_rows_kept(std::size_t rows, std::size_t kept);
    /// Ends a pending verify keeping all rows (live stays at base + 1).
    void drop_slots() noexcept;
    /// Tree verify (HALO_MTP_TREE): the last forward ran a chain of `rows` rows with n_slots >=
    /// rows, plus one extra leaf row whose parent is chain row 0. The forward ran the leaf as a
    /// one-row step reading the ring slot that holds the state after row 0 (physical
    /// (base + rows) mod P) and wrote its state to (base + rows + 1) mod P. This makes that state
    /// the live one (the sequence continues [x, leaf]): live = (base + rows + 1) mod P, one
    /// integer. Requires a pending verify with rows == slot_rows(), slots_valid() >= rows and
    /// P >= rows + 2 (so that slot is distinct from live and every chain slot). Error(Api)
    /// otherwise, nothing changes. Ends the pending verify.
    void commit_tree_leaf(std::size_t rows);
    /// True when a tree verify of a `rows`-row chain fits the ring (P >= rows + 2).
    [[nodiscard]] bool fits_tree_leaf(std::size_t rows) const noexcept { return p_ >= rows + 2; }

    void reset() noexcept;  ///< live = slot 0, zeroed; drops a pending verify
    [[nodiscard]] GdnSnapshot snapshot() const;
    /// Writes the snapshot into slab[live] (the other slots are never read before being
    /// rewritten). Error(Api) if the snapshot has the wrong size.
    void restore(const GdnSnapshot& s);
    void copy_from(const GdnState& other);  ///< live state only; shapes must match

    /// Bytes moved by commit_rows_kept(rows, kept): 0 — under the ring the commit is one
    /// integer (ADR-001 §5.3), where the pre-ring form copied a slot into the live state.
    [[nodiscard]] std::uint64_t commit_bytes(std::size_t rows, std::size_t kept) const noexcept;

private:
    // Slab layout (elements): recurrent rings at [layer * P + slot] * recurrent_floats, then
    // conv rings at conv_base + [layer * P + slot] * conv_floats.
    [[nodiscard]] std::size_t rec_at(std::size_t layer, std::size_t slot) const noexcept {
        return (layer * p_ + slot) * shape_.recurrent_floats();
    }
    [[nodiscard]] std::size_t conv_base() const noexcept { return shape_.n_layers * p_ * shape_.recurrent_floats(); }
    [[nodiscard]] std::size_t conv_at(std::size_t layer, std::size_t slot) const noexcept {
        return conv_base() + (layer * p_ + slot) * shape_.conv_floats();
    }
    [[nodiscard]] float* live_rec(std::size_t layer) noexcept { return host_.data() + rec_at(layer, live_); }
    [[nodiscard]] float* live_conv(std::size_t layer) noexcept { return host_.data() + conv_at(layer, live_); }
    /// One physical slot of one layer's ring as a backend operand (attached only).
    [[nodiscard]] backend::TensorRef rec_slot_ref(std::size_t layer, std::size_t slot) const {
        return {dev_.get(), rec_at(layer, slot) * sizeof(float), shape_.recurrent_floats() * sizeof(float), 0};
    }
    [[nodiscard]] backend::TensorRef conv_slot_ref(std::size_t layer, std::size_t slot) const {
        return {dev_.get(), conv_at(layer, slot) * sizeof(float), shape_.conv_floats() * sizeof(float), 0};
    }
    void check_layer(std::size_t layer) const;
    /// read views (recurrent/conv/slab): current when host-coherent or right after pull().
    void check_current(const char* what) const;
    /// ring views (recurrent_ring/conv_ring): only meaningful over the canonical host slab.
    void check_coherent(const char* what) const;
    void zero_live();          ///< zero slab[live] of every layer (host)
    void upload_live();        ///< host slab[live] -> device (device-resident states)
    void gather_live(float* dst) const;        ///< host slab[live] -> [layer][rec|conv]
    void scatter_live(const float* src);       ///< [layer][rec|conv] -> host slab[live]

    GdnShape shape_;
    std::size_t p_;             ///< physical states per layer: max(2, max_slots + 1)
    std::vector<float> host_;   ///< the slab (host canonical, or the device mirror)
    backend::Backend* be_ = nullptr;
    std::unique_ptr<backend::Buffer> dev_;
    bool coherent_ = true;      ///< host_ is the storage (unattached or zero-copy)
    bool pulled_ = false;       ///< device-resident: host_ mirrors the device (set by pull())
    bool needs_zero_ = false;   ///< device-resident: slab[live] must be zeroed before use
    std::uint32_t live_ = 0;
    std::uint32_t base_ = 0;    ///< live at the last begin_step
    // Pending-verify bookkeeping (ADR-001 §5.3): set by mark_slots_written with n_slots > 0,
    // cleared by commit_rows_kept / drop_slots / reset / restore.
    std::size_t pend_rows_ = 0;
    std::size_t pend_slots_ = 0;
    bool pending_ = false;
};

}  // namespace halo::state
