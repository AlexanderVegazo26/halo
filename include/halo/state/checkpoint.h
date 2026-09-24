#pragma once
// GDN prefix checkpoints (D-013): full GDN-state snapshots keyed by the exact token prefix
// they were taken after, kept under a byte budget.
//
// A hybrid model cannot reuse KV alone: the GDN state at position P summarizes the whole
// prefix. A prefix-cache hit therefore restores the newest checkpoint at a position
// <= the longest common prefix (LCP), reuses the KV blocks up to it, and recomputes only
// the tail. Where checkpoints are *placed* (preamble end, user-message starts, N-k, minimum
// spacing) is the runtime's policy; this store only holds and finds them.
//
// A checkpoint also carries the trunk hidden h_{P-1} (SequenceState::last_hidden): without
// it the MTP drafter has no hidden to pair with the next token after a restore (D-005).
//
// Budget: bytes() of every entry (snapshot + hidden + token ids) sum to <= budget. Insert
// evicts least-recently-used entries until the new one fits; an entry larger than the whole
// budget is rejected (returns false, store unchanged). D-013 places the budget in the GPU
// pool; the CPU reference keeps it in host memory, sized by the memory planner.
//
// Thread-safety: none (owned by the single scheduler/engine worker).

#include <cstddef>
#include <cstdint>
#include <list>
#include <span>
#include <vector>

#include "halo/state/gdn_state.h"

namespace halo::state {

struct Checkpoint {
    std::vector<std::int32_t> tokens;  ///< the prefix: state is after tokens[0..P)
    GdnSnapshot gdn;
    std::vector<float> last_hidden;    ///< trunk hidden h_{P-1}
    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return gdn.bytes() + last_hidden.size() * sizeof(float) + tokens.size() * sizeof(std::int32_t);
    }
};

class CheckpointStore {
public:
    explicit CheckpointStore(std::uint64_t budget_bytes) : budget_(budget_bytes) {}

    /// Stores `c`, replacing an entry with the same token prefix and evicting LRU entries as
    /// needed. Returns false (and changes nothing) if `c` alone exceeds the budget or has no
    /// tokens.
    bool insert(Checkpoint c);

    /// The longest checkpoint whose tokens are a prefix of `query` and whose length is
    /// <= max_len (pass the LCP with the cached sequence, or query.size()). nullptr if none.
    /// Marks the hit most recently used. The pointer is valid until the next insert/clear.
    [[nodiscard]] const Checkpoint* find(std::span<const std::int32_t> query, std::size_t max_len);

    void clear() noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::uint64_t used_bytes() const noexcept { return used_; }
    [[nodiscard]] std::uint64_t budget_bytes() const noexcept { return budget_; }
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }

private:
    std::uint64_t budget_;
    std::uint64_t used_ = 0;
    std::uint64_t evictions_ = 0;
    std::list<Checkpoint> entries_;  // front = most recently used
};

}  // namespace halo::state
