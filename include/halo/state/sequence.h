#pragma once
// SequenceState: everything one sequence carries between forward calls (ARCHITECTURE
// "SequenceState"): trunk KV, MTP-layer KV, GDN recurrent + conv state with rollback slots,
// and the MTP catch-up bookkeeping of D-005.
//
// MTP pairing (D-005): the MTP row at RoPE position p is MTP(embed(x_p), h_{p-1}), where
// h_{p-1} is the trunk's final output-normed hidden at position p-1. With L = length()
// trunk rows committed (positions 0..L-1), the pairs for positions 1..L-1 are known. They
// are either already in mtp_kv (MTP KV row j holds position j+1) or queued here:
//
//     mtp_kv->length() + mtp_queue_size() == L - 1          (L >= 1, model with MTP)
//     last_hidden == h_{L-1}
//
// The pair for position L also needs x_L, the pending (sampled, not yet fed) token, which
// is not part of this state. The queue lets plain decode (MTP disabled by the profit gate)
// defer the MTP catch-up and batch it into the next draft or flush (one weight pass for
// many rows instead of one per token).
//
// Thread-safety: none (one sequence, one owner).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "halo/kv_cache/paged_kv.h"
#include "halo/state/gdn_state.h"

namespace halo::state {

struct SequenceState {
    /// `mtp` may be null (model without MTP). max_slots = n_draft + 1 (D-012).
    SequenceState(kv_cache::KvPool& trunk, kv_cache::KvPool* mtp, const GdnShape& shape, std::size_t max_slots)
        : kv(trunk), gdn(shape, max_slots) {
        if (mtp != nullptr) mtp_kv.emplace(*mtp);
    }

    kv_cache::SequenceKv kv;
    std::optional<kv_cache::SequenceKv> mtp_kv;
    GdnState gdn;
    /// Trunk final hidden of row length()-1; empty when length() == 0.
    std::vector<float> last_hidden;
    /// Queued MTP pairs for positions mtp_kv->length()+1 .. length()-1: token x_p and the
    /// trunk hidden h_{p-1} (row-major, n_embd floats per pair).
    std::vector<std::int32_t> mtp_queue_tokens;
    std::vector<float> mtp_queue_hidden;

    [[nodiscard]] std::size_t length() const noexcept { return kv.length(); }
    [[nodiscard]] std::size_t mtp_queue_size() const noexcept { return mtp_queue_tokens.size(); }

    /// Empty sequence: KV released, GDN zeroed, MTP bookkeeping dropped.
    void reset() noexcept {
        kv.clear();
        if (mtp_kv) mtp_kv->clear();
        gdn.reset();
        last_hidden.clear();
        mtp_queue_tokens.clear();
        mtp_queue_hidden.clear();
    }
};

}  // namespace halo::state
