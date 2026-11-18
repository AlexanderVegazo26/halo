// Shared by test_speculative.cpp (CPU) and test_vulkan_speculative.cpp: one speculative tick run either as
// Qwen35::mtp_draft_chain_verify (the draft chain and the trunk verify in one command stream, the verify
// rows embedding the chain's device ids) or as the two calls it replaces, mtp_draft_chain then forward
// with the drafts as host tokens. Both must give bitwise the same drafts, verify rows, costs and state.
#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "halo/models/qwen35.h"
#include "halo/state/sequence.h"

namespace halo::test {

/// One sequence of the tick: ready to draft (MTP queue flushed, so mtp_kv length == length() - 1),
/// `x` its pending token, `k` its drafts (0 = no chain step: a plain one-row verify, chain_of = -1).
struct ChainVerifySeq {
    state::SequenceState* seq = nullptr;
    std::int32_t x = 0;
    std::size_t k = 0;
};

struct ChainVerifyTick {
    models::StepResult draft;   ///< one entry per sequence with k > 0, in order
    models::StepResult verify;  ///< one entry per sequence
};

/// Runs the tick fused (mtp_draft_chain_verify) or as mtp_draft_chain + forward.
inline ChainVerifyTick run_chain_verify_tick(const models::Qwen35& m, std::span<const ChainVerifySeq> in, bool fused) {
    const std::size_t n = in.size();
    std::vector<models::MtpStep> mtp;
    std::vector<std::size_t> ks;
    std::vector<std::ptrdiff_t> chain_of(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        const ChainVerifySeq& c = in[i];
        if (c.k == 0) continue;
        chain_of[i] = static_cast<std::ptrdiff_t>(mtp.size());
        models::MtpStep s;
        s.tokens = std::span(&c.x, 1);
        s.hidden = c.seq->last_hidden;
        s.kv = &*c.seq->mtp_kv;
        s.first_position = static_cast<std::int32_t>(c.seq->length());
        mtp.push_back(s);
        ks.push_back(c.k);
    }
    ChainVerifyTick t;
    if (!fused) m.mtp_draft_chain(mtp, ks, t.draft);
    std::vector<std::vector<std::int32_t>> fed(n);
    std::vector<std::vector<std::size_t>> rows(n);
    std::vector<models::SeqStep> verify(n);
    for (std::size_t i = 0; i < n; ++i) {
        const ChainVerifySeq& c = in[i];
        fed[i].assign(1, c.x);
        for (std::size_t d = 0; d < c.k; ++d) {
            // Fused: placeholder ids the chain's device ids replace; otherwise the drafts as host tokens.
            fed[i].push_back(fused ? 0 : t.draft.seqs.at(static_cast<std::size_t>(chain_of[i])).argmax.at(d).index);
        }
        for (std::size_t r = 0; r < fed[i].size(); ++r) rows[i].push_back(r);
        models::SeqStep& v = verify[i];
        v.tokens = fed[i];
        v.kv = &c.seq->kv;
        v.gdn = &c.seq->gdn;
        v.logit_rows = rows[i];
        v.logits = models::LogitsMode::Argmax;
        v.want_hidden = true;
        v.n_state_slots = c.k > 0 ? c.k + 1 : 0;
        v.gdn_path = models::GdnPath::Recurrent;
    }
    if (fused) {
        m.mtp_draft_chain_verify(mtp, ks, t.draft, verify, chain_of, t.verify);
    } else {
        m.forward(verify, t.verify);
    }
    return t;
}

inline void expect_same_cost(const models::StepCost& a, const models::StepCost& b, const std::string& what) {
    EXPECT_EQ(a.weight_passes, b.weight_passes) << what << ": weight_passes";
    EXPECT_EQ(a.weight_bytes, b.weight_bytes) << what << ": weight_bytes";
    EXPECT_EQ(a.embedding_bytes, b.embedding_bytes) << what << ": embedding_bytes";
    EXPECT_EQ(a.activation_bytes, b.activation_bytes) << what << ": activation_bytes";
    EXPECT_EQ(a.state_bytes, b.state_bytes) << what << ": state_bytes";
    EXPECT_EQ(a.kv_bytes, b.kv_bytes) << what << ": kv_bytes";
}

/// Fused tick `f` against the two-call tick `r`: every draft, verify row, hidden and cost bit for bit.
inline void expect_same_tick(const ChainVerifyTick& f, const ChainVerifyTick& r, const std::string& what) {
    ASSERT_EQ(f.draft.seqs.size(), r.draft.seqs.size()) << what << ": chain steps";
    for (std::size_t j = 0; j < r.draft.seqs.size(); ++j) {
        const auto& a = f.draft.seqs[j].argmax;
        const auto& b = r.draft.seqs[j].argmax;
        ASSERT_EQ(a.size(), b.size()) << what << ": chain step " << j << " draft count";
        for (std::size_t d = 0; d < b.size(); ++d) {
            EXPECT_EQ(a[d].index, b[d].index) << what << ": chain step " << j << " depth " << d + 1 << " draft id";
            EXPECT_EQ(a[d].value, b[d].value) << what << ": chain step " << j << " depth " << d + 1 << " draft logit";
        }
    }
    ASSERT_EQ(f.verify.seqs.size(), r.verify.seqs.size()) << what << ": verify steps";
    for (std::size_t i = 0; i < r.verify.seqs.size(); ++i) {
        const auto& a = f.verify.seqs[i].argmax;
        const auto& b = r.verify.seqs[i].argmax;
        ASSERT_EQ(a.size(), b.size()) << what << ": verify step " << i << " rows";
        for (std::size_t row = 0; row < b.size(); ++row) {
            EXPECT_EQ(a[row].index, b[row].index) << what << ": verify step " << i << " row " << row
                                                  << " target id (row embedded from the chain's device id?)";
            EXPECT_EQ(a[row].value, b[row].value) << what << ": verify step " << i << " row " << row << " target logit";
        }
        EXPECT_TRUE(f.verify.seqs[i].hidden == r.verify.seqs[i].hidden) << what << ": verify step " << i << " hidden rows differ";
    }
    expect_same_cost(f.draft.cost, r.draft.cost, what + " draft cost");
    expect_same_cost(f.verify.cost, r.verify.cost, what + " verify cost");
}

/// The state one sequence was left in by each side of the tick, bit for bit: trunk KV, GDN state and its
/// pending verify rows, MTP KV. `dump(kv)` returns a KV's rows (the backend decides how to read them).
template <class Dump>
void expect_same_seq_state(state::SequenceState& a, state::SequenceState& b, Dump&& dump, const std::string& what) {
    EXPECT_EQ(a.kv.length(), b.kv.length()) << what << ": trunk KV length after the verify";
    EXPECT_TRUE(dump(a.kv) == dump(b.kv)) << what << ": trunk KV rows differ";
    EXPECT_TRUE(a.gdn.snapshot().data == b.gdn.snapshot().data) << what << ": GDN state differs";
    EXPECT_EQ(a.gdn.slot_rows(), b.gdn.slot_rows()) << what << ": pending verify rows";
    ASSERT_TRUE(a.mtp_kv && b.mtp_kv) << what << ": both sequences must keep their MTP KV";
    EXPECT_EQ(a.mtp_kv->length(), b.mtp_kv->length()) << what << ": MTP KV length after the chain";
    EXPECT_TRUE(dump(*a.mtp_kv) == dump(*b.mtp_kv)) << what << ": MTP KV rows differ";
}

}  // namespace halo::test
