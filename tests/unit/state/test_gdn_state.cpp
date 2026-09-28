// GDN state (D-003, D-012): shape sizes, rollback slots driven by the real cpu GDN and
// conv ops, commit_rows_kept == running only the kept rows (bitwise), error atomicity,
// snapshot / restore; and the D-013 CheckpointStore.

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/core/error.h"
#include "halo/state/checkpoint.h"
#include "halo/state/gdn_state.h"
#include "halo/state/sequence.h"

using halo::state::Checkpoint;
using halo::state::CheckpointStore;
using halo::state::GdnShape;
using halo::state::GdnState;

namespace {

// 2 layers, 2 k heads / 4 v heads (tiled), d_k = d_v = 8, conv kernel 4 over 48 channels.
const halo::cpu::GdnDims kDims{2, 4, 8, 8, halo::cpu::GdnHeadMapping::Tiled};
const GdnShape kShape{2, 4, 8, 8, 4, 48};

struct Inputs {
    std::size_t T;
    std::vector<float> q, k, v, g, beta, x, w;
};

Inputs make_inputs(std::size_t T, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd;
    std::uniform_real_distribution<float> ud(0.05f, 0.95f);
    Inputs in{T, {}, {}, {}, {}, {}, {}, {}};
    in.q.resize(T * 16);
    in.k.resize(T * 16);
    in.v.resize(T * 32);
    in.g.resize(T * 4);
    in.beta.resize(T * 4);
    in.x.resize(T * 48);
    in.w.resize(48 * 4);
    for (auto* vec : {&in.q, &in.k, &in.v, &in.x, &in.w}) {
        for (auto& f : *vec) f = nd(rng);
    }
    for (auto& f : in.g) f = -ud(rng);
    for (auto& f : in.beta) f = ud(rng);
    return in;
}

/// Runs rows [r0, r0 + n) of `in` through layer `layer` of `s` (conv then GDN) over the
/// state ring, writing n_slots slots.
void run(GdnState& s, std::size_t layer, const Inputs& in, std::size_t r0, std::size_t n, std::size_t n_slots) {
    using halo::cpu::ConstRows;
    using halo::cpu::Rows;
    std::vector<float> x(in.x.begin() + static_cast<std::ptrdiff_t>(r0 * 48), in.x.begin() + static_cast<std::ptrdiff_t>((r0 + n) * 48));
    halo::cpu::causal_conv1d_silu(ConstRows(std::span<const float>(x), n, 48), ConstRows(std::span<const float>(in.w), 48, 4),
                                  s.conv_ring(layer, n_slots), Rows(std::span(x), n, 48));
    const halo::cpu::GdnInputs gi{
        ConstRows(&in.q[r0 * 16], n, 16, 16), ConstRows(&in.k[r0 * 16], n, 16, 16), ConstRows(&in.v[r0 * 32], n, 32, 32),
        ConstRows(&in.g[r0 * 4], n, 4, 4),    ConstRows(&in.beta[r0 * 4], n, 4, 4),
    };
    std::vector<float> o(n * 32);
    halo::cpu::gated_delta_rule_recurrent(kDims, gi, s.recurrent_ring(layer, n_slots), Rows(std::span(o), n, 32),
                                          halo::cpu::GdnQkParams{.qk_l2norm = true, .q_scale = 0.25f});
}

/// A forward over n rows, as models::Qwen35 drives the state: record the base, run every
/// layer, commit (live advance) with the slots recorded.
void run_all_layers(GdnState& s, const Inputs& a, const Inputs& b, std::size_t r0, std::size_t n, std::size_t n_slots) {
    s.begin_step(n, n_slots);
    run(s, 0, a, r0, n, n_slots);
    run(s, 1, b, r0, n, n_slots);
    s.mark_slots_written(n, n_slots);
}

}  // namespace

TEST(GdnShape, RealQwen38SizesMatchD003) {
    // D-003: 48 GDN layers x 48 v heads x 128 x 128 fp32 = 144 MiB; conv 48 x 3 x 10240 fp32.
    const GdnShape real{48, 48, 128, 128, 4, 10240};
    EXPECT_EQ(real.n_layers * real.recurrent_floats() * sizeof(float), 144ull << 20);
    EXPECT_EQ(real.n_layers * real.conv_floats() * sizeof(float), 48ull * 3 * 10240 * 4);
    EXPECT_EQ(real.total_bytes(), (144ull << 20) + 48ull * 3 * 10240 * 4);
    EXPECT_THROW((void)GdnShape{}.recurrent_floats(), halo::Error);
    EXPECT_THROW((void)(GdnShape{1, 1, 1, 1, 0, 1}.conv_floats()), halo::Error);
    EXPECT_THROW(GdnState(GdnShape{SIZE_MAX / 4, 4, 8, 8, 4, 48}, 1), halo::Error);
}

TEST(GdnState, CommitRowsKeptEqualsRunningOnlyTheKeptRowsBitwise) {
    const std::size_t T = 5;  // verify rows = n_draft + 1
    const Inputs a = make_inputs(3 + T, 1), b = make_inputs(3 + T, 2);
    for (std::size_t kept = 1; kept <= T; ++kept) {
        SCOPED_TRACE(kept);
        // history of 3 rows, then a slot-writing call over T rows, keep `kept`
        GdnState spec(kShape, T);
        run_all_layers(spec, a, b, 0, 3, 0);
        run_all_layers(spec, a, b, 3, T, T);
        EXPECT_EQ(spec.slot_rows(), T);
        EXPECT_EQ(spec.slots_valid(), T);
        spec.commit_rows_kept(T, kept);
        EXPECT_EQ(spec.slots_valid(), 0u) << "commit ends the pending verify";
        // reference: the same history, then only the kept rows, no slots
        GdnState ref(kShape, 0);
        run_all_layers(ref, a, b, 0, 3, 0);
        run_all_layers(ref, a, b, 3, kept, 0);
        EXPECT_EQ(spec.snapshot().data, ref.snapshot().data) << "recurrent + conv state of every layer";
        EXPECT_EQ(spec.commit_bytes(T, kept), 0u) << "the ring commit moves one integer, not a state";
    }
}

TEST(GdnState, RingCommitIsOneIntegerFromTheRecordedBase) {
    // ADR-001 §5.3: decode/prefill commit is live = (base+1) mod P; a verify commit is
    // live = (base + 1 + (T - m)) mod P; a failure leaves live unchanged.
    const Inputs a = make_inputs(8, 7), b = make_inputs(8, 8);
    GdnState s(kShape, 2);  // P = 3
    EXPECT_EQ(s.live(), 0u);
    EXPECT_EQ(s.ring_size(), 3u);
    s.begin_step(3, 0);  // a step that then "fails": no mark -> live unchanged
    EXPECT_EQ(s.live(), 0u);
    run_all_layers(s, a, b, 0, 3, 0);
    EXPECT_EQ(s.live(), 1u) << "decode commit: live = base + 1";
    run_all_layers(s, a, b, 3, 2, 0);
    EXPECT_EQ(s.live(), 2u);
    run_all_layers(s, a, b, 5, 3, 0);
    EXPECT_EQ(s.live(), 0u) << "the ring wraps mod P";
    run_all_layers(s, a, b, 0, 5, 2);
    EXPECT_EQ(s.live(), 1u) << "after a clean status, before the verify commit";
    EXPECT_THROW(s.begin_step(1, 0), halo::Error) << "a forward with an uncommitted verify is Error(Api)";
    EXPECT_EQ(s.live(), 1u);
    EXPECT_THROW(s.commit_rows_kept(5, 3), halo::Error) << "slot 2 was never written (only 2 slots requested)";
    s.commit_rows_kept(5, 4);  // keep 4 of 5: live = base(0) + 1 + slot 1
    EXPECT_EQ(s.live(), 2u);
    EXPECT_EQ(s.slots_valid(), 0u);
    // drop_slots accepts the whole call.
    run_all_layers(s, a, b, 0, 4, 2);
    EXPECT_EQ(s.live(), 0u);
    s.drop_slots();
    EXPECT_EQ(s.live(), 0u);
    run_all_layers(s, a, b, 0, 1, 0);
    EXPECT_EQ(s.live(), 1u);
}

TEST(GdnState, FewerSlotsThanRowsLimitsRollbackDepth) {
    const Inputs a = make_inputs(6, 3), b = make_inputs(6, 4);
    GdnState s(kShape, 4);
    run_all_layers(s, a, b, 0, 6, 2);  // only states after rows 5 and 4
    EXPECT_EQ(s.slots_valid(), 2u);
    const auto before = s.snapshot().data;
    EXPECT_THROW(s.commit_rows_kept(6, 4), halo::Error) << "slot 2 was not written";
    EXPECT_THROW(s.commit_rows_kept(5, 5), halo::Error) << "row count mismatch";
    EXPECT_THROW(s.commit_rows_kept(6, 0), halo::Error) << "must keep >= 1 row";
    EXPECT_THROW(s.commit_rows_kept(6, 7), halo::Error);
    EXPECT_EQ(s.snapshot().data, before) << "a rejected commit changes nothing";
    EXPECT_EQ(s.slots_valid(), 2u);
    s.commit_rows_kept(6, 5);
    GdnState ref(kShape, 0);
    run_all_layers(ref, a, b, 0, 5, 0);
    EXPECT_EQ(s.snapshot().data, ref.snapshot().data);
    EXPECT_THROW(s.begin_step(1, 5), halo::Error) << "more slots than the ring holds";
    EXPECT_THROW((void)s.recurrent(2), halo::Error) << "layer out of range";
}

TEST(GdnState, SnapshotRestoreCopyAndReset) {
    const Inputs a = make_inputs(7, 5), b = make_inputs(7, 6);
    GdnState s(kShape, 2);
    run_all_layers(s, a, b, 0, 4, 0);
    const auto snap = s.snapshot();
    EXPECT_EQ(snap.bytes(), kShape.total_bytes());
    run_all_layers(s, a, b, 4, 3, 2);
    EXPECT_NE(s.snapshot().data, snap.data);
    s.restore(snap);
    EXPECT_EQ(s.snapshot().data, snap.data);
    EXPECT_EQ(s.slots_valid(), 0u) << "restore drops slots";
    GdnState t(kShape, 0);
    t.copy_from(s);
    EXPECT_EQ(t.snapshot().data, snap.data);
    halo::state::GdnSnapshot bad;
    bad.data.resize(3);
    EXPECT_THROW(s.restore(bad), halo::Error);
    EXPECT_EQ(s.snapshot().data, snap.data);
    s.reset();
    for (const float f : s.snapshot().data) ASSERT_EQ(f, 0.0f);
    GdnState other(GdnShape{1, 4, 8, 8, 4, 48}, 0);
    EXPECT_THROW(other.copy_from(s), halo::Error);
}

// ---- CheckpointStore (D-013) -----------------------------------------------------------

namespace {

Checkpoint ckpt(std::vector<std::int32_t> toks, std::size_t floats, float fill) {
    Checkpoint c;
    c.tokens = std::move(toks);
    c.gdn.data.assign(floats, fill);
    c.last_hidden.assign(4, fill);
    return c;
}

}  // namespace

TEST(CheckpointStore, LongestPrefixAtOrBelowTheLimit) {
    CheckpointStore st(1 << 20);
    ASSERT_TRUE(st.insert(ckpt({1, 2}, 8, 2)));
    ASSERT_TRUE(st.insert(ckpt({1, 2, 3, 4}, 8, 4)));
    ASSERT_TRUE(st.insert(ckpt({1, 2, 3, 4, 5, 6}, 8, 6)));
    ASSERT_TRUE(st.insert(ckpt({9, 9, 9}, 8, 9)));
    const std::vector<std::int32_t> q = {1, 2, 3, 4, 5, 7, 7};
    const Checkpoint* c = st.find(q, q.size());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->tokens.size(), 4u) << "{1..6} diverges at 5; {1..4} is the longest prefix";
    EXPECT_EQ(c->last_hidden[0], 4.0f);
    c = st.find(q, 3);  // LCP limit below 4
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->tokens.size(), 2u);
    EXPECT_EQ(st.find(q, 1), nullptr);
    const std::vector<std::int32_t> none = {5, 5};
    EXPECT_EQ(st.find(none, 2), nullptr);
    const std::vector<std::int32_t> exact = {1, 2, 3, 4, 5, 6};
    ASSERT_NE(st.find(exact, 6), nullptr);
    EXPECT_EQ(st.find(exact, 6)->tokens.size(), 6u) << "a checkpoint at exactly the limit is usable";
}

TEST(CheckpointStore, BudgetEvictsLeastRecentlyUsedAndRejectsOversize) {
    const std::uint64_t one = ckpt({1}, 100, 0).bytes();
    CheckpointStore st(3 * one + 2);  // room for three single-token entries
    ASSERT_TRUE(st.insert(ckpt({1}, 100, 1)));
    ASSERT_TRUE(st.insert(ckpt({2}, 100, 2)));
    ASSERT_TRUE(st.insert(ckpt({3}, 100, 3)));
    EXPECT_EQ(st.used_bytes(), 3 * one);
    const std::vector<std::int32_t> q1 = {1};
    ASSERT_NE(st.find(q1, 1), nullptr);  // {1} becomes most recent; {2} is now LRU
    ASSERT_TRUE(st.insert(ckpt({4}, 100, 4)));
    EXPECT_EQ(st.size(), 3u);
    EXPECT_EQ(st.evictions(), 1u);
    const std::vector<std::int32_t> q2 = {2};
    EXPECT_EQ(st.find(q2, 1), nullptr) << "LRU entry evicted";
    EXPECT_NE(st.find(q1, 1), nullptr);
    EXPECT_LE(st.used_bytes(), st.budget_bytes());
    // An entry larger than the whole budget is rejected and changes nothing.
    EXPECT_FALSE(st.insert(ckpt({7}, 1000, 7)));
    EXPECT_EQ(st.size(), 3u);
    EXPECT_EQ(st.used_bytes(), 3 * one);
    EXPECT_FALSE(st.insert(ckpt({}, 1, 0))) << "an empty prefix is not a checkpoint";
    // Same prefix replaces, never duplicates.
    ASSERT_TRUE(st.insert(ckpt({4}, 100, 44)));
    EXPECT_EQ(st.size(), 3u);
    const std::vector<std::int32_t> q4 = {4};
    EXPECT_EQ(st.find(q4, 1)->last_hidden[0], 44.0f);
    st.clear();
    EXPECT_EQ(st.size(), 0u);
    EXPECT_EQ(st.used_bytes(), 0u);
}

TEST(SequenceState, ResetDropsEverything) {
    halo::kv_cache::KvPool trunk({1, 4, 4}, 4), mtp({1, 4, 4}, 4);
    halo::state::SequenceState s(trunk, &mtp, kShape, 2);
    ASSERT_TRUE(s.mtp_kv.has_value());
    s.kv.reserve(3);
    s.kv.commit(3);
    s.mtp_kv->reserve(2);
    s.mtp_kv->commit(2);
    s.last_hidden.assign(4, 1.0f);
    s.mtp_queue_tokens.push_back(5);
    s.mtp_queue_hidden.assign(4, 2.0f);
    s.gdn.recurrent(0)[0] = 3.0f;
    s.reset();
    EXPECT_EQ(s.length(), 0u);
    EXPECT_EQ(s.mtp_kv->length(), 0u);
    EXPECT_EQ(trunk.used_blocks(), 0u);
    EXPECT_EQ(mtp.used_blocks(), 0u);
    EXPECT_TRUE(s.last_hidden.empty());
    EXPECT_EQ(s.mtp_queue_size(), 0u);
    EXPECT_EQ(s.gdn.recurrent(0)[0], 0.0f);
    halo::state::SequenceState no_mtp(trunk, nullptr, kShape, 0);
    EXPECT_FALSE(no_mtp.mtp_kv.has_value());
}
