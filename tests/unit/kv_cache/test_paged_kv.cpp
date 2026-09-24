// Paged KV cache: fixed pool, typed exhaustion, all-or-nothing reserve, refcounted prefix
// sharing with copy-on-write, truncate, move, and PagedRows views into cpu::attention_gqa.

#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/core/error.h"
#include "halo/kv_cache/paged_kv.h"

using halo::kv_cache::KvLayout;
using halo::kv_cache::KvPool;
using halo::kv_cache::SequenceKv;

namespace {

constexpr KvLayout kLayout{2, 8, 4};  // 2 layers, kv_dim 8, 4 rows per block

/// Deterministic K/V row content for (layer, row, tag).
std::vector<float> row_of(std::size_t layer, std::size_t row, float tag, bool v) {
    std::vector<float> r(kLayout.kv_dim);
    for (std::size_t c = 0; c < r.size(); ++c) {
        r[c] = tag + static_cast<float>(layer * 1000 + row * 10 + c) + (v ? 0.5f : 0.0f);
    }
    return r;
}

void append(SequenceKv& kv, std::size_t n, float tag) {
    kv.reserve(n);
    const std::size_t len = kv.length();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t l = 0; l < kLayout.n_layers; ++l) {
            kv.write(l, len + i, row_of(l, len + i, tag, false), row_of(l, len + i, tag, true));
        }
    }
    kv.commit(n);
}

/// Every row [0, length) of every layer, K and V, read through the paged views.
std::vector<float> dump(const SequenceKv& kv) {
    std::vector<float> out;
    std::vector<const float*> tk, tv;
    for (std::size_t l = 0; l < kLayout.n_layers; ++l) {
        const auto k = kv.keys(l, kv.length(), tk);
        const auto v = kv.values(l, kv.length(), tv);
        for (std::size_t r = 0; r < kv.length(); ++r) {
            out.insert(out.end(), k.row_ptr(r), k.row_ptr(r) + kLayout.kv_dim);
            out.insert(out.end(), v.row_ptr(r), v.row_ptr(r) + kLayout.kv_dim);
        }
    }
    return out;
}

std::vector<float> expected(std::size_t len, std::span<const std::pair<std::size_t, float>> tags) {
    // tags: (first row, tag) segments in order
    std::vector<float> out;
    for (std::size_t l = 0; l < kLayout.n_layers; ++l) {
        for (std::size_t r = 0; r < len; ++r) {
            float tag = 0;
            for (const auto& [first, t] : tags) {
                if (r >= first) tag = t;
            }
            const auto k = row_of(l, r, tag, false), v = row_of(l, r, tag, true);
            out.insert(out.end(), k.begin(), k.end());
            out.insert(out.end(), v.begin(), v.end());
        }
    }
    return out;
}

}  // namespace

TEST(KvPool, LayoutAndConfigErrors) {
    EXPECT_EQ(kLayout.block_floats(), 2u * 2 * 4 * 8);
    EXPECT_THROW((void)KvLayout{}.block_floats(), halo::Error);
    EXPECT_THROW(KvPool(kLayout, 0), halo::Error);
    try {
        (void)KvLayout{SIZE_MAX / 2, 8, 4}.block_floats();
        ADD_FAILURE() << "expected overflow error";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Config);
    }
}

TEST(KvPool, ExhaustionIsTypedMemoryErrorAndAllocateNIsAllOrNothing) {
    KvPool pool(kLayout, 3);
    auto two = pool.allocate_n(2);
    EXPECT_EQ(pool.free_blocks(), 1u);
    try {
        (void)pool.allocate_n(2);
        ADD_FAILURE() << "expected MEMORY_ERROR";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Memory) << e.what();
    }
    EXPECT_EQ(pool.free_blocks(), 1u) << "a failed allocate_n must allocate nothing";
    const auto b = pool.allocate();
    EXPECT_EQ(pool.refcount(b), 1u);
    EXPECT_THROW((void)pool.allocate(), halo::Error);
    pool.retain(b);
    EXPECT_EQ(pool.refcount(b), 2u);
    pool.release(b);
    pool.release(b);
    EXPECT_EQ(pool.refcount(b), 0u);
    pool.release(b);  // double release is ignored, never a double free
    for (const auto x : two) pool.release(x);
    EXPECT_EQ(pool.free_blocks(), 3u);
    EXPECT_THROW(pool.retain(b), halo::Error) << "retain of a free block";
}

TEST(SequenceKv, AppendReadTruncateAndRelease) {
    KvPool pool(kLayout, 8);
    {
        SequenceKv kv(pool);
        append(kv, 6, 0);
        EXPECT_EQ(kv.length(), 6u);
        EXPECT_EQ(kv.blocks().size(), 2u);
        EXPECT_EQ(pool.used_blocks(), 2u);
        const std::pair<std::size_t, float> seg[] = {{0, 0.0f}};
        EXPECT_EQ(dump(kv), expected(6, seg));
        kv.truncate(4);  // exactly one block
        EXPECT_EQ(kv.length(), 4u);
        EXPECT_EQ(pool.used_blocks(), 1u);
        kv.truncate(9);  // never grows
        EXPECT_EQ(kv.length(), 4u);
        append(kv, 3, 7);
        const std::pair<std::size_t, float> seg2[] = {{0, 0.0f}, {4, 7.0f}};
        EXPECT_EQ(dump(kv), expected(7, seg2));
        // write/commit outside the reserved range are programming errors, not UB
        EXPECT_THROW(kv.write(0, 3, row_of(0, 3, 0, false), row_of(0, 3, 0, true)), halo::Error) << "below length";
        EXPECT_THROW(kv.write(0, 8, row_of(0, 8, 0, false), row_of(0, 8, 0, true)), halo::Error) << "past capacity";
        EXPECT_THROW(kv.write(2, 7, row_of(0, 7, 0, false), row_of(0, 7, 0, true)), halo::Error) << "bad layer";
        EXPECT_THROW(kv.commit(2), halo::Error);
    }
    EXPECT_EQ(pool.used_blocks(), 0u) << "destructor releases every block";
}

TEST(SequenceKv, ReserveOnExhaustionIsAtomic) {
    KvPool pool(kLayout, 2);
    SequenceKv kv(pool);
    append(kv, 5, 1);  // 2 blocks
    const auto before = dump(kv);
    const std::vector<halo::kv_cache::BlockId> table(kv.blocks().begin(), kv.blocks().end());
    try {
        kv.reserve(4);  // needs a third block
        ADD_FAILURE() << "expected MEMORY_ERROR";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Memory);
    }
    EXPECT_EQ(kv.length(), 5u);
    EXPECT_EQ(kv.capacity(), 8u);
    EXPECT_EQ(std::vector<halo::kv_cache::BlockId>(kv.blocks().begin(), kv.blocks().end()), table);
    EXPECT_EQ(dump(kv), before);
    EXPECT_EQ(pool.free_blocks(), 0u);
    kv.reserve(3);  // fits in the partially used block
    EXPECT_EQ(pool.free_blocks(), 0u);
}

TEST(SequenceKv, SharedPrefixCopyOnWriteLeavesSourceUnchanged) {
    KvPool pool(kLayout, 16);
    SequenceKv a(pool);
    append(a, 6, 1);  // blocks: [0..3] full, [4..5] partial
    const auto a_before = dump(a);
    SequenceKv b(pool);
    b.share_prefix(a, 6);
    EXPECT_EQ(pool.used_blocks(), 2u) << "sharing copies nothing";
    EXPECT_EQ(pool.refcount(a.blocks()[1]), 2u);
    EXPECT_EQ(dump(b), a_before);
    // b writes into the shared partial block -> copy-on-write of that block only
    append(b, 3, 9);
    EXPECT_EQ(pool.used_blocks(), 4u) << "one COW copy + one fresh block";
    EXPECT_EQ(b.blocks()[0], a.blocks()[0]) << "the full block stays shared";
    EXPECT_NE(b.blocks()[1], a.blocks()[1]);
    EXPECT_EQ(dump(a), a_before) << "the source is never modified";
    const std::pair<std::size_t, float> seg[] = {{0, 1.0f}, {6, 9.0f}};
    EXPECT_EQ(dump(b), expected(9, seg));
    // the source writing past the shared point triggers COW too (block 0 still shared)
    SequenceKv c(pool);
    c.share_prefix(a, 3);  // inside block 0
    append(c, 1, 5);
    const std::pair<std::size_t, float> seg_c[] = {{0, 1.0f}, {3, 5.0f}};
    EXPECT_EQ(dump(c), expected(4, seg_c));
    EXPECT_EQ(dump(a), a_before);
    EXPECT_EQ(dump(b).size(), expected(9, seg).size());
    EXPECT_EQ(dump(b), expected(9, seg));
    // truncating a shared sequence releases only its references
    const auto used = pool.used_blocks();
    c.clear();
    EXPECT_LT(pool.used_blocks(), used);
    EXPECT_EQ(dump(a), a_before);
    // share_prefix preconditions
    EXPECT_THROW(b.share_prefix(a, 1), halo::Error) << "target not empty";
    SequenceKv d(pool);
    EXPECT_THROW(d.share_prefix(a, 7), halo::Error) << "longer than the source";
    KvPool other(kLayout, 2);
    SequenceKv e(other);
    EXPECT_THROW(e.share_prefix(a, 1), halo::Error) << "different pools";
}

TEST(SequenceKv, TruncateInsideSharedBlockThenAppendCopies) {
    KvPool pool(kLayout, 8);
    SequenceKv a(pool);
    append(a, 4, 1);
    SequenceKv b(pool);
    b.share_prefix(a, 4);
    b.truncate(2);           // still references block 0 (shared)
    append(b, 1, 3);         // row 2 lives in the shared block -> must COW
    const std::pair<std::size_t, float> seg_a[] = {{0, 1.0f}};
    EXPECT_EQ(dump(a), expected(4, seg_a));
    const std::pair<std::size_t, float> seg_b[] = {{0, 1.0f}, {2, 3.0f}};
    EXPECT_EQ(dump(b), expected(3, seg_b));
}

TEST(SequenceKv, MoveTransfersOwnership) {
    KvPool pool(kLayout, 4);
    SequenceKv a(pool);
    append(a, 5, 2);
    const auto snap = dump(a);
    SequenceKv b(std::move(a));
    EXPECT_EQ(dump(b), snap);
    EXPECT_EQ(a.length(), 0u);  // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_EQ(pool.used_blocks(), 2u);
    SequenceKv c(pool);
    append(c, 1, 0);
    c = std::move(b);
    EXPECT_EQ(dump(c), snap);
    EXPECT_EQ(pool.used_blocks(), 2u) << "move-assign released c's old block";
}

TEST(SequenceKv, PagedViewsDriveAttentionLikeContiguousRows) {
    // 2 KV heads x 4 dims, 4 query heads; 11 history rows over 3 blocks.
    const halo::cpu::AttentionDims dims{4, 2, 4};
    KvPool pool(kLayout, 4);
    SequenceKv kv(pool);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    const std::size_t T = 11;
    std::vector<float> K(T * 8), V(T * 8);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);
    kv.reserve(T);
    for (std::size_t r = 0; r < T; ++r) {
        for (std::size_t l = 0; l < 2; ++l) {
            kv.write(l, r, std::span<const float>(&K[r * 8], 8), std::span<const float>(&V[r * 8], 8));
        }
    }
    kv.commit(T);
    const std::size_t nq = 3;  // last 3 rows are the queries
    std::vector<float> q(nq * 16);
    for (auto& x : q) x = nd(rng);
    std::vector<float> paged(nq * 16), contig(nq * 16);
    std::vector<const float*> tk, tv;
    halo::cpu::attention_gqa(dims, halo::cpu::ConstRows(std::span<const float>(q), nq, 16), kv.keys(1, T, tk), kv.values(1, T, tv),
                             T - nq, 0.5f, halo::cpu::Rows(std::span(paged), nq, 16));
    halo::cpu::attention_gqa(dims, halo::cpu::ConstRows(std::span<const float>(q), nq, 16),
                             halo::cpu::PagedRows::contiguous(K.data(), T, 8, 8), halo::cpu::PagedRows::contiguous(V.data(), T, 8, 8),
                             T - nq, 0.5f, halo::cpu::Rows(std::span(contig), nq, 16));
    EXPECT_EQ(paged, contig) << "the paged view must present exactly the written rows";
    EXPECT_THROW((void)kv.keys(0, 13, tk), halo::Error) << "view past capacity";
}
