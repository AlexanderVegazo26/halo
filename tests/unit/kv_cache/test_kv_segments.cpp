// Segmented (per-layer) KV pool: a Placement::PerLayer pool stores one backend buffer per attention
// layer so a pool larger than a backend's single-buffer cap (RADV: 4 GiB) can exist; the attention /
// kv-write kernels are called per layer with that layer's buffer and (n_layers = 1, layer = 0).
//
// CPU-verifiable logic only (host accessors, block bookkeeping, and the CPU backend ops through
// KvPool::layer_image()): a sequence whose blocks are scattered over every segment must read, write,
// copy-on-write and attend exactly like the same sequence in a contiguous single-image pool. The
// Vulkan shaders are unchanged by design (they already take n_layers / layer); their segmented
// behaviour is covered by the same call shape in tests/unit/vulkan, not here.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/thread_pool.h"
#include "halo/core/error.h"
#include "halo/kv_cache/paged_kv.h"

namespace cpu = halo::cpu;
namespace hb = halo::backend;
using halo::kv_cache::BlockId;
using halo::kv_cache::KvLayout;
using halo::kv_cache::KvPool;
using halo::kv_cache::Placement;
using halo::kv_cache::SequenceKv;

namespace {

constexpr std::size_t kLayers = 3, kNKv = 2, kHd = 16, kKvd = kNKv * kHd, kBt = 4, kNHead = 4;
constexpr KvLayout kLayout{kLayers, kKvd, kBt};

std::vector<float> rnd(std::size_t n, std::uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// What one scripted run produced; compared across placements.
struct ScriptResult {
    std::vector<std::vector<float>> attn;  ///< backend attention output per (sequence, layer)
    std::vector<std::vector<float>> ref;   ///< cpu::attention_gqa over the host views, same order
    std::vector<float> image;              ///< KvPool::read_all() (logical block-major)
    std::vector<std::vector<BlockId>> tables;
    std::size_t segments = 0;
    std::size_t total_blocks = 0;
};

/// Two sequences of 37 and 30 rows appended in interleaved chunks through the CPU backend's
/// kv_write, so their blocks are scattered over the pool; then a 3-query attention per layer.
ScriptResult run_script(Placement placement, std::size_t n_blocks) {
    cpu::ThreadPool tp{2};
    const std::unique_ptr<hb::Backend> be = hb::make_cpu_backend(&tp);
    const std::unique_ptr<hb::Stream> st = be->create_stream();
    KvPool pool(kLayout, n_blocks, placement);
    pool.attach(*be);
    std::vector<std::unique_ptr<hb::Buffer>> keep;
    const auto imp = [&](auto& vec) {
        keep.push_back(be->import_host(std::as_writable_bytes(std::span(vec))));
        return hb::TensorRef::of(*keep.back());
    };

    constexpr std::size_t kRows[2] = {37, 30};
    std::vector<std::vector<float>> kdata, vdata;  // [seq * kLayers + layer] -> kRows * kKvd
    for (std::size_t s = 0; s < 2; ++s) {
        for (std::size_t l = 0; l < kLayers; ++l) {
            kdata.push_back(rnd(kRows[s] * kKvd, static_cast<std::uint32_t>(100 + s * 10 + l)));
            vdata.push_back(rnd(kRows[s] * kKvd, static_cast<std::uint32_t>(200 + s * 10 + l)));
        }
    }
    SequenceKv seqs[2] = {SequenceKv(pool), SequenceKv(pool)};
    const auto append = [&](std::size_t s, std::size_t n) {
        SequenceKv& kv = seqs[s];
        kv.reserve(n);
        const std::size_t len = kv.length();
        std::vector<std::uint32_t> table(kv.blocks().begin(), kv.blocks().end());
        const hb::TensorRef rt = imp(table);
        for (std::size_t l = 0; l < kLayers; ++l) {
            std::vector<float> k(kdata[s * kLayers + l].begin() + static_cast<std::ptrdiff_t>(len * kKvd),
                                 kdata[s * kLayers + l].begin() + static_cast<std::ptrdiff_t>((len + n) * kKvd));
            std::vector<float> v(vdata[s * kLayers + l].begin() + static_cast<std::ptrdiff_t>(len * kKvd),
                                 vdata[s * kLayers + l].begin() + static_cast<std::ptrdiff_t>((len + n) * kKvd));
            const KvPool::LayerImage img = pool.layer_image(l);
            hb::KvWriteArgs w{};
            w.kv_pool = img.ref;
            w.block_table = rt;
            w.k = imp(k);
            w.v = imp(v);
            w.n_pool_blocks = static_cast<std::uint32_t>(pool.total_blocks());
            w.n_layers = img.n_layers;
            w.layer = img.layer;
            w.block_tokens = kBt;
            w.kv_dim = kKvd;
            w.n_block_table = static_cast<std::uint32_t>(table.size());
            w.start = static_cast<std::uint32_t>(len);
            w.n_tokens = static_cast<std::uint32_t>(n);
            be->kv_write(*st, w);
        }
        st->submit();
        st->wait();
        kv.commit(n);
    };
    constexpr std::size_t kChunkA[4] = {9, 9, 9, 10}, kChunkB[4] = {7, 7, 7, 9};
    for (std::size_t i = 0; i < 4; ++i) {
        append(0, kChunkA[i]);
        append(1, kChunkB[i]);
    }

    ScriptResult run;
    constexpr std::size_t T = 3;
    const float scale = 0.25f;
    for (std::size_t s = 0; s < 2; ++s) {
        const std::size_t len = seqs[s].length();
        std::vector<std::uint32_t> table(seqs[s].blocks().begin(), seqs[s].blocks().end());
        run.tables.emplace_back(seqs[s].blocks().begin(), seqs[s].blocks().end());
        const hb::TensorRef rt = imp(table);
        for (std::size_t l = 0; l < kLayers; ++l) {
            std::vector<float> q = rnd(T * kNHead * kHd, static_cast<std::uint32_t>(300 + s * 10 + l));
            std::vector<float> out(T * kNHead * kHd, 0.0f), ref(T * kNHead * kHd, 0.0f);
            const KvPool::LayerImage img = pool.layer_image(l);
            hb::AttentionArgs a{};
            a.q = imp(q);
            a.kv_pool = img.ref;
            a.block_table = rt;
            a.out = imp(out);
            a.n_pool_blocks = static_cast<std::uint32_t>(pool.total_blocks());
            a.n_layers = img.n_layers;
            a.layer = img.layer;
            a.block_tokens = kBt;
            a.n_block_table = static_cast<std::uint32_t>(table.size());
            a.n_head = kNHead;
            a.n_kv_head = kNKv;
            a.head_dim = kHd;
            a.n_tokens = T;
            a.q_offset = static_cast<std::uint32_t>(len - T);
            a.scale = scale;
            be->attention(*st, a);
            st->submit();
            st->wait();
            std::vector<const float*> kt, vt;
            const cpu::PagedRows keys = seqs[s].keys(l, len, kt);
            const cpu::PagedRows vals = seqs[s].values(l, len, vt);
            cpu::attention_gqa({kNHead, kNKv, kHd}, cpu::ConstRows(std::span<const float>(q), T, kNHead * kHd), keys, vals,
                               len - T, scale, cpu::Rows(std::span(ref), T, kNHead * kHd));
            run.attn.push_back(std::move(out));
            run.ref.push_back(std::move(ref));
        }
    }
    run.image = pool.read_all();
    run.segments = pool.segment_count();
    run.total_blocks = pool.total_blocks();
    return run;
}

}  // namespace

// ---- cap arithmetic and the address mapping ---------------------------------------------------

TEST(KvSegments, CapArithmeticForTheProductionLayout) {
    // 27B shape (memory::qwen38_27b_shape): 16 attention layers, 4 KV heads x 256, 16-token blocks.
    const KvLayout f32{16, 1024, 16};
    EXPECT_EQ(f32.block_bytes(), 2u << 20) << "2 MiB per block = 128 KiB per token";
    EXPECT_EQ(f32.layer_block_bytes(), 128u << 10);
    constexpr std::uint64_t cap = 4ull << 30;
    // Contiguous: the pre-segmentation limit, "a 4 GiB pool holds about 32k tokens" (docs/vulkan.md).
    EXPECT_EQ(KvPool::max_blocks(f32, Placement::Single, cap), 2048u);
    // Split per layer: n_layers times more blocks per buffer, 512Ki tokens of fp32.
    EXPECT_EQ(KvPool::max_blocks(f32, Placement::PerLayer, cap), 32768u);
    EXPECT_EQ(KvPool::max_blocks(f32, Placement::PerLayer, 0), std::numeric_limits<BlockId>::max()) << "0 = no cap";
    EXPECT_EQ(KvPool::max_blocks(f32, Placement::PerLayer, 100), 0u) << "not even one block";
    const KvLayout f16{16, 1024, 16, hb::KvType::F16}, q8{16, 1024, 16, hb::KvType::Q8};
    EXPECT_EQ(KvPool::max_blocks(f16, Placement::PerLayer, cap), 65536u);
    EXPECT_EQ(f16.block_bytes(), 1u << 20);
    EXPECT_EQ(q8.layer_block_bytes(), 36u * 1024 * 16 / 16) << "36 KiB per layer-block: 16 tokens x 2 x 1152 B";
    EXPECT_EQ(KvPool::max_blocks(q8, Placement::PerLayer, cap), cap / q8.layer_block_bytes());

    // 128k tokens on 2 sequences fits per layer; so does the same plus 2 cache slots' worth of
    // blocks only if the cache share shrinks by the 8 blocks it overshoots (the engine does that).
    const std::size_t per_seq = (131072 + 3 + 1 + 15) / 16 + 1;  // blocks_for(128k) with a 2-deep MTP draft
    EXPECT_LE(2 * per_seq, KvPool::max_blocks(f32, Placement::PerLayer, cap));
    EXPECT_GT(4 * per_seq, KvPool::max_blocks(f32, Placement::PerLayer, cap));
    EXPECT_GT(2 * per_seq, KvPool::max_blocks(f32, Placement::Single, cap)) << "the old contiguous pool could not hold it";

    EXPECT_EQ(KvPool::segment_bytes_for(f32, Placement::Single, 100), 100ull * f32.block_bytes());
    EXPECT_EQ(KvPool::segment_bytes_for(f32, Placement::PerLayer, 100), 100ull * f32.layer_block_bytes());
}

TEST(KvSegments, HostMappingIsLayerMajorDisjointAndCoversThePool) {
    constexpr std::size_t n_blocks = 7;
    KvPool single(kLayout, n_blocks, Placement::Single), per(kLayout, n_blocks, Placement::PerLayer);
    EXPECT_EQ(single.segment_count(), 1u);
    EXPECT_EQ(per.segment_count(), kLayers);
    EXPECT_EQ(per.total_bytes(), single.total_bytes());
    EXPECT_EQ(per.segment_bytes() * kLayers, per.total_bytes());
    EXPECT_EQ(single.segment_bytes(), single.total_bytes());

    const std::size_t row_block = kBt * kKvd;                // one K (or V) slice of a block
    const std::size_t layer_block = 2 * row_block;           // one layer's slice of a block
    const std::size_t slab = n_blocks * layer_block;         // one segment
    const float* base = per.k_rows(0, 0);
    std::vector<int> seen(n_blocks * kLayers * 2, 0);
    for (BlockId b = 0; b < n_blocks; ++b) {
        for (std::size_t l = 0; l < kLayers; ++l) {
            const std::ptrdiff_t k = per.k_rows(b, l) - base, v = per.v_rows(b, l) - base;
            EXPECT_EQ(static_cast<std::size_t>(k), l * slab + b * layer_block) << "segment l, block b";
            EXPECT_EQ(static_cast<std::size_t>(v), static_cast<std::size_t>(k) + row_block);
            for (const std::ptrdiff_t off : {k, v}) {
                ASSERT_EQ(static_cast<std::size_t>(off) % row_block, 0u);
                ASSERT_LT(static_cast<std::size_t>(off) / row_block, seen.size());
                ++seen[static_cast<std::size_t>(off) / row_block];
            }
            // The contiguous pool keeps the block-major mapping the existing tests and kernels assume.
            EXPECT_EQ(static_cast<std::size_t>(single.k_rows(b, l) - single.k_rows(0, 0)),
                      b * kLayout.block_floats() + l * layer_block);
        }
    }
    for (const int c : seen) EXPECT_EQ(c, 1) << "every (block, layer, K|V) slice is addressed exactly once";
}

TEST(KvSegments, LayerImageAddressingAndErrors) {
    cpu::ThreadPool tp{1};
    const std::unique_ptr<hb::Backend> be = hb::make_cpu_backend(&tp);
    KvPool per(kLayout, 5, Placement::PerLayer), single(kLayout, 5, Placement::Single);
    EXPECT_THROW((void)per.layer_image(0), halo::Error) << "unattached";
    per.attach(*be);
    single.attach(*be);
    EXPECT_THROW((void)per.storage_ref(), halo::Error) << "a per-layer pool has no single image";
    EXPECT_THROW((void)per.layer_image(kLayers), halo::Error) << "layer out of range";
    const hb::TensorRef whole = single.storage_ref();
    for (std::size_t l = 0; l < kLayers; ++l) {
        const KvPool::LayerImage s = single.layer_image(l), p = per.layer_image(l);
        EXPECT_EQ(s.ref.buffer, whole.buffer);
        EXPECT_EQ(s.n_layers, kLayers);
        EXPECT_EQ(s.layer, l);
        EXPECT_EQ(p.n_layers, 1u);
        EXPECT_EQ(p.layer, 0u);
        // The layer's buffer wraps exactly that layer's host slab (zero-copy on the CPU backend).
        EXPECT_EQ(static_cast<const void*>(p.ref.buffer->host_ptr()), static_cast<const void*>(per.k_rows(0, l)));
        EXPECT_EQ(p.ref.buffer->bytes(), per.segment_bytes());
        if (l > 0) EXPECT_NE(p.ref.buffer, per.layer_image(l - 1).ref.buffer) << "one buffer per segment";
    }
}

// ---- behaviour equals the contiguous pool -----------------------------------------------------

TEST(KvSegments, SequencesScatteredOverAllSegmentsAttendLikeTheContiguousPool) {
    constexpr std::size_t n_blocks = 40;
    const ScriptResult single = run_script(Placement::Single, n_blocks);
    const ScriptResult per = run_script(Placement::PerLayer, n_blocks);
    EXPECT_EQ(single.segments, 1u);
    EXPECT_EQ(per.segments, kLayers);
    // Same allocation order: identical block tables (the block id -> segment offset mapping is the
    // only difference), and the tables really are scattered (interleaved chunks of two sequences).
    ASSERT_EQ(single.tables, per.tables);
    bool scattered = false;
    for (std::size_t i = 1; i < per.tables[0].size(); ++i) scattered |= per.tables[0][i] != per.tables[0][i - 1] + 1;
    EXPECT_TRUE(scattered) << "the script must not produce a contiguous block run";
    EXPECT_GE(per.tables[0].size(), 10u);
    ASSERT_EQ(single.attn.size(), 2 * kLayers);
    ASSERT_EQ(per.attn.size(), 2 * kLayers);
    for (std::size_t i = 0; i < per.attn.size(); ++i) {
        EXPECT_TRUE(bits_equal(per.attn[i], single.attn[i])) << "(sequence, layer) " << i << " differs between placements";
        EXPECT_TRUE(bits_equal(per.attn[i], per.ref[i])) << "(sequence, layer) " << i << " differs from cpu::attention_gqa";
        EXPECT_TRUE(bits_equal(single.attn[i], single.ref[i])) << "(sequence, layer) " << i;
    }
    EXPECT_TRUE(bits_equal(per.image, single.image)) << "the logical pool image differs";
}

TEST(KvSegments, CopyOnWriteInAPerLayerPoolMatchesTheContiguousPool) {
    const auto script = [](Placement placement, std::vector<float>& a_rows, std::vector<float>& b_rows, std::vector<float>& image,
                           std::uint32_t& shared_rc, BlockId& a_tail, BlockId& b_tail) {
        cpu::ThreadPool tp{1};
        const std::unique_ptr<hb::Backend> be = hb::make_cpu_backend(&tp);
        KvPool pool(kLayout, 24, placement);
        pool.attach(*be);
        SequenceKv a(pool), b(pool);
        const auto put = [&](SequenceKv& s, std::size_t row, float tag) {
            for (std::size_t l = 0; l < kLayers; ++l) {
                std::vector<float> k(kKvd), v(kKvd);
                for (std::size_t c = 0; c < kKvd; ++c) {
                    k[c] = tag + static_cast<float>(l * 1000 + row * 10 + c);
                    v[c] = k[c] + 0.5f;
                }
                s.write(l, row, k, v);
            }
        };
        a.reserve(37);
        for (std::size_t r = 0; r < 37; ++r) put(a, r, 1.0f);
        a.commit(37);
        b.share_prefix(a, 21);  // 21 rows = 5 full blocks + 1 row of the sixth
        b.reserve(2);           // rows 21, 22 fall into the shared sixth block: copy-on-write
        for (std::size_t r = 21; r < 23; ++r) put(b, r, 5000.0f);
        b.commit(2);
        shared_rc = pool.refcount(a.blocks()[0]);
        a_tail = a.blocks()[5];
        b_tail = b.blocks()[5];
        const auto dump = [&](const SequenceKv& s, std::vector<float>& out) {
            for (std::size_t l = 0; l < kLayers; ++l) {
                std::vector<const float*> tk, tv;
                const cpu::PagedRows k = s.keys(l, s.length(), tk), v = s.values(l, s.length(), tv);
                for (std::size_t r = 0; r < s.length(); ++r) {
                    out.insert(out.end(), k.row_ptr(r), k.row_ptr(r) + kKvd);
                    out.insert(out.end(), v.row_ptr(r), v.row_ptr(r) + kKvd);
                }
            }
        };
        dump(a, a_rows);
        dump(b, b_rows);
        image = pool.read_all();
    };
    std::vector<float> sa, sb, si, pa, pb, pi;
    std::uint32_t src = 0, prc = 0;
    BlockId sat = 0, sbt = 0, pat = 0, pbt = 0;
    script(Placement::Single, sa, sb, si, src, sat, sbt);
    script(Placement::PerLayer, pa, pb, pi, prc, pat, pbt);
    EXPECT_EQ(src, 2u) << "the first five blocks are shared";
    EXPECT_EQ(prc, 2u);
    EXPECT_NE(sat, sbt) << "the partially shared block was copied";
    EXPECT_NE(pat, pbt);
    EXPECT_TRUE(bits_equal(pa, sa)) << "the source sequence changed under copy-on-write";
    EXPECT_TRUE(bits_equal(pb, sb));
    EXPECT_TRUE(bits_equal(pi, si));
    // Rows 0..20 of the copy equal the source (layer 0, K of row 20: the shared part of the copied block).
    const std::size_t stride = 2 * kKvd;
    EXPECT_TRUE(std::equal(pb.begin() + 20 * stride, pb.begin() + 21 * stride, pa.begin() + 20 * stride));
    EXPECT_FALSE(std::equal(pb.begin() + 21 * stride, pb.begin() + 22 * stride, pa.begin() + 21 * stride)) << "row 21 diverged";
}
