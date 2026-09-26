// KV write and paged GQA ATTENTION of the Vulkan backend (WS-F2 V2), differentially tested
// against halo::kv_cache (SequenceKv::write) and cpu::attention_gqa on the same inputs
// (TRD §30; code review M-2). Verified on lavapipe only (DECISIONS D-001).
//
// The K/V history lives in a real halo::kv_cache::KvPool (two sequences grown in alternation,
// so block tables are fragmented; a partially filled last block; layer 1 of 2). The Vulkan ops
// read a byte copy of the pool storage plus the sequence's block ids; the CPU reads
// SequenceKv::keys()/values(). `PoolLayoutIsTheDocumentedOne` pins the layout both assume.
//
// KV write is a copy: the pool after the Vulkan write must equal, byte for byte, the pool after
// SequenceKv::write of the same rows (so nothing else — other layers, other sequences' blocks —
// is touched).
//
// ATTENTION bound (first order, u = 2^-24), per output element (t, h, d):
//   |o_vk - o_cpu| <= u * (4N + 8*n_tiles + 8R + 24 + 4D) * sum_s w_s |v_s[d]| + FLT_MIN
// N = keys attended, n_tiles = ceil(N / attention_workgroup), R = max_s (m - x_s) the score
// range, w_s the softmax weights, D = (head_dim/8 + 4) * max_s sum_i |q_i k_s,i| * |scale|.
// Terms (relative to the weights, then scaled by sum w|v|):
//   - exp of the shifted score: vk 3 + 2R ulp (Vulkan precision rule for exp), cpu ~1;
//     rounding of the argument x_s - m: R per side                                -> 4R + 4
//   - vk running-max rescale: each alpha = exp(m - m') costs 3 + 2|dm| + |dm| ulp and one
//     multiply of l and of o: over the tiles, sum |dm| <= R                       -> 3R + 6 n_tiles
//   - normalising sum (cpu sequential N; vk tile trees + chain <= N + n_tiles)     -> 2N + n_tiles
//   - weighted accumulation (cpu N; vk N + n_tiles), final division 1 per side     -> 2N + n_tiles + 2
//   - scores: both sides use the CPU's 8-lane dot, uncontracted, so they are identical when the
//     device honours `precise`; the bound does not rely on that: a score error of at most D ulp
//     per side moves every weight by at most 2D relative                          -> 4D
// Summed and rounded up: 4N + 8 n_tiles + 8R + 24 + 4D. Weights, R and D are computed in double
// from the CPU-side inputs (error-bound scaling only).

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "halo/kv_cache/paged_kv.h"
#include "vk_test_util.h"

namespace hv = halo::vulkan;
namespace hc = halo::cpu;
namespace kv = halo::kv_cache;
using hv::test::download;
using hv::test::upload;

namespace {

constexpr std::size_t k_layers = 2;
constexpr std::size_t k_layer = 1;
constexpr double k_u = 0x1p-24;
constexpr double k_floor = 1.1754943508222875e-38;  // FLT_MIN

std::vector<float> uniform(std::mt19937& rng, std::size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

/// A pool holding one sequence `seq` of S rows (and a decoy sequence interleaved with it, so
/// seq's block table is fragmented). K and V rows are distinct random values; `rows_written`
/// of seq are filled (the rest reserved but unwritten, for KV-write tests).
struct PoolFixture {
    std::unique_ptr<kv::KvPool> pool;
    std::unique_ptr<kv::SequenceKv> seq, other;
    std::size_t kv_dim = 0, bt = 0;

    PoolFixture(std::size_t kv_dim_, std::size_t bt_, std::size_t S, std::size_t rows_written, std::mt19937& rng)
        : kv_dim(kv_dim_), bt(bt_) {
        const std::size_t blocks = (S + bt - 1) / bt;
        pool = std::make_unique<kv::KvPool>(kv::KvLayout{k_layers, kv_dim, bt}, 2 * blocks + 3);
        seq = std::make_unique<kv::SequenceKv>(*pool);
        other = std::make_unique<kv::SequenceKv>(*pool);
        std::vector<float> krow(kv_dim), vrow(kv_dim);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (std::size_t s = 0; s < S; ++s) {
            if (s % bt == 0) {  // the decoy takes the next block first
                other->reserve(bt);
                for (std::size_t j = 0; j < bt; ++j) {
                    for (auto& x : krow) x = u(rng);
                    for (std::size_t L = 0; L < k_layers; ++L) other->write(L, other->length() + j, krow, krow);
                }
                other->commit(bt);
            }
            if (s < rows_written) {  // committed history
                seq->reserve(1);
                for (std::size_t L = 0; L < k_layers; ++L) {
                    for (auto& x : krow) x = u(rng) * 2.0f;
                    for (auto& x : vrow) x = u(rng);
                    seq->write(L, s, krow, vrow);
                }
                seq->commit(1);
            } else {  // reserved (writable) rows [rows_written, s], one block at a time
                seq->reserve(s + 1 - seq->length());
            }
        }
        EXPECT_GE(seq->capacity(), S);
    }

    [[nodiscard]] std::size_t pool_floats() const { return pool->total_blocks() * pool->layout().block_floats(); }
    [[nodiscard]] std::vector<float> pool_bytes() const {
        std::vector<float> v(pool_floats());
        std::memcpy(v.data(), pool->k_rows(0, 0), v.size() * sizeof(float));
        return v;
    }
    [[nodiscard]] std::vector<std::uint32_t> table() const {
        return {seq->blocks().begin(), seq->blocks().end()};
    }
};

bool bitwise(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

/// Data uploaded at byte `lead` of a buffer whose other bytes are 0xFF, used through a view at
/// that offset. Leads that are not multiples of the device alignment (16 on lavapipe) make every
/// kernel offset term (the view's sub-alignment remainder) load-bearing.
struct Placed {
    hv::Buffer buf;
    std::uint64_t lead = 0, bytes = 0, row_stride = 0;
    [[nodiscard]] hv::BufferView view() const { return hv::BufferView(buf, lead, bytes, row_stride); }
    template <typename T>
    [[nodiscard]] std::vector<T> read(std::size_t count) const {
        std::vector<T> out(count);
        buf.download(std::span<T>(out), lead);
        return out;
    }
};

template <typename T>
Placed place(const std::shared_ptr<hv::Context>& ctx, std::span<const T> data, std::uint64_t lead,
             std::uint64_t row_stride = 0, hv::MemoryUsage usage = hv::MemoryUsage::HostCached) {
    std::vector<std::uint8_t> bytes(lead + data.size_bytes() + 16, 0xFF);
    std::memcpy(bytes.data() + lead, data.data(), data.size_bytes());
    return Placed{upload(ctx, std::span<const std::uint8_t>(bytes), usage), lead, data.size_bytes(), row_stride};
}

/// A status word at byte 4 of a 16-byte buffer, pre-set to garbage (the op must zero it).
Placed place_status(const std::shared_ptr<hv::Context>& ctx) {
    const std::vector<std::uint32_t> w{0xDEADBEEFu};
    return place(ctx, std::span<const std::uint32_t>(w), 4);
}

struct AttnCase {
    std::uint32_t n_head = 24, n_kv = 4, hd = 256;  // qwen35 (D-004)
    std::uint32_t bt = 16;
    std::uint32_t T = 1;
    std::uint32_t q_offset = 40;
    std::uint32_t q_head_stride = 0;  // 0 = dense; 2*hd = interleaved [Q | gate]
    std::uint32_t wg = 128;
    bool bad_block = false;
    std::uint32_t seed = 1;
    float scale = 0.0f;  // 0 = 1/sqrt(head_dim)
};

std::string describe(const AttnCase& c) {
    return "heads " + std::to_string(c.n_head) + "/" + std::to_string(c.n_kv) + " hd " + std::to_string(c.hd) + " bt " +
           std::to_string(c.bt) + " T " + std::to_string(c.T) + " q_offset " + std::to_string(c.q_offset) + " qhs " +
           std::to_string(c.q_head_stride) + " wg " + std::to_string(c.wg);
}

/// Runs the Vulkan attention and cpu::attention_gqa on the same pool and q; checks the bound.
void check_attention(const std::shared_ptr<hv::Context>& ctx, const AttnCase& c) {
    const std::string what = describe(c);
    SCOPED_TRACE(what);
    std::mt19937 rng(c.seed);
    const std::size_t kv_dim = std::size_t{c.n_kv} * c.hd;
    const std::size_t S = c.q_offset + c.T;
    PoolFixture f(kv_dim, c.bt, S, S, rng);
    ASSERT_NE(S % c.bt, 0u) << "cases are chosen with a partially filled last block";

    const std::size_t hs = c.q_head_stride == 0 ? c.hd : c.q_head_stride;
    const std::size_t q_row = std::size_t{c.n_head} * hs;
    const std::vector<float> qbuf = uniform(rng, c.T * q_row, -1.0f, 1.0f);
    const std::size_t qcols = std::size_t{c.n_head} * c.hd;
    std::vector<float> qd(c.T * qcols);  // the same q values, dense, for the CPU op
    for (std::size_t t = 0; t < c.T; ++t)
        for (std::size_t h = 0; h < c.n_head; ++h) std::copy_n(&qbuf[t * q_row + h * hs], c.hd, &qd[t * qcols + h * c.hd]);
    const float scale = c.scale != 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.hd));
    std::vector<const float*> kt, vt;
    const hc::PagedRows kp = f.seq->keys(k_layer, S, kt);
    const hc::PagedRows vp = f.seq->values(k_layer, S, vt);
    std::vector<float> ref(c.T * qcols);
    hc::attention_gqa(hc::AttentionDims{c.n_head, c.n_kv, c.hd}, hc::ConstRows(qd.data(), c.T, qcols, qcols), kp, vp,
                      c.q_offset, scale, hc::Rows(ref.data(), c.T, qcols, qcols));

    hv::OpsOptions o;
    o.attention_workgroup = c.wg;
    hv::Ops ops(ctx, o);
    std::vector<std::uint32_t> table = f.table();
    if (c.bad_block) table[table.size() / 2] = static_cast<std::uint32_t>(f.pool->total_blocks());
    const std::vector<float> poolv = f.pool_bytes();
    // Every operand at a lead that is not a multiple of 16 (see Placed).
    const Placed bq = place(ctx, std::span<const float>(qbuf), 4, q_row * 4);
    const Placed bp = place(ctx, std::span<const float>(poolv), 12);
    const Placed bt = place(ctx, std::span<const std::uint32_t>(table), 8);
    const std::vector<float> init(ref.size(), 999.0f);
    const Placed bo = place(ctx, std::span<const float>(init), 4);
    const Placed bs = place_status(ctx);
    hv::AttentionArgs a;
    a.q = bq.view();
    a.q_head_stride = c.q_head_stride;
    a.kv_pool = bp.view();
    a.n_pool_blocks = static_cast<std::uint32_t>(f.pool->total_blocks());
    a.n_layers = k_layers;
    a.layer = k_layer;
    a.block_tokens = c.bt;
    a.block_table = bt.view();
    a.n_head = c.n_head;
    a.n_kv_head = c.n_kv;
    a.head_dim = c.hd;
    a.n_tokens = c.T;
    a.q_offset = c.q_offset;
    a.scale = scale;
    a.out = bo.view();
    a.status = bs.view();
    hv::Stream s(ctx);
    ops.attention(s, a);
    s.submit_and_wait();
    const std::uint32_t status = hv::read_status(bs.buf, bs.lead);
    if (c.bad_block) {
        EXPECT_EQ(status, hv::k_status_bad_block);
        EXPECT_THROW(hv::check_status(status, "attention"), halo::Error);
        return;
    }
    EXPECT_EQ(status, 0u);
    const std::vector<float> out = bo.read<float>(ref.size());

    const std::size_t group = c.n_head / c.n_kv;
    std::size_t bad = 0;
    double worst = 0.0, worst_abs = 0.0;
    for (std::size_t t = 0; t < c.T; ++t) {
        const std::size_t n = c.q_offset + t + 1;
        const double n_tiles = std::ceil(double(n) / c.wg);
        for (std::size_t h = 0; h < c.n_head; ++h) {
            const std::size_t kvh = h / group;
            const float* qh = &qd[t * qcols + h * c.hd];
            std::vector<double> x(n);
            double m = -1e300, dmax = 0.0;
            for (std::size_t sidx = 0; sidx < n; ++sidx) {
                const float* kr = kp.row_ptr(sidx) + kvh * c.hd;
                double dot = 0.0, mag = 0.0;
                for (std::size_t d = 0; d < c.hd; ++d) {
                    dot += double(qh[d]) * double(kr[d]);
                    mag += std::fabs(double(qh[d]) * double(kr[d]));
                }
                x[sidx] = dot * double(scale);
                m = std::max(m, x[sidx]);
                dmax = std::max(dmax, mag * std::fabs(double(scale)));
            }
            const double D = (double(c.hd) / 8.0 + 4.0) * dmax;
            double R = 0.0, l = 0.0;
            for (std::size_t sidx = 0; sidx < n; ++sidx) {
                R = std::max(R, m - x[sidx]);
                l += std::exp(x[sidx] - m);
            }
            for (std::size_t d = 0; d < c.hd; ++d) {
                double wv = 0.0;
                for (std::size_t sidx = 0; sidx < n; ++sidx)
                    wv += std::exp(x[sidx] - m) / l * std::fabs(double(vp.row_ptr(sidx)[kvh * c.hd + d]));
                const double bound = k_u * (4.0 * double(n) + 8.0 * n_tiles + 8.0 * R + 24.0 + 4.0 * D) * wv + k_floor;
                const std::size_t i = t * qcols + h * c.hd + d;
                const double diff = std::fabs(double(out[i]) - double(ref[i]));
                worst = std::max(worst, std::isfinite(double(out[i])) ? diff / bound : 1e300);
                worst_abs = std::max(worst_abs, diff);
                if (!(diff <= bound) && bad++ == 0) {
                    ADD_FAILURE() << "t " << t << " h " << h << " d " << d << ": cpu " << ref[i] << " vk " << out[i]
                                  << " |d| " << diff << " bound " << bound;
                }
            }
        }
    }
    EXPECT_EQ(bad, 0u);
    std::cout << "[vk-kv] attention " << what << ": worst |vk-cpu|/bound = " << worst << ", worst |vk-cpu| = "
              << worst_abs << "\n";
}

}  // namespace

TEST(VkKv, PoolLayoutIsTheDocumentedOne) {
    // ops.h "The paged KV pool": block[layer][K|V][token][kv_dim], blocks back to back.
    const std::size_t kv_dim = 12, bt = 4;
    kv::KvPool pool(kv::KvLayout{3, kv_dim, bt}, 5);
    const float* base = pool.k_rows(0, 0);
    const std::size_t block_floats = 3 * 2 * bt * kv_dim;
    EXPECT_EQ(pool.layout().block_floats(), block_floats);
    for (kv::BlockId b = 0; b < 5; ++b)
        for (std::size_t L = 0; L < 3; ++L) {
            EXPECT_EQ(pool.k_rows(b, L) - base, std::ptrdiff_t(b * block_floats + L * 2 * bt * kv_dim));
            EXPECT_EQ(pool.v_rows(b, L) - pool.k_rows(b, L), std::ptrdiff_t(bt * kv_dim));
        }
}

TEST(VkKv, KvWriteMatchesSequenceKvWriteBytewise) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    struct Case {
        std::size_t kv_dim, bt, S, start, T;
    };
    // Decode (one row into a partially filled block), MTP verify (rows crossing a block
    // boundary), prefill-like (many blocks, fragmented table), qwen35 kv_dim 1024.
    for (const Case c : {Case{1024, 16, 41, 40, 1}, Case{1024, 16, 37, 30, 7}, Case{96, 4, 45, 3, 42}}) {
        SCOPED_TRACE("kv_dim " + std::to_string(c.kv_dim) + " bt " + std::to_string(c.bt) + " start " +
                     std::to_string(c.start) + " T " + std::to_string(c.T));
        std::mt19937 rng(static_cast<unsigned>(c.kv_dim + c.S));
        PoolFixture f(c.kv_dim, c.bt, c.S, c.start, rng);
        const std::vector<float> before = f.pool_bytes();
        const std::vector<float> k = uniform(rng, c.T * c.kv_dim, -3.0f, 3.0f);
        const std::vector<float> v = uniform(rng, c.T * c.kv_dim, -1.0f, 1.0f);
        for (std::size_t t = 0; t < c.T; ++t) {
            f.seq->write(k_layer, c.start + t, std::span(k).subspan(t * c.kv_dim, c.kv_dim),
                         std::span(v).subspan(t * c.kv_dim, c.kv_dim));
        }
        const std::vector<float> want = f.pool_bytes();
        ASSERT_FALSE(bitwise(before, want));
        hv::Ops ops(ctx);
        const Placed bp = place(ctx, std::span<const float>(before), 12);  // leads: see Placed
        const std::vector<std::uint32_t> table = f.table();
        const Placed bt = place(ctx, std::span<const std::uint32_t>(table), 4);
        const Placed bk = place(ctx, std::span<const float>(k), 8), bv = place(ctx, std::span<const float>(v), 12);
        const Placed bs = place_status(ctx);
        hv::KvWriteArgs a;
        a.kv_pool = bp.view();
        a.n_pool_blocks = static_cast<std::uint32_t>(f.pool->total_blocks());
        a.n_layers = k_layers;
        a.layer = k_layer;
        a.block_tokens = static_cast<std::uint32_t>(c.bt);
        a.kv_dim = static_cast<std::uint32_t>(c.kv_dim);
        a.block_table = bt.view();
        a.start = static_cast<std::uint32_t>(c.start);
        a.n_tokens = static_cast<std::uint32_t>(c.T);
        a.k = bk.view();
        a.v = bv.view();
        a.status = bs.view();
        hv::Stream s(ctx);
        ops.kv_write(s, a);
        s.submit_and_wait();
        EXPECT_EQ(hv::read_status(bs.buf, bs.lead), 0u);
        EXPECT_TRUE(bitwise(bp.read<float>(want.size()), want)) << "pool differs from SequenceKv::write";
    }
}

TEST(VkKv, KvWriteBadBlockSetsStatusAndSkipsOnlyThoseRows) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::size_t kv_dim = 64, bt = 4, S = 13, start = 2, T = 9;  // rows 2..10 span blocks 0..2
    std::mt19937 rng(5);
    PoolFixture f(kv_dim, bt, S, start, rng);
    const std::vector<float> before = f.pool_bytes();
    const std::vector<float> k = uniform(rng, T * kv_dim, -3.0f, 3.0f), v = uniform(rng, T * kv_dim, -1.0f, 1.0f);
    // Expected: every row except those in block 1 (rows 4..7) written.
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t row = start + t;
        if (row / bt == 1) continue;
        f.seq->write(k_layer, row, std::span(k).subspan(t * kv_dim, kv_dim), std::span(v).subspan(t * kv_dim, kv_dim));
    }
    const std::vector<float> want = f.pool_bytes();
    std::vector<std::uint32_t> table = f.table();
    table[1] = static_cast<std::uint32_t>(f.pool->total_blocks());  // the boundary value
    hv::Ops ops(ctx);
    const Placed bp = place(ctx, std::span<const float>(before), 12);
    const Placed bt_ = place(ctx, std::span<const std::uint32_t>(table), 4);
    const Placed bk = place(ctx, std::span<const float>(k), 8), bv = place(ctx, std::span<const float>(v), 12);
    const Placed bs = place_status(ctx);
    hv::KvWriteArgs a;
    a.kv_pool = bp.view();
    a.n_pool_blocks = static_cast<std::uint32_t>(f.pool->total_blocks());
    a.n_layers = k_layers;
    a.layer = k_layer;
    a.block_tokens = bt;
    a.kv_dim = kv_dim;
    a.block_table = bt_.view();
    a.start = start;
    a.n_tokens = T;
    a.k = bk.view();
    a.v = bv.view();
    a.status = bs.view();
    hv::Stream s(ctx);
    ops.kv_write(s, a);
    s.submit_and_wait();
    const std::uint32_t status = hv::read_status(bs.buf, bs.lead);
    EXPECT_EQ(status, hv::k_status_bad_block);
    EXPECT_TRUE(bitwise(bp.read<float>(want.size()), want));
    try {
        hv::check_status(status, "kv_write");
        ADD_FAILURE() << "bad block accepted";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Kernel);
        EXPECT_NE(std::string(e.what()).find("bad-block"), std::string::npos) << e.what();
    }
}

TEST(VkKv, AttentionMatchesCpuQwen35Shapes) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    check_attention(ctx, {.T = 1, .q_offset = 40, .seed = 801});                          // decode, 41 rows
    check_attention(ctx, {.T = 4, .q_offset = 37, .q_head_stride = 512, .seed = 802});    // verify, [Q|gate]
    check_attention(ctx, {.T = 3, .q_offset = 300, .seed = 803});                         // 3 tiles, partial last
}

TEST(VkKv, AttentionWorkgroupSizesAndOddShapes) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    // hd 256 at the smallest workgroup that holds it (4 dims per thread) and at 256.
    check_attention(ctx, {.T = 2, .q_offset = 150, .wg = 64, .seed = 811});
    check_attention(ctx, {.T = 2, .q_offset = 150, .q_head_stride = 512, .wg = 256, .seed = 812});
    // Tiny-model dims (hd 64, 4/2 heads, bt 4): a 33-token prefill, many tiles of 32.
    check_attention(ctx, {.n_head = 4, .n_kv = 2, .hd = 64, .bt = 4, .T = 33, .q_offset = 0, .wg = 32, .seed = 813});
    // Odd sizes: the 8-lane dot tail (hd 20), 3 query heads per KV head, bt 5.
    check_attention(ctx, {.n_head = 6, .n_kv = 2, .hd = 20, .bt = 5, .T = 3, .q_offset = 0, .wg = 32, .seed = 814});
    // Large scores (scale 8: score range ~100s): exp(x - m) overflows fp32 unless m is the
    // true running maximum, so this is the case that pins the tile-max reduction.
    check_attention(ctx, {.n_head = 4, .n_kv = 2, .hd = 64, .bt = 4, .T = 2, .q_offset = 149, .wg = 32, .seed = 816,
                          .scale = 8.0f});
    // Long history: 1001 keys = 8 tiles of 128.
    check_attention(ctx, {.n_head = 8, .n_kv = 2, .hd = 64, .bt = 16, .T = 1, .q_offset = 1000, .seed = 815});
}

TEST(VkKv, AttentionBadBlockSetsStatus) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    check_attention(ctx, {.T = 2, .q_offset = 40, .bad_block = true, .seed = 821});
}

TEST(VkKv, KvWriteThenAttentionMatchesSequenceKvWriteThenCpuAttention) {
    // The seam: the new rows are written on the device and attended in the same recording, vs
    // SequenceKv::write + cpu::attention_gqa (the CPU forward's order).
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint32_t n_head = 24, n_kv = 4, hd = 256, bt = 16, q_offset = 45, T = 4;
    const std::size_t kv_dim = std::size_t{n_kv} * hd, S = q_offset + T, qcols = std::size_t{n_head} * hd;
    std::mt19937 rng(900);
    PoolFixture f(kv_dim, bt, S, q_offset, rng);
    const std::vector<float> before = f.pool_bytes();
    const std::vector<float> k = uniform(rng, T * kv_dim, -2.0f, 2.0f), v = uniform(rng, T * kv_dim, -1.0f, 1.0f);
    const std::vector<float> q = uniform(rng, T * qcols, -1.0f, 1.0f);
    for (std::size_t t = 0; t < T; ++t)
        f.seq->write(k_layer, q_offset + t, std::span(k).subspan(t * kv_dim, kv_dim),
                     std::span(v).subspan(t * kv_dim, kv_dim));
    const float scale = 1.0f / 16.0f;
    std::vector<const float*> kt, vt;
    const hc::PagedRows kp = f.seq->keys(k_layer, S, kt), vp = f.seq->values(k_layer, S, vt);
    std::vector<float> ref(T * qcols);
    hc::attention_gqa(hc::AttentionDims{n_head, n_kv, hd}, hc::ConstRows(q.data(), T, qcols, qcols), kp, vp, q_offset,
                      scale, hc::Rows(ref.data(), T, qcols, qcols));

    hv::Ops ops(ctx);  // operands at offset 0 here (the zero-remainder path; the other tests use leads)
    hv::Buffer bp = upload(ctx, std::span<const float>(before));
    const std::vector<std::uint32_t> table = f.table();
    hv::Buffer btab = upload(ctx, std::span<const std::uint32_t>(table));
    hv::Buffer bk = upload(ctx, std::span<const float>(k)), bv = upload(ctx, std::span<const float>(v));
    hv::Buffer bq = upload(ctx, std::span<const float>(q));
    hv::Buffer bo = hv::Buffer::create(ctx, ref.size() * 4, hv::MemoryUsage::HostCached);
    hv::Buffer bs = hv::Buffer::create(ctx, 8, hv::MemoryUsage::HostCached);  // two status words
    hv::KvWriteArgs w;
    w.kv_pool = bp;
    w.n_pool_blocks = static_cast<std::uint32_t>(f.pool->total_blocks());
    w.n_layers = k_layers;
    w.layer = k_layer;
    w.block_tokens = bt;
    w.kv_dim = static_cast<std::uint32_t>(kv_dim);
    w.block_table = btab;
    w.start = q_offset;
    w.n_tokens = T;
    w.k = bk;
    w.v = bv;
    w.status = hv::BufferView(bs, 0, 4);
    hv::AttentionArgs a;
    a.q = bq;
    a.kv_pool = bp;
    a.n_pool_blocks = w.n_pool_blocks;
    a.n_layers = k_layers;
    a.layer = k_layer;
    a.block_tokens = bt;
    a.block_table = btab;
    a.n_head = n_head;
    a.n_kv_head = n_kv;
    a.head_dim = hd;
    a.n_tokens = T;
    a.q_offset = q_offset;
    a.scale = scale;
    a.out = bo;
    a.status = hv::BufferView(bs, 4, 4);
    hv::Stream s(ctx);
    ops.kv_write(s, w);
    ops.attention(s, a);
    s.submit_and_wait();
    EXPECT_EQ(hv::read_status(bs, 0), 0u);
    EXPECT_EQ(hv::read_status(bs, 4), 0u);
    const std::vector<float> out = download<float>(bo, ref.size());
    // Same bound as check_attention; the new rows' K/V are the CPU pool's bytes (KV write is a
    // bit-exact copy, tested above), so the bound's inputs are the CPU-side values.
    const std::size_t group = n_head / n_kv;
    double worst = 0.0;
    std::size_t bad = 0;
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t n = q_offset + t + 1;
        const double n_tiles = std::ceil(double(n) / ops.options().attention_workgroup);
        for (std::size_t h = 0; h < n_head; ++h) {
            const std::size_t kvh = h / group;
            std::vector<double> x(n);
            double m = -1e300, dmax = 0.0;
            for (std::size_t sidx = 0; sidx < n; ++sidx) {
                double dot = 0.0, mag = 0.0;
                for (std::size_t d = 0; d < hd; ++d) {
                    const double p = double(q[t * qcols + h * hd + d]) * double(kp.row_ptr(sidx)[kvh * hd + d]);
                    dot += p;
                    mag += std::fabs(p);
                }
                x[sidx] = dot * double(scale);
                m = std::max(m, x[sidx]);
                dmax = std::max(dmax, mag * double(scale));
            }
            double R = 0.0, l = 0.0;
            for (const double xs : x) {
                R = std::max(R, m - xs);
                l += std::exp(xs - m);
            }
            const double D = (hd / 8.0 + 4.0) * dmax;
            for (std::size_t d = 0; d < hd; ++d) {
                double wv = 0.0;
                for (std::size_t sidx = 0; sidx < n; ++sidx)
                    wv += std::exp(x[sidx] - m) / l * std::fabs(double(vp.row_ptr(sidx)[kvh * hd + d]));
                const double bound = k_u * (4.0 * double(n) + 8.0 * n_tiles + 8.0 * R + 24.0 + 4.0 * D) * wv + k_floor;
                const std::size_t i = t * qcols + h * hd + d;
                const double diff = std::fabs(double(out[i]) - double(ref[i]));
                worst = std::max(worst, std::isfinite(double(out[i])) ? diff / bound : 1e300);
                if (!(diff <= bound)) ++bad;
            }
        }
    }
    EXPECT_EQ(bad, 0u);
    std::cout << "[vk-kv] kv_write -> attention seam: worst |vk-cpu|/bound = " << worst << "\n";
}

TEST(VkKv, Validation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t hd = 64, n_kv = 2, bt = 4, blocks = 3;
    const std::size_t block_floats = std::size_t{k_layers} * 2 * bt * n_kv * hd;
    hv::Buffer pool = hv::Buffer::create(ctx, blocks * block_floats * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer tab = hv::Buffer::create(ctx, 3 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer q = hv::Buffer::create(ctx, 2 * 4 * hd * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer out = hv::Buffer::create(ctx, 2 * 4 * hd * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer st = hv::Buffer::create(ctx, 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    auto base = [&] {
        hv::AttentionArgs a;
        a.q = q;
        a.kv_pool = pool;
        a.n_pool_blocks = blocks;
        a.n_layers = k_layers;
        a.layer = k_layer;
        a.block_tokens = bt;
        a.block_table = tab;
        a.n_head = 4;
        a.n_kv_head = n_kv;
        a.head_dim = hd;
        a.n_tokens = 2;
        a.q_offset = 9;  // 11 rows -> 3 table entries
        a.scale = 0.125f;
        a.out = out;
        a.status = st;
        return a;
    };
    auto expect_code = [&](const hv::AttentionArgs& a, halo::ErrorCode code, const char* what) {
        try {
            ops.attention(s, a);
            ADD_FAILURE() << what << ": accepted";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), code) << what << ": " << e.what();
        }
    };
    EXPECT_NO_THROW(ops.attention(s, base()));
    auto a = base();
    a.q_offset = 10;  // 12 rows: still 3 table entries
    EXPECT_NO_THROW(ops.attention(s, a));
    a.q_offset = 11;  // 13 rows need a 4th entry
    expect_code(a, halo::ErrorCode::Kernel, "table too short");
    a = base();
    a.layer = 2;
    expect_code(a, halo::ErrorCode::Kernel, "layer out of range");
    a = base();
    a.n_head = 3;
    expect_code(a, halo::ErrorCode::Kernel, "n_head not a multiple of n_kv_head");
    a = base();
    a.out = q;
    expect_code(a, halo::ErrorCode::Kernel, "out overlaps q");
    a = base();
    a.status = hv::BufferView(pool, 0, 4);
    expect_code(a, halo::ErrorCode::Kernel, "status inside the pool");
    a = base();
    a.n_pool_blocks = blocks + 1;
    expect_code(a, halo::ErrorCode::Kernel, "pool buffer smaller than n_pool_blocks");
    a = base();
    a.n_pool_blocks = 1u << 30;  // ~2 TiB of blocks: beyond one descriptor
    expect_code(a, halo::ErrorCode::Unsupported, "pool beyond one descriptor");
    a = base();
    a.head_dim = 512;
    a.n_head = 1;
    a.n_kv_head = 1;
    expect_code(a, halo::ErrorCode::Unsupported, "head_dim 512");
    {
        hv::OpsOptions o;
        o.attention_workgroup = 32;
        hv::Ops small(ctx, o);
        hv::AttentionArgs b = base();
        b.head_dim = 256;
        b.n_head = 1;
        b.n_kv_head = 1;
        try {
            small.attention(s, b);
            ADD_FAILURE() << "head_dim 256 with workgroup 32 accepted";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported) << e.what();
        }
    }
    hv::OpsOptions bad_wg;
    bad_wg.attention_workgroup = 96;
    EXPECT_THROW(hv::Ops(ctx, bad_wg), halo::Error);
    // KV write: k overlapping the pool, a row past the table.
    hv::KvWriteArgs w;
    w.kv_pool = pool;
    w.n_pool_blocks = blocks;
    w.n_layers = k_layers;
    w.layer = k_layer;
    w.block_tokens = bt;
    w.kv_dim = n_kv * hd;
    w.block_table = tab;
    w.start = 10;
    w.n_tokens = 2;
    w.k = q;
    w.v = hv::BufferView(q, 0, 2 * n_kv * hd * 4);
    w.status = st;
    EXPECT_NO_THROW(ops.kv_write(s, w));
    hv::KvWriteArgs w2 = w;
    w2.k = hv::BufferView(pool, 0, 2 * n_kv * hd * 4);
    EXPECT_THROW(ops.kv_write(s, w2), halo::Error);
    w2 = w;
    w2.start = 11;  // row 12 needs a 4th table entry
    EXPECT_THROW(ops.kv_write(s, w2), halo::Error);
    s.submit_and_wait();
}
