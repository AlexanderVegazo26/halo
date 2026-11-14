// KV write and paged GQA ATTENTION of the Vulkan backend (WS-F2 V2), differentially tested
// against halo::kv_cache (SequenceKv::write) and cpu::attention_gqa on the same inputs
// (TRD §30; code review M-2). Verified on lavapipe (DECISIONS D-001) and RADV (2026-11-14).
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
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "halo/kv_cache/paged_kv.h"
#include "halo/tensor/fp16.h"
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

// ---- host reference of the quantized pool formats (ops.h "The paged KV pool") ---------------
// F16: two halves per word, element 2p in the low half, round to nearest even
// (tensor::fp32_to_fp16). Q8: each 32-element group is [fp32 scale = max|x| / 127 | 32 x int8,
// 4 per word, element 4w+b in byte b], q = clamp(roundEven(x * (127 / max|x|)), -127, 127), all
// zero when max|x| = 0. Decoding is exact: half -> float widening, float(q) * scale (one fp32
// multiply), which is what attention_kv_main.glsl reads.

const char* kv_name(hv::KvType t) { return t == hv::KvType::F16 ? "f16" : t == hv::KvType::Q8 ? "q8" : "f32"; }

std::size_t row_words(hv::KvType t, std::size_t kv_dim) {
    return t == hv::KvType::F16 ? kv_dim / 2 : t == hv::KvType::Q8 ? kv_dim / 32 * 9 : kv_dim;
}

void encode_row(hv::KvType t, std::span<const float> x, std::uint32_t* out) {
    if (t == hv::KvType::F32) {
        std::memcpy(out, x.data(), x.size_bytes());
    } else if (t == hv::KvType::F16) {
        for (std::size_t p = 0; p < x.size() / 2; ++p)
            out[p] = std::uint32_t{halo::tensor::fp32_to_fp16(x[2 * p])} |
                     (std::uint32_t{halo::tensor::fp32_to_fp16(x[2 * p + 1])} << 16);
    } else {
        for (std::size_t g = 0; g < x.size() / 32; ++g) {
            const float* s = x.data() + g * 32;
            float amax = 0.0f;
            for (std::size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(s[j]));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            std::uint32_t* w = out + g * 9;
            w[0] = std::bit_cast<std::uint32_t>(amax / 127.0f);
            for (std::size_t j = 0; j < 32; ++j) {
                if (j % 4 == 0) w[1 + j / 4] = 0;
                const int q = std::clamp(static_cast<int>(std::nearbyint(s[j] * inv)), -127, 127);
                w[1 + j / 4] |= (static_cast<std::uint32_t>(q) & 0xFFu) << (8 * (j % 4));
            }
        }
    }
}

void decode_row(hv::KvType t, const std::uint32_t* in, std::span<float> x) {
    if (t == hv::KvType::F32) {
        std::memcpy(x.data(), in, x.size_bytes());
    } else if (t == hv::KvType::F16) {
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = halo::tensor::fp16_to_fp32(static_cast<std::uint16_t>(in[i / 2] >> (16 * (i % 2))));
    } else {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const std::uint32_t* w = in + (i / 32) * 9;
            const auto q = static_cast<std::int8_t>((w[1 + (i % 32) / 4] >> (8 * (i % 4))) & 0xFFu);
            x[i] = static_cast<float>(q) * std::bit_cast<float>(w[0]);
        }
    }
}

/// The whole pool image (rows of kv_dim floats, back to back) in format t.
std::vector<std::uint32_t> encode_pool(hv::KvType t, std::span<const float> pool, std::size_t kv_dim) {
    const std::size_t rows = pool.size() / kv_dim, rw = row_words(t, kv_dim);
    std::vector<std::uint32_t> out(rows * rw);
    for (std::size_t r = 0; r < rows; ++r) encode_row(t, pool.subspan(r * kv_dim, kv_dim), out.data() + r * rw);
    return out;
}

/// Overwrites seq's rows [0, S) of `layer` in the host pool with the values the device reads
/// from `image` (the fp32 pool's layout in format t), so the CPU reference attends exactly those.
void load_decoded(PoolFixture& f, hv::KvType t, std::span<const std::uint32_t> image, std::size_t layer, std::size_t S) {
    const std::size_t rw = row_words(t, f.kv_dim);
    const std::vector<std::uint32_t> table = f.table();
    for (std::size_t s = 0; s < S; ++s) {
        const kv::BlockId id = table[s / f.bt];
        const std::size_t krow = ((std::size_t{id} * k_layers + layer) * 2) * f.bt + s % f.bt;
        decode_row(t, image.data() + krow * rw, std::span(f.pool->k_rows(id, layer) + (s % f.bt) * f.kv_dim, f.kv_dim));
        decode_row(t, image.data() + (krow + f.bt) * rw,
                   std::span(f.pool->v_rows(id, layer) + (s % f.bt) * f.kv_dim, f.kv_dim));
    }
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
    hv::KvType kv = hv::KvType::F32;  // pool format (HALO_KV_FP16 / HALO_KV_TYPE)
};

std::string describe(const AttnCase& c) {
    return std::string("kv ") + kv_name(c.kv) + " heads " + std::to_string(c.n_head) + "/" + std::to_string(c.n_kv) +
           " hd " + std::to_string(c.hd) + " bt " + std::to_string(c.bt) + " T " + std::to_string(c.T) + " q_offset " +
           std::to_string(c.q_offset) + " qhs " + std::to_string(c.q_head_stride) + " wg " + std::to_string(c.wg);
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
    // The device pool in the case's format; the CPU then attends the values the device decodes
    // from it (the only fp16/q8 error is kv_write's storage rounding, pinned by the KvWrite tests).
    const std::vector<std::uint32_t> image = encode_pool(c.kv, f.pool_bytes(), kv_dim);
    if (c.kv != hv::KvType::F32) load_decoded(f, c.kv, image, k_layer, S);
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
    // Every operand at a lead that is not a multiple of 16 (see Placed).
    const Placed bq = place(ctx, std::span<const float>(qbuf), 4, q_row * 4);
    const Placed bp = place(ctx, std::span<const std::uint32_t>(image), 12);
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
    a.kv_type = c.kv;
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

// ---- fp16 / q8 KV pools (opt-in HALO_KV_FP16=1 / HALO_KV_TYPE=q8; next.md step 9) ----------
// kv_write_f16 / kv_write_q8 against the host encoders above, attention_f16 / attention_q8
// against cpu::attention_gqa on the decoded values (the fp32 bound: decoding is exact).

namespace {

/// Runs kv_write in format t over a pool image `before` (encode_pool of f's fp32 pool) and
/// returns the device image afterwards; status must stay 0.
std::vector<std::uint32_t> device_kv_write(const std::shared_ptr<hv::Context>& ctx, const PoolFixture& f,
                                           hv::KvType t, std::span<const std::uint32_t> before,
                                           std::span<const float> k, std::span<const float> v, std::size_t start,
                                           std::size_t T) {
    hv::Ops ops(ctx);
    const Placed bp = place(ctx, before, 12);
    const std::vector<std::uint32_t> table = f.table();
    const Placed bt = place(ctx, std::span<const std::uint32_t>(table), 4);
    const Placed bk = place(ctx, k, 8), bv = place(ctx, v, 12);
    const Placed bs = place_status(ctx);
    hv::KvWriteArgs a;
    a.kv_pool = bp.view();
    a.n_pool_blocks = static_cast<std::uint32_t>(f.pool->total_blocks());
    a.n_layers = k_layers;
    a.layer = k_layer;
    a.block_tokens = static_cast<std::uint32_t>(f.bt);
    a.kv_dim = static_cast<std::uint32_t>(f.kv_dim);
    a.block_table = bt.view();
    a.start = static_cast<std::uint32_t>(start);
    a.n_tokens = static_cast<std::uint32_t>(T);
    a.k = bk.view();
    a.v = bv.view();
    a.status = bs.view();
    a.kv_type = t;
    hv::Stream s(ctx);
    ops.kv_write(s, a);
    s.submit_and_wait();
    EXPECT_EQ(hv::read_status(bs.buf, bs.lead), 0u) << kv_name(t) << " kv_write status";
    return bp.read<std::uint32_t>(before.size());
}

/// Pool-image row index (rows of row_words) of history row s, K (kv 0) or V (kv 1), k_layer.
std::size_t image_row(const PoolFixture& f, std::size_t s, std::size_t kv) {
    return ((std::size_t{f.table()[s / f.bt]} * k_layers + k_layer) * 2 + kv) * f.bt + s % f.bt;
}

/// K and V rows of a kv_write case: uniform values, with special rows for format edges.
struct WriteRows {
    std::vector<float> k, v;
};

WriteRows write_rows(std::mt19937& rng, std::size_t T, std::size_t kv_dim) {
    WriteRows w{uniform(rng, T * kv_dim, -3.0f, 3.0f), uniform(rng, T * kv_dim, -1.0f, 1.0f)};
    // fp16 rounding edges in the first K row: the largest half, the value just under the
    // overflow midpoint (-> 65504) and the midpoint itself (-> inf under round-to-nearest-even,
    // 65504 under round-toward-zero), two ties (1 + 2^-11 -> 1, 1 + 3 * 2^-11 -> 1 + 2^-9),
    // half subnormals (exact 2^-20; 0.75 * 2^-24 -> 2^-24; the tie 2.5 * 2^-24 -> 2 * 2^-24), a
    // value far past the half range (-> inf), a NaN (-> 0x7E00), -0 and a negative value.
    const float edges[] = {65504.0f,   65519.0f,   65520.0f, 1.0f + 0x1p-11f, 1.0f + 3 * 0x1p-11f, 0x1p-20f,
                           0x1.8p-25f, 0x1.4p-23f, 1.0e5f,   std::numeric_limits<float>::quiet_NaN(),
                           -0.0f,      -2.5f};
    std::copy(std::begin(edges), std::end(edges), w.k.begin());
    // q8 group edges in the first V row (kv_dim >= 96): an all-zero group, a group whose only
    // nonzero is its negative maximum, a group of one repeated value.
    if (kv_dim >= 96) {
        std::fill_n(w.v.begin(), 32, 0.0f);
        std::fill_n(w.v.begin() + 32, 32, 0.0f);
        w.v[45] = -0.75f;
        std::fill_n(w.v.begin() + 64, 32, 0.3f);
    }
    return w;
}

struct WriteCase {
    std::size_t kv_dim, bt, S, start, T;
};
// The fp32 test's cases: decode into a partly filled block, MTP verify across a block boundary
// (qwen35 kv_dim 1024), prefill-like over a fragmented table.
constexpr WriteCase k_write_cases[] = {{1024, 16, 41, 40, 1}, {1024, 16, 37, 30, 7}, {96, 4, 45, 3, 42}};

}  // namespace

TEST(VkKv, KvWriteF16StoresRoundToNearestEvenHalves) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const WriteCase c : k_write_cases) {
        SCOPED_TRACE("kv_dim " + std::to_string(c.kv_dim) + " bt " + std::to_string(c.bt) + " start " +
                     std::to_string(c.start) + " T " + std::to_string(c.T));
        std::mt19937 rng(static_cast<unsigned>(c.kv_dim + c.S + 7));
        PoolFixture f(c.kv_dim, c.bt, c.S, c.start, rng);
        const std::vector<std::uint32_t> before = encode_pool(hv::KvType::F16, f.pool_bytes(), c.kv_dim);
        const WriteRows w = write_rows(rng, c.T, c.kv_dim);
        for (std::size_t t = 0; t < c.T; ++t)
            f.seq->write(k_layer, c.start + t, std::span(w.k).subspan(t * c.kv_dim, c.kv_dim),
                         std::span(w.v).subspan(t * c.kv_dim, c.kv_dim));
        const std::vector<std::uint32_t> want = encode_pool(hv::KvType::F16, f.pool_bytes(), c.kv_dim);
        ASSERT_NE(before, want);
        const std::vector<std::uint32_t> got = device_kv_write(ctx, f, hv::KvType::F16, before, w.k, w.v, c.start, c.T);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < want.size(); ++i) {
            if (got[i] != want[i] && bad++ < 8) {
                const std::size_t row = i / (c.kv_dim / 2), col = 2 * (i % (c.kv_dim / 2));
                ADD_FAILURE() << "pool word " << i << " (image row " << row << ", elements " << col << ".." << col + 1
                              << "): device 0x" << std::hex << got[i] << ", round-to-nearest-even halves 0x" << want[i]
                              << std::dec;
            }
        }
        EXPECT_EQ(bad, 0u) << "fp16 pool words differing from tensor::fp32_to_fp16 of SequenceKv::write's rows";
    }
}

TEST(VkKv, KvWriteQ8StoresMaxScaleAndNearestInt8PerGroup) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const WriteCase c : k_write_cases) {
        SCOPED_TRACE("kv_dim " + std::to_string(c.kv_dim) + " bt " + std::to_string(c.bt) + " start " +
                     std::to_string(c.start) + " T " + std::to_string(c.T));
        std::mt19937 rng(static_cast<unsigned>(c.kv_dim + c.S + 9));
        PoolFixture f(c.kv_dim, c.bt, c.S, c.start, rng);
        const std::size_t rw = c.kv_dim / 32 * 9;
        const std::vector<std::uint32_t> before = encode_pool(hv::KvType::Q8, f.pool_bytes(), c.kv_dim);
        WriteRows w = write_rows(rng, c.T, c.kv_dim);
        // The fp16 NaN edge has no q8 contract (GLSL max of a NaN is implementation-defined).
        std::replace_if(w.k.begin(), w.k.end(), [](float x) { return std::isnan(x); }, 0.0f);
        const std::vector<std::uint32_t> got = device_kv_write(ctx, f, hv::KvType::Q8, before, w.k, w.v, c.start, c.T);
        // Rows the write must not touch stay byte for byte.
        std::vector<bool> written(before.size() / rw, false);
        for (std::size_t t = 0; t < c.T; ++t)
            for (std::size_t kv = 0; kv < 2; ++kv) written[image_row(f, c.start + t, kv)] = true;
        std::size_t untouched_bad = 0;
        for (std::size_t r = 0; r < written.size(); ++r)
            if (!written[r] && !std::equal(got.begin() + r * rw, got.begin() + (r + 1) * rw, before.begin() + r * rw))
                ++untouched_bad;
        EXPECT_EQ(untouched_bad, 0u) << "pool rows outside the written ones changed";
        // Written groups: scale = max|x| / 127 (GLSL division: within 3 ulp), q the nearest
        // integer of x * 127 / max|x| up to the rounding of the reciprocal (|q - r| <= 0.5 + 1e-4),
        // |q| = 127 at the maximum, all zero for an all-zero group.
        std::size_t bad = 0, groups = 0, host_equal = 0;
        for (std::size_t t = 0; t < c.T; ++t) {
            for (std::size_t kv = 0; kv < 2; ++kv) {
                const float* x = (kv == 0 ? w.k : w.v).data() + t * c.kv_dim;
                const std::uint32_t* row = got.data() + image_row(f, c.start + t, kv) * rw;
                std::vector<std::uint32_t> host(rw);
                encode_row(hv::KvType::Q8, std::span(x, c.kv_dim), host.data());
                for (std::size_t g = 0; g < c.kv_dim / 32; ++g, ++groups) {
                    const std::uint32_t* gw = row + g * 9;
                    host_equal += std::equal(gw, gw + 9, host.data() + g * 9);
                    double amax = 0.0;
                    for (std::size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(double(x[g * 32 + j])));
                    const float want_scale = static_cast<float>(amax) / 127.0f;
                    const float sg = std::bit_cast<float>(gw[0]);
                    const double ulp = double(std::nextafter(want_scale, INFINITY) - want_scale);
                    std::string why;
                    if (!(std::fabs(double(sg) - double(want_scale)) <= 3.0 * ulp))
                        why = "scale " + std::to_string(sg) + " vs max|x|/127 " + std::to_string(want_scale);
                    for (std::size_t j = 0; j < 32 && why.empty(); ++j) {
                        const int q = static_cast<std::int8_t>((gw[1 + j / 4] >> (8 * (j % 4))) & 0xFFu);
                        const double r = amax > 0.0 ? double(x[g * 32 + j]) * 127.0 / amax : 0.0;
                        const bool at_max = amax > 0.0 && std::fabs(double(x[g * 32 + j])) == amax;
                        if (q < -127 || q > 127 || std::fabs(q - r) > 0.5 + 1e-4 || (at_max && std::abs(q) != 127))
                            why = "element " + std::to_string(j) + " x " + std::to_string(x[g * 32 + j]) + " -> q " +
                                  std::to_string(q) + ", x*127/max|x| = " + std::to_string(r);
                    }
                    if (!why.empty() && bad++ < 8)
                        ADD_FAILURE() << "row " << c.start + t << (kv == 0 ? " K" : " V") << " group " << g << ": " << why;
                }
            }
        }
        EXPECT_EQ(bad, 0u) << "of " << groups << " written q8 groups";
        std::cout << "[vk-kv] q8 kv_write kv_dim " << c.kv_dim << ": " << host_equal << "/" << groups
                  << " groups bitwise the host encoder\n";
    }
}

TEST(VkKv, AttentionOverF16AndQ8PoolsMatchesCpuOnTheDecodedValues) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const hv::KvType t : {hv::KvType::F16, hv::KvType::Q8}) {
        check_attention(ctx, {.T = 1, .q_offset = 40, .seed = 831, .kv = t});                        // decode
        check_attention(ctx, {.T = 4, .q_offset = 37, .q_head_stride = 512, .seed = 832, .kv = t});  // verify, [Q|gate]
        check_attention(ctx, {.T = 3, .q_offset = 300, .seed = 833, .kv = t});                       // 3 tiles
        check_attention(ctx, {.T = 2, .q_offset = 150, .wg = 64, .seed = 834, .kv = t});             // 4 dims/thread
        check_attention(ctx, {.n_head = 4, .n_kv = 2, .hd = 64, .bt = 4, .T = 33, .q_offset = 0, .wg = 32, .seed = 835,
                              .kv = t});  // tiny-model prefill
        check_attention(ctx, {.n_head = 4, .n_kv = 2, .hd = 64, .bt = 4, .T = 2, .q_offset = 149, .wg = 32, .seed = 836,
                              .scale = 8.0f, .kv = t});  // large scores: the tile max
        check_attention(ctx, {.n_head = 8, .n_kv = 2, .hd = 64, .bt = 16, .T = 1, .q_offset = 1000, .seed = 837,
                              .kv = t});  // 8 tiles
        check_attention(ctx, {.T = 2, .q_offset = 40, .bad_block = true, .seed = 838, .kv = t});
    }
    // fp16 only needs an even head_dim: the 8-lane dot tail (hd 20), 3 query heads per KV head.
    check_attention(ctx, {.n_head = 6, .n_kv = 2, .hd = 20, .bt = 5, .T = 3, .q_offset = 0, .wg = 32, .seed = 839,
                          .kv = hv::KvType::F16});
}

TEST(VkKv, QuantizedKvWriteThenAttentionMatchesCpuOnTheWrittenValues) {
    // The decode seam in one recording, as qwen35 runs it: the new rows are stored in the pool's
    // format and attended at once; the CPU attends the values decoded from the device's image.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const hv::KvType t : {hv::KvType::F16, hv::KvType::Q8}) {
        SCOPED_TRACE(kv_name(t));
        const std::uint32_t n_head = 24, n_kv = 4, hd = 256, bt = 16, q_offset = 45, T = 4;
        const std::size_t kv_dim = std::size_t{n_kv} * hd, S = q_offset + T, qcols = std::size_t{n_head} * hd;
        std::mt19937 rng(910 + static_cast<unsigned>(t));
        PoolFixture f(kv_dim, bt, S, q_offset, rng);
        const std::vector<std::uint32_t> before = encode_pool(t, f.pool_bytes(), kv_dim);
        const std::vector<float> k = uniform(rng, T * kv_dim, -2.0f, 2.0f), v = uniform(rng, T * kv_dim, -1.0f, 1.0f);
        const std::vector<float> q = uniform(rng, T * qcols, -1.0f, 1.0f);
        const float scale = 1.0f / 16.0f;
        hv::Ops ops(ctx);
        hv::Buffer bp = upload(ctx, std::span<const std::uint32_t>(before));
        const std::vector<std::uint32_t> table = f.table();
        hv::Buffer btab = upload(ctx, std::span<const std::uint32_t>(table));
        hv::Buffer bk = upload(ctx, std::span<const float>(k)), bv = upload(ctx, std::span<const float>(v));
        hv::Buffer bq = upload(ctx, std::span<const float>(q));
        hv::Buffer bo = hv::Buffer::create(ctx, T * qcols * 4, hv::MemoryUsage::HostCached);
        hv::Buffer bs = hv::Buffer::create(ctx, 8, hv::MemoryUsage::HostCached);
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
        w.kv_type = t;
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
        a.kv_type = t;
        hv::Stream s(ctx);
        ops.kv_write(s, w);
        ops.attention(s, a);
        s.submit_and_wait();
        EXPECT_EQ(hv::read_status(bs, 0), 0u);
        EXPECT_EQ(hv::read_status(bs, 4), 0u);
        const std::vector<std::uint32_t> image = download<std::uint32_t>(bp, before.size());
        // The written rows decode to the inputs within the format's storage error.
        std::vector<float> dk(kv_dim), dv(kv_dim);
        std::size_t far = 0;
        const std::size_t rw = row_words(t, kv_dim);
        for (std::size_t r = 0; r < T; ++r) {
            decode_row(t, image.data() + image_row(f, q_offset + r, 0) * rw, dk);
            decode_row(t, image.data() + image_row(f, q_offset + r, 1) * rw, dv);
            for (std::size_t i = 0; i < kv_dim; ++i) {
                // fp16: 2^-11 relative (2^-25 absolute below the half normals); q8: half a step of
                // max|x| / 127 (|k| <= 2, |v| <= 1).
                const auto h16 = [](float x) { return std::max(std::fabs(double(x)) * 0x1p-11, 0x1p-25); };
                const double tk = t == hv::KvType::F16 ? h16(k[r * kv_dim + i]) : 2.0 / 254 * 1.0001;
                const double tv = t == hv::KvType::F16 ? h16(v[r * kv_dim + i]) : 1.0 / 254 * 1.0001;
                far += std::fabs(double(dk[i]) - double(k[r * kv_dim + i])) > tk ||
                       std::fabs(double(dv[i]) - double(v[r * kv_dim + i])) > tv;
            }
        }
        EXPECT_EQ(far, 0u) << "written elements decoding further than the format's storage error from the input";
        load_decoded(f, t, image, k_layer, S);
        std::vector<const float*> kt, vt;
        const hc::PagedRows kp = f.seq->keys(k_layer, S, kt), vp = f.seq->values(k_layer, S, vt);
        std::vector<float> ref(T * qcols);
        hc::attention_gqa(hc::AttentionDims{n_head, n_kv, hd}, hc::ConstRows(q.data(), T, qcols, qcols), kp, vp,
                          q_offset, scale, hc::Rows(ref.data(), T, qcols, qcols));
        const std::vector<float> out = download<float>(bo, ref.size());
        // The fp32 seam's bound (VkKv.KvWriteThenAttention...) on the decoded values.
        const std::size_t group = n_head / n_kv;
        std::size_t bad = 0;
        double worst = 0.0;
        for (std::size_t tt = 0; tt < T; ++tt) {
            const std::size_t n = q_offset + tt + 1;
            const double n_tiles = std::ceil(double(n) / ops.options().attention_workgroup);
            for (std::size_t h = 0; h < n_head; ++h) {
                const std::size_t kvh = h / group;
                std::vector<double> x(n);
                double m = -1e300, dmax = 0.0;
                for (std::size_t sidx = 0; sidx < n; ++sidx) {
                    double dot = 0.0, mag = 0.0;
                    for (std::size_t d = 0; d < hd; ++d) {
                        const double p = double(q[tt * qcols + h * hd + d]) * double(kp.row_ptr(sidx)[kvh * hd + d]);
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
                    const double bound =
                        k_u * (4.0 * double(n) + 8.0 * n_tiles + 8.0 * R + 24.0 + 4.0 * D) * wv + k_floor;
                    const std::size_t i = tt * qcols + h * hd + d;
                    const double diff = std::fabs(double(out[i]) - double(ref[i]));
                    worst = std::max(worst, std::isfinite(double(out[i])) ? diff / bound : 1e300);
                    if (!(diff <= bound) && bad++ == 0)
                        ADD_FAILURE() << "t " << tt << " h " << h << " d " << d << ": cpu " << ref[i] << " vk " << out[i]
                                      << " bound " << bound;
                }
            }
        }
        EXPECT_EQ(bad, 0u);
        std::cout << "[vk-kv] " << kv_name(t) << " kv_write -> attention seam: worst |vk-cpu|/bound = " << worst << "\n";
    }
}

TEST(VkKv, QuantizedPoolsRejectHeadsOffTheirWordOrGroupBoundary) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer pool = hv::Buffer::create(ctx, 1 << 16, hv::MemoryUsage::DeviceLocal);
    hv::Buffer tab = hv::Buffer::create(ctx, 64, hv::MemoryUsage::DeviceLocal);
    hv::Buffer q = hv::Buffer::create(ctx, 1 << 14, hv::MemoryUsage::DeviceLocal);
    hv::Buffer out = hv::Buffer::create(ctx, 1 << 14, hv::MemoryUsage::DeviceLocal);
    hv::Buffer st = hv::Buffer::create(ctx, 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    auto attn = [&](hv::KvType t, std::uint32_t hd) {
        hv::AttentionArgs a;
        a.q = q;
        a.kv_pool = pool;
        a.n_pool_blocks = 2;
        a.block_tokens = 4;
        a.block_table = tab;
        a.n_head = 2;
        a.n_kv_head = 1;
        a.head_dim = hd;
        a.n_tokens = 1;
        a.q_offset = 3;
        a.scale = 0.125f;
        a.out = out;
        a.status = st;
        a.kv_type = t;
        ops.attention(s, a);
    };
    auto write = [&](hv::KvType t, std::uint32_t kv_dim) {
        hv::KvWriteArgs w;
        w.kv_pool = pool;
        w.n_pool_blocks = 2;
        w.block_tokens = 4;
        w.kv_dim = kv_dim;
        w.block_table = tab;
        w.start = 3;
        w.k = hv::BufferView(q, 0, kv_dim * 4);
        w.v = hv::BufferView(q, 4096, kv_dim * 4);
        w.status = st;
        w.kv_type = t;
        ops.kv_write(s, w);
    };
    auto unsupported = [](const char* what, auto&& fn) {
        try {
            fn();
            ADD_FAILURE() << what << ": accepted";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported) << what << ": " << e.what();
        }
    };
    EXPECT_NO_THROW(attn(hv::KvType::F16, 20));
    EXPECT_NO_THROW(attn(hv::KvType::Q8, 64));
    EXPECT_NO_THROW(write(hv::KvType::F16, 20));
    EXPECT_NO_THROW(write(hv::KvType::Q8, 96));
    unsupported("f16 attention, odd head_dim 21", [&] { attn(hv::KvType::F16, 21); });
    unsupported("q8 attention, head_dim 48", [&] { attn(hv::KvType::Q8, 48); });
    unsupported("f16 kv_write, odd kv_dim 21", [&] { write(hv::KvType::F16, 21); });
    unsupported("q8 kv_write, kv_dim 80", [&] { write(hv::KvType::Q8, 80); });
    s.submit_and_wait();
}
