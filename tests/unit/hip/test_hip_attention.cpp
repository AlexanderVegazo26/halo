// ATTENTION (paged GQA) of the HIP backend, differentially tested against
// cpu::attention_gqa on the same inputs (TRD §30, review M-2).
//
// The K/V history lives in a real halo::kv_cache::KvPool (fragmented block tables, a
// partially filled last block, a non-zero layer); the HIP op reads a byte copy of the pool
// storage plus the sequence's block ids, and the CPU reads SequenceKv::keys()/values().
//
//   attn_exact_b128 : the CPU's order -> must be BIT-IDENTICAL (emulation).
//   attn_online_*   : running-max single pass -> within the bound derived below.
//
// Bound (first order, u = 2^-24), per output element d:
//   |o_hip - o_cpu| <= u * (4*N + 3*n_tiles + 4*R + 16) * sum_s w_s |v_s[d]|
// N = keys attended, R = max_s (m - x_s) the score range, w_s the softmax weights.
// Terms: each side's exp (<= 2u each), the rounding of the exponent argument (<= u*R per side)
// and of the running-max rescale chain (<= u*R + n_tiles*u), the normalizing sum (<= N u per
// side) and the weighted accumulation (<= (N + n_tiles) u). Weights and R are computed in
// double from the CPU-side inputs. The RoPE head-stride (TD-9) test is in test_hip_head.cpp
// (no kv_cache dependency).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/hip/registry.h"
#include "halo/kv_cache/paged_kv.h"
#include "hip_test_util.h"

namespace halo::hip::test {
namespace {

struct AttnCase {
    std::uint32_t n_head = 24, n_kv = 4, hd = 256;  // qwen35 (D-004)
    std::uint32_t bt = 16;                          // block tokens
    std::uint32_t T = 1;
    std::uint32_t q_offset = 40;
    std::uint32_t q_head_stride = 0;  // 0 = dense; 2*hd = interleaved [Q | gate]
    std::string variant = "attn_online_b128";
    int bad_block = 0;                // 1 = corrupt one table entry
    std::uint32_t seed = 1;
};

void check_attention(const AttnCase& c, Runner& r) {
    SCOPED_TRACE(r.name() + " " + c.variant);
    SCOPED_TRACE(testing::Message() << "heads " << c.n_head << "/" << c.n_kv << " hd " << c.hd << " bt " << c.bt << " T "
                                    << c.T << " q_offset " << c.q_offset << " q_head_stride " << c.q_head_stride);
    std::mt19937 rng(c.seed);
    const std::size_t kv_dim = static_cast<std::size_t>(c.n_kv) * c.hd;
    const std::size_t S = c.q_offset + c.T;
    constexpr std::size_t kLayers = 2;
    constexpr std::size_t kLayer = 1;
    const std::size_t blocks_needed = (S + c.bt - 1) / c.bt;
    kv_cache::KvPool pool(kv_cache::KvLayout{kLayers, kv_dim, c.bt}, 2 * blocks_needed + 3);
    // Two sequences grown in alternation, so each one's block table is fragmented.
    kv_cache::SequenceKv seq(pool), other(pool);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> krow(kv_dim), vrow(kv_dim);
    for (std::size_t s = 0; s < S; ++s) {
        if (s % c.bt == 0) {
            other.reserve(c.bt);
            for (std::size_t j = 0; j < c.bt; ++j) {
                for (auto& x : krow) x = u(rng);
                other.write(0, other.length() + j, krow, krow);
            }
            other.commit(c.bt);
        }
        seq.reserve(1);
        for (std::size_t L = 0; L < kLayers; ++L) {
            for (auto& x : krow) x = u(rng) * 2.0f;
            for (auto& x : vrow) x = u(rng);
            seq.write(L, s, krow, vrow);
        }
        seq.commit(1);
    }
    ASSERT_EQ(seq.blocks().size(), blocks_needed);
    ASSERT_NE(S % c.bt, 0u) << "cases are chosen with a partially filled last block";

    const std::size_t hs = c.q_head_stride == 0 ? c.hd : c.q_head_stride;
    const std::size_t q_row = static_cast<std::size_t>(c.n_head) * hs;
    std::vector<float> qbuf = uniform(rng, c.T * q_row, -1.0f, 1.0f);
    std::vector<float> qd(c.T * static_cast<std::size_t>(c.n_head) * c.hd);  // dense copy for the CPU
    for (std::size_t t = 0; t < c.T; ++t) {
        for (std::size_t h = 0; h < c.n_head; ++h) {
            std::copy_n(&qbuf[t * q_row + h * hs], c.hd, &qd[(t * c.n_head + h) * c.hd]);
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.hd));
    std::vector<const float*> kt, vt;
    const cpu::PagedRows kp = seq.keys(kLayer, S, kt);
    const cpu::PagedRows vp = seq.values(kLayer, S, vt);
    const std::size_t qcols = static_cast<std::size_t>(c.n_head) * c.hd;
    std::vector<float> ref(c.T * qcols, 0.0f);
    cpu::attention_gqa(cpu::AttentionDims{c.n_head, c.n_kv, c.hd}, cpu::ConstRows(qd.data(), c.T, qcols, qcols), kp, vp,
                       c.q_offset, scale, cpu::Rows(ref.data(), c.T, qcols, qcols));

    // HIP operands: a byte copy of the whole pool, the block ids, q (possibly interleaved).
    const std::size_t pool_floats = pool.total_blocks() * pool.layout().block_floats();
    std::vector<float> poolv(pool_floats);
    std::memcpy(poolv.data(), pool.k_rows(0, 0), pool_floats * sizeof(float));
    std::vector<float> table(seq.blocks().size());
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = std::bit_cast<float>(seq.blocks()[i]);
    if (c.bad_block != 0) table[table.size() / 2] = std::bit_cast<float>(static_cast<std::uint32_t>(pool.total_blocks()));
    std::vector<float> out(ref.size(), 999.0f);
    OpsOptions o;
    o.attention = c.variant;
    const Ops ops(o);
    std::vector<float> ws(ops.attention_workspace_bytes(c.T, c.n_head, c.q_offset) / 4 + 1, 0.0f);
    std::vector<std::uint32_t> status{0xDEADBEEFu};
    const Buffer bq = r.make(qbuf), bp = r.make(poolv), bt = r.make(table), bo = r.make(out), bw = r.make(ws),
                 bs = r.make_words(status);
    AttentionArgs a;
    a.q = BufferView(bq, 0, 0, q_row * 4);
    a.q_head_stride = c.q_head_stride;
    a.kv_pool = bp;
    a.n_pool_blocks = static_cast<std::uint32_t>(pool.total_blocks());
    a.n_layers = kLayers;
    a.layer = kLayer;
    a.block_tokens = c.bt;
    a.block_table = bt;
    a.n_head = c.n_head;
    a.n_kv_head = c.n_kv;
    a.head_dim = c.hd;
    a.n_tokens = c.T;
    a.q_offset = c.q_offset;
    a.scale = scale;
    a.out = bo;
    a.workspace = bw;
    a.status = bs;
    ops.attention(r.target(), a);
    r.finish();
    if (c.bad_block != 0) {
        try {
            Ops::check_status(r.target(), bs);
            ADD_FAILURE() << "bad block id accepted";
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Kernel);
            EXPECT_NE(std::string(e.what()).find("block table id outside the KV pool"), std::string::npos) << e.what();
        }
        return;
    }
    EXPECT_NO_THROW(Ops::check_status(r.target(), bs));
    r.fetch(bo, out);
    if (c.variant == "attn_exact_b128" && r.exact()) {
        EXPECT_TRUE(matches(ref, out, true, "out (exact variant, bitwise vs cpu::attention_gqa)"));
        return;
    }
    // Derived bound (see the file comment), per element. Device runs use the same bound (its
    // exp term assumes <= 2 ulp per side, to be calibrated on the EVO-X2): the fixed
    // kDevAbs/kDevRel is not derived for this op and could be tighter than legitimate rounding
    // differences at long N.
    const double uu = std::ldexp(1.0, -24);
    const std::size_t group = c.n_head / c.n_kv;
    const unsigned block = find_variant("ATTENTION", c.variant).block;
    std::size_t bad = 0;
    double worst = 0.0;
    double worst_abs = 0.0;
    for (std::size_t t = 0; t < c.T; ++t) {
        const std::size_t n = c.q_offset + t + 1;
        const double n_tiles = std::ceil(static_cast<double>(n) / block);
        for (std::size_t h = 0; h < c.n_head; ++h) {
            const std::size_t kvh = h / group;
            std::vector<double> x(n);
            double m = -1e300;
            for (std::size_t s = 0; s < n; ++s) {
                double dot = 0.0;
                for (std::size_t d = 0; d < c.hd; ++d) {
                    dot += static_cast<double>(qd[(t * c.n_head + h) * c.hd + d]) *
                           static_cast<double>(kp.row_ptr(s)[kvh * c.hd + d]);
                }
                x[s] = dot * static_cast<double>(scale);
                m = std::max(m, x[s]);
            }
            double R = 0.0, l = 0.0;
            for (std::size_t s = 0; s < n; ++s) {
                R = std::max(R, m - x[s]);
                l += std::exp(x[s] - m);
            }
            for (std::size_t d = 0; d < c.hd; ++d) {
                double wv = 0.0;
                for (std::size_t s = 0; s < n; ++s) {
                    wv += std::exp(x[s] - m) / l * std::fabs(static_cast<double>(vp.row_ptr(s)[kvh * c.hd + d]));
                }
                const double bound =
                    uu * (4.0 * static_cast<double>(n) + 3.0 * n_tiles + 4.0 * R + 16.0) * wv * (1 + 1e-3) + 1e-30;
                const std::size_t i = (t * c.n_head + h) * c.hd + d;
                const double diff = std::fabs(static_cast<double>(out[i]) - static_cast<double>(ref[i]));
                worst = std::max(worst, diff / bound);
                worst_abs = std::max(worst_abs, diff);
                if (!(diff <= bound) && bad++ == 0) {
                    ADD_FAILURE() << "t " << t << " h " << h << " d " << d << ": cpu " << ref[i] << " hip " << out[i]
                                  << " |d| " << diff << " bound " << bound;
                }
            }
        }
    }
    EXPECT_EQ(bad, 0u);
    std::printf("    %s %s: worst |d| / bound = %.4f, worst |d| = %.3g\n", r.name().c_str(), c.variant.c_str(), worst,
                worst_abs);
}

std::vector<AttnCase> attn_cases() {
    std::vector<AttnCase> v;
    for (const char* var : {"attn_exact_b128", "attn_online_b128", "attn_online_b64"}) {
        // qwen35 decode: 41 history rows = 3 blocks of 16, the last holding 9.
        v.push_back({.T = 1, .q_offset = 40, .variant = var, .seed = 801});
        // MTP verify (causal, T = 4) reading Q in place from the interleaved [Q | gate] row.
        v.push_back({.T = 4, .q_offset = 29, .q_head_stride = 512, .variant = var, .seed = 802});
        // Odd sizes: 8-lane dot tail (hd 20), 3 query heads per KV head, history == T.
        v.push_back({.n_head = 6, .n_kv = 2, .hd = 20, .bt = 5, .T = 3, .q_offset = 0, .variant = var, .seed = 803});
        // Long history: 1001 keys = 8 tiles of 128 (16 of 64), 63 blocks.
        v.push_back({.n_head = 8, .n_kv = 2, .hd = 64, .bt = 16, .T = 1, .q_offset = 1000, .variant = var, .seed = 804});
    }
    return v;
}

TEST(HipAttnEmu, PagedGqaVsCpuAttention) {
    for (const AttnCase& c : attn_cases()) {
        for (auto& r : emulation_runners()) check_attention(c, *r);
    }
}

TEST(HipAttnEmu, BadBlockIdRaises) {
    for (const char* var : {"attn_exact_b128", "attn_online_b128"}) {
        const AttnCase c{.T = 2, .q_offset = 40, .variant = var, .bad_block = 1, .seed = 805};
        for (auto& r : emulation_runners()) check_attention(c, *r);
    }
}

TEST(HipAttnDevice, PagedGqaVsCpuAttention) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const AttnCase& c : attn_cases()) check_attention(c, r);
}

TEST(HipAttnEmu, RejectsBadShapes) {
    std::vector<float> f(1u << 16, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * 4);
    AttentionArgs a;
    a.q = BufferView(b, 0, 4096);
    a.kv_pool = BufferView(b, 8192, 65536);
    a.n_pool_blocks = 1;
    a.block_tokens = 16;
    a.block_table = BufferView(b, 131072, 64);
    a.n_head = 2;
    a.n_kv_head = 1;
    a.head_dim = 32;
    a.out = BufferView(b, 196608, 4096);
    a.status = BufferView(b, 250000, 4);
    ASSERT_NO_THROW(Ops().attention(Target::emulation(), a));  // valid baseline
    auto expect = [&](const char* needle) {
        try {
            Ops().attention(Target::emulation(), a);
            ADD_FAILURE() << "no error; expected: " << needle;
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Kernel) << e.what();
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    a.head_dim = 288;
    expect("head_dim 288 exceeds");
    a.head_dim = 32;
    a.n_head = 3;
    a.n_kv_head = 2;
    expect("bad dims");
    a.n_head = 2;
    a.n_kv_head = 1;
    a.q_offset = 16;  // 17 rows need 2 table entries; the view holds 1
    a.block_table = BufferView(b, 131072, 4);
    expect("block_table needs 8 bytes");
    a.q_offset = 0;
    a.block_table = BufferView(b, 131072, 64);
    a.layer = 1;
    expect("bad pool layout");
    a.layer = 0;
    a.out = BufferView(b, 8192 + 64, 256);
    expect("out overlaps kv_pool");
}

}  // namespace
}  // namespace halo::hip::test
