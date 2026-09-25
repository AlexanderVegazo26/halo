// KV write (append) of the HIP backend, differentially tested against
// halo::kv_cache::SequenceKv::write on the same pool and inputs (TRD §30, review M-2): the
// whole pool must be byte-identical afterwards. Device tests skip on the dev host (D-001).

#include <algorithm>
#include <cstring>

#include "halo/kv_cache/paged_kv.h"
#include "hip_test_util.h"

namespace halo::hip::test {
namespace {

struct KvCase {
    std::uint32_t kv_dim = 1024;  // qwen35: 4 KV heads x 256 (D-004)
    std::uint32_t bt = 16;
    std::uint32_t prefix = 13;    // rows already committed
    std::uint32_t T = 5;          // rows appended (13..17 crosses the 16-row block boundary)
    std::uint32_t seed = 1;
};

void check_kv_write(const KvCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "kv_dim " << c.kv_dim << " bt " << c.bt << " prefix " << c.prefix << " T "
                                    << c.T);
    constexpr std::size_t kLayers = 2;
    constexpr std::size_t kLayer = 1;
    const std::size_t S = c.prefix + c.T;
    kv_cache::KvPool pool(kv_cache::KvLayout{kLayers, c.kv_dim, c.bt}, (S + c.bt - 1) / c.bt * 2 + 2);
    // Every pool float gets a defined value first, so the byte comparison covers the whole pool.
    for (std::size_t bl = 0; bl < pool.total_blocks(); ++bl) {
        for (std::size_t L = 0; L < kLayers; ++L) {
            std::fill_n(pool.k_rows(static_cast<kv_cache::BlockId>(bl), L), static_cast<std::size_t>(c.bt) * c.kv_dim, 777.0f);
            std::fill_n(pool.v_rows(static_cast<kv_cache::BlockId>(bl), L), static_cast<std::size_t>(c.bt) * c.kv_dim, -777.0f);
        }
    }
    kv_cache::SequenceKv seq(pool), other(pool);
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> row(c.kv_dim);
    // A committed prefix in both layers, with another sequence interleaved (fragmented table).
    for (std::size_t s = 0; s < c.prefix; ++s) {
        if (s % c.bt == 0) {
            other.reserve(c.bt);
            other.commit(0);
        }
        seq.reserve(1);
        for (std::size_t L = 0; L < kLayers; ++L) {
            for (auto& x : row) x = u(rng);
            seq.write(L, s, row, row);
        }
        seq.commit(1);
    }
    seq.reserve(c.T);  // allocates the blocks the append needs
    const std::size_t pool_floats = pool.total_blocks() * pool.layout().block_floats();
    std::vector<float> before(pool_floats);
    std::memcpy(before.data(), pool.k_rows(0, 0), pool_floats * sizeof(float));
    // New rows, then the CPU reference write.
    std::vector<float> k = uniform(rng, c.T * static_cast<std::size_t>(c.kv_dim), -2.0f, 2.0f);
    std::vector<float> v = uniform(rng, c.T * static_cast<std::size_t>(c.kv_dim), -2.0f, 2.0f);
    for (std::size_t t = 0; t < c.T; ++t) {
        seq.write(kLayer, c.prefix + t, std::span<const float>(&k[t * c.kv_dim], c.kv_dim),
                  std::span<const float>(&v[t * c.kv_dim], c.kv_dim));
    }
    std::vector<float> after(pool_floats);
    std::memcpy(after.data(), pool.k_rows(0, 0), pool_floats * sizeof(float));
    // HIP write on the pre-append copy.
    std::vector<float> table(seq.blocks().size());
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = std::bit_cast<float>(seq.blocks()[i]);
    std::vector<std::uint32_t> status{0xDEADBEEFu};
    const Buffer bp = r.make(before), bt = r.make(table), bk = r.make(k), bv = r.make(v), bs = r.make_words(status);
    KvWriteArgs a;
    a.kv_pool = bp;
    a.n_pool_blocks = static_cast<std::uint32_t>(pool.total_blocks());
    a.n_layers = kLayers;
    a.layer = kLayer;
    a.block_tokens = c.bt;
    a.kv_dim = c.kv_dim;
    a.block_table = bt;
    a.start = c.prefix;
    a.n_tokens = c.T;
    a.k = bk;
    a.v = bv;
    a.status = bs;
    Ops().kv_write(r.target(), a);
    r.finish();
    EXPECT_NO_THROW(Ops::check_status(r.target(), bs));
    r.fetch(bp, before);
    // Pure copies: bitwise on the device too. The whole pool (other layers, other blocks,
    // the prefix rows) must match what SequenceKv::write left.
    EXPECT_TRUE(matches(after, before, true, "whole pool after append"));
}

std::vector<KvCase> kv_cases() {
    return {
        {.kv_dim = 1024, .bt = 16, .prefix = 13, .T = 5, .seed = 1201},  // MTP verify crossing a block boundary
        {.kv_dim = 1024, .bt = 16, .prefix = 40, .T = 1, .seed = 1202},  // decode step into a partial block
        {.kv_dim = 40, .bt = 5, .prefix = 0, .T = 12, .seed = 1203},     // prefill chunk over 3 blocks
    };
}

TEST(HipKvEmu, WriteMatchesSequenceKvWrite) {
    for (const KvCase& c : kv_cases()) {
        for (auto& r : emulation_runners()) check_kv_write(c, *r);
    }
}

TEST(HipKvDevice, WriteMatchesSequenceKvWrite) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const KvCase& c : kv_cases()) check_kv_write(c, r);
}

TEST(HipKvEmu, WriteBadBlockRaises) {
    std::vector<float> f(1u << 16, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * 4);
    const std::uint32_t tbl[2] = {0, 9};  // 1 pool block: id 9 is out of range
    std::memcpy(&f[40000], tbl, sizeof(tbl));
    KvWriteArgs a;
    a.kv_pool = BufferView(b, 0, 2 * 16 * 8 * 4);
    a.n_pool_blocks = 1;
    a.n_layers = 1;
    a.block_tokens = 16;
    a.kv_dim = 8;
    a.block_table = BufferView(b, 160000, 8);
    a.start = 15;
    a.n_tokens = 2;  // rows 15 (block 0) and 16 (block 1 = id 9)
    a.k = BufferView(b, 170000, 64);
    a.v = BufferView(b, 180000, 64);
    a.status = BufferView(b, 190000, 4);
    Ops().kv_write(Target::emulation(), a);
    try {
        Ops::check_status(Target::emulation(), a.status);
        ADD_FAILURE() << "block id 9 of 1 accepted";
    } catch (const Error& e) {
        EXPECT_NE(std::string(e.what()).find("block table id outside the KV pool"), std::string::npos) << e.what();
    }
}

}  // namespace
}  // namespace halo::hip::test
