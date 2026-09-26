// WS-BI-1 migration gate (ADR-001 §6.2): models::Qwen35 composed over the CPU backend must be
// bit-identical to the pre-backend forward (legacy_qwen35.cpp, a verbatim copy compiled the
// same way) on every tiny-model file.
//
// Two sides, each with its own KV pools, sequences and GDN states, are driven through the same
// operation sequence. After every call the test compares, bit for bit (never with ==):
//   - every result: argmax index and value bits, full logits rows, hidden rows, per-layer
//     inputs (capture_layer_inputs), the resolved GDN paths;
//   - the WHOLE storage of both KV pools (filled with the same sentinel first, so a write to
//     a wrong row or a read of an unwritten row shows up), every block table and length;
//   - every GDN layer's live recurrent and conv state, all max_slots slots, slot bookkeeping;
//   - the D-011 cost: every StepCost field exactly; activation_bytes is the legacy value minus
//     exactly the removed [Q | gate] de-interleave copy (16 * rows * n_head * head_dim bytes
//     per attention layer call).
// The operation sequence varies the dimensions the invariants are stated over: a prefill
// crossing GDN chunk and KV block boundaries, S = 1 and ragged S = 3 batches that mix chunked
// prefill, decode and slot-writing verify in one call, verify K = 1..4 followed by
// commit_rows_kept for every m, a copy-on-write shared prefix, MTP catch-up plus a 2-deep draft
// chain, and the Full, Argmax and None logits modes. Parameters: every tiny-*.gguf present,
// GDN chunk 64 and 8, a thread pool and a null pool, the pool constructor and the
// Qwen35(model, Backend&) constructor.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/core/error.h"
#include "legacy_qwen35.h"
#include "tiny_golden.h"

using halo::kv_cache::KvPool;
using halo::models::ForwardOptions;
using halo::models::GdnPath;
using halo::models::LogitsMode;
using halo::models::MtpStep;
using halo::models::Qwen35;
using halo::models::Qwen35Options;
using halo::models::SeqStep;
using halo::models::StepResult;
using halo::models::legacy::LegacyQwen35;
using halo::test::TestSeq;
namespace hb = halo::backend;

namespace {

constexpr std::size_t kSlots = 4;   // verify K <= 4
constexpr std::size_t kBlocks = 96;
constexpr std::size_t kSeqs = 5;
constexpr float kSentinel = -7.25f;

::testing::AssertionResult same_bits(std::span<const float> a, std::span<const float> b, const std::string& what) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << what << ": " << a.size() << " vs " << b.size() << " floats";
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) {
            return ::testing::AssertionFailure() << what << ": first difference at element " << i << " of " << a.size()
                                                 << " (new " << a[i] << ", legacy " << b[i] << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

std::span<const float> pool_storage(const KvPool& p) {
    return {p.k_rows(0, 0), p.total_blocks() * p.layout().block_floats()};
}

using ForwardFn = std::function<void(std::span<const SeqStep>, StepResult&, const ForwardOptions&)>;
using MtpFn = std::function<void(std::span<const MtpStep>, StepResult&)>;

/// One side of the gate: a model's entry points plus its own pools and sequences.
struct Side {
    ForwardFn forward;
    MtpFn mtp_forward;
    std::unique_ptr<KvPool> kv_pool, mtp_pool;
    std::vector<std::unique_ptr<TestSeq>> seqs;
    std::map<std::string, StepResult> saved;

    Side(ForwardFn f, MtpFn m, const halo::kv_cache::KvLayout& kvl, const halo::kv_cache::KvLayout& mtpl,
         const halo::state::GdnShape& shape)
        : forward(std::move(f)), mtp_forward(std::move(m)), kv_pool(std::make_unique<KvPool>(kvl, kBlocks)),
          mtp_pool(std::make_unique<KvPool>(mtpl, kBlocks)) {
        for (KvPool* p : {kv_pool.get(), mtp_pool.get()}) {
            std::fill_n(p->k_rows(0, 0), p->total_blocks() * p->layout().block_floats(), kSentinel);
        }
        for (std::size_t i = 0; i < kSeqs; ++i) seqs.push_back(std::make_unique<TestSeq>(*kv_pool, *mtp_pool, shape, kSlots));
    }
    TestSeq& seq(std::size_t i) { return *seqs.at(i); }
};

/// Drives both sides in lockstep and compares after every call.
class Gate {
public:
    Gate(const Qwen35& nw, const LegacyQwen35& lg)
        : n_([&nw](auto s, auto& r, const auto& o) { nw.forward(s, r, o); }, [&nw](auto s, auto& r) { nw.mtp_forward(s, r); },
             nw.kv_layout(), nw.mtp_kv_layout(), nw.gdn_shape()),
          l_([&lg](auto s, auto& r, const auto& o) { lg.forward(s, r, o); }, [&lg](auto s, auto& r) { lg.mtp_forward(s, r); },
             lg.kv_layout(), lg.mtp_kv_layout(), lg.gdn_shape()),
          qd_(static_cast<std::uint64_t>(nw.hparams().n_head) * nw.hparams().key_length),
          n_attn_(nw.hparams().n_attn_layers), E_(nw.n_embd()) {}

    /// make(Side&) -> std::vector<SeqStep> over that side's sequences.
    template <class Make>
    void forward(const std::string& what, Make&& make, const ForwardOptions& opts = {}, const std::string& save = {}) {
        SCOPED_TRACE(what);
        const std::vector<SeqStep> sn = make(n_);
        const std::vector<SeqStep> sl = make(l_);
        StepResult rn, rl;
        n_.forward(sn, rn, opts);
        l_.forward(sl, rl, opts);
        std::uint64_t rows = 0;
        for (const SeqStep& s : sn) rows += s.tokens.size();
        compare_results(rn, rl, 16 * rows * qd_ * n_attn_);
        compare_state();
        if (!save.empty()) {
            n_.saved[save] = std::move(rn);
            l_.saved[save] = std::move(rl);
        }
    }

    template <class Make>
    void mtp(const std::string& what, Make&& make, const std::string& save = {}) {
        SCOPED_TRACE(what);
        const std::vector<MtpStep> sn = make(n_);
        const std::vector<MtpStep> sl = make(l_);
        StepResult rn, rl;
        n_.mtp_forward(sn, rn);
        l_.mtp_forward(sl, rl);
        std::uint64_t rows = 0;
        for (const MtpStep& s : sn) rows += s.tokens.size();
        compare_results(rn, rl, 16 * rows * qd_);
        compare_state();
        if (!save.empty()) {
            n_.saved[save] = std::move(rn);
            l_.saved[save] = std::move(rl);
        }
    }

    /// The same host-side state operation on both sides (truncate, commit, share), then compare.
    template <class Op>
    void both(const std::string& what, Op&& op) {
        SCOPED_TRACE(what);
        op(n_);
        op(l_);
        compare_state();
    }

    [[nodiscard]] std::size_t n_embd() const noexcept { return E_; }
    [[nodiscard]] Side& new_side() noexcept { return n_; }

private:
    static void compare_results(const StepResult& n, const StepResult& l, std::uint64_t act_delta) {
        ASSERT_EQ(n.seqs.size(), l.seqs.size());
        for (std::size_t i = 0; i < n.seqs.size(); ++i) {
            const auto& a = n.seqs[i];
            const auto& b = l.seqs[i];
            ASSERT_EQ(a.argmax.size(), b.argmax.size()) << "seq " << i;
            for (std::size_t j = 0; j < a.argmax.size(); ++j) {
                EXPECT_EQ(a.argmax[j].index, b.argmax[j].index) << "seq " << i << " logit row " << j;
                EXPECT_EQ(std::bit_cast<std::uint32_t>(a.argmax[j].value), std::bit_cast<std::uint32_t>(b.argmax[j].value))
                    << "seq " << i << " logit row " << j;
            }
            EXPECT_TRUE(same_bits(a.logits, b.logits, "logits of seq " + std::to_string(i)));
            EXPECT_TRUE(same_bits(a.hidden, b.hidden, "hidden of seq " + std::to_string(i)));
        }
        EXPECT_EQ(n.gdn_paths, l.gdn_paths);
        ASSERT_EQ(n.layer_inputs.size(), l.layer_inputs.size());
        for (std::size_t li = 0; li < n.layer_inputs.size(); ++li) {
            EXPECT_TRUE(same_bits(n.layer_inputs[li], l.layer_inputs[li], "layer input " + std::to_string(li)));
        }
        EXPECT_EQ(n.cost.weight_bytes, l.cost.weight_bytes);
        EXPECT_EQ(n.cost.weight_passes, l.cost.weight_passes);
        EXPECT_EQ(n.cost.embedding_bytes, l.cost.embedding_bytes);
        EXPECT_EQ(n.cost.state_bytes, l.cost.state_bytes);
        EXPECT_EQ(n.cost.kv_bytes, l.cost.kv_bytes);
        EXPECT_EQ(l.cost.activation_bytes - n.cost.activation_bytes, act_delta)
            << "activation bytes must drop by exactly the removed de-interleave copy (new " << n.cost.activation_bytes
            << ", legacy " << l.cost.activation_bytes << ")";
    }

    void compare_state() {
        EXPECT_TRUE(same_bits(pool_storage(*n_.kv_pool), pool_storage(*l_.kv_pool), "trunk KV pool storage"));
        EXPECT_TRUE(same_bits(pool_storage(*n_.mtp_pool), pool_storage(*l_.mtp_pool), "MTP KV pool storage"));
        for (std::size_t i = 0; i < kSeqs; ++i) {
            TestSeq& a = n_.seq(i);
            TestSeq& b = l_.seq(i);
            const std::string s = "seq " + std::to_string(i);
            EXPECT_EQ(a.kv.length(), b.kv.length()) << s;
            EXPECT_TRUE(std::ranges::equal(a.kv.blocks(), b.kv.blocks())) << s << " block table";
            EXPECT_EQ(a.mtp_kv.length(), b.mtp_kv.length()) << s;
            EXPECT_TRUE(std::ranges::equal(a.mtp_kv.blocks(), b.mtp_kv.blocks())) << s << " MTP block table";
            EXPECT_EQ(a.gdn.slot_rows(), b.gdn.slot_rows()) << s;
            EXPECT_EQ(a.gdn.slots_valid(), b.gdn.slots_valid()) << s;
            for (std::size_t layer = 0; layer < a.gdn.shape().n_layers; ++layer) {
                const std::string sl = s + " GDN layer " + std::to_string(layer);
                EXPECT_TRUE(same_bits(a.gdn.recurrent(layer), b.gdn.recurrent(layer), sl + " live recurrent"));
                const auto ca = a.gdn.conv(layer);
                const auto cb = b.gdn.conv(layer);
                EXPECT_TRUE(same_bits({ca.data(), ca.rows() * ca.cols()}, {cb.data(), cb.rows() * cb.cols()}, sl + " live conv"));
                EXPECT_TRUE(same_bits(a.gdn.recurrent_slots(layer, kSlots), b.gdn.recurrent_slots(layer, kSlots),
                                      sl + " recurrent slots"));
                EXPECT_TRUE(same_bits(a.gdn.conv_slots(layer, kSlots), b.gdn.conv_slots(layer, kSlots), sl + " conv slots"));
            }
        }
    }

    Side n_, l_;
    std::uint64_t qd_, n_attn_;
    std::size_t E_;
};

/// Deterministic token ids in [0, n_vocab).
class Tokens {
public:
    explicit Tokens(std::size_t n_vocab) : n_vocab_(n_vocab) {}
    std::vector<std::int32_t> next(std::size_t n) {
        std::vector<std::int32_t> v(n);
        for (auto& t : v) {
            state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
            t = static_cast<std::int32_t>((state_ >> 33) % n_vocab_);
        }
        return v;
    }

private:
    std::size_t n_vocab_;
    std::uint64_t state_ = 0x243F6A8885A308D3ULL;
};

SeqStep seq_step(TestSeq& q, std::span<const std::int32_t> tokens, std::span<const std::size_t> rows, LogitsMode mode,
                 bool hidden = false, std::size_t slots = 0, GdnPath path = GdnPath::Auto) {
    SeqStep s;
    s.tokens = tokens;
    s.kv = &q.kv;
    s.gdn = &q.gdn;
    s.logit_rows = rows;
    s.logits = mode;
    s.want_hidden = hidden;
    s.n_state_slots = slots;
    s.gdn_path = path;
    return s;
}

#define GATE_STEP(x)                                              \
    do {                                                          \
        x;                                                        \
        if (::testing::Test::HasFailure()) return;                \
    } while (false)

/// The §6.2 operation sequence.
void run_scenario(Gate& g, std::size_t n_vocab, bool has_mtp) {
    Tokens gen(n_vocab);
    const std::size_t E = g.n_embd();

    // 1. S = 1 prefill of 70 rows: crosses the GDN chunk (64, or 8 for the small-chunk
    //    parameter) and four KV blocks of 16; per-layer inputs captured.
    const auto p0 = gen.next(70);
    const std::vector<std::size_t> r0{0, 37, 69};
    GATE_STEP(g.forward(
        "prefill S=1, 70 rows, Full", [&](Side& s) { return std::vector{seq_step(s.seq(0), p0, r0, LogitsMode::Full, true)}; },
        {true}, "p0"));

    // 2. Decode, Argmax.
    const auto d0 = gen.next(1);
    const std::vector<std::size_t> row0{0};
    GATE_STEP(g.forward("decode S=1, Argmax",
                        [&](Side& s) { return std::vector{seq_step(s.seq(0), d0, row0, LogitsMode::Argmax)}; }));

    // 3. Ragged S = 3: decode (Full) + chunked prefill of 19 rows (Argmax, 2 rows out of order)
    //    + one-row prefill (None, hidden); per-layer inputs captured.
    const auto d1 = gen.next(1);
    const auto p1 = gen.next(19);
    const auto p2 = gen.next(1);
    const std::vector<std::size_t> r1{18, 3};
    const std::vector<std::size_t> none{};
    GATE_STEP(g.forward(
        "ragged S=3: decode + prefill 19 + prefill 1",
        [&](Side& s) {
            return std::vector{seq_step(s.seq(0), d1, row0, LogitsMode::Full), seq_step(s.seq(1), p1, r1, LogitsMode::Argmax, true),
                               seq_step(s.seq(2), p2, none, LogitsMode::None, true)};
        },
        {true}, "ragged"));

    // 4. Explicit GDN paths: recurrent multi-row without slots, chunked, and Auto.
    const auto a1 = gen.next(3);
    const auto a2 = gen.next(5);
    const auto a3 = gen.next(9);
    const std::vector<std::size_t> ra1{2};
    const std::vector<std::size_t> ra2{4, 0};
    GATE_STEP(g.forward("explicit paths: Recurrent 3 + Chunked 5 + Auto 9", [&](Side& s) {
        return std::vector{seq_step(s.seq(1), a1, ra1, LogitsMode::Full, false, 0, GdnPath::Recurrent),
                           seq_step(s.seq(2), a2, ra2, LogitsMode::Argmax, false, 0, GdnPath::Chunked),
                           seq_step(s.seq(3), a3, none, LogitsMode::None)};
    }));

    // 5. Verify K = 1..4 with slots, then commit_rows_kept for every m; in the same call a
    //    one-row decode and a 6-row chunked prefill of other sequences.
    for (std::size_t K = 1; K <= kSlots; ++K) {
        for (std::size_t m = 1; m <= K; ++m) {
            const std::string tag = "verify K=" + std::to_string(K) + " m=" + std::to_string(m);
            const auto v = gen.next(K);
            const auto dd = gen.next(1);
            const auto pp = gen.next(6);
            std::vector<std::size_t> all(K);
            std::iota(all.begin(), all.end(), 0);
            const std::vector<std::size_t> rp{5};
            const std::size_t len0 = g.new_side().seq(0).kv.length();
            GATE_STEP(g.forward(tag, [&](Side& s) {
                return std::vector{seq_step(s.seq(0), v, all, LogitsMode::Full, true, K),
                                   seq_step(s.seq(1), dd, row0, LogitsMode::Argmax),
                                   seq_step(s.seq(2), pp, rp, LogitsMode::Argmax)};
            }));
            GATE_STEP(g.both(tag + " commit", [&](Side& s) {
                s.seq(0).kv.truncate(len0 + m);
                s.seq(0).gdn.commit_rows_kept(K, m);
            }));
        }
    }

    // 6. Copy-on-write shared prefix: seq 4 shares seq 0's KV up to 5 rows before its end (a
    //    partially filled block), then both write into the shared block in one call.
    GATE_STEP(g.both("share prefix", [&](Side& s) {
        s.seq(4).kv.share_prefix(s.seq(0).kv, s.seq(0).kv.length() - 5);
        s.seq(4).gdn.copy_from(s.seq(0).gdn);
    }));
    const auto c4 = gen.next(4);
    const auto c0 = gen.next(1);
    const std::vector<std::size_t> rc4{3, 1};
    GATE_STEP(g.forward("COW: shared-prefix sequence + its source in one call", [&](Side& s) {
        return std::vector{seq_step(s.seq(4), c4, rc4, LogitsMode::Full), seq_step(s.seq(0), c0, row0, LogitsMode::Argmax)};
    }));

    if (!has_mtp) return;
    // 7. MTP catch-up (S = 2) on the trunk hidden rows, then a 2-deep draft chain.
    const std::span<const std::int32_t> m0_tokens = std::span(p0).subspan(1);
    const std::span<const std::int32_t> m1_tokens = std::span(p1).subspan(1);
    const std::vector<std::size_t> rm0{0, 68};
    const std::vector<std::size_t> rm1{17};
    GATE_STEP(g.mtp(
        "MTP catch-up S=2",
        [&](Side& s) {
            MtpStep a;
            a.tokens = m0_tokens;
            a.hidden = std::span<const float>(s.saved.at("p0").seqs[0].hidden).first(m0_tokens.size() * E);
            a.kv = &s.seq(0).mtp_kv;
            a.first_position = 1;
            a.logit_rows = rm0;
            a.logits = LogitsMode::Full;
            a.want_hidden = true;
            MtpStep b;
            b.tokens = m1_tokens;
            b.hidden = std::span<const float>(s.saved.at("ragged").seqs[1].hidden).first(m1_tokens.size() * E);
            b.kv = &s.seq(1).mtp_kv;
            b.first_position = 1;
            b.logit_rows = rm1;
            b.logits = LogitsMode::Argmax;
            return std::vector{a, b};
        },
        "m0"));
    std::int32_t pos = 1 + static_cast<std::int32_t>(m0_tokens.size());
    const char* prev = "m0";
    const char* depth_tags[] = {"m1", "m2"};
    for (std::size_t depth = 0; depth < 2; ++depth) {
        std::map<const Side*, std::int32_t> tok;  // each side drafts from its own argmax
        GATE_STEP(g.mtp(
            "MTP draft depth " + std::to_string(depth + 1),
            [&](Side& s) {
                const auto& o = s.saved.at(prev).seqs[0];
                tok[&s] = o.argmax.back().index;
                MtpStep a;
                a.tokens = std::span(&tok[&s], 1);
                a.hidden = std::span<const float>(o.hidden).last(E);
                a.kv = &s.seq(0).mtp_kv;
                a.first_position = pos;
                a.logit_rows = depth == 0 ? std::span<const std::size_t>(row0) : std::span<const std::size_t>(none);
                a.logits = depth == 0 ? LogitsMode::Argmax : LogitsMode::None;
                a.want_hidden = true;
                return std::vector{a};
            },
            depth_tags[depth]));
        prev = depth_tags[depth];
        ++pos;
    }
}

// ---------------------------------------------------------------------------------------

enum class Ctor : std::uint8_t { Pool, Backend };

struct GateParam {
    const char* file;
    std::size_t chunk;
    bool threads;
    Ctor ctor;
};

std::string param_name(const ::testing::TestParamInfo<GateParam>& info) {
    const GateParam& p = info.param;
    std::string f = p.file;
    f = f.substr(5, f.size() - 10);  // tiny-XXX.gguf -> XXX
    return f + "_chunk" + std::to_string(p.chunk) + (p.threads ? "_pool" : "_nopool") +
           (p.ctor == Ctor::Pool ? "_poolctor" : "_backendctor");
}

class BackendGate : public ::testing::TestWithParam<GateParam> {};

TEST_P(BackendGate, ForwardOverCpuBackendIsBitIdenticalToLegacy) {
    const GateParam& p = GetParam();
    const auto path = halo::test::tiny_dir() / p.file;
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " missing";
    const auto nm = halo::model::NormalizedModel::load(path);
    std::unique_ptr<halo::cpu::ThreadPool> pool;
    if (p.threads) pool = std::make_unique<halo::cpu::ThreadPool>(halo::cpu::ThreadPool::default_threads());
    const Qwen35Options opts{p.chunk};
    std::unique_ptr<hb::Backend> be;
    std::unique_ptr<Qwen35> nw;
    if (p.ctor == Ctor::Backend) {
        be = hb::make_cpu_backend(pool.get());
        nw = std::make_unique<Qwen35>(nm, *be, opts);
        ASSERT_EQ(&nw->backend(), be.get());
    } else {
        nw = std::make_unique<Qwen35>(nm, pool.get(), opts);
    }
    EXPECT_EQ(nw->backend().kind(), hb::Kind::Cpu);
    const LegacyQwen35 lg(nm, pool.get(), opts);
    ASSERT_EQ(nw->has_mtp(), lg.has_mtp());
    if (std::string_view(p.file) == "tiny-f32.gguf") ASSERT_TRUE(nw->has_mtp()) << "the MTP part of the scenario must run";
    EXPECT_EQ(nw->trunk_weight_bytes(), lg.trunk_weight_bytes());
    EXPECT_EQ(nw->lm_head_bytes(), lg.lm_head_bytes());
    EXPECT_EQ(nw->mtp_block_bytes(), lg.mtp_block_bytes());
    EXPECT_EQ(nw->mtp_head_bytes(), lg.mtp_head_bytes());
    EXPECT_EQ(nw->embedding_row_bytes(), lg.embedding_row_bytes());
    Gate g(*nw, lg);
    run_scenario(g, nw->n_vocab(), nw->has_mtp());
}

// Every tiny file with the default chunk; the other dimensions on f32 and one K-quant.
INSTANTIATE_TEST_SUITE_P(
    TinyFiles, BackendGate,
    ::testing::Values(GateParam{"tiny-f32.gguf", 64, true, Ctor::Pool}, GateParam{"tiny-q8_0.gguf", 64, true, Ctor::Pool},
                      GateParam{"tiny-q4_k_m.gguf", 64, true, Ctor::Pool}, GateParam{"tiny-q6_k.gguf", 64, true, Ctor::Pool},
                      GateParam{"tiny-iq4_xs.gguf", 64, true, Ctor::Pool}, GateParam{"tiny-q3_k_m.gguf", 64, true, Ctor::Pool},
                      GateParam{"tiny-q5_k_m.gguf", 64, true, Ctor::Pool}, GateParam{"tiny-q4_0.gguf", 64, true, Ctor::Pool},
                      GateParam{"tiny-f32.gguf", 8, false, Ctor::Backend}, GateParam{"tiny-f32.gguf", 64, true, Ctor::Backend},
                      GateParam{"tiny-q4_k_m.gguf", 8, true, Ctor::Backend}),
    param_name);

// ---------------------------------------------------------------------------------------
// Backend Limits (ADR-001 §5.2): checked at construction.
// ---------------------------------------------------------------------------------------

/// The CPU backend with other Limits (every call forwards to it).
class LimitedBackend final : public hb::Backend {
public:
    LimitedBackend(hb::Backend& inner, hb::Limits limits) : in_(inner), lim_(limits) {}
    [[nodiscard]] hb::Kind kind() const noexcept override { return in_.kind(); }
    [[nodiscard]] std::string describe() const override { return "limited " + in_.describe(); }
    [[nodiscard]] hb::Limits limits() const noexcept override { return lim_; }
    std::unique_ptr<hb::Buffer> allocate(std::uint64_t b, hb::Tier t) override { return in_.allocate(b, t); }
    std::unique_ptr<hb::Buffer> import_host(std::span<std::byte> b) override { return in_.import_host(b); }
    std::unique_ptr<hb::Buffer> import_host_readonly(std::span<const std::byte> b) override { return in_.import_host_readonly(b); }
    std::unique_ptr<hb::Stream> create_stream() override { return in_.create_stream(); }
    void upload(hb::Stream& s, hb::TensorRef d, std::span<const std::byte> b) override { in_.upload(s, d, b); }
    void download(hb::Stream& s, hb::TensorRef d, std::span<std::byte> b) override { in_.download(s, d, b); }
    [[nodiscard]] std::span<const hb::VariantInfo> variants(hb::OpId op, std::string_view f) const override {
        return in_.variants(op, f);
    }
    void get_rows(hb::Stream& s, const hb::GetRowsArgs& a) override { in_.get_rows(s, a); }
    void rms_norm(hb::Stream& s, const hb::RmsNormArgs& a) override { in_.rms_norm(s, a); }
    void add_rms_norm(hb::Stream& s, const hb::AddRmsNormArgs& a) override { in_.add_rms_norm(s, a); }
    void gemv(hb::Stream& s, const hb::GemvArgs& a) override { in_.gemv(s, a); }
    void gdn_gates(hb::Stream& s, const hb::GdnGateArgs& a) override { in_.gdn_gates(s, a); }
    void conv1d_silu(hb::Stream& s, const hb::Conv1dArgs& a) override { in_.conv1d_silu(s, a); }
    void gated_delta_rule(hb::Stream& s, const hb::GdnArgs& a) override { in_.gated_delta_rule(s, a); }
    void gated_rms_norm(hb::Stream& s, const hb::GatedNormArgs& a) override { in_.gated_rms_norm(s, a); }
    void partial_rope(hb::Stream& s, const hb::RopeArgs& a) override { in_.partial_rope(s, a); }
    void kv_write(hb::Stream& s, const hb::KvWriteArgs& a) override { in_.kv_write(s, a); }
    void attention(hb::Stream& s, const hb::AttentionArgs& a) override { in_.attention(s, a); }
    void swiglu(hb::Stream& s, const hb::EltwiseArgs& a) override { in_.swiglu(s, a); }
    void mul_sigmoid(hb::Stream& s, const hb::EltwiseArgs& a) override { in_.mul_sigmoid(s, a); }
    void add(hb::Stream& s, const hb::EltwiseArgs& a) override { in_.add(s, a); }
    void lm_head(hb::Stream& s, const hb::LmHeadArgs& a) override { in_.lm_head(s, a); }
    void argmax(hb::Stream& s, const hb::ArgmaxArgs& a) override { in_.argmax(s, a); }
    void top_k(hb::Stream& s, const hb::TopKArgs& a) override { in_.top_k(s, a); }
    void copy(hb::Stream& s, const hb::CopyArgs& a) override { in_.copy(s, a); }

private:
    hb::Backend& in_;
    hb::Limits lim_;
};

class BackendLimits : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = halo::test::tiny_dir() / "tiny-f32.gguf";
        if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " missing";
        nm_ = std::make_unique<halo::model::NormalizedModel>(halo::model::NormalizedModel::load(path));
        cpu_ = hb::make_cpu_backend(nullptr);
        full_ = cpu_->limits();
    }
    /// The error code of constructing Qwen35 over the CPU backend with `lim` (nullopt: none).
    std::optional<halo::ErrorCode> construct(const hb::Limits& lim, std::size_t chunk = 64) const {
        LimitedBackend be(*cpu_, lim);
        try {
            const Qwen35 m(*nm_, be, Qwen35Options{chunk});
        } catch (const halo::Error& e) {
            return e.code();
        }
        return std::nullopt;
    }
    std::unique_ptr<halo::model::NormalizedModel> nm_;
    std::unique_ptr<hb::Backend> cpu_;
    hb::Limits full_{};
};

// The tiny model: d_k 32, head dim 64, conv kernel 4, rope dims 16 (make_tiny_model.py).
TEST_F(BackendLimits, EveryLimitIsCheckedAtConstruction) {
    EXPECT_EQ(construct(full_), std::nullopt);
    const auto& hp = nm_->hparams();
    struct Case {
        const char* what;
        std::uint32_t hb::Limits::*field;
        std::uint64_t model_value;  // the model's dimension (the chunk: the option, 64)
    };
    const Case cases[] = {
        {"max_gdn_dk", &hb::Limits::max_gdn_dk, hp.gdn_head_k_dim},
        {"max_head_dim", &hb::Limits::max_head_dim, hp.key_length},
        {"max_conv_k", &hb::Limits::max_conv_k, hp.ssm_conv_kernel},
        {"max_rope_dims", &hb::Limits::max_rope_dims, hp.rope_dim},
        {"max_gdn_chunk", &hb::Limits::max_gdn_chunk, 64},
    };
    for (const Case& c : cases) {
        ASSERT_GT(c.model_value, 1u) << c.what;
        hb::Limits below = full_;
        below.*c.field = static_cast<std::uint32_t>(c.model_value - 1);
        EXPECT_EQ(construct(below), halo::ErrorCode::Unsupported) << c.what;
        hb::Limits at = full_;
        at.*c.field = static_cast<std::uint32_t>(c.model_value);
        EXPECT_EQ(construct(at), std::nullopt) << c.what << " at the limit";
    }
    // A backend without the chunked form does not care about the chunk length.
    hb::Limits nochunk = full_;
    nochunk.gdn_chunked = false;
    nochunk.max_gdn_chunk = 0;
    EXPECT_EQ(construct(nochunk), std::nullopt);
}

// Contract change (documented in qwen35.h): a chunk above the CPU backend's 1024 now fails at
// construction instead of at the first chunked forward.
TEST_F(BackendLimits, ChunkAboveCpuLimitFailsAtConstruction) {
    EXPECT_THROW(
        {
            try {
                const Qwen35 m(*nm_, static_cast<halo::cpu::ThreadPool*>(nullptr), Qwen35Options{1025});
            } catch (const halo::Error& e) {
                EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
                throw;
            }
        },
        halo::Error);
    EXPECT_NO_THROW({ const Qwen35 m(*nm_, static_cast<halo::cpu::ThreadPool*>(nullptr), Qwen35Options{1024}); });
    EXPECT_EQ(construct(full_, 4097), halo::ErrorCode::Config);
}

// Without a chunked form, Auto prefill runs the recurrent form (bit-identical to the legacy
// forward forced to Recurrent) and an explicit Chunked request is Error(Unsupported) before
// any state changes.
TEST_F(BackendLimits, BackendWithoutChunkedFormRunsRecurrent) {
    hb::Limits nochunk = full_;
    nochunk.gdn_chunked = false;
    LimitedBackend be(*cpu_, nochunk);
    const Qwen35 nw(*nm_, be);
    const LegacyQwen35 lg(*nm_, nullptr);
    KvPool pn(nw.kv_layout(), 8), pl(lg.kv_layout(), 8), mn(nw.mtp_kv_layout(), 1), ml(lg.mtp_kv_layout(), 1);
    TestSeq qn(pn, mn, nw.gdn_shape(), 1), ql(pl, ml, lg.gdn_shape(), 1);
    Tokens gen(nw.n_vocab());
    const auto t = gen.next(12);
    const std::vector<std::size_t> rows{11};
    StepResult rn, rl;
    nw.forward(std::vector{seq_step(qn, t, rows, LogitsMode::Full)}, rn);
    lg.forward(std::vector{seq_step(ql, t, rows, LogitsMode::Full, false, 0, GdnPath::Recurrent)}, rl);
    ASSERT_EQ(rn.gdn_paths, std::vector<GdnPath>{GdnPath::Recurrent});
    EXPECT_TRUE(same_bits(rn.seqs[0].logits, rl.seqs[0].logits, "logits"));
    for (std::size_t layer = 0; layer < qn.gdn.shape().n_layers; ++layer) {
        EXPECT_TRUE(same_bits(qn.gdn.recurrent(layer), ql.gdn.recurrent(layer), "recurrent " + std::to_string(layer)));
    }

    const std::vector<float> before(qn.gdn.recurrent(0).begin(), qn.gdn.recurrent(0).end());
    try {
        nw.forward(std::vector{seq_step(qn, t, rows, LogitsMode::Argmax, false, 0, GdnPath::Chunked)}, rn);
        ADD_FAILURE() << "explicit Chunked on a backend without it must throw";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
    }
    EXPECT_EQ(qn.kv.length(), 12u);
    EXPECT_TRUE(same_bits(qn.gdn.recurrent(0), before, "state after the rejected call"));
}

}  // namespace
