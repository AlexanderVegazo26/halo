// WS-BI-3 (ADR-001 §6.3): the whole qwen35 forward (models::Qwen35 over Backend&) on the HIP
// backend in host emulation vs the same forward on the CPU backend, on the tiny model.
//
// Two sides, each with its own KV pools (sentinel-filled), sequences and GDN states, run the
// same operation sequence; after every call the test compares every result (argmax index and
// value, full logits rows, hidden rows, per-layer inputs, resolved GDN paths, StepCost), the
// WHOLE storage of both KV pools, every block table and length, and every GDN layer's live
// recurrent and conv state and all slots.
//
//  (a) HipEmulation with the bitwise profile (hip_bitwise_defaults: gemv_generic_b64,
//      attn_exact_b128, every other default already bitwise): everything bit for bit, in both
//      emulation orders. A mismatch is a finding (A-2), never a tolerance to widen.
//  (b) HipEmulation with the stock defaults (wave GEMV, online attention): per-layer relative L2
//      error <= tau(l), fixed below BEFORE the first measurement, and argmax index equality on
//      every logit row whose CPU top-2 gap is decisive. Also non-vacuity: some compared value
//      must differ bitwise (the defaults really compute differently).
//
// tau (derived, first order; docs/hip.md "GEMV reduction orders"): a wave GEMV over K inputs
// differs from the CPU by at most (d_cpu + d_wave) u sum_i |x_i w_i| per output, d_cpu =
// K/8 + 3 + K mod 8 + 1, d_wave = ceil(K/32) + 5, u = 2^-24. Relative to |y| the factor
// sum|x w| / |y| is taken as sqrt(K) (random-sign sums; the tiny model's weights are random).
// The tiny model's largest K is 512 (n_ff), so per GEMV eps = 89 u sqrt(512) = 1.2e-4. With the
// assumption that no layer amplifies a relative input error by more than 2 (RMS-normalised
// residual blocks), the error after n GEMVs is at most 2 n eps: tau(l) = 2 * eps * (GEMVs
// before layer l's input). The online-attention term (u (4N + ...) at N <= 100 keys) is below
// 1e-5 and is covered by the factor 2. This is an estimate under stated assumptions, not a
// rigorous bound; it cannot detect 1-ulp-scale errors (those are what (a) and the per-op bounds
// in test_hip_backend.cpp detect).

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/backend/hip_backend.h"
#include "halo/core/error.h"
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
using halo::test::TestSeq;
namespace hb = halo::backend;

namespace {

constexpr std::size_t kSlots = 4;
constexpr std::size_t kBlocks = 64;
constexpr std::size_t kSeqs = 5;
constexpr float kSentinel = -7.25f;
constexpr double kU = 5.9604644775390625e-08;
const std::set<halo::DType> kHipTypes{halo::DType::F32,  halo::DType::F16,  halo::DType::Q4_0,  halo::DType::Q8_0,
                                      halo::DType::Q3_K, halo::DType::Q4_K, halo::DType::Q5_K,  halo::DType::Q6_K,
                                      halo::DType::IQ3_S, halo::DType::IQ4_NL, halo::DType::IQ4_XS};

/// Per-GEMV relative error estimate at the tiny model's largest K (see the file comment).
double eps_gemv() {
    constexpr double K = 512;
    const double d = (512 / 8 + 3 + 512 % 8 + 1) + (512 + 31) / 32 + 5;
    return d * kU * std::sqrt(K);
}

std::span<const float> pool_storage(const KvPool& p) { return {p.k_rows(0, 0), p.total_blocks() * p.layout().block_floats()}; }

/// How the two sides are compared.
struct Mode {
    bool bitwise = true;
    double tau = 0;  // relative L2 bound for (b); per-layer bounds scale it
    // (b) statistics
    std::size_t differing_values = 0;
    std::size_t argmax_checked = 0;
    double min_gap_over_diff = 1e300;  // min over Full rows of (CPU top-2 gap) / max |logit difference|
    double worst_ratio = 0;  // max rel-L2 / bound
};

::testing::AssertionResult compare(Mode& m, std::span<const float> hip, std::span<const float> cpu, double bound,
                                   const std::string& what) {
    if (hip.size() != cpu.size()) return ::testing::AssertionFailure() << what << ": " << hip.size() << " vs " << cpu.size();
    std::size_t diff = 0, first = 0;
    double num = 0, den = 0;
    for (std::size_t i = 0; i < hip.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(hip[i]) != std::bit_cast<std::uint32_t>(cpu[i])) {
            if (diff == 0) first = i;
            ++diff;
        }
        const double d = double(hip[i]) - double(cpu[i]);
        num += d * d;
        den += double(cpu[i]) * double(cpu[i]);
    }
    if (m.bitwise) {
        if (diff == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << what << ": " << diff << " of " << hip.size() << " differ; first at " << first
                                             << " (hip " << hip[first] << ", cpu " << cpu[first] << ")";
    }
    m.differing_values += diff;
    if (diff == 0) return ::testing::AssertionSuccess();
    // Values that are not finite on either side must match exactly (sentinels, NaN poison).
    for (std::size_t i = 0; i < hip.size(); ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(hip[i])) {
            if (std::bit_cast<std::uint32_t>(hip[i]) != std::bit_cast<std::uint32_t>(cpu[i])) {
                return ::testing::AssertionFailure() << what << ": non-finite mismatch at " << i;
            }
        }
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    m.worst_ratio = std::max(m.worst_ratio, rel / bound);
    if (rel <= bound) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << what << ": relative L2 " << rel << " > bound " << bound;
}

using ForwardFn = std::function<void(std::span<const SeqStep>, StepResult&, const ForwardOptions&)>;
using MtpFn = std::function<void(std::span<const MtpStep>, StepResult&)>;

struct Side {
    const Qwen35* model;
    std::unique_ptr<KvPool> kv_pool, mtp_pool;
    std::vector<std::unique_ptr<TestSeq>> seqs;
    std::map<std::string, StepResult> saved;

    explicit Side(const Qwen35& m)
        : model(&m), kv_pool(std::make_unique<KvPool>(m.kv_layout(), kBlocks)), mtp_pool(std::make_unique<KvPool>(m.mtp_kv_layout(), kBlocks)) {
        for (KvPool* p : {kv_pool.get(), mtp_pool.get()}) {
            std::fill_n(p->k_rows(0, 0), p->total_blocks() * p->layout().block_floats(), kSentinel);
        }
        for (std::size_t i = 0; i < kSeqs; ++i) seqs.push_back(std::make_unique<TestSeq>(*kv_pool, *mtp_pool, m.gdn_shape(), kSlots));
    }
    TestSeq& seq(std::size_t i) { return *seqs.at(i); }
};

/// The number of GEMVs before layer l's input (layer 0 = embeddings; see the tau comment).
std::size_t gemvs_before(const Qwen35& m, std::size_t layer) {
    const auto& hp = m.hparams();
    std::size_t n = 0;
    for (std::size_t l = 0; l < layer && l < hp.n_layer; ++l) {
        const bool attn = hp.layer_kind[l] == halo::model::LayerKind::FullAttention;
        n += attn ? 7 : 8;
    }
    return n;
}

class Diff {
public:
    Diff(const Qwen35& hip, const Qwen35& cpu, Mode& mode) : h_(hip), c_(cpu), mode_(mode), E_(cpu.n_embd()) {}

    template <class Make>
    void forward(const std::string& what, Make&& make, const ForwardOptions& opts = {}, const std::string& save = {}) {
        SCOPED_TRACE(what);
        const std::vector<SeqStep> sh = make(h_);
        const std::vector<SeqStep> sc = make(c_);
        StepResult rh, rc;
        h_.model->forward(sh, rh, opts);
        c_.model->forward(sc, rc, opts);
        compare_results(rh, rc, final_bound());
        compare_state();
        if (!save.empty()) {
            h_.saved[save] = std::move(rh);
            c_.saved[save] = std::move(rc);
        }
    }

    template <class Make>
    void mtp(const std::string& what, Make&& make, const std::string& save = {}) {
        SCOPED_TRACE(what);
        const std::vector<MtpStep> sh = make(h_);
        const std::vector<MtpStep> sc = make(c_);
        StepResult rh, rc;
        h_.model->mtp_forward(sh, rh);
        c_.model->mtp_forward(sc, rc);
        compare_results(rh, rc, final_bound() + 2 * 8 * eps_gemv());  // + eh_proj and one attention block
        compare_state();
        if (!save.empty()) {
            h_.saved[save] = std::move(rh);
            c_.saved[save] = std::move(rc);
        }
    }

    template <class Op>
    void both(const std::string& what, Op&& op) {
        SCOPED_TRACE(what);
        op(h_);
        op(c_);
        compare_state();
    }

    [[nodiscard]] std::size_t n_embd() const noexcept { return E_; }
    [[nodiscard]] Side& hip_side() noexcept { return h_; }

private:
    [[nodiscard]] double final_bound() const {
        return 2 * eps_gemv() * static_cast<double>(gemvs_before(*c_.model, c_.model->hparams().n_layer) + 1);
    }

    void compare_results(const StepResult& h, const StepResult& c, double bound) {
        ASSERT_EQ(h.seqs.size(), c.seqs.size());
        for (std::size_t i = 0; i < h.seqs.size(); ++i) {
            const auto& a = h.seqs[i];
            const auto& b = c.seqs[i];
            const std::string s = "seq " + std::to_string(i);
            ASSERT_EQ(a.argmax.size(), b.argmax.size()) << s;
            const std::size_t V = c_.model->n_vocab();
            for (std::size_t j = 0; j < a.argmax.size(); ++j) {
                if (mode_.bitwise) {
                    EXPECT_EQ(a.argmax[j].index, b.argmax[j].index) << s << " logit row " << j;
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(a.argmax[j].value), std::bit_cast<std::uint32_t>(b.argmax[j].value))
                        << s << " logit row " << j;
                    continue;
                }
                // (b), ADR §6.3: argmax token equality along the whole trajectory. Emulation is
                // deterministic, so this cannot flake between runs; how close it is to flipping is
                // reported as the smallest CPU top-2 gap over the largest logit difference.
                ++mode_.argmax_checked;
                EXPECT_EQ(a.argmax[j].index, b.argmax[j].index) << s << " logit row " << j;
                if (b.logits.size() < (j + 1) * V || a.logits.size() < (j + 1) * V) continue;
                std::vector<float> row(b.logits.begin() + static_cast<std::ptrdiff_t>(j * V),
                                       b.logits.begin() + static_cast<std::ptrdiff_t>((j + 1) * V));
                double dmax = 0;
                for (std::size_t n = 0; n < V; ++n) dmax = std::max(dmax, std::fabs(double(a.logits[j * V + n]) - double(row[n])));
                std::partial_sort(row.begin(), row.begin() + 2, row.end(), std::greater<>());
                const double gap = double(row[0]) - double(row[1]);
                mode_.min_gap_over_diff = std::min(mode_.min_gap_over_diff, dmax > 0 ? gap / dmax : 1e300);
            }
            EXPECT_TRUE(compare(mode_, a.logits, b.logits, bound, "logits of " + s));
            EXPECT_TRUE(compare(mode_, a.hidden, b.hidden, bound, "hidden of " + s));
        }
        EXPECT_EQ(h.gdn_paths, c.gdn_paths);
        ASSERT_EQ(h.layer_inputs.size(), c.layer_inputs.size());
        for (std::size_t li = 0; li < h.layer_inputs.size(); ++li) {
            const double lb = 2 * eps_gemv() * static_cast<double>(std::max<std::size_t>(1, gemvs_before(*c_.model, li)));
            EXPECT_TRUE(compare(mode_, h.layer_inputs[li], c.layer_inputs[li], lb, "layer input " + std::to_string(li)));
        }
        // The cost model is backend-independent.
        EXPECT_EQ(h.cost.weight_bytes, c.cost.weight_bytes);
        EXPECT_EQ(h.cost.weight_passes, c.cost.weight_passes);
        EXPECT_EQ(h.cost.embedding_bytes, c.cost.embedding_bytes);
        EXPECT_EQ(h.cost.activation_bytes, c.cost.activation_bytes);
        EXPECT_EQ(h.cost.state_bytes, c.cost.state_bytes);
        EXPECT_EQ(h.cost.kv_bytes, c.cost.kv_bytes);
    }

    void compare_state() {
        const double b = final_bound();
        EXPECT_TRUE(compare(mode_, pool_storage(*h_.kv_pool), pool_storage(*c_.kv_pool), b, "trunk KV pool storage"));
        EXPECT_TRUE(compare(mode_, pool_storage(*h_.mtp_pool), pool_storage(*c_.mtp_pool), b, "MTP KV pool storage"));
        for (std::size_t i = 0; i < kSeqs; ++i) {
            TestSeq& a = h_.seq(i);
            TestSeq& c = c_.seq(i);
            const std::string s = "seq " + std::to_string(i);
            EXPECT_EQ(a.kv.length(), c.kv.length()) << s;
            EXPECT_TRUE(std::ranges::equal(a.kv.blocks(), c.kv.blocks())) << s << " block table";
            EXPECT_EQ(a.mtp_kv.length(), c.mtp_kv.length()) << s;
            EXPECT_TRUE(std::ranges::equal(a.mtp_kv.blocks(), c.mtp_kv.blocks())) << s << " MTP block table";
            EXPECT_EQ(a.gdn.slot_rows(), c.gdn.slot_rows()) << s;
            EXPECT_EQ(a.gdn.slots_valid(), c.gdn.slots_valid()) << s;
            EXPECT_EQ(a.gdn.live(), c.gdn.live()) << s << " ring live";
            // The whole ring slab (ADR-001 §5.3): live state + every physical slot.
            EXPECT_TRUE(compare(mode_, a.gdn.slab(), c.gdn.slab(), b, s + " GDN ring slab"));
            for (std::size_t layer = 0; layer < a.gdn.shape().n_layers; ++layer) {
                const std::string sl = s + " GDN layer " + std::to_string(layer);
                EXPECT_TRUE(compare(mode_, a.gdn.recurrent(layer), c.gdn.recurrent(layer), b, sl + " live recurrent"));
                const auto ca = a.gdn.conv(layer);
                const auto cc = c.gdn.conv(layer);
                EXPECT_TRUE(compare(mode_, {ca.data(), ca.rows() * ca.cols()}, {cc.data(), cc.rows() * cc.cols()}, b, sl + " live conv"));
            }
        }
    }

    Side h_, c_;
    Mode& mode_;
    std::size_t E_;
};

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

#define DIFF_STEP(x)                               \
    do {                                           \
        x;                                         \
        if (::testing::Test::HasFailure()) return; \
    } while (false)

/// The ADR §6.2 operation sequence (as the WS-BI-1 gate), with few logit rows: the emulated
/// LM head reads all 248,320 vocabulary rows per requested row.
void run_scenario(Diff& g, std::size_t n_vocab, bool has_mtp) {
    Tokens gen(n_vocab);
    const std::size_t E = g.n_embd();
    const std::vector<std::size_t> row0{0};
    const std::vector<std::size_t> none{};

    // 1. S = 1 prefill of 70 rows: crosses the GDN chunk (64 / 8) and four KV blocks of 16.
    const auto p0 = gen.next(70);
    const std::vector<std::size_t> r0{69, 37};
    DIFF_STEP(g.forward(
        "prefill S=1, 70 rows, Full", [&](Side& s) { return std::vector{seq_step(s.seq(0), p0, r0, LogitsMode::Full, true)}; },
        {true}, "p0"));

    // 2. Decode, Argmax.
    const auto d0 = gen.next(1);
    DIFF_STEP(g.forward("decode S=1, Argmax", [&](Side& s) { return std::vector{seq_step(s.seq(0), d0, row0, LogitsMode::Argmax)}; }));

    // 3. Ragged S = 3: decode (Full) + chunked prefill of 19 rows (hidden) + one-row prefill (None).
    const auto d1 = gen.next(1);
    const auto p1 = gen.next(19);
    const auto p2 = gen.next(1);
    DIFF_STEP(g.forward(
        "ragged S=3: decode + prefill 19 + prefill 1",
        [&](Side& s) {
            return std::vector{seq_step(s.seq(0), d1, row0, LogitsMode::Full), seq_step(s.seq(1), p1, none, LogitsMode::None, true),
                               seq_step(s.seq(2), p2, none, LogitsMode::None, true)};
        },
        {true}, "ragged"));

    // 4. Explicit GDN paths: recurrent multi-row without slots, chunked, and Auto.
    const auto a1 = gen.next(3);
    const auto a2 = gen.next(5);
    const auto a3 = gen.next(9);
    DIFF_STEP(g.forward("explicit paths: Recurrent 3 + Chunked 5 + Auto 9", [&](Side& s) {
        return std::vector{seq_step(s.seq(1), a1, none, LogitsMode::None, false, 0, GdnPath::Recurrent),
                           seq_step(s.seq(2), a2, none, LogitsMode::None, false, 0, GdnPath::Chunked),
                           seq_step(s.seq(3), a3, none, LogitsMode::None)};
    }));

    // 5. Verify K = 1..4 with slots, then commit_rows_kept for every m; in the same call a
    //    decode and a 6-row chunked prefill of other sequences. One logit row per call.
    for (std::size_t K = 1; K <= kSlots; ++K) {
        for (std::size_t m = 1; m <= K; ++m) {
            const std::string tag = "verify K=" + std::to_string(K) + " m=" + std::to_string(m);
            const auto v = gen.next(K);
            const auto dd = gen.next(1);
            const auto pp = gen.next(6);
            const std::vector<std::size_t> last{K - 1};
            const std::size_t len0 = g.hip_side().seq(0).kv.length();
            const bool full = K == 4 && m == 2;
            DIFF_STEP(g.forward(tag, [&](Side& s) {
                return std::vector{seq_step(s.seq(0), v, full ? last : none, full ? LogitsMode::Full : LogitsMode::None, true, K),
                                   seq_step(s.seq(1), dd, none, LogitsMode::None), seq_step(s.seq(2), pp, none, LogitsMode::None)};
            }));
            DIFF_STEP(g.both(tag + " commit", [&](Side& s) {
                s.seq(0).kv.truncate(len0 + m);
                s.seq(0).gdn.commit_rows_kept(K, m);
            }));
        }
    }

    // 6. Copy-on-write shared prefix, then both sequences write into the shared block.
    DIFF_STEP(g.both("share prefix", [&](Side& s) {
        s.seq(4).kv.share_prefix(s.seq(0).kv, s.seq(0).kv.length() - 5);
        s.seq(4).gdn.copy_from(s.seq(0).gdn);
    }));
    const auto c4 = gen.next(4);
    const auto c0 = gen.next(1);
    const std::vector<std::size_t> rc4{3};
    DIFF_STEP(g.forward("COW: shared-prefix sequence + its source in one call", [&](Side& s) {
        return std::vector{seq_step(s.seq(4), c4, rc4, LogitsMode::Full), seq_step(s.seq(0), c0, none, LogitsMode::None)};
    }));

    if (!has_mtp) return;
    // 7. MTP catch-up (S = 2) on the trunk hidden rows, then a 2-deep draft chain drafted from
    //    each side's own argmax.
    const std::span<const std::int32_t> m0_tokens = std::span(p0).subspan(1);
    const std::span<const std::int32_t> m1_tokens = std::span(p1).subspan(1);
    const std::vector<std::size_t> rm0{68};
    DIFF_STEP(g.mtp(
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
            b.logits = LogitsMode::None;
            return std::vector{a, b};
        },
        "m0"));
    std::int32_t pos = 1 + static_cast<std::int32_t>(m0_tokens.size());
    const char* prev = "m0";
    const char* depth_tags[] = {"m1", "m2"};
    for (std::size_t depth = 0; depth < 2; ++depth) {
        std::map<const Side*, std::int32_t> tok;
        DIFF_STEP(g.mtp(
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
                a.logits = depth == 0 ? LogitsMode::Full : LogitsMode::None;
                a.want_hidden = true;
                return std::vector{a};
            },
            depth_tags[depth]));
        prev = depth_tags[depth];
        ++pos;
    }
}

struct Param {
    const char* file;
    std::size_t chunk;
    bool bitwise;
    bool reverse;
};

std::string param_name(const ::testing::TestParamInfo<Param>& info) {
    const Param& p = info.param;
    std::string f = p.file;
    f = f.substr(5, f.size() - 10);
    return f + "_chunk" + std::to_string(p.chunk) + (p.bitwise ? "_bitwise" : "_bounded") + (p.reverse ? "_reverse" : "_forward");
}

class HipForward : public ::testing::TestWithParam<Param> {};

TEST_P(HipForward, MatchesTheCpuBackend) {
    const Param& p = GetParam();
    const auto path = halo::test::tiny_dir() / p.file;
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " missing";
    const auto nm = halo::model::NormalizedModel::load(path);
    hb::HipBackendOptions o;
    o.emulation_reverse_order = p.reverse;
    if (p.bitwise) o.defaults = hb::hip_bitwise_defaults();
    const auto hip = hb::make_hip_backend(o);
    ASSERT_EQ(hip->kind(), hb::Kind::HipEmulation);
    const auto cpu = hb::make_cpu_backend(nullptr);
    const Qwen35Options opts{p.chunk};
    const Qwen35 mh(nm, *hip, opts);
    const Qwen35 mc(nm, *cpu, opts);
    ASSERT_EQ(&mh.backend(), hip.get());
    if (std::string_view(p.file) == "tiny-f32.gguf") ASSERT_TRUE(mh.has_mtp()) << "the MTP part of the scenario must run";
    // A matrix type without a HIP kernel (docs/hip.md: F32, F16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K,
    // Q6_K, IQ3_S, IQ4_NL, IQ4_XS) must fail the first forward with Error(Unsupported), never
    // silently: the file is covered by that assertion instead of the differential.
    std::string unsupported;
    for (const auto& t : nm.gguf().tensors()) {
        if (t.n_dims < 2 || t.name.find("ssm_conv1d") != std::string::npos) continue;  // dequantized on the host at load
        if (!kHipTypes.contains(t.type)) unsupported = t.name + " (" + std::string(halo::traits(t.type).name) + ")";
    }
    Mode mode;
    mode.bitwise = p.bitwise;
    Diff g(mh, mc, mode);
    if (!unsupported.empty()) {
        try {
            run_scenario(g, mh.n_vocab(), mh.has_mtp());
            ADD_FAILURE() << "the forward ran although " << unsupported << " has no HIP kernel";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported) << e.what();
            std::printf("    %s: %s has no HIP kernel -> Error(Unsupported) at the first forward (%s)\n", p.file, unsupported.c_str(),
                        e.what());
        }
        return;
    }
    run_scenario(g, mh.n_vocab(), mh.has_mtp());
    if (!p.bitwise) {
        std::printf("    (b) differing values %zu, worst rel-L2/bound %.3g, argmax rows %zu, min top-2 gap / max |d logit| %.3g\n",
                    mode.differing_values, mode.worst_ratio, mode.argmax_checked, mode.min_gap_over_diff);
        EXPECT_GT(mode.differing_values, 0u) << "the default profile must compute differently from the CPU (non-vacuity)";
        EXPECT_GT(mode.argmax_checked, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Tiny, HipForward,
    ::testing::Values(Param{"tiny-f32.gguf", 64, true, false}, Param{"tiny-q8_0.gguf", 64, true, false},
                      Param{"tiny-q4_k_m.gguf", 64, true, false}, Param{"tiny-q6_k.gguf", 64, true, false},
                      Param{"tiny-iq4_xs.gguf", 64, true, false}, Param{"tiny-q3_k_m.gguf", 64, true, false},
                      Param{"tiny-q5_k_m.gguf", 64, true, false}, Param{"tiny-q4_0.gguf", 64, true, false},
                      Param{"tiny-f32.gguf", 8, true, true}, Param{"tiny-q6_k.gguf", 64, true, true},
                      Param{"tiny-f32.gguf", 64, false, false}, Param{"tiny-q6_k.gguf", 64, false, true}),
    param_name);

// Data errors surface through the forward exactly as on the CPU backend (Error(Kernel)).
TEST(HipForwardErrors, NoForwardStartsOnAnUnsupportedChunk) {
    const auto path = halo::test::tiny_dir() / "tiny-f32.gguf";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " missing";
    const auto nm = halo::model::NormalizedModel::load(path);
    const auto hip = hb::make_hip_backend();
    // The HIP chunked GDN caps the chunk at 64 (LDS tiles): construction fails early, as ADR §5.2 wants.
    try {
        const Qwen35 m(nm, *hip, Qwen35Options{65});
        ADD_FAILURE() << "chunk 65 accepted";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported) << e.what();
    }
    EXPECT_NO_THROW({ const Qwen35 m(nm, *hip, Qwen35Options{64}); });
}

}  // namespace
