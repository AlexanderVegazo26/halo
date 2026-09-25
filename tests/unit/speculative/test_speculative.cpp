// Milestone 2: MTP speculative decoding on the tiny qwen35 model (D-005, D-012, FR-009).
//
// On this random tiny model the real MTP almost never agrees with the trunk (golden
// mtp_argmax vs trunk argmax: 0 of 149 rows for p0, 0 of 3 for p1), so real drafting only
// exercises rejection. Acceptance of every depth is driven by *oracle* drafts (forced_drafts
// = the true greedy continuation with one wrong token at a chosen depth).
//
// Equality is always bitwise: tokens, trunk KV rows, GDN recurrent + conv state, the
// carried hidden, and the MTP KV after flushing the catch-up queue of both sides.

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "../models/tiny_golden.h"
#include "halo/backends/cpu/ops.h"
#include "halo/core/error.h"
#include "halo/speculative/speculative.h"

using halo::speculative::GateConfig;
using halo::speculative::GateMode;
using halo::speculative::ProfitGate;
using halo::speculative::SpecConfig;
using halo::speculative::Speculator;
using halo::speculative::StepRequest;
using halo::speculative::Tick;
using halo::state::SequenceState;
using halo::test::Golden;
using halo::test::TinyModel;

namespace {

using Toks = std::vector<std::int32_t>;

class Spec : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        golden_ = Golden::load();
        tiny_ = TinyModel::load("tiny-f32.gguf", 1024);
    }
    static void TearDownTestSuite() { tiny_.reset(); }
    void SetUp() override {
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        if (!tiny_) GTEST_SKIP() << "tiny-f32.gguf missing under " << halo::test::tiny_dir();
    }

    static const halo::models::Qwen35& model() { return *tiny_->model; }
    static std::unique_ptr<SequenceState> seq(std::size_t slots = 4, halo::kv_cache::KvPool* trunk = nullptr) {
        return std::make_unique<SequenceState>(trunk ? *trunk : *tiny_->kv_pool, tiny_->mtp_pool.get(), model().gdn_shape(), slots);
    }
    static Speculator spec(GateMode mode = GateMode::Always, std::size_t max_draft = 3) {
        SpecConfig c;
        c.max_draft = max_draft;
        c.gate.mode = mode;
        return Speculator(model(), c);
    }
    static Toks prompt(const char* p) { return golden_->i32(std::string(p) + ".tokens"); }

    static inline std::optional<Golden> golden_;
    static inline std::unique_ptr<TinyModel> tiny_;
};

/// One-request tick.
Tick step1(Speculator& sp, const StepRequest& r) {
    Tick t;
    sp.step(std::span(&r, 1), t);
    return t;
}

Toks prefill(Speculator& sp, SequenceState& s, std::span<const std::int32_t> p) {
    StepRequest r;
    r.seq = &s;
    r.prefill = p;
    return step1(sp, r).out[0].tokens;
}

/// Greedy generation of n tokens (the first from the prefill), k drafts per step.
Toks generate(Speculator& sp, SequenceState& s, std::span<const std::int32_t> p, std::size_t n, std::size_t k) {
    Toks out = prefill(sp, s, p);
    while (out.size() < n) {
        StepRequest r;
        r.seq = &s;
        r.token = out.back();
        r.max_draft = k;
        r.max_emit = n - out.size();
        const Tick t = step1(sp, r);
        EXPECT_FALSE(t.out[0].tokens.empty());
        out.insert(out.end(), t.out[0].tokens.begin(), t.out[0].tokens.end());
    }
    return out;
}

/// Plain decode of `toks` one at a time (reference history), returns the argmaxes.
Toks feed_plain(Speculator& sp, SequenceState& s, std::span<const std::int32_t> toks) {
    Toks out;
    for (const std::int32_t t : toks) {
        StepRequest r;
        r.seq = &s;
        r.token = t;
        out.push_back(step1(sp, r).out[0].tokens.at(0));
    }
    return out;
}

std::vector<float> dump_kv(const halo::kv_cache::SequenceKv& kv) {
    std::vector<float> out;
    const auto& l = kv.pool().layout();
    std::vector<const float*> t;
    for (std::size_t layer = 0; layer < l.n_layers; ++layer) {
        for (const bool k : {true, false}) {
            const auto v = k ? kv.keys(layer, kv.length(), t) : kv.values(layer, kv.length(), t);
            for (std::size_t r = 0; r < kv.length(); ++r) out.insert(out.end(), v.row_ptr(r), v.row_ptr(r) + l.kv_dim);
        }
    }
    return out;
}

/// Bitwise equality of two sequences' complete state (flushes both MTP queues first).
void expect_same_state(Speculator& sp, SequenceState& a, SequenceState& b) {
    SequenceState* both[] = {&a, &b};
    sp.flush_mtp(both);
    EXPECT_EQ(a.length(), b.length());
    EXPECT_TRUE(dump_kv(a.kv) == dump_kv(b.kv)) << "trunk KV rows differ";
    EXPECT_TRUE(a.gdn.snapshot().data == b.gdn.snapshot().data) << "GDN recurrent/conv state differs";
    EXPECT_TRUE(a.last_hidden == b.last_hidden) << "carried trunk hidden (pending_h) differs";
    ASSERT_TRUE(a.mtp_kv && b.mtp_kv);
    EXPECT_EQ(a.mtp_kv->length(), b.mtp_kv->length());
    EXPECT_EQ(a.mtp_kv->length() + 1, a.length()) << "MTP KV must hold positions 1..L-1 after a flush";
    EXPECT_TRUE(dump_kv(*a.mtp_kv) == dump_kv(*b.mtp_kv)) << "MTP KV rows differ";
}

/// Oracle drafts: the true continuation g[m+1 .. m+k] with a wrong token at depth `wrong`
/// (wrong == k: all correct).
Toks oracle(const Toks& g, std::size_t m, std::size_t k, std::size_t wrong, std::size_t n_vocab) {
    Toks d(g.begin() + static_cast<std::ptrdiff_t>(m + 1), g.begin() + static_cast<std::ptrdiff_t>(m + 1 + k));
    if (wrong < k) d[wrong] = static_cast<std::int32_t>((static_cast<std::size_t>(d[wrong]) + 1) % n_vocab);
    return d;
}

}  // namespace

// ---- the key invariant: greedy MTP ON == MTP OFF ----------------------------------------

// One instance per prompt keeps each ctest well under the default timeout under ASan
// (the three prompts in one test took ~610 s there).
class SpecPrompt : public Spec, public ::testing::WithParamInterface<const char*> {};

TEST_P(SpecPrompt, GreedyWithRealMtpDraftsEqualsPlainGreedyAndGolden) {
    constexpr std::size_t kN = 16;
    {
        const char* p = GetParam();
        SCOPED_TRACE(p);
        const Toks pr = prompt(p);
        auto off = spec(GateMode::Off);
        auto so = seq();
        const Toks plain = generate(off, *so, pr, kN, 3);
        EXPECT_EQ(off.metrics().spec_steps, 0u);
        const Toks gold = golden_->i32(std::string(p) + ".decode_tokens");
        EXPECT_TRUE(std::equal(gold.begin(), gold.begin() + static_cast<std::ptrdiff_t>(std::min(gold.size(), kN)), plain.begin()))
            << "plain greedy must reproduce the transformers greedy decode";
        for (std::size_t k = 1; k <= 3; ++k) {
            SCOPED_TRACE(k);
            auto on = spec(GateMode::Always);
            auto s = seq();
            EXPECT_EQ(generate(on, *s, pr, kN, k), plain);
            const auto& mt = on.metrics();
            if (pr.size() > 0) EXPECT_GT(mt.spec_steps, 0u) << "MTP drafting never ran";
            std::printf("[%s k=%zu] spec steps %llu, drafted %llu, accepted %llu\n", p, k,
                        static_cast<unsigned long long>(mt.spec_steps), static_cast<unsigned long long>(mt.drafted),
                        static_cast<unsigned long long>(mt.accepted));
            expect_same_state(on, *s, *so);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Prompts, SpecPrompt, ::testing::Values("p0", "p1", "p2"));

TEST_F(Spec, OracleDraftsAcceptEveryDepthAndStayIdenticalToPlain) {
    constexpr std::size_t kK = 3, kN = 24;
    const Toks pr = prompt("p1");
    auto off = spec(GateMode::Off);
    auto ref = seq();
    const Toks g = generate(off, *ref, pr, kN + kK + 1, 0);
    auto sp = spec(GateMode::Always);
    auto s = seq();
    Toks out = prefill(sp, *s, pr);
    std::vector<std::size_t> seen(kK + 1, 0);
    for (std::size_t step = 0; out.size() < kN; ++step) {
        const std::size_t wrong = step % (kK + 1);
        const Toks d = oracle(g, out.size() - 1, kK, wrong, model().n_vocab());
        StepRequest r;
        r.seq = s.get();
        r.token = out.back();
        r.forced_drafts = d;
        const Tick t = step1(sp, r);
        EXPECT_EQ(t.out[0].accepted, wrong) << "step " << step;
        EXPECT_EQ(t.out[0].tokens.size(), wrong + 1);
        ++seen[t.out[0].accepted];
        out.insert(out.end(), t.out[0].tokens.begin(), t.out[0].tokens.end());
    }
    EXPECT_TRUE(std::equal(out.begin(), out.end(), g.begin())) << "emitted tokens must be the plain greedy sequence";
    for (std::size_t a = 0; a <= kK; ++a) EXPECT_GT(seen[a], 0u) << "acceptance " << a << " never exercised";
    // Reference state: plain decode of exactly the tokens the spec path fed.
    auto plain = seq();
    auto off2 = spec(GateMode::Off);
    (void)prefill(off2, *plain, pr);
    (void)feed_plain(off2, *plain, std::span(out).first(out.size() - 1));
    expect_same_state(sp, *s, *plain);
}

// ---- D-012: verify + rollback to r == decode r ------------------------------------------

TEST_F(Spec, VerifyThenRollbackToRIsDecodeR) {
    constexpr std::size_t kK = 3;
    const Toks pr = prompt("p0");  // 150 tokens: history crosses KV blocks and GDN chunks
    auto off = spec(GateMode::Off);
    auto gs = seq();
    const Toks g = generate(off, *gs, pr, 12, 0);
    const std::size_t m = 3;  // pending token g[m], history = prompt + g[0..m)
    for (std::size_t r = 1; r <= kK + 1; ++r) {
        SCOPED_TRACE(r);
        auto sp = spec(GateMode::Always);
        auto s = seq();
        (void)prefill(sp, *s, pr);
        (void)feed_plain(sp, *s, std::span(g).first(m));
        const Toks d = oracle(g, m, kK, kK, model().n_vocab());  // all correct
        StepRequest q;
        q.seq = s.get();
        q.token = g[m];
        q.forced_drafts = d;
        Tick t;
        sp.draft(std::span(&q, 1), t);
        sp.verify(std::span(&q, 1), t);
        ASSERT_EQ(t.out[0].accepted, kK);
        ASSERT_EQ(t.out[0].n_keep, kK + 1);
        const std::size_t bad[] = {kK + 2};
        EXPECT_THROW(sp.commit(std::span(&q, 1), t, bad), halo::Error) << "cannot keep more than was accepted";
        const std::size_t zero[] = {0};
        EXPECT_THROW(sp.commit(std::span(&q, 1), t, zero), halo::Error);
        const std::size_t keep[] = {r};
        sp.commit(std::span(&q, 1), t, keep);
        EXPECT_EQ(t.out[0].tokens, Toks(g.begin() + static_cast<std::ptrdiff_t>(m + 1), g.begin() + static_cast<std::ptrdiff_t>(m + 1 + r)));
        auto ref = seq();
        (void)prefill(off, *ref, pr);
        (void)feed_plain(off, *ref, std::span(g).first(m + r));
        expect_same_state(sp, *s, *ref);
        // Both continue identically.
        StepRequest c1;
        c1.seq = s.get();
        c1.token = g[m + r];
        c1.max_draft = 2;
        StepRequest c2 = c1;
        c2.seq = ref.get();
        EXPECT_EQ(step1(sp, c1).out[0].tokens.at(0), g[m + r + 1]);
        EXPECT_EQ(step1(off, c2).out[0].tokens.at(0), g[m + r + 1]);
    }
}

TEST_F(Spec, MaxEmitAndStopTokensCapAcceptanceConsistently) {
    constexpr std::size_t kK = 3;
    const Toks pr = prompt("p1");
    auto off = spec(GateMode::Off);
    auto gs = seq();
    const Toks g = generate(off, *gs, pr, 10, 0);
    for (const bool use_stop : {false, true}) {
        SCOPED_TRACE(use_stop);
        auto sp = spec(GateMode::Always);
        auto s = seq();
        (void)prefill(sp, *s, pr);
        const Toks d = oracle(g, 0, kK, kK, model().n_vocab());
        StepRequest q;
        q.seq = s.get();
        q.token = g[0];
        q.forced_drafts = d;
        const std::int32_t stop[] = {g[2]};
        if (use_stop) q.stop_tokens = stop; else q.max_emit = 2;
        const Tick t = step1(sp, q);
        EXPECT_EQ(t.out[0].accepted, kK) << "acceptance is measured before the caps";
        EXPECT_EQ(t.out[0].tokens, (Toks{g[1], g[2]}));
        auto ref = seq();
        (void)prefill(off, *ref, pr);
        (void)feed_plain(off, *ref, std::span(g).first(2));
        expect_same_state(sp, *s, *ref);
    }
}

// ---- RR-006: cancellation / failure at every phase ----------------------------------------

TEST_F(Spec, AbortAfterRealDraftRestoresTheExactPreTickState) {
    const Toks pr = prompt("p0");
    auto sp = spec(GateMode::Always);
    auto s = seq(), twin = seq();
    const Toks first = prefill(sp, *s, pr);
    (void)prefill(sp, *twin, pr);
    StepRequest q;
    q.seq = s.get();
    q.token = first[0];
    q.max_draft = 3;
    Tick t;
    const std::size_t queued = s->mtp_queue_size();
    sp.draft(std::span(&q, 1), t);
    ASSERT_EQ(t.out[0].drafts.size(), 3u) << "real MTP chain of depth 3";
    EXPECT_EQ(s->mtp_kv->length(), pr.size() + 2) << "catch-up + x row + 2 chain rows";
    sp.abort_draft(std::span(&q, 1), t);
    EXPECT_EQ(s->mtp_queue_size(), queued) << "the catch-up queue is only consumed at commit";
    EXPECT_THROW(sp.verify(std::span(&q, 1), t), halo::Error) << "verify after abort";
    // Untouched twin: equal (flush both, then compare everything).
    expect_same_state(sp, *s, *twin);
    // And the aborted sequence still decodes like its twin.
    StepRequest a = q, b = q;
    b.seq = twin.get();
    EXPECT_EQ(step1(sp, a).out[0].tokens, step1(sp, b).out[0].tokens);
}

TEST_F(Spec, CancelRightAfterVerifyIsCommitOneAndContinuesLikePlain) {
    const Toks pr = prompt("p2");
    auto off = spec(GateMode::Off);
    auto gs = seq();
    const Toks g = generate(off, *gs, pr, 8, 0);
    auto sp = spec(GateMode::Always);
    auto s = seq();
    (void)prefill(sp, *s, pr);
    (void)feed_plain(sp, *s, std::span(g).first(2));
    StepRequest q;
    q.seq = s.get();
    q.token = g[2];
    q.max_draft = 2;
    Tick t;
    sp.draft(std::span(&q, 1), t);
    sp.verify(std::span(&q, 1), t);
    const std::size_t one[] = {1};
    sp.commit(std::span(&q, 1), t, one);  // the client cancelled before any token was shown
    EXPECT_EQ(t.out[0].tokens, Toks{g[3]});
    auto ref = seq();
    (void)prefill(off, *ref, pr);
    (void)feed_plain(off, *ref, std::span(g).first(3));
    expect_same_state(sp, *s, *ref);
    // Resume after cancellation: the continuation is still the plain greedy one.
    Toks cont{g[3]};
    while (cont.size() < 5) {
        StepRequest r;
        r.seq = s.get();
        r.token = cont.back();
        r.max_draft = 2;
        const Tick tt = step1(sp, r);
        cont.insert(cont.end(), tt.out[0].tokens.begin(), tt.out[0].tokens.end());
    }
    EXPECT_TRUE(std::equal(cont.begin(), cont.begin() + 5, g.begin() + 3));
}

TEST_F(Spec, KvExhaustionDuringVerifyLeavesEverySequenceAtItsPreTickState) {
    const Toks pr = prompt("p0");  // 150 rows; 10 blocks of 16 = 160 rows
    halo::kv_cache::KvPool small(model().kv_layout(), 10);
    auto sp = spec(GateMode::Always);
    auto off = spec(GateMode::Off);
    auto s = seq(4, &small);
    auto gs = seq();
    const Toks g = generate(off, *gs, pr, 16, 0);
    (void)prefill(sp, *s, pr);
    (void)feed_plain(sp, *s, std::span(g).first(8));  // length 158
    auto twin = seq();
    (void)prefill(off, *twin, pr);
    (void)feed_plain(off, *twin, std::span(g).first(8));
    const Toks d = oracle(g, 8, 3, 3, model().n_vocab());  // 4 rows -> 162 > 160
    StepRequest q;
    q.seq = s.get();
    q.token = g[8];
    q.forced_drafts = d;
    Tick t;
    try {
        sp.step(std::span(&q, 1), t);
        ADD_FAILURE() << "expected MEMORY_ERROR";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Memory) << e.what();
    }
    expect_same_state(sp, *s, *twin);
    // A plain step still fits (159 <= 160) and is correct.
    StepRequest p;
    p.seq = s.get();
    p.token = g[8];
    EXPECT_EQ(step1(sp, p).out[0].tokens, Toks{g[9]});
}

// ---- the draft chain (D-005 amendment) ---------------------------------------------------

TEST_F(Spec, DraftChainEqualsDirectMtpCallsFedTheirOwnHidden) {
    const Toks pr = prompt("p0");
    auto sp = spec(GateMode::Always);
    auto s = seq(), twin = seq();
    const Toks first = prefill(sp, *s, pr);
    (void)prefill(sp, *twin, pr);
    StepRequest q;
    q.seq = s.get();
    q.token = first[0];
    q.max_draft = 3;
    Tick t;
    sp.draft(std::span(&q, 1), t);
    // Direct: flush the twin's queue, then (x, h_{L-1}) at L, then (d_i, mtp_hidden_i).
    SequenceState* tw[] = {twin.get()};
    sp.flush_mtp(tw);
    const std::size_t L = twin->length();
    ASSERT_EQ(twin->mtp_kv->length(), L - 1);
    std::int32_t tok = first[0];
    std::vector<float> h = twin->last_hidden;
    Toks direct;
    for (std::size_t i = 0; i < 3; ++i) {
        halo::models::MtpStep m;
        m.tokens = std::span(&tok, 1);
        m.hidden = h;
        m.kv = &*twin->mtp_kv;
        m.first_position = static_cast<std::int32_t>(L + i);
        const std::size_t row0 = 0;
        m.logit_rows = std::span(&row0, 1);
        m.want_hidden = true;
        halo::models::StepResult r;
        model().mtp_forward(std::span(&m, 1), r);
        direct.push_back(r.seqs[0].argmax[0].index);
        tok = direct.back();
        h = r.seqs[0].hidden;
    }
    EXPECT_EQ(t.out[0].drafts, direct);
    EXPECT_EQ(dump_kv(*s->mtp_kv), dump_kv(*twin->mtp_kv)) << "chain rows written at the same MTP positions";
    sp.abort_draft(std::span(&q, 1), t);
}

// ---- MTP pairing anchored to the golden-validated convention ----------------------------------
// Every other MTP-state check compares the speculator with itself (spec path vs plain path
// share the pairing code). This one compares against direct model calls with exactly the
// pairing MtpEndToEndOnOwnHidden validated against the golden in M1: MTP row i =
// MTP(embed(tokens[i+1]), final_hidden[i]) at position i+1.

TEST_F(Spec, ChunkedPrefillMtpCatchUpEqualsDirectGoldenPairing) {
    const Toks pr = prompt("p0");
    ASSERT_EQ(pr.size(), 150u);
    auto sp = spec(GateMode::Always);
    auto s = seq();
    StepRequest c1;
    c1.seq = s.get();
    c1.prefill = std::span(pr).first(64);
    c1.want_output = false;
    (void)step1(sp, c1);
    StepRequest c2;
    c2.seq = s.get();
    c2.prefill = std::span(pr).subspan(64);  // exercises the (fed[0], last_hidden) pair at len0 > 0
    const Toks first = step1(sp, c2).out[0].tokens;
    SequenceState* one[] = {s.get()};
    sp.flush_mtp(one);
    ASSERT_EQ(s->mtp_kv->length(), 149u);
    // Direct reference: one-shot trunk forward (chunk boundaries 0/64/128 as above) + one
    // direct mtp_forward over the golden pairing.
    auto ref = seq();
    halo::models::SeqStep t;
    t.tokens = pr;
    t.kv = &ref->kv;
    t.gdn = &ref->gdn;
    t.logits = halo::models::LogitsMode::None;
    t.want_hidden = true;
    halo::models::StepResult tr;
    model().forward(std::span(&t, 1), tr);
    const std::size_t E = model().n_embd();
    const std::vector<float>& h = tr.seqs[0].hidden;
    EXPECT_TRUE(s->last_hidden == std::vector<float>(h.end() - static_cast<std::ptrdiff_t>(E), h.end())) << "pending_h = h_149";
    halo::models::MtpStep m;
    m.tokens = std::span(pr).subspan(1);
    m.hidden = std::span(h).first(149 * E);
    m.kv = &*ref->mtp_kv;
    m.first_position = 1;
    m.logits = halo::models::LogitsMode::None;
    halo::models::StepResult mr;
    model().mtp_forward(std::span(&m, 1), mr);
    EXPECT_TRUE(dump_kv(*s->mtp_kv) == dump_kv(*ref->mtp_kv)) << "MTP catch-up rows differ from the golden pairing";
    // First draft = direct MTP((x, h_149)) at position 150.
    const Toks gold = golden_->i32("p0.decode_tokens");
    ASSERT_EQ(first, Toks{gold[0]});
    StepRequest q;
    q.seq = s.get();
    q.token = first[0];
    q.max_draft = 1;
    Tick tk;
    sp.draft(std::span(&q, 1), tk);
    halo::models::MtpStep d;
    d.tokens = std::span(&first[0], 1);
    d.hidden = std::span(h).subspan(149 * E, E);
    d.kv = &*ref->mtp_kv;
    d.first_position = 150;
    const std::size_t row0 = 0;
    d.logit_rows = std::span(&row0, 1);
    halo::models::StepResult dr;
    model().mtp_forward(std::span(&d, 1), dr);
    ASSERT_EQ(tk.out[0].drafts.size(), 1u);
    EXPECT_EQ(tk.out[0].drafts[0], dr.seqs[0].argmax[0].index);
    sp.abort_draft(std::span(&q, 1), tk);
}

// ---- review N-1: a failed MTP flush must not let the queue grow ------------------------------

TEST_F(Spec, FailedMtpFlushDropsMtpInsteadOfGrowingTheQueue) {
    halo::kv_cache::KvPool small_mtp(model().mtp_kv_layout(), 2);  // 32 MTP rows: p0's 149 pairs cannot flush
    auto s = std::make_unique<SequenceState>(*tiny_->kv_pool, &small_mtp, model().gdn_shape(), 4);
    auto sp = spec(GateMode::Off);  // nothing drafts, so only the flush consumes the queue
    Toks out = prefill(sp, *s, prompt("p0"));
    EXPECT_FALSE(s->mtp_kv.has_value()) << "MTP must be dropped after the failed flush";
    EXPECT_EQ(s->mtp_queue_size(), 0u);
    EXPECT_EQ(sp.metrics().mtp_dropped, 1u);
    EXPECT_EQ(small_mtp.used_blocks(), 0u);
    while (out.size() < 6) {  // decoding continues, greedy output unaffected
        StepRequest r;
        r.seq = s.get();
        r.token = out.back();
        r.max_draft = 2;
        const Tick t = step1(sp, r);
        out.insert(out.end(), t.out[0].tokens.begin(), t.out[0].tokens.end());
        EXPECT_EQ(s->mtp_queue_size(), 0u) << "no queue growth without MTP";
    }
    auto off = spec(GateMode::Off);
    auto ref = seq();
    EXPECT_EQ(out, generate(off, *ref, prompt("p0"), 6, 0));
}

// ---- batch-shaped tick -----------------------------------------------------------------------

TEST_F(Spec, BatchedTickEqualsEachSequenceAlone) {
    const Toks p0 = prompt("p0"), p1 = prompt("p1"), p2 = prompt("p2");
    auto off = spec(GateMode::Off);
    auto gs = seq();
    const Toks g1 = generate(off, *gs, p1, 20, 0);
    // A: real MTP k=2; B: oracle k=2; C: plain; D: non-greedy (caller "samples" argmax of logits).
    const std::size_t V = model().n_vocab();
    auto run = [&](bool batched, std::vector<std::unique_ptr<SequenceState>>& st, std::vector<Toks>& toks, std::uint32_t& passes) {
        auto sp = spec(GateMode::Always);
        st.clear();
        for (int i = 0; i < 4; ++i) st.push_back(seq());
        toks.assign(4, {});
        const Toks* prompts[] = {&p0, &p1, &p2, &p1};
        auto tick = [&](std::vector<StepRequest>& reqs) {
            std::vector<Tick> ts;
            if (batched) {
                ts.resize(1);
                sp.step(reqs, ts[0]);
                passes = sp.metrics().last_weight_passes;
            } else {
                for (auto& r : reqs) ts.push_back(step1(sp, r));
            }
            for (std::size_t i = 0; i < 4; ++i) {
                const auto& o = batched ? ts[0].out[i] : ts[i].out[0];
                if (i == 3) {
                    EXPECT_TRUE(o.tokens.empty());
                    EXPECT_EQ(o.logits.size(), V);
                    toks[i].push_back(halo::cpu::top_k(o.logits, 1)[0].index);
                } else {
                    toks[i].insert(toks[i].end(), o.tokens.begin(), o.tokens.end());
                }
            }
        };
        std::vector<StepRequest> reqs(4);
        for (std::size_t i = 0; i < 4; ++i) {
            reqs[i].seq = st[i].get();
            reqs[i].prefill = *prompts[i];
            reqs[i].greedy = i != 3;
        }
        tick(reqs);
        std::vector<Toks> drafts(4);
        for (int stepn = 0; stepn < 5; ++stepn) {
            for (std::size_t i = 0; i < 4; ++i) {
                reqs[i] = {};
                reqs[i].seq = st[i].get();
                reqs[i].token = toks[i].back();
                reqs[i].greedy = i != 3;
                reqs[i].max_draft = i == 0 ? 2 : 0;
            }
            drafts[1] = oracle(g1, toks[1].size() - 1, 2, static_cast<std::size_t>(stepn) % 3, V);
            reqs[1].forced_drafts = drafts[1];
            tick(reqs);
        }
    };
    std::vector<std::unique_ptr<SequenceState>> sb, sa;
    std::vector<Toks> tb, ta;
    std::uint32_t passes = 0, unused = 0;
    run(true, sb, tb, passes);
    run(false, sa, ta, unused);
    for (std::size_t i = 0; i < 4; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(tb[i], ta[i]);
        auto sp = spec(GateMode::Always);
        expect_same_state(sp, *sb[i], *sa[i]);
    }
    EXPECT_TRUE(std::equal(tb[1].begin(), tb[1].end(), g1.begin())) << "oracle sequence == plain greedy";
    // One trunk pass + one MTP pass per draft depth for the whole batch (depth 1 serves A's
    // draft and B's catch-up; depth 2 only A).
    EXPECT_EQ(passes, 3u);
}

// ---- profit gate (FR-009) ------------------------------------------------------------------

TEST(ProfitGateUnit, DisablesBelowThresholdReprobesAndReenables) {
    GateConfig c;
    c.mode = GateMode::Auto;
    c.window = 4;
    c.probe_interval = 10;
    ProfitGate g(c, 100);
    EXPECT_TRUE(g.allow());
    for (int i = 0; i < 3; ++i) g.record_spec(1, 150);
    EXPECT_TRUE(g.allow()) << "no decision before a full window";
    g.record_spec(1, 150);  // 4 tokens / 600 bytes * 100 = 0.667x
    EXPECT_FALSE(g.allow());
    EXPECT_DOUBLE_EQ(g.last_speedup(), 4.0 * 100 / 600);
    EXPECT_EQ(g.disables(), 1u);
    for (int i = 0; i < 9; ++i) g.record_plain();
    EXPECT_FALSE(g.allow());
    g.record_plain();
    EXPECT_TRUE(g.allow()) << "re-probe after probe_interval plain steps";
    EXPECT_EQ(g.probes(), 1u);
    for (int i = 0; i < 4; ++i) g.record_spec(3, 150);  // 2.0x
    EXPECT_TRUE(g.allow());
    EXPECT_DOUBLE_EQ(g.last_speedup(), 2.0);
    // exactly the threshold stays on: 21 tokens * 100 / 2000 bytes = 1.05
    for (int i = 0; i < 3; ++i) g.record_spec(5, 500);
    g.record_spec(6, 500);
    EXPECT_DOUBLE_EQ(g.last_speedup(), 1.05);
    EXPECT_TRUE(g.allow());
    c.mode = GateMode::Always;
    ProfitGate a(c, 100);
    for (int i = 0; i < 8; ++i) a.record_spec(1, 1000);
    EXPECT_TRUE(a.allow());
    c.mode = GateMode::Off;
    EXPECT_FALSE(ProfitGate(c, 100).allow());
}

TEST_F(Spec, AutoGateDisablesMtpOnTheTinyModelFromMeasuredAcceptance) {
    // Tiny model: the 248,320 x 256 F32 LM head (254 MB) dwarfs the trunk, so every draft
    // costs ~one head pass; with the measured ~0 acceptance speculation cannot pay.
    SpecConfig c;
    c.max_draft = 2;
    c.gate.mode = GateMode::Auto;
    c.gate.window = 4;
    c.gate.probe_interval = 6;
    Speculator sp(model(), c);
    auto s = seq();
    Toks out = prefill(sp, *s, prompt("p1"));
    std::vector<std::size_t> k_seen;
    for (int i = 0; i < 14; ++i) {
        StepRequest r;
        r.seq = s.get();
        r.token = out.back();
        r.max_draft = 2;
        const Tick t = step1(sp, r);
        k_seen.push_back(t.out[0].drafts.size());
        out.insert(out.end(), t.out[0].tokens.begin(), t.out[0].tokens.end());
    }
    const auto& cm = sp.cost_model();
    EXPECT_GT(cm.mtp_head, cm.trunk) << "premise: head-dominated tiny model";
    EXPECT_EQ(sp.gate().disables(), 2u) << "off after the first window, on for a probe, off again";
    EXPECT_EQ(sp.gate().probes(), 1u);
    EXPECT_LT(sp.gate().last_speedup(), 1.05);
    const double expect = static_cast<double>(4 + sp.metrics().accepted) * static_cast<double>(cm.plain_step()) /
                          (4.0 * static_cast<double>(cm.spec_step(2)));
    std::printf("tiny gate: last speedup %.4f (accepted %llu of %llu drafts)\n", sp.gate().last_speedup(),
                static_cast<unsigned long long>(sp.metrics().accepted), static_cast<unsigned long long>(sp.metrics().drafted));
    if (sp.metrics().accepted == 0) EXPECT_DOUBLE_EQ(sp.gate().last_speedup(), expect);
    const std::vector<std::size_t> want = {2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2};
    EXPECT_EQ(k_seen, want);
    // Output is unaffected by the gate.
    auto off = spec(GateMode::Off);
    auto ref = seq();
    EXPECT_EQ(out, generate(off, *ref, prompt("p1"), out.size(), 0));
    // The queue built while the gate was off was consumed by the probe's depth-1 calls: the
    // resulting state equals the plain run's (history = every emitted token but the last).
    auto plain = seq();
    (void)prefill(off, *plain, prompt("p1"));
    (void)feed_plain(off, *plain, std::span(out).first(out.size() - 1));
    expect_same_state(sp, *s, *plain);
}

// ---- cost model: a verify + rollback step costs no extra weight pass (D-012 / D-014) -----------

TEST_F(Spec, VerifyCostsOneWeightPassAndRollbackCostsOnlySlotTraffic) {
    const Toks pr = prompt("p1");
    auto a = seq(), b = seq();
    auto sp = spec(GateMode::Always);
    const Toks f = prefill(sp, *a, pr);
    (void)prefill(sp, *b, pr);
    const std::int32_t row[] = {f[0]};
    const std::int32_t rows4[] = {f[0], 1, 2, 3};
    halo::models::SeqStep d, v;
    const std::size_t r0 = 0;
    const std::size_t all[] = {0, 1, 2, 3};
    d.tokens = row;
    d.kv = &a->kv;
    d.gdn = &a->gdn;
    d.logit_rows = std::span(&r0, 1);
    v.tokens = rows4;
    v.kv = &b->kv;
    v.gdn = &b->gdn;
    v.logit_rows = all;
    v.n_state_slots = 4;
    halo::models::StepResult rd, rv;
    model().forward(std::span(&d, 1), rd);
    model().forward(std::span(&v, 1), rv);
    EXPECT_EQ(rd.cost.weight_passes, 1u);
    EXPECT_EQ(rv.cost.weight_passes, 1u) << "4 verify rows share one weight pass";
    EXPECT_EQ(rv.cost.weight_bytes, rd.cost.weight_bytes) << "verify reads exactly the decode step's weights";
    const auto& sh = model().gdn_shape();
    const std::uint64_t state = sh.total_bytes();
    // Slot traffic: K = 4 slots written (slot 0 duplicates the live state; see report).
    EXPECT_EQ(rv.cost.state_bytes - rd.cost.state_bytes, 4 * state);
    // Rollback itself: a partial accept copies one slot into the live state (read + write).
    EXPECT_EQ(b->gdn.commit_bytes(4, 2), 2 * state);
    EXPECT_EQ(b->gdn.commit_bytes(4, 4), 0u);
    // The speculator's tick with k drafts = 1 trunk pass + k MTP passes.
    auto s = seq();
    const Toks f2 = prefill(sp, *s, pr);
    StepRequest q;
    q.seq = s.get();
    q.token = f2[0];
    q.max_draft = 3;
    (void)step1(sp, q);
    EXPECT_EQ(sp.metrics().last_weight_passes, 4u);
    std::printf("tiny: trunk %llu B, head %llu B, mtp block %llu B, state %llu B/seq\n",
                static_cast<unsigned long long>(sp.cost_model().trunk), static_cast<unsigned long long>(sp.cost_model().head),
                static_cast<unsigned long long>(sp.cost_model().mtp_block), static_cast<unsigned long long>(state));
}

TEST_F(Spec, RequestValidation) {
    auto sp = spec(GateMode::Always);
    auto s = seq(2);  // 2 slots -> at most 1 draft
    const Toks f = prefill(sp, *s, prompt("p1"));
    const std::int32_t d2[] = {1, 2};
    StepRequest q;
    q.seq = s.get();
    q.token = f[0];
    q.forced_drafts = d2;
    Tick t;
    EXPECT_THROW(sp.step(std::span(&q, 1), t), halo::Error) << "more drafts than slots";
    q.forced_drafts = {};
    q.max_emit = 0;
    EXPECT_THROW(sp.step(std::span(&q, 1), t), halo::Error);
    q.max_emit = 4;
    q.max_draft = 5;  // capped to slots - 1
    t = step1(sp, q);
    EXPECT_EQ(t.out[0].drafts.size(), 1u);
    StepRequest bad;
    Tick t2;
    EXPECT_THROW(sp.step(std::span(&bad, 1), t2), halo::Error) << "no sequence";
    Tick t3;
    EXPECT_THROW(sp.commit(std::span(&q, 1), t3), halo::Error) << "commit before verify";
    // A sequence whose MTP bookkeeping was corrupted is rejected, not silently used.
    s->mtp_queue_tokens.push_back(1);
    q.token = t.out[0].tokens.back();
    Tick t4;
    EXPECT_THROW(sp.step(std::span(&q, 1), t4), halo::Error);
}
