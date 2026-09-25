// Milestone 1: the CPU qwen35 forward vs transformers (trunk) and the NumPy port of
// llama.cpp's graph_mtp (MTP) on the tiny model (python/tools/make_tiny_model.py).
//
// Tolerance model (fixed before any HALO-vs-golden difference was measured; every check
// prints error / tolerance as "ratio"):
//
//   Both sides compute the same fp32 function with different reduction orders (HALO: 8
//   interleaved partial sums; torch: its own order; the MTP golden: float64). Per decoder
//   layer the largest reductions are the FFN down projection (n = 512) and, in GDN layers,
//   the state recursion over T tokens (tests/unit/cpu_kernels/tolerance.h Model R and G):
//     eps_red = 8 sqrt(512) eps          = 2.2e-5
//     eps_gdn = 8 eps sqrt(T (d_k + 64)) = 1.1e-4   (T = 150, d_k = 32)
//     eps_layer = eps_red + eps_gdn     ~= 1.3e-4   (relative to the row's max |value|)
//   Errors of successive layers add (linear, not quadrature: a deliberate over-estimate),
//   and the RMS norms, gates and SiLUs may amplify them by a small factor; we allow 4:
//     layer_in.i       : |a - ref| <= 4 i eps_layer max|ref row|
//     final_hidden     : 4 L eps_layer max|ref row|, L = 8            (= 4.2e-3 relative)
//     logits           : (4 L eps_layer + 8 sqrt(n_embd) eps) max|ref row|
//     MTP (golden h in): one block = 2 sublayers + eh_proj (n = 512):
//                        4 * 3 * eps_red max|ref row|                  (= 2.6e-4 relative)
//   An argmax is compared only where the top-1 margin exceeds 2x the logits tolerance
//   (the golden's margin for trunk rows; HALO's own margin for MTP rows, which have none).
//
// Quantized files (q8_0, q4_k_m, q6_k): a bitwise differential against the F32 forward over
// the same dequantized weights; see "quantized files" below for why not a golden tolerance.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <numeric>
#include <tuple>

#include "halo/backends/cpu/ops.h"
#include "halo/tensor/quant.h"
#include "halo/tokenizer/tokenizer.h"
#include "tiny_golden.h"

using halo::models::GdnPath;
using halo::models::LogitsMode;
using halo::models::MtpStep;
using halo::models::SeqStep;
using halo::models::StepResult;
using halo::test::Golden;
using halo::test::row_err;
using halo::test::TinyModel;

namespace {

constexpr double kEps = 1.1920928955078125e-07;
constexpr std::size_t kLayers = 8;
const double kEpsRed = 8.0 * std::sqrt(512.0) * kEps;
const double kEpsGdn = 8.0 * kEps * std::sqrt(150.0 * (32 + 64));
const double kEpsLayer = kEpsRed + kEpsGdn;
double tol_layer_rel(std::size_t i) { return 4.0 * static_cast<double>(i) * kEpsLayer; }
const double kFinalRel = 4.0 * kLayers * kEpsLayer;
const double kLogitRel = kFinalRel + 8.0 * std::sqrt(256.0) * kEps;
const double kMtpRel = 4.0 * 3.0 * kEpsRed;
const double kMtpLogitRel = kMtpRel + 8.0 * std::sqrt(256.0) * kEps;

const char* const kPrompts[] = {"p0", "p1", "p2"};

struct PromptRun {
    std::vector<std::int32_t> tokens;
    StepResult prefill;             // layer inputs, hidden, full logits of every row
    std::vector<std::vector<float>> decode_logits;  // 12 teacher-forced decode steps
    std::vector<GdnPath> decode_paths;
};

/// Prefill (one shot, all rows Full) + 12 teacher-forced decode steps on the golden tokens.
PromptRun run_prompt(const TinyModel& t, const Golden& g, const std::string& p, bool capture) {
    PromptRun r;
    r.tokens = g.i32(p + ".tokens");
    auto seq = t.seq();
    std::vector<std::size_t> rows(r.tokens.size());
    std::iota(rows.begin(), rows.end(), 0);
    SeqStep s;
    s.tokens = r.tokens;
    s.kv = &seq->kv;
    s.gdn = &seq->gdn;
    s.logit_rows = rows;
    s.logits = LogitsMode::Full;
    s.want_hidden = true;
    t.model->forward(std::span(&s, 1), r.prefill, {capture});
    const auto dec = g.i32(p + ".decode_tokens");
    for (const std::int32_t tok : dec) {
        StepResult sr;
        SeqStep d;
        const std::size_t row0 = 0;
        d.tokens = std::span(&tok, 1);
        d.kv = &seq->kv;
        d.gdn = &seq->gdn;
        d.logit_rows = std::span(&row0, 1);
        d.logits = LogitsMode::Full;
        t.model->forward(std::span(&d, 1), sr);
        r.decode_logits.push_back(std::move(sr.seqs[0].logits));
        r.decode_paths.push_back(sr.gdn_paths.at(0));
    }
    return r;
}

class Qwen35F32 : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        golden_ = Golden::load();
        if (!golden_) return;
        tiny_ = TinyModel::load("tiny-f32.gguf");
        if (!tiny_) return;
        for (const char* p : kPrompts) runs_[p] = run_prompt(*tiny_, *golden_, p, true);
    }
    static void TearDownTestSuite() {
        runs_.clear();
        tiny_.reset();
    }
    void SetUp() override {
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        if (!tiny_) GTEST_SKIP() << "tiny-f32.gguf missing under " << halo::test::tiny_dir();
    }
    static inline std::optional<Golden> golden_;
    static inline std::unique_ptr<TinyModel> tiny_;
    static inline std::map<std::string, PromptRun> runs_;
};

}  // namespace

TEST_F(Qwen35F32, EmbeddingAndLayerInputsMatchTransformers) {
    const std::size_t E = tiny_->model->n_embd();
    for (const char* p : kPrompts) {
        SCOPED_TRACE(p);
        const PromptRun& r = runs_.at(p);
        ASSERT_EQ(r.prefill.layer_inputs.size(), kLayers);
        for (std::size_t li = 0; li < kLayers; ++li) {
            const auto ref = golden_->f32(std::string(p) + ".layer_in." + std::to_string(li));
            ASSERT_EQ(ref.size(), r.tokens.size() * E);
            double worst = 0;
            for (std::size_t t = 0; t < r.tokens.size(); ++t) {
                const auto e = row_err(&r.prefill.layer_inputs[li][t * E], &ref[t * E], E);
                const double tol = tol_layer_rel(li) * e.max_abs_ref;
                if (li == 0) {
                    EXPECT_EQ(e.max_abs_err, 0.0) << "embedding row " << t << " must be exact for F32";
                } else {
                    EXPECT_LE(e.max_abs_err, tol) << "layer_in." << li << " row " << t;
                    worst = std::max(worst, e.max_abs_err / tol);
                }
            }
            if (li > 0) std::printf("[%s] layer_in.%zu worst ratio %.3f\n", p, li, worst);
        }
    }
}

TEST_F(Qwen35F32, FinalHiddenMatchesTransformers) {
    const std::size_t E = tiny_->model->n_embd();
    for (const char* p : kPrompts) {
        SCOPED_TRACE(p);
        const PromptRun& r = runs_.at(p);
        const auto ref = golden_->f32(std::string(p) + ".final_hidden");
        ASSERT_EQ(r.prefill.seqs[0].hidden.size(), ref.size());
        double worst = 0;
        for (std::size_t t = 0; t < r.tokens.size(); ++t) {
            const auto e = row_err(&r.prefill.seqs[0].hidden[t * E], &ref[t * E], E);
            const double tol = kFinalRel * e.max_abs_ref;
            EXPECT_LE(e.max_abs_err, tol) << "row " << t;
            worst = std::max(worst, e.max_abs_err / tol);
        }
        std::printf("[%s] final_hidden worst ratio %.3f\n", p, worst);
    }
}

TEST_F(Qwen35F32, LogitsRowsAndArgmaxMatchTransformers) {
    const std::size_t V = tiny_->model->n_vocab();
    for (const char* p : kPrompts) {
        SCOPED_TRACE(p);
        const PromptRun& r = runs_.at(p);
        const auto rows = golden_->i32(std::string(p) + ".logits_rows");
        const auto ref = golden_->f32(std::string(p) + ".logits");
        ASSERT_EQ(ref.size(), rows.size() * V);
        double worst = 0, max_ref = 0;
        for (std::size_t j = 0; j < rows.size(); ++j) {
            const auto row = static_cast<std::size_t>(rows[j]);
            const auto e = row_err(&r.prefill.seqs[0].logits[row * V], &ref[j * V], V);
            const double tol = kLogitRel * e.max_abs_ref;
            EXPECT_LE(e.max_abs_err, tol) << "logits row " << row;
            worst = std::max(worst, e.max_abs_err / tol);
            max_ref = std::max(max_ref, e.max_abs_ref);
        }
        std::printf("[%s] logits worst ratio %.3f\n", p, worst);
        // Argmax of every row where the golden top-1 margin exceeds twice that row's logits
        // tolerance (tolerance from HALO's own row max |logit|; the golden stores full
        // logits only for logits_rows).
        const auto am = golden_->i32(std::string(p) + ".argmax");
        const auto margin = golden_->f32(std::string(p) + ".top1_margin");
        const std::size_t n_rows = r.tokens.size();
        std::size_t compared = 0;
        for (std::size_t t = 0; t < n_rows; ++t) {
            double row_max = 0;
            for (std::size_t c = 0; c < V; ++c) {
                row_max = std::max(row_max, std::abs(static_cast<double>(r.prefill.seqs[0].logits[t * V + c])));
            }
            const double tol = kLogitRel * row_max;
            if (static_cast<double>(margin[t]) <= 2.0 * tol) {
                // Near-tie row: the golden's winner must be within 2 tol of HALO's best.
                const float mine = r.prefill.seqs[0].logits[t * V + static_cast<std::size_t>(am[t])];
                EXPECT_GE(static_cast<double>(mine), static_cast<double>(r.prefill.seqs[0].argmax[t].value) - 2.0 * tol)
                    << "near-tie row " << t;
                continue;
            }
            ++compared;
            EXPECT_EQ(r.prefill.seqs[0].argmax[t].index, am[t]) << "row " << t << " margin " << margin[t];
        }
        std::printf("[%s] argmax compared on %zu of %zu rows (max row logit %.3f)\n", p, compared, n_rows, max_ref);
        // Every row is checked one way or the other; the exact-argmax comparison must still
        // cover most of the long prompt (p0) to be a meaningful test.
        if (n_rows >= 100) EXPECT_GE(compared, n_rows * 3 / 4) << "too few rows above the margin threshold";
    }
}

TEST_F(Qwen35F32, GreedyDecodeWithCacheMatchesTransformers) {
    for (const char* p : kPrompts) {
        SCOPED_TRACE(p);
        const PromptRun& r = runs_.at(p);
        const auto dec = golden_->i32(std::string(p) + ".decode_tokens");
        const auto top_ids = golden_->i32(std::string(p) + ".decode_top16_ids");
        const auto top_lg = golden_->f32(std::string(p) + ".decode_top16_logits");
        // decode_tokens[0] is the argmax of the prefill's last row.
        EXPECT_EQ(r.prefill.seqs[0].argmax.back().index, dec[0]);
        double worst = 0;
        for (std::size_t i = 0; i < dec.size(); ++i) {
            EXPECT_EQ(r.decode_paths[i], GdnPath::Recurrent) << "decode must use the recurrent GDN path";
            const auto& lg = r.decode_logits[i];
            double max_abs = 0;
            for (std::size_t j = 0; j < 16; ++j) max_abs = std::max(max_abs, std::abs(static_cast<double>(top_lg[i * 16 + j])));
            const double tol = kLogitRel * max_abs;
            for (std::size_t j = 0; j < 16; ++j) {
                const auto id = static_cast<std::size_t>(top_ids[i * 16 + j]);
                const double err = std::abs(static_cast<double>(lg[id]) - static_cast<double>(top_lg[i * 16 + j]));
                EXPECT_LE(err, tol) << "step " << i << " top-" << j << " id " << id;
                worst = std::max(worst, err / tol);
            }
            // Our own top-16 ids where consecutive golden logits are separated by > 2 tol.
            const auto ours = halo::cpu::top_k(lg, 16);
            for (std::size_t j = 0; j < 16; ++j) {
                const bool sep_above = j == 0 || static_cast<double>(top_lg[i * 16 + j - 1] - top_lg[i * 16 + j]) > 2 * tol;
                const bool sep_below = j == 15 || static_cast<double>(top_lg[i * 16 + j] - top_lg[i * 16 + j + 1]) > 2 * tol;
                if (sep_above && sep_below) EXPECT_EQ(ours[j].index, top_ids[i * 16 + j]) << "step " << i << " rank " << j;
            }
            if (i + 1 < dec.size()) EXPECT_EQ(ours[0].index, dec[i + 1]) << "greedy chain at step " << i;
        }
        std::printf("[%s] decode top-16 logits worst ratio %.3f\n", p, worst);
    }
}

TEST_F(Qwen35F32, PrefillUsesChunkedPathAndChunkedDiffersFromRecurrent) {
    const PromptRun& r = runs_.at("p0");
    ASSERT_EQ(r.prefill.gdn_paths.size(), 1u);
    EXPECT_EQ(r.prefill.gdn_paths[0], GdnPath::Chunked);
    // The two paths are different arithmetic (not bitwise equal) but the same function.
    auto seq = tiny_->seq();
    SeqStep s;
    s.tokens = r.tokens;
    s.kv = &seq->kv;
    s.gdn = &seq->gdn;
    s.want_hidden = true;
    s.logits = LogitsMode::None;
    s.gdn_path = GdnPath::Recurrent;
    StepResult rr;
    tiny_->model->forward(std::span(&s, 1), rr);
    EXPECT_EQ(rr.gdn_paths[0], GdnPath::Recurrent);
    EXPECT_NE(rr.seqs[0].hidden, r.prefill.seqs[0].hidden);
    const std::size_t E = tiny_->model->n_embd();
    for (std::size_t t = 0; t < r.tokens.size(); ++t) {
        const auto e = row_err(&rr.seqs[0].hidden[t * E], &r.prefill.seqs[0].hidden[t * E], E);
        EXPECT_LE(e.max_abs_err, 2 * kFinalRel * e.max_abs_ref) << "row " << t;
    }
}

namespace {

struct PiecesOut {
    std::vector<float> hidden;
    std::vector<float> last_logits;
    std::vector<float> gdn;
};

PiecesOut prefill_in_pieces(const TinyModel& t, std::span<const std::int32_t> tokens, std::span<const std::size_t> pieces) {
    auto seq = t.seq();
    PiecesOut o;
    std::size_t at = 0;
    for (const std::size_t n : pieces) {
        SeqStep s;
        s.tokens = tokens.subspan(at, n);
        s.kv = &seq->kv;
        s.gdn = &seq->gdn;
        s.want_hidden = true;
        const std::size_t last = n - 1;
        s.logit_rows = std::span(&last, 1);
        s.logits = LogitsMode::Full;
        s.gdn_path = GdnPath::Chunked;
        StepResult r;
        t.model->forward(std::span(&s, 1), r);
        o.hidden.insert(o.hidden.end(), r.seqs[0].hidden.begin(), r.seqs[0].hidden.end());
        o.last_logits = r.seqs[0].logits;
        at += n;
    }
    EXPECT_EQ(at, tokens.size());
    o.gdn = seq->gdn.snapshot().data;
    return o;
}

}  // namespace

TEST_F(Qwen35F32, PrefillInPiecesEqualsOneShot) {
    const PromptRun& r = runs_.at("p0");
    const std::size_t one[] = {150};
    const std::size_t aligned[] = {64, 64, 22};
    const std::size_t unaligned[] = {10, 50, 89, 1};
    const PiecesOut ref = prefill_in_pieces(*tiny_, r.tokens, one);
    EXPECT_EQ(ref.hidden, r.prefill.seqs[0].hidden) << "a fresh run must reproduce the suite's prefill bitwise";
    const PiecesOut a = prefill_in_pieces(*tiny_, r.tokens, aligned);
    EXPECT_EQ(a.hidden, ref.hidden) << "pieces aligned to the 64-token chunk must be bitwise equal";
    EXPECT_EQ(a.last_logits, ref.last_logits);
    EXPECT_EQ(a.gdn, ref.gdn);
    const PiecesOut u = prefill_in_pieces(*tiny_, r.tokens, unaligned);
    const std::size_t E = tiny_->model->n_embd();
    double worst = 0;
    for (std::size_t t = 0; t < r.tokens.size(); ++t) {
        const auto e = row_err(&u.hidden[t * E], &ref.hidden[t * E], E);
        const double tol = 2 * kFinalRel * e.max_abs_ref;  // two fp32 runs, each within kFinalRel of exact
        EXPECT_LE(e.max_abs_err, tol) << "row " << t;
        worst = std::max(worst, e.max_abs_err / tol);
    }
    std::printf("unaligned pieces vs one-shot worst ratio %.3f\n", worst);
}

TEST_F(Qwen35F32, BatchedSequencesAreBitIdenticalToAlone) {
    auto s0 = tiny_->seq(), s1 = tiny_->seq(), s2 = tiny_->seq();
    std::vector<std::vector<std::int32_t>> toks;
    for (const char* p : kPrompts) toks.push_back(runs_.at(p).tokens);
    std::vector<std::vector<std::size_t>> rows(3);
    SeqStep steps[3];
    halo::test::TestSeq* seqs[] = {s0.get(), s1.get(), s2.get()};
    for (std::size_t i = 0; i < 3; ++i) {
        rows[i].resize(toks[i].size());
        std::iota(rows[i].begin(), rows[i].end(), 0);
        steps[i].tokens = toks[i];
        steps[i].kv = &seqs[i]->kv;
        steps[i].gdn = &seqs[i]->gdn;
        steps[i].logit_rows = rows[i];
        steps[i].logits = LogitsMode::Full;
        steps[i].want_hidden = true;
    }
    StepResult r;
    tiny_->model->forward(steps, r);
    EXPECT_EQ(r.cost.weight_passes, 1u);
    for (std::size_t i = 0; i < 3; ++i) {
        const PromptRun& alone = runs_.at(kPrompts[i]);
        EXPECT_EQ(r.seqs[i].hidden, alone.prefill.seqs[0].hidden) << kPrompts[i];
        EXPECT_EQ(r.seqs[i].logits, alone.prefill.seqs[0].logits) << kPrompts[i];
    }
}

// ---- MTP (D-005) ------------------------------------------------------------------------

namespace {

StepResult run_mtp(const TinyModel& t, std::span<const std::int32_t> next_tokens, std::span<const float> hidden,
                   halo::test::TestSeq& seq) {
    std::vector<std::size_t> rows(next_tokens.size());
    std::iota(rows.begin(), rows.end(), 0);
    MtpStep m;
    m.tokens = next_tokens;
    m.hidden = hidden;
    m.kv = &seq.mtp_kv;
    m.first_position = 1;
    m.logit_rows = rows;
    m.logits = LogitsMode::Full;
    m.want_hidden = true;
    StepResult r;
    t.model->mtp_forward(std::span(&m, 1), r);
    return r;
}

void check_mtp(const TinyModel& t, const Golden& g, const std::string& p, const StepResult& r, double hid_rel,
               double logit_rel) {
    const std::size_t E = t.model->n_embd();
    const std::size_t V = t.model->n_vocab();
    const auto ref_h = g.f32(p + ".mtp_hidden");
    const auto ref_am = g.i32(p + ".mtp_argmax");
    const auto ref_last = g.f32(p + ".mtp_logits_last");
    const std::size_t n = ref_am.size();
    ASSERT_EQ(r.seqs[0].hidden.size(), n * E);
    double worst = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto e = row_err(&r.seqs[0].hidden[i * E], &ref_h[i * E], E);
        const double tol = hid_rel * e.max_abs_ref;
        EXPECT_LE(e.max_abs_err, tol) << p << " mtp_hidden row " << i;
        worst = std::max(worst, e.max_abs_err / tol);
    }
    const auto e = row_err(&r.seqs[0].logits[(n - 1) * V], ref_last.data(), V);
    const double ltol = logit_rel * e.max_abs_ref;
    EXPECT_LE(e.max_abs_err, ltol) << p << " mtp_logits_last";
    std::size_t compared = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto top = halo::cpu::top_k(std::span(&r.seqs[0].logits[i * V], V), 2);
        const double tol = logit_rel * static_cast<double>(std::max(std::abs(top[0].value), std::abs(top[1].value)));
        EXPECT_EQ(top[0].index, r.seqs[0].argmax[i].index);
        if (static_cast<double>(top[0].value - top[1].value) <= 2 * tol) continue;
        ++compared;
        EXPECT_EQ(top[0].index, ref_am[i]) << p << " mtp row " << i;
    }
    std::printf("[%s] mtp hidden worst ratio %.3f, last-logits ratio %.3f, argmax compared %zu/%zu\n", p.c_str(), worst,
                e.max_abs_err / ltol, compared, n);
    EXPECT_GE(compared, n * 8 / 10);
}

}  // namespace

TEST_F(Qwen35F32, MtpMatchesGoldenGivenGoldenHidden) {
    ASSERT_TRUE(tiny_->model->has_mtp());
    const std::size_t E = tiny_->model->n_embd();
    for (const char* p : {"p0", "p1"}) {
        SCOPED_TRACE(p);
        const auto toks = golden_->i32(std::string(p) + ".tokens");
        const auto fh = golden_->f32(std::string(p) + ".final_hidden");
        auto seq = tiny_->seq();
        const std::size_t n = toks.size() - 1;
        const StepResult r = run_mtp(*tiny_, std::span(toks).subspan(1), std::span(fh).first(n * E), *seq);
        check_mtp(*tiny_, *golden_, p, r, kMtpRel, kMtpLogitRel);
        EXPECT_EQ(seq->mtp_kv.length(), n);
    }
    EXPECT_FALSE(golden_->has("p2.mtp_hidden")) << "p2 has one token: no MTP row exists (checked, not skipped silently)";
}

TEST_F(Qwen35F32, MtpEndToEndOnOwnHidden) {
    const std::size_t E = tiny_->model->n_embd();
    for (const char* p : {"p0", "p1"}) {
        SCOPED_TRACE(p);
        const PromptRun& run = runs_.at(p);
        const std::size_t n = run.tokens.size() - 1;
        auto seq = tiny_->seq();
        const StepResult r = run_mtp(*tiny_, std::span(run.tokens).subspan(1),
                                     std::span(run.prefill.seqs[0].hidden).first(n * E), *seq);
        check_mtp(*tiny_, *golden_, p, r, kFinalRel + kMtpRel, kLogitRel + kMtpLogitRel);
    }
}

TEST_F(Qwen35F32, MtpInPiecesEqualsOneShotAndDraftChainIsSelfConsistent) {
    // No golden exists for the draft chain (D-005 amendment: "a draft-chain golden is
    // owed"); this checks self-consistency only: (1) MTP rows computed one at a time
    // (the drafting pattern) are bit-identical to one batched call; (2) a chain step fed
    // the MTP's own post-shared_head_norm hidden equals a teacher-forced call on the same
    // (token, hidden) pair.
    const std::size_t E = tiny_->model->n_embd();
    const auto& run = runs_.at("p1");
    const std::size_t n = run.tokens.size() - 1;
    const std::span<const float> h = std::span(run.prefill.seqs[0].hidden).first(n * E);
    auto a = tiny_->seq();
    const StepResult all = run_mtp(*tiny_, std::span(run.tokens).subspan(1), h, *a);
    auto b = tiny_->seq();
    std::vector<float> hid;
    for (std::size_t i = 0; i < n; ++i) {
        MtpStep m;
        m.tokens = std::span(run.tokens).subspan(1 + i, 1);
        m.hidden = h.subspan(i * E, E);
        m.kv = &b->mtp_kv;
        m.first_position = static_cast<std::int32_t>(1 + i);
        m.want_hidden = true;
        m.logits = LogitsMode::None;
        StepResult r;
        tiny_->model->mtp_forward(std::span(&m, 1), r);
        hid.insert(hid.end(), r.seqs[0].hidden.begin(), r.seqs[0].hidden.end());
    }
    EXPECT_EQ(hid, all.seqs[0].hidden);
    // Chain: draft d1 from (x_n = decode_tokens[0], h_{n-1}), then d2 from (d1, mtp_hidden(d1-row)).
    const std::int32_t x_n = golden_->i32("p1.decode_tokens")[0];
    const std::span<const float> h_last = std::span(run.prefill.seqs[0].hidden).subspan(n * E, E);
    auto c = tiny_->seq();
    auto run1 = [&](halo::test::TestSeq& s, std::int32_t tok, std::span<const float> hin, std::int32_t pos) {
        MtpStep m;
        m.tokens = std::span(&tok, 1);
        m.hidden = hin;
        m.kv = &s.mtp_kv;
        m.first_position = pos;
        const std::size_t row0 = 0;
        m.logit_rows = std::span(&row0, 1);
        m.want_hidden = true;
        StepResult r;
        tiny_->model->mtp_forward(std::span(&m, 1), r);
        return r;
    };
    c->mtp_kv.share_prefix(a->mtp_kv, n);  // history rows for positions 1..n
    const auto pos_n = static_cast<std::int32_t>(n + 1);
    const StepResult d1 = run1(*c, x_n, h_last, pos_n);
    const StepResult d2 = run1(*c, d1.seqs[0].argmax[0].index, d1.seqs[0].hidden, pos_n + 1);
    // Teacher-forced replay of the same two pairs in one call from the same history.
    auto d = tiny_->seq();
    d->mtp_kv.share_prefix(a->mtp_kv, n);
    const std::int32_t pair_tokens[] = {x_n, d1.seqs[0].argmax[0].index};
    std::vector<float> pair_h(h_last.begin(), h_last.end());
    pair_h.insert(pair_h.end(), d1.seqs[0].hidden.begin(), d1.seqs[0].hidden.end());
    MtpStep m;
    m.tokens = pair_tokens;
    m.hidden = pair_h;
    m.kv = &d->mtp_kv;
    m.first_position = pos_n;
    const std::size_t rows2[] = {0, 1};
    m.logit_rows = rows2;
    m.want_hidden = true;
    StepResult both;
    tiny_->model->mtp_forward(std::span(&m, 1), both);
    EXPECT_EQ(both.seqs[0].argmax[0], d1.seqs[0].argmax[0]);
    EXPECT_EQ(both.seqs[0].argmax[1], d2.seqs[0].argmax[0]);
    EXPECT_EQ(std::vector<float>(both.seqs[0].hidden.begin() + static_cast<std::ptrdiff_t>(E), both.seqs[0].hidden.end()),
              d2.seqs[0].hidden);
    EXPECT_EQ(c->mtp_kv.length(), n + 2);
}

// ---- failure atomicity ------------------------------------------------------------------

TEST_F(Qwen35F32, InvalidStepLeavesStateUnchanged) {
    auto seq = tiny_->seq();
    const auto& toks = runs_.at("p1").tokens;
    SeqStep s;
    s.tokens = toks;
    s.kv = &seq->kv;
    s.gdn = &seq->gdn;
    s.logits = LogitsMode::None;
    StepResult r;
    tiny_->model->forward(std::span(&s, 1), r);
    const auto before = seq->gdn.snapshot().data;
    const std::int32_t bad[] = {5, static_cast<std::int32_t>(tiny_->model->n_vocab())};
    s.tokens = bad;
    EXPECT_THROW(tiny_->model->forward(std::span(&s, 1), r), halo::Error);
    EXPECT_EQ(seq->kv.length(), toks.size());
    EXPECT_EQ(seq->gdn.snapshot().data, before);
    const std::size_t bad_row = 9;
    s.tokens = std::span(toks).first(1);
    s.logit_rows = std::span(&bad_row, 1);
    s.logits = LogitsMode::Argmax;
    EXPECT_THROW(tiny_->model->forward(std::span(&s, 1), r), halo::Error);
    EXPECT_EQ(seq->kv.length(), toks.size());
}

TEST_F(Qwen35F32, KvPoolExhaustionIsMemoryErrorAndAtomic) {
    halo::kv_cache::KvPool small(tiny_->model->kv_layout(), 2);  // 32 rows
    halo::kv_cache::SequenceKv kv(small);
    halo::state::GdnState gdn(tiny_->model->gdn_shape(), 0);
    const auto& toks = runs_.at("p0").tokens;
    SeqStep s;
    s.tokens = std::span(toks).first(20);
    s.kv = &kv;
    s.gdn = &gdn;
    s.logits = LogitsMode::None;
    StepResult r;
    tiny_->model->forward(std::span(&s, 1), r);
    const auto before = gdn.snapshot().data;
    s.tokens = std::span(toks).subspan(20, 20);
    try {
        tiny_->model->forward(std::span(&s, 1), r);
        ADD_FAILURE() << "expected MEMORY_ERROR";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Memory) << e.what();
    }
    EXPECT_EQ(kv.length(), 20u);
    EXPECT_EQ(gdn.snapshot().data, before);
}

// ---- quantized files -----------------------------------------------------------------------
// Differential, bitwise: HALO on a quantized file == HALO on an in-memory F32 GGUF image of
// the same file's dequantized tensors (identical metadata). The F32 path is tied to
// transformers above, so by composition the quantized path computes the transformers
// function over the dequantized weights; any error in the quant plumbing (tensor choice,
// row offsets, head slabs, embedding rows, MTP embedding/head) breaks bitwise equality.
//
// Why not a tolerance against the golden: an a-priori noise model (each matmul adds an
// independent relative error rho = max relative RMS weight error, quadrature over 4L + 2
// matmuls, 3 sigma) was written before measuring and is refuted by the measurement on this
// random tiny network: layer-input error grows smoothly with depth to ~10-15 rho by layer 7
// for every layer kind (amplification, not a defect), so the model under-predicts, and a
// tolerance loose enough to pass leaves no argmax row comparable. The measurement is still
// printed (rho, per-layer error growth vs the golden) for the report.

namespace {

void put_u32(std::vector<std::byte>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFFu));
}
void put_u64(std::vector<std::byte>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFFu));
}

/// In-memory GGUF with the same header KVs as `src` and every tensor dequantized to F32.
halo::model::GgufFile dequantized_image(const halo::model::GgufFile& src) {
    const auto& ts = src.tensors();
    std::uint64_t infos = 0;
    for (const auto& t : ts) infos += 8 + t.name.size() + 4 + 8ULL * t.n_dims + 4 + 8;
    const std::uint64_t kv_end = src.header_size() - infos;
    const auto raw = src.tensor_data(ts.front());  // any tensor: we need the file base
    const std::byte* file_base = raw.data() - src.data_offset() - ts.front().offset;
    std::vector<std::byte> b(file_base, file_base + kv_end);  // magic, version, counts, KVs
    const std::uint64_t align = src.alignment();
    std::vector<std::uint64_t> offs;
    std::uint64_t off = 0;
    for (const auto& t : ts) {
        offs.push_back(off);
        off += (t.n_elements * 4 + align - 1) / align * align;
    }
    for (std::size_t i = 0; i < ts.size(); ++i) {
        put_u64(b, ts[i].name.size());
        for (const char c : ts[i].name) b.push_back(static_cast<std::byte>(c));
        put_u32(b, ts[i].n_dims);
        for (std::uint32_t d = 0; d < ts[i].n_dims; ++d) put_u64(b, static_cast<std::uint64_t>(ts[i].ne[d]));
        put_u32(b, static_cast<std::uint32_t>(halo::DType::F32));
        put_u64(b, offs[i]);
    }
    b.resize((b.size() + align - 1) / align * align);
    const std::size_t data0 = b.size();
    b.resize(data0 + off);
    for (std::size_t i = 0; i < ts.size(); ++i) {
        std::vector<float> f(ts[i].n_elements);
        halo::tensor::dequantize_row(ts[i].type, src.tensor_data(ts[i]), f);
        std::memcpy(b.data() + data0 + offs[i], f.data(), f.size() * sizeof(float));
    }
    return halo::model::GgufFile::parse(std::move(b), halo::model::GgufMode::Full, src.source() + " (dequantized)");
}

struct QuantRun {
    PromptRun trunk;
    StepResult mtp;
};

QuantRun run_quant(const TinyModel& t, const Golden& g, const std::string& p) {
    QuantRun q;
    q.trunk = run_prompt(t, g, p, true);
    const std::size_t E = t.model->n_embd();
    const std::size_t n = q.trunk.tokens.size() - 1;
    if (n > 0) {
        auto seq = t.seq();
        std::vector<std::size_t> rows(n);
        std::iota(rows.begin(), rows.end(), 0);
        MtpStep m;
        m.tokens = std::span(q.trunk.tokens).subspan(1);
        m.hidden = std::span(q.trunk.prefill.seqs[0].hidden).first(n * E);
        m.kv = &seq->mtp_kv;
        m.logit_rows = rows;
        m.logits = LogitsMode::Full;
        m.want_hidden = true;
        t.model->mtp_forward(std::span(&m, 1), q.mtp);
    }
    return q;
}

double rel_rms_weight_error(const halo::model::WeightRef& q, const halo::model::WeightRef& f) {
    const auto mq = halo::models::weight_matrix(q);
    const auto mf = halo::models::weight_matrix(f);
    std::vector<float> sq, sf;
    double se = 0, sr = 0;
    constexpr std::size_t kBlock = 4096;
    for (std::size_t r0 = 0; r0 < mq.rows(); r0 += kBlock) {
        const std::size_t n = std::min(kBlock, mq.rows() - r0);
        const auto bq = mq.row_block(r0, n, sq);
        const auto bf = mf.row_block(r0, n, sf);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < mq.cols(); ++c) {
                const double d = static_cast<double>(bq(r, c)) - static_cast<double>(bf(r, c));
                se += d * d;
                sr += static_cast<double>(bf(r, c)) * static_cast<double>(bf(r, c));
            }
        }
    }
    return std::sqrt(se / sr);
}

/// max relative RMS error over every 2-D trunk weight + embedding + LM head (diagnostic).
double measure_rho(const halo::model::NormalizedModel& q, const halo::model::NormalizedModel& f) {
    double rho = 0;
    const auto upd = [&](const halo::model::WeightRef& a, const halo::model::WeightRef& b) {
        if (a.present()) rho = std::max(rho, rel_rms_weight_error(a, b));
    };
    for (std::uint32_t il = 0; il < q.hparams().n_layer; ++il) {
        const auto& a = q.layer(il);
        const auto& b = f.layer(il);
        if (a.kind == halo::model::LayerKind::FullAttention) {
            upd(a.attn.q, b.attn.q);
            upd(a.attn.k, b.attn.k);
            upd(a.attn.v, b.attn.v);
            upd(a.attn.output, b.attn.output);
        } else {
            upd(a.gdn.qkv, b.gdn.qkv);
            upd(a.gdn.gate, b.gdn.gate);
            upd(a.gdn.beta, b.gdn.beta);
            upd(a.gdn.alpha, b.gdn.alpha);
            upd(a.gdn.out, b.gdn.out);
        }
        upd(a.ffn_gate, b.ffn_gate);
        upd(a.ffn_up, b.ffn_up);
        upd(a.ffn_down, b.ffn_down);
    }
    upd(q.token_embd(), f.token_embd());
    upd(q.output(), f.output());
    return rho;
}

// One instance per (file, prompt): three prompts in one test took ~1000 s under ASan with
// the full suite running in parallel, too close to ctest's 1500 s default timeout.
class Qwen35Quant : public ::testing::TestWithParam<std::tuple<const char*, const char*>> {};

}  // namespace

TEST_P(Qwen35Quant, BitIdenticalToF32ForwardOverDequantizedWeights) {
    const auto golden = Golden::load();
    if (!golden) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
    const char* const file = std::get<0>(GetParam());
    const char* const p = std::get<1>(GetParam());
    const auto qt = TinyModel::load(file, 64);
    if (!qt) GTEST_SKIP() << file << " missing under " << halo::test::tiny_dir();
    // At least one non-F32 matrix, or the differential is vacuous.
    ASSERT_NE(qt->nm->layer(0).ffn_down.type(), halo::DType::F32);
    ASSERT_NE(qt->nm->output().type(), halo::DType::F32);

    TinyModel dq;
    dq.nm = std::make_unique<halo::model::NormalizedModel>(halo::model::NormalizedModel::from_gguf(dequantized_image(qt->nm->gguf())));
    ASSERT_EQ(dq.nm->output().type(), halo::DType::F32);
    dq.pool = std::make_unique<halo::cpu::ThreadPool>(halo::cpu::ThreadPool::default_threads());
    dq.model = std::make_unique<halo::models::Qwen35>(*dq.nm, dq.pool.get());
    dq.kv_pool = std::make_unique<halo::kv_cache::KvPool>(dq.model->kv_layout(), 64);
    dq.mtp_pool = std::make_unique<halo::kv_cache::KvPool>(dq.model->mtp_kv_layout(), 64);
    ASSERT_EQ(qt->model->has_mtp(), dq.model->has_mtp());

    const std::size_t E = qt->model->n_embd();
    {
        SCOPED_TRACE(p);
        const QuantRun a = run_quant(*qt, *golden, p);
        const QuantRun b = run_quant(dq, *golden, p);
        EXPECT_EQ(a.trunk.prefill.layer_inputs, b.trunk.prefill.layer_inputs);
        EXPECT_EQ(a.trunk.prefill.seqs[0].hidden, b.trunk.prefill.seqs[0].hidden);
        EXPECT_EQ(a.trunk.prefill.seqs[0].logits, b.trunk.prefill.seqs[0].logits);
        EXPECT_EQ(a.trunk.decode_logits, b.trunk.decode_logits);
        EXPECT_EQ(a.mtp.seqs.empty(), b.mtp.seqs.empty());
        if (!a.mtp.seqs.empty()) {
            EXPECT_EQ(a.mtp.seqs[0].hidden, b.mtp.seqs[0].hidden);
            EXPECT_EQ(a.mtp.seqs[0].logits, b.mtp.seqs[0].logits);
        }
        // Diagnostic only (see above): error growth against the transformers golden.
        std::printf("[%s/%s] layer_in rel rms err vs golden:", file, p);
        for (std::size_t li = 0; li < kLayers; ++li) {
            const auto ref = golden->f32(std::string(p) + ".layer_in." + std::to_string(li));
            const auto e = row_err(a.trunk.prefill.layer_inputs[li].data(), ref.data(), ref.size());
            std::printf(" %.4f", e.rms_err / e.rms_ref);
        }
        const auto ref_h = golden->f32(std::string(p) + ".final_hidden");
        const auto eh = row_err(a.trunk.prefill.seqs[0].hidden.data(), ref_h.data(), ref_h.size());
        std::printf(" | final_hidden %.4f\n", eh.rms_err / eh.rms_ref);
        (void)E;
    }
    const auto f32p = halo::test::tiny_dir() / "tiny-f32.gguf";
    if (std::string(p) == "p0" && std::filesystem::exists(f32p)) {
        const auto f32 = halo::model::NormalizedModel::load(f32p);
        std::printf("[%s] rho (max relative RMS weight error vs tiny-f32) %.5f\n", file, measure_rho(*qt->nm, f32));
    }
}

INSTANTIATE_TEST_SUITE_P(Files, Qwen35Quant,
                         ::testing::Combine(::testing::Values("tiny-q8_0.gguf", "tiny-q4_k_m.gguf", "tiny-q6_k.gguf"),
                                            ::testing::Values("p0", "p1", "p2")),
                         [](const auto& info) {
                             const std::string n = std::get<0>(info.param);
                             return n.substr(5, n.size() - 10) + "_" + std::get<1>(info.param);  // tiny-XXX.gguf -> XXX_pN
                         });
