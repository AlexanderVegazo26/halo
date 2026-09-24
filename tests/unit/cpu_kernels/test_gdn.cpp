// Gated DeltaNet CPU reference tests (TRD §29): recurrent and chunked vs an independent
// fp64 recurrence, chunked == recurrent across chunk boundaries, state continuity under
// arbitrary splits, head-mapping permutation, rollback slots, thread-count invariance.
// Tolerances: Model G in tolerance.h (fixed before measurement).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/traffic.h"
#include "halo/core/error.h"
#include "reference.h"
#include "tolerance.h"

// This file compares against fp64 reference math and promotes float operands to double on
// purpose. -Wdouble-promotion exists to catch *accidental* promotion in float code; it stays
// enabled for the CPU kernels themselves (backends/cpu) and is disabled for this file only.
#pragma GCC diagnostic ignored "-Wdouble-promotion"

namespace {

using namespace halo::cpu;
using namespace halo::cpu::test;

enum class Regime { Model, WeakDecay, StrongDecay, BetaLow, BetaHigh };

const char* regime_name(Regime r) {
    switch (r) {
        case Regime::Model: return "model";
        case Regime::WeakDecay: return "weak_decay";
        case Regime::StrongDecay: return "strong_decay";
        case Regime::BetaLow: return "beta_low";
        case Regime::BetaHigh: return "beta_high";
    }
    return "?";
}

struct Data {
    GdnDims dims;
    std::size_t T = 0;
    std::vector<float> q, k, v, g, beta, s0;
    [[nodiscard]] std::size_t qk_cols() const { return dims.n_k_heads * dims.d_k; }
    [[nodiscard]] std::size_t v_cols() const { return dims.n_v_heads * dims.d_v; }
    [[nodiscard]] std::size_t state_n() const { return dims.n_v_heads * dims.d_k * dims.d_v; }
    /// Inputs for rows [t0, t0 + n).
    [[nodiscard]] GdnInputs rows(std::size_t t0, std::size_t n) const {
        const std::size_t nv = dims.n_v_heads;
        return {ConstRows(q.data() + t0 * qk_cols(), n, qk_cols(), qk_cols()),
                ConstRows(k.data() + t0 * qk_cols(), n, qk_cols(), qk_cols()),
                ConstRows(v.data() + t0 * v_cols(), n, v_cols(), v_cols()),
                ConstRows(g.data() + t0 * nv, n, nv, nv), ConstRows(beta.data() + t0 * nv, n, nv, nv)};
    }
};

float softplus_f(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

Data make(GdnDims dims, std::size_t T, Regime regime, bool init, std::uint64_t seed) {
    Rng rng(seed);
    Data d;
    d.dims = dims;
    d.T = T;
    const std::size_t nv = dims.n_v_heads;
    d.q = rng.normal(T * d.qk_cols());
    d.k = rng.normal(T * d.qk_cols());
    d.v = rng.normal(T * d.v_cols());
    d.g.resize(T * nv);
    d.beta.resize(T * nv);
    auto a = rng.normal(T * nv);
    auto dt = rng.normal(nv, 0.5f);
    auto bz = rng.normal(T * nv, 2.0f);
    auto u = rng.uniform(T * nv, 0.0f, 1.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t j = 0; j < nv; ++j) {
            const std::size_t i = t * nv + j;
            // Model: A spans weak (0.01) to very strong (16) decay across heads, as HF init.
            const float A = nv == 1 ? 1.0f : 0.01f + (16.0f - 0.01f) * static_cast<float>(j) / static_cast<float>(nv - 1);
            d.g[i] = -A * softplus_f(a[i] + dt[j]);
            d.beta[i] = 1.0f / (1.0f + std::exp(-bz[i]));
            switch (regime) {
                case Regime::Model: break;
                case Regime::WeakDecay: d.g[i] = -1e-3f * u[i]; break;
                case Regime::StrongDecay: d.g[i] = -10.0f - 20.0f * u[i]; break;  // cumsum underflows exp
                case Regime::BetaLow: d.beta[i] = 1e-3f * u[i]; break;
                case Regime::BetaHigh: d.beta[i] = 1.0f - 1e-3f * u[i]; break;
            }
        }
    }
    d.s0 = init ? rng.normal(d.state_n(), 0.5f) : std::vector<float>(d.state_n(), 0.0f);
    return d;
}

struct Result {
    std::vector<float> out, state, slots;
};

enum class Form { Recurrent, Chunked };

Result run(const Data& d, Form form, std::size_t t0, std::size_t n, std::vector<float> state, std::size_t chunk = 64,
           ThreadPool* pool = nullptr, std::size_t n_slots = 0, bool l2 = true) {
    Result r;
    r.out.assign(n * d.v_cols(), 0.0f);
    r.slots.assign(n_slots * d.state_n(), 12345.0f);  // sentinel: unwritten slots keep it
    Rows out(r.out.data(), n, d.v_cols(), d.v_cols());
    if (form == Form::Recurrent)
        gated_delta_rule_recurrent(d.dims, d.rows(t0, n), state, out, l2, pool, r.slots);
    else
        gated_delta_rule_chunked(d.dims, d.rows(t0, n), state, out, l2, chunk, pool, r.slots);
    r.state = std::move(state);
    return r;
}

struct Ref {
    std::vector<double> out, state, out_terms;
};

Ref ref(const Data& d, std::size_t n) {
    Ref r;
    r.state.assign(d.s0.begin(), d.s0.end());
    const GdnRefDims rd{d.dims.n_k_heads, d.dims.n_v_heads, d.dims.d_k, d.dims.d_v,
                        d.dims.mapping == GdnHeadMapping::Tiled};
    r.out = gdn_ref(rd, n, d.q, d.k, d.v, d.g, d.beta, r.state, true, &r.out_terms);
    return r;
}

/// Max over heads of (err / Model-G tol) for outputs and state; asserts <= 1 (x factor).
/// out_terms: fp64 per-element sum of |terms| of the output dot product (Model G scale).
double check_vs(const Data& d, std::size_t T, const std::vector<float>& out, const std::vector<float>& state,
                const std::vector<double>& ref_out, const std::vector<double>& ref_state,
                const std::vector<double>& out_terms, double factor, const std::string& what) {
    const std::size_t nv = d.dims.n_v_heads, dv = d.dims.d_v, sh = d.dims.d_k * dv;
    double worst = 0;
    const std::size_t t_eff = std::max<std::size_t>(T, 1);
    for (std::size_t j = 0; j < nv; ++j) {
        double so = 0, eo = 0, ss = 0, es = 0;
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t c = 0; c < dv; ++c) {
                const std::size_t i = t * nv * dv + j * dv + c;
                so = std::max(so, out_terms[i]);
                eo = std::max(eo, std::abs(out[i] - ref_out[i]));
            }
        for (std::size_t e = 0; e < sh; ++e) {
            ss = std::max(ss, std::abs(ref_state[j * sh + e]));
            es = std::max(es, std::abs(state[j * sh + e] - ref_state[j * sh + e]));
        }
        const double to = factor * tol_gdn(t_eff, d.dims.d_k, so) + kDenormFloor;
        const double ts = factor * tol_gdn(t_eff, d.dims.d_k, ss) + kDenormFloor;
        EXPECT_LE(eo, to) << what << " head " << j << " out err " << eo << " scale " << so;
        EXPECT_LE(es, ts) << what << " head " << j << " state err " << es << " scale " << ss;
        worst = std::max({worst, eo / to, es / ts});
    }
    return worst;
}

/// State-only check after `tokens` processed rows (Model G at that length).
double check_state(const Data& d, std::size_t tokens, std::span<const float> state, const std::vector<double>& ref_state,
                   double factor, const std::string& what) {
    const std::size_t sh = d.dims.d_k * d.dims.d_v;
    double worst = 0;
    for (std::size_t j = 0; j < d.dims.n_v_heads; ++j) {
        double ss = 0, es = 0;
        for (std::size_t e = 0; e < sh; ++e) {
            ss = std::max(ss, std::abs(ref_state[j * sh + e]));
            es = std::max(es, std::abs(state[j * sh + e] - ref_state[j * sh + e]));
        }
        const double ts = factor * tol_gdn(tokens, d.dims.d_k, ss) + kDenormFloor;
        EXPECT_LE(es, ts) << what << " head " << j << " state err " << es << " scale " << ss;
        worst = std::max(worst, es / ts);
    }
    return worst;
}

std::vector<double> to_d(const std::vector<float>& v) { return {v.begin(), v.end()}; }

const GdnDims kR1{2, 2, 16, 16, GdnHeadMapping::Tiled};
const GdnDims kR3{2, 6, 32, 32, GdnHeadMapping::Tiled};
const GdnDims kRect{2, 6, 16, 24, GdnHeadMapping::Tiled};

GdnDims with(GdnDims d, GdnHeadMapping m) {
    d.mapping = m;
    return d;
}

// ------------------------------------------------------------------------------ sweeps

TEST(CpuGdn, ChunkedAndRecurrentMatchFp64AcrossLengths) {
    double worst_rec = 0, worst_ch = 0, worst_cross = 0;
    std::uint64_t seed = 100;
    for (const GdnDims& base : {kR1, kR3, kRect}) {
        for (GdnHeadMapping m : {GdnHeadMapping::Tiled, GdnHeadMapping::Grouped}) {
            for (std::size_t T : {0u, 1u, 63u, 64u, 65u, 128u, 129u, 300u}) {
                for (bool init : {false, true}) {
                    const Data d = make(with(base, m), T, Regime::Model, init, ++seed);
                    const std::string what = "nv" + std::to_string(base.n_v_heads) + " dk" + std::to_string(base.d_k) +
                                             " dv" + std::to_string(base.d_v) +
                                             (m == GdnHeadMapping::Tiled ? " tiled" : " grouped") + " T" +
                                             std::to_string(T) + (init ? " init" : " zero");
                    const Ref r = ref(d, T);
                    const Result rec = run(d, Form::Recurrent, 0, T, d.s0);
                    const Result ch = run(d, Form::Chunked, 0, T, d.s0);
                    worst_rec = std::max(worst_rec, check_vs(d, T, rec.out, rec.state, r.out, r.state, r.out_terms, 1.0, what + " rec"));
                    worst_ch = std::max(worst_ch, check_vs(d, T, ch.out, ch.state, r.out, r.state, r.out_terms, 1.0, what + " chunk"));
                    worst_cross = std::max(worst_cross, check_vs(d, T, ch.out, ch.state, to_d(rec.out), to_d(rec.state),
                                                                 r.out_terms, 2.0, what + " chunk-vs-rec"));
                    if (T == 0) {
                        EXPECT_TRUE(bitwise_equal(rec.state, d.s0));
                        EXPECT_TRUE(bitwise_equal(ch.state, d.s0));
                    }
                }
            }
        }
    }
    std::printf("[tol] gdn sweep worst err/tol: recurrent %.3g, chunked %.3g, chunked-vs-recurrent (2x) %.3g\n",
                worst_rec, worst_ch, worst_cross);
}

TEST(CpuGdn, ChunkSizesIncludingOneAndLargerThanSequence) {
    double worst = 0;
    std::uint64_t seed = 500;
    for (std::size_t T : {1u, 65u, 129u, 300u}) {
        const Data d = make(kR3, T, Regime::Model, true, ++seed);
        const Ref r = ref(d, T);
        for (std::size_t cs : {1u, 16u, 64u, 128u}) {
            const Result ch = run(d, Form::Chunked, 0, T, d.s0, cs);
            worst = std::max(worst, check_vs(d, T, ch.out, ch.state, r.out, r.state, r.out_terms, 1.0,
                                             "T" + std::to_string(T) + " cs" + std::to_string(cs)));
        }
    }
    std::printf("[tol] gdn chunk sizes worst err/tol %.3g\n", worst);
}

TEST(CpuGdn, NumericalRegimes) {
    std::uint64_t seed = 700;
    for (Regime rg : {Regime::WeakDecay, Regime::StrongDecay, Regime::BetaLow, Regime::BetaHigh}) {
        const Data d = make(kR3, 300, rg, true, ++seed);
        const Ref r = ref(d, 300);
        const Result rec = run(d, Form::Recurrent, 0, 300, d.s0);
        const Result ch = run(d, Form::Chunked, 0, 300, d.s0);
        for (float e : ch.out) ASSERT_TRUE(std::isfinite(e)) << regime_name(rg);
        for (float e : ch.state) ASSERT_TRUE(std::isfinite(e)) << regime_name(rg);
        const double wr = check_vs(d, 300, rec.out, rec.state, r.out, r.state, r.out_terms, 1.0, std::string(regime_name(rg)) + " rec");
        const double wc = check_vs(d, 300, ch.out, ch.state, r.out, r.state, r.out_terms, 1.0, std::string(regime_name(rg)) + " chunk");
        std::printf("[tol] gdn regime %-12s worst err/tol: recurrent %.3g chunked %.3g\n", regime_name(rg), wr, wc);
    }
}

TEST(CpuGdn, RealLayerDimsShortSequence) {
    const GdnDims real{16, 48, 128, 128, GdnHeadMapping::Tiled};
    const Data d = make(real, 70, Regime::Model, true, 900);
    const Ref r = ref(d, 70);
    const Result rec = run(d, Form::Recurrent, 0, 70, d.s0);
    const Result ch = run(d, Form::Chunked, 0, 70, d.s0);
    const double wr = check_vs(d, 70, rec.out, rec.state, r.out, r.state, r.out_terms, 1.0, "real rec");
    const double wc = check_vs(d, 70, ch.out, ch.state, r.out, r.state, r.out_terms, 1.0, "real chunk");
    std::printf("[tol] gdn real dims (48v/16k/128) T=70 worst err/tol: recurrent %.3g chunked %.3g\n", wr, wc);
}

// ------------------------------------------------------------------- state continuity

TEST(CpuGdn, StateContinuityUnderArbitrarySplits) {
    // prefill 100 + decode 1 x 50 + prefill 37, and an irregular split crossing boundaries.
    std::vector<std::size_t> split_a{100};
    split_a.insert(split_a.end(), 50, 1);
    split_a.push_back(37);
    const std::vector<std::size_t> split_b{1, 63, 64, 2, 57, 1, 1, 70};  // sums to 259
    for (const auto& split : {split_a, split_b}) {
        std::size_t T = 0;
        for (auto n : split) T += n;
        const Data d = make(kR3, T, Regime::Model, true, 1000 + T);
        const Ref r = ref(d, T);
        const Result one_ch = run(d, Form::Chunked, 0, T, d.s0);
        const Result one_rec = run(d, Form::Recurrent, 0, T, d.s0);
        // Mixed: chunked for multi-token pieces (prefill), recurrent for single tokens (decode).
        std::vector<float> state = d.s0, out(T * d.v_cols());
        std::vector<float> rec_state = d.s0, rec_out(T * d.v_cols());
        std::size_t t0 = 0;
        for (std::size_t n : split) {
            const Result piece = run(d, n == 1 ? Form::Recurrent : Form::Chunked, t0, n, state);
            std::copy(piece.out.begin(), piece.out.end(), out.begin() + static_cast<std::ptrdiff_t>(t0 * d.v_cols()));
            state = piece.state;
            const Result rp = run(d, Form::Recurrent, t0, n, rec_state);
            std::copy(rp.out.begin(), rp.out.end(), rec_out.begin() + static_cast<std::ptrdiff_t>(t0 * d.v_cols()));
            rec_state = rp.state;
            t0 += n;
        }
        const std::string what = "split T=" + std::to_string(T);
        const double w1 = check_vs(d, T, out, state, r.out, r.state, r.out_terms, 1.0, what + " mixed vs fp64");
        const double w2 = check_vs(d, T, out, state, to_d(one_ch.out), to_d(one_ch.state), r.out_terms, 2.0, what + " mixed vs one-pass chunked");
        const double w3 = check_vs(d, T, out, state, to_d(one_rec.out), to_d(one_rec.state), r.out_terms, 2.0, what + " mixed vs one-pass recurrent");
        std::printf("[tol] gdn %s worst err/tol: vs fp64 %.3g, vs chunked %.3g, vs recurrent %.3g\n", what.c_str(), w1, w2, w3);
        // The recurrent form is split-invariant bit for bit.
        EXPECT_TRUE(bitwise_equal(rec_out, one_rec.out)) << what;
        EXPECT_TRUE(bitwise_equal(rec_state, one_rec.state)) << what;
    }
}

TEST(CpuGdn, ChunkedSplitAtChunkBoundariesIsBitwise) {
    const Data d = make(kR3, 200, Regime::Model, true, 1200);
    const Result one = run(d, Form::Chunked, 0, 200, d.s0, 64);
    const Result a = run(d, Form::Chunked, 0, 128, d.s0, 64);
    const Result b = run(d, Form::Chunked, 128, 72, a.state, 64);
    std::vector<float> out = a.out;
    out.insert(out.end(), b.out.begin(), b.out.end());
    EXPECT_TRUE(bitwise_equal(out, one.out));
    EXPECT_TRUE(bitwise_equal(b.state, one.state));
}

// ---------------------------------------------------------------------- head mapping

/// grouped head index -> tiled head index (llama.cpp _reorder_v_heads).
std::size_t tiled_index(std::size_t grouped, std::size_t n_k, std::size_t ratio) {
    return (grouped % ratio) * n_k + grouped / ratio;
}

TEST(CpuGdn, TiledOnPermutedHeadsEqualsGroupedBitwise) {
    for (const GdnDims& base : {kR3, kRect}) {
        const std::size_t nk = base.n_k_heads, nv = base.n_v_heads, ratio = nv / nk, dv = base.d_v, sh = base.d_k * dv;
        const Data gd = make(with(base, GdnHeadMapping::Grouped), 150, Regime::Model, true, 1300 + dv);
        Data td = gd;
        td.dims.mapping = GdnHeadMapping::Tiled;
        for (std::size_t t = 0; t < gd.T; ++t)
            for (std::size_t j = 0; j < nv; ++j) {
                const std::size_t tj = tiled_index(j, nk, ratio);
                std::copy_n(&gd.v[t * nv * dv + j * dv], dv, &td.v[t * nv * dv + tj * dv]);
                td.g[t * nv + tj] = gd.g[t * nv + j];
                td.beta[t * nv + tj] = gd.beta[t * nv + j];
            }
        for (std::size_t j = 0; j < nv; ++j) std::copy_n(&gd.s0[j * sh], sh, &td.s0[tiled_index(j, nk, ratio) * sh]);
        for (Form f : {Form::Recurrent, Form::Chunked}) {
            const Result g = run(gd, f, 0, gd.T, gd.s0);
            const Result t = run(td, f, 0, td.T, td.s0);
            for (std::size_t j = 0; j < nv; ++j) {
                const std::size_t tj = tiled_index(j, nk, ratio);
                for (std::size_t tt = 0; tt < gd.T; ++tt)
                    ASSERT_TRUE(bitwise_equal(std::span(g.out).subspan(tt * nv * dv + j * dv, dv),
                                              std::span(t.out).subspan(tt * nv * dv + tj * dv, dv)))
                        << "head " << j << " token " << tt;
                ASSERT_TRUE(bitwise_equal(std::span(g.state).subspan(j * sh, sh), std::span(t.state).subspan(tj * sh, sh)));
            }
        }
        // And the mappings themselves.
        for (std::size_t j = 0; j < nv; ++j) {
            EXPECT_EQ(gdn_k_head(with(base, GdnHeadMapping::Tiled), j), j % nk);
            EXPECT_EQ(gdn_k_head(with(base, GdnHeadMapping::Grouped), j), j / ratio);
        }
    }
}

TEST(CpuGdn, PrenormalizedInputsEqualInKernelL2Bitwise) {
    Data d = make(kR3, 70, Regime::Model, true, 1400);
    Data pre = d;
    l2_norm_heads(ConstRows(std::span<const float>(d.q), d.T, d.qk_cols()), kR3.n_k_heads, kR3.d_k,
                  Rows(std::span<float>(pre.q), d.T, d.qk_cols()));
    l2_norm_heads(ConstRows(std::span<const float>(d.k), d.T, d.qk_cols()), kR3.n_k_heads, kR3.d_k,
                  Rows(std::span<float>(pre.k), d.T, d.qk_cols()));
    for (Form f : {Form::Recurrent, Form::Chunked}) {
        const Result a = run(d, f, 0, d.T, d.s0, 64, nullptr, 0, true);
        const Result b = run(pre, f, 0, d.T, d.s0, 64, nullptr, 0, false);
        EXPECT_TRUE(bitwise_equal(a.out, b.out));
        EXPECT_TRUE(bitwise_equal(a.state, b.state));
    }
}

// --------------------------------------------------------------------- rollback slots

TEST(CpuGdn, SlotsEqualFinalStateOfTruncatedRunBitwise) {
    constexpr std::size_t kSlots = 4;
    for (Form f : {Form::Recurrent, Form::Chunked}) {
        for (std::size_t T : {1u, 2u, 5u, 17u, 40u}) {
            const Data d = make(kR3, T, Regime::Model, true, 1500 + T);
            // chunk 16: the last rows of T=17/40 straddle a chunk boundary.
            const Result full = run(d, f, 0, T, d.s0, 16, nullptr, kSlots);
            const Result no_slots = run(d, f, 0, T, d.s0, 16, nullptr, 0);
            EXPECT_TRUE(bitwise_equal(full.out, no_slots.out)) << "slots changed out";
            EXPECT_TRUE(bitwise_equal(full.state, no_slots.state)) << "slots changed state";
            const std::size_t sn = d.state_n();
            for (std::size_t s = 0; s < kSlots; ++s) {
                const auto slot = std::span(full.slots).subspan(s * sn, sn);
                if (s >= T) {
                    for (float e : slot) ASSERT_EQ(e, 12345.0f) << "slot " << s << " >= T was written";
                    continue;
                }
                const std::size_t rows = T - s;  // rows [0, T-1-s]
                const Result trunc = run(d, f, 0, rows, d.s0, 16);
                EXPECT_TRUE(bitwise_equal(slot, trunc.state))
                    << (f == Form::Chunked ? "chunked" : "recurrent") << " T=" << T << " slot " << s;
            }
        }
    }
}

TEST(CpuGdn, RollbackTakeSlotEqualsDecodingFewerRows) {
    constexpr std::size_t kSlots = 4, T = 23;
    const Data d = make(kR3, T, Regime::Model, true, 1600);
    const Result verify_rec = run(d, Form::Recurrent, 0, T, d.s0, 64, nullptr, kSlots);
    const Result verify_ch = run(d, Form::Chunked, 0, T, d.s0, 64, nullptr, kSlots);
    const std::size_t sn = d.state_n();
    for (std::size_t j = 0; j < kSlots; ++j) {
        // "decode T - j rows": one recurrent call per row.
        std::vector<float> st = d.s0;
        for (std::size_t t = 0; t < T - j; ++t) st = run(d, Form::Recurrent, t, 1, st).state;
        EXPECT_TRUE(bitwise_equal(std::span(verify_rec.slots).subspan(j * sn, sn), st)) << "recurrent slot " << j;
        // Chunked verify (prefill-style) vs decode: different forms, Model-G tolerance.
        const Ref r = ref(d, T - j);
        const auto ch_slot = std::span<const float>(verify_ch.slots).subspan(j * sn, sn);
        const double w1 = check_state(d, T - j, ch_slot, to_d(st), 2.0, "chunked slot " + std::to_string(j) + " vs decode");
        const double w2 = check_state(d, T - j, ch_slot, r.state, 1.0, "chunked slot " + std::to_string(j) + " vs fp64");
        std::printf("[tol] gdn rollback slot %zu (chunked verify) worst err/tol: vs decode %.3g, vs fp64 %.3g\n", j, w1, w2);
    }
}

// ------------------------------------------------------------- thread-count invariance

TEST(CpuGdn, ThreadCountInvarianceBitwise) {
    const Data d = make(kR3, 150, Regime::Model, true, 1700);
    const Result base_rec = run(d, Form::Recurrent, 0, d.T, d.s0, 64, nullptr, 3);
    const Result base_ch = run(d, Form::Chunked, 0, d.T, d.s0, 64, nullptr, 3);
    for (std::size_t threads : {1u, 2u, 3u, 7u, 16u, 64u}) {
        ThreadPool pool(threads);
        const Result rec = run(d, Form::Recurrent, 0, d.T, d.s0, 64, &pool, 3);
        const Result ch = run(d, Form::Chunked, 0, d.T, d.s0, 64, &pool, 3);
        EXPECT_TRUE(bitwise_equal(rec.out, base_rec.out) && bitwise_equal(rec.state, base_rec.state) &&
                    bitwise_equal(rec.slots, base_rec.slots))
            << "recurrent threads=" << threads;
        EXPECT_TRUE(bitwise_equal(ch.out, base_ch.out) && bitwise_equal(ch.state, base_ch.state) &&
                    bitwise_equal(ch.slots, base_ch.slots))
            << "chunked threads=" << threads;
    }
}

// -------------------------------------------------------------------------- validation

TEST(CpuGdn, RejectsBadShapesOverlapAndPositiveDecay) {
    Data d = make(kR3, 10, Regime::Model, true, 1800);
    std::vector<float> state = d.s0, out(d.T * d.v_cols());
    Rows o(std::span<float>(out), d.T, d.v_cols());
    GdnDims bad = kR3;
    bad.n_v_heads = 5;
    EXPECT_THROW(gated_delta_rule_recurrent(bad, d.rows(0, d.T), state, o, true), halo::Error);
    std::vector<float> small_state(state.size() - 1);
    EXPECT_THROW(gated_delta_rule_recurrent(kR3, d.rows(0, d.T), small_state, o, true), halo::Error);
    // out overlapping the state.
    std::vector<float> both(state.size() + out.size());
    EXPECT_THROW(gated_delta_rule_recurrent(kR3, d.rows(0, d.T), std::span(both).first(state.size()),
                                            Rows(both.data() + 16, d.T, d.v_cols(), d.v_cols()), true),
                 halo::Error);
    std::vector<float> slots(state.size() * 2 + 1);
    EXPECT_THROW(gated_delta_rule_recurrent(kR3, d.rows(0, d.T), state, o, true, nullptr, slots), halo::Error);
    EXPECT_THROW(gated_delta_rule_chunked(kR3, d.rows(0, d.T), state, o, true, 0), halo::Error);
    d.g[7] = 0.25f;
    EXPECT_THROW(gated_delta_rule_chunked(kR3, d.rows(0, d.T), state, o, true), halo::Error);
    EXPECT_NO_THROW(gated_delta_rule_recurrent(kR3, d.rows(0, d.T), state, o, true));
}

TEST(CpuTraffic, GdnAndConvCompulsoryBytes) {
    const GdnDims real{16, 48, 128, 128, GdnHeadMapping::Tiled};
    const std::uint64_t state = 48ull * 128 * 128 * 4;  // 3 MiB per layer per sequence (D-003)
    EXPECT_EQ(state, 3ull << 20);
    const OpTraffic t1 = traffic_gated_delta_rule(real, 1);
    EXPECT_EQ(t1.bytes_read, 4ull * (2 * 2048 + 6144 + 96) + state);
    EXPECT_EQ(t1.bytes_written, 4ull * 6144 + state);
    // Slots: only min(T, n_slots) are written.
    EXPECT_EQ(traffic_gated_delta_rule(real, 1, 3).bytes_written, t1.bytes_written + state);
    EXPECT_EQ(traffic_gated_delta_rule(real, 5, 3).bytes_written, 4ull * 5 * 6144 + 4 * state);
    const OpTraffic c = traffic_causal_conv1d(1, 10240, 4, 2);
    EXPECT_EQ(c.bytes_read, 4ull * (10240 + 10240 * 4 + 3 * 10240));
    EXPECT_EQ(c.bytes_written, 4ull * (10240 + 3 * 10240 + 1 * 3 * 10240));
    // LM head, Q6_K 5120 x 248320 (D-007): 210 B per 256 weights.
    const std::uint64_t head_bytes = 5120ull * 248320 / 256 * 210;
    EXPECT_EQ(traffic_matmul_argmax(5120, head_bytes).bytes_read, 4ull * 5120 + head_bytes);
    EXPECT_EQ(traffic_attention({24, 4, 256}, 1, 99).bytes_read, 4ull * (24 * 256 + 2 * 100 * 4 * 256));
}

}  // namespace
