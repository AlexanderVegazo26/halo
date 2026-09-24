// Sampler chain tests (WS-H, TRD §23). No reference data needed.
//
// Every filter is checked against a direct reference written here (full sort + prefix
// scan, independent of the implementation's nth_element / index sorts), and the complete
// chain is checked statistically: with fixed logits the empirical token frequencies must
// match the analytic post-filter distribution (Pearson chi-square, alpha = 1e-3, bins with
// expected count < 5 merged, fixed seed so the test is deterministic).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "halo/core/error.h"
#include "halo/sampling/sampler.h"

namespace {

using halo::Error;
using halo::ErrorCode;
using halo::SamplingParams;
using halo::sampling::CandidateSet;
using halo::sampling::exact_prefilter_ok;
using halo::sampling::Sampler;
using halo::sampling::SamplerConfig;
namespace chain = halo::sampling::chain;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

template <typename F>
ErrorCode code_of(F&& f) {
    try {
        f();
    } catch (const Error& e) {
        return e.code();
    }
    ADD_FAILURE() << "expected halo::Error";
    return ErrorCode::Cancelled;
}

Sampler make(const SamplingParams& p, std::size_t vocab) {
    SamplerConfig cfg;
    cfg.vocab_size = vocab;
    return Sampler::create(p, nullptr, cfg);
}

std::vector<chain::Cand> cands_of(const std::vector<double>& logits) {
    std::vector<chain::Cand> c;
    for (std::size_t i = 0; i < logits.size(); ++i) c.push_back({static_cast<std::int32_t>(i), logits[i]});
    return c;
}

std::vector<std::int32_t> ids_of(std::vector<chain::Cand> c) {
    std::vector<std::int32_t> ids;
    for (const auto& x : c) ids.push_back(x.id);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// ---- direct references (probabilities by explicit softmax over a sorted copy) --------

struct Ref {
    std::int32_t id;
    double logit;
};

std::vector<double> ref_softmax(const std::vector<Ref>& v) {
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& x : v) m = std::max(m, x.logit);
    std::vector<double> p;
    double s = 0;
    for (const auto& x : v) {
        p.push_back(std::exp(x.logit - m));
        s += p.back();
    }
    for (auto& x : p) x /= s;
    return p;
}

std::vector<Ref> ref_sorted_desc(std::vector<Ref> v) {
    std::stable_sort(v.begin(), v.end(), [](const Ref& a, const Ref& b) {
        return a.logit != b.logit ? a.logit > b.logit : a.id < b.id;
    });
    return v;
}

std::vector<Ref> ref_top_k(std::vector<Ref> v, int k) {
    if (k <= 0 || static_cast<std::size_t>(k) >= v.size()) return v;
    v = ref_sorted_desc(v);
    v.resize(static_cast<std::size_t>(k));
    return v;
}

std::vector<Ref> ref_top_p(std::vector<Ref> v, double p) {
    if (p >= 1.0) return v;
    v = ref_sorted_desc(v);
    const auto pr = ref_softmax(v);
    double cum = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        cum += pr[i];
        if (cum >= p) {
            v.resize(i + 1);
            break;
        }
    }
    return v;
}

std::vector<Ref> ref_min_p(std::vector<Ref> v, double p) {
    if (p <= 0.0) return v;
    const auto pr = ref_softmax(v);
    const double pmax = *std::max_element(pr.begin(), pr.end());
    std::vector<Ref> out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (pr[i] >= p * pmax * (1 - 1e-12)) out.push_back(v[i]);
    }
    return out;
}

std::vector<Ref> ref_typical(std::vector<Ref> v, double p) {
    if (p >= 1.0 || v.size() <= 1) return v;
    const auto pr = ref_softmax(v);
    double h = 0;
    for (const double x : pr) h -= x > 0 ? x * std::log(x) : 0.0;
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        const double sa = std::fabs(-std::log(pr[a]) - h), sb = std::fabs(-std::log(pr[b]) - h);
        return sa != sb ? sa < sb : v[a].id < v[b].id;
    });
    std::vector<Ref> out;
    double cum = 0;
    for (const auto i : idx) {
        out.push_back(v[i]);
        cum += pr[i];
        if (cum > p) break;
    }
    return out;
}

/// Analytic distribution of the whole chain (no penalties): id -> probability.
std::map<std::int32_t, double> ref_distribution(const std::vector<float>& logits, const SamplingParams& p) {
    std::vector<Ref> v;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (std::isinf(logits[i])) continue;
        v.push_back({static_cast<std::int32_t>(i), static_cast<double>(logits[i]) / static_cast<double>(p.temperature)});
    }
    v = ref_top_k(v, p.top_k);
    v = ref_typical(v, static_cast<double>(p.typical_p));
    v = ref_top_p(v, static_cast<double>(p.top_p));
    v = ref_min_p(v, static_cast<double>(p.min_p));
    const auto pr = ref_softmax(v);
    std::map<std::int32_t, double> out;
    for (std::size_t i = 0; i < v.size(); ++i) out[v[i].id] = pr[i];
    return out;
}

/// Upper alpha = 1e-3 critical value of chi-square(df) by Wilson-Hilferty (z = 3.0902).
double chi2_critical(int df) {
    const double d = df;
    const double z = 3.0902;
    const double a = 2.0 / (9.0 * d);
    return d * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

struct Chi2 {
    double stat = 0;
    int df = 0;
    std::size_t outside = 0;  // draws of ids with zero analytic probability
};

Chi2 chi_square(const std::map<std::int32_t, double>& expect, const std::map<std::int32_t, std::size_t>& got,
                std::size_t n) {
    Chi2 r;
    for (const auto& [id, c] : got) {
        if (!expect.contains(id)) r.outside += c;
    }
    // Merge bins with expected count < 5 into one pooled bin.
    double pool_e = 0, pool_o = 0;
    int bins = 0;
    for (const auto& [id, p] : expect) {
        const double e = p * static_cast<double>(n);
        const auto it = got.find(id);
        const double o = it == got.end() ? 0.0 : static_cast<double>(it->second);
        if (e < 5.0) {
            pool_e += e;
            pool_o += o;
            continue;
        }
        r.stat += (o - e) * (o - e) / e;
        ++bins;
    }
    if (pool_e >= 5.0) {
        r.stat += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        ++bins;
    }
    r.df = std::max(1, bins - 1);
    return r;
}

std::vector<float> fixed_logits() {
    // 16 tokens: ties (ids 2/3, 9/10), a -inf, a spread of magnitudes.
    return {1.5f, 0.2f, 2.0f, 2.0f, -1.0f, 0.7f, kNegInf, 3.1f, -2.5f, 1.1f, 1.1f, 0.0f, -0.4f, 2.6f, -3.0f, 0.9f};
}

// ---------------------------------------------------------------------------------------

TEST(SamplerParams, ValidateRejectsBadValuesWithApi) {
    const auto bad = [](auto mutate) {
        SamplingParams p;
        mutate(p);
        return code_of([&] { halo::sampling::validate(p); });
    };
    EXPECT_EQ(bad([](SamplingParams& p) { p.temperature = NAN; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.temperature = INFINITY; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.top_k = -1; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.top_p = 1.5f; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.top_p = NAN; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.min_p = -0.1f; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.typical_p = 2.0f; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.repetition_penalty = 0.0f; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.presence_penalty = NAN; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.frequency_penalty = INFINITY; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) { p.penalty_last_n = -2; }), ErrorCode::Api);
    EXPECT_EQ(bad([](SamplingParams& p) {
                  p.json_object = true;
                  p.json_schema = "{}";
              }),
              ErrorCode::Api);
    SamplingParams ok;
    EXPECT_NO_THROW(halo::sampling::validate(ok));
    // Sampler::create validates too.
    SamplingParams p;
    p.top_p = -1.0f;
    EXPECT_EQ(code_of([&] { (void)make(p, 8); }), ErrorCode::Api);
    EXPECT_EQ(code_of([&] { (void)Sampler::create(SamplingParams{}, nullptr); }), ErrorCode::Api);  // no vocab
}

TEST(SamplerInput, NanAndPosInfAreKernelErrorsNegInfIsLegal) {
    SamplingParams p;
    p.seed = 1;
    auto s = make(p, 4);
    const std::vector<std::int32_t> none;
    std::vector<float> row{0.0f, NAN, 1.0f, 2.0f};
    EXPECT_EQ(code_of([&] { (void)s.sample(row, none); }), ErrorCode::Kernel);
    row[1] = INFINITY;
    EXPECT_EQ(code_of([&] { (void)s.sample(row, none); }), ErrorCode::Kernel);
    row = {kNegInf, kNegInf, kNegInf, kNegInf};
    EXPECT_EQ(code_of([&] { (void)s.sample(row, none); }), ErrorCode::Api);
    row = {0.0f, 1.0f};  // wrong length
    EXPECT_EQ(code_of([&] { (void)s.sample(row, none); }), ErrorCode::Api);
    // Candidate-set validation.
    const std::vector<std::int32_t> ids{0, 2, 2};
    const std::vector<float> lg{1.0f, 2.0f, 3.0f};
    EXPECT_EQ(code_of([&] { (void)s.sample(CandidateSet{ids, lg}, none); }), ErrorCode::Api);  // duplicate
    const std::vector<std::int32_t> ids_oob{0, 4};
    const std::vector<float> lg2{1.0f, 2.0f};
    EXPECT_EQ(code_of([&] { (void)s.sample(CandidateSet{ids_oob, lg2}, none); }), ErrorCode::Api);
    const std::vector<std::int32_t> ids_ok{0, 1};
    const std::vector<float> lg_nan{1.0f, NAN};
    EXPECT_EQ(code_of([&] { (void)s.sample(CandidateSet{ids_ok, lg_nan}, none); }), ErrorCode::Kernel);
    EXPECT_EQ(code_of([&] { (void)s.sample(CandidateSet{ids_ok, lg}, none); }), ErrorCode::Api);  // size mismatch
    const std::vector<std::int32_t> bad_hist{9};
    const std::vector<float> row4{0.0f, 1.0f, 2.0f, 3.0f};
    EXPECT_EQ(code_of([&] { (void)s.sample(row4, bad_hist); }), ErrorCode::Api);
}

TEST(SamplerInput, AllNegInfExceptOneAlwaysPicksIt) {
    for (const float temp : {0.0f, 0.3f, 1.0f, 5.0f}) {
        SamplingParams p;
        p.temperature = temp;
        p.top_p = 0.5f;
        p.min_p = 0.2f;
        p.typical_p = 0.3f;
        p.seed = 7;
        auto s = make(p, 1000);
        std::vector<float> row(1000, kNegInf);
        row[613] = -50.0f;
        for (int i = 0; i < 200; ++i) ASSERT_EQ(s.sample(row, {}), 613) << "temperature " << temp;
    }
}

TEST(SamplerGreedy, ExactArgmaxTiesToLowestId) {
    SamplingParams p;
    p.temperature = 0.0f;
    auto s = make(p, 6);
    const std::vector<float> row{1.0f, 3.0f, -2.0f, 3.0f, 3.0f, 0.5f};
    EXPECT_EQ(s.sample(row, {}), 1);
    // Greedy consumes no randomness and ignores truncation filters.
    p.top_k = 1;
    p.top_p = 0.01f;
    auto s2 = make(p, 6);
    EXPECT_EQ(s2.sample(row, {}), 1);
    // Candidate path: same answer regardless of candidate order.
    const std::vector<std::int32_t> ids{4, 3, 1, 0};
    const std::vector<float> lg{3.0f, 3.0f, 3.0f, 1.0f};
    EXPECT_EQ(s.sample(CandidateSet{ids, lg}, {}), 1);
    // Tiny positive temperature is sampling, not greedy, and stays finite.
    p = SamplingParams{};
    p.temperature = 1e-6f;
    p.seed = 3;
    auto s3 = make(p, 6);
    const std::vector<float> row2{1.0f, 3.0f, -2.0f, 2.999f, 0.0f, 0.5f};
    for (int i = 0; i < 50; ++i) EXPECT_EQ(s3.sample(row2, {}), 1);
}

TEST(SamplerPenalties, WindowCountsSemantics) {
    const std::vector<std::int32_t> h{5, 1, 5, 2, 5, 1};
    using V = std::vector<std::pair<std::int32_t, std::int32_t>>;
    EXPECT_EQ(chain::window_counts(h, 0), V{});
    EXPECT_EQ(chain::window_counts(h, -1), (V{{1, 2}, {2, 1}, {5, 3}}));
    EXPECT_EQ(chain::window_counts(h, 3), (V{{1, 1}, {2, 1}, {5, 1}}));  // last 3: 2, 5, 1
    EXPECT_EQ(chain::window_counts(h, 100), (V{{1, 2}, {2, 1}, {5, 3}}));
}

TEST(SamplerPenalties, MatchLlamaCppFormula) {
    SamplingParams p;
    p.repetition_penalty = 1.5f;
    p.frequency_penalty = 0.25f;
    p.presence_penalty = 0.5f;
    auto c = cands_of({2.0, -1.0, 0.0, 4.0, 1.0});
    const std::vector<std::pair<std::int32_t, std::int32_t>> counts{{0, 3}, {1, 1}, {2, 2}, {7, 4}};
    chain::apply_penalties(c, p, counts);
    // logit>0: divide; logit<=0: multiply; then - c*freq - presence (once).
    EXPECT_DOUBLE_EQ(c[0].logit, 2.0 / 1.5 - 3 * 0.25 - 0.5);
    EXPECT_DOUBLE_EQ(c[1].logit, -1.0 * 1.5 - 1 * 0.25 - 0.5);
    EXPECT_DOUBLE_EQ(c[2].logit, 0.0 * 1.5 - 2 * 0.25 - 0.5);
    EXPECT_DOUBLE_EQ(c[3].logit, 4.0);  // unseen
    EXPECT_DOUBLE_EQ(c[4].logit, 1.0);
    // Neutral params or last_n == 0: no-op.
    auto d = cands_of({2.0, -1.0});
    SamplingParams off = p;
    off.penalty_last_n = 0;
    chain::apply_penalties(d, off, counts);
    EXPECT_DOUBLE_EQ(d[0].logit, 2.0);
    EXPECT_DOUBLE_EQ(d[1].logit, -1.0);
}

TEST(SamplerPenalties, AppliedThroughSamplerOverHistoryWindow) {
    // Token 0 leads by 1.0; a repetition penalty of 3 on a positive logit (2.0 -> 0.667)
    // flips greedy to token 1 only while token 0 is inside the window.
    SamplingParams p;
    p.temperature = 0.0f;
    p.repetition_penalty = 3.0f;
    p.penalty_last_n = 2;
    auto s = make(p, 3);
    const std::vector<float> row{2.0f, 1.0f, -5.0f};
    EXPECT_EQ(s.sample(row, std::vector<std::int32_t>{}), 0);
    EXPECT_EQ(s.sample(row, std::vector<std::int32_t>{2, 0}), 1);
    EXPECT_EQ(s.sample(row, std::vector<std::int32_t>{0, 2, 2}), 0);  // 0 fell out of last 2
    const std::vector<std::int32_t> ids{2, 0, 1};
    const std::vector<float> lg{-5.0f, 2.0f, 1.0f};
    EXPECT_EQ(s.sample(CandidateSet{ids, lg}, std::vector<std::int32_t>{0}), 1);
}

TEST(SamplerFilters, TopKMatchesReference) {
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nd(0.0, 2.0);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> lg(50);
        for (auto& x : lg) x = std::round(nd(rng) * 4) / 4;  // many ties
        std::vector<Ref> r;
        for (std::size_t i = 0; i < lg.size(); ++i) r.push_back({static_cast<std::int32_t>(i), lg[i]});
        for (const int k : {0, 1, 3, 17, 49, 50, 51, 1000}) {
            auto c = cands_of(lg);
            std::shuffle(c.begin(), c.end(), rng);  // order independence
            chain::apply_top_k(c, k);
            std::vector<chain::Cand> expect;
            for (const auto& x : ref_top_k(r, k)) expect.push_back({x.id, x.logit});
            ASSERT_EQ(ids_of(c), ids_of(expect)) << "trial " << trial << " k " << k;
        }
    }
}

TEST(SamplerFilters, TopPMatchesReferenceIncludingEdges) {
    std::mt19937_64 rng(12);
    std::normal_distribution<double> nd(0.0, 1.5);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> lg(40);
        for (auto& x : lg) x = std::round(nd(rng) * 8) / 8;
        std::vector<Ref> r;
        for (std::size_t i = 0; i < lg.size(); ++i) r.push_back({static_cast<std::int32_t>(i), lg[i]});
        for (const float p : {0.0f, 0.05f, 0.5f, 0.9f, 0.999f, 1.0f}) {
            auto c = cands_of(lg);
            std::shuffle(c.begin(), c.end(), rng);
            chain::apply_top_p(c, p);
            std::vector<chain::Cand> expect;
            for (const auto& x : ref_top_p(r, static_cast<double>(p))) expect.push_back({x.id, x.logit});
            ASSERT_EQ(ids_of(c), ids_of(expect)) << "trial " << trial << " p " << p;
        }
    }
    // top_p = 1: exact no-op; probs .5/.25/.25 with p = .74 keeps 2; the .25 tie keeps the lower id.
    auto c = cands_of({std::log(2.0), 0.0, 0.0});
    chain::apply_top_p(c, 1.0f);
    EXPECT_EQ(c.size(), 3U);
    chain::apply_top_p(c, 0.74f);
    EXPECT_EQ(ids_of(c), (std::vector<std::int32_t>{0, 1}));  // tie 1/2 -> lower id kept
    // Cumulative probability exactly equal to top_p stops there (>=, llama.cpp): four equal
    // logits give exact 0.25 steps in double, so p = 0.5 keeps exactly two (the lower ids).
    auto e = cands_of({0.0, 0.0, 0.0, 0.0});
    chain::apply_top_p(e, 0.5f);
    EXPECT_EQ(ids_of(e), (std::vector<std::int32_t>{0, 1}));
}

TEST(SamplerDraw, InverseCdfInIdOrderIndependentOfInputOrder) {
    // Weights by id: 1 -> 1, 2 -> 2, 3 -> 1 (total 4). In id order the CDF steps are
    // [0, .25) -> 1, [.25, .75) -> 2, [.75, 1) -> 3, whatever order the set arrives in.
    const std::vector<chain::Cand> base{{3, 0.0}, {1, 0.0}, {2, std::log(2.0)}};
    const std::vector<std::pair<double, std::int32_t>> expect{
        {0.0, 1}, {0.2, 1}, {0.3, 2}, {0.7, 2}, {0.8, 3}, {0.999, 3}};
    std::vector<chain::Cand> perm = base;
    std::sort(perm.begin(), perm.end(), [](const chain::Cand& a, const chain::Cand& b) { return a.id < b.id; });
    do {
        for (const auto& [u, id] : expect) {
            auto c = perm;
            EXPECT_EQ(chain::draw(c, u), id) << "u " << u << " first id " << perm.front().id;
        }
    } while (std::next_permutation(perm.begin(), perm.end(),
                                   [](const chain::Cand& a, const chain::Cand& b) { return a.id < b.id; }));
}

TEST(SamplerFilters, MinPMatchesReferenceIncludingEdges) {
    std::mt19937_64 rng(13);
    std::normal_distribution<double> nd(0.0, 1.5);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> lg(40);
        for (auto& x : lg) x = nd(rng);
        std::vector<Ref> r;
        for (std::size_t i = 0; i < lg.size(); ++i) r.push_back({static_cast<std::int32_t>(i), lg[i]});
        for (const float p : {0.0f, 0.01f, 0.2f, 0.7f, 1.0f}) {
            auto c = cands_of(lg);
            chain::apply_min_p(c, p);
            std::vector<chain::Cand> expect;
            for (const auto& x : ref_min_p(r, static_cast<double>(p))) expect.push_back({x.id, x.logit});
            ASSERT_EQ(ids_of(c), ids_of(expect)) << "trial " << trial << " p " << p;
        }
    }
    auto c = cands_of({1.0, 1.0, 0.0});
    chain::apply_min_p(c, 1.0f);  // keeps every max tie
    EXPECT_EQ(ids_of(c), (std::vector<std::int32_t>{0, 1}));
}

TEST(SamplerFilters, TypicalMatchesReference) {
    std::mt19937_64 rng(14);
    std::normal_distribution<double> nd(0.0, 1.5);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> lg(30);
        for (auto& x : lg) x = nd(rng);
        std::vector<Ref> r;
        for (std::size_t i = 0; i < lg.size(); ++i) r.push_back({static_cast<std::int32_t>(i), lg[i]});
        for (const float p : {0.0f, 0.2f, 0.5f, 0.95f, 1.0f}) {
            auto c = cands_of(lg);
            std::shuffle(c.begin(), c.end(), rng);
            chain::apply_typical(c, p);
            std::vector<chain::Cand> expect;
            for (const auto& x : ref_typical(r, static_cast<double>(p))) expect.push_back({x.id, x.logit});
            ASSERT_EQ(ids_of(c), ids_of(expect)) << "trial " << trial << " p " << p;
        }
    }
}

TEST(SamplerFilters, ProbabilitiesAreStableForHugeLogits) {
    const auto p = chain::probabilities(cands_of({1e30, 1e30 - 1e15, -1e30}));
    EXPECT_TRUE(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]));
    EXPECT_NEAR(p[0], 1.0, 1e-12);
    const auto q = chain::probabilities(cands_of({1000.0, 1000.0}));
    EXPECT_DOUBLE_EQ(q[0], 0.5);
}

struct DistCase {
    const char* name;
    float temperature;
    int top_k;
    float typical_p;
    float top_p;
    float min_p;
};

TEST(SamplerDistribution, EmpiricalMatchesAnalyticChiSquare) {
    const std::vector<DistCase> cases{
        {"plain", 1.0f, 0, 1.0f, 1.0f, 0.0f},        {"hot", 2.5f, 0, 1.0f, 1.0f, 0.0f},
        {"cold", 0.4f, 0, 1.0f, 1.0f, 0.0f},         {"top_k", 1.0f, 5, 1.0f, 1.0f, 0.0f},
        {"top_p", 1.0f, 0, 1.0f, 0.8f, 0.0f},        {"min_p", 1.0f, 0, 1.0f, 1.0f, 0.1f},
        {"typical", 1.0f, 0, 0.6f, 1.0f, 0.0f},      {"all", 0.8f, 10, 0.9f, 0.9f, 0.05f},
        {"k_gt_n", 1.3f, 1000, 1.0f, 0.95f, 0.0f},
    };
    const auto logits = fixed_logits();
    constexpr std::size_t kDraws = 100000;
    for (const auto& dc : cases) {
        SamplingParams p;
        p.temperature = dc.temperature;
        p.top_k = dc.top_k;
        p.typical_p = dc.typical_p;
        p.top_p = dc.top_p;
        p.min_p = dc.min_p;
        p.seed = 0xC0FFEE;
        auto s = make(p, logits.size());
        std::map<std::int32_t, std::size_t> got;
        for (std::size_t i = 0; i < kDraws; ++i) ++got[s.sample(logits, {})];
        const auto expect = ref_distribution(logits, p);
        const auto r = chi_square(expect, got, kDraws);
        EXPECT_EQ(r.outside, 0U) << dc.name << ": drew a token the filters exclude";
        EXPECT_LT(r.stat, chi2_critical(r.df)) << dc.name << " chi2=" << r.stat << " df=" << r.df;
    }
}

TEST(SamplerDeterminism, SeedResetAndOrderIndependence) {
    SamplingParams p;
    p.seed = 42;
    p.top_p = 0.95f;
    std::mt19937 g(5);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<std::vector<float>> rows(300, std::vector<float>(64));
    for (auto& r : rows) {
        for (auto& x : r) x = nd(g);
    }
    auto a = make(p, 64);
    auto b = make(p, 64);
    std::vector<std::int32_t> ta, tb, tc;
    for (const auto& r : rows) ta.push_back(a.sample(r, {}));
    for (const auto& r : rows) tb.push_back(b.sample(r, {}));
    EXPECT_EQ(ta, tb);
    a.reset();
    for (const auto& r : rows) tc.push_back(a.sample(r, {}));
    EXPECT_EQ(ta, tc);
    p.seed = 43;
    auto c = make(p, 64);
    std::vector<std::int32_t> td;
    for (const auto& r : rows) td.push_back(c.sample(r, {}));
    EXPECT_NE(ta, td);
    EXPECT_EQ(a.seed(), 42U);
    // Candidate order does not change the draw.
    p.seed = 9;
    auto e = make(p, 64);
    auto f = make(p, 64);
    std::vector<std::int32_t> ids(64);
    std::iota(ids.begin(), ids.end(), 0);
    for (const auto& r : rows) {
        std::vector<std::int32_t> perm = ids;
        std::shuffle(perm.begin(), perm.end(), g);
        std::vector<float> lg;
        for (const auto id : perm) lg.push_back(r[static_cast<std::size_t>(id)]);
        ASSERT_EQ(e.sample(r, {}), f.sample(CandidateSet{perm, lg}, {}));
    }
}

TEST(SamplerPrefilter, ExactPrefilterOkTruthTable) {
    SamplingParams p;
    p.temperature = 0.0f;
    EXPECT_TRUE(exact_prefilter_ok(p, 1));
    EXPECT_FALSE(exact_prefilter_ok(p, 0));
    p.temperature = 1.0f;
    EXPECT_FALSE(exact_prefilter_ok(p, 1024));  // top_k disabled: full row needed
    p.top_k = 40;
    EXPECT_TRUE(exact_prefilter_ok(p, 1024));
    EXPECT_TRUE(exact_prefilter_ok(p, 40));
    EXPECT_FALSE(exact_prefilter_ok(p, 39));
    SamplingParams q = p;
    q.repetition_penalty = 1.1f;
    EXPECT_FALSE(exact_prefilter_ok(q, 1024));
    q = p;
    q.presence_penalty = -0.5f;
    EXPECT_FALSE(exact_prefilter_ok(q, 1024));
    q.penalty_last_n = 0;  // penalties disabled by the window
    EXPECT_TRUE(exact_prefilter_ok(q, 1024));
    q = p;
    q.json_object = true;
    EXPECT_FALSE(exact_prefilter_ok(q, 1024));
    q = p;
    q.temperature = 0.0f;
    q.frequency_penalty = 0.1f;
    EXPECT_FALSE(exact_prefilter_ok(q, 1024));
}

TEST(SamplerPrefilter, PrefilteredEqualsFullRowWheneverExact) {
    constexpr std::size_t kVocab = 5000;
    std::mt19937 g(77);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const std::vector<SamplingParams> configs = [] {
        std::vector<SamplingParams> v;
        SamplingParams p;
        p.temperature = 0.0f;
        v.push_back(p);
        for (const int k : {1, 8, 40, 256}) {
            for (const float t : {0.3f, 1.0f, 1.7f}) {
                SamplingParams q;
                q.temperature = t;
                q.top_k = k;
                q.top_p = 0.9f;
                q.min_p = 0.02f;
                q.typical_p = k > 8 ? 0.95f : 1.0f;
                v.push_back(q);
            }
        }
        return v;
    }();
    std::size_t compared = 0;
    for (std::size_t ci = 0; ci < configs.size(); ++ci) {
        SamplingParams p = configs[ci];
        p.seed = 1000 + ci;
        const std::size_t k = 256;
        ASSERT_TRUE(exact_prefilter_ok(p, k)) << ci;
        auto full = make(p, kVocab);
        auto pre = make(p, kVocab);
        for (int step = 0; step < 200; ++step) {
            std::vector<float> row(kVocab);
            for (auto& x : row) x = std::round(nd(g) * 2.0f) / 2.0f;  // ties at the k boundary
            const auto top = halo::cpu::top_k(row, k);
            std::vector<std::int32_t> ids;
            std::vector<float> lg;
            for (const auto& e : top) {
                ids.push_back(e.index);
                lg.push_back(e.value);
            }
            ASSERT_EQ(full.sample(row, {}), pre.sample(CandidateSet{ids, lg}, {})) << "config " << ci << " step " << step;
            ++compared;
        }
    }
    EXPECT_EQ(compared, configs.size() * 200);
}

TEST(SamplerPrefilter, PenaltyCanMakePrefilterInexact) {
    // Why exact_prefilter_ok() refuses penalties: demoting the top candidate lets a token
    // the k=1 prefilter never saw become the true argmax.
    SamplingParams p;
    p.temperature = 0.0f;
    p.repetition_penalty = 4.0f;
    ASSERT_FALSE(exact_prefilter_ok(p, 1));
    auto s = make(p, 3);
    const std::vector<float> row{4.0f, 3.0f, 0.0f};
    const std::vector<std::int32_t> hist{0};
    const std::vector<std::int32_t> ids{0};
    const std::vector<float> lg{4.0f};
    EXPECT_EQ(s.sample(row, hist), 1);
    EXPECT_EQ(s.sample(CandidateSet{ids, lg}, hist), 0);
}

}  // namespace
