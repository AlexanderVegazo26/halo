// CPU reference operators vs golden outputs computed by the transformers Qwen3.5 functions
// themselves (python/tools/make_kernel_golden.py -> $HALO_REF_DIR/kernel_golden).
//
// Tolerances: the models in tolerance.h, doubled, because both sides are fp32 with
// different summation orders (|ours - hf| <= |ours - exact| + |hf - exact|).
// If the golden directory is absent every test is SKIPPED (reported, never a pass); if it
// is present but a required case/tensor is missing, the test FAILS.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/backends/cpu/ops.h"
#include "reference.h"
#include "tolerance.h"

// This file compares against fp64 reference math and promotes float operands to double on
// purpose. -Wdouble-promotion exists to catch *accidental* promotion in float code; it stays
// enabled for the CPU kernels themselves (backends/cpu) and is disabled for this file only.
#pragma GCC diagnostic ignored "-Wdouble-promotion"

namespace {

using namespace halo::cpu;
using namespace halo::cpu::test;
namespace fs = std::filesystem;

const fs::path kDir = fs::path(HALO_REF_DIR) / "kernel_golden";

const std::vector<std::string> kRequiredCases{
    "rms_norm.a", "rms_norm.b", "gated_norm", "l2norm", "rope", "conv", "gdn.r3", "gdn.r3_rect",
    "gdn.r1", "gdn.real", "attn.small", "attn.real", "elem", "matmul", "topk"};

class CpuGolden : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(kDir / "manifest.json")) {
            GTEST_SKIP() << "golden data not found at " << kDir
                         << " (generate with python/tools/make_kernel_golden.py)";
        }
        std::ifstream f(kDir / "manifest.json");
        manifest_ = nlohmann::json::parse(f);
    }

    const nlohmann::json& meta(const std::string& c) const {
        if (!manifest_["cases"].contains(c)) throw std::runtime_error("golden case missing: " + c);
        return manifest_["cases"][c];
    }

    template <class T>
    std::vector<T> load(const std::string& name) const {
        const auto& ts = manifest_["tensors"];
        if (!ts.contains(name)) throw std::runtime_error("golden tensor missing: " + name);
        const auto& e = ts[name];
        const std::string want = std::is_same_v<T, float> ? "float32" : "int32";
        if (e["dtype"].get<std::string>() != want) throw std::runtime_error(name + ": dtype " + e["dtype"].dump());
        std::size_t n = 1;
        for (const auto& d : e["shape"]) n *= d.get<std::size_t>();
        std::vector<T> v(n);
        std::ifstream f(kDir / e["file"].get<std::string>(), std::ios::binary);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
        if (!f || f.gcount() != static_cast<std::streamsize>(n * sizeof(T))) throw std::runtime_error(name + ": short read");
        return v;
    }
    std::vector<float> f32(const std::string& n) const { return load<float>(n); }

    nlohmann::json manifest_;
};

ConstRows cv(const std::vector<float>& v, std::size_t r, std::size_t c) { return {std::span<const float>(v), r, c}; }
Rows mv(std::vector<float>& v, std::size_t r, std::size_t c) { return {std::span<float>(v), r, c}; }

TEST_F(CpuGolden, ManifestHasAllRequiredCasesAndTorchPath) {
    for (const auto& c : kRequiredCases) EXPECT_TRUE(manifest_["cases"].contains(c)) << c;
    EXPECT_EQ(manifest_["reference"]["functions_called_via"], "inspect.unwrap (torch path)");
    std::printf("[golden] transformers %s, torch %s, %zu tensors\n",
                manifest_["transformers"].get<std::string>().c_str(), manifest_["torch"].get<std::string>().c_str(),
                manifest_["tensors"].size());
}

TEST_F(CpuGolden, RmsNormAndGatedNormAndL2) {
    double worst = 0;
    for (const std::string c : {"rms_norm.a", "rms_norm.b"}) {
        const std::size_t r = meta(c)["rows"], d = meta(c)["dim"];
        const auto x = f32(c + ".x"), w = f32(c + ".w"), y = f32(c + ".y");
        std::vector<float> o(x.size());
        rms_norm(cv(x, r, d), w, 1e-6f, mv(o, r, d));
        for (std::size_t i = 0; i < o.size(); ++i) {
            const double tol = 2 * (tol_reduction(d, std::abs(y[i])) + tol_elementwise(y[i]));
            worst = std::max(worst, std::abs(o[i] - y[i]) / tol);
            ASSERT_LE(std::abs(o[i] - y[i]), tol) << c << " i=" << i;
        }
    }
    {
        const std::size_t r = meta("gated_norm")["rows"], d = meta("gated_norm")["dim"];
        const auto x = f32("gated_norm.x"), z = f32("gated_norm.z"), w = f32("gated_norm.w"), y = f32("gated_norm.y");
        std::vector<float> o(x.size());
        gated_rms_norm(cv(x, r, d), w, cv(z, r, d), 1e-6f, mv(o, r, d));
        for (std::size_t i = 0; i < o.size(); ++i) {
            const double tol = 2 * (tol_reduction(d, std::abs(y[i])) + 2 * tol_elementwise(y[i]));
            worst = std::max(worst, std::abs(o[i] - y[i]) / tol);
            ASSERT_LE(std::abs(o[i] - y[i]), tol) << "gated_norm i=" << i;
        }
    }
    {
        const auto& m = meta("l2norm");
        const std::size_t t = m["tokens"], h = m["heads"], d = m["head_dim"];
        const auto x = f32("l2norm.x"), y = f32("l2norm.y");
        std::vector<float> o(x.size());
        l2_norm_heads(cv(x, t, h * d), h, d, mv(o, t, h * d));
        for (std::size_t i = 0; i < o.size(); ++i) {
            const double tol = 2 * (tol_reduction(d, std::abs(y[i])) + tol_elementwise(y[i]));
            worst = std::max(worst, std::abs(o[i] - y[i]) / tol);
            ASSERT_LE(std::abs(o[i] - y[i]), tol) << "l2norm i=" << i;
        }
    }
    std::printf("[tol] golden norms worst err/tol = %.3g\n", worst);
}

TEST_F(CpuGolden, RopeInvFreqBitwiseAndOutputs) {
    const auto& m = meta("rope");
    const std::size_t t = m["tokens"], h = m["heads"], hd = m["head_dim"], rot = m["rot_dims"];
    const float theta = m["theta"].get<float>();
    // Bitwise inverse frequencies => bitwise fp32 angles at every position (one multiply),
    // so the tolerance below need not grow with the position (tested up to 262143).
    const auto inv_hf = f32("rope.inv_freq");
    const auto inv = rope_inv_freq(rot, theta);
    ASSERT_EQ(inv.size(), inv_hf.size());
    for (std::size_t i = 0; i < inv.size(); ++i) ASSERT_EQ(inv[i], inv_hf[i]) << "inv_freq[" << i << "]";
    const auto x = f32("rope.x"), y = f32("rope.y");
    const auto pos = load<std::int32_t>("rope.positions");
    auto o = x;
    partial_rope_neox(mv(o, t, h * hd), h, hd, pos, rot, theta);
    double worst = 0;
    for (std::size_t r = 0; r < t; ++r)
        for (std::size_t hh = 0; hh < h; ++hh)
            for (std::size_t i = 0; i < hd; ++i) {
                const std::size_t e = r * h * hd + hh * hd + i;
                const std::size_t pair = i < rot / 2 ? i + rot / 2 : (i < rot ? i - rot / 2 : i);
                const double scale = std::abs(x[e]) + (i < rot ? std::abs(x[r * h * hd + hh * hd + pair]) : 0.0);
                const double tol = 2 * tol_elementwise(scale);
                worst = std::max(worst, std::abs(o[e] - y[e]) / tol);
                ASSERT_LE(std::abs(o[e] - y[e]), tol) << "pos " << pos[r] << " head " << hh << " dim " << i;
            }
    std::printf("[tol] golden rope worst err/tol = %.3g\n", worst);
}

TEST_F(CpuGolden, Conv1dPrefillAndUpdate) {
    const auto& m = meta("conv");
    const std::size_t c = m["channels"], k = m["kernel"];
    const auto w = f32("conv.w");
    double worst = 0;
    for (const std::string mode : {"prefill", "update"}) {
        const std::size_t t = m[mode + "_tokens"];
        const auto x = f32("conv." + mode + ".x"), y = f32("conv." + mode + ".y"),
                   st_out = f32("conv." + mode + ".state_out");
        std::vector<float> st = mode == "update" ? f32("conv.update.state_in") : std::vector<float>((k - 1) * c, 0.0f);
        const auto st0 = st;
        std::vector<float> o(x.size());
        causal_conv1d_silu(cv(x, t, c), cv(w, c, k), mv(st, k - 1, c), mv(o, t, c));
        EXPECT_TRUE(bitwise_equal(st, st_out)) << mode << " conv state";
        for (std::size_t tt = 0; tt < t; ++tt)
            for (std::size_t ch = 0; ch < c; ++ch) {
                double sabs = 0;
                for (std::size_t j = 0; j < k; ++j) {
                    const long idx = long(tt) - long(k - 1) + long(j);
                    const double xin = idx >= 0 ? x[std::size_t(idx) * c + ch] : st0[std::size_t(long(k - 1) + idx) * c + ch];
                    sabs += std::abs(w[ch * k + j] * xin);
                }
                const std::size_t e = tt * c + ch;
                const double tol = 2 * (1.1 * tol_reduction(k, sabs) + tol_elementwise(y[e]));
                worst = std::max(worst, std::abs(o[e] - y[e]) / tol);
                ASSERT_LE(std::abs(o[e] - y[e]), tol) << mode << " t=" << tt << " c=" << ch;
            }
    }
    std::printf("[tol] golden conv1d worst err/tol = %.3g\n", worst);
}

TEST_F(CpuGolden, GatedDeltaRuleGroupedAndTiledBothForms) {
    for (const std::string c : {"gdn.r3", "gdn.r3_rect", "gdn.r1", "gdn.real"}) {
        const auto& m = meta(c);
        const std::size_t T = m["tokens"], nk = m["n_k_heads"], nv = m["n_v_heads"], dk = m["d_k"], dv = m["d_v"];
        const auto q = f32(c + ".q"), k = f32(c + ".k");
        for (const std::string layout : {"grouped", "tiled"}) {
            const std::string p = c + "." + layout + ".";
            const auto v = f32(p + "v"), g = f32(p + "g"), beta = f32(p + "beta"), s0 = f32(p + "state_in");
            const GdnDims dims{nk, nv, dk, dv, layout == "tiled" ? GdnHeadMapping::Tiled : GdnHeadMapping::Grouped};
            const GdnInputs in{cv(q, T, nk * dk), cv(k, T, nk * dk), cv(v, T, nv * dv), cv(g, T, nv), cv(beta, T, nv)};
            // Model G output scale (sum of |terms| of each output dot product) from fp64.
            std::vector<double> ref_state(s0.begin(), s0.end()), out_terms;
            (void)gdn_ref({nk, nv, dk, dv, layout == "tiled"}, T, q, k, v, g, beta, ref_state, true,
                          1.0 / std::sqrt(static_cast<double>(dk)), &out_terms);
            for (const bool chunked : {false, true}) {
                std::vector<float> st = s0, out(T * nv * dv);
                if (chunked)
                    gated_delta_rule_chunked(dims, in, st, mv(out, T, nv * dv), true, m["chunk_size"].get<std::size_t>());
                else
                    gated_delta_rule_recurrent(dims, in, st, mv(out, T, nv * dv), true);
                double worst = 0;
                for (const std::string hf : {"chunk", "recur"}) {
                    const auto ho = f32(p + "out_" + hf), hs = f32(p + "state_" + hf);
                    for (std::size_t j = 0; j < nv; ++j) {
                        double so = 0, eo = 0, ss = 0, es = 0;
                        for (std::size_t t = 0; t < T; ++t)
                            for (std::size_t cc = 0; cc < dv; ++cc) {
                                const std::size_t i = t * nv * dv + j * dv + cc;
                                so = std::max(so, out_terms[i]);
                                eo = std::max(eo, double(std::abs(out[i] - ho[i])));
                            }
                        for (std::size_t e = 0; e < dk * dv; ++e) {
                            ss = std::max(ss, double(std::abs(hs[j * dk * dv + e])));
                            es = std::max(es, double(std::abs(st[j * dk * dv + e] - hs[j * dk * dv + e])));
                        }
                        const double to = 2 * tol_gdn(T, dk, so) + kDenormFloor, ts = 2 * tol_gdn(T, dk, ss) + kDenormFloor;
                        EXPECT_LE(eo, to) << p << (chunked ? "chunked" : "recurrent") << " vs hf " << hf << " head " << j;
                        EXPECT_LE(es, ts) << p << (chunked ? "chunked" : "recurrent") << " vs hf " << hf << " state head " << j;
                        worst = std::max({worst, eo / to, es / ts});
                    }
                }
                std::printf("[tol] golden %s %s %-9s worst err/tol = %.3g\n", c.c_str(), layout.c_str(),
                            chunked ? "chunked" : "recurrent", worst);
            }
        }
    }
}

TEST_F(CpuGolden, AttentionAndOutputGate) {
    for (const std::string c : {"attn.small", "attn.real"}) {
        const auto& m = meta(c);
        const std::size_t nh = m["n_head"], nkv = m["n_kv_head"], hd = m["head_dim"], t = m["tokens"], s = m["history"],
                          off = m["q_offset"];
        const float scale = m["scale"].get<float>();
        const auto q = f32(c + ".q"), k = f32(c + ".k"), v = f32(c + ".v"), y = f32(c + ".out"), gate = f32(c + ".gate"),
                   gated = f32(c + ".gated");
        std::vector<float> o(t * nh * hd);
        attention_gqa({nh, nkv, hd}, cv(q, t, nh * hd), PagedRows::contiguous(k.data(), s, nkv * hd, nkv * hd),
                      PagedRows::contiguous(v.data(), s, nkv * hd, nkv * hd), off, scale, mv(o, t, nh * hd));
        double worst = 0;
        for (std::size_t tt = 0; tt < t; ++tt)
            for (std::size_t h = 0; h < nh; ++h) {
                const std::size_t kvh = h / (nh / nkv), n = off + tt + 1;
                double qn = 0, a = 0, vmax = 0;
                for (std::size_t i = 0; i < hd; ++i) qn += double(q[tt * nh * hd + h * hd + i]) * q[tt * nh * hd + h * hd + i];
                for (std::size_t ss = 0; ss < n; ++ss) {
                    double kn = 0;
                    for (std::size_t i = 0; i < hd; ++i) {
                        kn += double(k[ss * nkv * hd + kvh * hd + i]) * k[ss * nkv * hd + kvh * hd + i];
                        vmax = std::max(vmax, double(std::abs(v[ss * nkv * hd + kvh * hd + i])));
                    }
                    a = std::max(a, std::sqrt(qn * kn) * scale);
                }
                const double tol = 2 * (8.0 * kEps * (std::sqrt(double(hd)) * a + std::sqrt(double(n)) + 4.0) * vmax);
                for (std::size_t i = 0; i < hd; ++i) {
                    const std::size_t e = tt * nh * hd + h * hd + i;
                    worst = std::max(worst, std::abs(o[e] - y[e]) / tol);
                    ASSERT_LE(std::abs(o[e] - y[e]), tol) << c << " t=" << tt << " h=" << h << " i=" << i;
                }
            }
        std::vector<float> og(o.size());
        mul_sigmoid(cv(o, t, nh * hd), cv(gate, t, nh * hd), mv(og, t, nh * hd));
        for (std::size_t e = 0; e < og.size(); ++e) {
            // Our attention error |o - y| passes through a factor sigmoid <= 1, plus Model E
            // on both sides for the gate multiply.
            const double tol = 2 * tol_elementwise(gated[e]) + std::abs(o[e] - y[e]);
            ASSERT_LE(std::abs(og[e] - gated[e]), tol) << c << " gated e=" << e;
        }
        std::printf("[tol] golden %s worst err/tol = %.3g\n", c.c_str(), worst);
    }
}

TEST_F(CpuGolden, ElementwiseSoftmaxMatmulTopK) {
    const std::size_t r = meta("elem")["rows"], d = meta("elem")["dim"];
    const auto x = f32("elem.x"), b = f32("elem.b");
    std::vector<float> o(x.size());
    double worst = 0;
    auto cmp = [&](const std::string& name, double factor) {
        const auto y = f32("elem." + name);
        for (std::size_t i = 0; i < o.size(); ++i) {
            const double tol = factor * tol_elementwise(y[i]);
            if (factor == 0.0) {
                ASSERT_EQ(o[i], y[i]) << name << " i=" << i;
                continue;
            }
            worst = std::max(worst, std::abs(o[i] - y[i]) / tol);
            ASSERT_LE(std::abs(o[i] - y[i]), tol) << name << " x=" << x[i];
        }
    };
    silu(cv(x, r, d), mv(o, r, d));
    cmp("silu", 2);
    sigmoid(cv(x, r, d), mv(o, r, d));
    cmp("sigmoid", 2);
    softplus(cv(x, r, d), mv(o, r, d));
    cmp("softplus", 2);
    swiglu(cv(x, r, d), cv(b, r, d), mv(o, r, d));
    cmp("swiglu", 2);
    add(cv(x, r, d), cv(b, r, d), mv(o, r, d));
    cmp("add", 0);  // one correctly rounded op on both sides: exact
    mul(cv(x, r, d), cv(b, r, d), mv(o, r, d));
    cmp("mul", 0);
    std::printf("[tol] golden elementwise worst err/tol = %.3g\n", worst);

    const auto sx = f32("elem.softmax_x"), sy = f32("elem.softmax");
    softmax_rows(cv(sx, r, d), mv(o, r, d));
    for (std::size_t row = 0; row < r; ++row) {
        float mx = sx[row * d];
        for (std::size_t i = 0; i < d; ++i) mx = std::max(mx, sx[row * d + i]);
        for (std::size_t i = 0; i < d; ++i) {
            const std::size_t e = row * d + i;
            const double ref = sy[e];
            const double tol = 2 * ((kEps * std::abs(double(sx[e]) - mx)) * ref + tol_reduction(d, ref) + tol_elementwise(ref));
            ASSERT_LE(std::abs(o[e] - ref), tol) << "softmax row " << row << " i=" << i;
        }
    }

    const std::size_t t = meta("matmul")["tokens"], kk = meta("matmul")["k"], n = meta("matmul")["n"];
    const auto mx = f32("matmul.x"), mw = f32("matmul.w"), my = f32("matmul.y");
    std::vector<float> y(t * n);
    matmul(cv(mx, t, kk), WeightMatrix::dense(cv(mw, n, kk)), mv(y, t, n));
    for (std::size_t i = 0; i < t; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double sabs = 0;
            for (std::size_t e = 0; e < kk; ++e) sabs += std::abs(double(mx[i * kk + e]) * mw[j * kk + e]);
            ASSERT_LE(std::abs(y[i * n + j] - my[i * n + j]), 2 * tol_reduction(kk, sabs)) << "matmul " << i << "," << j;
        }

    const auto logits = f32("topk.logits"), vals = f32("topk.values");
    const auto idx = load<std::int32_t>("topk.indices");
    const std::size_t k = meta("topk")["k"];
    const auto tk = top_k(logits, k);
    for (std::size_t i = 0; i < k; ++i) {
        EXPECT_EQ(tk[i].index, idx[i]) << i;
        EXPECT_EQ(tk[i].value, vals[i]) << i;
    }
    EXPECT_EQ(argmax(logits).index, meta("topk")["argmax"].get<std::int32_t>());
}

}  // namespace
