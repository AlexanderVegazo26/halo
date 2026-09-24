#pragma once
// Independent fp64 reference implementations for the CPU kernel tests.
// Written directly from the math in DECISIONS.md D-004 (not ported from the fp32
// kernels): plain nested loops, dense contiguous layouts, no chunking.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <vector>

// fp64 reference code: float -> double promotion is intentional here (see test files).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"

namespace halo::cpu::test {

struct GdnRefDims {
    std::size_t n_k, n_v, d_k, d_v;
    bool tiled;  // v head j -> k head j % n_k (tiled) or j / (n_v / n_k) (grouped)
};

[[nodiscard]] inline std::size_t ref_k_head(const GdnRefDims& d, std::size_t j) {
    return d.tiled ? j % d.n_k : j / (d.n_v / d.n_k);
}

/// fp64 gated delta rule, recurrent form, D-004 item 6, with the D-016 q/k contract: when
/// `l2`, q and k are L2-normalized per head (eps 1e-6); then q *= q_scale (always explicit
/// here -- no implicit 1/sqrt(d_k)). Inputs dense token-major:
/// q,k [T, n_k*d_k], v [T, n_v*d_v], g/beta [T, n_v]; state [n_v, d_k, d_v] in/out.
/// Returns out [T, n_v*d_v]. If out_terms is given it receives, per output element,
/// sum_i |S[i][c] q[i]| -- the magnitude of the terms of the final dot product (the scale of
/// Model R / Model G in tolerance.h; unlike |out| it is not shrunk by cancellation).
inline std::vector<double> gdn_ref(const GdnRefDims& d, std::size_t T, std::span<const float> q,
                                   std::span<const float> k, std::span<const float> v,
                                   std::span<const float> g, std::span<const float> beta,
                                   std::vector<double>& state, bool l2, double q_scale,
                                   std::vector<double>* out_terms = nullptr) {
    std::vector<double> out(T * d.n_v * d.d_v, 0.0);
    if (out_terms) out_terms->assign(out.size(), 0.0);
    std::vector<double> qn(d.d_k), kn(d.d_k), kv(d.d_v), delta(d.d_v);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t j = 0; j < d.n_v; ++j) {
            const std::size_t kh = ref_k_head(d, j);
            double sq = 0, sk = 0;
            for (std::size_t i = 0; i < d.d_k; ++i) {
                qn[i] = double(q[t * d.n_k * d.d_k + kh * d.d_k + i]);
                kn[i] = double(k[t * d.n_k * d.d_k + kh * d.d_k + i]);
                sq += qn[i] * qn[i];
                sk += kn[i] * kn[i];
            }
            const double iq = l2 ? 1.0 / std::sqrt(sq + 1e-6) : 1.0;
            const double ik = l2 ? 1.0 / std::sqrt(sk + 1e-6) : 1.0;
            for (std::size_t i = 0; i < d.d_k; ++i) {
                qn[i] *= iq * q_scale;
                kn[i] *= ik;
            }
            double* S = state.data() + j * d.d_k * d.d_v;
            const double decay = std::exp(static_cast<double>(g[t * d.n_v + j]));
            const double b = double(beta[t * d.n_v + j]);
            for (std::size_t e = 0; e < d.d_k * d.d_v; ++e) S[e] *= decay;
            for (std::size_t c = 0; c < d.d_v; ++c) {
                kv[c] = 0;
                for (std::size_t i = 0; i < d.d_k; ++i) kv[c] += S[i * d.d_v + c] * kn[i];
                delta[c] = (double(v[t * d.n_v * d.d_v + j * d.d_v + c]) - kv[c]) * b;
            }
            for (std::size_t i = 0; i < d.d_k; ++i)
                for (std::size_t c = 0; c < d.d_v; ++c) S[i * d.d_v + c] += kn[i] * delta[c];
            for (std::size_t c = 0; c < d.d_v; ++c) {
                double o = 0, terms = 0;
                for (std::size_t i = 0; i < d.d_k; ++i) {
                    o += S[i * d.d_v + c] * qn[i];
                    terms += std::abs(S[i * d.d_v + c] * qn[i]);
                }
                out[t * d.n_v * d.d_v + j * d.d_v + c] = o;
                if (out_terms) (*out_terms)[t * d.n_v * d.d_v + j * d.d_v + c] = terms;
            }
        }
    }
    return out;
}

/// Deterministic random fill helpers.
struct Rng {
    std::mt19937_64 gen;
    explicit Rng(std::uint64_t seed) : gen(seed) {}
    std::vector<float> normal(std::size_t n, float sd = 1.0f) {
        std::normal_distribution<float> nd(0.0f, sd);
        std::vector<float> v(n);
        for (auto& e : v) e = nd(gen);
        return v;
    }
    std::vector<float> uniform(std::size_t n, float lo, float hi) {
        std::uniform_real_distribution<float> ud(lo, hi);
        std::vector<float> v(n);
        for (auto& e : v) e = ud(gen);
        return v;
    }
};

[[nodiscard]] inline bool bitwise_equal(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace halo::cpu::test

#pragma GCC diagnostic pop
