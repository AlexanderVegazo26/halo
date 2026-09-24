// Micro-benchmark: CPU reference Gated DeltaNet at Qwen3.8-27B layer dims
// (48 value heads, 16 key heads, d_k = d_v = 128; D-003).
//
// NOT a HALO performance claim (DECISIONS.md D-001): timings are from whatever host runs
// this binary, typically the dev laptop / WSL2. One GDN layer, one sequence.
//
//   halo_bench_cpu_gdn [threads] [prefill_tokens]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/cpu/ops.h"

namespace {

using namespace halo::cpu;
using Clock = std::chrono::steady_clock;

struct Data {
    std::vector<float> q, k, v, g, beta;
};

Data make_data(std::size_t n_tok, const GdnDims& d) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    Data x;
    x.q.resize(n_tok * d.n_k_heads * d.d_k);
    x.k.resize(x.q.size());
    x.v.resize(n_tok * d.n_v_heads * d.d_v);
    x.g.resize(n_tok * d.n_v_heads);
    x.beta.resize(x.g.size());
    for (auto& e : x.q) e = nd(rng);
    for (auto& e : x.k) e = nd(rng);
    for (auto& e : x.v) e = nd(rng);
    for (auto& e : x.g) e = -0.05f - 0.5f * ud(rng);
    for (auto& e : x.beta) e = ud(rng);
    return x;
}

GdnInputs view(const Data& x, std::size_t n_tok, const GdnDims& d) {
    return {ConstRows(std::span<const float>(x.q), n_tok, d.n_k_heads * d.d_k),
            ConstRows(std::span<const float>(x.k), n_tok, d.n_k_heads * d.d_k),
            ConstRows(std::span<const float>(x.v), n_tok, d.n_v_heads * d.d_v),
            ConstRows(std::span<const float>(x.g), n_tok, d.n_v_heads),
            ConstRows(std::span<const float>(x.beta), n_tok, d.n_v_heads)};
}

template <class F>
double best_ms(int reps, F&& f) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        f();
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        best = ms < best ? ms : best;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t threads = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : ThreadPool::default_threads();
    const std::size_t prefill = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 512;
    const GdnDims dims{16, 48, 128, 128, GdnHeadMapping::Tiled};
    ThreadPool pool(threads == 0 ? 1 : threads);
    std::vector<float> state(dims.n_v_heads * dims.d_k * dims.d_v, 0.0f);

    std::printf("HALO CPU reference GDN micro-benchmark -- DEV-HOST TIMINGS, NOT A HALO PERF CLAIM (D-001)\n");
    std::printf("dims: n_v=48 n_k=16 d_k=d_v=128 (one layer, one sequence), threads=%zu, fp32\n", pool.size());

    // Decode: recurrent, T = 1.
    {
        const Data x = make_data(1, dims);
        const GdnInputs in = view(x, 1, dims);
        std::vector<float> out(dims.n_v_heads * dims.d_v);
        Rows o(std::span<float>(out), 1, out.size());
        constexpr int kSteps = 200;
        const double ms = best_ms(5, [&] {
            for (int i = 0; i < kSteps; ++i) gated_delta_rule_recurrent(dims, in, state, o, true, &pool);
        });
        std::printf("recurrent decode  T=1    : %8.4f ms/token (best of 5 x %d steps)\n", ms / kSteps, kSteps);
    }
    // Prefill: recurrent vs chunked(64).
    {
        const Data x = make_data(prefill, dims);
        const GdnInputs in = view(x, prefill, dims);
        std::vector<float> out(prefill * dims.n_v_heads * dims.d_v);
        Rows o(std::span<float>(out), prefill, dims.n_v_heads * dims.d_v);
        const double rec = best_ms(3, [&] { gated_delta_rule_recurrent(dims, in, state, o, true, &pool); });
        std::printf("recurrent prefill T=%-5zu: %8.2f ms (%8.1f tok/s)\n", prefill, rec, 1e3 * static_cast<double>(prefill) / rec);
        for (const std::size_t cs : {16, 64}) {
            const double ch = best_ms(3, [&] { gated_delta_rule_chunked(dims, in, state, o, true, cs, &pool); });
            std::printf("chunked(%3zu) prefill T=%-5zu: %8.2f ms (%8.1f tok/s)\n", cs, prefill, ch,
                        1e3 * static_cast<double>(prefill) / ch);
        }
    }
    return 0;
}
