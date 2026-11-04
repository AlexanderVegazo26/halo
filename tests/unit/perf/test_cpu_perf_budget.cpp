// Performance budgets for the CPU reference kernels (ISO 25010 performance efficiency).
//
// Two kinds of check, both single-threaded (pool = nullptr) so the numbers do not depend on
// the runner's core count:
//   * Budget: best-of-N wall time of one kernel call at a fixed shape must stay under a
//     generous absolute limit (~10x the dev-host baseline below), so only a gross regression
//     (a lost vectorization, an accidental copy of the weights, a debug-only path left on)
//     fails, never runner noise.
//   * Scaling: the kernel's cost must grow with the shape as its complexity says (attention
//     decode linear in the history length, the GDN recurrence linear in tokens, matmul linear
//     in output rows). Measured as a ratio of two sizes on the same host, so it is largely
//     hardware-independent and catches an accidental O(n^2).
//
// Dev-host baseline (2026-11-04, RelWithDebInfo clang-21, no -march, one thread) is the comment
// next to each budget; the scaling ratios measured 3.99-4.16 for a 4x size step. Timings are NOT a HALO performance claim (DECISIONS.md D-001).
// Skipped (with a reason) in Debug and sanitizer builds, where timings mean nothing.
// ctest runs these RUN_SERIAL (label `perf`) so parallel tests don't steal the core.
// Rerun: ctest --test-dir build -L perf --output-on-failure

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "halo/backends/cpu/ops.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define HALO_PERF_SANITIZED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define HALO_PERF_SANITIZED 1
#endif

namespace {

using namespace halo::cpu;
using Clock = std::chrono::steady_clock;

// Why timings are not checked in this build, or nullptr when they are.
const char* skip_reason() {
#if defined(HALO_PERF_SANITIZED)
    return "sanitizer build: timings are not meaningful (perf budgets run in the RelWithDebInfo lane)";
#elif !defined(NDEBUG)
    return "Debug build (no NDEBUG): timings are not meaningful (perf budgets run in the RelWithDebInfo lane)";
#else
    return nullptr;
#endif
}

#define HALO_PERF_SKIP_IF_UNMEASURABLE()            \
    do {                                            \
        if (const char* why = skip_reason()) {      \
            GTEST_SKIP() << why;                    \
        }                                           \
    } while (0)

// Keeps results observable so the optimizer cannot drop a kernel call.
volatile float g_sink = 0.0f;

void consume(std::span<const float> v) {
    float s = 0.0f;
    for (std::size_t i = 0; i < v.size(); i += 61) s += v[i];
    g_sink = g_sink + s;
}

std::vector<float> random_vec(std::size_t n, std::uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(lo, hi);
    std::vector<float> v(n);
    for (auto& e : v) e = ud(rng);
    return v;
}

// Best wall time in ms over `reps` calls after one warm-up call. The best (not the mean) is
// the least noisy statistic for "how fast can this code go": noise only ever adds time.
double best_ms(int reps, const std::function<void()>& f) {
    f();
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        f();
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        best = std::min(best, ms);
    }
    return best;
}

void report(const std::string& what, double ms, double budget_ms) {
    std::printf("[perf] %-44s best %9.3f ms  budget %8.1f ms  (%5.1f%% of budget)\n", what.c_str(), ms, budget_ms,
                100.0 * ms / budget_ms);
    ::testing::Test::RecordProperty(what + " ms", std::to_string(ms));
}

void expect_within_budget(const std::string& what, double ms, double budget_ms) {
    report(what, ms, budget_ms);
    EXPECT_LE(ms, budget_ms) << what << ": best of N took " << ms << " ms, budget is " << budget_ms
                             << " ms (~10x the 2026-11-04 dev-host baseline): a gross slowdown of the CPU "
                                "reference kernel, e.g. lost vectorization or an extra copy per call";
}

// cost(large) / cost(small) must stay under `max_ratio` when the size grows by `size_factor`.
void expect_scaling(const std::string& what, double small_ms, double large_ms, double size_factor,
                    double max_ratio, const char* complexity) {
    const double ratio = large_ms / small_ms;
    std::printf("[perf] %-44s %9.3f ms -> %9.3f ms for %gx the size: ratio %.2f (limit %.1f, %s)\n", what.c_str(),
                small_ms, large_ms, size_factor, ratio, max_ratio, complexity);
    ::testing::Test::RecordProperty(what + " ratio", std::to_string(ratio));
    EXPECT_LE(ratio, max_ratio) << what << ": growing the size " << size_factor << "x made the call " << ratio
                                << "x slower (" << small_ms << " ms -> " << large_ms << " ms); " << complexity
                                << " allows at most " << max_ratio << "x";
}

// ---- workloads -----------------------------------------------------------------------------

struct MatmulCase {
    std::size_t T, N, K;
    std::vector<float> x, w, y;
    explicit MatmulCase(std::size_t t, std::size_t n, std::size_t k)
        : T(t), N(n), K(k), x(random_vec(t * k, 1)), w(random_vec(n * k, 2)), y(t * n) {}
    void run() {
        const auto wm = WeightMatrix::dense(ConstRows(std::span<const float>(w), N, K));
        matmul(ConstRows(std::span<const float>(x), T, K), wm, Rows(std::span<float>(y), T, N));
        consume(y);
    }
};

struct AttnCase {
    AttentionDims dims;
    std::size_t T, S;
    std::vector<float> q, k, v, out;
    AttnCase(AttentionDims d, std::size_t t, std::size_t s)
        : dims(d),
          T(t),
          S(s),
          q(random_vec(t * d.n_head * d.head_dim, 3)),
          k(random_vec(s * d.n_kv_head * d.head_dim, 4)),
          v(random_vec(s * d.n_kv_head * d.head_dim, 5)),
          out(t * d.n_head * d.head_dim) {}
    void run() {
        const std::size_t kv_cols = dims.n_kv_head * dims.head_dim;
        const auto kp = PagedRows::contiguous(k.data(), S, kv_cols, kv_cols);
        const auto vp = PagedRows::contiguous(v.data(), S, kv_cols, kv_cols);
        const std::size_t q_cols = dims.n_head * dims.head_dim;
        attention_gqa(dims, ConstRows(std::span<const float>(q), T, q_cols), kp, vp, S - T,
                      1.0f / std::sqrt(static_cast<float>(dims.head_dim)), Rows(std::span<float>(out), T, q_cols));
        consume(out);
    }
};

// Qwen3.x GDN layer dims (D-003): 16 key heads, 48 value heads, d_k = d_v = 128.
constexpr GdnDims kGdn{16, 48, 128, 128, GdnHeadMapping::Tiled};

struct GdnCase {
    std::size_t T;
    std::vector<float> q, k, v, g, beta, state, out;
    explicit GdnCase(std::size_t t)
        : T(t),
          q(random_vec(t * kGdn.n_k_heads * kGdn.d_k, 6)),
          k(random_vec(t * kGdn.n_k_heads * kGdn.d_k, 7)),
          v(random_vec(t * kGdn.n_v_heads * kGdn.d_v, 8)),
          g(random_vec(t * kGdn.n_v_heads, 9, -0.55f, -0.05f)),
          beta(random_vec(t * kGdn.n_v_heads, 10, 0.0f, 1.0f)),
          state(kGdn.n_v_heads * kGdn.d_k * kGdn.d_v),
          out(t * kGdn.n_v_heads * kGdn.d_v) {}
    GdnInputs inputs() const {
        const std::size_t qk = kGdn.n_k_heads * kGdn.d_k, vc = kGdn.n_v_heads * kGdn.d_v;
        return {ConstRows(std::span<const float>(q), T, qk), ConstRows(std::span<const float>(k), T, qk),
                ConstRows(std::span<const float>(v), T, vc), ConstRows(std::span<const float>(g), T, kGdn.n_v_heads),
                ConstRows(std::span<const float>(beta), T, kGdn.n_v_heads)};
    }
    void recurrent() {
        std::fill(state.begin(), state.end(), 0.0f);
        gated_delta_rule_recurrent(kGdn, inputs(), std::span<float>(state),
                                   Rows(std::span<float>(out), T, kGdn.n_v_heads * kGdn.d_v));
        consume(out);
    }
    void chunked() {
        std::fill(state.begin(), state.end(), 0.0f);
        gated_delta_rule_chunked(kGdn, inputs(), std::span<float>(state),
                                 Rows(std::span<float>(out), T, kGdn.n_v_heads * kGdn.d_v));
        consume(out);
    }
};

// Attention dims of a Qwen3.x full-attention layer: 16 query heads over 4 KV heads, d = 128.
constexpr AttentionDims kAttn{16, 4, 128};

// ---- budgets ---------------------------------------------------------------------------------

TEST(CpuPerfBudget, MatmulDecodeAndPrefillStayWithinBudget) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    MatmulCase decode(1, 2048, 2048);  // one token through a 2048x2048 projection
    expect_within_budget("matmul decode T=1 2048x2048", best_ms(15, [&] { decode.run(); }), 12.0);  // 1.1 ms
    MatmulCase prefill(32, 2048, 2048);  // a 32-token prefill chunk through the same projection
    expect_within_budget("matmul prefill T=32 2048x2048", best_ms(5, [&] { prefill.run(); }), 350.0);  // 35 ms
}

TEST(CpuPerfBudget, ArgmaxAndTopKOverAFullVocabularyStayWithinBudget) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    const auto logits = random_vec(151'936, 11);  // Qwen vocabulary
    std::int32_t idx = 0;
    expect_within_budget("argmax vocab=151936", best_ms(25, [&] { idx += argmax(logits).index; }), 2.0);  // 0.16 ms
    expect_within_budget("top_k k=40 vocab=151936", best_ms(25, [&] { idx += top_k(logits, 40).front().index; }),
                         3.0);  // 0.30 ms
    g_sink = g_sink + static_cast<float>(idx);
}

TEST(CpuPerfBudget, AttentionDecodeAndPrefillStayWithinBudget) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    AttnCase decode(kAttn, 1, 4096);  // one new token over a 4k history
    expect_within_budget("attention decode T=1 S=4096", best_ms(15, [&] { decode.run(); }), 60.0);  // 5.8 ms
    AttnCase prefill(kAttn, 64, 1024);  // 64-token prefill chunk at offset 960
    expect_within_budget("attention prefill T=64 S=1024", best_ms(5, [&] { prefill.run(); }), 900.0);  // 90 ms
}

TEST(CpuPerfBudget, GatedDeltaNetDecodeAndPrefillStayWithinBudget) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    GdnCase decode(1);
    expect_within_budget("gdn recurrent decode T=1", best_ms(25, [&] { decode.recurrent(); }), 5.0);  // 0.49 ms
    GdnCase prefill(128);
    expect_within_budget("gdn chunked prefill T=128", best_ms(5, [&] { prefill.chunked(); }), 1000.0);  // 98 ms
}

TEST(CpuPerfBudget, NormsAndElementwiseOpsStayWithinBudget) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    constexpr std::size_t R = 256, D = 4096;
    const auto x = random_vec(R * D, 12), z = random_vec(R * D, 13), w = random_vec(D, 14, 0.5f, 1.5f);
    std::vector<float> out(R * D);
    const ConstRows xr(std::span<const float>(x), R, D), zr(std::span<const float>(z), R, D);
    const Rows outr(std::span<float>(out), R, D);
    expect_within_budget("rms_norm 256x4096", best_ms(10, [&] {
                             rms_norm(xr, w, 1e-6f, outr);
                             consume(out);
                         }),
                         5.0);  // 0.42 ms
    expect_within_budget("swiglu 256x4096", best_ms(10, [&] {
                             swiglu(xr, zr, outr);
                             consume(out);
                         }),
                         60.0);  // 5.7 ms
    expect_within_budget("softmax_rows 256x4096", best_ms(10, [&] {
                             softmax_rows(xr, outr);
                             consume(out);
                         }),
                         100.0);  // 9.8 ms
}

// ---- complexity ------------------------------------------------------------------------------

TEST(CpuPerfScaling, AttentionDecodeIsLinearInHistoryLength) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    AttnCase small(kAttn, 1, 1024), large(kAttn, 1, 4096);
    const double a = best_ms(15, [&] { small.run(); });
    const double b = best_ms(15, [&] { large.run(); });
    expect_scaling("attention decode S 1024 -> 4096", a, b, 4.0, 7.0, "O(S) per decoded token");
}

TEST(CpuPerfScaling, GatedDeltaNetRecurrenceIsLinearInTokens) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    GdnCase small(16), large(64);
    const double a = best_ms(7, [&] { small.recurrent(); });
    const double b = best_ms(7, [&] { large.recurrent(); });
    expect_scaling("gdn recurrent T 16 -> 64", a, b, 4.0, 7.0, "O(T) with a fixed-size state");
}

TEST(CpuPerfScaling, MatmulDecodeIsLinearInOutputRows) {
    HALO_PERF_SKIP_IF_UNMEASURABLE();
    MatmulCase small(1, 512, 2048), large(1, 2048, 2048);
    const double a = best_ms(25, [&] { small.run(); });
    const double b = best_ms(15, [&] { large.run(); });
    expect_scaling("matmul decode N 512 -> 2048", a, b, 4.0, 7.0, "O(N*K) per token");
}

}  // namespace
