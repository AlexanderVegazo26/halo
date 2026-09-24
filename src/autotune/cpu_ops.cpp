#include "halo/autotune/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <format>

#include "halo/backends/cpu/traffic.h"
#include "halo/backends/cpu/weight_matrix.h"
#include "halo/core/error.h"

namespace halo::autotune {

namespace {

/// Deterministic values in [-1, 1) (SplitMix64; portable).
void fill(std::vector<float>& v, std::uint64_t seed) {
    std::uint64_t s = seed;
    for (auto& x : v) {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        x = static_cast<float>(static_cast<double>(z >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
}

void check_threads(const std::vector<unsigned>& t) {
    HALO_CHECK(!t.empty(), ErrorCode::Config, "cpu tunable: no thread counts");
    for (const unsigned n : t) HALO_CHECK(n >= 1 && n <= 1024, ErrorCode::Config, "cpu tunable: threads {}", n);
}

void make_pools(const std::vector<unsigned>& counts, std::map<unsigned, std::unique_ptr<cpu::ThreadPool>>& pools) {
    for (const unsigned n : counts) {
        if (!pools.contains(n)) pools.emplace(n, std::make_unique<cpu::ThreadPool>(n));
    }
}

unsigned as_unsigned(std::int64_t v, const char* what) {
    HALO_CHECK(v >= 1 && v <= 1 << 20, ErrorCode::Config, "cpu tunable: {} = {} out of range", what, v);
    return static_cast<unsigned>(v);
}

}  // namespace

// ---- matmul -------------------------------------------------------------------------------

CpuMatmulTunable::CpuMatmulTunable(std::size_t tokens, std::size_t k, std::size_t n,
                                   std::vector<unsigned> thread_counts, std::uint64_t seed)
    : t_(tokens), k_(k), n_(n), threads_(std::move(thread_counts)) {
    HALO_CHECK(t_ >= 1 && k_ >= 1 && n_ >= 1 && t_ <= (1U << 16) && k_ <= (1U << 16) && n_ <= (1U << 20),
               ErrorCode::Config, "CpuMatmulTunable: shape {}x{}x{} out of range", t_, k_, n_);
    check_threads(threads_);
    x_.resize(t_ * k_);
    w_.resize(n_ * k_);
    y_.resize(t_ * n_);
    fill(x_, seed);
    fill(w_, seed + 1);
    make_pools(threads_, pools_);
    // Reference: 1 thread, no pool.
    ref_.resize(t_ * n_);
    cpu::matmul(cpu::ConstRows(std::span<const float>(x_), t_, k_),
                cpu::WeightMatrix::dense(cpu::ConstRows(std::span<const float>(w_), n_, k_)),
                cpu::Rows(std::span<float>(ref_), t_, n_), nullptr);
}

CpuMatmulTunable::~CpuMatmulTunable() = default;

OpKey CpuMatmulTunable::key() const { return {"MATMUL", std::format("T={},K={},N={},w=f32", t_, k_, n_)}; }

std::vector<Candidate> CpuMatmulTunable::candidates() const {
    std::vector<Candidate> out;
    for (const unsigned n : threads_) out.push_back(Candidate{{{"threads", n}}});
    return out;
}

cpu::ThreadPool& CpuMatmulTunable::pool(const Candidate& c) {
    const unsigned n = as_unsigned(c.get("threads"), "threads");
    const auto it = pools_.find(n);
    HALO_CHECK(it != pools_.end(), ErrorCode::Config, "CpuMatmulTunable: no pool for {} threads", n);
    return *it->second;
}

Measurement CpuMatmulTunable::run(const Candidate& c) {
    cpu::matmul(cpu::ConstRows(std::span<const float>(x_), t_, k_),
                cpu::WeightMatrix::dense(cpu::ConstRows(std::span<const float>(w_), n_, k_)),
                cpu::Rows(std::span<float>(y_), t_, n_), &pool(c));
    return {};
}

bool CpuMatmulTunable::validate(const Candidate& c, std::string& why) {
    (void)run(c);
    if (y_ != ref_) {
        why = std::format("matmul with {} differs from the 1-thread result", c.to_string());
        return false;
    }
    return true;
}

std::optional<CostInputs> CpuMatmulTunable::cost(const Candidate& c) const {
    CostInputs in;
    in.flops = 2.0 * static_cast<double>(t_) * static_cast<double>(k_) * static_cast<double>(n_);
    in.bytes = cpu::traffic_matmul(t_, k_, n_, 4ULL * k_ * n_).total();
    in.tier = hardware::MemoryTier::Host;
    in.parallelism = std::min<unsigned>(as_unsigned(c.get("threads"), "threads"), static_cast<unsigned>(n_));
    in.launches = 1;
    in.syncs = 1;
    return in;
}

// ---- chunked GDN --------------------------------------------------------------------------

CpuGdnChunkedTunable::CpuGdnChunkedTunable(const cpu::GdnDims& dims, std::size_t tokens,
                                           std::vector<unsigned> chunk_sizes, std::vector<unsigned> thread_counts,
                                           std::uint64_t seed, float gdn_tolerance)
    : dims_(dims), t_(tokens), chunks_(std::move(chunk_sizes)), threads_(std::move(thread_counts)), tol_(gdn_tolerance) {
    HALO_CHECK(dims_.n_k_heads >= 1 && dims_.n_v_heads >= 1 && dims_.d_k >= 1 && dims_.d_v >= 1 &&
                   dims_.n_v_heads % dims_.n_k_heads == 0 && dims_.d_k <= 512 && dims_.d_v <= 512 &&
                   dims_.n_v_heads <= 256,
               ErrorCode::Config, "CpuGdnChunkedTunable: invalid dims");
    HALO_CHECK(t_ >= 1 && t_ <= (1U << 16), ErrorCode::Config, "CpuGdnChunkedTunable: tokens {} out of range", t_);
    HALO_CHECK(!chunks_.empty(), ErrorCode::Config, "CpuGdnChunkedTunable: no chunk sizes");
    for (const unsigned c : chunks_) {
        HALO_CHECK(c >= 1 && c <= 1024, ErrorCode::Config, "CpuGdnChunkedTunable: chunk {} not in [1, 1024]", c);
    }
    check_threads(threads_);
    const std::size_t qk = dims_.n_k_heads * dims_.d_k;
    const std::size_t vv = dims_.n_v_heads * dims_.d_v;
    q_.resize(t_ * qk);
    k_.resize(t_ * qk);
    v_.resize(t_ * vv);
    g_.resize(t_ * dims_.n_v_heads);
    beta_.resize(t_ * dims_.n_v_heads);
    state0_.resize(dims_.n_v_heads * dims_.d_k * dims_.d_v);
    fill(q_, seed);
    fill(k_, seed + 1);
    fill(v_, seed + 2);
    fill(g_, seed + 3);
    fill(beta_, seed + 4);
    fill(state0_, seed + 5);
    for (auto& x : g_) x = -0.05f - 0.25f * std::fabs(x);  // log decay: must be <= 0
    for (auto& x : beta_) x = 0.5f + 0.5f * std::fabs(x) * 0.9f;  // in (0.5, 0.95)
    for (auto& x : state0_) x *= 0.1f;
    out_.resize(t_ * vv);
    make_pools(threads_, pools_);
    // Reference: the recurrent form, single thread.
    state_ = state0_;
    ref_.resize(t_ * vv);
    const cpu::GdnInputs in{cpu::ConstRows(std::span<const float>(q_), t_, qk),
                            cpu::ConstRows(std::span<const float>(k_), t_, qk),
                            cpu::ConstRows(std::span<const float>(v_), t_, vv),
                            cpu::ConstRows(std::span<const float>(g_), t_, dims_.n_v_heads),
                            cpu::ConstRows(std::span<const float>(beta_), t_, dims_.n_v_heads)};
    cpu::gated_delta_rule_recurrent(dims_, in, std::span<float>(state_), cpu::Rows(std::span<float>(ref_), t_, vv),
                                    cpu::GdnQkParams{}, nullptr);
}

CpuGdnChunkedTunable::~CpuGdnChunkedTunable() = default;

OpKey CpuGdnChunkedTunable::key() const {
    return {"GATED_DELTANET", std::format("T={},Hk={},Hv={},dk={},dv={},impl=chunked", t_, dims_.n_k_heads,
                                          dims_.n_v_heads, dims_.d_k, dims_.d_v)};
}

std::vector<Candidate> CpuGdnChunkedTunable::candidates() const {
    std::vector<Candidate> out;
    for (const unsigned c : chunks_) {
        for (const unsigned n : threads_) out.push_back(Candidate{{{"chunk", c}, {"threads", n}}});
    }
    return out;
}

cpu::ThreadPool& CpuGdnChunkedTunable::pool(unsigned threads) {
    const auto it = pools_.find(threads);
    HALO_CHECK(it != pools_.end(), ErrorCode::Config, "CpuGdnChunkedTunable: no pool for {} threads", threads);
    return *it->second;
}

void CpuGdnChunkedTunable::run_chunked(unsigned chunk, unsigned threads, std::vector<float>& out) {
    const std::size_t qk = dims_.n_k_heads * dims_.d_k;
    const std::size_t vv = dims_.n_v_heads * dims_.d_v;
    // The state copy is part of each timed run (it is O(Hv·dk·dv), small next to the op).
    std::copy(state0_.begin(), state0_.end(), state_.begin());
    const cpu::GdnInputs in{cpu::ConstRows(std::span<const float>(q_), t_, qk),
                            cpu::ConstRows(std::span<const float>(k_), t_, qk),
                            cpu::ConstRows(std::span<const float>(v_), t_, vv),
                            cpu::ConstRows(std::span<const float>(g_), t_, dims_.n_v_heads),
                            cpu::ConstRows(std::span<const float>(beta_), t_, dims_.n_v_heads)};
    cpu::gated_delta_rule_chunked(dims_, in, std::span<float>(state_), cpu::Rows(std::span<float>(out), t_, vv),
                                  cpu::GdnQkParams{}, chunk, &pool(threads));
}

Measurement CpuGdnChunkedTunable::run(const Candidate& c) {
    run_chunked(as_unsigned(c.get("chunk"), "chunk"), as_unsigned(c.get("threads"), "threads"), out_);
    return {};
}

bool CpuGdnChunkedTunable::validate(const Candidate& c, std::string& why) {
    (void)run(c);
    float worst = 0.0f;
    for (std::size_t i = 0; i < out_.size(); ++i) {
        const float d = std::fabs(out_[i] - ref_[i]);
        if (!(d <= worst)) worst = d;  // also propagates NaN
    }
    if (!(worst <= tol_)) {
        why = std::format("chunked output differs from recurrent by {} > {}", worst, tol_);
        return false;
    }
    return true;
}

std::optional<CostInputs> CpuGdnChunkedTunable::cost(const Candidate& c) const {
    const double cs = static_cast<double>(as_unsigned(c.get("chunk"), "chunk"));
    const double t = static_cast<double>(t_);
    const double dk = static_cast<double>(dims_.d_k);
    const double dv = static_cast<double>(dims_.d_v);
    const double h = static_cast<double>(dims_.n_v_heads);
    const double eff_cs = std::min(cs, t);
    // Per value head: intra-chunk ut/attn (2·cs·dk per token), forward substitution
    // (cs·(dk+dv) per token), attn·v_new (cs·dv per token), state terms (~4·dk·dv per token)
    // and one state update per chunk (cs·dk·dv per chunk -> dk·dv per token). Estimate.
    const double per_tok = eff_cs * (2 * dk + dk + dv + dv) + 5 * dk * dv;
    CostInputs in;
    in.flops = 2.0 * h * t * per_tok;
    in.bytes = cpu::traffic_gated_delta_rule(dims_, t_).total();
    in.tier = hardware::MemoryTier::Host;
    in.parallelism = std::min<unsigned>(as_unsigned(c.get("threads"), "threads"),
                                        static_cast<unsigned>(dims_.n_v_heads));
    in.launches = 1;
    in.syncs = 1;
    return in;
}

}  // namespace halo::autotune
