#pragma once
// TunableOps over the real CPU reference kernels (library halo_autotune_cpu, which links
// halo_backend_cpu; the runtime-lookup library halo_autotune does not).
//
// Candidates:
//   CpuMatmulTunable     threads            (halo::cpu::matmul, dense f32 weights)
//   CpuGdnChunkedTunable chunk x threads    (halo::cpu::gated_delta_rule_chunked)
// Thread pools are created once per thread count in the constructor (never inside a timed
// run). Inputs are deterministic from a seed.
//
// Correctness gates (validate): matmul output must be bit-identical to the 1-thread result
// (the ThreadPool contract); chunked GDN output must match the recurrent form within
// max |diff| <= gdn_tolerance (the two forms differ by fp32 rounding only).

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "halo/autotune/tuner.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/thread_pool.h"

namespace halo::autotune {

class CpuMatmulTunable final : public TunableOp {
public:
    /// y[T, N] = x[T, K] W^T. thread_counts must be non-empty and each >= 1.
    CpuMatmulTunable(std::size_t tokens, std::size_t k, std::size_t n, std::vector<unsigned> thread_counts,
                     std::uint64_t seed = 1);
    ~CpuMatmulTunable() override;

    [[nodiscard]] OpKey key() const override;
    [[nodiscard]] std::string backend() const override { return "cpu"; }
    [[nodiscard]] std::vector<Candidate> candidates() const override;
    Measurement run(const Candidate& c) override;
    bool validate(const Candidate& c, std::string& why) override;
    [[nodiscard]] std::optional<CostInputs> cost(const Candidate& c) const override;

private:
    cpu::ThreadPool& pool(const Candidate& c);
    std::size_t t_, k_, n_;
    std::vector<unsigned> threads_;
    std::map<unsigned, std::unique_ptr<cpu::ThreadPool>> pools_;
    std::vector<float> x_, w_, y_, ref_;
};

class CpuGdnChunkedTunable final : public TunableOp {
public:
    CpuGdnChunkedTunable(const cpu::GdnDims& dims, std::size_t tokens, std::vector<unsigned> chunk_sizes,
                         std::vector<unsigned> thread_counts, std::uint64_t seed = 2, float gdn_tolerance = 1e-3f);
    ~CpuGdnChunkedTunable() override;

    [[nodiscard]] OpKey key() const override;
    [[nodiscard]] std::string backend() const override { return "cpu"; }
    [[nodiscard]] std::vector<Candidate> candidates() const override;
    Measurement run(const Candidate& c) override;
    bool validate(const Candidate& c, std::string& why) override;
    /// flops: an operation-count estimate of the chunked algorithm (O(T·cs·(2dk+dv)) for the
    /// intra-chunk terms + O(T·dk·dv) for the state terms, per value head); bytes: the
    /// compulsory traffic of halo::cpu::traffic_gated_delta_rule.
    [[nodiscard]] std::optional<CostInputs> cost(const Candidate& c) const override;

private:
    cpu::ThreadPool& pool(unsigned threads);
    void run_chunked(unsigned chunk, unsigned threads, std::vector<float>& out);
    cpu::GdnDims dims_;
    std::size_t t_;
    std::vector<unsigned> chunks_, threads_;
    float tol_;
    std::map<unsigned, std::unique_ptr<cpu::ThreadPool>> pools_;
    std::vector<float> q_, k_, v_, g_, beta_, state0_, state_, out_, ref_;
};

}  // namespace halo::autotune
