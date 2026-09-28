#pragma once
// TEST-ONLY: the pre-backend qwen35 forward (src/models/qwen35.cpp at b874c51, unchanged since
// 8065734; verbatim in
// legacy_qwen35.cpp), the reference of the WS-BI-1 migration gate (ADR-001 §6.2). Same
// public API as models::Qwen35 over the same step/result types. Delete at the end of WS-BI-2.
// WS-BI-2 note: the GDN/conv state calls moved to the state ring overloads (ADR §5.3), which
// are bit-identical to the slot overloads this file used before (tests/unit/cpu_kernels).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "halo/models/qwen35.h"

namespace halo::models::legacy {

class LegacyQwen35 {
public:
    LegacyQwen35(const model::NormalizedModel& model, cpu::ThreadPool* pool);
    LegacyQwen35(const model::NormalizedModel& model, cpu::ThreadPool* pool, const Qwen35Options& options);
    ~LegacyQwen35();
    LegacyQwen35(const LegacyQwen35&) = delete;
    LegacyQwen35& operator=(const LegacyQwen35&) = delete;

    [[nodiscard]] const model::Qwen35HParams& hparams() const noexcept;
    [[nodiscard]] const model::NormalizedModel& model() const noexcept { return *model_; }
    [[nodiscard]] bool has_mtp() const noexcept;
    [[nodiscard]] std::size_t n_vocab() const noexcept;
    [[nodiscard]] std::size_t n_embd() const noexcept;
    [[nodiscard]] std::size_t gdn_chunk() const noexcept;
    [[nodiscard]] kv_cache::KvLayout kv_layout(std::size_t block_tokens = 16) const;
    [[nodiscard]] kv_cache::KvLayout mtp_kv_layout(std::size_t block_tokens = 16) const;
    [[nodiscard]] state::GdnShape gdn_shape() const;
    void forward(std::span<const SeqStep> steps, StepResult& out, const ForwardOptions& opts = {}) const;
    void mtp_forward(std::span<const MtpStep> steps, StepResult& out) const;
    [[nodiscard]] std::uint64_t trunk_weight_bytes() const noexcept;
    [[nodiscard]] std::uint64_t lm_head_bytes() const noexcept;
    [[nodiscard]] std::uint64_t mtp_block_bytes() const noexcept;
    [[nodiscard]] std::uint64_t mtp_head_bytes() const noexcept;
    [[nodiscard]] std::uint64_t embedding_row_bytes() const noexcept;

    struct Impl;

private:
    const model::NormalizedModel* model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace halo::models::legacy
