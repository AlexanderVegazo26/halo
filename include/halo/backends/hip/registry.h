#pragma once
/// \file
/// HALO HIP kernel registry (TRD §9 operator ids, §17 hip_kernel_registry, §56): every
/// kernel variant the backend can run, with its launch parameters, so the autotuner can
/// enumerate candidates and a StepPlan can name one. Ops select a variant by name
/// (OpsOptions); unknown names are rejected.

#include <span>
#include <string_view>
#include <vector>

namespace halo::hip {

struct KernelVariant {
    std::string_view op;          ///< TRD §9 operator id, e.g. "GATED_DELTANET"
    std::string_view form;        ///< sub-form of the op ("recurrent", "chunked"), or ""
    std::string_view name;        ///< unique variant name, e.g. "gdn_recurrent_b128"
    std::string_view kernels;     ///< device kernel(s) launched
    unsigned block = 0;           ///< threads per workgroup
    std::string_view mapping;     ///< work decomposition (grid / thread roles)
    std::string_view decisions;   ///< DECISIONS.md entries implemented
    bool is_default = false;      ///< the variant an Ops uses when not told otherwise
};

/// All registered variants (static storage).
[[nodiscard]] std::span<const KernelVariant> kernel_variants() noexcept;
/// The variants of one operator id (possibly empty).
[[nodiscard]] std::vector<KernelVariant> kernel_variants(std::string_view op);
/// The named variant of `op`; throws Error(Config) when it does not exist.
[[nodiscard]] const KernelVariant& find_variant(std::string_view op, std::string_view name);
/// The default variant of (`op`, `form`); throws Error(Config) when there is none.
[[nodiscard]] const KernelVariant& default_variant(std::string_view op, std::string_view form = "");

}  // namespace halo::hip
