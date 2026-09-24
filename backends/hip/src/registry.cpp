// HIP kernel registry: the variant table (TRD §9 ids, §56 kernel selection).

#include "halo/backends/hip/registry.h"

#include <array>

#include "halo/core/error.h"

namespace halo::hip {

namespace {

constexpr std::string_view kGdnDecisions = "D-003 D-004(5,6) D-012 D-016";

constexpr std::array<KernelVariant, 21> kVariants{{
    {"GATED_DELTANET", "recurrent", "gdn_recurrent_b128", "k_gdn_recurrent", 128,
     "grid (n_v, ceil(d_v/128)); thread = one value column, S[:,c] in registers (d_k <= 128)",
     kGdnDecisions, true},
    {"GATED_DELTANET", "recurrent", "gdn_recurrent_b64", "k_gdn_recurrent", 64,
     "grid (n_v, ceil(d_v/64)); 2x the workgroups of b128 for d_v = 128", kGdnDecisions, false},
    {"GATED_DELTANET", "chunked", "gdn_chunked_b64", "k_gdn_check_g, k_gdn_chunk_intra, k_gdn_chunk_state", 64,
     "check: 1 thread per (t, head); intra: grid (n_v, chunks in group); state: grid (n_v, ceil(d_v/64)), "
     "thread = one value column, S[:,c] in LDS (32 KiB)",
     kGdnDecisions, true},
    {"GATED_DELTANET", "chunked", "gdn_chunked_b32", "k_gdn_check_g, k_gdn_chunk_intra, k_gdn_chunk_state", 32,
     "as gdn_chunked_b64 with 32-thread workgroups (one wave32; same static 32 KiB LDS tile)", kGdnDecisions, false},
    {"CONV1D_SHORT", "", "conv1d_silu_b256", "k_conv1d_silu", 256,
     "grid ceil(C/256); thread = one channel, K-1 history in registers (K <= 8)", "D-004(2) D-012", true},
    {"CONV1D_SHORT", "", "conv1d_silu_b64", "k_conv1d_silu", 64, "grid ceil(C/64)", "D-004(2) D-012", false},
    {"GATED_NORM", "", "gated_norm_b128", "k_norm", 128,
     "grid rows; lanes 0..7 = the CPU's 8 partial sums, thread 0 combines, then element-wise", "D-004(7)", true},
    {"GATED_NORM", "", "gated_norm_b32", "k_norm", 32, "as gated_norm_b128 with 32 threads (one wave32)",
     "D-004(7)", false},
    {"QUANT_GEMV", "", "gemv_wave32_r4", "k_gemv_wave", 128,
     "grid (ceil(N/4), T); one wave32 per row; lane owns 8-element groups l, l+32, ...; fixed LDS tree",
     "D-007 D-014", true},
    {"QUANT_GEMV", "", "gemv_wave32_r8", "k_gemv_wave", 256, "as gemv_wave32_r4 with 8 rows per workgroup",
     "D-007 D-014", false},
    {"QUANT_GEMV", "", "gemv_generic_b64", "k_gemv_generic", 64,
     "grid (ceil(N/8), T); 8 lanes per row in the CPU dot order (bit-identical to cpu::matmul); per-element "
     "dequant, slow",
     "D-007 D-014", false},
    {"ARGMAX_FUSED", "", "argmax_b256", "k_argmax_partial, k_argmax_reduce", 256,
     "partial: 16 logits per thread + LDS tree per workgroup; reduce: one workgroup per vector. Also the "
     "LM head's stage 2 (stage 1 = the QUANT_GEMV variant's argmax epilogue)",
     "D-016", true},
    {"TOP_K", "", "topk_bitonic_b256", "k_topk", 256,
     "rounds of 2048-candidate bitonic sorts in LDS keeping k per chunk until one chunk remains (k <= 1024)",
     "TRD 19, D-016", true},
    {"RMS_NORM", "", "rms_norm_b128", "k_norm", 128, "as gated_norm_b128 without the gate", "D-004", true},
    {"RMS_NORM", "", "rms_norm_b32", "k_norm", 32, "one wave32 per row", "D-004", false},
    {"PARTIAL_ROPE", "", "rope_neox_b128", "k_rope", 128,
     "grid (T, ceil(heads*half/128)); cos/sin per (t, i) in LDS; thread = one rotated pair", "D-004", true},
    {"SWIGLU", "", "swiglu_b256", "k_eltwise", 256, "grid (rows, ceil(cols/256)); thread = one element", "D-004",
     true},
    {"MUL_SIGMOID", "", "mul_sigmoid_b256", "k_eltwise", 256, "grid (rows, ceil(cols/256)); thread = one element",
     "D-004", true},
    {"ATTENTION", "", "attn_online_b128", "k_attn_check, k_attn_online", 128,
     "grid (T, n_head); key tiles of 128 with a running max; thread = one key score, then 2 output dims",
     "D-004", true},
    {"ATTENTION", "", "attn_online_b64", "k_attn_check, k_attn_online", 64, "as attn_online_b128 with 64-key tiles",
     "D-004", false},
    {"ATTENTION", "", "attn_exact_b128", "k_attn_check, k_attn_exact", 128,
     "grid (T, n_head); the CPU's three passes and sequential sums (bit-identical, serial, slow)", "D-004",
     false},
}};

}  // namespace

std::span<const KernelVariant> kernel_variants() noexcept { return kVariants; }

std::vector<KernelVariant> kernel_variants(std::string_view op) {
    std::vector<KernelVariant> out;
    for (const KernelVariant& v : kVariants) {
        if (v.op == op) out.push_back(v);
    }
    return out;
}

const KernelVariant& find_variant(std::string_view op, std::string_view name) {
    for (const KernelVariant& v : kVariants) {
        if (v.op == op && v.name == name) return v;
    }
    throw_error(ErrorCode::Config, "HIP: no kernel variant '{}' for operator {}", name, op);
}

const KernelVariant& default_variant(std::string_view op, std::string_view form) {
    for (const KernelVariant& v : kVariants) {
        if (v.op == op && v.form == form && v.is_default) return v;
    }
    throw_error(ErrorCode::Config, "HIP: no default kernel variant for operator {} {}", op, form);
}

}  // namespace halo::hip
