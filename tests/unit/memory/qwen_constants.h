#pragma once
// Real Qwen3.8-27B numbers used by the planner tests.

#include "halo/memory/planner.h"

namespace halo_test {

inline constexpr std::uint64_t KiB = 1ULL << 10;
inline constexpr std::uint64_t MiB = 1ULL << 20;
inline constexpr std::uint64_t GiB = 1ULL << 30;

/// Per-class bytes of unsloth Qwen3.8-27B-UD-Q4_K_XL.gguf, computed from the real GGUF
/// header (/root/halo-ref/unsloth-ud-q4kxl.summary.json): for every tensor,
/// prod(dims) / block_elems × block_bytes using gguf-py GGML_QUANT_SIZES. Classes:
/// token_embd → embeddings; output → lm_head; output_norm and blk.*.*norm (except
/// ssm_norm) → norms; blk.64.* → mtp; blk.*.ffn_* → ffn; blk.*.{ssm_*,attn_qkv,attn_gate}
/// → gdn; remaining blk.*.attn_* → attention. Total 17,548,181,504 B (≈ 17.55 GB).
inline halo::memory::WeightBytes ud_q4_k_xl_weights() {
    halo::memory::WeightBytes w;
    w.embeddings = 715161600;
    w.lm_head = 1042944000;
    w.attention = 1146511360;
    w.gdn = 3673090048;
    w.ffn = 10616791040;
    w.norms = 2674688;
    w.mtp = 351008768;
    return w;
}
inline constexpr std::uint64_t kUdWeightsTotal = 17548181504ULL;

// D-003 facts.
inline constexpr std::uint64_t kGdnRecurrentPerSeq = 48ULL * 48 * 128 * 128 * 4;  // 144 MiB
inline constexpr std::uint64_t kGdnConvPerSeq = 48ULL * 3 * 10240 * 4;            // 5.625 MiB
inline constexpr std::uint64_t kGdnStatePerCopy = kGdnRecurrentPerSeq + kGdnConvPerSeq;
// TRD §2.2 / D-003: 16 layers × 4 KV heads × 256 × (K,V) × 2 B.
inline constexpr std::uint64_t kKvF16PerToken = 16ULL * 4 * 256 * 2 * 2;  // 64 KiB
inline constexpr std::uint64_t kMtpKvF16PerToken = 1ULL * 4 * 256 * 2 * 2;  // one layer

inline halo::memory::ModelShape qwen() { return halo::memory::qwen38_27b_shape(ud_q4_k_xl_weights()); }

}  // namespace halo_test
