#pragma once
/// \file
/// First HALO Vulkan compute operators (TRD §9 ids in brackets). Each method validates
/// shapes and buffer sizes on the host (overflow-checked), then records one or more
/// dispatches into `stream`; nothing executes until the stream is submitted.
///
/// All tensors are dense, row-major, fp32 unless stated. Buffers passed as outputs must
/// be distinct from inputs (no in-place aliasing).
///
/// Status: correctness-first kernels validated on lavapipe only (DECISIONS D-001). They
/// use fixed-order shared-memory tree reductions (deterministic run to run) and one
/// workgroup per output row / head; no performance claim is made for them.
///
/// Thread safety: an Ops instance is not thread-safe (it builds kernels lazily into an
/// internal cache); use one per thread or guard it externally.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/kernel.h"
#include "halo/tensor/dtype.h"

namespace halo::vulkan {

class Context;

struct OpsOptions {
    /// Workgroup size (power of two, 32..1024) of the reduction kernels (rms_norm,
    /// matvec_*, argmax). Passed as specialization constant 0.
    std::uint32_t reduce_workgroup = 256;
    /// Workgroup size of gated_delta_rule_decode (threads map to d_v columns; need not be a
    /// power of two).
    std::uint32_t gdn_workgroup = 128;
    /// Largest d_k gated_delta_rule_decode accepts (shared-memory q/k tile size).
    std::uint32_t gdn_max_dk = 256;
};

/// Byte size of one row of `cols` elements of weight type `t` (F32, Q8_0, Q4_K, Q6_K).
/// Throws Error(Unsupported) for other types, Error(Kernel) if cols is not a multiple
/// of the block size.
[[nodiscard]] std::uint64_t matvec_row_bytes(DType t, std::uint32_t cols);

/// Inputs of a recurrent Gated DeltaNet decode over T = n_tokens rows (DECISIONS D-004
/// item 6). T = 1 is the single-token decode step; T > 1 is the MTP-verify form, which
/// can also record per-row state slots for rollback (architecture decision C-1).
///
/// q/k contract — DECISIONS **D-016**, identical to `halo::cpu::GdnQkParams` (same two
/// fields, same defaults, same order of operations, same validation):
///  - q, k: [T, n_k, d_k], the RAW per-head rows (after conv + SiLU).
///  - if `qk_l2norm`: q <- q * (1 / sqrt(sum(q^2) + 1e-6)), likewise k, per head (in-kernel
///    reduction, fixed order).
///  - then q <- q * q_scale (k is never scaled). `q_scale` = nullopt means 1/sqrt(d_k); it
///    must be finite (Error(Kernel) otherwise). The kernel applies exactly this and
///    nothing else. The qwen35 forward passes qk_l2norm = true, q_scale = 1/sqrt(128).
///    Callers holding pre-normalized, pre-scaled q/k pass qk_l2norm = false, q_scale = 1.
///
/// Other conventions (shared with backends/cpu `gated_delta_rule_recurrent`):
///  - v: [T, n_v, d_v].  g: [T, n_v] log-decay (the kernel applies exp(g); g <= 0 in the
///    model).  beta: [T, n_v], already sigmoid-activated.
///  - Head mapping is GGUF *tiled*: value head j uses key head `j % n_k` (D-004 item 5).
///  - A state is [n_v, d_k, d_v] fp32, element (j, a, b) at `(j*d_k + a)*d_v + b`
///    (a = key dim, b = value dim, d_v fastest; HF layout — ggml stores the transpose).
///  - out: [T, n_v, d_v].
/// Per row t and head j:  S *= exp(g);  kv = Sᵀk;  δ = (v − kv)·β;  S += k δᵀ;  o = Sᵀq.
///
/// State placement (all offsets in **float elements**, no alignment requirement):
///  - input state: `state` at `state_offset`.
///  - final state: `state_out` (nullptr = `state`) at `state_out_offset` (nullopt = same
///    offset as the input). Default = in place. When input and output live in the same
///    buffer their regions must be identical or disjoint.
///  - `state_slots` (optional): `n_slots` states back to back starting at `slots_offset`;
///    slot s receives the state after row T-1-s (slot 0 = most recent = final state) for
///    s < min(T, n_slots); slots s >= T are left untouched. Rolling back r rows = using
///    slot r as the state. Must not overlap the input/output state regions or any input.
struct GdnDecodeArgs {
    const Buffer* q = nullptr;
    const Buffer* k = nullptr;
    const Buffer* v = nullptr;
    const Buffer* g = nullptr;
    const Buffer* beta = nullptr;
    Buffer* state = nullptr;  ///< input state (and output when state_out is null)
    Buffer* out = nullptr;
    std::uint32_t n_v = 0;
    std::uint32_t n_k = 0;
    std::uint32_t d_k = 0;
    std::uint32_t d_v = 0;

    std::uint32_t n_tokens = 1;  ///< T
    std::uint64_t state_offset = 0;
    Buffer* state_out = nullptr;
    std::optional<std::uint64_t> state_out_offset;
    Buffer* state_slots = nullptr;
    std::uint64_t slots_offset = 0;
    std::uint32_t n_slots = 0;

    bool qk_l2norm = true;         ///< D-016: L2-normalize q and k per head in the kernel
    std::optional<float> q_scale;  ///< D-016: q multiplier after normalization; nullopt = 1/sqrt(d_k)
};

/// Result layout written by argmax(): three 32-bit words
///   {index (uint32), value (float bits), nan (1 if any logit was NaN, else 0)}.
/// Read it with read_argmax() / decode_argmax(), which raise Error(Kernel) when the NaN
/// word is set, like cpu::argmax (DECISIONS D-016, code review S-3).
inline constexpr std::uint32_t k_argmax_result_bytes = 12;
/// Index word when no element was eligible (every logit NaN); decode_argmax never returns it.
inline constexpr std::uint32_t k_argmax_none = 0xFFFFFFFFu;

struct ArgmaxResult {
    std::uint32_t index = 0;
    float value = 0.0f;
};

/// Decodes the three result words. Throws Error(Kernel) when the NaN word is set (a NaN
/// logit is a bug upstream and must not be sampled silently; D-016), or when there is no
/// index.
[[nodiscard]] ArgmaxResult decode_argmax(std::span<const std::uint32_t, 3> words);
/// Downloads k_argmax_result_bytes at byte `offset` of `result` (after the stream that
/// wrote it has been waited on) and decodes them (same errors as decode_argmax).
[[nodiscard]] ArgmaxResult read_argmax(const Buffer& result, std::uint64_t offset = 0);

class Ops {
public:
    explicit Ops(std::shared_ptr<Context> ctx, OpsOptions options = {});

    [[nodiscard]] const OpsOptions& options() const noexcept { return options_; }

    /// [RMS_NORM] y[r,i] = x[r,i] / sqrt(mean_i(x[r,·]²) + eps) * w[i].
    /// x, y: [rows, cols]; w: [cols]. Plain x̂·w (GGUF stores 1+w; D-004).
    void rms_norm(Stream& stream, const Buffer& x, const Buffer& w, Buffer& y, std::uint32_t rows,
                  std::uint32_t cols, float eps);

    /// [QUANT_GEMV / MATMUL] y[r] = Σ_c W[r,c]·x[c]. W is `rows` rows of `cols` elements in
    /// ggml layout for `wtype` (F32 row-major; Q8_0/Q4_K/Q6_K: consecutive ggml blocks per
    /// row, rows back to back). x: [cols] fp32; y: [rows] fp32. rows may exceed the
    /// device workgroup-count limit (a 2-D grid is used). The W buffer must hold at least
    /// rows*matvec_row_bytes(wtype, cols) bytes rounded up to a multiple of 4 (quantized
    /// blocks are read as 32-bit words; Q8_0/Q6_K rows are not word-aligned).
    void matvec(Stream& stream, DType wtype, const Buffer& w, const Buffer& x, Buffer& y,
                std::uint32_t rows, std::uint32_t cols);

    /// [GATED_DELTANET] recurrent decode over args.n_tokens rows; see GdnDecodeArgs.
    void gated_delta_rule_decode(Stream& stream, const GdnDecodeArgs& args);

    /// [ARGMAX_FUSED building block] index of the maximum of `logits[0..n)`. Two passes:
    /// per-workgroup partials into `scratch` (argmax_scratch_bytes(n) bytes), then one
    /// workgroup reduces them into `result` (k_argmax_result_bytes; read it with
    /// read_argmax). Ties resolve to the lowest index; -inf is an ordinary value; any NaN
    /// sets the result's NaN word, so read_argmax raises Error(Kernel) (D-016). Every
    /// dispatch rewrites all three words, so a result buffer can be reused.
    void argmax(Stream& stream, const Buffer& logits, std::uint32_t n, Buffer& scratch,
                Buffer& result);
    [[nodiscard]] std::uint64_t argmax_scratch_bytes(std::uint32_t n) const;
    [[nodiscard]] std::uint32_t argmax_partials(std::uint32_t n) const;

private:
    const Kernel& kernel(const std::string& shader, std::uint32_t num_buffers,
                         std::uint32_t push_bytes, std::vector<SpecConstant> spec,
                         std::array<std::uint32_t, 3> local);

    std::shared_ptr<Context> ctx_;
    OpsOptions options_;
    std::map<std::string, std::unique_ptr<Kernel>> kernels_;
};

}  // namespace halo::vulkan
