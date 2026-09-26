#pragma once
/// \file
/// HALO Vulkan compute operators (TRD §9 ids in brackets). Each method validates shapes,
/// views and buffer sizes on the host (overflow-checked), then records one or more
/// dispatches into `stream`; nothing executes until the stream is submitted.
///
/// Semantics are those of the CPU reference operators in `halo/backends/cpu/ops.h`: every
/// kernel here is accepted only by differential tests against `halo::cpu` and
/// `halo::tensor` on the same input buffers (TRD §30; DECISIONS D-016; code review M-2).
///
/// Operands are `BufferView`s (code review S-1): a byte range of a Buffer with an optional
/// row stride, so weights can share one arena buffer and a fused projection row (e.g. the
/// GDN qkv row) can be read as q / k / v views without copies. A view's offset need not be
/// a multiple of minStorageBufferOffsetAlignment: each operand is bound at its own offset
/// rounded down to that alignment, with only its own extent (so an operand deep inside a
/// large arena is not limited by maxStorageBufferRange), and the sub-alignment remainder
/// is passed to the shader as a push constant. fp32 operands need 4-byte-aligned offsets
/// and strides; quantized weights may start at any byte.
///
/// Aliasing: an output may not overlap any input or other output, compared as byte ranges
/// (the extent a view touches, [offset, offset + (rows-1)*stride + row_bytes)) within the
/// same VkBuffer. The only exception is the GDN state, whose input and output regions may
/// be identical (in place). Strided views are compared by their whole extent, so two
/// element-disjoint interleaved views of one buffer are conservatively rejected when one
/// of them is an output.
///
/// Status: correctness-first kernels validated on lavapipe only (DECISIONS D-001). They
/// use fixed-order shared-memory tree reductions (deterministic run to run) and one
/// workgroup per output row / head; no performance claim is made for them.
///
/// Thread safety: an Ops instance is not thread-safe (it builds kernels lazily into an
/// internal cache); use one per thread or guard it externally.

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/kernel.h"
#include "halo/tensor/dtype.h"

namespace halo::vulkan {

class Context;

/// A byte range of a Buffer used as an operator operand (code review S-1).
///  - `offset`: bytes from the start of the buffer (any value; see the file comment).
///  - `bytes`: size of the range the operator may touch; 0 = to the end of the buffer.
///    The operator checks that everything it reads or writes lies inside the range.
///  - `row_stride`: bytes between the starts of consecutive rows of a 2-D operand; 0 =
///    dense (the operator's natural row size). Ignored for 1-D operands.
/// A view is non-owning; the Buffer must outlive every recording that uses it. The view is
/// const (it names memory), but the memory of an output view is written.
struct BufferView {
    const Buffer* buffer = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t row_stride = 0;

    BufferView() = default;
    /// The whole buffer, dense (implicit so a Buffer can be passed where a view is expected).
    BufferView(const Buffer& b) noexcept : buffer(&b) {}  // NOLINT(google-explicit-constructor)
    BufferView(const Buffer& b, std::uint64_t offset_bytes, std::uint64_t size_bytes = 0,
               std::uint64_t row_stride_bytes = 0) noexcept
        : buffer(&b), offset(offset_bytes), bytes(size_bytes), row_stride(row_stride_bytes) {}

    [[nodiscard]] bool empty() const noexcept { return buffer == nullptr; }
};

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

/// Byte size of one row of `cols` elements of weight type `t` (F32, Q8_0, Q4_K, Q5_K,
/// Q6_K, IQ4_XS). Throws Error(Unsupported) for other types, Error(Kernel) if cols is not a
/// multiple of the block size.
[[nodiscard]] std::uint64_t matvec_row_bytes(DType t, std::uint32_t cols);

/// Inputs of a recurrent Gated DeltaNet decode over T = n_tokens rows (DECISIONS D-004
/// item 6). T = 1 is the single-token decode step; T > 1 is the MTP-verify form, which can
/// also record per-row state slots for rollback (DECISIONS D-012).
///
/// q/k contract — DECISIONS **D-016**, identical to `halo::cpu::GdnQkParams` (same two
/// fields, same defaults, same order of operations):
///  - q, k are the RAW per-head rows (after conv + SiLU).
///  - if `qk_l2norm`: q <- q * rsqrt(sum(q^2) + 1e-6), k <- k * rsqrt(sum(k^2) + 1e-6)
///    per head (in-kernel reduction).
///  - q <- q * q_scale (after the optional normalization; k is never scaled). `q_scale` =
///    nullopt means 1/sqrt(d_k); it must be finite. The kernel applies exactly this and
///    nothing else. The qwen35 forward passes qk_l2norm = true, q_scale = 1/sqrt(128).
///
/// Other operands (token-major; a view's row_stride is the byte distance between tokens):
///  - q, k: T rows of [n_k * d_k]; v: T rows of [n_v * d_v]; g, beta: T rows of [n_v]
///    (g is the log-decay, the kernel applies exp(g); beta is already sigmoid-activated).
///    q, k, v may be views into one fused qkv buffer (e.g. offset 0 / 2048*4 / 4096*4 with
///    row_stride 10240*4).
///  - out: T rows of [n_v * d_v].
///  - Head mapping is GGUF *tiled*: value head j uses key head `j % n_k` (D-004 item 5).
///  - A state is [n_v, d_k, d_v] fp32, element (j, a, b) at `(j*d_k + a)*d_v + b` (a = key
///    dim, b = value dim, d_v fastest; HF layout — ggml stores the transpose), the same
///    layout as the CPU op.
/// Per row t and head j:  S *= exp(g);  kv = Sᵀk;  δ = (v − kv)·β;  S += k δᵀ;  o = Sᵀq.
///
/// State placement (views; row_stride is ignored, extent = one state):
///  - `state`: the input state region.
///  - `state_out`: the final-state region; nullopt = in place (same region as `state`).
///    Input and output regions in one buffer must be identical or disjoint.
///  - `state_slots` (optional, empty view = none): `n_slots` states back to back; slot s
///    receives the state after row T-1-s (slot 0 = most recent = final state) for
///    s < min(T, n_slots); slots s >= T are left untouched. Rolling back r rows = using
///    slot r as the state. Must not overlap the state regions, `out`, or any input.
struct GdnDecodeArgs {
    BufferView q, k, v, g, beta;
    BufferView state;
    std::optional<BufferView> state_out;
    BufferView state_slots;
    BufferView out;
    std::uint32_t n_v = 0;
    std::uint32_t n_k = 0;
    std::uint32_t d_k = 0;
    std::uint32_t d_v = 0;
    std::uint32_t n_tokens = 1;  ///< T
    std::uint32_t n_slots = 0;
    bool qk_l2norm = true;         ///< D-016
    std::optional<float> q_scale;  ///< D-016; nullopt = 1/sqrt(d_k)
};

/// Result layout written by argmax(): three 32-bit words
///   {index (uint32), value (float bits), nan (0 or 1: some logit was NaN)}.
/// Read it with read_argmax(), which raises Error(Kernel) on NaN like cpu::argmax (D-016).
inline constexpr std::uint32_t k_argmax_result_bytes = 12;
/// Index word when every logit was NaN (never returned by read_argmax, which throws).
inline constexpr std::uint32_t k_argmax_none = 0xFFFFFFFFu;

struct ArgmaxResult {
    std::uint32_t index = 0;
    float value = 0.0f;
};

/// Decodes the three result words. Throws Error(Kernel) when the NaN flag is set (a NaN
/// logit is a bug upstream and must not be sampled silently; cpu::argmax raises the same
/// error, DECISIONS D-016).
[[nodiscard]] ArgmaxResult decode_argmax(std::span<const std::uint32_t, 3> words);
/// Downloads k_argmax_result_bytes at `offset` of `result` (after the stream that wrote it
/// has been waited on) and decodes them.
[[nodiscard]] ArgmaxResult read_argmax(const Buffer& result, std::uint64_t offset = 0);

// ---------------------------------------------------------------- layer ops (WS-F2 V1)
// Argument structs mirror halo::hip (include/halo/backends/hip/ops.h) over vulkan::BufferView.
// Each op's semantics are those of the named halo::cpu op; "may alias exactly" means the
// output may name exactly the same elements (same buffer, first byte, stride and extent) as
// that input, and must be disjoint from every other operand.

/// [CONV1D_SHORT] (cpu::causal_conv1d_silu): causal depthwise conv1d (no bias) + SiLU.
///  - x: n_tokens rows of [channels]; weight: channels rows of [kernel] (tap kernel-1 = the
///    current input); kernel in 1..8.
///  - conv_state: kernel-1 rows of [channels], in/out (oldest first); empty when kernel = 1.
///  - out: n_tokens rows of [channels]; may alias x exactly.
///  - state_slots (D-012; empty = none): n_slots dense [kernel-1, channels] states; slot s =
///    conv state after row T-1-s for s < min(T, n_slots); slots s >= T are left untouched.
/// The accumulation order is the CPU op's (taps oldest first, then the current input), and
/// no multiply-add is contracted, so the pre-activation is bit-identical to halo::cpu; the
/// SiLU uses the device exp. conv_state and the slots are bit-identical copies of inputs.
struct Conv1dArgs {
    BufferView x{}, weight{}, conv_state{}, out{};
    BufferView state_slots{};
    std::uint32_t n_tokens = 0;
    std::uint32_t channels = 0;
    std::uint32_t kernel = 4;
    std::uint32_t n_slots = 0;
};

/// [GATED_NORM] (cpu::gated_rms_norm): out[r] = (w * (x[r] * rsqrt(mean(x[r]^2) + eps))) *
/// silu(z[r]), per row (a row = one value head). x, z, out: rows rows of [cols]; w: [cols].
/// out may alias x or z exactly.
struct GatedNormArgs {
    BufferView x{}, z{}, w{}, out{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    float eps = 1e-6f;
};

/// [PARTIAL_ROPE] (cpu::partial_rope_neox), in place on x: n_tokens rows, each n_heads heads
/// of head_dim elements, head h starting at h * head_stride (TD-9; 0 = head_dim, dense). Only
/// the first rot_dims of each head rotate (NeoX rotate-half); the rest (and anything between
/// heads, e.g. the gate halves of qwen35's interleaved [Q | gate] attn_q row, head_dim 256,
/// head_stride 512) are never touched.
///  - cos_sin: n_tokens rows of [rot_dims] fp32 = rope_cos_sin_table(positions, rot_dims,
///    theta): per token rot_dims/2 cosines then rot_dims/2 sines of the CPU op's fp32 angles,
///    evaluated in double and rounded (Vulkan has no double sin/cos, and its float sin/cos
///    is only accurate to 2^-11 absolute, so the table is built on the host). With the table
///    the kernel is the CPU op's two products and one add per element, uncontracted:
///    bit-identical to halo::cpu.
///  - x's row_stride should be explicit when head_stride > head_dim.
struct RopeArgs {
    BufferView x{};
    BufferView cos_sin{};
    std::uint32_t n_tokens = 0;
    std::uint32_t n_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rot_dims = 0;
    std::uint32_t head_stride = 0;  ///< TD-9; 0 = head_dim
};

/// The RopeArgs::cos_sin table for `positions` (exactly the CPU op's angles; see RopeArgs).
[[nodiscard]] std::vector<float> rope_cos_sin_table(std::span<const std::int32_t> positions, std::uint32_t rot_dims,
                                                    float theta);

/// [SWIGLU] out = silu(a) * b (a = gate, b = up); [MUL_SIGMOID] out = a * sigmoid(b) (a =
/// attention output, b = gate); [ADD] out = a + b. rows x cols; out may alias a or b exactly.
struct EltwiseArgs {
    BufferView a{}, b{}, out{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

/// [ADD + RMS_NORM fused] h = a + b, then y = rms_norm(h) * w — h bit-identical to cpu::add,
/// y equal to Ops::rms_norm(h) bit for bit (same reduction order). h may alias a or b
/// exactly (the in-place residual accumulate, ADR-001 §5.2); y must not overlap a, b, h, w.
struct AddRmsNormArgs {
    BufferView a{}, b{}, h{}, w{}, y{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    float eps = 1e-6f;
};

/// [GDN gates] qwen35's per-head gate inputs, fused, in the CPU forward's exact order
/// (src/models/qwen35.cpp): beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias), softplus
/// with torch's threshold 20 and an accurate log1p. b, a: rows x [n_v]; dt_bias, ssm_a:
/// [n_v]. beta may alias b exactly, g may alias a exactly.
struct GdnGateArgs {
    BufferView b{}, a{}, dt_bias{}, ssm_a{}, beta{}, g{};
    std::uint32_t rows = 0;
    std::uint32_t n_v = 0;
};

class Ops {
public:
    explicit Ops(std::shared_ptr<Context> ctx, OpsOptions options = {});

    [[nodiscard]] const OpsOptions& options() const noexcept { return options_; }

    /// [RMS_NORM] y[r,i] = (x[r,i] * rsqrt(mean_i(x[r,·]²) + eps)) * w[i], like cpu::rms_norm.
    /// x, y: `rows` rows of `cols` fp32 (row strides from the views); w: [cols]. Plain x̂·w
    /// (GGUF stores 1+w; D-004). y must not overlap x or w (no in-place, unlike the CPU op).
    void rms_norm(Stream& stream, const BufferView& x, const BufferView& w, const BufferView& y,
                  std::uint32_t rows, std::uint32_t cols, float eps);

    /// [QUANT_GEMV / MATMUL] y[r] = Σ_c W[r,c]·x[c]. W is `rows` rows of `cols` elements in
    /// ggml layout for `wtype` (F32 row-major; quantized: consecutive ggml blocks per row);
    /// the W view's row_stride (0 = matvec_row_bytes) separates rows. x: [cols] fp32; y:
    /// [rows] fp32. rows may exceed the device workgroup-count limit (a 2-D grid is used).
    /// Quantized blocks are read as 32-bit words, so the W buffer must extend to the end of
    /// the word holding the last byte read (checked).
    void matvec(Stream& stream, DType wtype, const BufferView& w, const BufferView& x, const BufferView& y,
                std::uint32_t rows, std::uint32_t cols);

    /// [GATED_DELTANET] recurrent decode over args.n_tokens rows; see GdnDecodeArgs.
    void gated_delta_rule_decode(Stream& stream, const GdnDecodeArgs& args);

    /// [ARGMAX_FUSED building block] index of the maximum of `logits[0..n)`. Two passes:
    /// per-workgroup partials into `scratch` (argmax_scratch_bytes(n) bytes), then one
    /// workgroup reduces them into `result` (k_argmax_result_bytes, see read_argmax). Ties
    /// resolve to the lowest index; -inf is an ordinary value; any NaN sets the result's
    /// NaN flag (read_argmax then throws). Every dispatch rewrites the flag, so a result
    /// buffer can be reused.
    void argmax(Stream& stream, const BufferView& logits, std::uint32_t n, const BufferView& scratch,
                const BufferView& result);
    [[nodiscard]] std::uint64_t argmax_scratch_bytes(std::uint32_t n) const;
    [[nodiscard]] std::uint32_t argmax_partials(std::uint32_t n) const;

    /// [CONV1D_SHORT] see Conv1dArgs.
    void causal_conv1d_silu(Stream& stream, const Conv1dArgs& args);
    /// [GATED_NORM] see GatedNormArgs.
    void gated_rms_norm(Stream& stream, const GatedNormArgs& args);
    /// [PARTIAL_ROPE] in place; see RopeArgs.
    void partial_rope_neox(Stream& stream, const RopeArgs& args);
    /// [SWIGLU] / [MUL_SIGMOID] / [ADD]; see EltwiseArgs.
    void swiglu(Stream& stream, const EltwiseArgs& args);
    void mul_sigmoid(Stream& stream, const EltwiseArgs& args);
    void add(Stream& stream, const EltwiseArgs& args);
    /// [ADD + RMS_NORM] see AddRmsNormArgs.
    void add_rms_norm(Stream& stream, const AddRmsNormArgs& args);
    /// [GDN gates] see GdnGateArgs.
    void gdn_gates(Stream& stream, const GdnGateArgs& args);

private:
    void eltwise(Stream& stream, const EltwiseArgs& args, std::uint32_t op_code, std::string_view name);
    const Kernel& kernel(const std::string& shader, std::uint32_t num_buffers,
                         std::uint32_t push_bytes, std::vector<SpecConstant> spec,
                         std::array<std::uint32_t, 3> local);

    std::shared_ptr<Context> ctx_;
    OpsOptions options_;
    std::map<std::string, std::unique_ptr<Kernel>> kernels_;
};

}  // namespace halo::vulkan
