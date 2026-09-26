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
    /// Workgroup size of attention (power of two, 32..1024): keys per tile, and each thread
    /// accumulates head_dim / attention_workgroup (<= 4) output elements.
    std::uint32_t attention_workgroup = 128;
    /// Largest weight / table range one gemv or get_rows dispatch binds; larger W (e.g. a
    /// 248320-row embedding or LM head) is processed in row slabs that each fit. 0 = the
    /// device's maxStorageBufferRange. (A smaller value only splits more; for tests.)
    std::uint64_t max_binding_bytes = 0;
};

/// Byte size of one row of `cols` elements of weight type `t` (F32, Q8_0, Q4_K, Q5_K,
/// Q6_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S: every type of the UD-Q4_K_XL pack, D-007 / D-014).
/// Throws Error(Unsupported) for other types, Error(Kernel) if cols is not a multiple of the
/// block size.
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

// ---------------------------------------------------------------- paged ops (WS-F2 V2)
// Status words: GET_ROWS, KV write and ATTENTION detect bad data (an id or block-table entry
// out of range) on the device. Each call takes its own 4-byte `status` view, which the op
// zeroes (a recorded fill) before its dispatch and the kernel ORs bits into. Two calls must
// not share a word (the second call's fill would erase the first one's bits). After the
// stream was waited on, read it with read_status() and pass it to check_status(). Bit values
// are halo::hip's (ADR-001 §5.5).
inline constexpr std::uint32_t k_status_bad_block = 4u;  ///< = hip::kStatusBadBlock
inline constexpr std::uint32_t k_status_bad_index = 8u;  ///< = hip::kStatusBadIndex
inline constexpr std::uint64_t k_status_bytes = 4;

/// Downloads the status word at `offset` of `status` (after the writing stream was waited on).
[[nodiscard]] std::uint32_t read_status(const Buffer& status, std::uint64_t offset = 0);
/// Throws Error(Kernel) naming `op` and the set bits when `word` is non-zero.
void check_status(std::uint32_t word, std::string_view op);

/// [GET_ROWS] embedding lookup: out[t] = tensor::dequantize_row(W[ids[t]]) for every GEMV
/// weight type (every type of matvec_row_bytes; the pack's token_embd is Q4_K).
/// Bit-identical to tensor::dequantize_row: the kernels evaluate its expressions uncontracted
/// (verified on lavapipe; fp16 scales are converted by unpackHalf2x16, which may flush fp16
/// denormals on some devices).
///  - w: n_rows rows of matvec_row_bytes(wtype, cols) (row_stride 0 = dense).
///  - ids: n_ids int32. out: n_ids rows of [cols] fp32; must not overlap w, ids or status.
///  - status: an id outside [0, n_rows) sets k_status_bad_index, and that row of out is not
///    written (W is never read through it).
struct GetRowsArgs {
    DType wtype = DType::F32;
    BufferView w{};
    std::uint32_t n_rows = 0;
    std::uint32_t cols = 0;
    BufferView ids{};
    std::uint32_t n_ids = 1;
    BufferView out{};
    BufferView status{};
};

/// The paged KV pool (a halo::kv_cache::KvPool's storage, byte for byte): n_pool_blocks
/// blocks of n_layers * 2 * block_tokens * kv_dim fp32, block[layer][K|V][token][kv_dim]
/// (kv_dim = n_kv_head * head_dim). History row s of a sequence lives in block
/// block_table[s / block_tokens] (uint32 ids), token s % block_tokens.
///
/// Limit of this backend: the whole pool is bound as one storage-buffer descriptor, so it
/// must fit maxStorageBufferRange and 2^32 fp32 elements; a larger pool is rejected with
/// Error(Unsupported). (At fp32, the 27B model's 16 attention layers take 128 KiB per token,
/// so a 4 GiB pool holds about 32k tokens in total.)

/// [KV write] Writes n_tokens K and V rows at history rows start .. start+n_tokens-1 of
/// `layer` through the block table — exactly kv_cache::SequenceKv::write. Every block the
/// rows fall into must be owned by this sequence alone (copy-on-write is kv_cache's job).
///  - k, v: n_tokens rows of [kv_dim]; must not overlap kv_pool.
///  - block_table: must have an entry for every block the rows fall into.
///  - status: a table entry >= n_pool_blocks sets k_status_bad_block; that row is not written.
struct KvWriteArgs {
    BufferView kv_pool{};
    std::uint32_t n_pool_blocks = 0;
    std::uint32_t n_layers = 1;
    std::uint32_t layer = 0;
    std::uint32_t block_tokens = 16;
    std::uint32_t kv_dim = 0;
    BufferView block_table{};
    std::uint32_t start = 0;
    std::uint32_t n_tokens = 1;
    BufferView k{}, v{};
    BufferView status{};
};

/// [ATTENTION] causal GQA softmax attention over the paged K/V history (cpu::attention_gqa).
///  - q: n_tokens rows of n_head heads of head_dim; q_head_stride = elements between heads
///    (0 = head_dim; 512 reads Q in place from qwen35's interleaved [Q | gate] attn_q row;
///    give q's row_stride explicitly then).
///  - kv_pool / layer / block_tokens / block_table: see "The paged KV pool" above; the table
///    must cover q_offset + n_tokens rows (the last block may be partially filled).
///  - Query t attends history rows 0 ..= q_offset + t; query head h uses KV head
///    h / (n_head / n_kv_head); scores = (q . k) * scale, with the CPU's 8-lane dot order.
///  - out: n_tokens rows of [n_head * head_dim]; must not overlap any operand.
///  - status: a table entry the call needs that is >= n_pool_blocks sets k_status_bad_block;
///    those rows are skipped (never read) and out is then undefined.
/// Online softmax (running max, one pass over key tiles of attention_workgroup keys), so the
/// result differs from the CPU's exact three-pass order within the bound in test_vk_kv.cpp.
/// Limits: head_dim <= 256 and head_dim <= 4 * attention_workgroup (Error(Unsupported)).
/// Unlike hip::AttentionArgs there is no workspace (the online form needs none).
struct AttentionArgs {
    BufferView q{};
    std::uint32_t q_head_stride = 0;
    BufferView kv_pool{};
    std::uint32_t n_pool_blocks = 0;
    std::uint32_t n_layers = 1;
    std::uint32_t layer = 0;
    std::uint32_t block_tokens = 16;
    BufferView block_table{};
    std::uint32_t n_head = 0;
    std::uint32_t n_kv_head = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t n_tokens = 1;
    std::uint32_t q_offset = 0;
    float scale = 1.0f;
    BufferView out{};
    BufferView status{};
};

// ---------------------------------------------------------------- chunked GATED_DELTANET (WS-F2 V4)

/// Status bit: the chunked GDN saw a g that is not <= 0 (NaN included) and wrote nothing
/// (= hip::kStatusPositiveG; check_status raises Error(Kernel) exactly where
/// cpu::gated_delta_rule_chunked throws).
inline constexpr std::uint32_t k_status_positive_g = 1u;

/// GATED_DELTANET, chunked form (cpu::gated_delta_rule_chunked; transformers
/// torch_chunk_gated_delta_rule), for prefill: the same function and operands as
/// gated_delta_rule_decode (GdnDecodeArgs: D-016 q/k contract, tiled heads, state layout,
/// state_out / in place, rollback slots evaluated with the chunk's closed form exactly as the
/// CPU op), evaluated chunk_size tokens at a time. Every arithmetic step follows the CPU
/// op's order; exp, the L2-norm reduction order and contraction differ, so the result is
/// bounded against the CPU op, not bitwise (test_vk_chunked.cpp).
///  - chunk_size: 1..64 (qwen35 prefill uses 64).
///  - workspace: >= gdn_chunked_workspace_bytes(gdn, chunk_size) (per-head scratch).
///  - status: zeroed by the op; a g that is not <= 0 sets k_status_positive_g and then out,
///    state and slots are left untouched (the op writes nothing).
/// Parallel over value heads only (one workgroup per head, chunks in order): correct, but
/// not a tuned GPU prefill.
struct GdnChunkedArgs {
    GdnDecodeArgs gdn{};
    std::uint32_t chunk_size = 64;
    BufferView workspace{};
    BufferView status{};
};

[[nodiscard]] std::uint64_t gdn_chunked_workspace_bytes(const GdnDecodeArgs& gdn, std::uint32_t chunk_size);

// ---------------------------------------------------------------- GEMV, LM head, TOP_K (WS-F2 V3)

/// Status bit: TOP_K saw a NaN logit (= hip::kStatusNaN; check_status raises Error(Kernel),
/// as cpu::top_k does).
inline constexpr std::uint32_t k_status_nan = 2u;

/// [QUANT_GEMV / MATMUL] (cpu::matmul with a tensor::dequantize_row WeightMatrix):
///   y[t][n] = sum_i x[t][i] * W[n][i],  t < n_vec, n < rows, i < cols.
///  - wtype: every type of matvec_row_bytes. W: rows rows of matvec_row_bytes(wtype, cols)
///    (row_stride 0 = dense). Quantized rows may start at any byte; F32 rows need 4-byte
///    alignment.
///  - x: n_vec rows of [cols] fp32; y: n_vec rows of [rows] fp32; y must not overlap x or W.
/// n_vec > 1 is the batched form (prefill chunks, MTP verify): each workgroup dequantizes its
/// W row once and applies it to up to 8 vectors per dispatch (larger n_vec is split into
/// dispatches of 8). The per-vector arithmetic is exactly the n_vec = 1 matvec's, so a
/// vector's result does not depend on the batch it is in (tested bitwise).
struct GemvArgs {
    DType wtype = DType::F32;
    BufferView w{}, x{}, y{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t n_vec = 1;
};

/// [LOGITS_MATMUL + ARGMAX] (cpu::matmul then cpu::argmax per vector; = cpu::matmul_argmax):
/// argmax_n of y[t][n], ties to the lowest n, NaN -> the vector's NaN word is set (read_argmax
/// then throws, D-016). Composed on the device from gemv and the two-pass argmax (no host
/// round trip; not a fused kernel, no performance claim).
///  - gemv.y: the logits (kept on the device); may be empty, then they go to the workspace.
///  - workspace: >= Ops::lm_head_workspace_bytes(rows, n_vec, gemv.y empty).
///  - result: n_vec * k_argmax_result_bytes; vector t at byte t * k_argmax_result_bytes
///    (read_argmax(result_buffer, result.offset + t * 12)).
///  - valid_rows: rows at/after this index are excluded from the argmax (GGUF LM-head padding
///    past the real vocabulary). 0 = no clamp (argmax over all gemv.rows). The logits are
///    unaffected; a NaN in any row -- padded or not -- still sets the vector's NaN word.
struct LmHeadArgs {
    GemvArgs gemv{};
    BufferView workspace{};
    BufferView result{};
    std::uint32_t valid_rows = 0;
};

/// [TOP_K] (cpu::top_k, per vector): the k largest logits, value descending, ties by lower
/// index first; 1 <= k <= min(n, 1024). Exact (comparisons only): ids and values are
/// bit-identical to the CPU.
///  - logits: n_vec rows of [n] (row_stride = distance between vectors).
///  - ids: n_vec rows of [k] int32; values: n_vec rows of [k] fp32.
///  - workspace: >= topk_workspace_bytes(n, k, n_vec) (0 when n <= 2048).
///  - status: a NaN logit sets k_status_nan (outputs then undefined), as cpu::top_k throws.
/// Chunked bitonic selection: chunks of 2048 are sorted in shared memory (16 KiB) and reduced
/// to their best k until one chunk per vector remains (ceil(log_{2048/k}(n/k)) + 1 passes).
struct TopKArgs {
    BufferView logits{};
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t n_vec = 1;
    BufferView workspace{};
    BufferView ids{};
    BufferView values{};
    BufferView status{};
};

[[nodiscard]] std::uint64_t topk_workspace_bytes(std::uint32_t n, std::uint32_t k, std::uint32_t n_vec);

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
    /// buffer can be reused. `valid` > 0 excludes indices >= valid from the ranking (the LM
    /// head's valid_rows clamp); the NaN scan still covers all n elements.
    void argmax(Stream& stream, const BufferView& logits, std::uint32_t n, const BufferView& scratch,
                const BufferView& result, std::uint32_t valid = 0);
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

    /// [GET_ROWS] see GetRowsArgs; then check_status(read_status(...)).
    void get_rows(Stream& stream, const GetRowsArgs& args);
    /// [KV write] see KvWriteArgs; then check_status(read_status(...)).
    void kv_write(Stream& stream, const KvWriteArgs& args);
    /// [ATTENTION] see AttentionArgs; then check_status(read_status(...)).
    void attention(Stream& stream, const AttentionArgs& args);

    /// [GATED_DELTANET] chunked form; see GdnChunkedArgs; then check_status(read_status(...)).
    void gated_delta_rule_chunked(Stream& stream, const GdnChunkedArgs& args);

    /// [QUANT_GEMV / MATMUL] batched; see GemvArgs. matvec() is gemv with n_vec = 1.
    void gemv(Stream& stream, const GemvArgs& args);
    /// [LOGITS_MATMUL + ARGMAX] see LmHeadArgs.
    void lm_head(Stream& stream, const LmHeadArgs& args);
    [[nodiscard]] std::uint64_t lm_head_workspace_bytes(std::uint32_t rows, std::uint32_t n_vec,
                                                        bool logits_in_workspace) const;
    /// [TOP_K] see TopKArgs; then check_status(read_status(...)).
    void top_k(Stream& stream, const TopKArgs& args);

private:
    void eltwise(Stream& stream, const EltwiseArgs& args, std::uint32_t op_code, std::string_view name);
    void gemv_impl(Stream& stream, const GemvArgs& args, std::string_view op);
    void gdn_impl(Stream& stream, const GdnDecodeArgs& args, const GdnChunkedArgs* chunked);
    const Kernel& kernel(const std::string& shader, std::uint32_t num_buffers,
                         std::uint32_t push_bytes, std::vector<SpecConstant> spec,
                         std::array<std::uint32_t, 3> local);

    std::shared_ptr<Context> ctx_;
    OpsOptions options_;
    std::map<std::string, std::unique_ptr<Kernel>> kernels_;
};

}  // namespace halo::vulkan
