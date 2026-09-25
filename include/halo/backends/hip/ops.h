#pragma once
/// \file
/// HALO HIP operators (TRD §9 ids in brackets). Each method validates shapes, views,
/// aliasing and buffer ranges on the host (overflow-checked, before anything runs), then
/// either enqueues the kernel(s) on a device Stream or runs the identical kernel bodies in
/// the host emulation (Target::emulation), which is how the dev host tests them.
///
/// Semantics are those of the CPU reference operators in `halo/backends/cpu/ops.h`; these
/// kernels are accepted only by differential tests against `halo::cpu` on the same inputs
/// (TRD §30; DECISIONS D-016; code review M-2). The host emulation is bit-identical to
/// halo::cpu (tested); on the device only the transcendental functions (expf) differ.
///
/// Operands are BufferViews (the Vulkan backend's concept, code review S-1): a byte range
/// of a Buffer plus an optional row stride, so q/k/v can be views into one fused qkv row.
/// fp32 operands need 4-byte-aligned offsets and strides. Aliasing is checked on address
/// ranges: an output may not overlap any other operand except where a method says
/// "may alias", and then only exactly (same address and stride).
///
/// Memory: a device Target requires Device/HostPinned/Managed buffers; the emulation
/// Target requires MemoryTier::Host buffers (Buffer::wrap_host). Mixing is rejected.
///
/// Status (DECISIONS D-001): device execution is unverified (no GPU on the dev host).
/// Thread safety: an Ops is immutable after construction and may be shared; a Stream may
/// not be used from two threads at once.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "halo/backends/hip/runtime.h"
#include "halo/tensor/dtype.h"

namespace halo::hip {

/// A byte range of a Buffer used as an operand.
///  - offset: bytes from the start of the buffer.
///  - bytes: size of the range the op may touch; 0 = to the end of the buffer.
///  - row_stride: bytes between consecutive rows of a 2-D operand; 0 = dense.
struct BufferView {
    const Buffer* buffer = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t row_stride = 0;

    BufferView() = default;
    BufferView(const Buffer& b) noexcept : buffer(&b) {}  // NOLINT(google-explicit-constructor)
    BufferView(const Buffer& b, std::uint64_t offset_bytes, std::uint64_t size_bytes = 0,
               std::uint64_t row_stride_bytes = 0) noexcept
        : buffer(&b), offset(offset_bytes), bytes(size_bytes), row_stride(row_stride_bytes) {}

    [[nodiscard]] bool empty() const noexcept { return buffer == nullptr; }
};

/// Where an op runs: enqueued on a device stream, or executed synchronously by the host
/// emulation of the kernels (forward or reverse thread/workgroup order).
class Target {
public:
    [[nodiscard]] static Target device(const Stream& s) noexcept { return Target(&s, false); }
    [[nodiscard]] static Target emulation(bool reverse_order = false) noexcept { return Target(nullptr, reverse_order); }

    [[nodiscard]] bool is_device() const noexcept { return stream_ != nullptr; }
    [[nodiscard]] const Stream* stream() const noexcept { return stream_; }
    [[nodiscard]] bool reverse_order() const noexcept { return reverse_; }

private:
    Target(const Stream* s, bool reverse) noexcept : stream_(s), reverse_(reverse) {}
    const Stream* stream_;
    bool reverse_;
};

/// How value head j selects its key/query head (same meaning as cpu::GdnHeadMapping).
enum class GdnHeadMapping {
    Tiled,    ///< GGUF order: k head = j % n_k (D-004 item 5) — the qwen35 model's order
    Grouped,  ///< HF order: k head = j / (n_v / n_k)
};

/// Operands of GATED_DELTANET (both forms). Same contract as cpu::gated_delta_rule_*:
///  - q/k — DECISIONS **D-016**: RAW per-head rows (after conv + SiLU). If `qk_l2norm`:
///    q <- q * rsqrt(sum(q^2) + 1e-6), k likewise, per head, in-kernel; then
///    q <- q * q_scale (nullopt = 1/sqrt(d_k); must be finite). Nothing else is applied.
///  - q, k: T rows of [n_k * d_k]; v: T rows of [n_v * d_v]; g, beta: T rows of [n_v]
///    (g = log-decay, the kernel applies exp(g); beta already sigmoid-ed). Token-major; the
///    view's row_stride is the distance between tokens.
///  - state: [n_v, d_k, d_v] fp32, d_v fastest (the CPU layout). `state_out` = nullopt
///    means in place; otherwise it must be disjoint from `state` (identical = in place).
///  - out: T rows of [n_v * d_v]; must not overlap any other operand.
///  - state_slots (D-012; empty = none): n_slots states back to back; slot s receives the
///    state after row T-1-s for s < min(T, n_slots); slots s >= T are left untouched.
/// Limits of this backend: d_k <= 128 (register-resident state column), n_tokens >= 1.
struct GdnArgs {
    BufferView q{}, k{}, v{}, g{}, beta{};
    BufferView state{};
    std::optional<BufferView> state_out{};
    BufferView state_slots{};
    BufferView out{};
    std::uint32_t n_k = 0;
    std::uint32_t n_v = 0;
    std::uint32_t d_k = 0;
    std::uint32_t d_v = 0;
    std::uint32_t n_tokens = 1;
    std::uint32_t n_slots = 0;
    GdnHeadMapping mapping = GdnHeadMapping::Tiled;
    bool qk_l2norm = true;         ///< D-016
    std::optional<float> q_scale{};  ///< D-016; nullopt = 1/sqrt(d_k)
};

/// Extra operands of the chunked form.
///  - chunk_size: 1..64 (the CPU op accepts up to 1024; this backend's LDS tiles cap it at
///    64, which is what qwen35 prefill uses).
///  - workspace: at least gdn_chunked_workspace_bytes(args, 1) bytes (fp32 aligned). The op
///    processes as many chunks per launch group as fit; more workspace = fewer groups.
///  - status: one 32-bit word the op zeroes, then sets to kStatusPositiveG when some g is
///    not <= 0 (NaN included) — the kernels then write nothing (out, state and slots keep
///    their contents). Call Ops::check_status after the work completed: it raises
///    Error(Kernel) exactly where cpu::gated_delta_rule_chunked would.
struct GdnChunkedArgs {
    GdnArgs gdn{};
    std::uint32_t chunk_size = 64;
    BufferView workspace{};
    BufferView status{};
};

/// Workspace bytes for `chunks_per_group` chunks (clamped to the chunk count) of every head.
[[nodiscard]] std::uint64_t gdn_chunked_workspace_bytes(const GdnArgs& args, std::uint32_t chunk_size,
                                                        std::uint32_t chunks_per_group);

/// Status bits written by ops that detect data errors on the device.
inline constexpr std::uint32_t kStatusPositiveG = 1u;
inline constexpr std::uint32_t kStatusNaN = 2u;
inline constexpr std::uint32_t kStatusBadBlock = 4u;

/// CONV1D_SHORT (cpu::causal_conv1d_silu): causal depthwise conv1d (no bias) + SiLU.
///  - x: T rows of [C]; weight: C rows of [K] (tap K-1 = current input); K in 1..8.
///  - conv_state: K-1 rows of [C], in/out (oldest first).
///  - out: T rows of [C]; may alias x exactly.
///  - state_slots (optional): n_slots * (K-1) * C floats, slot s = conv state after row
///    T-1-s for s < min(T, n_slots) (dense [K-1, C] per slot).
struct Conv1dArgs {
    BufferView x{}, weight{}, conv_state{}, out{};
    BufferView state_slots{};
    std::uint32_t n_tokens = 0;
    std::uint32_t channels = 0;
    std::uint32_t kernel = 4;
    std::uint32_t n_slots = 0;
};

/// GATED_NORM (cpu::gated_rms_norm): out[r] = (w * (x̂[r])) * silu(z[r]), x̂ = x * rsqrt(
/// mean(x^2) + eps), per row (a row = one value head, cols = d_v). x, z, out: `rows` rows
/// of [cols]; w: [cols]. out may alias x or z exactly.
struct GatedNormArgs {
    BufferView x{}, z{}, w{}, out{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    float eps = 1e-6f;
};

/// QUANT_GEMV / MATMUL for decode (cpu::matmul with a dequantized WeightMatrix):
///   y[t][n] = sum_i x[t][i] * W[n][i],  t < n_vec, n < rows, i < cols.
///  - wtype: F32, F16, Q8_0, Q4_K, Q5_K, Q6_K and the D-014 second tier IQ4_XS, IQ4_NL, Q3_K,
///    IQ3_S (every type of the UD-Q4_K_XL pack), and Q4_0 (the ggml-org MTP pack, D-006)
///    (ggml block layouts; DECISIONS D-007, D-014).
///    Other types raise Error(Unsupported). cols must be a multiple of the block size.
///  - w: `rows` rows of row_bytes(wtype, cols) bytes; the view's row_stride (0 = dense) is
///    the byte distance between rows. Quantized/F16 rows may start at any byte; F32 rows
///    need 4-byte alignment.
///  - x: n_vec rows of [cols] fp32; y: n_vec rows of [rows] fp32; y must not overlap x or w.
/// Dequantized weights are bit-identical to halo::tensor::dequantize_row. The generic
/// variant also reproduces cpu::detail::dot's summation order (bit-identical to the CPU
/// matmul); the wave variants use a different, fixed order (see docs/hip.md for the bound).
struct GemvArgs {
    DType wtype = DType::F32;
    BufferView w{}, x{}, y{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t n_vec = 1;
};

/// ARGMAX result words of one vector: {index (uint32), value (float bits), nan (0/1)} — the
/// Vulkan backend's layout. Read them with Ops::read_argmax, which raises Error(Kernel) when
/// the NaN word is set (D-016; cpu::argmax raises the same error).
inline constexpr std::uint32_t k_argmax_result_bytes = 12;

struct ArgmaxResult {
    std::uint32_t index = 0;
    float value = 0.0f;
};

/// Decodes 3 words per vector; throws Error(Kernel) naming the vector if its NaN word is set.
[[nodiscard]] std::vector<ArgmaxResult> decode_argmax(std::span<const std::uint32_t> words);

/// [LOGITS_MATMUL + ARGMAX_FUSED] (cpu::matmul_argmax, per vector): argmax_n of
/// y[t][n] = Σ_i x[t][i]·W[n][i], ties to the lowest n, NaN -> error. The GEMV variant
/// (OpsOptions::gemv) computes the logits and each workgroup ranks its own rows (epilogue);
/// a second kernel reduces the partials. The logits are also written when gemv.y is set,
/// and stay on the device either way (TRD §19).
///  - workspace: >= Ops::lm_head_workspace_bytes(rows, n_vec).
///  - result: n_vec * k_argmax_result_bytes.
struct LmHeadArgs {
    GemvArgs gemv{};  ///< gemv.y may be empty (argmax only)
    BufferView workspace{};
    BufferView result{};
};

/// [ARGMAX_FUSED building block] argmax of each of n_vec logits vectors of length n (cpu::argmax).
struct ArgmaxArgs {
    BufferView logits{};  ///< n_vec rows of [n]; row_stride = distance between vectors
    std::uint32_t n = 0;
    std::uint32_t n_vec = 1;
    BufferView workspace{};  ///< >= Ops::argmax_workspace_bytes(n, n_vec)
    BufferView result{};     ///< n_vec * k_argmax_result_bytes
};

/// [TOP_K] (cpu::top_k, per vector; TRD §19 GPU pre-filter): the k largest logits, sorted by
/// value descending, ties by lower index first; 1 <= k <= min(n, 1024).
///  - ids: n_vec rows of [k] int32; values: n_vec rows of [k] fp32 (dense).
///  - workspace: >= topk_workspace_bytes(n, k, n_vec) (0 when n <= 2048).
///  - status: one word, zeroed by the op, kStatusNaN set when some logit is NaN; call
///    Ops::check_status afterwards (raises Error(Kernel), as cpu::top_k).
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

/// [RMS_NORM] (cpu::rms_norm): out[r] = (x[r] * rsqrt(mean(x[r]^2) + eps)) * w, per row.
/// Per-head norms are rows = T * heads with cols = head_dim. out may alias x exactly.
struct RmsNormArgs {
    BufferView x{}, w{}, out{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    float eps = 1e-6f;
};

/// [PARTIAL_ROPE] (cpu::partial_rope_neox), in place on x: n_tokens rows of
/// [n_heads * head_dim]; positions: n_tokens int32. Only the first rot_dims of each head
/// rotate (NeoX rotate-half). inv_freq is computed on the host exactly as the CPU op does;
/// cos/sin of the fp32 angle are evaluated in double. rot_dims even, <= head_dim, <= 128.
/// x's row_stride should be given explicitly when head_stride > head_dim (the dense default
/// would be the extent of the last head, not the full interleaved row).
/// head_stride (TD-9): elements between the starts of consecutive heads within a row; 0 means
/// head_dim (dense, the CPU op's layout). qwen35's attn_q output interleaves [Q | gate] per
/// head (D-004), so RoPE on Q in place is head_dim = 256, head_stride = 512; the gate halves are
/// never touched. Must be >= head_dim.
struct RopeArgs {
    BufferView x{};
    BufferView positions{};
    std::uint32_t n_tokens = 0;
    std::uint32_t n_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rot_dims = 0;
    float theta = 10000.0f;
    std::uint32_t head_stride = 0;  ///< TD-9; 0 = head_dim
};

/// [SWIGLU] out = silu(a) * b (a = gate, b = up); [MUL_SIGMOID] out = a * sigmoid(b)
/// (a = attention output, b = gate). rows x cols; out may alias a or b exactly.
struct EltwiseArgs {
    BufferView a{}, b{}, out{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

/// [ATTENTION] causal GQA softmax attention over a paged K/V history (cpu::attention_gqa).
///  - q: n_tokens rows of n_head heads of head_dim; q_head_stride = elements between heads
///    (0 = head_dim; 512 reads Q in place from qwen35's interleaved [Q | gate] attn_q row).
///  - kv_pool: a halo::kv_cache::KvPool's storage, byte for byte: n_pool_blocks blocks of
///    n_layers * 2 * block_tokens * kv_dim floats, block[layer][K|V][token][kv_dim], with
///    kv_dim = n_kv_head * head_dim. `layer` selects the attention layer.
///  - block_table: uint32 block ids; history row s is in block table[s / block_tokens]. It
///    must cover q_offset + n_tokens rows (the last block may be partially filled).
///  - Query t attends history rows 0 ..= q_offset + t; query head h uses KV head
///    h / (n_head / n_kv_head); scores = (q . k) * scale.
///  - out: n_tokens rows of [n_head * head_dim]; must not overlap any operand.
///  - status: one word, zeroed by the op; kStatusBadBlock when a table entry the call reads
///    is >= n_pool_blocks (those rows are skipped, never read); check_status raises
///    Error(Kernel). The contents of `out` are then undefined. The table lives on the device, so it is validated there.
///  - workspace: attention_workspace_bytes() (the exact variant's score rows; 0 for online).
/// Limits of this backend: head_dim <= 256.
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
    BufferView workspace{};
    BufferView status{};
};

/// Status bit: GET_ROWS saw a token id outside [0, n_rows).
inline constexpr std::uint32_t kStatusBadIndex = 8u;

/// [KV write] Appends n_tokens K and V rows at history positions start .. start+n_tokens-1 of
/// `layer`, through the block table — exactly kv_cache::SequenceKv::write (same pool layout
/// as AttentionArgs). Every block the rows fall into must be exclusively owned by this
/// sequence (refcount 1): copy-on-write of shared prefix blocks is kv_cache's job
/// (SequenceKv::reserve), not this kernel's. k, v: n_tokens rows of [kv_dim]. status: zeroed
/// by the op; kStatusBadBlock for a table id >= n_pool_blocks (that row is not written).
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

/// [GET_ROWS] Embedding lookup: out[t] = dequantize_row(W[ids[t]]) for any GEMV weight type
/// (the pack's token_embd is Q4_K). Bit-identical to tensor::dequantize_row. w: n_rows rows of
/// row_bytes(wtype, cols) (row_stride 0 = dense); ids: n_ids int32; out: n_ids rows of
/// [cols]. status: zeroed by the op; kStatusBadIndex for an id outside [0, n_rows) (that row
/// of out is not written).
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

/// [ADD + RMS_NORM fused] h = a + b, then y = rms_norm(h) * w — bit-identical to cpu::add
/// followed by cpu::rms_norm. h may alias a or b exactly (in-place residual); y must not
/// overlap a, b, h or w.
struct AddRmsNormArgs {
    BufferView a{}, b{}, h{}, w{}, y{};
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    float eps = 1e-6f;
};

/// Variant choice per operator (names from the kernel registry, registry.h).
struct OpsOptions {
    std::string gdn_recurrent = "gdn_recurrent_b128";
    std::string gdn_chunked = "gdn_chunked_b64";
    std::string conv1d = "conv1d_silu_b256";
    std::string gated_norm = "gated_norm_b128";
    std::string gemv = "gemv_wave32_r4";
    std::string argmax = "argmax_b256";
    std::string topk = "topk_bitonic_b256";
    std::string rms_norm = "rms_norm_b128";
    std::string rope = "rope_neox_b128";
    std::string swiglu = "swiglu_b256";
    std::string mul_sigmoid = "mul_sigmoid_b256";
    std::string attention = "attn_online_b128";
    std::string kv_write = "kv_write_b256";
    std::string get_rows = "get_rows_b256";
    std::string add = "add_b256";
    std::string add_rms_norm = "add_rms_norm_b128";
};

class Ops {
public:
    /// Throws Error(Config) when a variant name is not registered for its operator.
    explicit Ops(OpsOptions options = {});

    [[nodiscard]] const OpsOptions& options() const noexcept { return options_; }

    /// [GATED_DELTANET] recurrent form (decode, T = 1; MTP verify, T > 1 with slots).
    void gated_delta_rule_recurrent(const Target& target, const GdnArgs& args) const;
    /// [GATED_DELTANET] chunked form (prefill); numerically the CPU chunked form.
    void gated_delta_rule_chunked(const Target& target, const GdnChunkedArgs& args) const;
    /// [CONV1D_SHORT]
    void causal_conv1d_silu(const Target& target, const Conv1dArgs& args) const;
    /// [GATED_NORM]
    void gated_rms_norm(const Target& target, const GatedNormArgs& args) const;
    /// [QUANT_GEMV / MATMUL]
    void gemv(const Target& target, const GemvArgs& args) const;
    /// [LOGITS_MATMUL + ARGMAX_FUSED]; read the result with read_argmax.
    void lm_head_argmax(const Target& target, const LmHeadArgs& args) const;
    [[nodiscard]] std::uint64_t lm_head_workspace_bytes(std::uint32_t rows, std::uint32_t n_vec) const;
    /// [ARGMAX_FUSED building block] over existing logits; read with read_argmax.
    void argmax(const Target& target, const ArgmaxArgs& args) const;
    [[nodiscard]] std::uint64_t argmax_workspace_bytes(std::uint32_t n, std::uint32_t n_vec) const;
    /// Reads n_vec argmax results (device: synchronizes the target stream first); throws
    /// Error(Kernel) when a vector contained NaN.
    [[nodiscard]] static std::vector<ArgmaxResult> read_argmax(const Target& target, const BufferView& result,
                                                               std::uint32_t n_vec);
    /// [TOP_K]; then check_status(status).
    void top_k(const Target& target, const TopKArgs& args) const;
    /// [RMS_NORM]
    void rms_norm(const Target& target, const RmsNormArgs& args) const;
    /// [PARTIAL_ROPE]
    void partial_rope_neox(const Target& target, const RopeArgs& args) const;
    /// [SWIGLU]
    void swiglu(const Target& target, const EltwiseArgs& args) const;
    /// [MUL_SIGMOID]
    void mul_sigmoid(const Target& target, const EltwiseArgs& args) const;
    /// [ATTENTION] paged GQA decode / verify; then check_status(status).
    void attention(const Target& target, const AttentionArgs& args) const;
    [[nodiscard]] std::uint64_t attention_workspace_bytes(std::uint32_t n_tokens, std::uint32_t n_head,
                                                          std::uint32_t q_offset) const;
    /// [KV write]; then check_status(status).
    void kv_write(const Target& target, const KvWriteArgs& args) const;
    /// [GET_ROWS]; then check_status(status).
    void get_rows(const Target& target, const GetRowsArgs& args) const;
    /// [ADD] out = a + b (cpu::add); out may alias a or b exactly.
    void add(const Target& target, const EltwiseArgs& args) const;
    /// [ADD + RMS_NORM]
    void add_rms_norm(const Target& target, const AddRmsNormArgs& args) const;

    /// Reads a status word (device: synchronizes the target stream, then copies it) and
    /// throws Error(Kernel) naming the set bits; returns normally when it is zero.
    static void check_status(const Target& target, const BufferView& status);

private:
    OpsOptions options_;
    unsigned gdn_rec_block_ = 0;
    unsigned gdn_chunk_block_ = 0;
    unsigned conv_block_ = 0;
    unsigned norm_block_ = 0;
    unsigned gemv_block_ = 0;
    bool gemv_generic_ = false;
    unsigned argmax_block_ = 0;
    unsigned topk_block_ = 0;
    unsigned rms_block_ = 0;
    unsigned rope_block_ = 0;
    unsigned swiglu_block_ = 0;
    unsigned mulsig_block_ = 0;
    unsigned attn_block_ = 0;
    bool attn_exact_ = false;
    unsigned kvw_block_ = 0;
    unsigned rows_block_ = 0;
    unsigned add_block_ = 0;
    unsigned addnorm_block_ = 0;
};

}  // namespace halo::hip
