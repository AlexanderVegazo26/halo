#pragma once
/// \file
/// HALO backend interface (docs/adr/ADR-001-backend-interface.md, accepted by DECISIONS D-017).
///
/// One typed, per-op, asynchronous-enqueue interface. The single qwen35 forward
/// (models/qwen35) is composed over it, so there is exactly one model definition for every
/// backend (ADR §4). Each op method validates every operand on the host and throws before
/// anything is enqueued; the work then runs in stream order. The host reads results only
/// after Stream::wait() (CPU: the op has already run when the call returns).
///
/// Operands are TensorRefs: a byte range of a Buffer plus an optional row stride (the
/// hip::BufferView / vulkan::BufferView concept, made backend-neutral). fp32 operands need
/// 4-byte-aligned addresses and strides.
///
/// Neutral aliasing rule (ADR §5.2, the intersection of CPU, HIP and Vulkan): an output may
/// not overlap any other operand, with exactly two exceptions — the residual accumulate
/// `h = a + b` with h identical to a (ADD, ADD+RMS_NORM), and PARTIAL_ROPE, which is in place
/// by contract. Every backend enforces this rule even where its kernels could do more, so a
/// forward that passes on the CPU backend never relies on an alias another backend rejects.
///
/// Kernel selection is an argument: every args struct carries a KernelChoice (ADR §5.6);
/// variant ids are explicit, append-only and enumerable through Backend::variants().
///
/// Error mapping (ADR §5.5): shape / alias / range / alignment problems -> Error(Kernel);
/// dtype or limit problems -> Error(Unsupported); unknown variant id -> Error(Config);
/// allocation failure -> Error(Memory).
///
/// WS-BI-1 scope. Status words (StatusRef) are part of the contract, but no backend
/// implements them yet: every args struct's `status` must be empty, and a data error
/// (NaN logit, positive g in chunked GDN, bad block id) raises Error(Kernel) synchronously
/// on the CPU backend, exactly as the halo::cpu function does. Status words, per-sequence
/// owners and the GDN state ring (ADR §5.3) arrive in WS-BI-2.
///
/// Threading: a Backend and its Streams are driven by one thread (the engine worker). The
/// CPU backend is the exception: its ops are synchronous and share only the ThreadPool, so
/// several threads may run ops on disjoint buffers concurrently.
///
/// Ownership: Buffers come from Backend::allocate / import_host and must not outlive the
/// Backend. Imported host memory must outlive its Buffer.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "halo/core/error.h"
#include "halo/tensor/dtype.h"

namespace halo::cpu {
class ThreadPool;
}

namespace halo::backend {

class Backend;

enum class Kind : std::uint8_t { Cpu, Hip, HipEmulation, Vulkan };
enum class Tier : std::uint8_t { Vram, Gtt, Host };

[[nodiscard]] std::string_view to_string(Kind k) noexcept;

/// Memory owned (or imported) by one Backend.
class Buffer {
public:
    virtual ~Buffer() = default;
    [[nodiscard]] virtual std::uint64_t bytes() const noexcept = 0;
    [[nodiscard]] virtual Tier tier() const noexcept = 0;
    /// The backend that owns this buffer; ops reject buffers of another backend.
    [[nodiscard]] virtual const Backend* backend() const noexcept = 0;
    /// False for a read-only import (e.g. the mmapped GGUF bytes); ops reject it as an output.
    [[nodiscard]] virtual bool writable() const noexcept = 0;
    /// Host address of byte 0, or null if the buffer is not host-addressable (ADR A-4: a
    /// design must never require host mapping of a GPU buffer).
    [[nodiscard]] virtual const std::byte* host_data() const noexcept = 0;
    /// As host_data(), but null unless the buffer is also writable.
    [[nodiscard]] virtual std::byte* host_ptr() const noexcept = 0;

protected:
    Buffer() = default;
    Buffer(const Buffer&) = default;
    Buffer& operator=(const Buffer&) = default;
};

/// An operand: bytes [offset, offset + bytes) of `buffer` (bytes 0 = to the end of the
/// buffer); row_stride = bytes between consecutive rows of a 2-D operand (0 = dense).
struct TensorRef {
    const Buffer* buffer = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t row_stride = 0;

    [[nodiscard]] bool empty() const noexcept { return buffer == nullptr; }
    /// Whole buffer, dense.
    [[nodiscard]] static TensorRef of(const Buffer& b) noexcept { return {&b, 0, 0, 0}; }
    /// The same buffer from `offset` + `delta` bytes on; a bounded view (bytes != 0) shrinks
    /// by delta. Error(Kernel) if delta reaches past a bounded view's end (the result would
    /// otherwise become bytes == 0, i.e. unbounded), or on offset overflow.
    [[nodiscard]] TensorRef shifted(std::uint64_t delta) const {
        HALO_CHECK(bytes == 0 || delta < bytes, ErrorCode::Kernel, "TensorRef::shifted: {} bytes past a {}-byte view", delta,
                   bytes);
        HALO_CHECK(offset <= std::numeric_limits<std::uint64_t>::max() - delta, ErrorCode::Kernel,
                   "TensorRef::shifted: offset overflow");
        return {buffer, offset + delta, bytes == 0 ? 0 : bytes - delta, row_stride};
    }
    [[nodiscard]] TensorRef with_stride(std::uint64_t stride_bytes) const noexcept {
        return {buffer, offset, bytes, stride_bytes};
    }
};

/// Status bits a device op sets when it detects a data error (the HIP backend's values).
inline constexpr std::uint32_t kStatusPositiveG = 1u;
inline constexpr std::uint32_t kStatusNaN = 2u;
inline constexpr std::uint32_t kStatusBadBlock = 4u;
inline constexpr std::uint32_t kStatusBadIndex = 8u;
inline constexpr std::uint32_t kBatchWide = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kNoStatus = std::numeric_limits<std::uint32_t>::max();

/// One word of the step's status array and the sequence that owns it (ADR §5.5).
/// WS-BI-1: must be empty (word == kNoStatus) on every backend.
struct StatusRef {
    std::uint32_t word = kNoStatus;
    std::uint32_t owner = kBatchWide;
    [[nodiscard]] bool empty() const noexcept { return word == kNoStatus; }
};

/// Which registered variant runs the op; 0 = the backend's default (ADR §5.6).
struct KernelChoice {
    std::uint32_t variant_id = 0;
};

/// TRD §9 operators of the interface. Values are stable (append only).
enum class OpId : std::uint16_t {
    GetRows = 0,
    RmsNorm = 1,
    AddRmsNorm = 2,
    Gemv = 3,
    GdnGates = 4,
    Conv1dSilu = 5,
    GatedDeltaRule = 6,
    GatedRmsNorm = 7,
    PartialRope = 8,
    KvWrite = 9,
    Attention = 10,
    Swiglu = 11,
    MulSigmoid = 12,
    Add = 13,
    LmHead = 14,
    Argmax = 15,
    TopK = 16,
    Copy = 17,
};
inline constexpr std::size_t kOpCount = 18;

[[nodiscard]] std::string_view op_name(OpId op) noexcept;

/// One registered kernel variant. `id` is explicit and append-only, never an index.
struct VariantInfo {
    std::uint32_t id = 0;
    std::string_view name;
    OpId op = OpId::GetRows;
    std::string_view form;  ///< e.g. "recurrent" / "chunked" for GATED_DELTANET; else empty
};

/// Operand limits of a backend; Qwen35 checks them at construction (Error(Unsupported)).
struct Limits {
    std::uint32_t max_gdn_dk = 0;
    std::uint32_t max_gdn_chunk = 0;
    std::uint32_t max_head_dim = 0;
    std::uint32_t max_conv_k = 0;
    std::uint32_t max_top_k = 0;
    std::uint32_t max_rope_dims = 0;
    bool gdn_chunked = false;  ///< the chunked GATED_DELTANET form exists
};

// ---------------------------------------------------------------------------------------
// Args structs. Sizes in elements unless named *_bytes. Composition notes name the CPU
// reference (a fixed sequence of halo::cpu / halo::tensor calls, ADR §5.4).
// ---------------------------------------------------------------------------------------

/// GET_ROWS (= tensor::dequantize_row per id): out[i] = dequant(table row ids[i]).
///  - table: n_rows rows of row_bytes(type, cols) (row_stride 0 = that size); ids: n_ids int32;
///  - out: n_ids rows of [cols] fp32. An id outside [0, n_rows) is a data error.
struct GetRowsArgs {
    DType type = DType::F32;
    TensorRef table{}, ids{}, out{};
    std::uint32_t n_rows = 0, cols = 0, n_ids = 0;
    KernelChoice kernel{};
    StatusRef status{};
};

/// RMS_NORM (cpu::rms_norm): rows x cols; w: [cols]. out must not overlap x.
/// Per-head norms use rows = T * heads, cols = head_dim, and x.row_stride = the head stride.
struct RmsNormArgs {
    TensorRef x{}, w{}, out{};
    std::uint32_t rows = 0, cols = 0;
    float eps = 1e-6f;
    KernelChoice kernel{};
};

/// ADD + RMS_NORM (= cpu::add(a, b -> h); cpu::rms_norm(h -> out)). h may be identical to a
/// (the residual accumulate); out must not overlap a, b or h.
struct AddRmsNormArgs {
    TensorRef a{}, b{}, h{}, w{}, out{};
    std::uint32_t rows = 0, cols = 0;
    float eps = 1e-6f;
    KernelChoice kernel{};
};

/// MATMUL / QUANT_GEMV (cpu::matmul over a WeightMatrix): y[t][n] = sum_i x[t][i] W[n][i].
///  - w: `rows` rows of row_bytes(wtype, cols) bytes (row_stride 0 = that size). F32 rows are
///    read in place; other types are dequantized with tensor::dequantize_row.
///  - x: n_vec rows of [cols]; y: n_vec rows of [rows]; y must not overlap x or w.
struct GemvArgs {
    DType wtype = DType::F32;
    TensorRef w{}, x{}, y{};
    std::uint32_t rows = 0, cols = 0, n_vec = 1;
    KernelChoice kernel{};
};

/// GDN gates (= the qwen35 sequence: beta_out = cpu::sigmoid(beta); t = alpha + dt_bias;
/// t = cpu::softplus(t); g_out = a * t). alpha, beta, g_out, beta_out: n_tokens rows of
/// [n_heads]; dt_bias, a: [n_heads]. Outputs must not overlap any operand or each other.
struct GdnGateArgs {
    TensorRef alpha{}, beta{}, dt_bias{}, a{}, g_out{}, beta_out{};
    std::uint32_t n_tokens = 0, n_heads = 0;
    KernelChoice kernel{};
};

/// CONV1D_SHORT (cpu::causal_conv1d_silu). x, out: n_tokens rows of [channels]; weight:
/// channels rows of [kernel_size]; conv_state: kernel_size-1 rows of [channels], in/out (WS-BI-1:
/// in place, as today); state_slots: n_slots dense states (D-012). out must not overlap x.
struct Conv1dArgs {
    TensorRef x{}, weight{}, conv_state{}, out{}, state_slots{};
    std::uint32_t n_tokens = 0, channels = 0, kernel_size = 4, n_slots = 0;
    KernelChoice kernel{};
};

enum class GdnForm : std::uint8_t { Recurrent, Chunked };
enum class GdnHeadMapping : std::uint8_t { Tiled, Grouped };

/// GATED_DELTANET (cpu::gated_delta_rule_recurrent / _chunked). D-016: q, k are raw; the
/// forward always sets qk_l2norm and q_scale explicitly. state: [n_v, d_k, d_v] in/out
/// (WS-BI-1: in place); state_slots: n_slots states (D-012); out must not overlap anything.
struct GdnArgs {
    GdnForm form = GdnForm::Recurrent;
    TensorRef q{}, k{}, v{}, g{}, beta{}, state{}, state_slots{}, out{};
    std::uint32_t n_k = 0, n_v = 0, d_k = 0, d_v = 0, n_tokens = 0, n_slots = 0;
    GdnHeadMapping mapping = GdnHeadMapping::Tiled;
    bool qk_l2norm = true;
    std::optional<float> q_scale{};  ///< nullopt = 1/sqrt(d_k) (D-016 amendment)
    std::uint32_t chunk_size = 64;   ///< Chunked only
    KernelChoice kernel{};
    StatusRef status{};
};

/// GATED_NORM (cpu::gated_rms_norm): rows x cols, w: [cols]. out must not overlap x or z.
struct GatedNormArgs {
    TensorRef x{}, z{}, w{}, out{};
    std::uint32_t rows = 0, cols = 0;
    float eps = 1e-6f;
    KernelChoice kernel{};
};

/// PARTIAL_ROPE (cpu::partial_rope_neox), in place on x: n_tokens rows of n_heads heads,
/// head h starting at element h * head_stride of the row (0 = head_dim, dense). positions:
/// n_tokens int32. A strided call is decomposed per head into dense single-head calls.
struct RopeArgs {
    TensorRef x{}, positions{};
    std::uint32_t n_tokens = 0, n_heads = 0, head_dim = 0, rot_dims = 0;
    float theta = 10000.0f;
    std::uint32_t head_stride = 0;
    KernelChoice kernel{};
};

/// KV write: n_tokens K and V rows at history rows start .. start + n_tokens - 1 of `layer`
/// through the block table, into a kv_cache::KvPool storage image (block[layer][K|V][token]
/// [kv_dim] fp32, n_pool_blocks blocks). Rows must lie in blocks the caller reserved.
struct KvWriteArgs {
    TensorRef kv_pool{}, block_table{}, k{}, v{};
    std::uint32_t n_pool_blocks = 0, n_layers = 1, layer = 0, block_tokens = 16, kv_dim = 0;
    std::uint32_t n_block_table = 0;  ///< entries in block_table
    std::uint32_t start = 0, n_tokens = 1;
    KernelChoice kernel{};
    StatusRef status{};
};

/// ATTENTION (cpu::attention_gqa) over the same pool image and block table. q: n_tokens
/// rows of n_head heads, head h at element h * q_head_stride (0 = head_dim). Query t attends
/// history rows 0 ..= q_offset + t. out: n_tokens rows of [n_head * head_dim], distinct.
struct AttentionArgs {
    TensorRef q{}, kv_pool{}, block_table{}, out{};
    std::uint32_t q_head_stride = 0;
    std::uint32_t n_pool_blocks = 0, n_layers = 1, layer = 0, block_tokens = 16;
    std::uint32_t n_block_table = 0;
    std::uint32_t n_head = 0, n_kv_head = 0, head_dim = 0, n_tokens = 1, q_offset = 0;
    float scale = 1.0f;
    KernelChoice kernel{};
    StatusRef status{};
};

/// SWIGLU out = silu(a) * b; MUL_SIGMOID out = a * sigmoid(b); ADD out = a + b.
/// rows x cols. out must not overlap a or b, except ADD's out identical to a.
struct EltwiseArgs {
    TensorRef a{}, b{}, out{};
    std::uint32_t rows = 0, cols = 0;
    KernelChoice kernel{};
};

/// ARGMAX result words of one vector: {index u32, value (f32 bits), nan (0/1)} (HIP layout).
inline constexpr std::uint32_t kArgmaxResultBytes = 12;

/// LM head: gemv over the vocabulary + fused argmax (= Impl::head: cpu::matmul in slabs of
/// 8192 rows, argmax over exactly those values, ties to the lowest index; a NaN logit is a
/// data error). gemv.y (optional): the full logits, n_vec rows of [rows].
/// result: n_vec * kArgmaxResultBytes.
struct LmHeadArgs {
    GemvArgs gemv{};
    TensorRef result{};
    /// M2: rows at/after this index are excluded from the argmax (GGUF LM-head padding past
    /// the tokenizer's real vocabulary). 0 = no clamp (argmax over all gemv.rows), the
    /// pre-M2 behavior; gemv.y (the full logits row, when requested) is unaffected either
    /// way -- the caller still gets every row's raw value, only the *argmax* is clamped.
    std::uint32_t valid_rows = 0;
    KernelChoice kernel{};
    StatusRef status{};
};

/// ARGMAX (cpu::argmax per vector): logits n_vec rows of [n]; result as LmHeadArgs.
struct ArgmaxArgs {
    TensorRef logits{}, result{};
    std::uint32_t n = 0, n_vec = 1;
    KernelChoice kernel{};
    StatusRef status{};
};

/// TOP_K (cpu::top_k per vector): ids int32 / values f32, n_vec rows of [k] (dense).
struct TopKArgs {
    TensorRef logits{}, ids{}, values{};
    std::uint32_t n = 0, k = 0, n_vec = 1;
    KernelChoice kernel{};
    StatusRef status{};
};

/// COPY: `bytes` bytes from src to dst (dense). Must not overlap.
struct CopyArgs {
    TensorRef src{}, dst{};
    std::uint64_t bytes = 0;
    KernelChoice kernel{};
};

/// A type-erased op call (tests, the tuner, future per-op mixing). EltwiseArgs serves three
/// ops, so the OpId selects which.
struct OpInvocation {
    OpId op = OpId::Copy;
    std::variant<GetRowsArgs, RmsNormArgs, AddRmsNormArgs, GemvArgs, GdnGateArgs, Conv1dArgs, GdnArgs, GatedNormArgs,
                 RopeArgs, KvWriteArgs, AttentionArgs, EltwiseArgs, LmHeadArgs, ArgmaxArgs, TopKArgs, CopyArgs>
        args;
};

/// Ordered work queue; one per engine worker; a single-threaded object.
class Stream {
public:
    virtual ~Stream() = default;
    virtual void submit() = 0;          ///< CPU: no-op
    virtual void wait() = 0;            ///< throws the translated device error
    virtual void abort() noexcept = 0;  ///< drain / discard (CPU: nothing)

protected:
    Stream() = default;
    Stream(const Stream&) = default;
    Stream& operator=(const Stream&) = default;
};

class Backend {
public:
    virtual ~Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    [[nodiscard]] virtual Kind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string describe() const = 0;
    [[nodiscard]] virtual Limits limits() const noexcept = 0;

    // ---- memory ----------------------------------------------------------------------
    /// Zero-filled on the CPU backend. Error(Memory) on failure.
    [[nodiscard]] virtual std::unique_ptr<Buffer> allocate(std::uint64_t bytes, Tier tier) = 0;
    /// Wraps caller-owned host memory (CPU: zero-copy). The memory must outlive the Buffer.
    [[nodiscard]] virtual std::unique_ptr<Buffer> import_host(std::span<std::byte> bytes) = 0;
    /// Read-only import (the mmapped GGUF weights): writable() is false.
    [[nodiscard]] virtual std::unique_ptr<Buffer> import_host_readonly(std::span<const std::byte> bytes) = 0;
    [[nodiscard]] virtual std::unique_ptr<Stream> create_stream() = 0;
    /// Ordered in the stream; `src` must stay valid until wait() returns.
    virtual void upload(Stream& s, TensorRef dst, std::span<const std::byte> src) = 0;
    /// `dst` is valid after wait() returns (CPU: immediately).
    virtual void download(Stream& s, TensorRef src, std::span<std::byte> dst) = 0;

    // ---- registry --------------------------------------------------------------------
    /// Registered variants of `op` (optionally of one form). Never empty for an implemented op.
    [[nodiscard]] virtual std::span<const VariantInfo> variants(OpId op, std::string_view form = {}) const = 0;

    // ---- ops: validate on the host (throw before enqueuing), then enqueue --------------
    virtual void get_rows(Stream&, const GetRowsArgs&) = 0;
    virtual void rms_norm(Stream&, const RmsNormArgs&) = 0;
    virtual void add_rms_norm(Stream&, const AddRmsNormArgs&) = 0;
    virtual void gemv(Stream&, const GemvArgs&) = 0;
    virtual void gdn_gates(Stream&, const GdnGateArgs&) = 0;
    virtual void conv1d_silu(Stream&, const Conv1dArgs&) = 0;
    virtual void gated_delta_rule(Stream&, const GdnArgs&) = 0;
    virtual void gated_rms_norm(Stream&, const GatedNormArgs&) = 0;
    virtual void partial_rope(Stream&, const RopeArgs&) = 0;
    virtual void kv_write(Stream&, const KvWriteArgs&) = 0;
    virtual void attention(Stream&, const AttentionArgs&) = 0;
    virtual void swiglu(Stream&, const EltwiseArgs&) = 0;
    virtual void mul_sigmoid(Stream&, const EltwiseArgs&) = 0;
    virtual void add(Stream&, const EltwiseArgs&) = 0;
    virtual void lm_head(Stream&, const LmHeadArgs&) = 0;
    virtual void argmax(Stream&, const ArgmaxArgs&) = 0;
    virtual void top_k(Stream&, const TopKArgs&) = 0;
    virtual void copy(Stream&, const CopyArgs&) = 0;

    /// Dispatches to the typed method. Error(Api) if `inv.args` does not hold the args type
    /// of `inv.op`.
    void run_op(Stream& s, const OpInvocation& inv);

protected:
    Backend() = default;
};

/// The CPU reference backend over halo::cpu (WS-BI-1). `pool` may be null
/// (single-threaded) and must outlive the backend. One "reference" variant (id 0) per op.
[[nodiscard]] std::unique_ptr<Backend> make_cpu_backend(cpu::ThreadPool* pool);

/// Decodes argmax result words (n_vec * 3 u32). Error(Kernel) naming the vector if its NaN
/// word is set (D-016).
struct ArgmaxResult {
    std::int32_t index = -1;
    float value = 0.0f;
};
[[nodiscard]] std::vector<ArgmaxResult> decode_argmax(std::span<const std::byte> words);

}  // namespace halo::backend
