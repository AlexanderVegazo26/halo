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
#include <string>

#include "halo/backends/hip/runtime.h"

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

/// Variant choice per operator (names from the kernel registry, registry.h).
struct OpsOptions {
    std::string gdn_recurrent = "gdn_recurrent_b128";
    std::string gdn_chunked = "gdn_chunked_b64";
    std::string conv1d = "conv1d_silu_b256";
    std::string gated_norm = "gated_norm_b128";
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

    /// Reads a status word (device: synchronizes the target stream, then copies it) and
    /// throws Error(Kernel) naming the set bits; returns normally when it is zero.
    static void check_status(const Target& target, const BufferView& status);

private:
    OpsOptions options_;
    unsigned gdn_rec_block_ = 0;
    unsigned gdn_chunk_block_ = 0;
    unsigned conv_block_ = 0;
    unsigned norm_block_ = 0;
};

}  // namespace halo::hip
