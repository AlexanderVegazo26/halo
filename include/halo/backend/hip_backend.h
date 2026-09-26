#pragma once
/// \file
/// The HIP backend behind the Backend interface (ADR-001 §5.1, §9 WS-BI-3).
///
/// An adapter over halo::hip::Ops: every TensorRef becomes a hip::BufferView (byte for byte:
/// same buffer, offset, bytes and row stride), every KernelChoice selects one prebuilt
/// hip::Ops (one per registered variant, built at construction; ADR §5.1 "per-variant Ops
/// cache"), and the workspaces and status words the HIP kernels need but the interface does
/// not carry are allocated per stream and live until Stream::wait() / abort().
///
/// Two modes (ADR §5.6: they are different Kinds and never share a profile label):
///  - Emulation (Kind::HipEmulation): the HIP kernel bodies run on the host (hip::Target::
///    emulation), over host memory. Bit-identical to halo::cpu for the variants docs/hip.md
///    marks bitwise (ADR A-2). This is how the dev host runs the backend.
///  - Device (Kind::Hip): a real gfx1151 device through a hip::Stream. DECISIONS D-001: the
///    dev host has no AMD GPU, so this mode is compile-checked only; make_hip_backend raises
///    Error(Device) there.
///
/// Contract beyond backend.h (both modes):
///  - The neutral aliasing rule, buffer ownership, read-only outputs and foreign streams are
///    enforced exactly as on the CPU backend (the HIP kernels alone would accept more).
///  - Operand limits (Limits) raise Error(Unsupported) before any HIP code runs: GDN d_k <= 128,
///    chunk <= 64, head_dim <= 256, conv kernel <= 8, top-k <= 1024, rope dims <= 128.
///  - Data errors (GET_ROWS id out of range, KV/attention block id outside the pool, positive
///    g in chunked GDN, NaN in standalone ARGMAX / TOP_K) raise Error(Kernel) naming the op:
///    in emulation right after the op (the CPU backend's timing), on a device at the next
///    Stream::wait() (ADR §5.5). A NaN logit in the fused LM head instead poisons only its
///    own row ({-1, NaN} in the result words, H1) -- the caller decodes it per sequence.
///    Status words (StatusRef) must be empty, as on the CPU backend (WS-BI-1 contract);
///    per-sequence status arrives with WS-BI-2's interface change.
///  - Device mode: import_host_readonly COPIES the bytes into a read-only VRAM buffer (D-017:
///    weights are copied at load until owner question Q3 is answered); import_host raises
///    Error(Unsupported), because a copy would break the caller's expectation that writes are
///    visible in its host memory (host state moves onto backend buffers in WS-BI-2/6). Buffers
///    are not host-addressable (host_data() == nullptr, ADR A-4).
///  - Emulation: allocate() fills new buffers with 0xFF bytes (NaN) by default, because device
///    memory is not zero-filled; a caller relying on the CPU backend's zero fill fails here.
///
/// Registry: variants(op, form) lists this backend's variants with explicit, append-only ids
/// starting at 1 (0 = the configured default, KernelChoice{}); the id -> HIP variant name map
/// is pinned by a test. Order: registry order, grouped by op and form.
///
/// Threading: a HIP backend and its streams are driven by one thread (ADR §5.1).

#include <cstdint>
#include <memory>

#include "halo/backend/backend.h"
#include "halo/backends/hip/ops.h"

namespace halo::backend {

enum class HipMode : std::uint8_t { Emulation, Device };

struct HipBackendOptions {
    HipMode mode = HipMode::Emulation;
    int device_ordinal = 0;               ///< Device mode
    bool emulation_reverse_order = false;  ///< Emulation: run threads/workgroups in reverse order
    bool emulation_poison_alloc = true;    ///< Emulation: allocate() fills with 0xFF (NaN)
    /// The variant each op runs for KernelChoice{} (variant_id 0), by HIP registry name.
    /// Error(Config) at construction for an unknown name.
    hip::OpsOptions defaults{};
};

/// The bitwise profile: defaults whose every op is bit-identical to halo::cpu in emulation
/// (docs/hip.md: gemv_generic_b64, attn_exact_b128; every other default already is).
[[nodiscard]] hip::OpsOptions hip_bitwise_defaults();

/// Throws Error(Config) for an unknown default variant, Error(Device) when Device mode finds
/// no usable device (always on the dev host, D-001).
[[nodiscard]] std::unique_ptr<Backend> make_hip_backend(const HipBackendOptions& options = {});

}  // namespace halo::backend
