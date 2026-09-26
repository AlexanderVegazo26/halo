#pragma once
/// \file
/// The Vulkan backend (ADR-001 §5.1, WS-BI-5 / WS-F2 V5): halo::backend::Backend over the
/// halo::vulkan operators (backends/vulkan). Built into the separate library
/// halo_backend_vulkan_adapter when the Vulkan backend is configured (src/backend/CMakeLists.txt),
/// so halo_backend itself never links Vulkan.
///
/// Semantics per op: the halo::vulkan op named in the adapter, each accepted by a differential
/// test against halo::cpu (tests/unit/vulkan); the interface's neutral aliasing rule
/// (backend.h) is enforced here even where a Vulkan kernel accepts more.
///
/// Memory (ADR §5.2):
///  - allocate: device memory for Tier::Vram, host-visible memory for Gtt, host-cached for
///    Host; zero-filled (as the CPU backend). host_data() / host_ptr() are always null (A-4).
///  - import_host_readonly: a device COPY made at import time; the caller's bytes must stay
///    unchanged while the Buffer lives (they are also read on the host, e.g. RoPE positions).
///  - import_host (writable): a device mirror of the caller's memory. The bytes are copied in
///    at import time; after every successful Stream::wait() of this backend the device copy
///    is written back to the caller's memory (for every live writable import). The caller
///    must not modify the memory while the Buffer lives. Two live writable imports may not
///    overlap (Error(Kernel)).
///  - upload / download: staged copies recorded in stream order; a download's destination is
///    filled at wait().
///
/// Data errors (bad row id / block id, positive g in chunked GDN, NaN in TOP_K) are detected
/// on the device into stream-owned status words and raised as Error(Kernel) by
/// Stream::wait(), naming the op — the CPU backend raises the same Error(Kernel)
/// synchronously. In that case wait() does NOT write back imports (the host keeps its
/// pre-step state). Per-sequence StatusRef owners (ADR §5.5) need the step status array of
/// WS-BI-2: a non-empty StatusRef is Error(Unsupported), as on the CPU backend. A NaN logit in
/// LM_HEAD / ARGMAX is reported through the result's NaN word (backend::decode_argmax maps it
/// to the poisoned-row sentinel {-1, NaN}; only that row is affected, H1).
///
/// Not implemented: the GDN / conv state ring (ADR §5.3, WS-BI-2; state is in place, as on the
/// CPU backend), KernelChoice variants other than 0, Grouped GDN head mapping, weight types
/// outside vulkan::matvec_row_bytes (Error(Unsupported)), RoPE positions that are not a
/// read-only host import (Error(Unsupported)). Device loss surfaces as Error(Backend).
///
/// Threading: one thread drives the backend and its streams (vulkan::Ops is not thread-safe).

#include <memory>

#include "halo/backend/backend.h"
#include "halo/backends/vulkan/ops.h"

namespace halo::vulkan {
class Context;
}

namespace halo::backend {

struct VulkanBackendOptions {
    vulkan::OpsOptions ops{};
};

/// The Vulkan backend over `ctx` (which it keeps alive).
[[nodiscard]] std::unique_ptr<Backend> make_vulkan_backend(std::shared_ptr<vulkan::Context> ctx,
                                                           const VulkanBackendOptions& options = {});

}  // namespace halo::backend
