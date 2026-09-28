// LM-head operators (WS-F2 V3): logits GEMV + argmax per vector, and TOP_K. Host side:
// validate every view up front (ops_internal.h), then record the dispatches.
// Semantics: cpu::matmul + cpu::argmax (= cpu::matmul_argmax) and cpu::top_k (ops.h).

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "ops_internal.h"
#include "vk_util.h"

namespace halo::vulkan {

using detail::Access;
using detail::checked_add;
using detail::checked_mul;
using detail::k_f32;
using detail::operand;
using detail::Operand;
using detail::require_disjoint;

namespace {

constexpr std::uint32_t k_topk_chunk = 2048;  // topk.comp CHUNK
constexpr std::uint32_t k_topk_max_k = k_topk_chunk / 2;
constexpr std::uint32_t k_topk_wg = 256;
constexpr std::uint64_t k_topk_shared_bytes = std::uint64_t{k_topk_chunk} * 8;

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

/// Byte distance between the vectors of a 2-D view (operand()'s rule).
std::uint64_t vec_stride(const BufferView& v, std::uint32_t n_vec, std::uint64_t row_bytes) {
    return (n_vec == 1 || v.row_stride == 0) ? row_bytes : v.row_stride;
}

}  // namespace

std::uint64_t Ops::lm_head_workspace_bytes(std::uint32_t rows, std::uint32_t n_vec, bool logits_in_workspace) const {
    constexpr std::string_view op = "lm_head";
    const std::uint64_t scratch = checked_mul(std::uint64_t{n_vec}, argmax_scratch_bytes(rows), op);
    if (!logits_in_workspace) return scratch;
    return checked_add(checked_mul(checked_mul(std::uint64_t{n_vec}, rows, op), k_f32, op), scratch, op);
}

void Ops::lm_head(Stream& stream, const LmHeadArgs& a) {
    constexpr std::string_view op = "lm_head";
    const GemvArgs& g = a.gemv;
    HALO_CHECK(g.rows > 0 && g.cols > 0 && g.n_vec > 0, ErrorCode::Kernel, "{}: empty shape {}x{} n_vec={}", op,
               g.rows, g.cols, g.n_vec);
    const bool in_ws = g.y.empty();
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t logit_row = std::uint64_t{g.rows} * k_f32;
    const std::uint64_t scratch_one = argmax_scratch_bytes(g.rows);
    // Validate everything before recording anything.
    const Operand ow = operand(g.w, g.rows, matvec_row_bytes(g.wtype, g.cols),
                               g.wtype == DType::F32 ? Access::Floats : Access::Words, align, op, "W");
    const Operand ox = operand(g.x, g.n_vec, std::uint64_t{g.cols} * k_f32, Access::Floats, align, op, "x");
    const Operand ows =
        operand(a.workspace, 1, lm_head_workspace_bytes(g.rows, g.n_vec, in_ws), Access::Floats, align, op, "workspace");
    const Operand ores =
        operand(a.result, 1, std::uint64_t{g.n_vec} * k_argmax_result_bytes, Access::Floats, align, op, "result");
    require_disjoint(ows, "workspace", {{&ow, "W"}, {&ox, "x"}, {&ores, "result"}}, op);
    require_disjoint(ores, "result", {{&ow, "W"}, {&ox, "x"}}, op);
    BufferView logits = in_ws ? BufferView(*a.workspace.buffer, a.workspace.offset, g.n_vec * logit_row) : g.y;
    if (!in_ws) {
        const Operand oy = operand(g.y, g.n_vec, logit_row, Access::Floats, align, op, "y (logits)");
        require_disjoint(oy, "y (logits)", {{&ow, "W"}, {&ox, "x"}, {&ows, "workspace"}, {&ores, "result"}}, op);
    }
    const std::uint64_t l_stride = in_ws ? logit_row : vec_stride(g.y, g.n_vec, logit_row);
    const std::uint64_t scratch_base = a.workspace.offset + (in_ws ? g.n_vec * logit_row : 0);

    GemvArgs gg = g;
    gg.y = logits;
    gemv_impl(stream, gg, op);
    for (std::uint32_t t = 0; t < g.n_vec; ++t) {
        argmax(stream, BufferView(*logits.buffer, logits.offset + t * l_stride, logit_row),
               g.rows, BufferView(*a.workspace.buffer, scratch_base + t * scratch_one, scratch_one),
               BufferView(*a.result.buffer, a.result.offset + std::uint64_t{t} * k_argmax_result_bytes,
                          k_argmax_result_bytes),
               a.valid_rows);
    }
}

std::uint64_t topk_workspace_bytes(std::uint32_t n, std::uint32_t k, std::uint32_t n_vec) {
    if (n <= k_topk_chunk) return 0;
    // Two ping-pong candidate regions of n_vec * ceil(n / CHUNK) * k (value, index) pairs.
    return 2 * std::uint64_t{n_vec} * ceil_div(n, k_topk_chunk) * k * 8;
}

void Ops::top_k(Stream& stream, const TopKArgs& a) {
    constexpr std::string_view op = "top_k";
    HALO_CHECK(a.n > 0 && a.n_vec > 0, ErrorCode::Kernel, "{}: empty shape n={} n_vec={}", op, a.n, a.n_vec);
    HALO_CHECK(a.n < 0x80000000u, ErrorCode::Kernel, "{}: n={} exceeds the int32 index range", op, a.n);
    HALO_CHECK(a.k >= 1 && a.k <= a.n, ErrorCode::Kernel, "{}: k = {} for {} entries", op, a.k, a.n);
    HALO_CHECK(a.k <= k_topk_max_k, ErrorCode::Unsupported, "{}: k = {} exceeds this backend's limit {}", op, a.k,
               k_topk_max_k);
    const DeviceInfo& info = ctx_->info();
    HALO_CHECK(info.max_shared_memory >= k_topk_shared_bytes && info.max_workgroup_size[0] >= k_topk_wg &&
                   info.max_workgroup_invocations >= k_topk_wg,
               ErrorCode::Unsupported, "{}: needs {} bytes of shared memory and {} invocations", op,
               k_topk_shared_bytes, k_topk_wg);
    const std::uint64_t align = info.min_storage_buffer_offset_alignment;
    const Operand ol = operand(a.logits, a.n_vec, std::uint64_t{a.n} * k_f32, Access::Floats, align, op, "logits");
    const Operand oi = operand(a.ids, a.n_vec, std::uint64_t{a.k} * 4, Access::Floats, align, op, "ids");
    const Operand ov = operand(a.values, a.n_vec, std::uint64_t{a.k} * k_f32, Access::Floats, align, op, "values");
    const Operand os = operand(a.status, 1, k_status_bytes, Access::Floats, align, op, "status");
    require_disjoint(oi, "ids", {{&ol, "logits"}, {&ov, "values"}, {&os, "status"}}, op);
    require_disjoint(ov, "values", {{&ol, "logits"}, {&os, "status"}}, op);
    require_disjoint(os, "status", {{&ol, "logits"}}, op);
    const std::uint64_t ws_bytes = topk_workspace_bytes(a.n, a.k, a.n_vec);
    // Candidate regions: [A values | A ids | B values | B ids], each n_vec * cap entries.
    const std::uint64_t cap = ceil_div(a.n, k_topk_chunk) * a.k;
    const std::uint64_t region = std::uint64_t{a.n_vec} * cap * 4;
    std::array<Operand, 4> reg{};
    if (ws_bytes > 0) {
        const Operand ows = operand(a.workspace, 1, ws_bytes, Access::Floats, align, op, "workspace");
        require_disjoint(ows, "workspace",
                         {{&ol, "logits"}, {&oi, "ids"}, {&ov, "values"}, {&os, "status"}}, op);
        for (std::size_t r = 0; r < 4; ++r) {
            reg[r] = operand(BufferView(*a.workspace.buffer, a.workspace.offset + r * region, region), 1, region,
                             Access::Floats, align, op, "workspace region");
        }
    }
    const std::uint32_t cap32 = detail::to_u32(cap, "top_k candidates per vector");

    struct Push {
        std::uint32_t count, n_chunks, k, n_vec, sv_off, sv_stride, si_off, si_stride, dv_off, dv_stride, di_off,
            di_stride, st_off;
    };
    const Kernel& k_first = kernel("topk", 5, sizeof(Push), {{0, k_topk_wg}, {1, 1}}, {k_topk_wg, 1, 1});
    const Kernel& k_next = kernel("topk", 5, sizeof(Push), {{0, k_topk_wg}, {1, 0}}, {k_topk_wg, 1, 1});
    // Pass plan (validated before recording).
    std::uint64_t count = a.n;
    std::uint32_t passes = 0;
    while (true) {
        ++passes;
        const std::uint64_t chunks = ceil_div(count, k_topk_chunk);
        (void)grid_1d(chunks * a.n_vec, info.max_workgroup_count[0], info.max_workgroup_count[1]);
        if (chunks == 1) break;
        count = chunks * a.k;
    }

    stream.fill(*a.status.buffer, a.status.offset, k_status_bytes, 0u);
    count = a.n;
    bool first = true;
    std::size_t cur = 0;  // destination region pair: 0 = A, 1 = B
    for (std::uint32_t p = 0; p < passes; ++p) {
        const std::uint32_t chunks = static_cast<std::uint32_t>(ceil_div(count, k_topk_chunk));
        const bool final = chunks == 1;
        const Operand& sv = first ? ol : reg[2 * (1 - cur)];
        const Operand& si = first ? ol : reg[2 * (1 - cur) + 1];
        const Operand& dv = final ? ov : reg[2 * cur];
        const Operand& di = final ? oi : reg[2 * cur + 1];
        const Push push{static_cast<std::uint32_t>(count),
                        chunks,
                        a.k,
                        a.n_vec,
                        sv.off,
                        first ? ol.stride : cap32,
                        si.off,
                        first ? 0u : cap32,
                        dv.off,
                        final ? ov.stride : cap32,
                        di.off,
                        final ? oi.stride : cap32,
                        os.off};
        const std::array bindings{sv.binding, si.binding, dv.binding, di.binding, os.binding};
        stream.dispatch(first ? k_first : k_next, bindings, push,
                        grid_1d(std::uint64_t{chunks} * a.n_vec, info.max_workgroup_count[0],
                                info.max_workgroup_count[1]),
                        0b11100);  // dst values/ids + status written
        count = std::uint64_t{chunks} * a.k;
        first = false;
        cur = 1 - cur;
    }
}

}  // namespace halo::vulkan
