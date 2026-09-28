// Paged-history and lookup operators (WS-F2 V2): GET_ROWS, KV write and ATTENTION over the
// kv_cache pool layout, plus their device status words. Host side: validate every view
// (ops_internal.h), check aliasing, zero the status word, record one dispatch.
// Semantics: tensor::dequantize_row, kv_cache::SequenceKv::write, cpu::attention_gqa (ops.h).

#include <array>
#include <cmath>
#include <format>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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
using detail::to_u32;

namespace {

constexpr std::uint32_t k_rows_wg = 256;

/// Validates the 4-byte status view.
Operand status_operand(const BufferView& v, std::uint64_t align, std::string_view op) {
    Operand o = operand(v, 1, k_status_bytes, Access::Floats, align, op, "status");
    return o;
}

void zero_status(Stream& stream, const BufferView& v) { stream.fill(*v.buffer, v.offset, k_status_bytes, 0u); }

constexpr std::uint32_t k_max_head_dim = 256;       // attention.comp MAX_HD
constexpr std::uint32_t k_attn_dims_per_thread = 4;  // attention.comp DIMS

/// Pool geometry shared by kv_write and attention (ops.h "The paged KV pool").
struct PoolGeometry {
    std::uint32_t block_floats = 0;
    std::uint32_t layer_off = 0;
};

PoolGeometry pool_geometry(std::uint32_t n_layers, std::uint32_t layer, std::uint32_t block_tokens,
                           std::uint64_t kv_dim, std::string_view op) {
    HALO_CHECK(n_layers > 0 && layer < n_layers, ErrorCode::Kernel, "{}: layer {} outside [0, {})", op, layer,
               n_layers);
    HALO_CHECK(block_tokens > 0 && kv_dim > 0, ErrorCode::Kernel, "{}: block_tokens {} / kv_dim {} must be > 0", op,
               block_tokens, kv_dim);
    const std::uint64_t kv_block = checked_mul(checked_mul(2, std::uint64_t{block_tokens}, op), kv_dim, op);
    PoolGeometry g;
    g.block_floats = to_u32(checked_mul(kv_block, n_layers, op), std::format("{} block floats", op));
    g.layer_off = static_cast<std::uint32_t>(kv_block * layer);  // < block_floats
    return g;
}

/// The whole pool is bound as one descriptor with 32-bit float indices; a larger pool is a
/// limit of this backend (Error(Unsupported), ADR-001 §5.5), not a shape error.
Operand pool_operand(const BufferView& v, std::uint32_t n_blocks, std::uint32_t block_floats, std::uint64_t align,
                     const DeviceInfo& info, std::string_view op) {
    HALO_CHECK(n_blocks > 0, ErrorCode::Kernel, "{}: n_pool_blocks is 0", op);
    const std::uint64_t bytes = checked_mul(checked_mul(std::uint64_t{n_blocks}, block_floats, op), k_f32, op);
    const std::uint64_t bound = checked_add(bytes, v.offset % align, op);
    HALO_CHECK(bound <= info.max_storage_buffer_range && bound / k_f32 <= 0xFFFFFFFFull, ErrorCode::Unsupported,
               "{}: the KV pool ({} bytes) exceeds this backend's limit: one descriptor of at most "
               "maxStorageBufferRange ({} bytes) and 2^32 floats",
               op, bytes, info.max_storage_buffer_range);
    return operand(v, n_blocks, std::uint64_t{block_floats} * k_f32, Access::Floats, align, op, "kv_pool");
}

/// Block-table view covering history rows [0, rows_end).
Operand table_operand(const BufferView& v, std::uint64_t rows_end, std::uint32_t block_tokens, std::uint64_t align,
                      std::string_view op) {
    const std::uint64_t entries = (rows_end + block_tokens - 1) / block_tokens;
    return operand(v, 1, entries * 4, Access::Floats, align, op, "block_table");
}


}  // namespace

std::uint32_t read_status(const Buffer& status, std::uint64_t offset) {
    std::array<std::uint32_t, 1> w{};
    status.download(std::span<std::uint32_t>(w), offset);
    return w[0];
}

void check_status(std::uint32_t word, std::string_view op) {
    if (word == 0) return;
    std::string what;
    if ((word & k_status_bad_block) != 0) what += " bad-block (a block-table entry >= n_pool_blocks)";
    if ((word & k_status_bad_index) != 0) what += " bad-index (a row id outside [0, n_rows))";
    if ((word & ~(k_status_bad_block | k_status_bad_index)) != 0) what += " unknown bits";
    throw_error(ErrorCode::Kernel, "{}: device status 0x{:x}:{}", op, word, what);
}

void Ops::get_rows(Stream& stream, const GetRowsArgs& a) {
    constexpr std::string_view op = "get_rows";
    HALO_CHECK(a.n_rows > 0 && a.cols > 0 && a.n_ids > 0, ErrorCode::Kernel, "{}: empty shape rows={} cols={} ids={}",
               op, a.n_rows, a.cols, a.n_ids);
    const std::string shader = detail::weight_shader("get_rows", a.wtype);
    const std::uint64_t row_bytes = matvec_row_bytes(a.wtype, a.cols);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const Operand ow = operand(a.w, a.n_rows, row_bytes, a.wtype == DType::F32 ? Access::Floats : Access::Words, align,
                               op, "w");
    const Operand oi = operand(a.ids, 1, std::uint64_t{a.n_ids} * 4, Access::Floats, align, op, "ids");
    const Operand oo = operand(a.out, a.n_ids, std::uint64_t{a.cols} * k_f32, Access::Floats, align, op, "out");
    const Operand os = status_operand(a.status, align, op);
    require_disjoint(oo, "out", {{&ow, "w"}, {&oi, "ids"}, {&os, "status"}}, op);
    require_disjoint(os, "status", {{&ow, "w"}, {&oi, "ids"}}, op);
    struct Push {
        std::uint32_t n_rows, cols, n_ids, w_off, w_stride, ids_off, o_off, o_stride, st_off, row_base, slab_rows, flag_bad;
    };
    const Kernel& k = kernel(shader, 4, sizeof(Push), {{0, k_rows_wg}}, {k_rows_wg, 1, 1});
    const DeviceInfo& info = ctx_->info();
    const GroupCount groups = grid_1d(a.n_ids, info.max_workgroup_count[0], info.max_workgroup_count[1]);
    // A table larger than one binding (e.g. the 248320-row embedding on a device with a small
    // maxStorageBufferRange) is gathered slab by slab: each dispatch binds rows
    // [row_base, row_base + slab_rows) and writes the ids that fall into them.
    const std::uint64_t limit = detail::binding_limit(options_, info);
    const std::uint64_t per =
        ow.binding.range > limit ? detail::rows_per_slab(limit, align, ow.stride * (a.wtype == DType::F32 ? k_f32 : 1),
                                                         row_bytes, op)
                                 : a.n_rows;
    const std::uint64_t ws = a.n_rows == 1 || a.w.row_stride == 0 ? row_bytes : a.w.row_stride;
    // Validate every slab binding before recording anything.
    std::vector<Operand> slabs;
    for (std::uint64_t r0 = 0; r0 < a.n_rows; r0 += per) {
        const std::uint64_t n = std::min<std::uint64_t>(per, a.n_rows - r0);
        slabs.push_back(per == a.n_rows ? ow
                                        : operand(BufferView(*a.w.buffer, a.w.offset + r0 * ws, (n - 1) * ws + row_bytes, ws),
                                                  n, row_bytes, a.wtype == DType::F32 ? Access::Floats : Access::Words,
                                                  align, op, "w (slab)"));
    }
    zero_status(stream, a.status);
    for (std::size_t si = 0; si < slabs.size(); ++si) {
        const std::uint64_t r0 = si * per;
        const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::uint64_t>(per, a.n_rows - r0));
        const Push push{a.n_rows, a.cols,    a.n_ids,  slabs[si].off, slabs[si].stride, oi.off,
                        oo.off,   oo.stride, os.off,   static_cast<std::uint32_t>(r0),  n,  si == 0 ? 1u : 0u};
        const std::array bindings{slabs[si].binding, oi.binding, oo.binding, os.binding};
        stream.dispatch(k, bindings, push, groups, 0b1100);  // out + status written
    }
}

void Ops::kv_write(Stream& stream, const KvWriteArgs& a) {
    constexpr std::string_view op = "kv_write";
    HALO_CHECK(a.n_tokens > 0, ErrorCode::Kernel, "{}: n_tokens is 0", op);
    const PoolGeometry g = pool_geometry(a.n_layers, a.layer, a.block_tokens, a.kv_dim, op);
    const DeviceInfo& info = ctx_->info();
    const std::uint64_t align = info.min_storage_buffer_offset_alignment;
    const std::uint64_t rows_end = checked_add(std::uint64_t{a.start}, a.n_tokens, op);
    (void)to_u32(rows_end, "kv_write history row");
    const Operand op_pool = pool_operand(a.kv_pool, a.n_pool_blocks, g.block_floats, align, info, op);
    const Operand ot = table_operand(a.block_table, rows_end, a.block_tokens, align, op);
    const std::uint64_t row = std::uint64_t{a.kv_dim} * k_f32;
    const Operand ok = operand(a.k, a.n_tokens, row, Access::Floats, align, op, "k");
    const Operand ov = operand(a.v, a.n_tokens, row, Access::Floats, align, op, "v");
    const Operand os = status_operand(a.status, align, op);
    require_disjoint(op_pool, "kv_pool", {{&ot, "block_table"}, {&ok, "k"}, {&ov, "v"}, {&os, "status"}}, op);
    require_disjoint(os, "status", {{&ot, "block_table"}, {&ok, "k"}, {&ov, "v"}}, op);
    struct Push {
        std::uint32_t n_pool_blocks, block_floats, layer_off, bt, kv_dim, start, n_tokens, pool_off, tab_off, k_off,
            k_stride, v_off, v_stride, st_off;
    } push{a.n_pool_blocks, g.block_floats, g.layer_off, a.block_tokens, a.kv_dim, a.start, a.n_tokens,
           op_pool.off,     ot.off,         ok.off,      ok.stride,      ov.off,   ov.stride, os.off};
    const Kernel& k = kernel("kv_write", 5, sizeof(Push), {{0, k_rows_wg}}, {k_rows_wg, 1, 1});
    const std::array bindings{op_pool.binding, ot.binding, ok.binding, ov.binding, os.binding};
    const GroupCount groups = grid_1d(a.n_tokens, info.max_workgroup_count[0], info.max_workgroup_count[1]);
    zero_status(stream, a.status);
    stream.dispatch(k, bindings, push, groups, 0b10001);  // pool + status written
}

void Ops::attention(Stream& stream, const AttentionArgs& a) {
    constexpr std::string_view op = "attention";
    HALO_CHECK(a.n_tokens > 0 && a.n_head > 0 && a.n_kv_head > 0 && a.head_dim > 0 && a.n_head % a.n_kv_head == 0,
               ErrorCode::Kernel, "{}: bad shape T={} n_head={} n_kv_head={} head_dim={}", op, a.n_tokens, a.n_head,
               a.n_kv_head, a.head_dim);
    const std::uint32_t wg = options_.attention_workgroup;
    HALO_CHECK(a.head_dim <= k_max_head_dim && a.head_dim <= k_attn_dims_per_thread * wg, ErrorCode::Unsupported,
               "{}: head_dim {} exceeds this backend's limit (<= {} and <= {} * attention_workgroup {})", op,
               a.head_dim, k_max_head_dim, k_attn_dims_per_thread, wg);
    HALO_CHECK(std::isfinite(a.scale), ErrorCode::Kernel, "{}: scale must be finite", op);
    const std::uint32_t qhs = a.q_head_stride == 0 ? a.head_dim : a.q_head_stride;
    HALO_CHECK(qhs >= a.head_dim, ErrorCode::Kernel, "{}: q_head_stride {} < head_dim {}", op, qhs, a.head_dim);
    const std::uint64_t kv_dim = checked_mul(std::uint64_t{a.n_kv_head}, a.head_dim, op);
    const PoolGeometry g = pool_geometry(a.n_layers, a.layer, a.block_tokens, kv_dim, op);
    const DeviceInfo& info = ctx_->info();
    const std::uint64_t align = info.min_storage_buffer_offset_alignment;
    const std::uint64_t rows_end = checked_add(std::uint64_t{a.q_offset}, a.n_tokens, op);
    (void)to_u32(rows_end, "attention history rows");
    const std::uint32_t items = to_u32(checked_mul(std::uint64_t{a.n_tokens}, a.n_head, op), "attention items");
    const std::uint64_t q_row = checked_add(checked_mul(std::uint64_t{a.n_head} - 1, qhs, op), a.head_dim, op) * k_f32;
    const Operand oq = operand(a.q, a.n_tokens, q_row, Access::Floats, align, op, "q");
    const Operand op_pool = pool_operand(a.kv_pool, a.n_pool_blocks, g.block_floats, align, info, op);
    const Operand ot = table_operand(a.block_table, rows_end, a.block_tokens, align, op);
    const Operand oo = operand(a.out, a.n_tokens, std::uint64_t{a.n_head} * a.head_dim * k_f32, Access::Floats, align,
                               op, "out");
    const Operand os = status_operand(a.status, align, op);
    require_disjoint(oo, "out", {{&oq, "q"}, {&op_pool, "kv_pool"}, {&ot, "block_table"}, {&os, "status"}}, op);
    require_disjoint(os, "status", {{&oq, "q"}, {&op_pool, "kv_pool"}, {&ot, "block_table"}}, op);
    struct Push {
        std::uint32_t n_head, n_kv_head, head_dim, n_tokens, q_offset;
        float scale;
        std::uint32_t n_pool_blocks, block_floats, layer_off, bt, kv_dim, q_off, q_stride, q_head_stride, pool_off,
            tab_off, o_off, o_stride, st_off;
    } push{a.n_head,       a.n_kv_head,     a.head_dim,  a.n_tokens,      a.q_offset,
           a.scale,        a.n_pool_blocks, g.block_floats, g.layer_off,  a.block_tokens,
           static_cast<std::uint32_t>(kv_dim), oq.off,  oq.stride,       qhs,          op_pool.off,
           ot.off,         oo.off,          oo.stride,   os.off};
    const Kernel& k = kernel("attention", 5, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{oq.binding, op_pool.binding, ot.binding, oo.binding, os.binding};
    const GroupCount groups = grid_1d(items, info.max_workgroup_count[0], info.max_workgroup_count[1]);
    zero_status(stream, a.status);
    stream.dispatch(k, bindings, push, groups, 0b11000);  // out + status written
}

}  // namespace halo::vulkan
