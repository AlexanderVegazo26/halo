// Layer operators of the qwen35 forward beyond the first four (WS-F2 V1): CONV1D_SHORT,
// GATED_NORM, PARTIAL_ROPE (head_stride), SWIGLU, MUL_SIGMOID, ADD, ADD+RMS_NORM and the
// fused GDN gates. Host side: validate every view (ops_internal.h), check aliasing, then
// record one dispatch. Semantics: halo::cpu (see ops.h).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "ops_internal.h"
#include "vk_util.h"

namespace halo::vulkan {

using detail::Access;
using detail::checked_mul;
using detail::k_f32;
using detail::operand;
using detail::Operand;
using detail::overlaps;
using detail::require_disjoint;
using detail::require_disjoint_or_exact;
using detail::to_u32;

namespace {

constexpr std::uint32_t k_max_conv_kernel = 8;
constexpr std::uint32_t k_eltwise_wg = 256;

}  // namespace

std::vector<float> rope_cos_sin_table(std::span<const std::int32_t> positions, std::uint32_t rot_dims, float theta) {
    HALO_CHECK(rot_dims > 0 && rot_dims % 2 == 0, ErrorCode::Kernel, "rope: rot_dims {} must be even and > 0",
               rot_dims);
    const std::size_t half = rot_dims / 2;
    // Exactly cpu::rope_inv_freq and cpu::partial_rope_neox's angle / cos / sin.
    std::vector<float> inv(half);
    for (std::size_t i = 0; i < half; ++i) {
        const float e = static_cast<float>(2 * i) / static_cast<float>(rot_dims);
        inv[i] = 1.0f / std::pow(theta, e);
    }
    std::vector<float> t(positions.size() * rot_dims);
    for (std::size_t p = 0; p < positions.size(); ++p) {
        const float pos = static_cast<float>(positions[p]);
        for (std::size_t i = 0; i < half; ++i) {
            const float angle = pos * inv[i];
            t[p * rot_dims + i] = static_cast<float>(std::cos(static_cast<double>(angle)));
            t[p * rot_dims + half + i] = static_cast<float>(std::sin(static_cast<double>(angle)));
        }
    }
    return t;
}

void Ops::causal_conv1d_silu(Stream& stream, const Conv1dArgs& a) {
    constexpr std::string_view op = "causal_conv1d_silu";
    HALO_CHECK(a.n_tokens > 0 && a.channels > 0, ErrorCode::Kernel, "{}: empty shape T={} C={}", op, a.n_tokens,
               a.channels);
    HALO_CHECK(a.kernel >= 1 && a.kernel <= k_max_conv_kernel, ErrorCode::Kernel, "{}: kernel {} outside 1..{}", op,
               a.kernel, k_max_conv_kernel);
    const std::uint32_t hist = a.kernel - 1;
    HALO_CHECK(a.n_slots == 0 || (hist > 0 && !a.state_slots.empty()), ErrorCode::Kernel,
               "{}: n_slots={} needs kernel >= 2 and a state_slots view", op, a.n_slots);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t row = std::uint64_t{a.channels} * k_f32;
    const Operand ox = operand(a.x, a.n_tokens, row, Access::Floats, align, op, "x");
    const Operand ow = operand(a.weight, a.channels, std::uint64_t{a.kernel} * k_f32, Access::Floats, align, op, "weight");
    const Operand oo = operand(a.out, a.n_tokens, row, Access::Floats, align, op, "out");
    require_disjoint_or_exact(oo, "out", ox, "x", op);
    require_disjoint(oo, "out", {{&ow, "weight"}}, op);
    std::optional<Operand> os;
    if (hist > 0) {
        os = operand(a.conv_state, hist, row, Access::Floats, align, op, "conv_state");
        require_disjoint(*os, "conv_state", {{&ox, "x"}, {&ow, "weight"}, {&oo, "out"}}, op);
    }
    const std::uint64_t used = std::min<std::uint64_t>(a.n_tokens, a.n_slots);
    std::optional<Operand> osl;
    if (used > 0) {
        osl = operand(a.state_slots, 1, checked_mul(checked_mul(used, hist, op), row, op), Access::Floats, align, op,
                      "state_slots (written slots)");
        require_disjoint(*osl, "state_slots", {{&ox, "x"}, {&ow, "weight"}, {&oo, "out"}, {os ? &*os : nullptr, "conv_state"}}, op);
    }
    const std::uint32_t wg = k_eltwise_wg;
    struct Push {
        std::uint32_t n_tokens, channels, kernel, n_slots, x_off, x_stride, w_off, w_stride, st_off, st_stride, o_off,
            o_stride, sl_off;
    } push{a.n_tokens, a.channels, a.kernel,  static_cast<std::uint32_t>(used),
           ox.off,     ox.stride,  ow.off,    ow.stride,
           os ? os->off : 0u,      os ? os->stride : 0u, oo.off, oo.stride, osl ? osl->off : 0u};
    const Kernel& k = kernel("conv1d_silu", 5, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    // Placeholders (never accessed): x for an absent conv_state, out for absent slots.
    const std::array bindings{ox.binding, ow.binding, os ? os->binding : ox.binding, oo.binding,
                              osl ? osl->binding : oo.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push,
                    grid_1d((std::uint64_t{a.channels} + wg - 1) / wg, info.max_workgroup_count[0],
                            info.max_workgroup_count[1]));
}

void Ops::gated_rms_norm(Stream& stream, const GatedNormArgs& a) {
    constexpr std::string_view op = "gated_rms_norm";
    HALO_CHECK(a.rows > 0 && a.cols > 0, ErrorCode::Kernel, "{}: empty shape {}x{}", op, a.rows, a.cols);
    HALO_CHECK(a.eps >= 0.0f, ErrorCode::Kernel, "{}: negative eps", op);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t row = std::uint64_t{a.cols} * k_f32;
    const Operand ox = operand(a.x, a.rows, row, Access::Floats, align, op, "x");
    const Operand oz = operand(a.z, a.rows, row, Access::Floats, align, op, "z");
    const Operand ow = operand(a.w, 1, row, Access::Floats, align, op, "w");
    const Operand oo = operand(a.out, a.rows, row, Access::Floats, align, op, "out");
    require_disjoint_or_exact(oo, "out", ox, "x", op);
    require_disjoint_or_exact(oo, "out", oz, "z", op);
    require_disjoint(oo, "out", {{&ow, "w"}}, op);
    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push {
        std::uint32_t rows, cols;
        float eps;
        std::uint32_t x_off, x_stride, z_off, z_stride, w_off, y_off, y_stride;
    } push{a.rows, a.cols, a.eps, ox.off, ox.stride, oz.off, oz.stride, ow.off, oo.off, oo.stride};
    const Kernel& k = kernel("gated_rms_norm", 4, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{ox.binding, oz.binding, ow.binding, oo.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(a.rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

void Ops::partial_rope_neox(Stream& stream, const RopeArgs& a) {
    constexpr std::string_view op = "partial_rope_neox";
    HALO_CHECK(a.n_tokens > 0 && a.n_heads > 0 && a.head_dim > 0, ErrorCode::Kernel, "{}: empty shape", op);
    HALO_CHECK(a.rot_dims > 0 && a.rot_dims % 2 == 0 && a.rot_dims <= a.head_dim, ErrorCode::Kernel,
               "{}: rot_dims {} must be even, > 0 and <= head_dim {}", op, a.rot_dims, a.head_dim);
    const std::uint32_t hs = a.head_stride == 0 ? a.head_dim : a.head_stride;
    HALO_CHECK(hs >= a.head_dim, ErrorCode::Kernel, "{}: head_stride {} < head_dim {}", op, hs, a.head_dim);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    // A row spans from head 0 to the end of the last head.
    const std::uint64_t row_elems =
        detail::checked_add(checked_mul(std::uint64_t{a.n_heads} - 1, hs, op), a.head_dim, op);
    const Operand ox = operand(a.x, a.n_tokens, row_elems * k_f32, Access::Floats, align, op, "x");
    const Operand oc = operand(a.cos_sin, a.n_tokens, std::uint64_t{a.rot_dims} * k_f32, Access::Floats, align, op,
                               "cos_sin");
    require_disjoint(ox, "x", {{&oc, "cos_sin"}}, op);
    (void)to_u32(std::uint64_t{a.n_heads} * (a.rot_dims / 2), "partial_rope_neox pairs per token");
    const std::uint32_t wg = k_eltwise_wg;
    struct Push {
        std::uint32_t n_tokens, n_heads, head_stride, rot, x_off, x_stride, cs_off, cs_stride;
    } push{a.n_tokens, a.n_heads, hs, a.rot_dims, ox.off, ox.stride, oc.off, oc.stride};
    const Kernel& k = kernel("rope_neox", 2, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{ox.binding, oc.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(a.n_tokens, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

void Ops::eltwise(Stream& stream, const EltwiseArgs& a, std::uint32_t op_code, std::string_view op) {
    HALO_CHECK(a.rows > 0 && a.cols > 0, ErrorCode::Kernel, "{}: empty shape {}x{}", op, a.rows, a.cols);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t row = std::uint64_t{a.cols} * k_f32;
    const Operand oa = operand(a.a, a.rows, row, Access::Floats, align, op, "a");
    const Operand ob = operand(a.b, a.rows, row, Access::Floats, align, op, "b");
    const Operand oo = operand(a.out, a.rows, row, Access::Floats, align, op, "out");
    require_disjoint_or_exact(oo, "out", oa, "a", op);
    require_disjoint_or_exact(oo, "out", ob, "b", op);
    const std::uint32_t wg = k_eltwise_wg;
    struct Push {
        std::uint32_t rows, cols, a_off, a_stride, b_off, b_stride, o_off, o_stride;
    } push{a.rows, a.cols, oa.off, oa.stride, ob.off, ob.stride, oo.off, oo.stride};
    const Kernel& k = kernel("eltwise", 3, sizeof(Push), {{0, wg}, {1, op_code}}, {wg, 1, 1});
    const std::array bindings{oa.binding, ob.binding, oo.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(a.rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

void Ops::add(Stream& stream, const EltwiseArgs& args) { eltwise(stream, args, 0, "add"); }
void Ops::swiglu(Stream& stream, const EltwiseArgs& args) { eltwise(stream, args, 1, "swiglu"); }
void Ops::mul_sigmoid(Stream& stream, const EltwiseArgs& args) { eltwise(stream, args, 2, "mul_sigmoid"); }

void Ops::add_rms_norm(Stream& stream, const AddRmsNormArgs& a) {
    constexpr std::string_view op = "add_rms_norm";
    HALO_CHECK(a.rows > 0 && a.cols > 0, ErrorCode::Kernel, "{}: empty shape {}x{}", op, a.rows, a.cols);
    HALO_CHECK(a.eps >= 0.0f, ErrorCode::Kernel, "{}: negative eps", op);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t row = std::uint64_t{a.cols} * k_f32;
    const Operand oa = operand(a.a, a.rows, row, Access::Floats, align, op, "a");
    const Operand ob = operand(a.b, a.rows, row, Access::Floats, align, op, "b");
    const Operand oh = operand(a.h, a.rows, row, Access::Floats, align, op, "h");
    const Operand ow = operand(a.w, 1, row, Access::Floats, align, op, "w");
    const Operand oy = operand(a.y, a.rows, row, Access::Floats, align, op, "y");
    require_disjoint_or_exact(oh, "h", oa, "a", op);
    require_disjoint_or_exact(oh, "h", ob, "b", op);
    require_disjoint(oh, "h", {{&ow, "w"}}, op);
    require_disjoint(oy, "y", {{&oa, "a"}, {&ob, "b"}, {&oh, "h"}, {&ow, "w"}}, op);
    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push {
        std::uint32_t rows, cols;
        float eps;
        std::uint32_t a_off, a_stride, b_off, b_stride, h_off, h_stride, w_off, y_off, y_stride;
    } push{a.rows, a.cols, a.eps, oa.off, oa.stride, ob.off, ob.stride, oh.off, oh.stride, ow.off, oy.off, oy.stride};
    const Kernel& k = kernel("add_rms_norm", 5, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{oa.binding, ob.binding, oh.binding, ow.binding, oy.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(a.rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

void Ops::gdn_gates(Stream& stream, const GdnGateArgs& a) {
    constexpr std::string_view op = "gdn_gates";
    HALO_CHECK(a.rows > 0 && a.n_v > 0, ErrorCode::Kernel, "{}: empty shape {}x{}", op, a.rows, a.n_v);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t row = std::uint64_t{a.n_v} * k_f32;
    const Operand ob = operand(a.b, a.rows, row, Access::Floats, align, op, "b");
    const Operand oa = operand(a.a, a.rows, row, Access::Floats, align, op, "a");
    const Operand od = operand(a.dt_bias, 1, row, Access::Floats, align, op, "dt_bias");
    const Operand os = operand(a.ssm_a, 1, row, Access::Floats, align, op, "ssm_a");
    const Operand obt = operand(a.beta, a.rows, row, Access::Floats, align, op, "beta");
    const Operand og = operand(a.g, a.rows, row, Access::Floats, align, op, "g");
    require_disjoint_or_exact(obt, "beta", ob, "b", op);
    require_disjoint(obt, "beta", {{&oa, "a"}, {&od, "dt_bias"}, {&os, "ssm_a"}, {&og, "g"}}, op);
    require_disjoint_or_exact(og, "g", oa, "a", op);
    require_disjoint(og, "g", {{&ob, "b"}, {&od, "dt_bias"}, {&os, "ssm_a"}}, op);
    const std::uint32_t wg = k_eltwise_wg;
    struct Push {
        std::uint32_t rows, n_v, b_off, b_stride, a_off, a_stride, dt_off, sa_off, beta_off, beta_stride, g_off,
            g_stride;
    } push{a.rows, a.n_v, ob.off, ob.stride, oa.off, oa.stride, od.off, os.off, obt.off, obt.stride, og.off, og.stride};
    const Kernel& k = kernel("gdn_gates", 6, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{ob.binding, oa.binding, od.binding, os.binding, obt.binding, og.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(a.rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

}  // namespace halo::vulkan
