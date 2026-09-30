#include "halo/backends/vulkan/ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <format>
#include <initializer_list>
#include <utility>

#include "halo/backends/vulkan/context.h"
#include "halo/core/error.h"
#include "ops_internal.h"
#include "vk_util.h"

namespace halo::vulkan {

using detail::checked_add;
using detail::checked_mul;
using detail::to_u32;

namespace {

using detail::Access;
using detail::Operand;
using detail::operand;
using detail::overlaps;
using detail::require_disjoint;
using detail::k_f32;
// Elements handled per argmax pass-1 thread (chunk = workgroup * this).
constexpr std::uint32_t k_argmax_per_thread = 16;
// Words per argmax partial / result: {index, value bits, nan flag}.
constexpr std::uint32_t k_argmax_words = 3;
// Vectors per gemv dispatch (matvec_quant_main.glsl MAX_VEC).
constexpr std::uint32_t k_gemv_max_vec = 8;
// HALO_SUBGROUP_REDUCE=1 (default off): batched (n_vec > 1) matvecs of Q4_K / Q5_K / Q6_K / IQ4_XS use the
// matvec_<type>_sg shaders (subgroupAdd reduction, NV specialized to the dispatch's vector count) when the
// device offers arithmetic subgroup ops in compute shaders with subgroupSize >= 32 (the shaders stay correct
// for any workgroup / subgroup ratio above that). Results differ from the tree shaders in reduction order.
bool subgroup_reduce_enabled(const DeviceInfo& info) {
    static const bool requested = [] {
        const char* e = std::getenv("HALO_SUBGROUP_REDUCE");
        return e != nullptr && *e != '\0' && std::string_view(e) != "0";
    }();
    return requested && info.subgroup_size >= 32 && (info.subgroup_ops & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0 &&
           (info.subgroup_stages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
}
// Largest chunk of the chunked GDN (the HIP backend's cap too; qwen35 prefill uses 64).
constexpr std::uint32_t k_gdn_max_chunk = 64;

}  // namespace

std::uint64_t matvec_row_bytes(DType t, std::uint32_t cols) {
    auto blocks = [&](std::uint32_t qk, std::uint64_t bytes) {
        HALO_CHECK(cols % qk == 0, ErrorCode::Kernel, "matvec: cols={} is not a multiple of {} for this type",
                   cols, qk);
        return std::uint64_t{cols} / qk * bytes;
    };
    switch (t) {
        case DType::F32: return std::uint64_t{cols} * k_f32;
        case DType::Q8_0: return blocks(32, 34);
        case DType::Q4_K: return blocks(256, 144);
        case DType::Q5_K: return blocks(256, 176);
        case DType::Q6_K: return blocks(256, 210);
        case DType::IQ4_XS: return blocks(256, 136);
        case DType::IQ4_NL: return blocks(32, 18);
        case DType::Q3_K: return blocks(256, 110);
        case DType::IQ3_S: return blocks(256, 110);
        default:
            throw_error(ErrorCode::Unsupported,
                        "Vulkan matvec: weight type id {} not supported (F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, "
                        "IQ4_NL, Q3_K, IQ3_S)",
                        static_cast<std::uint32_t>(t));
    }
}

ArgmaxResult decode_argmax(std::span<const std::uint32_t, 3> words) {
    HALO_CHECK(words[2] == 0, ErrorCode::Kernel, "argmax: NaN in the logits (Vulkan kernel flag)");
    HALO_CHECK(words[0] != k_argmax_none, ErrorCode::Kernel, "argmax: no result (empty input)");
    return ArgmaxResult{words[0], std::bit_cast<float>(words[1])};
}

ArgmaxResult read_argmax(const Buffer& result, std::uint64_t offset) {
    std::array<std::uint32_t, 3> w{};
    result.download(std::span<std::uint32_t>(w), offset);
    return decode_argmax(w);
}

Ops::Ops(std::shared_ptr<Context> ctx, OpsOptions options) : ctx_(std::move(ctx)), options_(options) {
    HALO_CHECK(ctx_ != nullptr, ErrorCode::Kernel, "Ops: null context");
    const DeviceInfo& info = ctx_->info();
    const std::uint32_t rw = options_.reduce_workgroup;
    HALO_CHECK(std::has_single_bit(rw) && rw >= 32 && rw <= 1024 && rw <= info.max_workgroup_size[0] &&
                   rw <= info.max_workgroup_invocations,
               ErrorCode::Kernel, "Ops: reduce_workgroup={} must be a power of two in [32, min(1024, device)]",
               rw);
    const std::uint32_t gw = options_.gdn_workgroup;
    HALO_CHECK(gw >= 1 && gw <= info.max_workgroup_size[0] && gw <= info.max_workgroup_invocations,
               ErrorCode::Kernel, "Ops: gdn_workgroup={} outside device limits", gw);
    // gated_delta_rule_decode shared memory: q/k tiles (2 * MAX_DK) + two WG-wide reduction arrays.
    HALO_CHECK(options_.gdn_max_dk >= 1 &&
                   (std::uint64_t{options_.gdn_max_dk} + gw) * 2 * k_f32 <= info.max_shared_memory,
               ErrorCode::Kernel, "Ops: gdn_max_dk={} with gdn_workgroup={} exceeds shared memory ({} bytes)",
               options_.gdn_max_dk, gw, info.max_shared_memory);
    // argmax: three WG-wide shared arrays (value, index, NaN flag); rms_norm/matvec use one.
    HALO_CHECK(std::uint64_t{rw} * 12 <= info.max_shared_memory, ErrorCode::Kernel,
               "Ops: reduce_workgroup={} exceeds shared memory", rw);
    // Tuning knob for the gemv/matvec workgroup size (specialization constant 0).
    if (const char* env = std::getenv("HALO_VK_GEMV_WG"); env != nullptr && *env != '\0')
        options_.gemv_workgroup = static_cast<std::uint32_t>(std::stoul(env));
    if (options_.gemv_workgroup != 0)
        HALO_CHECK(std::has_single_bit(options_.gemv_workgroup) && options_.gemv_workgroup >= 32 &&
                       options_.gemv_workgroup <= info.max_workgroup_size[0],
                   ErrorCode::Kernel, "Ops: gemv_workgroup={} must be a power of two >= 32",
                   options_.gemv_workgroup);
    const std::uint32_t aw = options_.attention_workgroup;
    HALO_CHECK(std::has_single_bit(aw) && aw >= 32 && aw <= 1024 && aw <= info.max_workgroup_size[0] &&
                   aw <= info.max_workgroup_invocations,
               ErrorCode::Kernel, "Ops: attention_workgroup={} must be a power of two in [32, min(1024, device)]",
               aw);
    // attention shared memory: q tile (256 floats) + three WG-wide arrays (scores, reduction, row bases).
    HALO_CHECK((256 + 3 * std::uint64_t{aw}) * 4 <= info.max_shared_memory, ErrorCode::Kernel,
               "Ops: attention_workgroup={} exceeds shared memory ({} bytes)", aw, info.max_shared_memory);
}

const Kernel& Ops::kernel(const std::string& shader, std::uint32_t num_buffers, std::uint32_t push_bytes,
                          std::vector<SpecConstant> spec, std::array<std::uint32_t, 3> local) {
    std::string key = shader;
    for (const SpecConstant& s : spec) key += std::format(":{}={}", s.id, s.value);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) return *it->second;
    KernelDesc d;
    d.shader = &find_shader(shader);
    d.num_buffers = num_buffers;
    d.push_constant_bytes = push_bytes;
    d.spec_constants = std::move(spec);
    d.local_size = local;
    auto k = std::make_unique<Kernel>(ctx_, std::move(d));
    const Kernel& ref = *k;
    kernels_.emplace(std::move(key), std::move(k));
    return ref;
}

void Ops::rms_norm(Stream& stream, const BufferView& x, const BufferView& w, const BufferView& y,
                   std::uint32_t rows, std::uint32_t cols, float eps) {
    constexpr std::string_view op = "rms_norm";
    HALO_CHECK(rows > 0 && cols > 0, ErrorCode::Kernel, "rms_norm: empty shape {}x{}", rows, cols);
    HALO_CHECK(eps >= 0.0f, ErrorCode::Kernel, "rms_norm: negative eps");
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const std::uint64_t row_bytes = std::uint64_t{cols} * k_f32;
    const Operand ox = operand(x, rows, row_bytes, Access::Floats, align, op, "x");
    const Operand ow = operand(w, 1, row_bytes, Access::Floats, align, op, "w");
    const Operand oy = operand(y, rows, row_bytes, Access::Floats, align, op, "y");
    require_disjoint(oy, "y", {{&ox, "x"}, {&ow, "w"}}, op);

    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push {
        std::uint32_t rows, cols;
        float eps;
        std::uint32_t x_off, x_stride, w_off, y_off, y_stride;
    } push{rows, cols, eps, ox.off, ox.stride, ow.off, oy.off, oy.stride};
    const Kernel& k = kernel("rms_norm", 3, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{ox.binding, ow.binding, oy.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(rows, info.max_workgroup_count[0], info.max_workgroup_count[1]),
                    0b100);  // y written
}

void Ops::matvec(Stream& stream, DType wtype, const BufferView& w, const BufferView& x, const BufferView& y,
                 std::uint32_t rows, std::uint32_t cols) {
    gemv_impl(stream, GemvArgs{wtype, w, x, y, rows, cols, 1}, "matvec");
}

void Ops::gemv(Stream& stream, const GemvArgs& args) { gemv_impl(stream, args, "gemv"); }

void Ops::gemv_repacked(Stream& stream, const GemvArgs& args) { gemv_impl(stream, args, "gemv(repacked)", true); }

void Ops::gemv_impl(Stream& stream, const GemvArgs& a, std::string_view op, bool repacked) {
    HALO_CHECK(a.rows > 0 && a.cols > 0 && a.n_vec > 0, ErrorCode::Kernel, "{}: empty shape {}x{} n_vec={}", op,
               a.rows, a.cols, a.n_vec);
    // The repacked layout (halo/tensor/repack.h) keeps the row footprint of the GGUF layout, so
    // row_bytes and every extent check below are unchanged; only the shader differs ("_rp").
    const bool normed = !a.norm_w.empty();
    HALO_CHECK(!normed || (!repacked && a.n_vec == 1 && a.cols <= 16384 &&
                           (a.wtype == DType::Q4_K || a.wtype == DType::Q5_K || a.wtype == DType::Q6_K ||
                            a.wtype == DType::IQ4_XS)),
               ErrorCode::Unsupported,
               "{}: fused norm needs a non-repacked Q4_K/Q5_K/Q6_K/IQ4_XS weight, n_vec == 1 and cols <= 16384", op);
    HALO_CHECK(!repacked || a.wtype == DType::Q5_K || a.wtype == DType::Q6_K || a.wtype == DType::IQ4_XS,
               ErrorCode::Unsupported, "{}: no repacked matvec for weight type id {} (Q5_K, Q6_K, IQ4_XS)", op,
               static_cast<std::uint32_t>(a.wtype));
    const std::uint64_t row_bytes = matvec_row_bytes(a.wtype, a.cols);
    const std::string shader = detail::weight_shader("matvec", a.wtype) + (repacked ? "_rp" : "");
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    // F32 weights are indexed by element, quantized ones by byte (read as 32-bit words).
    const Operand ow =
        operand(a.w, a.rows, row_bytes, a.wtype == DType::F32 ? Access::Floats : Access::Words, align, op, "W");
    const Operand ox = operand(a.x, a.n_vec, std::uint64_t{a.cols} * k_f32, Access::Floats, align, op, "x");
    const Operand oy = operand(a.y, a.n_vec, std::uint64_t{a.rows} * k_f32, Access::Floats, align, op, "y");
    require_disjoint(oy, "y", {{&ow, "W"}, {&ox, "x"}}, op);
    // The _rp shaders load uvec4 / uvec2 at plane offsets relative to each row start: every
    // row (binding remainder + r * stride) must be 16-byte aligned. A single row has no stride.
    HALO_CHECK(!repacked || (ow.off % 16 == 0 && (a.rows == 1 || ow.stride % 16 == 0)), ErrorCode::Kernel,
               "{}: repacked W needs a 16-byte aligned view offset ({}) and row stride ({})", op, ow.off, ow.stride);

    // A W larger than one binding (e.g. the 248320-row embedding / LM head on a device with
    // a small maxStorageBufferRange) runs as row slabs; every slab writes its own rows of y.
    const std::uint64_t limit = detail::binding_limit(options_, ctx_->info());
    if (ow.binding.range > limit) {
        const std::uint64_t ws = (a.rows == 1 || a.w.row_stride == 0) ? row_bytes : a.w.row_stride;
        const std::uint64_t ys = (a.n_vec == 1 || a.y.row_stride == 0) ? std::uint64_t{a.rows} * k_f32 : a.y.row_stride;
        const std::uint64_t per = detail::rows_per_slab(limit, align, ws, row_bytes, op);
        for (std::uint64_t r0 = 0; r0 < a.rows; r0 += per) {
            const auto n = static_cast<std::uint32_t>(std::min<std::uint64_t>(per, a.rows - r0));
            GemvArgs s = a;
            s.rows = n;
            s.w = BufferView(*a.w.buffer, a.w.offset + r0 * ws, (n - 1) * ws + row_bytes, ws);
            s.y = BufferView(*a.y.buffer, a.y.offset + r0 * k_f32, (a.n_vec - 1) * ys + std::uint64_t{n} * k_f32, ys);
            gemv_impl(stream, s, op, repacked);
        }
        return;
    }

    // Prefill: more than one matvec dispatch's worth of vectors on a K-quant weight runs as a
    // tiled cooperative-matrix GEMM (fp16 tiles, fp32 accumulate; shaders/common/matmul_cm_main.glsl).
    // Opt-in (HALO_COOPMAT=1, and the device must have enabled the extension at context creation:
    // Context::coopmat_enabled()); numerics differ from the fp32 matvec, so never the default.
    // (Never on repacked weights: matmul_cm reads the raw GGUF blocks; those run the _rp matvec.)
    if (!repacked && a.n_vec > k_gemv_max_vec && ctx_->coopmat_enabled() &&
        (a.wtype == DType::Q4_K || a.wtype == DType::Q5_K || a.wtype == DType::Q6_K)) {
        struct CmPush {
            std::uint32_t rows, cols, n_vec, w_off, w_stride, x_off, x_stride, y_off, y_stride;
        };
        constexpr std::uint32_t cm_wg = 64;  // matmul_cm_main.glsl WG (constant_id 0)
        constexpr std::uint32_t cm_tile = 32;  // TM = TN in matmul_cm_main.glsl: one tile per subgroup
        const Kernel& kcm = kernel(detail::weight_shader("matmul_cm", a.wtype), 3, sizeof(CmPush), {{0, cm_wg}},
                                   {cm_wg, 1, 1});
        const std::uint64_t tiles = std::uint64_t{(a.n_vec + cm_tile - 1) / cm_tile} * ((a.rows + cm_tile - 1) / cm_tile);
        const DeviceInfo& cm_info = ctx_->info();
        // One workgroup per tile; a workgroup with several subgroups just runs extra ones past the grid.
        const GroupCount cm_groups = grid_1d(tiles, cm_info.max_workgroup_count[0], cm_info.max_workgroup_count[1]);
        const std::array cm_bindings{ow.binding, ox.binding, oy.binding};
        const CmPush cm_push{a.rows, a.cols, a.n_vec, ow.off, ow.stride, ox.off, ox.stride, oy.off, oy.stride};
        stream.dispatch(kcm, cm_bindings, cm_push, cm_groups, 0b100);  // y written
        return;
    }

    // Workgroup size. Types with a llama-style SWAR matvec main (q4_k/q5_k/q6_k/q3_k/
    // iq4_xs/iq4_nl/iq3_s; see shaders/common/matvec_<type>_main.glsl) run fastest at one
    // wave (64): 16 (8 for iq*) threads per block -> 4 (8) blocks per iteration, and both
    // ffn shapes (20/68 blocks) divide by 4.
    // Types on the shared matvec_quant_main.glsl keep reduce_workgroup (256 measured best
    // there). gemv_workgroup (HALO_VK_GEMV_WG) overrides for tuning.
    const bool swar_main = a.wtype == DType::Q4_K || a.wtype == DType::Q5_K || a.wtype == DType::Q6_K ||
                           a.wtype == DType::IQ4_XS || a.wtype == DType::IQ4_NL || a.wtype == DType::Q3_K ||
                           a.wtype == DType::IQ3_S;
    const std::uint32_t wg = options_.gemv_workgroup != 0 ? options_.gemv_workgroup
                             : swar_main                ? 64
                                                        : options_.reduce_workgroup;
    if (normed) {
        // Fused rms_norm + decode matvec (n_vec == 1): x is the raw row; the shader stages
        // (x * inv_rms) * norm_w in shared memory (XCOLS = cols floats) and runs the unmodified main.
        const Operand onw = operand(a.norm_w, 1, std::uint64_t{a.cols} * k_f32, Access::Floats, align, op, "norm_w");
        require_disjoint(oy, "y", {{&onw, "norm_w"}}, op);
        struct NPush {
            std::uint32_t rows, cols, n_vec, w_off, w_stride, x_off, x_stride, y_off, y_stride, xr_off, nw_off;
            float eps;
        };
        const Kernel& kn = kernel(shader + "_normed", 4, sizeof(NPush), {{0, wg}, {1, 0u}, {2, a.cols}}, {wg, 1, 1});
        const std::array nbindings{ow.binding, ox.binding, oy.binding, onw.binding};
        const DeviceInfo& ninfo = ctx_->info();
        // x_off = 0: the shader stages the normalized row at index 0; the raw view remainder is xr_off.
        const NPush npush{a.rows, a.cols, 1, ow.off, ow.stride, 0, 0, oy.off, oy.stride, ox.off, onw.off, a.norm_eps};
        stream.dispatch(kn, nbindings, npush, grid_1d(a.rows, ninfo.max_workgroup_count[0], ninfo.max_workgroup_count[1]),
                        0b100);  // y written
        return;
    }
    struct Push {
        std::uint32_t rows, cols, n_vec, w_off, w_stride, x_off, x_stride, y_off, y_stride;
    };
    // Specialization constant 1 (BATCHED): decode dispatches get a pipeline compiled without
    // the batched accumulator body (matvec_quant_main.glsl).
    const Kernel& k =
        kernel(shader, 3, sizeof(Push), {{0, wg}, {1, a.n_vec == 1 ? 0u : 1u}}, {wg, 1, 1});
    const std::array bindings{ow.binding, ox.binding, oy.binding};
    const DeviceInfo& info = ctx_->info();
    const GroupCount groups = grid_1d(a.rows, info.max_workgroup_count[0], info.max_workgroup_count[1]);
    // HALO_SUBGROUP_REDUCE=1: batched chunks (n > 1) of the four hot K-quant types run the *_sg shader
    // (subgroupAdd reduction, spec constant 2 = the chunk's vector count so the vector loops unroll).
    const bool sg_reduce = !repacked && a.n_vec > 1 &&
                           (a.wtype == DType::Q4_K || a.wtype == DType::Q5_K || a.wtype == DType::Q6_K ||
                            a.wtype == DType::IQ4_XS) &&
                           subgroup_reduce_enabled(info);
    // Up to k_gemv_max_vec vectors per dispatch (matvec_quant_main.glsl MAX_VEC). Offsets of
    // later chunks stay inside the index range operand() checked for the whole extent.
    for (std::uint32_t t0 = 0; t0 < a.n_vec; t0 += k_gemv_max_vec) {
        const std::uint32_t n = std::min(k_gemv_max_vec, a.n_vec - t0);
        const Kernel& kc = (sg_reduce && n > 1) ? kernel(shader + "_sg", 3, sizeof(Push), {{0, wg}, {2, n}}, {wg, 1, 1}) : k;
        const Push push{a.rows,
                        a.cols,
                        n,
                        ow.off,
                        ow.stride,
                        ox.off + t0 * ox.stride,
                        ox.stride,
                        oy.off + t0 * oy.stride,
                        oy.stride};
        stream.dispatch(kc, bindings, push, groups, 0b100);  // y written
    }
}

void Ops::gdn_impl(Stream& stream, const GdnDecodeArgs& a, const GdnChunkedArgs* chunked) {
    const std::string_view op = chunked == nullptr ? "gated_delta_rule_decode" : "gated_delta_rule_chunked";
    HALO_CHECK(a.n_v > 0 && a.n_k > 0 && a.d_k > 0 && a.d_v > 0 && a.n_tokens > 0, ErrorCode::Kernel,
               "{}: empty dims n_v={} n_k={} d_k={} d_v={} T={}", op, a.n_v, a.n_k, a.d_k, a.d_v, a.n_tokens);
    HALO_CHECK(a.n_v % a.n_k == 0, ErrorCode::Kernel, "{}: n_v={} is not a multiple of n_k={}", op, a.n_v, a.n_k);
    HALO_CHECK(a.d_k <= options_.gdn_max_dk, ErrorCode::Kernel, "{}: d_k={} > gdn_max_dk={}", op, a.d_k,
               options_.gdn_max_dk);
    const bool ring = a.ring.has_value();
    if (ring) {
        HALO_CHECK(a.ring->p >= 2 && a.ring->live < a.ring->p, ErrorCode::Kernel,
                   "{}: state ring p={} live={} is out of range", op, a.ring->p, a.ring->live);
        HALO_CHECK(a.n_slots <= a.ring->p - 1, ErrorCode::Kernel, "{}: {} slots requested, ring p - 1 = {}", op,
                   a.n_slots, a.ring->p - 1);
        HALO_CHECK(!a.state_out.has_value() && a.state_slots.empty(), ErrorCode::Kernel,
                   "{}: with a state ring, state_out/state_slots are derived from the slab and must be unset", op);
    } else {
        HALO_CHECK(a.n_slots == 0 || !a.state_slots.empty(), ErrorCode::Kernel,
                   "{}: n_slots={} without a state_slots view", op, a.n_slots);
    }
    const float q_scale = a.q_scale.value_or(1.0f / std::sqrt(static_cast<float>(a.d_k)));
    HALO_CHECK(std::isfinite(q_scale), ErrorCode::Kernel, "{}: q_scale {} is not finite", op, q_scale);

    const std::uint64_t T = a.n_tokens;
    const std::uint64_t qk_row = checked_mul(checked_mul(a.n_k, a.d_k, op), k_f32, op);
    const std::uint64_t v_row = checked_mul(checked_mul(a.n_v, a.d_v, op), k_f32, op);
    const std::uint64_t head_row = std::uint64_t{a.n_v} * k_f32;
    const std::uint64_t s_bytes = checked_mul(checked_mul(checked_mul(a.n_v, a.d_k, op), a.d_v, op), k_f32, op);
    // Written slots: s < min(T, n_slots); the region checked is the written one.
    const std::uint64_t used_slots = std::min<std::uint64_t>(T, a.n_slots);

    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const Operand oq = operand(a.q, T, qk_row, Access::Floats, align, op, "q");
    const Operand ok = operand(a.k, T, qk_row, Access::Floats, align, op, "k");
    const Operand ov = operand(a.v, T, v_row, Access::Floats, align, op, "v");
    const Operand og = operand(a.g, T, head_row, Access::Floats, align, op, "g");
    const Operand ob = operand(a.beta, T, head_row, Access::Floats, align, op, "beta");
    const Operand oo = operand(a.out, T, v_row, Access::Floats, align, op, "out");
    // Ring: `state` is the whole slab of p states; the kernel derives the input, output and
    // slot addresses from ring {p, live} (one binding for all three roles).
    const Operand oslab = ring ? operand(a.state, a.ring->p, s_bytes, Access::Floats, align, op, "state slab")
                               : Operand{};
    const Operand oin = ring ? oslab : operand(a.state, 1, s_bytes, Access::Floats, align, op, "state (input region)");
    const Operand osout = ring ? oslab
                               : operand(a.state_out.value_or(a.state), 1, s_bytes, Access::Floats, align, op,
                                         "state_out (output region)");
    std::optional<Operand> osl;
    if (!ring && used_slots > 0) {
        osl = operand(a.state_slots, 1, checked_mul(used_slots, s_bytes, op), Access::Floats, align, op,
                      "state_slots (written slots)");
    }

    const std::initializer_list<std::pair<const Operand*, std::string_view>> inputs{
        {&oq, "q"}, {&ok, "k"}, {&ov, "v"}, {&og, "g"}, {&ob, "beta"}};
    require_disjoint(oo, "out", inputs, op);
    require_disjoint(oo, "out", {{&oin, "state"}, {ring ? nullptr : &osout, "state_out"}, {osl ? &*osl : nullptr, "state_slots"}}, op);
    if (ring) {
        // The slab is one buffer by construction (slots share it); it must only be disjoint
        // from the inputs and out.
        require_disjoint(oslab, "state slab", inputs, op);
    } else {
        require_disjoint(osout, "state_out", inputs, op);
        // The input and output state regions are identical (in place) or disjoint.
        HALO_CHECK(!overlaps(oin, osout) || oin.begin == osout.begin, ErrorCode::Kernel,
                   "{}: input state bytes [{}, {}) and output state bytes [{}, {}) partially overlap", op, oin.begin,
                   oin.end, osout.begin, osout.end);
        if (osl) {
            require_disjoint(*osl, "state_slots", inputs, op);
            require_disjoint(*osl, "state_slots", {{&oin, "state"}, {&osout, "state_out"}}, op);
        }
    }

    const DeviceInfo& info = ctx_->info();
    const std::uint32_t wg = options_.gdn_workgroup;
    // Decode: split each head's d_v columns over workgroups of `dv_tile` columns (opt-in:
    // HALO_GDN_DV_TILE=32, default 0 = one group per head; unmeasured). The chunked kernel is one group per head.
    std::uint32_t dv_tile = a.d_v;
    if (chunked == nullptr) {
        static const std::uint32_t want = [] {
            const char* e = std::getenv("HALO_GDN_DV_TILE");
            return e ? static_cast<std::uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
        }();
        if (want > 0 && want < a.d_v) dv_tile = want;
    }
    // Opt-in register-resident decode variant (HALO_GDN_REG=1, default off; unmeasured): one thread
    // per d_v column holding its d_k state rows in registers, so the state is read/written once per
    // dispatch. Needs d_k <= k_gdn_reg_max_dk (register pressure) and dv_tile <= workgroup size.
    constexpr std::uint32_t k_gdn_reg_max_dk = 128;
    static const bool gdn_reg_env = [] {
        const char* e = std::getenv("HALO_GDN_REG");
        return e != nullptr && e[0] == '1';
    }();
    const bool use_reg = chunked == nullptr && gdn_reg_env && a.d_k <= k_gdn_reg_max_dk;
    if (use_reg) dv_tile = std::min(dv_tile, wg);
    const std::uint32_t dv_tiles = (a.d_v + dv_tile - 1) / dv_tile;
    const GroupCount groups = grid_1d(chunked == nullptr ? a.n_v * dv_tiles : a.n_v, info.max_workgroup_count[0],
                                      info.max_workgroup_count[1]);
    const std::uint32_t ring_p = ring ? a.ring->p : 0u;
    const std::uint32_t ring_live = ring ? a.ring->live : 0u;
    if (chunked == nullptr) {
        struct Push {
            std::uint32_t n_v, n_k, d_k, d_v, n_tokens, n_slots, in_off, out_off, slots_off, qk_l2norm;
            float q_scale;
            std::uint32_t q_off, q_stride, k_off, k_stride, v_off, v_stride, g_off, g_stride, b_off, b_stride, o_off,
                o_stride;
            std::uint32_t ring_p, ring_live;  // ADR-001 §5.3; ring_p == 0 = the in/off/slots form
            std::uint32_t dv_tile;            // d_v columns per workgroup
        } push{a.n_v,      a.n_k,      a.d_k,          a.d_v,
               a.n_tokens, static_cast<std::uint32_t>(used_slots), oin.off, ring ? 0u : osout.off,
               osl ? osl->off : 0u, a.qk_l2norm ? 1u : 0u, q_scale,  oq.off,
               oq.stride,  ok.off,     ok.stride,      ov.off,
               ov.stride,  og.off,     og.stride,      ob.off,
               ob.stride,  oo.off,     oo.stride,      ring_p,
               ring_live,  dv_tile};
        static_assert(sizeof(Push) <= 128, "push constants must fit the guaranteed 128-byte minimum");
        const Kernel& k = use_reg ? kernel("gated_delta_rule_decode_reg", 9, sizeof(Push), {{0, wg}, {2, a.d_k}},
                                           {wg, 1, 1})
                                  : kernel("gated_delta_rule_decode", 9, sizeof(Push),
                                           {{0, wg}, {1, options_.gdn_max_dk}}, {wg, 1, 1});
        // Without slots, binding 8 aliases `out` as a placeholder; the shader never writes it.
        // With a ring, bindings 5/6/8 all name the slab.
        const std::array bindings{oq.binding, ok.binding,    ov.binding, og.binding,
                                  ob.binding, oin.binding,   osout.binding, oo.binding,
                                  ring ? oslab.binding : (osl ? osl->binding : oo.binding)};
        stream.dispatch(k, bindings, push, groups, 0b111000000);  // state_out, out, slots written
        return;
    }

    // Chunked form: workspace and status, then the g pre-pass and the main kernel.
    const std::uint32_t cs = chunked->chunk_size;
    HALO_CHECK(cs >= 1 && cs <= k_gdn_max_chunk, ErrorCode::Kernel, "{}: chunk_size {} not in [1, {}]", op, cs,
               k_gdn_max_chunk);
    // gated_delta_rule_chunked.comp shared memory: q/k tiles (2 * MAX_DK), two WG-wide reduction
    // arrays and the 64 x 64 decay table (MAX_CS^2, sized for the host cap k_gdn_max_chunk).
    HALO_CHECK((2 * std::uint64_t{options_.gdn_max_dk} + 2 * std::uint64_t{wg} +
                std::uint64_t{k_gdn_max_chunk} * k_gdn_max_chunk) * k_f32 <= info.max_shared_memory,
               ErrorCode::Kernel, "{}: shared memory ({} bytes) too small for the decay table", op,
               info.max_shared_memory);
    const std::uint64_t ws_bytes = gdn_chunked_workspace_bytes(a, cs);
    const Operand ows = operand(chunked->workspace, 1, ws_bytes, Access::Floats, align, op, "workspace");
    const Operand ost = operand(chunked->status, 1, k_status_bytes, Access::Floats, align, op, "status");
    const std::initializer_list<std::pair<const Operand*, std::string_view>> all{
        {&oq, "q"},     {&ok, "k"},           {&ov, "v"},   {&og, "g"}, {&ob, "beta"}, {&oin, "state"},
        {&osout, "state_out"}, {&oo, "out"}, {osl ? &*osl : nullptr, "state_slots"}};
    require_disjoint(ows, "workspace", all, op);
    require_disjoint(ost, "status", all, op);
    require_disjoint(ost, "status", {{&ows, "workspace"}}, op);
    struct Push {
        std::uint32_t n_v, n_k, d_k, d_v, n_tokens, n_slots, in_off, out_off, slots_off, qk_l2norm;
        float q_scale;
        std::uint32_t q_off, q_stride, k_off, k_stride, v_off, v_stride, g_off, g_stride, b_off, b_stride, o_off,
            o_stride, cs, ws_off, st_off;
        std::uint32_t ring_p, ring_live;  // ADR-001 §5.3; ring_p == 0 = the in/off/slots form
    } push{a.n_v,      a.n_k,      a.d_k,          a.d_v,
           a.n_tokens, static_cast<std::uint32_t>(used_slots), oin.off, ring ? 0u : osout.off,
           osl ? osl->off : 0u, a.qk_l2norm ? 1u : 0u, q_scale,  oq.off,
           oq.stride,  ok.off,     ok.stride,      ov.off,
           ov.stride,  og.off,     og.stride,      ob.off,
           ob.stride,  oo.off,     oo.stride,      cs,
           ows.off,    ost.off,    ring_p,        ring_live};
    static_assert(sizeof(Push) <= 128, "push constants must fit the guaranteed 128-byte minimum");
    struct CheckPush {
        std::uint32_t n_tokens, n_v, g_off, g_stride, st_off;
    } check{a.n_tokens, a.n_v, og.off, og.stride, ost.off};
    constexpr std::uint32_t check_wg = 64;
    const Kernel& kc = kernel("gdn_gcheck", 2, sizeof(CheckPush), {{0, check_wg}}, {check_wg, 1, 1});
    const Kernel& k = kernel("gated_delta_rule_chunked", 11, sizeof(Push), {{0, wg}, {1, options_.gdn_max_dk}},
                             {wg, 1, 1});
    const GroupCount check_groups = grid_1d(a.n_tokens, info.max_workgroup_count[0], info.max_workgroup_count[1]);
    const std::array check_bindings{og.binding, ost.binding};
    const std::array bindings{oq.binding,  ok.binding,  ov.binding, og.binding,
                              ob.binding,  oin.binding, osout.binding, oo.binding,
                              ring ? oslab.binding : (osl ? osl->binding : oo.binding), ows.binding, ost.binding};
    stream.fill(*chunked->status.buffer, chunked->status.offset, k_status_bytes, 0u);
    stream.dispatch(kc, check_bindings, check, check_groups, 0b10);  // status written
    stream.dispatch(k, bindings, push, groups, 0b11111000000);  // state_out, out, slots, workspace, status
}

void Ops::gated_delta_rule_decode(Stream& stream, const GdnDecodeArgs& a) { gdn_impl(stream, a, nullptr); }

void Ops::gated_delta_rule_chunked(Stream& stream, const GdnChunkedArgs& a) { gdn_impl(stream, a.gdn, &a); }

std::uint64_t gdn_chunked_workspace_bytes(const GdnDecodeArgs& g, std::uint32_t chunk_size) {
    constexpr std::string_view op = "gated_delta_rule_chunked";
    const std::uint64_t cs = chunk_size;
    // gated_delta_rule_chunked.comp head_floats(): q, k, kb, kcd [cs x d_k]; vb, nv, v_new
    // [cs x d_v]; ut, attn [cs x cs]; gc [cs].
    const std::uint64_t per_head = checked_add(
        checked_add(checked_mul(4 * cs, g.d_k, op), checked_mul(3 * cs, g.d_v, op), op), 2 * cs * cs + cs, op);
    return checked_mul(checked_mul(per_head, g.n_v, op), k_f32, op);
}

std::uint32_t Ops::argmax_partials(std::uint32_t n) const {
    HALO_CHECK(n > 0, ErrorCode::Kernel, "argmax: empty input");
    const std::uint64_t chunk = std::uint64_t{options_.reduce_workgroup} * k_argmax_per_thread;
    return static_cast<std::uint32_t>((std::uint64_t{n} + chunk - 1) / chunk);
}

std::uint64_t Ops::argmax_scratch_bytes(std::uint32_t n) const {
    return std::uint64_t{argmax_partials(n)} * k_argmax_words * 4;
}

void Ops::argmax(Stream& stream, const BufferView& logits, std::uint32_t n, const BufferView& scratch,
                 const BufferView& result, std::uint32_t valid) {
    constexpr std::string_view op = "argmax";
    const std::uint32_t groups = argmax_partials(n);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    const Operand ol = operand(logits, 1, std::uint64_t{n} * k_f32, Access::Floats, align, op, "logits");
    const Operand os = operand(scratch, 1, argmax_scratch_bytes(n), Access::Floats, align, op, "scratch");
    const Operand orr = operand(result, 1, k_argmax_result_bytes, Access::Floats, align, op, "result");
    require_disjoint(os, "scratch", {{&ol, "logits"}, {&orr, "result"}}, op);
    require_disjoint(orr, "result", {{&ol, "logits"}}, op);

    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push1 {
        std::uint32_t n, chunk, groups, x_off, p_off, valid;
    } p1{n, wg * k_argmax_per_thread, groups, ol.off, os.off, valid};
    struct Push2 {
        std::uint32_t count, p_off, r_off;
    } p2{groups, os.off, orr.off};
    const Kernel& k1 = kernel("argmax_partial", 2, sizeof(Push1), {{0, wg}}, {wg, 1, 1});
    const Kernel& k2 = kernel("argmax_final", 2, sizeof(Push2), {{0, wg}}, {wg, 1, 1});
    const DeviceInfo& info = ctx_->info();
    const std::array b1{ol.binding, os.binding};
    stream.dispatch(k1, b1, p1, grid_1d(groups, info.max_workgroup_count[0], info.max_workgroup_count[1]),
                    0b10);  // scratch written
    const std::array b2{os.binding, orr.binding};
    stream.dispatch(k2, b2, p2, GroupCount{1, 1, 1}, 0b10);  // result written
}

}  // namespace halo::vulkan
