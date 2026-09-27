#include "halo/backends/vulkan/ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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
    stream.dispatch(k, bindings, push, grid_1d(rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

void Ops::matvec(Stream& stream, DType wtype, const BufferView& w, const BufferView& x, const BufferView& y,
                 std::uint32_t rows, std::uint32_t cols) {
    gemv_impl(stream, GemvArgs{wtype, w, x, y, rows, cols, 1}, "matvec");
}

void Ops::gemv(Stream& stream, const GemvArgs& args) { gemv_impl(stream, args, "gemv"); }

void Ops::gemv_impl(Stream& stream, const GemvArgs& a, std::string_view op) {
    HALO_CHECK(a.rows > 0 && a.cols > 0 && a.n_vec > 0, ErrorCode::Kernel, "{}: empty shape {}x{} n_vec={}", op,
               a.rows, a.cols, a.n_vec);
    const std::uint64_t row_bytes = matvec_row_bytes(a.wtype, a.cols);
    const std::string shader = detail::weight_shader("matvec", a.wtype);
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    // F32 weights are indexed by element, quantized ones by byte (read as 32-bit words).
    const Operand ow =
        operand(a.w, a.rows, row_bytes, a.wtype == DType::F32 ? Access::Floats : Access::Words, align, op, "W");
    const Operand ox = operand(a.x, a.n_vec, std::uint64_t{a.cols} * k_f32, Access::Floats, align, op, "x");
    const Operand oy = operand(a.y, a.n_vec, std::uint64_t{a.rows} * k_f32, Access::Floats, align, op, "y");
    require_disjoint(oy, "y", {{&ow, "W"}, {&ox, "x"}}, op);

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
            gemv_impl(stream, s, op);
        }
        return;
    }

    // Workgroup size. Types with a llama-style SWAR matvec main (q5_k pilot; see
    // shaders/common/matvec_q5_k_main.glsl) run fastest at one wave (64): 16 threads per
    // block -> 4 blocks per iteration, and both ffn shapes (20/68 blocks) divide by 4.
    // Types on the shared matvec_quant_main.glsl keep reduce_workgroup (256 measured best
    // there). gemv_workgroup (HALO_VK_GEMV_WG) overrides for tuning.
    const bool swar_main = a.wtype == DType::Q5_K;
    const std::uint32_t wg = options_.gemv_workgroup != 0 ? options_.gemv_workgroup
                             : swar_main                ? 64
                                                        : options_.reduce_workgroup;
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
    // Up to k_gemv_max_vec vectors per dispatch (matvec_quant_main.glsl MAX_VEC). Offsets of
    // later chunks stay inside the index range operand() checked for the whole extent.
    for (std::uint32_t t0 = 0; t0 < a.n_vec; t0 += k_gemv_max_vec) {
        const std::uint32_t n = std::min(k_gemv_max_vec, a.n_vec - t0);
        const Push push{a.rows,
                        a.cols,
                        n,
                        ow.off,
                        ow.stride,
                        ox.off + t0 * ox.stride,
                        ox.stride,
                        oy.off + t0 * oy.stride,
                        oy.stride};
        stream.dispatch(k, bindings, push, groups);
    }
}

void Ops::gdn_impl(Stream& stream, const GdnDecodeArgs& a, const GdnChunkedArgs* chunked) {
    const std::string_view op = chunked == nullptr ? "gated_delta_rule_decode" : "gated_delta_rule_chunked";
    HALO_CHECK(a.n_v > 0 && a.n_k > 0 && a.d_k > 0 && a.d_v > 0 && a.n_tokens > 0, ErrorCode::Kernel,
               "{}: empty dims n_v={} n_k={} d_k={} d_v={} T={}", op, a.n_v, a.n_k, a.d_k, a.d_v, a.n_tokens);
    HALO_CHECK(a.n_v % a.n_k == 0, ErrorCode::Kernel, "{}: n_v={} is not a multiple of n_k={}", op, a.n_v, a.n_k);
    HALO_CHECK(a.d_k <= options_.gdn_max_dk, ErrorCode::Kernel, "{}: d_k={} > gdn_max_dk={}", op, a.d_k,
               options_.gdn_max_dk);
    HALO_CHECK(a.n_slots == 0 || !a.state_slots.empty(), ErrorCode::Kernel,
               "{}: n_slots={} without a state_slots view", op, a.n_slots);
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
    const Operand oin = operand(a.state, 1, s_bytes, Access::Floats, align, op, "state (input region)");
    const Operand osout = operand(a.state_out.value_or(a.state), 1, s_bytes, Access::Floats, align, op,
                                  "state_out (output region)");
    std::optional<Operand> osl;
    if (used_slots > 0) {
        osl = operand(a.state_slots, 1, checked_mul(used_slots, s_bytes, op), Access::Floats, align, op,
                      "state_slots (written slots)");
    }

    const std::initializer_list<std::pair<const Operand*, std::string_view>> inputs{
        {&oq, "q"}, {&ok, "k"}, {&ov, "v"}, {&og, "g"}, {&ob, "beta"}};
    require_disjoint(oo, "out", inputs, op);
    require_disjoint(oo, "out", {{&oin, "state"}, {&osout, "state_out"}, {osl ? &*osl : nullptr, "state_slots"}}, op);
    require_disjoint(osout, "state_out", inputs, op);
    // The input and output state regions are identical (in place) or disjoint.
    HALO_CHECK(!overlaps(oin, osout) || oin.begin == osout.begin, ErrorCode::Kernel,
               "{}: input state bytes [{}, {}) and output state bytes [{}, {}) partially overlap", op, oin.begin,
               oin.end, osout.begin, osout.end);
    if (osl) {
        require_disjoint(*osl, "state_slots", inputs, op);
        require_disjoint(*osl, "state_slots", {{&oin, "state"}, {&osout, "state_out"}}, op);
    }

    const DeviceInfo& info = ctx_->info();
    const std::uint32_t wg = options_.gdn_workgroup;
    const GroupCount groups = grid_1d(a.n_v, info.max_workgroup_count[0], info.max_workgroup_count[1]);
    if (chunked == nullptr) {
        struct Push {
            std::uint32_t n_v, n_k, d_k, d_v, n_tokens, n_slots, in_off, out_off, slots_off, qk_l2norm;
            float q_scale;
            std::uint32_t q_off, q_stride, k_off, k_stride, v_off, v_stride, g_off, g_stride, b_off, b_stride, o_off,
                o_stride;
        } push{a.n_v,      a.n_k,      a.d_k,          a.d_v,
               a.n_tokens, static_cast<std::uint32_t>(used_slots), oin.off, osout.off,
               osl ? osl->off : 0u, a.qk_l2norm ? 1u : 0u, q_scale,  oq.off,
               oq.stride,  ok.off,     ok.stride,      ov.off,
               ov.stride,  og.off,     og.stride,      ob.off,
               ob.stride,  oo.off,     oo.stride};
        static_assert(sizeof(Push) <= 128, "push constants must fit the guaranteed 128-byte minimum");
        const Kernel& k = kernel("gated_delta_rule_decode", 9, sizeof(Push), {{0, wg}, {1, options_.gdn_max_dk}},
                                 {wg, 1, 1});
        // Without slots, binding 8 aliases `out` as a placeholder; the shader never writes it.
        const std::array bindings{oq.binding, ok.binding,    ov.binding, og.binding,
                                  ob.binding, oin.binding,   osout.binding, oo.binding,
                                  osl ? osl->binding : oo.binding};
        stream.dispatch(k, bindings, push, groups);
        return;
    }

    // Chunked form: workspace and status, then the g pre-pass and the main kernel.
    const std::uint32_t cs = chunked->chunk_size;
    HALO_CHECK(cs >= 1 && cs <= k_gdn_max_chunk, ErrorCode::Kernel, "{}: chunk_size {} not in [1, {}]", op, cs,
               k_gdn_max_chunk);
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
    } push{a.n_v,      a.n_k,      a.d_k,          a.d_v,
           a.n_tokens, static_cast<std::uint32_t>(used_slots), oin.off, osout.off,
           osl ? osl->off : 0u, a.qk_l2norm ? 1u : 0u, q_scale,  oq.off,
           oq.stride,  ok.off,     ok.stride,      ov.off,
           ov.stride,  og.off,     og.stride,      ob.off,
           ob.stride,  oo.off,     oo.stride,      cs,
           ows.off,    ost.off};
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
                              osl ? osl->binding : oo.binding, ows.binding, ost.binding};
    stream.fill(*chunked->status.buffer, chunked->status.offset, k_status_bytes, 0u);
    stream.dispatch(kc, check_bindings, check, check_groups);
    stream.dispatch(k, bindings, push, groups);
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
    stream.dispatch(k1, b1, p1, grid_1d(groups, info.max_workgroup_count[0], info.max_workgroup_count[1]));
    const std::array b2{os.binding, orr.binding};
    stream.dispatch(k2, b2, p2, GroupCount{1, 1, 1});
}

}  // namespace halo::vulkan
