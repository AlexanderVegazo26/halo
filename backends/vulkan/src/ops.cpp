#include "halo/backends/vulkan/ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <utility>

#include "halo/backends/vulkan/context.h"
#include "halo/core/error.h"
#include "vk_util.h"

namespace halo::vulkan {

using detail::checked_mul;
using detail::to_u32;

namespace {

constexpr std::uint64_t k_f32 = 4;
// Elements handled per argmax pass-1 thread (chunk = workgroup * this).
constexpr std::uint32_t k_argmax_per_thread = 16;

std::uint64_t round_up4(std::uint64_t v) { return (v + 3) / 4 * 4; }

// Binding covering [0, bytes) rounded up to a whole word (shaders read uint words), but
// never past the end of the buffer.
BufferBinding bind(const Buffer& b, std::uint64_t bytes) {
    return BufferBinding{&b, 0, std::min<std::uint64_t>(round_up4(bytes), b.size())};
}

void require_size(const Buffer& b, std::uint64_t bytes, std::string_view op, std::string_view what) {
    HALO_CHECK(b.valid(), ErrorCode::Kernel, "{}: buffer '{}' is empty", op, what);
    HALO_CHECK(b.size() >= bytes, ErrorCode::Kernel, "{}: buffer '{}' is {} bytes, needs {}", op, what,
               b.size(), bytes);
}

void require_distinct(const Buffer& out, std::initializer_list<const Buffer*> others, std::string_view op) {
    for (const Buffer* o : others) {
        HALO_CHECK(o == nullptr || o->handle() != out.handle(), ErrorCode::Kernel,
                   "{}: output buffer aliases an input (in-place is not supported)", op);
    }
}

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
        case DType::Q6_K: return blocks(256, 210);
        default:
            throw_error(ErrorCode::Unsupported, "Vulkan matvec: weight type id {} not supported (F32, Q8_0, "
                                                "Q4_K, Q6_K)",
                        static_cast<std::uint32_t>(t));
    }
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
    // gated_delta_rule_decode shared memory: q/k tiles (2 * MAX_DK floats) plus the two
    // WG-wide L2-norm reduction arrays (2 * WG floats, D-016).
    HALO_CHECK(options_.gdn_max_dk >= 1 &&
                   (std::uint64_t{options_.gdn_max_dk} + gw) * 2 * k_f32 <= info.max_shared_memory,
               ErrorCode::Kernel, "Ops: gdn_max_dk={} with gdn_workgroup={} exceeds shared memory ({} bytes)",
               options_.gdn_max_dk, gw, info.max_shared_memory);
    HALO_CHECK(std::uint64_t{rw} * 8 <= info.max_shared_memory, ErrorCode::Kernel,
               "Ops: reduce_workgroup={} exceeds shared memory", rw);
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

void Ops::rms_norm(Stream& stream, const Buffer& x, const Buffer& w, Buffer& y, std::uint32_t rows,
                   std::uint32_t cols, float eps) {
    constexpr std::string_view op = "rms_norm";
    HALO_CHECK(rows > 0 && cols > 0, ErrorCode::Kernel, "rms_norm: empty shape {}x{}", rows, cols);
    HALO_CHECK(eps >= 0.0f, ErrorCode::Kernel, "rms_norm: negative eps");
    const std::uint64_t n = checked_mul(rows, cols, op);
    (void)to_u32(n, "rms_norm elements");
    require_size(x, n * k_f32, op, "x");
    require_size(w, std::uint64_t{cols} * k_f32, op, "w");
    require_size(y, n * k_f32, op, "y");
    require_distinct(y, {&x, &w}, op);

    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push {
        std::uint32_t rows, cols;
        float eps;
    } push{rows, cols, eps};
    const Kernel& k = kernel("rms_norm", 3, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{bind(x, n * k_f32), bind(w, std::uint64_t{cols} * k_f32), bind(y, n * k_f32)};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

void Ops::matvec(Stream& stream, DType wtype, const Buffer& w, const Buffer& x, Buffer& y, std::uint32_t rows,
                 std::uint32_t cols) {
    constexpr std::string_view op = "matvec";
    HALO_CHECK(rows > 0 && cols > 0, ErrorCode::Kernel, "matvec: empty shape {}x{}", rows, cols);
    const std::uint64_t row_bytes = matvec_row_bytes(wtype, cols);
    const std::uint64_t w_bytes = checked_mul(rows, row_bytes, op);
    // Shaders index W with 32-bit element (F32) / byte (quant) offsets.
    (void)to_u32(round_up4(w_bytes), "matvec weight byte size");
    // Quantized rows are read as whole 32-bit words: the buffer must cover the last word.
    require_size(w, round_up4(w_bytes), op, "W (rounded up to 4 bytes)");
    require_size(x, std::uint64_t{cols} * k_f32, op, "x");
    require_size(y, std::uint64_t{rows} * k_f32, op, "y");
    require_distinct(y, {&w, &x}, op);

    std::string shader;
    switch (wtype) {
        case DType::F32: shader = "matvec_f32"; break;
        case DType::Q8_0: shader = "matvec_q8_0"; break;
        case DType::Q4_K: shader = "matvec_q4_k"; break;
        case DType::Q6_K: shader = "matvec_q6_k"; break;
        default: throw_error(ErrorCode::Unsupported, "Vulkan matvec: unsupported weight type");
    }
    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push {
        std::uint32_t rows, cols;
    } push{rows, cols};
    const Kernel& k = kernel(shader, 3, sizeof(Push), {{0, wg}}, {wg, 1, 1});
    const std::array bindings{bind(w, w_bytes), bind(x, std::uint64_t{cols} * k_f32),
                              bind(y, std::uint64_t{rows} * k_f32)};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(rows, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

namespace {

// [off, off+len) and [off2, off2+len2) in the same buffer overlap?
bool overlaps(std::uint64_t off, std::uint64_t len, std::uint64_t off2, std::uint64_t len2) {
    return off < off2 + len2 && off2 < off + len;
}

}  // namespace

void Ops::gated_delta_rule_decode(Stream& stream, const GdnDecodeArgs& a) {
    constexpr std::string_view op = "gated_delta_rule_decode";
    HALO_CHECK(a.q && a.k && a.v && a.g && a.beta && a.state && a.out, ErrorCode::Kernel,
               "{}: missing buffer", op);
    HALO_CHECK(a.n_v > 0 && a.n_k > 0 && a.d_k > 0 && a.d_v > 0 && a.n_tokens > 0, ErrorCode::Kernel,
               "{}: empty dims n_v={} n_k={} d_k={} d_v={} T={}", op, a.n_v, a.n_k, a.d_k, a.d_v, a.n_tokens);
    HALO_CHECK(a.d_k <= options_.gdn_max_dk, ErrorCode::Kernel, "{}: d_k={} > gdn_max_dk={}", op, a.d_k,
               options_.gdn_max_dk);
    HALO_CHECK(a.n_slots == 0 || a.state_slots != nullptr, ErrorCode::Kernel,
               "{}: n_slots={} without a state_slots buffer", op, a.n_slots);
    // D-016: same default and validation as cpu::gdn_q_scale.
    const float q_scale = a.q_scale.value_or(1.0f / std::sqrt(static_cast<float>(a.d_k)));
    HALO_CHECK(std::isfinite(q_scale), ErrorCode::Kernel, "{}: q_scale {} is not finite", op, q_scale);
    const std::uint64_t T = a.n_tokens;
    const std::uint64_t qk_bytes = checked_mul(checked_mul(checked_mul(T, a.n_k, op), a.d_k, op), k_f32, op);
    const std::uint64_t v_elems = checked_mul(checked_mul(T, a.n_v, op), a.d_v, op);
    const std::uint64_t v_bytes = v_elems * k_f32;
    const std::uint64_t head_bytes = checked_mul(T, a.n_v, op) * k_f32;
    const std::uint64_t s_elems = checked_mul(checked_mul(a.n_v, a.d_k, op), a.d_v, op);
    (void)to_u32(v_elems, "gated_delta_rule_decode v/out elements");
    (void)to_u32(qk_bytes / k_f32, "gated_delta_rule_decode q/k elements");

    Buffer& out_state = a.state_out != nullptr ? *a.state_out : *a.state;
    const std::uint64_t in_off = a.state_offset;
    const std::uint64_t out_off = a.state_out_offset.value_or(a.state_offset);
    // Written slots: s < min(T, n_slots); the region checked is the written one.
    const std::uint64_t used_slots = std::min<std::uint64_t>(T, a.n_slots);
    const std::uint64_t slots_elems = checked_mul(used_slots, s_elems, op);
    const std::uint64_t in_end = detail::checked_add(in_off, s_elems, op);
    const std::uint64_t out_end = detail::checked_add(out_off, s_elems, op);
    const std::uint64_t slots_end = detail::checked_add(a.slots_offset, slots_elems, op);

    require_size(*a.q, qk_bytes, op, "q");
    require_size(*a.k, qk_bytes, op, "k");
    require_size(*a.v, v_bytes, op, "v");
    require_size(*a.g, head_bytes, op, "g");
    require_size(*a.beta, head_bytes, op, "beta");
    require_size(*a.state, checked_mul(in_end, k_f32, op), op, "state (input region)");
    require_size(out_state, checked_mul(out_end, k_f32, op), op, "state_out (output region)");
    require_size(*a.out, v_bytes, op, "out");
    if (used_slots > 0) require_size(*a.state_slots, checked_mul(slots_end, k_f32, op), op, "state_slots");

    // Outputs must not alias inputs; the state regions are either identical (in place)
    // or disjoint; slots overlap nothing.
    require_distinct(*a.out, {a.q, a.k, a.v, a.g, a.beta, a.state, &out_state, a.state_slots}, op);
    require_distinct(out_state, {a.q, a.k, a.v, a.g, a.beta}, op);
    if (out_state.handle() == a.state->handle()) {
        HALO_CHECK(in_off == out_off || !overlaps(in_off, s_elems, out_off, s_elems), ErrorCode::Kernel,
                   "{}: input state [{}, +{}) and output state [{}, +{}) partially overlap", op, in_off, s_elems,
                   out_off, s_elems);
    }
    if (used_slots > 0) {
        require_distinct(*a.state_slots, {a.q, a.k, a.v, a.g, a.beta}, op);
        for (const auto& [buf, off] : {std::pair{a.state, in_off}, std::pair{&out_state, out_off}}) {
            HALO_CHECK(a.state_slots->handle() != buf->handle() ||
                           !overlaps(a.slots_offset, slots_elems, off, s_elems),
                       ErrorCode::Kernel, "{}: state slots [{}, +{}) overlap a state region [{}, +{})", op,
                       a.slots_offset, slots_elems, off, s_elems);
        }
    }

    // Each state region is bound at its own descriptor offset (rounded down to the
    // device alignment) with only its own length, so a region deep inside a large arena
    // is not limited by maxStorageBufferRange; the sub-alignment remainder is passed as a
    // push-constant element offset. Shader indices (remainder + region) must fit uint32.
    const std::uint64_t align = ctx_->info().min_storage_buffer_offset_alignment;
    struct Region {
        BufferBinding binding;
        std::uint32_t elem_rem;
    };
    auto region = [&](const Buffer& b, std::uint64_t elem_off, std::uint64_t elems, std::string_view what) {
        const std::uint64_t byte_off = elem_off * k_f32;
        const std::uint64_t desc_off = byte_off / align * align;
        const std::uint64_t rem = byte_off - desc_off;
        HALO_CHECK(rem % k_f32 == 0, ErrorCode::Kernel, "{}: {} offset not representable (alignment {})", op, what,
                   align);
        (void)to_u32(rem / k_f32 + elems, what);
        return Region{BufferBinding{&b, desc_off, rem + elems * k_f32}, static_cast<std::uint32_t>(rem / k_f32)};
    };
    const Region r_in = region(*a.state, in_off, s_elems, "input state region");
    const Region r_out = region(out_state, out_off, s_elems, "output state region");
    const Region r_slots = used_slots > 0 ? region(*a.state_slots, a.slots_offset, slots_elems, "slots region")
                                          : Region{bind(*a.out, v_bytes), 0};  // placeholder, never written

    const std::uint32_t wg = options_.gdn_workgroup;
    struct Push {
        std::uint32_t n_v, n_k, d_k, d_v, n_tokens, n_slots, in_off, out_off, slots_off, qk_l2norm;
        float q_scale;
    } push{a.n_v,          a.n_k,          a.d_k,
           a.d_v,          a.n_tokens,     static_cast<std::uint32_t>(used_slots),
           r_in.elem_rem,  r_out.elem_rem, r_slots.elem_rem,
           a.qk_l2norm ? 1u : 0u, q_scale};
    static_assert(sizeof(Push) == 44, "must match the gated_delta_rule_decode.comp push block");
    const Kernel& k = kernel("gated_delta_rule_decode", 9, sizeof(Push), {{0, wg}, {1, options_.gdn_max_dk}},
                             {wg, 1, 1});
    const std::array bindings{bind(*a.q, qk_bytes),
                              bind(*a.k, qk_bytes),
                              bind(*a.v, v_bytes),
                              bind(*a.g, head_bytes),
                              bind(*a.beta, head_bytes),
                              r_in.binding,
                              r_out.binding,
                              bind(*a.out, v_bytes),
                              r_slots.binding};
    const DeviceInfo& info = ctx_->info();
    stream.dispatch(k, bindings, push, grid_1d(a.n_v, info.max_workgroup_count[0], info.max_workgroup_count[1]));
}

std::uint32_t Ops::argmax_partials(std::uint32_t n) const {
    HALO_CHECK(n > 0, ErrorCode::Kernel, "argmax: empty input");
    const std::uint64_t chunk = std::uint64_t{options_.reduce_workgroup} * k_argmax_per_thread;
    return static_cast<std::uint32_t>((std::uint64_t{n} + chunk - 1) / chunk);
}

std::uint64_t Ops::argmax_scratch_bytes(std::uint32_t n) const {
    return std::uint64_t{argmax_partials(n)} * 8;
}

void Ops::argmax(Stream& stream, const Buffer& logits, std::uint32_t n, Buffer& scratch, Buffer& result) {
    constexpr std::string_view op = "argmax";
    const std::uint32_t groups = argmax_partials(n);
    const std::uint64_t logits_bytes = std::uint64_t{n} * k_f32;
    const std::uint64_t scratch_bytes = argmax_scratch_bytes(n);
    require_size(logits, logits_bytes, op, "logits");
    require_size(scratch, scratch_bytes, op, "scratch");
    require_size(result, k_argmax_result_bytes, op, "result");
    require_distinct(scratch, {&logits, &result}, op);
    require_distinct(result, {&logits}, op);

    const std::uint32_t wg = options_.reduce_workgroup;
    struct Push1 {
        std::uint32_t n, chunk, groups;
    } p1{n, wg * k_argmax_per_thread, groups};
    struct Push2 {
        std::uint32_t count;
    } p2{groups};
    const Kernel& k1 = kernel("argmax_partial", 2, sizeof(Push1), {{0, wg}}, {wg, 1, 1});
    const Kernel& k2 = kernel("argmax_final", 2, sizeof(Push2), {{0, wg}}, {wg, 1, 1});
    const DeviceInfo& info = ctx_->info();
    const std::array b1{bind(logits, logits_bytes), bind(scratch, scratch_bytes)};
    stream.dispatch(k1, b1, p1, grid_1d(groups, info.max_workgroup_count[0], info.max_workgroup_count[1]));
    const std::array b2{bind(scratch, scratch_bytes), bind(result, k_argmax_result_bytes)};
    stream.dispatch(k2, b2, p2, GroupCount{1, 1, 1});
}

}  // namespace halo::vulkan
