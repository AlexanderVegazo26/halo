// HIP operator layer: host-side validation (before anything runs), parameter building, and
// dispatch to the device launchers or the host emulation of the same kernel bodies.

#include "halo/backends/hip/ops.h"

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <functional>
#include <limits>

#include "halo/backends/hip/registry.h"
#include "launch.h"

namespace halo::hip {

static_assert(kStatusPositiveG == kern::kStatusPositiveG && kStatusNaN == kern::kStatusNaN);

namespace {

constexpr std::uint64_t kF32 = sizeof(float);
constexpr unsigned kMaxGridY = 65535;

/// Address range [begin, end) an operand touches (empty when begin == end).
struct Range {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    [[nodiscard]] bool empty() const noexcept { return begin == end; }
};

bool overlaps(const Range& a, const Range& b) noexcept {
    if (a.empty() || b.empty()) return false;
    return a.begin < b.end && b.begin < a.end;
}

/// A validated 2-D fp32 operand.
struct Rows {
    float* ptr = nullptr;
    std::uint64_t stride = 0;  // elements
    Range range;
};

std::uint64_t mul_checked(std::uint64_t a, std::uint64_t b, const char* op) {
    HALO_CHECK(a == 0 || b <= std::numeric_limits<std::uint64_t>::max() / a, ErrorCode::Kernel,
               "{}: size overflow {} * {}", op, a, b);
    return a * b;
}

void check_target(const Target& t, const Buffer& b, const char* op, const char* what) {
    if (t.is_device()) {
        HALO_CHECK(b.tier() != MemoryTier::Host, ErrorCode::Kernel,
                   "{}: {} is a host-emulation buffer but the target is a device stream", op, what);
    } else {
        HALO_CHECK(b.tier() == MemoryTier::Host, ErrorCode::Kernel,
                   "{}: {} is a {} buffer but the target is the host emulation (use Buffer::wrap_host)", op,
                   what, to_string(b.tier()));
    }
}

/// Resolves `rows` x `cols` fp32 elements of view `v` (row_stride 0 = dense).
Rows resolve(const Target& t, const BufferView& v, std::uint64_t rows, std::uint64_t cols, const char* op,
             const char* what) {
    HALO_CHECK(v.buffer != nullptr && !v.buffer->empty(), ErrorCode::Kernel, "{}: {} is missing", op, what);
    const Buffer& b = *v.buffer;
    check_target(t, b, op, what);
    const std::uint64_t row_bytes = mul_checked(cols, kF32, op);
    const std::uint64_t stride = v.row_stride == 0 ? row_bytes : v.row_stride;
    HALO_CHECK(v.offset % kF32 == 0 && stride % kF32 == 0, ErrorCode::Kernel,
               "{}: {} offset {} / row stride {} not 4-byte aligned", op, what, v.offset, stride);
    HALO_CHECK(rows <= 1 || stride >= row_bytes, ErrorCode::Kernel, "{}: {} row stride {} < row bytes {}", op,
               what, stride, row_bytes);
    HALO_CHECK(v.offset <= b.bytes(), ErrorCode::Kernel, "{}: {} offset {} beyond buffer of {} bytes", op, what,
               v.offset, b.bytes());
    const std::uint64_t avail = b.bytes() - v.offset;
    HALO_CHECK(v.bytes <= avail, ErrorCode::Kernel, "{}: {} view of {} bytes exceeds the {} bytes after offset",
               op, what, v.bytes, avail);
    const std::uint64_t limit = v.bytes == 0 ? avail : v.bytes;
    std::uint64_t extent = 0;
    if (rows > 0 && cols > 0) {
        extent = mul_checked(rows - 1, stride, op);
        HALO_CHECK(extent <= std::numeric_limits<std::uint64_t>::max() - row_bytes, ErrorCode::Kernel,
                   "{}: size overflow", op);
        extent += row_bytes;
    }
    HALO_CHECK(extent <= limit, ErrorCode::Kernel, "{}: {} needs {} bytes ({} rows x {} cols, stride {}), view has {}",
               op, what, extent, rows, cols, stride, limit);
    Rows r;
    r.ptr = reinterpret_cast<float*>(static_cast<char*>(b.data()) + v.offset);
    HALO_CHECK(reinterpret_cast<std::uintptr_t>(r.ptr) % alignof(float) == 0, ErrorCode::Kernel,
               "{}: {} is not 4-byte aligned", op, what);
    r.stride = stride / kF32;
    r.range.begin = reinterpret_cast<std::uintptr_t>(r.ptr);
    r.range.end = r.range.begin + extent;
    return r;
}

/// A validated byte-addressed 2-D operand (quantized weight rows).
struct ByteRows {
    const std::uint8_t* ptr = nullptr;
    std::uint64_t stride = 0;  // bytes
    Range range;
};

ByteRows resolve_bytes(const Target& t, const BufferView& v, std::uint64_t rows, std::uint64_t row_bytes,
                       std::uint64_t align, const char* op, const char* what) {
    HALO_CHECK(v.buffer != nullptr && !v.buffer->empty(), ErrorCode::Kernel, "{}: {} is missing", op, what);
    const Buffer& b = *v.buffer;
    check_target(t, b, op, what);
    const std::uint64_t stride = v.row_stride == 0 ? row_bytes : v.row_stride;
    HALO_CHECK(rows <= 1 || stride >= row_bytes, ErrorCode::Kernel, "{}: {} row stride {} < row bytes {}", op, what,
               stride, row_bytes);
    HALO_CHECK(v.offset <= b.bytes(), ErrorCode::Kernel, "{}: {} offset {} beyond buffer of {} bytes", op, what,
               v.offset, b.bytes());
    const std::uint64_t avail = b.bytes() - v.offset;
    HALO_CHECK(v.bytes <= avail, ErrorCode::Kernel, "{}: {} view of {} bytes exceeds the {} bytes after offset",
               op, what, v.bytes, avail);
    const std::uint64_t limit = v.bytes == 0 ? avail : v.bytes;
    std::uint64_t extent = 0;
    if (rows > 0 && row_bytes > 0) {
        extent = mul_checked(rows - 1, stride, op);
        HALO_CHECK(extent <= std::numeric_limits<std::uint64_t>::max() - row_bytes, ErrorCode::Kernel,
                   "{}: size overflow", op);
        extent += row_bytes;
    }
    HALO_CHECK(extent <= limit, ErrorCode::Kernel, "{}: {} needs {} bytes ({} rows of {} bytes, stride {}), view has {}",
               op, what, extent, rows, row_bytes, stride, limit);
    ByteRows r;
    r.ptr = static_cast<const std::uint8_t*>(b.data()) + v.offset;
    HALO_CHECK(reinterpret_cast<std::uintptr_t>(r.ptr) % align == 0 && stride % align == 0, ErrorCode::Kernel,
               "{}: {} is not {}-byte aligned", op, what, align);
    r.stride = stride;
    r.range.begin = reinterpret_cast<std::uintptr_t>(r.ptr);
    r.range.end = r.range.begin + extent;
    return r;
}

struct Named {
    const Rows* r;
    const char* what;
};

void check_disjoint(const Rows& out, const char* out_name, std::initializer_list<Named> others, const char* op) {
    for (const Named& o : others) {
        HALO_CHECK(!overlaps(out.range, o.r->range), ErrorCode::Kernel, "{}: {} overlaps {}", op, out_name, o.what);
    }
}

/// out may alias `in` only exactly (same address and stride).
void check_alias_exact_or_disjoint(const Rows& in, const Rows& out, const char* op, const char* what) {
    if (!overlaps(in.range, out.range)) return;
    HALO_CHECK(in.ptr == out.ptr && in.stride == out.stride, ErrorCode::Kernel,
               "{}: output partially overlaps {} (only exact aliasing is allowed)", op, what);
}

template <class P>
void dispatch(const Target& t, int (*dev)(const P&, const kern::Launch&, void*),
              void (*emu)(const P&, const kern::Launch&, bool), const P& p, const kern::Launch& l, const char* what) {
    if (l.grid_x == 0 || l.grid_y == 0) return;
    HALO_CHECK(l.grid_y <= kMaxGridY, ErrorCode::Kernel, "{}: grid.y {} exceeds {}", what, l.grid_y, kMaxGridY);
    if (t.is_device()) {
        check(dev(p, l, t.stream()->handle()), what);
    } else {
        emu(p, l, t.reverse_order());
    }
}

unsigned block_of(std::string_view op, std::string_view form, const std::string& name) {
    const KernelVariant& v = find_variant(op, name);
    HALO_CHECK(v.form == form, ErrorCode::Config, "HIP: variant '{}' is a {} {} kernel, not {}", name, op, v.form,
               form);
    return v.block;
}

float resolve_q_scale(const GdnArgs& a) {
    const float s = a.q_scale.value_or(1.0f / std::sqrt(static_cast<float>(a.d_k)));
    HALO_CHECK(std::isfinite(s), ErrorCode::Kernel, "gdn: q_scale {} is not finite", s);
    return s;
}

/// Validated GDN operands shared by both forms.
struct GdnResolved {
    Rows q, k, v, g, beta, state, state_out, slots, out;
    kern::GdnIn in;
    kern::GdnDimsK dims;
    std::uint64_t state_n = 0;
    float q_scale = 1.0f;
};

GdnResolved resolve_gdn(const Target& t, const GdnArgs& a, const char* op) {
    HALO_CHECK(a.n_k > 0 && a.n_v > 0 && a.d_k > 0 && a.d_v > 0, ErrorCode::Kernel,
               "{}: zero dimension (n_k {}, n_v {}, d_k {}, d_v {})", op, a.n_k, a.n_v, a.d_k, a.d_v);
    HALO_CHECK(a.n_v % a.n_k == 0, ErrorCode::Kernel, "{}: n_v {} is not a multiple of n_k {}", op, a.n_v, a.n_k);
    HALO_CHECK(a.d_k <= kern::kGdnMaxDk, ErrorCode::Kernel,
               "{}: d_k {} exceeds this backend's register-resident limit {}", op, a.d_k, kern::kGdnMaxDk);
    HALO_CHECK(a.n_tokens >= 1, ErrorCode::Kernel, "{}: n_tokens must be >= 1", op);
    GdnResolved r;
    const std::uint64_t T = a.n_tokens;
    const std::uint64_t qk_cols = mul_checked(a.n_k, a.d_k, op);
    const std::uint64_t v_cols = mul_checked(a.n_v, a.d_v, op);
    r.state_n = mul_checked(mul_checked(a.n_v, a.d_k, op), a.d_v, op);
    HALO_CHECK(r.state_n <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Kernel, "{}: state too large", op);
    r.q = resolve(t, a.q, T, qk_cols, op, "q");
    r.k = resolve(t, a.k, T, qk_cols, op, "k");
    r.v = resolve(t, a.v, T, v_cols, op, "v");
    r.g = resolve(t, a.g, T, a.n_v, op, "g");
    r.beta = resolve(t, a.beta, T, a.n_v, op, "beta");
    r.out = resolve(t, a.out, T, v_cols, op, "out");
    r.state = resolve(t, a.state, 1, r.state_n, op, "state");
    r.state_out = a.state_out ? resolve(t, *a.state_out, 1, r.state_n, op, "state_out") : r.state;
    if (r.state_out.ptr != r.state.ptr) {
        HALO_CHECK(!overlaps(r.state.range, r.state_out.range), ErrorCode::Kernel,
                   "{}: state_out partially overlaps state (must be identical or disjoint)", op);
    }
    if (a.n_slots > 0 || !a.state_slots.empty()) {
        HALO_CHECK(a.n_slots > 0 && !a.state_slots.empty(), ErrorCode::Kernel,
                   "{}: n_slots {} but state_slots is {}", op, a.n_slots, a.state_slots.empty() ? "empty" : "set");
        r.slots = resolve(t, a.state_slots, a.n_slots, r.state_n, op, "state_slots");
    }
    const std::initializer_list<Named> inputs{{&r.q, "q"}, {&r.k, "k"}, {&r.v, "v"}, {&r.g, "g"}, {&r.beta, "beta"}};
    check_disjoint(r.out, "out", inputs, op);
    check_disjoint(r.out, "out", {{&r.state, "state"}, {&r.state_out, "state_out"}}, op);
    check_disjoint(r.state, "state", inputs, op);
    check_disjoint(r.state_out, "state_out", inputs, op);
    check_disjoint(r.slots, "state_slots", inputs, op);
    check_disjoint(r.slots, "state_slots",
                   {{&r.state, "state"}, {&r.state_out, "state_out"}, {&r.out, "out"}}, op);
    r.q_scale = resolve_q_scale(a);
    r.in = kern::GdnIn{r.q.ptr, r.k.ptr, r.v.ptr, r.g.ptr, r.beta.ptr,
                       r.q.stride, r.k.stride, r.v.stride, r.g.stride, r.beta.stride};
    r.dims = kern::GdnDimsK{a.n_k, a.n_v, a.d_k, a.d_v, a.mapping == GdnHeadMapping::Grouped ? 1u : 0u};
    return r;
}

/// The single status word of `v` (4 bytes).
Rows resolve_status(const Target& t, const BufferView& v, const char* op) {
    return resolve(t, v, 1, 1, op, "status");
}

void zero_status(const Target& t, const Rows& s) {
    if (t.is_device()) {
        check(hipMemsetAsync(s.ptr, 0, sizeof(std::uint32_t), static_cast<hipStream_t>(t.stream()->handle())),
              "hipMemsetAsync(status)");
    } else {
        *reinterpret_cast<std::uint32_t*>(s.ptr) = 0;
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------

Ops::Ops(OpsOptions options) : options_(std::move(options)) {
    gdn_rec_block_ = block_of("GATED_DELTANET", "recurrent", options_.gdn_recurrent);
    gdn_chunk_block_ = block_of("GATED_DELTANET", "chunked", options_.gdn_chunked);
    HALO_CHECK(gdn_chunk_block_ <= kern::kGdnStateMaxBlock, ErrorCode::Config,
               "HIP: chunked GDN variant block {} exceeds the LDS state tile ({})", gdn_chunk_block_,
               kern::kGdnStateMaxBlock);
    conv_block_ = block_of("CONV1D_SHORT", "", options_.conv1d);
    norm_block_ = block_of("GATED_NORM", "", options_.gated_norm);
    gemv_block_ = block_of("QUANT_GEMV", "", options_.gemv);
    gemv_generic_ = find_variant("QUANT_GEMV", options_.gemv).kernels == "k_gemv_generic";
}

void Ops::gated_delta_rule_recurrent(const Target& target, const GdnArgs& args) const {
    constexpr const char* kOp = "hip::gated_delta_rule_recurrent";
    const GdnResolved r = resolve_gdn(target, args, kOp);
    kern::GdnRecParams p;
    p.in = r.in;
    p.dims = r.dims;
    p.n_tok = args.n_tokens;
    p.state_in = r.state.ptr;
    p.state_out = r.state_out.ptr;
    p.slots = r.slots.ptr;
    p.n_slots = r.slots.ptr != nullptr ? args.n_slots : 0;
    p.out = r.out.ptr;
    p.out_stride = r.out.stride;
    p.l2 = args.qk_l2norm ? 1u : 0u;
    p.q_scale = r.q_scale;
    dispatch(target, &detail::launch_gdn_recurrent, &detail::emulate_gdn_recurrent, p,
             kern::gdn_recurrent_launch(r.dims, gdn_rec_block_), kOp);
}

std::uint64_t gdn_chunked_workspace_bytes(const GdnArgs& args, std::uint32_t chunk_size,
                                          std::uint32_t chunks_per_group) {
    HALO_CHECK(chunk_size >= 1 && chunk_size <= kern::kGdnMaxChunk, ErrorCode::Kernel,
               "gdn_chunked_workspace_bytes: chunk_size {} not in [1, {}]", chunk_size, kern::kGdnMaxChunk);
    HALO_CHECK(args.n_tokens >= 1 && chunks_per_group >= 1, ErrorCode::Kernel,
               "gdn_chunked_workspace_bytes: n_tokens and chunks_per_group must be >= 1");
    const std::uint64_t n_chunks = (static_cast<std::uint64_t>(args.n_tokens) + chunk_size - 1) / chunk_size;
    const std::uint64_t g = chunks_per_group < n_chunks ? chunks_per_group : n_chunks;
    const kern::GdnChunkRecord rec = kern::gdn_chunk_record(chunk_size, args.d_k, args.d_v);
    constexpr const char* kOp = "gdn_chunked_workspace_bytes";
    return mul_checked(mul_checked(mul_checked(g, args.n_v, kOp), rec.floats, kOp), kF32, kOp);
}

void Ops::gated_delta_rule_chunked(const Target& target, const GdnChunkedArgs& args) const {
    constexpr const char* kOp = "hip::gated_delta_rule_chunked";
    const GdnArgs& a = args.gdn;
    const GdnResolved r = resolve_gdn(target, a, kOp);
    const std::uint32_t cs = args.chunk_size;
    HALO_CHECK(cs >= 1 && cs <= kern::kGdnMaxChunk, ErrorCode::Kernel,
               "{}: chunk_size {} not in [1, {}] (the CPU op accepts up to 1024; this backend's LDS tiles cap it)",
               kOp, cs, kern::kGdnMaxChunk);
    const kern::GdnChunkRecord rec = kern::gdn_chunk_record(cs, a.d_k, a.d_v);
    const std::uint64_t per_chunk = mul_checked(rec.floats, a.n_v, kOp);  // floats per chunk (all heads)
    HALO_CHECK(args.workspace.buffer != nullptr, ErrorCode::Kernel, "{}: workspace is missing", kOp);
    // Largest whole number of chunk records the workspace view holds.
    const Buffer& wb = *args.workspace.buffer;
    HALO_CHECK(args.workspace.offset <= wb.bytes(), ErrorCode::Kernel, "{}: workspace offset beyond buffer", kOp);
    const std::uint64_t ws_bytes = args.workspace.bytes == 0 ? wb.bytes() - args.workspace.offset : args.workspace.bytes;
    const std::uint32_t n_chunks = (a.n_tokens + cs - 1) / cs;
    const std::uint64_t fit = ws_bytes / kF32 / per_chunk;
    HALO_CHECK(fit >= 1, ErrorCode::Kernel, "{}: workspace of {} bytes is smaller than one chunk group ({} bytes)",
               kOp, ws_bytes, per_chunk * kF32);
    std::uint32_t group = fit < n_chunks ? static_cast<std::uint32_t>(fit) : n_chunks;
    if (group > kMaxGridY) group = kMaxGridY;  // K1 grid.y; clamp here so no launch can fail late
    const Rows ws = resolve(target, args.workspace, 1, per_chunk * group, kOp, "workspace");
    const Rows status = resolve_status(target, args.status, kOp);
    const std::initializer_list<Named> all{{&r.q, "q"},         {&r.k, "k"},     {&r.v, "v"},
                                           {&r.g, "g"},         {&r.beta, "beta"}, {&r.state, "state"},
                                           {&r.state_out, "state_out"}, {&r.slots, "state_slots"}, {&r.out, "out"}};
    check_disjoint(ws, "workspace", all, kOp);
    check_disjoint(status, "status", all, kOp);
    check_disjoint(status, "status", {{&ws, "workspace"}}, kOp);

    zero_status(target, status);
    auto* status_word = reinterpret_cast<std::uint32_t*>(status.ptr);
    kern::GdnCheckParams cp{r.g.ptr, r.g.stride, a.n_tokens, a.n_v, status_word};
    dispatch(target, &detail::launch_gdn_check_g, &detail::emulate_gdn_check_g, cp,
             kern::gdn_check_launch(a.n_tokens, a.n_v, gdn_chunk_block_), kOp);

    kern::GdnChunkParams p;
    p.in = r.in;
    p.dims = r.dims;
    p.n_tok = a.n_tokens;
    p.cs = cs;
    p.state_out = r.state_out.ptr;
    p.slots = r.slots.ptr;
    p.n_slots = r.slots.ptr != nullptr ? a.n_slots : 0;
    p.out = r.out.ptr;
    p.out_stride = r.out.stride;
    p.l2 = a.qk_l2norm ? 1u : 0u;
    p.q_scale = r.q_scale;
    p.ws = ws.ptr;
    p.status = status_word;
    for (std::uint32_t c0 = 0; c0 < n_chunks; c0 += group) {
        p.chunk0 = c0;
        p.n_chunks = (n_chunks - c0) < group ? (n_chunks - c0) : group;
        // Group 0 continues the caller's state; later groups continue what group 0 wrote.
        p.state_in = c0 == 0 ? r.state.ptr : r.state_out.ptr;
        dispatch(target, &detail::launch_gdn_chunk_intra, &detail::emulate_gdn_chunk_intra, p,
                 kern::gdn_intra_launch(r.dims, p.n_chunks, gdn_chunk_block_), kOp);
        dispatch(target, &detail::launch_gdn_chunk_state, &detail::emulate_gdn_chunk_state, p,
                 kern::gdn_state_launch(r.dims, gdn_chunk_block_), kOp);
    }
}

void Ops::causal_conv1d_silu(const Target& target, const Conv1dArgs& a) const {
    constexpr const char* kOp = "hip::causal_conv1d_silu";
    HALO_CHECK(a.kernel >= 1 && a.kernel <= kern::kConvMaxK, ErrorCode::Kernel, "{}: kernel size {} not in [1, {}]",
               kOp, a.kernel, kern::kConvMaxK);
    HALO_CHECK(a.channels >= 1 && a.n_tokens >= 1, ErrorCode::Kernel, "{}: channels and n_tokens must be >= 1", kOp);
    const std::uint64_t T = a.n_tokens;
    const std::uint64_t C = a.channels;
    const std::uint64_t hist = a.kernel - 1u;
    const Rows x = resolve(target, a.x, T, C, kOp, "x");
    const Rows w = resolve(target, a.weight, C, a.kernel, kOp, "weight");
    const Rows out = resolve(target, a.out, T, C, kOp, "out");
    Rows st;
    if (hist > 0) st = resolve(target, a.conv_state, hist, C, kOp, "conv_state");
    Rows slots;
    if (a.n_slots > 0 || !a.state_slots.empty()) {
        HALO_CHECK(hist > 0, ErrorCode::Kernel, "{}: state_slots need kernel size >= 2", kOp);
        HALO_CHECK(a.n_slots > 0 && !a.state_slots.empty(), ErrorCode::Kernel, "{}: n_slots / state_slots mismatch",
                   kOp);
        slots = resolve(target, BufferView(*a.state_slots.buffer, a.state_slots.offset, a.state_slots.bytes, 0),
                        a.n_slots, mul_checked(hist, C, kOp), kOp, "state_slots");
    }
    check_alias_exact_or_disjoint(x, out, kOp, "x");
    check_disjoint(out, "out", {{&st, "conv_state"}, {&w, "weight"}, {&slots, "state_slots"}}, kOp);
    check_disjoint(st, "conv_state", {{&x, "x"}, {&w, "weight"}, {&slots, "state_slots"}}, kOp);
    check_disjoint(slots, "state_slots", {{&x, "x"}, {&w, "weight"}}, kOp);
    kern::ConvParams p;
    p.x = x.ptr;
    p.x_stride = x.stride;
    p.w = w.ptr;
    p.w_stride = w.stride;
    p.state = st.ptr;
    p.state_stride = st.stride;
    p.out = out.ptr;
    p.out_stride = out.stride;
    p.slots = slots.ptr;
    p.n_slots = slots.ptr != nullptr ? a.n_slots : 0;
    p.n_tok = a.n_tokens;
    p.channels = a.channels;
    p.k = a.kernel;
    dispatch(target, &detail::launch_conv1d_silu, &detail::emulate_conv1d_silu, p,
             kern::conv_launch(a.channels, conv_block_), kOp);
}

void Ops::gated_rms_norm(const Target& target, const GatedNormArgs& a) const {
    constexpr const char* kOp = "hip::gated_rms_norm";
    HALO_CHECK(a.cols >= 1, ErrorCode::Kernel, "{}: cols must be >= 1", kOp);
    if (a.rows == 0) return;
    const Rows x = resolve(target, a.x, a.rows, a.cols, kOp, "x");
    const Rows z = resolve(target, a.z, a.rows, a.cols, kOp, "z");
    const Rows w = resolve(target, a.w, 1, a.cols, kOp, "w");
    const Rows out = resolve(target, a.out, a.rows, a.cols, kOp, "out");
    check_alias_exact_or_disjoint(x, out, kOp, "x");
    check_alias_exact_or_disjoint(z, out, kOp, "z");
    check_disjoint(out, "out", {{&w, "w"}}, kOp);
    kern::NormParams p;
    p.x = x.ptr;
    p.x_stride = x.stride;
    p.z = z.ptr;
    p.z_stride = z.stride;
    p.w = w.ptr;
    p.out = out.ptr;
    p.out_stride = out.stride;
    p.rows = a.rows;
    p.cols = a.cols;
    p.eps = a.eps;
    dispatch(target, &detail::launch_norm, &detail::emulate_norm, p, kern::norm_launch(a.rows, norm_block_), kOp);
}

void Ops::check_status(const Target& target, const BufferView& status) {
    constexpr const char* kOp = "hip::check_status";
    const Rows s = resolve_status(target, status, kOp);
    if (target.is_device()) target.stream()->synchronize();  // the word is written by that stream
    std::uint32_t word = 0;
    status.buffer->download(&word, sizeof(word), status.offset);
    static_cast<void>(s);
    if (word == 0) return;
    std::string what;
    if ((word & kStatusPositiveG) != 0) what += " g>0-or-NaN (chunked GATED_DELTANET requires g <= 0; nothing written)";
    if ((word & kStatusNaN) != 0) what += " NaN logit";
    throw_error(ErrorCode::Kernel, "HIP kernel status 0x{:x}:{}", word, what);
}

}  // namespace halo::hip

namespace halo::hip {

namespace {

kern::WType gemv_wtype(DType t, const char* op) {
    switch (t) {
        case DType::F32: return kern::WType::F32;
        case DType::F16: return kern::WType::F16;
        case DType::Q8_0: return kern::WType::Q8_0;
        case DType::Q4_K: return kern::WType::Q4_K;
        case DType::Q5_K: return kern::WType::Q5_K;
        case DType::Q6_K: return kern::WType::Q6_K;
        default: break;
    }
    throw_error(ErrorCode::Unsupported, "{}: weight type id {} has no HIP GEMV kernel (F32, F16, Q8_0, Q4_K, Q5_K, Q6_K)",
                op, static_cast<std::uint32_t>(t));
}

}  // namespace

void Ops::gemv(const Target& target, const GemvArgs& a) const {
    constexpr const char* kOp = "hip::gemv";
    const kern::WType wt = gemv_wtype(a.wtype, kOp);
    HALO_CHECK(a.rows >= 1 && a.cols >= 1 && a.n_vec >= 1, ErrorCode::Kernel,
               "{}: rows {}, cols {}, n_vec {} must all be >= 1", kOp, a.rows, a.cols, a.n_vec);
    const unsigned be = kern::wq_block_elems(wt);
    HALO_CHECK(a.cols % be == 0, ErrorCode::Kernel, "{}: cols {} is not a multiple of the block size {}", kOp, a.cols,
               be);
    const std::uint64_t row_bytes = mul_checked(a.cols / be, kern::wq_block_bytes(wt), kOp);
    const ByteRows w = resolve_bytes(target, a.w, a.rows, row_bytes, wt == kern::WType::F32 ? 4 : 1, kOp, "w");
    const Rows x = resolve(target, a.x, a.n_vec, a.cols, kOp, "x");
    const Rows y = resolve(target, a.y, a.n_vec, a.rows, kOp, "y");
    const Rows wr{nullptr, 0, w.range};
    check_disjoint(y, "y", {{&x, "x"}, {&wr, "w"}}, kOp);
    kern::GemvParams p;
    p.type = wt;
    p.w = w.ptr;
    p.w_stride = w.stride;
    p.x = x.ptr;
    p.x_stride = x.stride;
    p.y = y.ptr;
    p.y_stride = y.stride;
    p.rows = a.rows;
    p.cols = a.cols;
    p.n_vec = a.n_vec;
    if (gemv_generic_) {
        dispatch(target, &detail::launch_gemv_generic, &detail::emulate_gemv_generic, p,
                 kern::gemv_generic_launch(a.rows, a.n_vec, gemv_block_), kOp);
    } else {
        dispatch(target, &detail::launch_gemv_wave, &detail::emulate_gemv_wave, p,
                 kern::gemv_wave_launch(a.rows, a.n_vec, gemv_block_), kOp);
    }
}

}  // namespace halo::hip
