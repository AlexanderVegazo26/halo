// HIP operator layer: host-side validation (before anything runs), parameter building, and
// dispatch to the device launchers or the host emulation of the same kernel bodies.

#include "halo/backends/hip/ops.h"

#include <hip/hip_runtime_api.h>

#include <bit>
#include <cmath>
#include <functional>
#include <limits>

#include "halo/backends/hip/registry.h"
#include "launch.h"

namespace halo::hip {

static_assert(kStatusPositiveG == kern::kStatusPositiveG && kStatusNaN == kern::kStatusNaN &&
              kStatusBadBlock == kern::kStatusBadBlock && kStatusBadIndex == kern::kStatusBadIndex);

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
    argmax_block_ = block_of("ARGMAX_FUSED", "", options_.argmax);
    HALO_CHECK(argmax_block_ >= 1 && argmax_block_ <= kern::kArgMaxBlock && (argmax_block_ & (argmax_block_ - 1)) == 0,
               ErrorCode::Config, "HIP: argmax variant block {} must be a power of two <= {}", argmax_block_,
               kern::kArgMaxBlock);
    topk_block_ = block_of("TOP_K", "", options_.topk);
    rms_block_ = block_of("RMS_NORM", "", options_.rms_norm);
    rope_block_ = block_of("PARTIAL_ROPE", "", options_.rope);
    swiglu_block_ = block_of("SWIGLU", "", options_.swiglu);
    mulsig_block_ = block_of("MUL_SIGMOID", "", options_.mul_sigmoid);
    attn_block_ = block_of("ATTENTION", "", options_.attention);
    kvw_block_ = block_of("KV_WRITE", "", options_.kv_write);
    rows_block_ = block_of("GET_ROWS", "", options_.get_rows);
    add_block_ = block_of("ADD", "", options_.add);
    addnorm_block_ = block_of("ADD_RMS_NORM", "", options_.add_rms_norm);
    gate_block_ = block_of("GDN_GATE", "", options_.gdn_gate);
    gemm_block_ = block_of("QUANT_GEMM", "", options_.gemm);
    HALO_CHECK(gemm_block_ == kern::kGemmBlock, ErrorCode::Config, "HIP: GEMM variant block {} must be {} (16x16 tile)",
               gemm_block_, kern::kGemmBlock);
    attn_exact_ = find_variant("ATTENTION", options_.attention).kernels.find("exact") != std::string_view::npos;
    HALO_CHECK(attn_block_ >= 32 && attn_block_ <= kern::kAttnMaxBlock, ErrorCode::Config,
               "HIP: attention variant block {} not in [32, {}]", attn_block_, kern::kAttnMaxBlock);
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
    if ((word & kStatusBadBlock) != 0) what += " attention block table id outside the KV pool (out is undefined)";
    if ((word & kStatusBadIndex) != 0) what += " get_rows id outside the table (that output row is not written)";
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
        case DType::IQ4_NL: return kern::WType::IQ4_NL;
        case DType::IQ4_XS: return kern::WType::IQ4_XS;
        case DType::Q3_K: return kern::WType::Q3_K;
        case DType::IQ3_S: return kern::WType::IQ3_S;
        case DType::Q4_0: return kern::WType::Q4_0;
        default: break;
    }
    throw_error(ErrorCode::Unsupported, "{}: weight type id {} has no HIP GEMV kernel (F32, F16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_NL, IQ4_XS)",
                op, static_cast<std::uint32_t>(t));
}

}  // namespace

namespace {

struct GemvPlan {
    kern::GemvParams p;
    kern::Launch launch;
    Rows x, y;
    Range w_range;
};

/// Validates a GemvArgs (y may be empty when y_optional) and builds the launch.
GemvPlan plan_gemv(const Target& target, const GemvArgs& a, unsigned block, bool generic, bool y_optional,
                   const char* kOp) {
    const kern::WType wt = gemv_wtype(a.wtype, kOp);
    HALO_CHECK(a.rows >= 1 && a.cols >= 1 && a.n_vec >= 1, ErrorCode::Kernel,
               "{}: rows {}, cols {}, n_vec {} must all be >= 1", kOp, a.rows, a.cols, a.n_vec);
    const unsigned be = kern::wq_block_elems(wt);
    HALO_CHECK(a.cols % be == 0, ErrorCode::Kernel, "{}: cols {} is not a multiple of the block size {}", kOp, a.cols,
               be);
    const std::uint64_t row_bytes = mul_checked(a.cols / be, kern::wq_block_bytes(wt), kOp);
    const ByteRows w = resolve_bytes(target, a.w, a.rows, row_bytes, wt == kern::WType::F32 ? 4 : 1, kOp, "w");
    GemvPlan g;
    g.x = resolve(target, a.x, a.n_vec, a.cols, kOp, "x");
    if (!(y_optional && a.y.empty())) g.y = resolve(target, a.y, a.n_vec, a.rows, kOp, "y");
    g.w_range = w.range;
    const Rows wr{nullptr, 0, w.range};
    check_disjoint(g.y, "y", {{&g.x, "x"}, {&wr, "w"}}, kOp);
    kern::GemvParams& p = g.p;
    p.type = wt;
    p.w = w.ptr;
    p.w_stride = w.stride;
    p.x = g.x.ptr;
    p.x_stride = g.x.stride;
    p.y = g.y.ptr;
    p.y_stride = g.y.stride;
    p.rows = a.rows;
    p.cols = a.cols;
    p.n_vec = a.n_vec;
    g.launch = generic ? kern::gemv_generic_launch(a.rows, a.n_vec, block)
                       : kern::gemv_wave_launch(a.rows, a.n_vec, block);
    return g;
}

void run_gemv(const Target& target, const GemvPlan& g, bool generic, const char* kOp) {
    if (generic) {
        dispatch(target, &detail::launch_gemv_generic, &detail::emulate_gemv_generic, g.p, g.launch, kOp);
    } else {
        dispatch(target, &detail::launch_gemv_wave, &detail::emulate_gemv_wave, g.p, g.launch, kOp);
    }
}

}  // namespace

void Ops::gemv(const Target& target, const GemvArgs& a) const {
    constexpr const char* kOp = "hip::gemv";
    const GemvPlan g = plan_gemv(target, a, gemv_block_, gemv_generic_, false, kOp);
    run_gemv(target, g, gemv_generic_, kOp);
}

// ---- LM head / ARGMAX -------------------------------------------------------------------

std::vector<ArgmaxResult> decode_argmax(std::span<const std::uint32_t> words) {
    HALO_CHECK(words.size() % 3 == 0, ErrorCode::Kernel, "decode_argmax: {} words is not 3 per vector", words.size());
    std::vector<ArgmaxResult> out(words.size() / 3);
    for (std::size_t v = 0; v < out.size(); ++v) {
        const std::uint32_t idx = words[3 * v];
        const std::uint32_t nan = words[3 * v + 2];
        HALO_CHECK(nan == 0, ErrorCode::Kernel, "argmax: NaN logit in vector {} (D-016)", v);
        HALO_CHECK(idx != kern::kNoIndex, ErrorCode::Kernel, "argmax: vector {} has no candidate", v);
        out[v].index = idx;
        out[v].value = std::bit_cast<float>(words[3 * v + 1]);
    }
    return out;
}

std::uint64_t Ops::lm_head_workspace_bytes(std::uint32_t rows, std::uint32_t n_vec) const {
    const kern::Launch l = gemv_generic_ ? kern::gemv_generic_launch(rows, n_vec, gemv_block_)
                                         : kern::gemv_wave_launch(rows, n_vec, gemv_block_);
    return mul_checked(mul_checked(l.grid_x, n_vec, "lm_head_workspace_bytes"), sizeof(kern::ArgPart),
                       "lm_head_workspace_bytes");
}

std::uint64_t Ops::argmax_workspace_bytes(std::uint32_t n, std::uint32_t n_vec) const {
    const kern::Launch l = kern::argmax_partial_launch(n, n_vec, argmax_block_);
    return mul_checked(mul_checked(l.grid_x, n_vec, "argmax_workspace_bytes"), sizeof(kern::ArgPart),
                       "argmax_workspace_bytes");
}

void Ops::lm_head_argmax(const Target& target, const LmHeadArgs& a) const {
    constexpr const char* kOp = "hip::lm_head_argmax";
    GemvPlan g = plan_gemv(target, a.gemv, gemv_block_, gemv_generic_, true, kOp);
    const unsigned n_parts = g.launch.grid_x;
    const Rows ws = resolve(target, a.workspace, 1, static_cast<std::uint64_t>(n_parts) * a.gemv.n_vec * 4, kOp,
                            "workspace");
    const Rows res = resolve(target, a.result, 1, static_cast<std::uint64_t>(a.gemv.n_vec) * 3, kOp, "result");
    const Rows wr{nullptr, 0, g.w_range};
    check_disjoint(ws, "workspace", {{&g.x, "x"}, {&g.y, "y"}, {&wr, "w"}, {&res, "result"}}, kOp);
    check_disjoint(res, "result", {{&g.x, "x"}, {&g.y, "y"}, {&wr, "w"}}, kOp);
    g.p.part = reinterpret_cast<kern::ArgPart*>(ws.ptr);
    g.p.n_parts = n_parts;
    run_gemv(target, g, gemv_generic_, kOp);
    kern::ArgmaxParams r;
    r.part = g.p.part;
    r.n_parts = n_parts;
    r.result = reinterpret_cast<kern::ArgResult*>(res.ptr);
    dispatch(target, &detail::launch_argmax_reduce, &detail::emulate_argmax_reduce, r,
             kern::argmax_reduce_launch(a.gemv.n_vec, argmax_block_), kOp);
}

void Ops::argmax(const Target& target, const ArgmaxArgs& a) const {
    constexpr const char* kOp = "hip::argmax";
    HALO_CHECK(a.n >= 1 && a.n_vec >= 1, ErrorCode::Kernel, "{}: n and n_vec must be >= 1", kOp);
    HALO_CHECK(a.n < kern::kNoIndex, ErrorCode::Kernel, "{}: n {} too large", kOp, a.n);
    const Rows x = resolve(target, a.logits, a.n_vec, a.n, kOp, "logits");
    const kern::Launch l1 = kern::argmax_partial_launch(a.n, a.n_vec, argmax_block_);
    const Rows ws = resolve(target, a.workspace, 1, static_cast<std::uint64_t>(l1.grid_x) * a.n_vec * 4, kOp,
                            "workspace");
    const Rows res = resolve(target, a.result, 1, static_cast<std::uint64_t>(a.n_vec) * 3, kOp, "result");
    check_disjoint(ws, "workspace", {{&x, "logits"}, {&res, "result"}}, kOp);
    check_disjoint(res, "result", {{&x, "logits"}}, kOp);
    kern::ArgmaxParams p;
    p.x = x.ptr;
    p.x_stride = x.stride;
    p.n = a.n;
    p.part = reinterpret_cast<kern::ArgPart*>(ws.ptr);
    p.n_parts = l1.grid_x;
    p.result = reinterpret_cast<kern::ArgResult*>(res.ptr);
    dispatch(target, &detail::launch_argmax_partial, &detail::emulate_argmax_partial, p, l1, kOp);
    dispatch(target, &detail::launch_argmax_reduce, &detail::emulate_argmax_reduce, p,
             kern::argmax_reduce_launch(a.n_vec, argmax_block_), kOp);
}

std::vector<ArgmaxResult> Ops::read_argmax(const Target& target, const BufferView& result, std::uint32_t n_vec) {
    constexpr const char* kOp = "hip::read_argmax";
    HALO_CHECK(n_vec >= 1, ErrorCode::Kernel, "{}: n_vec must be >= 1", kOp);
    static_cast<void>(resolve(target, result, 1, static_cast<std::uint64_t>(n_vec) * 3, kOp, "result"));
    if (target.is_device()) target.stream()->synchronize();
    std::vector<std::uint32_t> words(static_cast<std::size_t>(n_vec) * 3);
    result.buffer->download(words.data(), words.size() * 4, result.offset);
    return decode_argmax(words);
}

// ---- TOP_K --------------------------------------------------------------------------------

namespace {

/// Candidates per vector entering each round: n, then blocks * k until one chunk remains.
std::vector<std::uint32_t> topk_rounds(std::uint32_t n, std::uint32_t k) {
    std::vector<std::uint32_t> m{n};
    while (m.back() > kern::kTopkChunk) {
        const std::uint32_t blocks = (m.back() + kern::kTopkChunk - 1) / kern::kTopkChunk;
        m.push_back(blocks * k);
    }
    return m;
}

}  // namespace

std::uint64_t topk_workspace_bytes(std::uint32_t n, std::uint32_t k, std::uint32_t n_vec) {
    HALO_CHECK(k >= 1 && k <= kern::kTopkMaxK && k <= n, ErrorCode::Kernel,
               "topk_workspace_bytes: k {} not in [1, min(n {}, {})]", k, n, kern::kTopkMaxK);
    const std::vector<std::uint32_t> m = topk_rounds(n, k);
    if (m.size() == 1) return 0;
    // Two ping-pong buffers, each sized for the largest intermediate round (m[1]).
    return mul_checked(mul_checked(2ull * m[1], n_vec, "topk_workspace_bytes"), sizeof(kern::TopkPair),
                       "topk_workspace_bytes");
}

void Ops::top_k(const Target& target, const TopKArgs& a) const {
    constexpr const char* kOp = "hip::top_k";
    HALO_CHECK(a.n_vec >= 1 && a.n >= 1, ErrorCode::Kernel, "{}: n and n_vec must be >= 1", kOp);
    HALO_CHECK(a.k >= 1 && a.k <= a.n && a.k <= kern::kTopkMaxK, ErrorCode::Kernel,
               "{}: k = {} for {} entries (this backend: k <= {})", kOp, a.k, a.n, kern::kTopkMaxK);
    HALO_CHECK(a.n < kern::kNoIndex, ErrorCode::Kernel, "{}: n {} too large", kOp, a.n);
    const std::vector<std::uint32_t> m = topk_rounds(a.n, a.k);
    const Rows x = resolve(target, a.logits, a.n_vec, a.n, kOp, "logits");
    const Rows ids = resolve(target, a.ids, a.n_vec, a.k, kOp, "ids");
    const Rows vals = resolve(target, a.values, a.n_vec, a.k, kOp, "values");
    HALO_CHECK(ids.stride == vals.stride, ErrorCode::Kernel, "{}: ids and values need the same row stride", kOp);
    const Rows status = resolve_status(target, a.status, kOp);
    const std::uint64_t pairs_per_vec = m.size() > 1 ? m[1] : 0;
    Rows ws;
    if (m.size() > 1) {
        ws = resolve(target, a.workspace, 1, 2 * pairs_per_vec * a.n_vec * 2, kOp, "workspace");
    }
    check_disjoint(ids, "ids", {{&x, "logits"}, {&vals, "values"}, {&status, "status"}, {&ws, "workspace"}}, kOp);
    check_disjoint(vals, "values", {{&x, "logits"}, {&status, "status"}, {&ws, "workspace"}}, kOp);
    check_disjoint(ws, "workspace", {{&x, "logits"}, {&status, "status"}}, kOp);
    check_disjoint(status, "status", {{&x, "logits"}}, kOp);
    zero_status(target, status);
    auto* pairs = reinterpret_cast<kern::TopkPair*>(ws.ptr);
    kern::TopkPair* buf[2] = {pairs, pairs == nullptr ? nullptr : pairs + pairs_per_vec * a.n_vec};
    for (std::size_t r = 0; r < m.size(); ++r) {
        kern::TopkParams p;
        p.m = m[r];
        p.k = a.k;
        p.status = reinterpret_cast<std::uint32_t*>(status.ptr);
        if (r == 0) {
            p.src_logits = x.ptr;
            p.src_stride = x.stride;
        } else {
            p.src_pairs = buf[(r - 1) % 2];
            p.src_stride = pairs_per_vec;
        }
        if (r + 1 == m.size()) {
            p.out_ids = reinterpret_cast<std::int32_t*>(ids.ptr);
            p.out_vals = vals.ptr;
            p.out_stride = ids.stride;
        } else {
            p.dst = buf[r % 2];
            p.dst_stride = pairs_per_vec;
        }
        dispatch(target, &detail::launch_topk, &detail::emulate_topk, p, kern::topk_launch(m[r], a.n_vec, topk_block_),
                 kOp);
    }
}

// ---- RMS_NORM / PARTIAL_ROPE / SWIGLU / MUL_SIGMOID -----------------------------------------

void Ops::rms_norm(const Target& target, const RmsNormArgs& a) const {
    constexpr const char* kOp = "hip::rms_norm";
    HALO_CHECK(a.cols >= 1, ErrorCode::Kernel, "{}: cols must be >= 1", kOp);
    if (a.rows == 0) return;
    const Rows x = resolve(target, a.x, a.rows, a.cols, kOp, "x");
    const Rows w = resolve(target, a.w, 1, a.cols, kOp, "w");
    const Rows out = resolve(target, a.out, a.rows, a.cols, kOp, "out");
    check_alias_exact_or_disjoint(x, out, kOp, "x");
    check_disjoint(out, "out", {{&w, "w"}}, kOp);
    kern::NormParams p;
    p.x = x.ptr;
    p.x_stride = x.stride;
    p.w = w.ptr;
    p.out = out.ptr;
    p.out_stride = out.stride;
    p.rows = a.rows;
    p.cols = a.cols;
    p.eps = a.eps;
    dispatch(target, &detail::launch_norm, &detail::emulate_norm, p, kern::norm_launch(a.rows, rms_block_), kOp);
}

void Ops::partial_rope_neox(const Target& target, const RopeArgs& a) const {
    constexpr const char* kOp = "hip::partial_rope_neox";
    HALO_CHECK(a.rot_dims > 0 && a.rot_dims % 2 == 0, ErrorCode::Kernel, "{}: rot_dims {} must be even and > 0", kOp,
               a.rot_dims);
    HALO_CHECK(a.rot_dims <= a.head_dim, ErrorCode::Kernel, "{}: rot_dims {} > head_dim {}", kOp, a.rot_dims,
               a.head_dim);
    HALO_CHECK(a.rot_dims / 2 <= kern::kRopeMaxHalf, ErrorCode::Kernel, "{}: rot_dims {} exceeds this backend's {}", kOp,
               a.rot_dims, 2 * kern::kRopeMaxHalf);
    HALO_CHECK(a.n_heads >= 1, ErrorCode::Kernel, "{}: n_heads must be >= 1", kOp);
    if (a.n_tokens == 0) return;
    const std::uint64_t hs = a.head_stride == 0 ? a.head_dim : a.head_stride;
    HALO_CHECK(hs >= a.head_dim, ErrorCode::Kernel, "{}: head_stride {} < head_dim {}", kOp, hs, a.head_dim);
    // Row extent: the last head ends at (n_heads - 1) * stride + head_dim.
    const Rows x = resolve(target, a.x, a.n_tokens, mul_checked(a.n_heads - 1, hs, kOp) + a.head_dim, kOp, "x");
    const Rows pos = resolve(target, a.positions, 1, a.n_tokens, kOp, "positions");
    check_disjoint(x, "x", {{&pos, "positions"}}, kOp);
    kern::RopeParams p;
    p.x = x.ptr;
    p.x_stride = x.stride;
    p.pos = reinterpret_cast<const std::int32_t*>(pos.ptr);
    p.n_heads = a.n_heads;
    p.head_dim = a.head_dim;
    p.head_stride = hs;
    p.half = a.rot_dims / 2;
    for (unsigned i = 0; i < p.half; ++i) {
        // cpu::rope_inv_freq: 1 / theta^(float(2i) / rot_dims), fp32 pow and divide.
        const float e = static_cast<float>(2 * i) / static_cast<float>(a.rot_dims);
        p.inv[i] = 1.0f / std::pow(a.theta, e);
    }
    dispatch(target, &detail::launch_rope, &detail::emulate_rope, p,
             kern::rope_launch(a.n_tokens, a.n_heads, p.half, rope_block_), kOp);
}

namespace {

void eltwise(const Target& target, const EltwiseArgs& a, kern::EwOp op, unsigned block, const char* kOp) {
    if (a.rows == 0 || a.cols == 0) return;
    const Rows ra = resolve(target, a.a, a.rows, a.cols, kOp, "a");
    const Rows rb = resolve(target, a.b, a.rows, a.cols, kOp, "b");
    const Rows out = resolve(target, a.out, a.rows, a.cols, kOp, "out");
    check_alias_exact_or_disjoint(ra, out, kOp, "a");
    check_alias_exact_or_disjoint(rb, out, kOp, "b");
    kern::EwParams p;
    p.op = op;
    p.a = ra.ptr;
    p.a_stride = ra.stride;
    p.b = rb.ptr;
    p.b_stride = rb.stride;
    p.out = out.ptr;
    p.out_stride = out.stride;
    p.rows = a.rows;
    p.cols = a.cols;
    dispatch(target, &detail::launch_eltwise, &detail::emulate_eltwise, p, kern::ew_launch(a.rows, a.cols, block), kOp);
}

}  // namespace

void Ops::swiglu(const Target& target, const EltwiseArgs& a) const {
    eltwise(target, a, kern::EwOp::SwiGlu, swiglu_block_, "hip::swiglu");
}

void Ops::mul_sigmoid(const Target& target, const EltwiseArgs& a) const {
    eltwise(target, a, kern::EwOp::MulSigmoid, mulsig_block_, "hip::mul_sigmoid");
}

}  // namespace halo::hip

// ---- ATTENTION ------------------------------------------------------------------------------

namespace halo::hip {

std::uint64_t Ops::attention_workspace_bytes(std::uint32_t n_tokens, std::uint32_t n_head, std::uint32_t q_offset) const {
    if (!attn_exact_) return 0;
    constexpr const char* kOp = "attention_workspace_bytes";
    return mul_checked(mul_checked(mul_checked(n_tokens, n_head, kOp), static_cast<std::uint64_t>(q_offset) + n_tokens,
                                   kOp),
                       4, kOp);
}

void Ops::attention(const Target& target, const AttentionArgs& a) const {
    constexpr const char* kOp = "hip::attention";
    HALO_CHECK(a.n_head > 0 && a.n_kv_head > 0 && a.head_dim > 0 && a.n_head % a.n_kv_head == 0, ErrorCode::Kernel,
               "{}: bad dims n_head {} n_kv_head {} head_dim {}", kOp, a.n_head, a.n_kv_head, a.head_dim);
    HALO_CHECK(a.head_dim <= kern::kAttnMaxHeadDim, ErrorCode::Kernel, "{}: head_dim {} exceeds this backend's {}", kOp,
               a.head_dim, kern::kAttnMaxHeadDim);
    HALO_CHECK(a.n_tokens >= 1, ErrorCode::Kernel, "{}: n_tokens must be >= 1", kOp);
    HALO_CHECK(a.block_tokens >= 1 && a.n_layers >= 1 && a.layer < a.n_layers && a.n_pool_blocks >= 1,
               ErrorCode::Kernel, "{}: bad pool layout (block_tokens {}, layer {} of {}, {} blocks)", kOp,
               a.block_tokens, a.layer, a.n_layers, a.n_pool_blocks);
    HALO_CHECK(std::isfinite(a.scale), ErrorCode::Kernel, "{}: scale {} is not finite", kOp, a.scale);
    const std::uint64_t kv_dim = mul_checked(a.n_kv_head, a.head_dim, kOp);
    HALO_CHECK(kv_dim <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Kernel, "{}: kv_dim too large", kOp);
    const std::uint64_t history = static_cast<std::uint64_t>(a.q_offset) + a.n_tokens;
    HALO_CHECK(history <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Kernel, "{}: history too long", kOp);
    const std::uint64_t n_table = (history + a.block_tokens - 1) / a.block_tokens;
    const std::uint64_t block_floats = mul_checked(mul_checked(2ull * a.n_layers, a.block_tokens, kOp), kv_dim, kOp);
    const std::uint64_t hs = a.q_head_stride == 0 ? a.head_dim : a.q_head_stride;
    HALO_CHECK(hs >= a.head_dim, ErrorCode::Kernel, "{}: q_head_stride {} < head_dim {}", kOp, hs, a.head_dim);
    const Rows q = resolve(target, a.q, a.n_tokens, mul_checked(a.n_head - 1, hs, kOp) + a.head_dim, kOp, "q");
    const Rows pool = resolve(target, a.kv_pool, 1, mul_checked(block_floats, a.n_pool_blocks, kOp), kOp, "kv_pool");
    const Rows table = resolve(target, a.block_table, 1, n_table, kOp, "block_table");
    const Rows out = resolve(target, a.out, a.n_tokens, mul_checked(a.n_head, a.head_dim, kOp), kOp, "out");
    const Rows status = resolve_status(target, a.status, kOp);
    Rows ws;
    const std::uint64_t ws_bytes = attention_workspace_bytes(a.n_tokens, a.n_head, a.q_offset);
    if (ws_bytes > 0) ws = resolve(target, a.workspace, 1, ws_bytes / 4, kOp, "workspace");
    check_disjoint(out, "out",
                   {{&q, "q"}, {&pool, "kv_pool"}, {&table, "block_table"}, {&status, "status"}, {&ws, "workspace"}},
                   kOp);
    check_disjoint(ws, "workspace", {{&q, "q"}, {&pool, "kv_pool"}, {&table, "block_table"}, {&status, "status"}}, kOp);
    check_disjoint(status, "status", {{&q, "q"}, {&pool, "kv_pool"}, {&table, "block_table"}}, kOp);

    kern::AttnParams p;
    p.q = q.ptr;
    p.q_stride = q.stride;
    p.q_head_stride = hs;
    p.pool = pool.ptr;
    p.block_floats = block_floats;
    p.k_off = static_cast<std::uint64_t>(a.layer) * 2 * a.block_tokens * kv_dim;
    p.v_off = p.k_off + static_cast<std::uint64_t>(a.block_tokens) * kv_dim;
    p.block_tokens = a.block_tokens;
    p.kv_dim = static_cast<unsigned>(kv_dim);
    p.n_pool_blocks = a.n_pool_blocks;
    p.table = reinterpret_cast<const std::uint32_t*>(table.ptr);
    p.n_table = static_cast<unsigned>(n_table);
    p.n_head = a.n_head;
    p.n_kv_head = a.n_kv_head;
    p.head_dim = a.head_dim;
    p.q_offset = a.q_offset;
    p.scale = a.scale;
    p.out = out.ptr;
    p.out_stride = out.stride;
    p.scores = ws.ptr;
    p.scores_stride = history;
    p.status = reinterpret_cast<std::uint32_t*>(status.ptr);
    zero_status(target, status);
    dispatch(target, &detail::launch_attn_check, &detail::emulate_attn_check, p,
             kern::attn_check_launch(p.n_table, 256), kOp);
    const kern::Launch l = kern::attn_launch(a.n_tokens, a.n_head, attn_block_);
    if (attn_exact_) {
        dispatch(target, &detail::launch_attn_exact, &detail::emulate_attn_exact, p, l, kOp);
    } else {
        dispatch(target, &detail::launch_attn_online, &detail::emulate_attn_online, p, l, kOp);
    }
}

}  // namespace halo::hip

// ---- KV write / GET_ROWS / ADD / ADD + RMS_NORM ---------------------------------------------

namespace halo::hip {

void Ops::kv_write(const Target& target, const KvWriteArgs& a) const {
    constexpr const char* kOp = "hip::kv_write";
    HALO_CHECK(a.block_tokens >= 1 && a.n_layers >= 1 && a.layer < a.n_layers && a.n_pool_blocks >= 1 && a.kv_dim >= 1,
               ErrorCode::Kernel, "{}: bad pool layout (block_tokens {}, layer {} of {}, {} blocks, kv_dim {})", kOp,
               a.block_tokens, a.layer, a.n_layers, a.n_pool_blocks, a.kv_dim);
    if (a.n_tokens == 0) return;
    const std::uint64_t end = static_cast<std::uint64_t>(a.start) + a.n_tokens;
    HALO_CHECK(end <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Kernel, "{}: position overflow", kOp);
    const std::uint64_t n_table = (end + a.block_tokens - 1) / a.block_tokens;
    const std::uint64_t block_floats = mul_checked(mul_checked(2ull * a.n_layers, a.block_tokens, kOp), a.kv_dim, kOp);
    const Rows pool = resolve(target, a.kv_pool, 1, mul_checked(block_floats, a.n_pool_blocks, kOp), kOp, "kv_pool");
    const Rows table = resolve(target, a.block_table, 1, n_table, kOp, "block_table");
    const Rows k = resolve(target, a.k, a.n_tokens, a.kv_dim, kOp, "k");
    const Rows v = resolve(target, a.v, a.n_tokens, a.kv_dim, kOp, "v");
    const Rows status = resolve_status(target, a.status, kOp);
    check_disjoint(pool, "kv_pool", {{&table, "block_table"}, {&k, "k"}, {&v, "v"}, {&status, "status"}}, kOp);
    check_disjoint(status, "status", {{&table, "block_table"}, {&k, "k"}, {&v, "v"}}, kOp);
    kern::KvWriteParams p;
    p.pool = pool.ptr;
    p.block_floats = block_floats;
    p.k_off = static_cast<std::uint64_t>(a.layer) * 2 * a.block_tokens * a.kv_dim;
    p.v_off = p.k_off + static_cast<std::uint64_t>(a.block_tokens) * a.kv_dim;
    p.block_tokens = a.block_tokens;
    p.kv_dim = a.kv_dim;
    p.n_pool_blocks = a.n_pool_blocks;
    p.table = reinterpret_cast<const std::uint32_t*>(table.ptr);
    p.start = a.start;
    p.k = k.ptr;
    p.k_stride = k.stride;
    p.v = v.ptr;
    p.v_stride = v.stride;
    p.status = reinterpret_cast<std::uint32_t*>(status.ptr);
    zero_status(target, status);
    dispatch(target, &detail::launch_kv_write, &detail::emulate_kv_write, p,
             kern::kv_write_launch(a.n_tokens, a.kv_dim, kvw_block_), kOp);
}

void Ops::get_rows(const Target& target, const GetRowsArgs& a) const {
    constexpr const char* kOp = "hip::get_rows";
    const kern::WType wt = gemv_wtype(a.wtype, kOp);
    HALO_CHECK(a.n_rows >= 1 && a.cols >= 1, ErrorCode::Kernel, "{}: n_rows and cols must be >= 1", kOp);
    HALO_CHECK(a.n_rows <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()), ErrorCode::Kernel,
               "{}: n_rows {} exceeds int32 ids", kOp, a.n_rows);
    if (a.n_ids == 0) return;
    const unsigned be = kern::wq_block_elems(wt);
    HALO_CHECK(a.cols % be == 0, ErrorCode::Kernel, "{}: cols {} is not a multiple of the block size {}", kOp, a.cols,
               be);
    const std::uint64_t row_bytes = mul_checked(a.cols / be, kern::wq_block_bytes(wt), kOp);
    const ByteRows w = resolve_bytes(target, a.w, a.n_rows, row_bytes, wt == kern::WType::F32 ? 4 : 1, kOp, "w");
    const Rows ids = resolve(target, a.ids, 1, a.n_ids, kOp, "ids");
    const Rows out = resolve(target, a.out, a.n_ids, a.cols, kOp, "out");
    const Rows status = resolve_status(target, a.status, kOp);
    const Rows wr{nullptr, 0, w.range};
    check_disjoint(out, "out", {{&wr, "w"}, {&ids, "ids"}, {&status, "status"}}, kOp);
    check_disjoint(status, "status", {{&wr, "w"}, {&ids, "ids"}}, kOp);
    kern::GetRowsParams p;
    p.type = wt;
    p.w = w.ptr;
    p.w_stride = w.stride;
    p.n_rows = a.n_rows;
    p.cols = a.cols;
    p.ids = reinterpret_cast<const std::int32_t*>(ids.ptr);
    p.out = out.ptr;
    p.out_stride = out.stride;
    p.status = reinterpret_cast<std::uint32_t*>(status.ptr);
    zero_status(target, status);
    dispatch(target, &detail::launch_get_rows, &detail::emulate_get_rows, p,
             kern::get_rows_launch(a.n_ids, a.cols, rows_block_), kOp);
}

void Ops::add(const Target& target, const EltwiseArgs& a) const {
    eltwise(target, a, kern::EwOp::Add, add_block_, "hip::add");
}

void Ops::add_rms_norm(const Target& target, const AddRmsNormArgs& a) const {
    constexpr const char* kOp = "hip::add_rms_norm";
    HALO_CHECK(a.cols >= 1, ErrorCode::Kernel, "{}: cols must be >= 1", kOp);
    if (a.rows == 0) return;
    const Rows ra = resolve(target, a.a, a.rows, a.cols, kOp, "a");
    const Rows rb = resolve(target, a.b, a.rows, a.cols, kOp, "b");
    const Rows h = resolve(target, a.h, a.rows, a.cols, kOp, "h");
    const Rows w = resolve(target, a.w, 1, a.cols, kOp, "w");
    const Rows y = resolve(target, a.y, a.rows, a.cols, kOp, "y");
    check_alias_exact_or_disjoint(ra, h, kOp, "a");
    check_alias_exact_or_disjoint(rb, h, kOp, "b");
    check_disjoint(h, "h", {{&w, "w"}}, kOp);
    check_disjoint(y, "y", {{&ra, "a"}, {&rb, "b"}, {&h, "h"}, {&w, "w"}}, kOp);
    kern::AddNormParams p;
    p.a = ra.ptr;
    p.a_stride = ra.stride;
    p.b = rb.ptr;
    p.b_stride = rb.stride;
    p.h = h.ptr;
    p.h_stride = h.stride;
    p.w = w.ptr;
    p.y = y.ptr;
    p.y_stride = y.stride;
    p.rows = a.rows;
    p.cols = a.cols;
    p.eps = a.eps;
    dispatch(target, &detail::launch_add_norm, &detail::emulate_add_norm, p,
             kern::add_norm_launch(a.rows, addnorm_block_), kOp);
}

}  // namespace halo::hip

// ---- GDN gates / QUANT_GEMM -----------------------------------------------------------------

namespace halo::hip {

void Ops::gdn_gates(const Target& target, const GdnGateArgs& a) const {
    constexpr const char* kOp = "hip::gdn_gates";
    HALO_CHECK(a.n_v >= 1, ErrorCode::Kernel, "{}: n_v must be >= 1", kOp);
    if (a.rows == 0) return;
    const Rows b = resolve(target, a.b, a.rows, a.n_v, kOp, "b");
    const Rows al = resolve(target, a.a, a.rows, a.n_v, kOp, "a");
    const Rows dt = resolve(target, a.dt_bias, 1, a.n_v, kOp, "dt_bias");
    const Rows sa = resolve(target, a.ssm_a, 1, a.n_v, kOp, "ssm_a");
    const Rows beta = resolve(target, a.beta, a.rows, a.n_v, kOp, "beta");
    const Rows g = resolve(target, a.g, a.rows, a.n_v, kOp, "g");
    check_alias_exact_or_disjoint(b, beta, kOp, "b");
    check_alias_exact_or_disjoint(al, g, kOp, "a");
    check_disjoint(beta, "beta", {{&al, "a"}, {&dt, "dt_bias"}, {&sa, "ssm_a"}, {&g, "g"}}, kOp);
    check_disjoint(g, "g", {{&b, "b"}, {&dt, "dt_bias"}, {&sa, "ssm_a"}}, kOp);
    kern::GdnGateParams p;
    p.b = b.ptr;
    p.b_stride = b.stride;
    p.a = al.ptr;
    p.a_stride = al.stride;
    p.dt_bias = dt.ptr;
    p.ssm_a = sa.ptr;
    p.beta = beta.ptr;
    p.beta_stride = beta.stride;
    p.g = g.ptr;
    p.g_stride = g.stride;
    p.rows = a.rows;
    p.n_v = a.n_v;
    dispatch(target, &detail::launch_gdn_gate, &detail::emulate_gdn_gate, p,
             kern::gdn_gate_launch(a.rows, a.n_v, gate_block_), kOp);
}

void Ops::gemm(const Target& target, const GemvArgs& a) const {
    constexpr const char* kOp = "hip::gemm";
    const GemvPlan g = plan_gemv(target, a, gemv_block_, gemv_generic_, false, kOp);
    dispatch(target, &detail::launch_gemm, &detail::emulate_gemm, g.p, kern::gemm_launch(a.rows, a.n_vec), kOp);
}

}  // namespace halo::hip
