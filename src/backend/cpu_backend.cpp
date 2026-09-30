// CPU reference backend (ADR-001 §5.1, WS-BI-1): the Backend interface over the frozen
// halo::cpu operators. Every op validates its TensorRefs (ownership, range, alignment,
// the neutral aliasing rule), builds cpu::RowsView / PagedRows / WeightMatrix views over
// the host bytes and calls exactly the halo::cpu (or halo::tensor) function(s) its contract
// names, so results are those of the reference by construction (DR-2).

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <new>
#include <string>

#include "halo/backend/backend.h"
#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/thread_pool.h"
#include "halo/backends/cpu/weight_matrix.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"

namespace halo::backend {

namespace {

constexpr std::uint64_t kF32 = 4;

// ---------------------------------------------------------------------------------------
// Buffers and streams
// ---------------------------------------------------------------------------------------

class CpuBuffer final : public Buffer {
public:
    CpuBuffer(const Backend* owner, std::unique_ptr<std::byte[]> storage, std::uint64_t bytes)
        : owner_(owner), storage_(std::move(storage)), bytes_(bytes), data_(storage_.get()), cdata_(storage_.get()),
          writable_(true) {}
    CpuBuffer(const Backend* owner, std::span<std::byte> host)
        : owner_(owner), bytes_(host.size()), data_(host.data()), cdata_(host.data()), writable_(true) {}
    CpuBuffer(const Backend* owner, std::span<const std::byte> host)
        : owner_(owner), bytes_(host.size()), data_(nullptr), cdata_(host.data()), writable_(false) {}

    [[nodiscard]] std::uint64_t bytes() const noexcept override { return bytes_; }
    [[nodiscard]] Tier tier() const noexcept override { return Tier::Host; }
    [[nodiscard]] const Backend* backend() const noexcept override { return owner_; }
    [[nodiscard]] bool writable() const noexcept override { return writable_; }
    [[nodiscard]] const std::byte* host_data() const noexcept override { return cdata_; }
    [[nodiscard]] std::byte* host_ptr() const noexcept override { return data_; }

private:
    const Backend* owner_;
    std::unique_ptr<std::byte[]> storage_;
    std::uint64_t bytes_;
    std::byte* data_;
    const std::byte* cdata_;
    bool writable_;
};

class CpuStream final : public Stream {
public:
    explicit CpuStream(const Backend* owner) : owner_(owner) {}
    void submit() override {}
    void wait() override {}
    void abort() noexcept override {}
    [[nodiscard]] const Backend* owner() const noexcept { return owner_; }

private:
    const Backend* owner_;
};

// ---------------------------------------------------------------------------------------
// Operand resolution
// ---------------------------------------------------------------------------------------

std::uint64_t mul_u64(std::uint64_t a, std::uint64_t b, const char* op, const char* what) {
    HALO_CHECK(b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b, ErrorCode::Kernel, "{}: {} size overflows", op,
               what);
    return a * b;
}

/// A validated operand: rows of `row_bytes` bytes, `stride` bytes apart.
struct Operand {
    const std::byte* base = nullptr;
    std::byte* wbase = nullptr;  // non-null for outputs
    std::uint64_t stride = 0;
    std::uint64_t extent = 0;    // bytes from base to one past the last byte touched
    const char* name = "";

    [[nodiscard]] const float* f() const noexcept { return static_cast<const float*>(static_cast<const void*>(base)); }
    [[nodiscard]] float* wf() const noexcept { return static_cast<float*>(static_cast<void*>(wbase)); }
    [[nodiscard]] std::size_t fstride() const noexcept { return static_cast<std::size_t>(stride / kF32); }
};

enum class Access : std::uint8_t { Read, Write };

class Resolver {
public:
    Resolver(const Backend* self, const char* op) : self_(self), op_(op) {}

    /// rows x row_bytes starting at ref; `align` = required alignment of the address and
    /// stride (4 for fp32/int32, 1 for quantized bytes). An empty ref is allowed only when
    /// the operand touches no bytes.
    Operand get(const TensorRef& ref, std::uint64_t rows, std::uint64_t row_bytes, std::uint64_t align, Access access,
                const char* name) const {
        Operand o;
        o.name = name;
        const std::uint64_t dense = row_bytes;
        o.stride = ref.row_stride != 0 ? ref.row_stride : dense;
        if (rows == 0 || row_bytes == 0) {
            if (ref.empty()) return o;
        }
        HALO_CHECK(!ref.empty(), ErrorCode::Kernel, "{}: operand {} is empty", op_, name);
        const Buffer& b = *ref.buffer;
        HALO_CHECK(b.backend() == self_, ErrorCode::Kernel, "{}: operand {} belongs to another backend", op_, name);
        HALO_CHECK(b.host_data() != nullptr, ErrorCode::Kernel, "{}: operand {} is not host-addressable", op_, name);
        HALO_CHECK(access == Access::Read || b.host_ptr() != nullptr, ErrorCode::Kernel, "{}: output {} is read-only", op_,
                   name);
        HALO_CHECK(rows <= 1 || o.stride >= row_bytes, ErrorCode::Kernel, "{}: operand {} row stride {} < row bytes {}", op_,
                   name, o.stride, row_bytes);
        o.extent = rows == 0 ? 0 : mul_u64(rows - 1, o.stride, op_, name) + row_bytes;
        HALO_CHECK(o.extent >= row_bytes || rows == 0, ErrorCode::Kernel, "{}: operand {} size overflows", op_, name);
        HALO_CHECK(ref.offset <= b.bytes(), ErrorCode::Kernel, "{}: operand {} offset {} past buffer end {}", op_, name,
                   ref.offset, b.bytes());
        const std::uint64_t avail = b.bytes() - ref.offset;
        const std::uint64_t limit = ref.bytes != 0 ? ref.bytes : avail;
        HALO_CHECK(limit <= avail, ErrorCode::Kernel, "{}: operand {} range [{}, +{}) exceeds buffer of {} bytes", op_, name,
                   ref.offset, ref.bytes, b.bytes());
        HALO_CHECK(o.extent <= limit, ErrorCode::Kernel, "{}: operand {} needs {} bytes, view has {}", op_, name, o.extent,
                   limit);
        o.base = b.host_data() + ref.offset;
        if (access == Access::Write) o.wbase = b.host_ptr() + ref.offset;
        HALO_CHECK(std::bit_cast<std::uintptr_t>(o.base) % align == 0 && o.stride % align == 0, ErrorCode::Kernel,
                   "{}: operand {} is not {}-byte aligned (offset {}, stride {})", op_, name, align, ref.offset, o.stride);
        return o;
    }

    /// fp32 rows x cols.
    Operand f32(const TensorRef& ref, std::uint64_t rows, std::uint64_t cols, Access access, const char* name) const {
        return get(ref, rows, mul_u64(cols, kF32, op_, name), kF32, access, name);
    }

    /// Rejects any overlap of `out` with `other`; identical views are allowed only if
    /// `allow_identical`.
    void no_overlap(const Operand& out, const Operand& other, bool allow_identical = false) const {
        if (out.extent == 0 || other.extent == 0) return;
        const auto lo1 = std::bit_cast<std::uintptr_t>(out.base);
        const auto lo2 = std::bit_cast<std::uintptr_t>(other.base);
        const bool overlap = lo1 < lo2 + other.extent && lo2 < lo1 + out.extent;
        if (!overlap) return;
        const bool identical = lo1 == lo2 && out.stride == other.stride && out.extent == other.extent;
        HALO_CHECK(allow_identical && identical, ErrorCode::Kernel, "{}: output {} overlaps operand {}{}", op_, out.name,
                   other.name, allow_identical ? " (only an identical view may alias)" : "");
    }

    [[nodiscard]] const char* op() const noexcept { return op_; }

private:
    const Backend* self_;
    const char* op_;
};

cpu::ConstRows crows(const Operand& o, std::size_t rows, std::size_t cols) {
    return rows == 0 || cols == 0 ? cpu::ConstRows(o.f(), rows, cols, cols) : cpu::ConstRows(o.f(), rows, cols, o.fstride());
}
cpu::Rows wrows(const Operand& o, std::size_t rows, std::size_t cols) {
    return rows == 0 || cols == 0 ? cpu::Rows(o.wf(), rows, cols, cols) : cpu::Rows(o.wf(), rows, cols, o.fstride());
}

void check_kernel(const KernelChoice& k, OpId op) {
    HALO_CHECK(k.variant_id == 0, ErrorCode::Config, "{}: unknown kernel variant id {} on the cpu backend (only 0 = reference)",
               op_name(op), k.variant_id);
}

void check_status(const StatusRef& s, OpId op) {
    HALO_CHECK(s.empty(), ErrorCode::Unsupported,
               "{}: status words are not implemented yet (ADR-001 WS-BI-2); the cpu backend raises data errors synchronously",
               op_name(op));
}

void check_nonzero(std::uint64_t v, const char* op, const char* what) {
    HALO_CHECK(v > 0, ErrorCode::Kernel, "{}: {} is 0", op, what);
}

std::uint64_t weight_row_bytes(DType t, std::uint32_t cols, const char* op) {
    HALO_CHECK(dequant_supported(t), ErrorCode::Unsupported, "{}: no dequantization for {}", op, traits(t).name);
    return tensor::row_bytes(t, cols);
}

/// The weight matrix exactly as models::weight_matrix builds it over the same bytes: F32 is
/// a dense in-place view, other types dequantize rows with tensor::dequantize_row.
cpu::WeightMatrix weight_matrix(DType type, const Operand& w, std::size_t first_row, std::size_t rows, std::size_t cols,
                                std::uint64_t rb) {
    const std::byte* base = w.base + first_row * w.stride;
    if (type == DType::F32) {
        return cpu::WeightMatrix::dense(
            cpu::ConstRows(static_cast<const float*>(static_cast<const void*>(base)), rows, cols, w.fstride()));
    }
    const std::uint64_t stride = w.stride;
    if (stride == rb) {
        return cpu::WeightMatrix::dequantized(rows, cols, [type, base, rb, cols](std::size_t first, std::size_t n, std::span<float> out) {
            tensor::dequantize_row(type, base + first * rb, out.data(), static_cast<std::int64_t>(n * cols));
        });
    }
    return cpu::WeightMatrix::dequantized(rows, cols, [type, base, stride, cols](std::size_t first, std::size_t n, std::span<float> out) {
        for (std::size_t r = 0; r < n; ++r) {
            tensor::dequantize_row(type, base + (first + r) * stride, out.data() + r * cols, static_cast<std::int64_t>(cols));
        }
    });
}

constexpr std::array<VariantInfo, kOpCount> kVariants{{
    {0, "reference", OpId::GetRows, {}},     {0, "reference", OpId::RmsNorm, {}},
    {0, "reference", OpId::AddRmsNorm, {}},  {0, "reference", OpId::Gemv, {}},
    {0, "reference", OpId::GdnGates, {}},    {0, "reference", OpId::Conv1dSilu, {}},
    {0, "reference", OpId::GatedDeltaRule, {}}, {0, "reference", OpId::GatedRmsNorm, {}},
    {0, "reference", OpId::PartialRope, {}}, {0, "reference", OpId::KvWrite, {}},
    {0, "reference", OpId::Attention, {}},   {0, "reference", OpId::Swiglu, {}},
    {0, "reference", OpId::MulSigmoid, {}},  {0, "reference", OpId::Add, {}},
    {0, "reference", OpId::LmHead, {}},      {0, "reference", OpId::Argmax, {}},
    {0, "reference", OpId::TopK, {}},        {0, "reference", OpId::Copy, {}},
}};

// ---------------------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------------------

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(cpu::ThreadPool* pool) : pool_(pool) {}

    [[nodiscard]] Kind kind() const noexcept override { return Kind::Cpu; }
    [[nodiscard]] std::string describe() const override {
        return std::format("cpu reference (halo::cpu, fp32, {} thread(s))", pool_ != nullptr ? pool_->size() : 1);
    }
    [[nodiscard]] Limits limits() const noexcept override {
        constexpr auto kMax = std::numeric_limits<std::uint32_t>::max();
        // chunk: cpu::gated_delta_rule_chunked accepts [1, 1024].
        Limits l{kMax, 1024, kMax, kMax, kMax, kMax, true};
        l.lm_head_ids = true;
        return l;
    }

    std::unique_ptr<Buffer> allocate(std::uint64_t bytes, Tier /*tier: host memory only*/) override {
        HALO_CHECK(bytes <= std::numeric_limits<std::size_t>::max(), ErrorCode::Memory, "cpu backend: {} bytes", bytes);
        std::unique_ptr<std::byte[]> storage;
        try {
            storage = std::make_unique<std::byte[]>(std::max<std::size_t>(static_cast<std::size_t>(bytes), 1));  // zero-filled
        } catch (const std::bad_alloc&) {
            throw_error(ErrorCode::Memory, "cpu backend: cannot allocate {} bytes", bytes);
        }
        return std::make_unique<CpuBuffer>(this, std::move(storage), bytes);
    }
    std::unique_ptr<Buffer> import_host(std::span<std::byte> bytes) override {
        return std::make_unique<CpuBuffer>(this, bytes);
    }
    std::unique_ptr<Buffer> import_host_readonly(std::span<const std::byte> bytes) override {
        return std::make_unique<CpuBuffer>(this, bytes);
    }
    std::unique_ptr<Stream> create_stream() override { return std::make_unique<CpuStream>(this); }

    void upload(Stream& s, TensorRef dst, std::span<const std::byte> src) override {
        check_stream(s, "upload");
        const Resolver r(this, "upload");
        const Operand d = r.get(dst, 1, src.size(), 1, Access::Write, "dst");
        if (!src.empty()) std::memcpy(d.wbase, src.data(), src.size());
    }
    void download(Stream& s, TensorRef src, std::span<std::byte> dst) override {
        check_stream(s, "download");
        const Resolver r(this, "download");
        const Operand o = r.get(src, 1, dst.size(), 1, Access::Read, "src");
        if (!dst.empty()) std::memcpy(dst.data(), o.base, dst.size());
    }

    [[nodiscard]] std::span<const VariantInfo> variants(OpId op, std::string_view /*form: one variant covers all*/) const override {
        const auto i = static_cast<std::size_t>(op);
        HALO_CHECK(i < kVariants.size(), ErrorCode::Api, "variants: unknown op id {}", i);
        return std::span<const VariantInfo>(&kVariants[i], 1);
    }

    // ---- ops ---------------------------------------------------------------------------

    void get_rows(Stream& s, const GetRowsArgs& a) override {
        check_stream(s, "GET_ROWS");
        check_kernel(a.kernel, OpId::GetRows);
        check_status(a.status, OpId::GetRows);
        const Resolver r(this, "GET_ROWS");
        const std::uint64_t rb = weight_row_bytes(a.type, a.cols, r.op());
        const Operand table = r.get(a.table, a.n_rows, rb, a.type == DType::F32 ? kF32 : 1, Access::Read, "table");
        const Operand ids = r.get(a.ids, 1, mul_u64(a.n_ids, 4, r.op(), "ids"), 4, Access::Read, "ids");
        const Operand out = r.f32(a.out, a.n_ids, a.cols, Access::Write, "out");
        r.no_overlap(out, table);
        r.no_overlap(out, ids);
        const auto* id = static_cast<const std::int32_t*>(static_cast<const void*>(ids.base));
        for (std::uint32_t i = 0; i < a.n_ids; ++i) {
            HALO_CHECK(id[i] >= 0 && static_cast<std::uint32_t>(id[i]) < a.n_rows, ErrorCode::Kernel,
                       "GET_ROWS: id {} at {} outside [0, {})", id[i], i, a.n_rows);
        }
        for (std::uint32_t i = 0; i < a.n_ids; ++i) {
            const std::byte* row = table.base + static_cast<std::uint64_t>(id[i]) * table.stride;
            tensor::dequantize_row(a.type, std::span<const std::byte>(row, rb), std::span<float>(out.wf() + i * out.fstride(), a.cols));
        }
    }

    void rms_norm(Stream& s, const RmsNormArgs& a) override {
        check_stream(s, "RMS_NORM");
        check_kernel(a.kernel, OpId::RmsNorm);
        const Resolver r(this, "RMS_NORM");
        const Operand x = r.f32(a.x, a.rows, a.cols, Access::Read, "x");
        const Operand w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, w);
        cpu::rms_norm(crows(x, a.rows, a.cols), std::span<const float>(w.f(), a.cols), a.eps, wrows(out, a.rows, a.cols), pool_);
    }

    void add_rms_norm(Stream& s, const AddRmsNormArgs& a) override {
        check_stream(s, "ADD_RMS_NORM");
        check_kernel(a.kernel, OpId::AddRmsNorm);
        const Resolver r(this, "ADD_RMS_NORM");
        const Operand x = r.f32(a.a, a.rows, a.cols, Access::Read, "a");
        const Operand y = r.f32(a.b, a.rows, a.cols, Access::Read, "b");
        const Operand h = r.f32(a.h, a.rows, a.cols, Access::Write, "h");
        const Operand w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(h, x, true);
        r.no_overlap(h, y);
        r.no_overlap(h, w);
        r.no_overlap(out, x);
        r.no_overlap(out, y);
        r.no_overlap(out, h);
        r.no_overlap(out, w);
        cpu::add(crows(x, a.rows, a.cols), crows(y, a.rows, a.cols), wrows(h, a.rows, a.cols), pool_);
        cpu::rms_norm(crows(h, a.rows, a.cols), std::span<const float>(w.f(), a.cols), a.eps, wrows(out, a.rows, a.cols), pool_);
    }

    void gemv(Stream& s, const GemvArgs& a) override {
        check_stream(s, "MATMUL");
        check_kernel(a.kernel, OpId::Gemv);
        HALO_CHECK(a.norm_w.empty(), ErrorCode::Unsupported, "MATMUL: a fused norm (GemvArgs::norm_w) is Vulkan-only");
        const Resolver r(this, "MATMUL");
        const std::uint64_t rb = weight_row_bytes(a.wtype, a.cols, r.op());
        const Operand w = r.get(a.w, a.rows, rb, a.wtype == DType::F32 ? kF32 : 1, Access::Read, "w");
        const Operand x = r.f32(a.x, a.n_vec, a.cols, Access::Read, "x");
        const Operand y = r.f32(a.y, a.n_vec, a.rows, Access::Write, "y");
        r.no_overlap(y, x);
        r.no_overlap(y, w);
        cpu::matmul(crows(x, a.n_vec, a.cols), weight_matrix(a.wtype, w, 0, a.rows, a.cols, rb), wrows(y, a.n_vec, a.rows), pool_);
    }

    void gdn_gates(Stream& s, const GdnGateArgs& a) override {
        check_stream(s, "GDN_GATES");
        check_kernel(a.kernel, OpId::GdnGates);
        const Resolver r(this, "GDN_GATES");
        const std::size_t T = a.n_tokens, nv = a.n_heads;
        const Operand alpha = r.f32(a.alpha, T, nv, Access::Read, "alpha");
        const Operand beta = r.f32(a.beta, T, nv, Access::Read, "beta");
        const Operand dt = r.f32(a.dt_bias, 1, nv, Access::Read, "dt_bias");
        const Operand av = r.f32(a.a, 1, nv, Access::Read, "a");
        const Operand g = r.f32(a.g_out, T, nv, Access::Write, "g_out");
        const Operand bo = r.f32(a.beta_out, T, nv, Access::Write, "beta_out");
        for (const Operand* o : {&g, &bo}) {
            r.no_overlap(*o, alpha);
            r.no_overlap(*o, beta);
            r.no_overlap(*o, dt);
            r.no_overlap(*o, av);
        }
        r.no_overlap(g, bo);
        // The qwen35 sequence (ADR §5.4): beta = sigmoid(beta); g = a * softplus(alpha + dt_bias).
        cpu::sigmoid(crows(beta, T, nv), wrows(bo, T, nv), pool_);
        const float* dtb = dt.f();
        const float* ssm_a = av.f();
        for (std::size_t t = 0; t < T; ++t) {
            const float* in = alpha.f() + t * alpha.fstride();
            float* o = g.wf() + t * g.fstride();
            for (std::size_t j = 0; j < nv; ++j) o[j] = in[j] + dtb[j];
        }
        cpu::softplus(crows(g, T, nv), wrows(g, T, nv), pool_);
        for (std::size_t t = 0; t < T; ++t) {
            float* o = g.wf() + t * g.fstride();
            for (std::size_t j = 0; j < nv; ++j) o[j] = ssm_a[j] * o[j];
        }
    }

    void conv1d_silu(Stream& s, const Conv1dArgs& a) override {
        check_stream(s, "CONV1D_SHORT");
        check_kernel(a.kernel, OpId::Conv1dSilu);
        const Resolver r(this, "CONV1D_SHORT");
        check_nonzero(a.kernel_size, r.op(), "kernel_size");
        const std::size_t T = a.n_tokens, C = a.channels, K = a.kernel_size;
        const Operand x = r.f32(a.x, T, C, Access::Read, "x");
        const Operand w = r.f32(a.weight, C, K, Access::Read, "weight");
        const Operand out = r.f32(a.out, T, C, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, w);
        const std::uint64_t state_floats = mul_u64(K - 1, C, r.op(), "conv_state");
        if (a.ring) {
            // ADR-001 §5.3: conv_state is the whole slab of P dense conv states.
            const Operand slab = r.f32(a.conv_state, 1, mul_u64(a.ring->p, state_floats, r.op(), "slab"),
                                       Access::Write, "conv_state slab");
            HALO_CHECK(a.state_slots.empty(), ErrorCode::Kernel,
                       "CONV1D_SHORT: with a state ring, state_slots is derived from the slab and must be empty");
            r.no_overlap(slab, x);
            r.no_overlap(slab, w);
            r.no_overlap(out, slab);
            const cpu::StateRingView ring{std::span<float>(slab.wf(), slab.extent / kF32), a.ring->p, a.ring->live,
                                          a.n_slots};
            cpu::causal_conv1d_silu(crows(x, T, C), crows(w, C, K), ring, wrows(out, T, C), pool_);
            return;
        }
        const Operand st = r.f32(a.conv_state, K - 1, C, Access::Write, "conv_state");
        const std::uint64_t slot_floats = mul_u64(mul_u64(a.n_slots, K - 1, r.op(), "slots"), C, r.op(), "slots");
        const Operand sl = r.f32(a.state_slots, 1, slot_floats, Access::Write, "state_slots");
        HALO_CHECK(st.extent == 0 || st.stride == C * kF32, ErrorCode::Kernel, "CONV1D_SHORT: conv_state must be dense");
        r.no_overlap(out, st);
        r.no_overlap(out, sl);
        r.no_overlap(st, x);
        r.no_overlap(st, w);
        r.no_overlap(sl, x);
        r.no_overlap(sl, w);
        r.no_overlap(sl, st);
        cpu::causal_conv1d_silu(crows(x, T, C), crows(w, C, K), wrows(st, K - 1, C), wrows(out, T, C), pool_,
                                std::span<float>(sl.wf(), slot_floats));
    }

    void gated_delta_rule(Stream& s, const GdnArgs& a) override {
        check_stream(s, "GATED_DELTANET");
        check_kernel(a.kernel, OpId::GatedDeltaRule);
        check_status(a.status, OpId::GatedDeltaRule);
        const Resolver r(this, "GATED_DELTANET");
        const std::size_t T = a.n_tokens;
        const std::size_t qk_cols = static_cast<std::size_t>(a.n_k) * a.d_k;
        const std::size_t v_cols = static_cast<std::size_t>(a.n_v) * a.d_v;
        const std::uint64_t state_floats = mul_u64(mul_u64(a.n_v, a.d_k, r.op(), "state"), a.d_v, r.op(), "state");
        const Operand q = r.f32(a.q, T, qk_cols, Access::Read, "q");
        const Operand k = r.f32(a.k, T, qk_cols, Access::Read, "k");
        const Operand v = r.f32(a.v, T, v_cols, Access::Read, "v");
        const Operand g = r.f32(a.g, T, a.n_v, Access::Read, "g");
        const Operand beta = r.f32(a.beta, T, a.n_v, Access::Read, "beta");
        const Operand out = r.f32(a.out, T, v_cols, Access::Write, "out");
        const cpu::GdnDims dims{a.n_k, a.n_v, a.d_k, a.d_v,
                                a.mapping == GdnHeadMapping::Tiled ? cpu::GdnHeadMapping::Tiled : cpu::GdnHeadMapping::Grouped};
        const cpu::GdnInputs in{crows(q, T, qk_cols), crows(k, T, qk_cols), crows(v, T, v_cols), crows(g, T, a.n_v),
                                crows(beta, T, a.n_v)};
        const cpu::GdnQkParams qk{.qk_l2norm = a.qk_l2norm, .q_scale = a.q_scale};
        if (a.ring) {
            // ADR-001 §5.3: state is the whole slab of P dense states; slots are derived.
            const Operand slab =
                r.f32(a.state, 1, mul_u64(a.ring->p, state_floats, r.op(), "slab"), Access::Write, "state slab");
            HALO_CHECK(a.state_slots.empty(), ErrorCode::Kernel,
                       "GATED_DELTANET: with a state ring, state_slots is derived from the slab and must be empty");
            for (const Operand* i : {&q, &k, &v, &g, &beta}) r.no_overlap(slab, *i);
            r.no_overlap(out, slab);
            const cpu::StateRingView ring{std::span<float>(slab.wf(), slab.extent / kF32), a.ring->p, a.ring->live,
                                          a.n_slots};
            if (a.form == GdnForm::Chunked) {
                cpu::gated_delta_rule_chunked(dims, in, ring, wrows(out, T, v_cols), qk, a.chunk_size, pool_);
            } else {
                cpu::gated_delta_rule_recurrent(dims, in, ring, wrows(out, T, v_cols), qk, pool_);
            }
            return;
        }
        const Operand st = r.f32(a.state, 1, state_floats, Access::Write, "state");
        const std::uint64_t slot_floats = mul_u64(a.n_slots, state_floats, r.op(), "slots");
        const Operand sl = r.f32(a.state_slots, 1, slot_floats, Access::Write, "state_slots");
        for (const Operand* o : {&st, &sl, &out}) {
            for (const Operand* i : {&q, &k, &v, &g, &beta}) r.no_overlap(*o, *i);
        }
        r.no_overlap(out, st);
        r.no_overlap(out, sl);
        r.no_overlap(sl, st);
        const std::span<float> state(st.wf(), state_floats);
        const std::span<float> slots(sl.wf(), slot_floats);
        if (a.form == GdnForm::Chunked) {
            cpu::gated_delta_rule_chunked(dims, in, state, wrows(out, T, v_cols), qk, a.chunk_size, pool_, slots);
        } else {
            cpu::gated_delta_rule_recurrent(dims, in, state, wrows(out, T, v_cols), qk, pool_, slots);
        }
    }

    void gated_rms_norm(Stream& s, const GatedNormArgs& a) override {
        check_stream(s, "GATED_NORM");
        check_kernel(a.kernel, OpId::GatedRmsNorm);
        const Resolver r(this, "GATED_NORM");
        const Operand x = r.f32(a.x, a.rows, a.cols, Access::Read, "x");
        const Operand z = r.f32(a.z, a.rows, a.cols, Access::Read, "z");
        const Operand w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, z);
        r.no_overlap(out, w);
        cpu::gated_rms_norm(crows(x, a.rows, a.cols), std::span<const float>(w.f(), a.cols), crows(z, a.rows, a.cols), a.eps,
                            wrows(out, a.rows, a.cols), pool_);
    }

    void partial_rope(Stream& s, const RopeArgs& a) override {
        check_stream(s, "PARTIAL_ROPE");
        check_kernel(a.kernel, OpId::PartialRope);
        const Resolver r(this, "PARTIAL_ROPE");
        check_nonzero(a.n_heads, r.op(), "n_heads");
        const std::size_t hs = a.head_stride == 0 ? a.head_dim : a.head_stride;
        HALO_CHECK(hs >= a.head_dim, ErrorCode::Kernel, "PARTIAL_ROPE: head_stride {} < head_dim {}", hs, a.head_dim);
        const std::size_t row_cols = (a.n_heads - 1) * hs + a.head_dim;
        const Operand x = r.f32(a.x, a.n_tokens, row_cols, Access::Write, "x");
        const Operand pos = r.get(a.positions, 1, mul_u64(a.n_tokens, 4, r.op(), "positions"), 4, Access::Read, "positions");
        r.no_overlap(x, pos);
        const std::span<const std::int32_t> positions(static_cast<const std::int32_t*>(static_cast<const void*>(pos.base)),
                                                      a.n_tokens);
        if (hs == a.head_dim) {
            cpu::partial_rope_neox(wrows(x, a.n_tokens, row_cols), a.n_heads, a.head_dim, positions, a.rot_dims, a.theta, pool_);
            return;
        }
        // Head-strided operand (ADR §5.2): one dense single-head call per head.
        for (std::size_t h = 0; h < a.n_heads; ++h) {
            const cpu::Rows xh(x.wf() + h * hs, a.n_tokens, a.head_dim, x.fstride());
            cpu::partial_rope_neox(xh, 1, a.head_dim, positions, a.rot_dims, a.theta, pool_);
        }
    }

    /// Validates the pool image and the block table; returns the table.
    struct PoolView {
        Operand pool;
        const std::uint32_t* table = nullptr;
        std::uint64_t block_floats = 0;
    };
    PoolView pool_view(const Resolver& r, const TensorRef& pool, const TensorRef& table, std::uint32_t n_pool_blocks,
                       std::uint32_t n_layers, std::uint32_t layer, std::uint32_t block_tokens, std::uint32_t kv_dim,
                       std::uint32_t n_table, std::uint64_t rows_needed, Access access) const {
        check_nonzero(block_tokens, r.op(), "block_tokens");
        check_nonzero(n_layers, r.op(), "n_layers");
        HALO_CHECK(layer < n_layers, ErrorCode::Kernel, "{}: layer {} of {}", r.op(), layer, n_layers);
        PoolView p;
        p.block_floats =
            mul_u64(mul_u64(mul_u64(n_layers, 2, r.op(), "pool"), block_tokens, r.op(), "pool"), kv_dim, r.op(), "pool");
        p.pool = r.f32(pool, n_pool_blocks, p.block_floats, access, "kv_pool");
        HALO_CHECK(p.pool.extent == 0 || p.pool.stride == p.block_floats * kF32, ErrorCode::Kernel, "{}: kv_pool must be dense",
                   r.op());
        const Operand t = r.get(table, 1, mul_u64(n_table, 4, r.op(), "block_table"), 4, Access::Read, "block_table");
        const std::uint64_t needed_blocks = (rows_needed + block_tokens - 1) / block_tokens;
        HALO_CHECK(needed_blocks <= n_table, ErrorCode::Kernel, "{}: {} rows need {} blocks, table has {}", r.op(), rows_needed,
                   needed_blocks, n_table);
        p.table = static_cast<const std::uint32_t*>(static_cast<const void*>(t.base));
        for (std::uint64_t b = 0; b < needed_blocks; ++b) {
            HALO_CHECK(p.table[b] < n_pool_blocks, ErrorCode::Kernel, "{}: block table entry {} = {} outside the pool of {}",
                       r.op(), b, p.table[b], n_pool_blocks);
        }
        return p;
    }

    void kv_write(Stream& s, const KvWriteArgs& a) override {
        check_stream(s, "KV_WRITE");
        check_kernel(a.kernel, OpId::KvWrite);
        check_status(a.status, OpId::KvWrite);
        HALO_CHECK(a.kv_type == KvType::F32, ErrorCode::Unsupported,
                   "KV_WRITE: KV type {} is implemented only on the Vulkan backend (the CPU backend stores fp32)",
                   to_string(a.kv_type));
        const Resolver r(this, "KV_WRITE");
        const std::uint64_t end = static_cast<std::uint64_t>(a.start) + a.n_tokens;
        const PoolView p = pool_view(r, a.kv_pool, a.block_table, a.n_pool_blocks, a.n_layers, a.layer, a.block_tokens,
                                     a.kv_dim, a.n_block_table, end, Access::Write);
        const Operand k = r.f32(a.k, a.n_tokens, a.kv_dim, Access::Read, "k");
        const Operand v = r.f32(a.v, a.n_tokens, a.kv_dim, Access::Read, "v");
        r.no_overlap(p.pool, k);
        r.no_overlap(p.pool, v);
        const std::size_t bt = a.block_tokens, kvd = a.kv_dim;
        for (std::size_t i = 0; i < a.n_tokens; ++i) {
            const std::size_t row = a.start + i;
            float* blk = p.pool.wf() + static_cast<std::size_t>(p.table[row / bt]) * p.block_floats;
            const std::size_t off = (row % bt) * kvd;
            // Exactly kv_cache::SequenceKv::write: block[layer][K|V][token][kv_dim].
            std::memcpy(blk + (a.layer * 2 + 0) * bt * kvd + off, k.f() + i * k.fstride(), kvd * sizeof(float));
            std::memcpy(blk + (a.layer * 2 + 1) * bt * kvd + off, v.f() + i * v.fstride(), kvd * sizeof(float));
        }
    }

    void attention(Stream& s, const AttentionArgs& a) override {
        check_stream(s, "ATTENTION");
        check_kernel(a.kernel, OpId::Attention);
        check_status(a.status, OpId::Attention);
        HALO_CHECK(a.kv_type == KvType::F32, ErrorCode::Unsupported,
                   "ATTENTION: KV type {} is implemented only on the Vulkan backend (the CPU backend stores fp32)",
                   to_string(a.kv_type));
        const Resolver r(this, "ATTENTION");
        check_nonzero(a.n_head, r.op(), "n_head");
        check_nonzero(a.n_kv_head, r.op(), "n_kv_head");
        HALO_CHECK(a.n_head % a.n_kv_head == 0, ErrorCode::Kernel, "ATTENTION: n_head {} not a multiple of n_kv_head {}",
                   a.n_head, a.n_kv_head);
        const std::size_t hd = a.head_dim, T = a.n_tokens;
        const std::size_t kvd = static_cast<std::size_t>(a.n_kv_head) * hd;
        const std::size_t qhs = a.q_head_stride == 0 ? hd : a.q_head_stride;
        HALO_CHECK(qhs >= hd, ErrorCode::Kernel, "ATTENTION: q_head_stride {} < head_dim {}", qhs, hd);
        const std::size_t q_cols = (a.n_head - 1) * qhs + hd;
        const std::uint64_t rows = static_cast<std::uint64_t>(a.q_offset) + T;
        const PoolView p = pool_view(r, a.kv_pool, a.block_table, a.n_pool_blocks, a.n_layers, a.layer, a.block_tokens,
                                     static_cast<std::uint32_t>(kvd), a.n_block_table, rows, Access::Read);
        const Operand q = r.f32(a.q, T, q_cols, Access::Read, "q");
        const Operand out = r.f32(a.out, T, static_cast<std::uint64_t>(a.n_head) * hd, Access::Write, "out");
        r.no_overlap(out, q);
        r.no_overlap(out, p.pool);
        const std::size_t bt = a.block_tokens;
        const std::size_t nb = static_cast<std::size_t>((rows + bt - 1) / bt);
        std::vector<const float*> kt(nb), vt(nb);
        for (std::size_t b = 0; b < nb; ++b) {
            const float* blk = p.pool.f() + static_cast<std::size_t>(p.table[b]) * p.block_floats;
            kt[b] = blk + (a.layer * 2 + 0) * bt * kvd;
            vt[b] = blk + (a.layer * 2 + 1) * bt * kvd;
        }
        if (!a.tree_parent.empty()) {
            // Tree attention (HALO_MTP_TREE): T parents; the row order of the T new rows is the history order.
            HALO_CHECK(qhs == hd, ErrorCode::Unsupported, "ATTENTION: tree attention needs a dense q (q_head_stride 0)");
            const Operand par = r.get(a.tree_parent, 1, mul_u64(T, 4, r.op(), "tree_parent"), 4, Access::Read, "tree_parent");
            r.no_overlap(out, par);
            const std::span<const std::int32_t> parents(static_cast<const std::int32_t*>(static_cast<const void*>(par.base)), T);
            const cpu::PagedRows keys = cpu::PagedRows::paged(kt, bt, rows, kvd, kvd);
            const cpu::PagedRows vals = cpu::PagedRows::paged(vt, bt, rows, kvd, kvd);
            cpu::attention_gqa_tree(cpu::AttentionDims{a.n_head, a.n_kv_head, hd}, crows(q, T, q_cols), keys, vals, a.q_offset,
                                    a.scale, parents, wrows(out, T, static_cast<std::size_t>(a.n_head) * hd), pool_);
            return;
        }
        if (qhs == hd) {
            const cpu::PagedRows keys = cpu::PagedRows::paged(kt, bt, rows, kvd, kvd);
            const cpu::PagedRows vals = cpu::PagedRows::paged(vt, bt, rows, kvd, kvd);
            cpu::attention_gqa(cpu::AttentionDims{a.n_head, a.n_kv_head, hd}, crows(q, T, q_cols), keys, vals, a.q_offset, a.scale,
                               wrows(out, T, static_cast<std::size_t>(a.n_head) * hd), pool_);
            return;
        }
        // Head-strided q (ADR §5.2): one single-head call per head over its KV head's columns.
        const std::size_t group = a.n_head / a.n_kv_head;
        std::vector<const float*> kh(nb), vh(nb);
        for (std::size_t h = 0; h < a.n_head; ++h) {
            const std::size_t kv_head = h / group;
            for (std::size_t b = 0; b < nb; ++b) {
                kh[b] = kt[b] + kv_head * hd;
                vh[b] = vt[b] + kv_head * hd;
            }
            const cpu::PagedRows keys = cpu::PagedRows::paged(kh, bt, rows, hd, kvd);
            const cpu::PagedRows vals = cpu::PagedRows::paged(vh, bt, rows, hd, kvd);
            const cpu::ConstRows qh(q.f() + h * qhs, T, hd, q.fstride());
            const cpu::Rows oh(out.wf() + h * hd, T, hd, out.fstride());
            cpu::attention_gqa(cpu::AttentionDims{1, 1, hd}, qh, keys, vals, a.q_offset, a.scale, oh, pool_);
        }
    }

    void eltwise(Stream& s, const EltwiseArgs& a, OpId op, bool allow_out_eq_a) {
        const std::string_view name = op_name(op);
        check_stream(s, name.data());
        check_kernel(a.kernel, op);
        const Resolver r(this, name.data());
        const Operand x = r.f32(a.a, a.rows, a.cols, Access::Read, "a");
        const Operand y = r.f32(a.b, a.rows, a.cols, Access::Read, "b");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x, allow_out_eq_a);
        r.no_overlap(out, y);
        const cpu::ConstRows ca = crows(x, a.rows, a.cols), cb = crows(y, a.rows, a.cols);
        const cpu::Rows o = wrows(out, a.rows, a.cols);
        switch (op) {
            case OpId::Swiglu: cpu::swiglu(ca, cb, o, pool_); break;
            case OpId::MulSigmoid: cpu::mul_sigmoid(ca, cb, o, pool_); break;
            default: cpu::add(ca, cb, o, pool_); break;
        }
    }
    void swiglu(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::Swiglu, false); }
    void mul_sigmoid(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::MulSigmoid, false); }
    void add(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::Add, true); }

    void lm_head(Stream& s, const LmHeadArgs& a) override {
        check_stream(s, "LM_HEAD");
        check_kernel(a.kernel, OpId::LmHead);
        check_kernel(a.gemv.kernel, OpId::LmHead);
        check_status(a.status, OpId::LmHead);
        const Resolver r(this, "LM_HEAD");
        const GemvArgs& g = a.gemv;
        const std::uint64_t rb = weight_row_bytes(g.wtype, g.cols, r.op());
        const Operand w = r.get(g.w, g.rows, rb, g.wtype == DType::F32 ? kF32 : 1, Access::Read, "w");
        const Operand x = r.f32(g.x, g.n_vec, g.cols, Access::Read, "x");
        const bool want_logits = !g.y.empty();
        const Operand y = want_logits ? r.f32(g.y, g.n_vec, g.rows, Access::Write, "logits") : Operand{};
        const Operand res = r.get(a.result, 1, mul_u64(g.n_vec, kArgmaxResultBytes, r.op(), "result"), 4, Access::Write, "result");
        for (const Operand* o : {&y, &res}) {
            r.no_overlap(*o, x);
            r.no_overlap(*o, w);
        }
        r.no_overlap(res, y);
        Operand ids_out{};
        if (!a.ids.empty()) {
            ids_out = r.get(a.ids, 1, mul_u64(g.n_vec, 4, r.op(), "ids"), 4, Access::Write, "ids");
            for (const Operand* o : {&x, &w, &y, &res}) r.no_overlap(ids_out, *o);
        }
        const std::size_t n = g.n_vec, E = g.cols, n_vocab = g.rows;
        if (n == 0) return;
        // M2: rows at/after valid_rows (GGUF LM-head padding past the tokenizer's real
        // vocabulary) are excluded from the argmax. The full logits row (gemv.y), if
        // requested, is unaffected -- every row still gets its raw value.
        const std::size_t vocab_limit = a.valid_rows > 0 ? std::min<std::size_t>(a.valid_rows, n_vocab) : n_vocab;
        // Exactly Impl::head: one pass over the head matrix in slabs; the argmax is taken over
        // exactly the values matmul produces (ties: lowest index); NaN is an error.
        const cpu::ConstRows xs = crows(x, n, E);
        constexpr std::size_t kSlab = 8192;
        const std::size_t slab = std::min(kSlab, n_vocab);
        std::vector<float> buf(n * slab);
        std::vector<cpu::TopKEntry> best(n);
        // H1/M2: a NaN logit poisons only its own row (sequence), not the whole batched call.
        // decode_argmax reports a poisoned row as ArgmaxResult{-1, NaN} instead of throwing, so
        // the engine can fail just that one sequence instead of every sequence in the tick.
        // (std::vector<bool> is not span-compatible, hence uint8_t here.)
        std::vector<std::uint8_t> poisoned(n, 0);
        for (std::size_t n0 = 0; n0 < n_vocab; n0 += slab) {
            const std::size_t ns = std::min(slab, n_vocab - n0);
            const cpu::WeightMatrix sub = weight_matrix(g.wtype, w, n0, ns, E, rb);
            cpu::matmul(xs, sub, cpu::Rows(buf.data(), n, ns, slab), pool_);
            for (std::size_t i = 0; i < n; ++i) {
                const float* row = &buf[i * slab];
                for (std::size_t c = 0; c < ns; ++c) {
                    if (std::isnan(row[c])) {
                        poisoned[i] = 1;
                        continue;
                    }
                    if (!poisoned[i] && n0 + c < vocab_limit && (best[i].index < 0 || row[c] > best[i].value))
                        best[i] = {static_cast<std::int32_t>(n0 + c), row[c]};
                }
                if (want_logits) std::copy_n(row, ns, y.wf() + i * y.fstride() + n0);
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (poisoned[i] != 0) best[i] = {-1, std::numeric_limits<float>::quiet_NaN()};
        }
        write_results(res, best, std::span<const std::uint8_t>(poisoned));
        if (ids_out.wbase != nullptr) {
            // LmHeadArgs::ids: dense winning ids; a poisoned vector (index -1) writes 0, a valid row.
            auto* out = static_cast<std::int32_t*>(static_cast<void*>(ids_out.wbase));
            for (std::size_t i = 0; i < n; ++i) out[i] = best[i].index < 0 ? 0 : best[i].index;
        }
    }

    void argmax(Stream& s, const ArgmaxArgs& a) override {
        check_stream(s, "ARGMAX_FUSED");
        check_kernel(a.kernel, OpId::Argmax);
        check_status(a.status, OpId::Argmax);
        const Resolver r(this, "ARGMAX_FUSED");
        const Operand l = r.f32(a.logits, a.n_vec, a.n, Access::Read, "logits");
        const Operand res = r.get(a.result, 1, mul_u64(a.n_vec, kArgmaxResultBytes, r.op(), "result"), 4, Access::Write, "result");
        r.no_overlap(res, l);
        std::vector<cpu::TopKEntry> best(a.n_vec);
        for (std::size_t i = 0; i < a.n_vec; ++i) best[i] = cpu::argmax(std::span<const float>(l.f() + i * l.fstride(), a.n));
        write_results(res, best);
    }

    void top_k(Stream& s, const TopKArgs& a) override {
        check_stream(s, "TOP_K");
        check_kernel(a.kernel, OpId::TopK);
        check_status(a.status, OpId::TopK);
        const Resolver r(this, "TOP_K");
        const Operand l = r.f32(a.logits, a.n_vec, a.n, Access::Read, "logits");
        const Operand ids = r.get(a.ids, a.n_vec, mul_u64(a.k, 4, r.op(), "ids"), 4, Access::Write, "ids");
        const Operand vals = r.f32(a.values, a.n_vec, a.k, Access::Write, "values");
        r.no_overlap(ids, l);
        r.no_overlap(vals, l);
        r.no_overlap(ids, vals);
        std::vector<std::vector<cpu::TopKEntry>> all(a.n_vec);
        for (std::size_t i = 0; i < a.n_vec; ++i) all[i] = cpu::top_k(std::span<const float>(l.f() + i * l.fstride(), a.n), a.k);
        for (std::size_t i = 0; i < a.n_vec; ++i) {
            auto* id = static_cast<std::int32_t*>(static_cast<void*>(ids.wbase + i * ids.stride));
            float* v = vals.wf() + i * vals.fstride();
            for (std::size_t j = 0; j < a.k; ++j) {
                id[j] = all[i][j].index;
                v[j] = all[i][j].value;
            }
        }
    }

    void copy(Stream& s, const CopyArgs& a) override {
        check_stream(s, "COPY");
        check_kernel(a.kernel, OpId::Copy);
        const Resolver r(this, "COPY");
        const Operand src = r.get(a.src, 1, a.bytes, 1, Access::Read, "src");
        const Operand dst = r.get(a.dst, 1, a.bytes, 1, Access::Write, "dst");
        r.no_overlap(dst, src);
        if (a.bytes != 0) std::memcpy(dst.wbase, src.base, a.bytes);
    }

private:
    void check_stream(const Stream& s, const char* op) const {
        const auto* cs = dynamic_cast<const CpuStream*>(&s);
        HALO_CHECK(cs != nullptr && cs->owner() == this, ErrorCode::Api, "{}: stream of another backend", op);
    }

    static void write_results(const Operand& res, std::span<const cpu::TopKEntry> best,
                              std::span<const std::uint8_t> poisoned = {}) {
        for (std::size_t i = 0; i < best.size(); ++i) {
            const bool p = i < poisoned.size() && poisoned[i] != 0;
            const std::array<std::uint32_t, 3> w{static_cast<std::uint32_t>(best[i].index), std::bit_cast<std::uint32_t>(best[i].value),
                                                 p ? 1u : 0u};
            std::memcpy(res.wbase + i * kArgmaxResultBytes, w.data(), sizeof(w));
        }
    }

    cpu::ThreadPool* pool_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(cpu::ThreadPool* pool) { return std::make_unique<CpuBackend>(pool); }

}  // namespace halo::backend
