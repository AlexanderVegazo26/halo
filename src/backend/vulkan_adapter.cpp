// The Vulkan backend adapter (ADR-001 §5.1, WS-BI-5 / WS-F2 V5): halo::backend::Backend over
// the halo::vulkan operators. See vulkan_adapter.h for the memory, status and aliasing
// contract. Every op validates its TensorRefs here against the interface's rules (ownership,
// logical range, alignment, the neutral aliasing rule of backend.h), converts them to
// vulkan::BufferViews and records the named halo::vulkan op into the stream.

#include "backend/vulkan_adapter.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <deque>
#include <format>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/kernel.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"

namespace halo::backend {

namespace {

namespace hv = halo::vulkan;

constexpr std::uint64_t kF32 = 4;
constexpr std::uint32_t kStatusChunkWords = 256;
constexpr std::uint32_t kMaxGdnChunk = 64;   // vulkan gated_delta_rule_chunked
constexpr std::uint32_t kMaxConvK = 8;       // vulkan causal_conv1d_silu
constexpr std::uint32_t kMaxTopK = 1024;     // vulkan top_k
constexpr std::uint32_t kMaxHeadDim = 256;   // vulkan attention (and <= 4 * attention_workgroup)

std::uint64_t round_up4(std::uint64_t v) { return (v + 3) / 4 * 4; }

std::uint64_t mul_u64(std::uint64_t a, std::uint64_t b, const char* op, const char* what) {
    HALO_CHECK(b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b, ErrorCode::Kernel, "{}: {} size overflows", op,
               what);
    return a * b;
}

class VulkanBackend;

// ---------------------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------------------

class VkBuf final : public Buffer {
public:
    VkBuf(VulkanBackend* owner, hv::Buffer dev, std::uint64_t bytes, Tier tier, bool writable, const std::byte* ro_host,
          std::span<std::byte> rw_host,
          std::uint64_t wb_limit = std::numeric_limits<std::uint64_t>::max());
    ~VkBuf() override;
    VkBuf(const VkBuf&) = delete;
    VkBuf& operator=(const VkBuf&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept override { return bytes_; }
    [[nodiscard]] Tier tier() const noexcept override { return tier_; }
    [[nodiscard]] const Backend* backend() const noexcept override;
    [[nodiscard]] bool writable() const noexcept override { return writable_; }
    // Not host-addressable (ADR A-4), imports included: the device copy is the operand.
    [[nodiscard]] const std::byte* host_data() const noexcept override { return nullptr; }
    [[nodiscard]] std::byte* host_ptr() const noexcept override { return nullptr; }

    [[nodiscard]] const hv::Buffer& dev() const noexcept { return dev_; }
    /// The caller's bytes of a read-only import (fixed while the buffer lives), else null.
    [[nodiscard]] const std::byte* ro_host() const noexcept { return ro_host_; }
    /// The caller's memory a writable import mirrors, else empty.
    [[nodiscard]] std::span<std::byte> rw_host() const noexcept { return rw_host_; }

    /// allocate() zero-fills lazily: the fill is recorded into the stream of the first op
    /// that touches the buffer (stream-ordered before it), never as its own submission.
    /// take_needs_zero() clears the flag; the recording stream re-marks it if the recording
    /// is discarded before submission (the fill never ran then).
    [[nodiscard]] bool take_needs_zero() const { return std::exchange(needs_zero_, false); }
    void remark_zero() const { needs_zero_ = true; }

    /// Records a device-written byte range of a writable import (merged, sorted). Only the
    /// union of these ranges, clamped to `wb_limit_`, is written back at Stream::wait().
    void dirty_add(std::uint64_t lo, std::uint64_t hi) const {
        if (rw_host_.empty()) return;
        hi = std::min({hi, bytes_, wb_limit_});
        if (hi <= lo) return;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
        merged.reserve(dirty_.size() + 1);
        bool placed = false;
        for (const auto& [a, b] : dirty_) {
            if (b < lo) {
                merged.emplace_back(a, b);
            } else if (hi < a) {
                if (!placed) {
                    merged.emplace_back(lo, hi);
                    placed = true;
                }
                merged.emplace_back(a, b);
            } else {  // overlaps or touches: absorb
                lo = std::min(lo, a);
                hi = std::max(hi, b);
            }
        }
        if (!placed) merged.emplace_back(lo, hi);
        dirty_ = std::move(merged);
    }
    /// The merged dirty ranges, resetting the tracker for the next wait interval.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::uint64_t>> dirty_take() const {
        return std::exchange(dirty_, {});
    }

private:
    VulkanBackend* owner_;
    hv::Buffer dev_;
    std::uint64_t bytes_;
    Tier tier_;
    bool writable_;
    const std::byte* ro_host_;
    std::span<std::byte> rw_host_;
    /// Write-back ceiling for write-only imports (the untouched suffix stays caller-owned).
    std::uint64_t wb_limit_ = std::numeric_limits<std::uint64_t>::max();
    mutable std::vector<std::pair<std::uint64_t, std::uint64_t>> dirty_;
    mutable bool needs_zero_ = false;
};

// ---------------------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------------------

class VkStream final : public Stream {
public:
    VkStream(VulkanBackend* owner, std::shared_ptr<hv::Context> ctx) : owner_(owner), ctx_(ctx), s_(std::move(ctx)) {}
    ~VkStream() override { abort(); }
    VkStream(const VkStream&) = delete;
    VkStream& operator=(const VkStream&) = delete;

    void submit() override {
        submitted_ = true;
        s_.submit();
    }
    void wait() override;
    void abort() noexcept override {
        // A discarded (never submitted) recording's zero-fills never ran: re-mark those
        // buffers so their next use fills them again. After a submission the fills ran (or
        // the device is lost and every buffer's content is undefined anyway).
        if (!submitted_) {
            for (const VkBuf* b : zeroed_) b->remark_zero();
        }
        s_.discard();
        clear();
    }
    /// Records the deferred zero-fill of an allocate()d buffer (see VkBuf::remark_zero),
    /// in stream order before the op that first touches it.
    void zero_fill(const VkBuf& b) {
        s_.fill(b.dev(), 0, b.dev().size(), 0u);
        zeroed_.push_back(&b);
    }

    [[nodiscard]] hv::Stream& s() noexcept { return s_; }
    [[nodiscard]] const VulkanBackend* owner() const noexcept { return owner_; }

    /// A fresh 4-byte status word for one op call (zeroed by the op itself).
    hv::BufferView status_word(std::string_view op) {
        if (status_ops_.size() == std::size_t{status_chunks_.size()} * kStatusChunkWords) {
            status_chunks_.push_back(hv::Buffer::create(ctx_, kStatusChunkWords * 4, hv::MemoryUsage::HostCached));
        }
        const std::size_t i = status_ops_.size();
        status_ops_.emplace_back(op);
        return hv::BufferView(status_chunks_[i / kStatusChunkWords], (i % kStatusChunkWords) * 4, 4);
    }
    /// Device scratch (workspace) alive until the next wait()/abort().
    hv::BufferView scratch(std::uint64_t bytes) {
        keep_.push_back(hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(bytes), 4), hv::MemoryUsage::DeviceLocal));
        return hv::BufferView(keep_.back(), 0, bytes);
    }
    /// Host bytes staged in a host-visible buffer alive until the next wait()/abort().
    const hv::Buffer& stage(std::span<const std::byte> bytes) {
        keep_.push_back(hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(bytes.size()), 4), hv::MemoryUsage::HostVisible));
        if (!bytes.empty()) keep_.back().upload(bytes);
        return keep_.back();
    }
    /// Records src -> a host-cached staging buffer; `dst` is filled at wait().
    void add_download(const hv::Buffer& src, std::uint64_t offset, std::span<std::byte> dst) {
        hv::Buffer staging = hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(dst.size()), 4), hv::MemoryUsage::HostCached);
        s_.copy(src, offset, staging, 0, dst.size());
        downloads_.push_back(Pending{std::move(staging), dst});
    }

private:
    struct Pending {
        hv::Buffer staging;
        std::span<std::byte> dst;
    };
    void clear() noexcept {
        keep_.clear();
        downloads_.clear();
        status_ops_.clear();
        zeroed_.clear();
        submitted_ = false;
    }

    VulkanBackend* owner_;
    std::shared_ptr<hv::Context> ctx_;
    hv::Stream s_;
    std::deque<hv::Buffer> keep_;  // deque: references stay valid as it grows
    std::vector<Pending> downloads_;
    std::deque<hv::Buffer> status_chunks_;  // reused across waits
    std::vector<std::string> status_ops_;
    std::vector<const VkBuf*> zeroed_;      // deferred zero-fills recorded in this recording
    bool submitted_ = false;                // the current recording was (at least once) submitted
};

// ---------------------------------------------------------------------------------------
// Operand resolution (the interface's rules; backend.h)
// ---------------------------------------------------------------------------------------

enum class Access : std::uint8_t { Read, Write };

struct Ref {
    hv::BufferView view{};
    const VkBuf* buf = nullptr;
    std::uint64_t begin = 0;   // byte offset in the buffer
    std::uint64_t extent = 0;  // bytes touched from begin
    std::uint64_t stride = 0;
    const char* name = "";
};

class Resolver {
public:
    Resolver(const Backend* self, const char* op, VkStream& st) : self_(self), op_(op), st_(&st) {}

    Ref get(const TensorRef& ref, std::uint64_t rows, std::uint64_t row_bytes, std::uint64_t align, Access access,
            const char* name, bool track = true) const {
        Ref o;
        o.name = name;
        o.stride = ref.row_stride != 0 ? ref.row_stride : row_bytes;
        if ((rows == 0 || row_bytes == 0) && ref.empty()) return o;
        HALO_CHECK(!ref.empty(), ErrorCode::Kernel, "{}: operand {} is empty", op_, name);
        const auto* b = dynamic_cast<const VkBuf*>(ref.buffer);
        HALO_CHECK(b != nullptr && b->backend() == self_, ErrorCode::Kernel, "{}: operand {} belongs to another backend", op_,
                   name);
        HALO_CHECK(access == Access::Read || b->writable(), ErrorCode::Kernel, "{}: output {} is read-only", op_, name);
        HALO_CHECK(rows <= 1 || o.stride >= row_bytes, ErrorCode::Kernel, "{}: operand {} row stride {} < row bytes {}", op_,
                   name, o.stride, row_bytes);
        o.extent = rows == 0 ? 0 : mul_u64(rows - 1, o.stride, op_, name) + row_bytes;
        HALO_CHECK(ref.offset <= b->bytes(), ErrorCode::Kernel, "{}: operand {} offset {} past buffer end {}", op_, name,
                   ref.offset, b->bytes());
        const std::uint64_t avail = b->bytes() - ref.offset;
        const std::uint64_t limit = ref.bytes != 0 ? ref.bytes : avail;
        HALO_CHECK(limit <= avail, ErrorCode::Kernel, "{}: operand {} range [{}, +{}) exceeds buffer of {} bytes", op_, name,
                   ref.offset, ref.bytes, b->bytes());
        HALO_CHECK(o.extent <= limit, ErrorCode::Kernel, "{}: operand {} needs {} bytes, view has {}", op_, name, o.extent,
                   limit);
        HALO_CHECK(ref.offset % align == 0 && o.stride % align == 0, ErrorCode::Kernel,
                   "{}: operand {} is not {}-byte aligned (offset {}, stride {})", op_, name, align, ref.offset, o.stride);
        o.buf = b;
        o.begin = ref.offset;
        o.view = hv::BufferView(b->dev(), ref.offset, limit, rows > 1 ? o.stride : 0);
        // The deferred zero-fill of an allocate()d buffer, stream-ordered before this op.
        if (b->take_needs_zero()) st_->zero_fill(*b);
        // Writable imports are mirrored to the caller's memory at wait(): track what ops write.
        if (track && access == Access::Write) b->dirty_add(o.begin, o.extent);
        return o;
    }

    Ref f32(const TensorRef& ref, std::uint64_t rows, std::uint64_t cols, Access access, const char* name,
            bool track = true) const {
        return get(ref, rows, mul_u64(cols, kF32, op_, name), kF32, access, name, track);
    }

    /// Rejects any overlap of `out` with `other`; identical views only if `allow_identical`.
    void no_overlap(const Ref& out, const Ref& other, bool allow_identical = false) const {
        if (out.extent == 0 || other.extent == 0 || out.buf != other.buf) return;
        const bool overlap = out.begin < other.begin + other.extent && other.begin < out.begin + out.extent;
        if (!overlap) return;
        const bool identical = out.begin == other.begin && out.stride == other.stride && out.extent == other.extent;
        HALO_CHECK(allow_identical && identical, ErrorCode::Kernel, "{}: output {} overlaps operand {}{}", op_, out.name,
                   other.name, allow_identical ? " (only an identical view may alias)" : "");
    }

    [[nodiscard]] const char* op() const noexcept { return op_; }

private:
    const Backend* self_;
    const char* op_;
    VkStream* st_;
};

void check_kernel(const KernelChoice& k, OpId op) {
    HALO_CHECK(k.variant_id == 0, ErrorCode::Config, "{}: unknown kernel variant id {} on the vulkan backend (only 0)",
               op_name(op), k.variant_id);
}

void check_status(const StatusRef& s, OpId op) {
    HALO_CHECK(s.empty(), ErrorCode::Unsupported,
               "{}: per-sequence status words need the step status array of ADR-001 WS-BI-2; the vulkan backend raises "
               "device data errors as Error(Kernel) at Stream::wait()",
               op_name(op));
}

std::uint64_t weight_row_bytes(DType t, std::uint32_t cols) { return hv::matvec_row_bytes(t, cols); }

constexpr std::array<VariantInfo, kOpCount> kVariants{{
    {0, "vulkan", OpId::GetRows, {}},        {0, "vulkan", OpId::RmsNorm, {}},
    {0, "vulkan", OpId::AddRmsNorm, {}},     {0, "vulkan", OpId::Gemv, {}},
    {0, "vulkan", OpId::GdnGates, {}},       {0, "vulkan", OpId::Conv1dSilu, {}},
    {0, "vulkan", OpId::GatedDeltaRule, {}}, {0, "vulkan", OpId::GatedRmsNorm, {}},
    {0, "vulkan", OpId::PartialRope, {}},    {0, "vulkan", OpId::KvWrite, {}},
    {0, "vulkan", OpId::Attention, {}},      {0, "vulkan", OpId::Swiglu, {}},
    {0, "vulkan", OpId::MulSigmoid, {}},     {0, "vulkan", OpId::Add, {}},
    {0, "vulkan", OpId::LmHead, {}},         {0, "vulkan", OpId::Argmax, {}},
    {0, "vulkan", OpId::TopK, {}},           {0, "vulkan", OpId::Copy, {}},
}};

// ---------------------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------------------

class VulkanBackend final : public Backend {
public:
    VulkanBackend(std::shared_ptr<hv::Context> ctx, const VulkanBackendOptions& o)
        : ctx_(std::move(ctx)), options_(o), ops_(ctx_, o.ops) {}
    ~VulkanBackend() override = default;
    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;

    [[nodiscard]] Kind kind() const noexcept override { return Kind::Vulkan; }
    [[nodiscard]] std::string describe() const override {
        const hv::DeviceInfo& i = ctx_->info();
        return std::format("vulkan ({}, {}{})", i.name, i.driver_name, i.correctness_only ? ", correctness-only CPU device" : "");
    }
    [[nodiscard]] Limits limits() const noexcept override {
        Limits l;
        l.max_gdn_dk = options_.ops.gdn_max_dk;
        l.max_gdn_chunk = kMaxGdnChunk;
        l.max_head_dim = std::min(kMaxHeadDim, 4 * options_.ops.attention_workgroup);
        l.max_conv_k = kMaxConvK;
        l.max_top_k = kMaxTopK;
        l.max_rope_dims = std::numeric_limits<std::uint32_t>::max();  // host cos/sin table; rot <= head_dim
        l.gdn_chunked = true;
        l.max_import_bytes = ctx_->info().max_memory_allocation_size;  // RADV: 4 GiB
        return l;
    }

    // ---- memory --------------------------------------------------------------------------

    std::unique_ptr<Buffer> allocate(std::uint64_t bytes, Tier tier) override {
        const hv::MemoryUsage usage = tier == Tier::Vram  ? hv::MemoryUsage::DeviceLocal
                                      : tier == Tier::Gtt ? hv::MemoryUsage::HostVisible
                                                          : hv::MemoryUsage::HostCached;
        hv::Buffer b = hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(bytes), 4), usage);
        // Zero-filled, as the CPU backend — but lazily (WS-BI-7): the fill is recorded, in
        // stream order, ahead of the first op that touches the buffer (Resolver::get). A
        // synchronous fill per allocation was a full GPU round trip each, and the step
        // arena allocates ~35 buffers per forward.
        auto r = std::make_unique<VkBuf>(this, std::move(b), bytes, tier, true, nullptr, std::span<std::byte>());
        r->remark_zero();
        return r;
    }
    std::unique_ptr<Buffer> import_host(std::span<std::byte> bytes) override {
        const auto lo = std::bit_cast<std::uintptr_t>(bytes.data());
        for (const VkBuf* b : imports_) {
            const auto blo = std::bit_cast<std::uintptr_t>(b->rw_host().data());
            HALO_CHECK(bytes.empty() || b->rw_host().empty() || lo + bytes.size() <= blo || blo + b->rw_host().size() <= lo,
                       ErrorCode::Kernel, "vulkan import_host: [{:#x}, +{}) overlaps a live writable import", lo, bytes.size());
        }
        // Host-cached, not device-local: a writable import is uploaded at import and mirrored
        // back at every Stream::wait(). As a mapped buffer both directions are plain memcpys
        // and the kernels read/write the state over the unified fabric; as a device-local
        // buffer each direction is a staged GPU transfer with a fence round-trip per 16 MiB
        // chunk, which dominated decode time (~500 blocking round-trips per token).
        hv::Buffer d = hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(bytes.size()), 4),
                                          hv::MemoryUsage::HostCached);
        if (!bytes.empty()) d.upload(bytes);
        return std::make_unique<VkBuf>(this, std::move(d), bytes.size(), Tier::Vram, true, nullptr, bytes);
    }
    std::unique_ptr<Buffer> import_host_writeonly(std::span<std::byte> bytes, std::uint64_t writeback_bytes) override {
        // Same host-cached placement as import_host, but no upload: the device only writes the
        // buffer (rollback-slot snapshots), and the write-back is clamped to the prefix the op
        // actually writes — the untouched suffix stays caller-owned (backend.h).
        hv::Buffer d = hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(bytes.size()), 4),
                                          hv::MemoryUsage::HostCached);
        return std::make_unique<VkBuf>(this, std::move(d), bytes.size(), Tier::Vram, true, nullptr, bytes,
                                       writeback_bytes);
    }
    std::unique_ptr<Buffer> import_host_readonly(std::span<const std::byte> bytes) override {
        hv::Buffer d = device_copy(bytes);
        return std::make_unique<VkBuf>(this, std::move(d), bytes.size(), Tier::Vram, false, bytes.data(),
                                       std::span<std::byte>());
    }
    std::unique_ptr<Stream> create_stream() override { return std::make_unique<VkStream>(this, ctx_); }

    void upload(Stream& s, TensorRef dst, std::span<const std::byte> src) override {
        VkStream& st = stream(s, "upload");
        const Resolver r(this, "upload", st);
        const Ref d = r.get(dst, 1, src.size(), 1, Access::Write, "dst");
        if (src.empty()) return;
        const hv::Buffer& staging = st.stage(src);
        st.s().copy(staging, 0, d.buf->dev(), d.begin, src.size());
    }
    void download(Stream& s, TensorRef src, std::span<std::byte> dst) override {
        VkStream& st = stream(s, "download");
        const Resolver r(this, "download", st);
        const Ref o = r.get(src, 1, dst.size(), 1, Access::Read, "src");
        if (dst.empty()) return;
        st.add_download(o.buf->dev(), o.begin, dst);
    }

    [[nodiscard]] std::span<const VariantInfo> variants(OpId op, std::string_view /*form*/) const override {
        const auto i = static_cast<std::size_t>(op);
        HALO_CHECK(i < kVariants.size(), ErrorCode::Api, "variants: unknown op id {}", i);
        return std::span<const VariantInfo>(&kVariants[i], 1);
    }

    // ---- ops -----------------------------------------------------------------------------

    void get_rows(Stream& s, const GetRowsArgs& a) override {
        VkStream& st = stream(s, "GET_ROWS");
        check_kernel(a.kernel, OpId::GetRows);
        check_status(a.status, OpId::GetRows);
        const Resolver r(this, "GET_ROWS", st);
        const std::uint64_t rb = weight_row_bytes(a.type, a.cols);
        const Ref table = r.get(a.table, a.n_rows, rb, a.type == DType::F32 ? kF32 : 1, Access::Read, "table");
        const Ref ids = r.get(a.ids, 1, mul_u64(a.n_ids, 4, r.op(), "ids"), 4, Access::Read, "ids");
        const Ref out = r.f32(a.out, a.n_ids, a.cols, Access::Write, "out");
        r.no_overlap(out, table);
        r.no_overlap(out, ids);
        if (a.n_ids == 0) return;
        ops_.get_rows(st.s(), hv::GetRowsArgs{a.type, table.view, a.n_rows, a.cols, ids.view, a.n_ids, out.view,
                                              st.status_word("GET_ROWS")});
    }

    void rms_norm(Stream& s, const RmsNormArgs& a) override {
        VkStream& st = stream(s, "RMS_NORM");
        check_kernel(a.kernel, OpId::RmsNorm);
        const Resolver r(this, "RMS_NORM", st);
        const Ref x = r.f32(a.x, a.rows, a.cols, Access::Read, "x");
        const Ref w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Ref out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, w);
        if (a.rows == 0) return;
        ops_.rms_norm(st.s(), x.view, w.view, out.view, a.rows, a.cols, a.eps);
    }

    void add_rms_norm(Stream& s, const AddRmsNormArgs& a) override {
        VkStream& st = stream(s, "ADD_RMS_NORM");
        check_kernel(a.kernel, OpId::AddRmsNorm);
        const Resolver r(this, "ADD_RMS_NORM", st);
        const Ref x = r.f32(a.a, a.rows, a.cols, Access::Read, "a");
        const Ref y = r.f32(a.b, a.rows, a.cols, Access::Read, "b");
        const Ref h = r.f32(a.h, a.rows, a.cols, Access::Write, "h");
        const Ref w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Ref out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(h, x, true);
        r.no_overlap(h, y);
        r.no_overlap(h, w);
        r.no_overlap(out, x);
        r.no_overlap(out, y);
        r.no_overlap(out, h);
        r.no_overlap(out, w);
        if (a.rows == 0) return;
        ops_.add_rms_norm(st.s(), hv::AddRmsNormArgs{x.view, y.view, h.view, w.view, out.view, a.rows, a.cols, a.eps});
    }

    void gemv(Stream& s, const GemvArgs& a) override {
        VkStream& st = stream(s, "MATMUL");
        check_kernel(a.kernel, OpId::Gemv);
        const Resolver r(this, "MATMUL", st);
        const hv::GemvArgs g = resolve_gemv(r, a, true);
        if (a.n_vec == 0 || a.rows == 0) return;
        ops_.gemv(st.s(), g);
    }

    void gdn_gates(Stream& s, const GdnGateArgs& a) override {
        VkStream& st = stream(s, "GDN_GATES");
        check_kernel(a.kernel, OpId::GdnGates);
        const Resolver r(this, "GDN_GATES", st);
        const std::uint32_t T = a.n_tokens, nv = a.n_heads;
        const Ref alpha = r.f32(a.alpha, T, nv, Access::Read, "alpha");
        const Ref beta = r.f32(a.beta, T, nv, Access::Read, "beta");
        const Ref dt = r.f32(a.dt_bias, 1, nv, Access::Read, "dt_bias");
        const Ref av = r.f32(a.a, 1, nv, Access::Read, "a");
        const Ref g = r.f32(a.g_out, T, nv, Access::Write, "g_out");
        const Ref bo = r.f32(a.beta_out, T, nv, Access::Write, "beta_out");
        for (const Ref* o : {&g, &bo}) {
            r.no_overlap(*o, alpha);
            r.no_overlap(*o, beta);
            r.no_overlap(*o, dt);
            r.no_overlap(*o, av);
        }
        r.no_overlap(g, bo);
        if (T == 0 || nv == 0) return;
        ops_.gdn_gates(st.s(), hv::GdnGateArgs{beta.view, alpha.view, dt.view, av.view, bo.view, g.view, T, nv});
    }

    void conv1d_silu(Stream& s, const Conv1dArgs& a) override {
        VkStream& st = stream(s, "CONV1D_SHORT");
        check_kernel(a.kernel, OpId::Conv1dSilu);
        const Resolver r(this, "CONV1D_SHORT", st);
        HALO_CHECK(a.kernel_size > 0, ErrorCode::Kernel, "CONV1D_SHORT: kernel_size is 0");
        HALO_CHECK(a.kernel_size <= kMaxConvK, ErrorCode::Unsupported, "CONV1D_SHORT: kernel {} exceeds the vulkan limit {}",
                   a.kernel_size, kMaxConvK);
        const std::uint32_t T = a.n_tokens, C = a.channels, K = a.kernel_size;
        const Ref x = r.f32(a.x, T, C, Access::Read, "x");
        const Ref w = r.f32(a.weight, C, K, Access::Read, "weight");
        const Ref out = r.f32(a.out, T, C, Access::Write, "out");
        const std::uint64_t state_floats = mul_u64(K - 1, C, r.op(), "conv_state");
        r.no_overlap(out, x);
        r.no_overlap(out, w);
        if (a.ring) {
            // ADR-001 §5.3: conv_state is the whole slab of P dense conv states; the kernel
            // derives input/final/slot addresses from the ring.
            const Ref slab = r.f32(a.conv_state, a.ring->p, state_floats, Access::Write, "conv_state slab");
            HALO_CHECK(a.state_slots.empty(), ErrorCode::Kernel,
                       "CONV1D_SHORT: with a state ring, state_slots is derived from the slab and must be empty");
            r.no_overlap(out, slab);
            r.no_overlap(slab, x);
            r.no_overlap(slab, w);
            if (T == 0 || C == 0) return;
            hv::Conv1dArgs c;
            c.x = x.view;
            c.weight = w.view;
            if (state_floats > 0) c.conv_state = slab.view;
            c.out = out.view;
            c.n_tokens = T;
            c.channels = C;
            c.kernel = K;
            c.n_slots = a.n_slots;
            c.ring = hv::GdnRing{a.ring->p, a.ring->live};
            ops_.causal_conv1d_silu(st.s(), c);
            return;
        }
        const Ref cs = r.f32(a.conv_state, K - 1, C, Access::Write, "conv_state");
        const std::uint64_t slot_floats = mul_u64(mul_u64(a.n_slots, K - 1, r.op(), "slots"), C, r.op(), "slots");
        const Ref sl = r.f32(a.state_slots, 1, slot_floats, Access::Write, "state_slots");
        HALO_CHECK(cs.extent == 0 || cs.stride == std::uint64_t{C} * kF32, ErrorCode::Kernel,
                   "CONV1D_SHORT: conv_state must be dense");
        r.no_overlap(out, cs);
        r.no_overlap(out, sl);
        r.no_overlap(cs, x);
        r.no_overlap(cs, w);
        r.no_overlap(sl, x);
        r.no_overlap(sl, w);
        r.no_overlap(sl, cs);
        if (T == 0 || C == 0) return;
        hv::Conv1dArgs c;
        c.x = x.view;
        c.weight = w.view;
        c.conv_state = cs.view;
        c.out = out.view;
        c.state_slots = slot_floats > 0 ? sl.view : hv::BufferView();
        c.n_tokens = T;
        c.channels = C;
        c.kernel = K;
        c.n_slots = slot_floats > 0 ? a.n_slots : 0;
        ops_.causal_conv1d_silu(st.s(), c);
    }

    void gated_delta_rule(Stream& s, const GdnArgs& a) override {
        VkStream& st = stream(s, "GATED_DELTANET");
        check_kernel(a.kernel, OpId::GatedDeltaRule);
        check_status(a.status, OpId::GatedDeltaRule);
        const Resolver r(this, "GATED_DELTANET", st);
        HALO_CHECK(a.mapping == GdnHeadMapping::Tiled, ErrorCode::Unsupported,
                   "GATED_DELTANET: the vulkan kernels implement the GGUF tiled head mapping only");
        HALO_CHECK(a.d_k <= options_.ops.gdn_max_dk, ErrorCode::Unsupported, "GATED_DELTANET: d_k {} exceeds the vulkan limit {}",
                   a.d_k, options_.ops.gdn_max_dk);
        const bool chunked = a.form == GdnForm::Chunked;
        if (chunked) {
            HALO_CHECK(a.chunk_size >= 1, ErrorCode::Kernel, "GATED_DELTANET: chunk_size is 0");
            HALO_CHECK(a.chunk_size <= kMaxGdnChunk, ErrorCode::Unsupported,
                       "GATED_DELTANET: chunk_size {} exceeds the vulkan limit {}", a.chunk_size, kMaxGdnChunk);
        }
        const std::uint32_t T = a.n_tokens;
        const std::uint64_t qk_cols = std::uint64_t{a.n_k} * a.d_k;
        const std::uint64_t v_cols = std::uint64_t{a.n_v} * a.d_v;
        const std::uint64_t state_floats = mul_u64(mul_u64(a.n_v, a.d_k, r.op(), "state"), a.d_v, r.op(), "state");
        const Ref q = r.f32(a.q, T, qk_cols, Access::Read, "q");
        const Ref k = r.f32(a.k, T, qk_cols, Access::Read, "k");
        const Ref v = r.f32(a.v, T, v_cols, Access::Read, "v");
        const Ref g = r.f32(a.g, T, a.n_v, Access::Read, "g");
        const Ref beta = r.f32(a.beta, T, a.n_v, Access::Read, "beta");
        const Ref out = r.f32(a.out, T, v_cols, Access::Write, "out");
        if (a.ring) {
            // ADR-001 §5.3: state is the whole slab of P dense states; the kernel derives
            // the input (slab[live]), final state and rollback slots from the ring.
            const Ref slab = r.f32(a.state, a.ring->p, state_floats, Access::Write, "state slab");
            HALO_CHECK(a.state_slots.empty(), ErrorCode::Kernel,
                       "GATED_DELTANET: with a state ring, state_slots is derived from the slab and must be empty");
            for (const Ref* i : {&q, &k, &v, &g, &beta}) r.no_overlap(slab, *i);
            r.no_overlap(out, slab);
            if (T == 0) return;
            hv::GdnDecodeArgs d;
            d.q = q.view;
            d.k = k.view;
            d.v = v.view;
            d.g = g.view;
            d.beta = beta.view;
            d.state = slab.view;
            d.out = out.view;
            d.n_v = a.n_v;
            d.n_k = a.n_k;
            d.d_k = a.d_k;
            d.d_v = a.d_v;
            d.n_tokens = T;
            d.n_slots = a.n_slots;
            d.qk_l2norm = a.qk_l2norm;
            d.q_scale = a.q_scale;
            d.ring = hv::GdnRing{a.ring->p, a.ring->live};
            if (!chunked) {
                ops_.gated_delta_rule_decode(st.s(), d);
                return;
            }
            hv::GdnChunkedArgs c;
            c.gdn = d;
            c.chunk_size = a.chunk_size;
            c.workspace = st.scratch(hv::gdn_chunked_workspace_bytes(d, a.chunk_size));
            c.status = st.status_word("GATED_DELTANET (chunked)");
            ops_.gated_delta_rule_chunked(st.s(), c);
            return;
        }
        const Ref stt = r.f32(a.state, 1, state_floats, Access::Write, "state");
        const std::uint64_t slot_floats = mul_u64(a.n_slots, state_floats, r.op(), "slots");
        const Ref sl = r.f32(a.state_slots, 1, slot_floats, Access::Write, "state_slots");
        for (const Ref* o : {&stt, &sl, &out}) {
            for (const Ref* i : {&q, &k, &v, &g, &beta}) r.no_overlap(*o, *i);
        }
        r.no_overlap(out, stt);
        r.no_overlap(out, sl);
        r.no_overlap(sl, stt);
        if (T == 0) return;
        hv::GdnDecodeArgs d;
        d.q = q.view;
        d.k = k.view;
        d.v = v.view;
        d.g = g.view;
        d.beta = beta.view;
        d.state = stt.view;
        d.state_slots = slot_floats > 0 ? sl.view : hv::BufferView();
        d.out = out.view;
        d.n_v = a.n_v;
        d.n_k = a.n_k;
        d.d_k = a.d_k;
        d.d_v = a.d_v;
        d.n_tokens = T;
        d.n_slots = slot_floats > 0 ? a.n_slots : 0;
        d.qk_l2norm = a.qk_l2norm;
        d.q_scale = a.q_scale;
        if (!chunked) {
            ops_.gated_delta_rule_decode(st.s(), d);
            return;
        }
        hv::GdnChunkedArgs c;
        c.gdn = d;
        c.chunk_size = a.chunk_size;
        c.workspace = st.scratch(hv::gdn_chunked_workspace_bytes(d, a.chunk_size));
        c.status = st.status_word("GATED_DELTANET (chunked)");
        ops_.gated_delta_rule_chunked(st.s(), c);
    }

    void gated_rms_norm(Stream& s, const GatedNormArgs& a) override {
        VkStream& st = stream(s, "GATED_NORM");
        check_kernel(a.kernel, OpId::GatedRmsNorm);
        const Resolver r(this, "GATED_NORM", st);
        const Ref x = r.f32(a.x, a.rows, a.cols, Access::Read, "x");
        const Ref z = r.f32(a.z, a.rows, a.cols, Access::Read, "z");
        const Ref w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Ref out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, z);
        r.no_overlap(out, w);
        if (a.rows == 0) return;
        ops_.gated_rms_norm(st.s(), hv::GatedNormArgs{x.view, z.view, w.view, out.view, a.rows, a.cols, a.eps});
    }

    void partial_rope(Stream& s, const RopeArgs& a) override {
        VkStream& st = stream(s, "PARTIAL_ROPE");
        check_kernel(a.kernel, OpId::PartialRope);
        const Resolver r(this, "PARTIAL_ROPE", st);
        HALO_CHECK(a.n_heads > 0, ErrorCode::Kernel, "PARTIAL_ROPE: n_heads is 0");
        const std::uint64_t hs = a.head_stride == 0 ? a.head_dim : a.head_stride;
        HALO_CHECK(hs >= a.head_dim, ErrorCode::Kernel, "PARTIAL_ROPE: head_stride {} < head_dim {}", hs, a.head_dim);
        const std::uint64_t row_cols = (std::uint64_t{a.n_heads} - 1) * hs + a.head_dim;
        const Ref x = r.f32(a.x, a.n_tokens, row_cols, Access::Write, "x");
        const Ref pos = r.get(a.positions, 1, mul_u64(a.n_tokens, 4, r.op(), "positions"), 4, Access::Read, "positions");
        r.no_overlap(x, pos);
        if (a.n_tokens == 0) return;
        // The cos/sin table is the CPU op's own fp32 angles evaluated in double on the host
        // (vulkan::rope_cos_sin_table), so the kernel is bit-identical to cpu::partial_rope_neox.
        // That needs the positions on the host: they must be a read-only host import.
        HALO_CHECK(pos.buf->ro_host() != nullptr, ErrorCode::Unsupported,
                   "PARTIAL_ROPE: the vulkan backend builds the cos/sin table on the host; positions must be a "
                   "read-only host import (import_host_readonly)");
        std::vector<std::int32_t> p(a.n_tokens);
        std::memcpy(p.data(), pos.buf->ro_host() + pos.begin, p.size() * 4);
        const std::vector<float> table = hv::rope_cos_sin_table(p, a.rot_dims, a.theta);
        const hv::Buffer& tb = st.stage(std::as_bytes(std::span(table)));
        hv::RopeArgs ra;
        ra.x = x.view;
        ra.cos_sin = tb;
        ra.n_tokens = a.n_tokens;
        ra.n_heads = a.n_heads;
        ra.head_dim = a.head_dim;
        ra.rot_dims = a.rot_dims;
        ra.head_stride = a.head_stride;
        ops_.partial_rope_neox(st.s(), ra);
    }

    void kv_write(Stream& s, const KvWriteArgs& a) override {
        VkStream& st = stream(s, "KV_WRITE");
        check_kernel(a.kernel, OpId::KvWrite);
        check_status(a.status, OpId::KvWrite);
        const Resolver r(this, "KV_WRITE", st);
        const std::uint64_t end = std::uint64_t{a.start} + a.n_tokens;
        const PoolRefs p = pool_refs(r, a.kv_pool, a.block_table, a.n_pool_blocks, a.n_layers, a.layer, a.block_tokens, a.kv_dim,
                                     a.n_block_table, end, Access::Write);
        const Ref k = r.f32(a.k, a.n_tokens, a.kv_dim, Access::Read, "k");
        const Ref v = r.f32(a.v, a.n_tokens, a.kv_dim, Access::Read, "v");
        r.no_overlap(p.pool, k);
        r.no_overlap(p.pool, v);
        if (a.n_tokens == 0) return;
        // The KV pool is a device-resident State-arena buffer (ADR-001 §5.2, WS-BI-2): the
        // kernel's writes stay on the device; there is no import write-back to track.
        hv::KvWriteArgs w;
        w.kv_pool = p.pool.view;
        w.n_pool_blocks = a.n_pool_blocks;
        w.n_layers = a.n_layers;
        w.layer = a.layer;
        w.block_tokens = a.block_tokens;
        w.kv_dim = a.kv_dim;
        w.block_table = p.table.view;
        w.start = a.start;
        w.n_tokens = a.n_tokens;
        w.k = k.view;
        w.v = v.view;
        w.status = st.status_word("KV_WRITE");
        ops_.kv_write(st.s(), w);
    }

    void attention(Stream& s, const AttentionArgs& a) override {
        VkStream& st = stream(s, "ATTENTION");
        check_kernel(a.kernel, OpId::Attention);
        check_status(a.status, OpId::Attention);
        const Resolver r(this, "ATTENTION", st);
        HALO_CHECK(a.n_head > 0 && a.n_kv_head > 0 && a.n_head % a.n_kv_head == 0, ErrorCode::Kernel,
                   "ATTENTION: n_head {} / n_kv_head {}", a.n_head, a.n_kv_head);
        const std::uint64_t hd = a.head_dim, T = a.n_tokens;
        const std::uint64_t kvd = std::uint64_t{a.n_kv_head} * hd;
        const std::uint64_t qhs = a.q_head_stride == 0 ? hd : a.q_head_stride;
        HALO_CHECK(qhs >= hd, ErrorCode::Kernel, "ATTENTION: q_head_stride {} < head_dim {}", qhs, hd);
        const std::uint64_t q_cols = (std::uint64_t{a.n_head} - 1) * qhs + hd;
        const std::uint64_t rows = std::uint64_t{a.q_offset} + T;
        HALO_CHECK(kvd <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Kernel, "ATTENTION: kv_dim overflows");
        const PoolRefs p = pool_refs(r, a.kv_pool, a.block_table, a.n_pool_blocks, a.n_layers, a.layer, a.block_tokens,
                                     static_cast<std::uint32_t>(kvd), a.n_block_table, rows, Access::Read);
        const Ref q = r.f32(a.q, T, q_cols, Access::Read, "q");
        const Ref out = r.f32(a.out, T, std::uint64_t{a.n_head} * hd, Access::Write, "out");
        r.no_overlap(out, q);
        r.no_overlap(out, p.pool);
        r.no_overlap(out, p.table);
        if (T == 0) return;
        hv::AttentionArgs at;
        at.q = q.view;
        at.q_head_stride = a.q_head_stride;
        at.kv_pool = p.pool.view;
        at.n_pool_blocks = a.n_pool_blocks;
        at.n_layers = a.n_layers;
        at.layer = a.layer;
        at.block_tokens = a.block_tokens;
        at.block_table = p.table.view;
        at.n_head = a.n_head;
        at.n_kv_head = a.n_kv_head;
        at.head_dim = a.head_dim;
        at.n_tokens = a.n_tokens;
        at.q_offset = a.q_offset;
        at.scale = a.scale;
        at.out = out.view;
        at.status = st.status_word("ATTENTION");
        ops_.attention(st.s(), at);
    }

    void swiglu(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::Swiglu); }
    void mul_sigmoid(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::MulSigmoid); }
    void add(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::Add); }

    void lm_head(Stream& s, const LmHeadArgs& a) override {
        VkStream& st = stream(s, "LM_HEAD");
        check_kernel(a.kernel, OpId::LmHead);
        check_kernel(a.gemv.kernel, OpId::LmHead);
        check_status(a.status, OpId::LmHead);
        const Resolver r(this, "LM_HEAD", st);
        const GemvArgs& g = a.gemv;
        const hv::GemvArgs hg = resolve_gemv(r, g, !g.y.empty());
        const Ref w = r.get(g.w, g.rows, weight_row_bytes(g.wtype, g.cols), g.wtype == DType::F32 ? kF32 : 1, Access::Read, "w");
        const Ref x = r.f32(g.x, g.n_vec, g.cols, Access::Read, "x");
        const Ref y = g.y.empty() ? Ref{} : r.f32(g.y, g.n_vec, g.rows, Access::Write, "logits");
        const Ref res =
            r.get(a.result, 1, mul_u64(g.n_vec, kArgmaxResultBytes, r.op(), "result"), 4, Access::Write, "result");
        for (const Ref* o : {&y, &res}) {
            r.no_overlap(*o, x);
            r.no_overlap(*o, w);
        }
        r.no_overlap(res, y);
        if (g.n_vec == 0 || g.rows == 0) return;
        hv::LmHeadArgs l;
        l.gemv = hg;
        l.workspace = st.scratch(ops_.lm_head_workspace_bytes(g.rows, g.n_vec, g.y.empty()));
        l.result = res.view;
        l.valid_rows = a.valid_rows;
        ops_.lm_head(st.s(), l);
    }

    void argmax(Stream& s, const ArgmaxArgs& a) override {
        VkStream& st = stream(s, "ARGMAX_FUSED");
        check_kernel(a.kernel, OpId::Argmax);
        check_status(a.status, OpId::Argmax);
        const Resolver r(this, "ARGMAX_FUSED", st);
        const Ref l = r.f32(a.logits, a.n_vec, a.n, Access::Read, "logits");
        const Ref res = r.get(a.result, 1, mul_u64(a.n_vec, kArgmaxResultBytes, r.op(), "result"), 4, Access::Write, "result");
        r.no_overlap(res, l);
        if (a.n_vec == 0 || a.n == 0) return;
        const std::uint64_t sb = ops_.argmax_scratch_bytes(a.n);
        const hv::BufferView scratch = st.scratch(sb * a.n_vec);
        for (std::uint32_t i = 0; i < a.n_vec; ++i) {
            ops_.argmax(st.s(), hv::BufferView(l.buf->dev(), l.begin + i * l.stride, std::uint64_t{a.n} * kF32), a.n,
                        hv::BufferView(*scratch.buffer, i * sb, sb),
                        hv::BufferView(res.buf->dev(), res.begin + std::uint64_t{i} * kArgmaxResultBytes, kArgmaxResultBytes));
        }
    }

    void top_k(Stream& s, const TopKArgs& a) override {
        VkStream& st = stream(s, "TOP_K");
        check_kernel(a.kernel, OpId::TopK);
        check_status(a.status, OpId::TopK);
        const Resolver r(this, "TOP_K", st);
        const Ref l = r.f32(a.logits, a.n_vec, a.n, Access::Read, "logits");
        const Ref ids = r.get(a.ids, a.n_vec, mul_u64(a.k, 4, r.op(), "ids"), 4, Access::Write, "ids");
        const Ref vals = r.f32(a.values, a.n_vec, a.k, Access::Write, "values");
        r.no_overlap(ids, l);
        r.no_overlap(vals, l);
        r.no_overlap(ids, vals);
        HALO_CHECK(a.k <= kMaxTopK, ErrorCode::Unsupported, "TOP_K: k {} exceeds the vulkan limit {}", a.k, kMaxTopK);
        if (a.n_vec == 0) return;
        hv::TopKArgs t;
        t.logits = l.view;
        t.n = a.n;
        t.k = a.k;
        t.n_vec = a.n_vec;
        const std::uint64_t ws = hv::topk_workspace_bytes(a.n, a.k, a.n_vec);
        if (ws > 0) t.workspace = st.scratch(ws);
        t.ids = ids.view;
        t.values = vals.view;
        t.status = st.status_word("TOP_K");
        ops_.top_k(st.s(), t);
    }

    void copy(Stream& s, const CopyArgs& a) override {
        VkStream& st = stream(s, "COPY");
        check_kernel(a.kernel, OpId::Copy);
        const Resolver r(this, "COPY", st);
        const Ref src = r.get(a.src, 1, a.bytes, 1, Access::Read, "src");
        const Ref dst = r.get(a.dst, 1, a.bytes, 1, Access::Write, "dst");
        r.no_overlap(dst, src);
        if (a.bytes == 0) return;
        st.s().copy(src.buf->dev(), src.begin, dst.buf->dev(), dst.begin, a.bytes);
    }

    // ---- adapter internals ------------------------------------------------------------------

    void register_import(VkBuf* b) { imports_.push_back(b); }
    void unregister_import(const VkBuf* b) noexcept { std::erase(imports_, b); }
    /// After a successful wait: the device-written (dirty) ranges of every live writable
    /// import -> the caller's memory. Ranges are tracked per write operand resolution, so a
    /// wait with no intervening import writes mirrors nothing.
    void write_back_imports() const {
        for (const VkBuf* b : imports_) {
            if (b->rw_host().empty()) continue;
            for (const auto& [lo, hi] : b->dirty_take()) b->dev().download(b->rw_host().subspan(lo, hi - lo), lo);
        }
    }

private:
    VkStream& stream(Stream& s, const char* op) const {
        auto* vs = dynamic_cast<VkStream*>(&s);
        HALO_CHECK(vs != nullptr && vs->owner() == this, ErrorCode::Api, "{}: stream of another backend", op);
        return *vs;
    }

    hv::Buffer device_copy(std::span<const std::byte> bytes) {
        hv::Buffer d = hv::Buffer::create(ctx_, std::max<std::uint64_t>(round_up4(bytes.size()), 4), hv::MemoryUsage::DeviceLocal);
        if (!bytes.empty()) d.upload(bytes);
        return d;
    }

    hv::GemvArgs resolve_gemv(const Resolver& r, const GemvArgs& a, bool want_y) const {
        const std::uint64_t rb = weight_row_bytes(a.wtype, a.cols);
        const Ref w = r.get(a.w, a.rows, rb, a.wtype == DType::F32 ? kF32 : 1, Access::Read, "w");
        const Ref x = r.f32(a.x, a.n_vec, a.cols, Access::Read, "x");
        const Ref y = want_y ? r.f32(a.y, a.n_vec, a.rows, Access::Write, "y") : Ref{};
        r.no_overlap(y, x);
        r.no_overlap(y, w);
        return hv::GemvArgs{a.wtype, w.view, x.view, y.view, a.rows, a.cols, a.n_vec};
    }

    struct PoolRefs {
        Ref pool, table;
    };
    PoolRefs pool_refs(const Resolver& r, const TensorRef& pool, const TensorRef& table, std::uint32_t n_pool_blocks,
                       std::uint32_t n_layers, std::uint32_t layer, std::uint32_t block_tokens, std::uint32_t kv_dim,
                       std::uint32_t n_table, std::uint64_t rows_needed, Access access) const {
        HALO_CHECK(block_tokens > 0 && n_layers > 0 && layer < n_layers, ErrorCode::Kernel,
                   "{}: block_tokens {} / layer {} of {}", r.op(), block_tokens, layer, n_layers);
        PoolRefs p;
        const std::uint64_t block_floats =
            mul_u64(mul_u64(mul_u64(n_layers, 2, r.op(), "pool"), block_tokens, r.op(), "pool"), kv_dim, r.op(), "pool");
        // Not dirty-tracked: the pool is a device-resident State-arena buffer (ADR-001 §5.2,
        // WS-BI-2 stage 2), so its writes never mirror back to host memory.
        p.pool = r.f32(pool, n_pool_blocks, block_floats, access, "kv_pool", /*track=*/false);
        HALO_CHECK(p.pool.extent == 0 || p.pool.stride == block_floats * kF32, ErrorCode::Kernel, "{}: kv_pool must be dense",
                   r.op());
        p.table = r.get(table, 1, mul_u64(n_table, 4, r.op(), "block_table"), 4, Access::Read, "block_table");
        const std::uint64_t needed = (rows_needed + block_tokens - 1) / block_tokens;
        HALO_CHECK(needed <= n_table, ErrorCode::Kernel, "{}: {} rows need {} blocks, table has {}", r.op(), rows_needed, needed,
                   n_table);
        // The device reads exactly the entries it needs, bounded by the table view.
        if (p.table.extent > 0) p.table.view.bytes = std::uint64_t{n_table} * 4;
        return p;
    }

    void eltwise(Stream& s, const EltwiseArgs& a, OpId op) {
        const char* name = op == OpId::Swiglu ? "SWIGLU" : op == OpId::MulSigmoid ? "MUL_SIGMOID" : "ADD";
        VkStream& st = stream(s, name);
        check_kernel(a.kernel, op);
        const Resolver r(this, name, st);
        const Ref x = r.f32(a.a, a.rows, a.cols, Access::Read, "a");
        const Ref y = r.f32(a.b, a.rows, a.cols, Access::Read, "b");
        const Ref out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x, op == OpId::Add);  // the residual accumulate out == a (ADD only)
        r.no_overlap(out, y);
        if (a.rows == 0) return;
        const hv::EltwiseArgs e{x.view, y.view, out.view, a.rows, a.cols};
        switch (op) {
            case OpId::Swiglu: ops_.swiglu(st.s(), e); break;
            case OpId::MulSigmoid: ops_.mul_sigmoid(st.s(), e); break;
            default: ops_.add(st.s(), e); break;
        }
    }

    std::shared_ptr<hv::Context> ctx_;
    VulkanBackendOptions options_;
    hv::Ops ops_;
    std::vector<VkBuf*> imports_;  // live writable imports (write-back at wait)
};

// ---------------------------------------------------------------------------------------

VkBuf::VkBuf(VulkanBackend* owner, hv::Buffer dev, std::uint64_t bytes, Tier tier, bool writable, const std::byte* ro_host,
             std::span<std::byte> rw_host, std::uint64_t wb_limit)
    : owner_(owner), dev_(std::move(dev)), bytes_(bytes), tier_(tier), writable_(writable), ro_host_(ro_host),
      rw_host_(rw_host), wb_limit_(std::min(wb_limit, bytes)) {
    if (!rw_host_.empty()) owner_->register_import(this);
}

VkBuf::~VkBuf() {
    if (!rw_host_.empty()) owner_->unregister_import(this);
}

const Backend* VkBuf::backend() const noexcept { return owner_; }

void VkStream::wait() {
    try {
        s_.submit();
        s_.wait();
        for (const Pending& d : downloads_) d.staging.download(d.dst);
    } catch (...) {
        abort();
        throw;
    }
    std::string errors;
    for (std::size_t i = 0; i < status_ops_.size(); ++i) {
        const std::uint32_t w = hv::read_status(status_chunks_[i / kStatusChunkWords], (i % kStatusChunkWords) * 4);
        if (w == 0) continue;
        if (!errors.empty()) errors += "; ";
        errors += std::format("{} (op #{}): status 0x{:x}{}{}{}", status_ops_[i], i, w,
                              (w & kStatusPositiveG) != 0 ? " positive-g" : "", (w & kStatusNaN) != 0 ? " NaN" : "",
                              (w & (kStatusBadBlock | kStatusBadIndex)) != 0 ? " out-of-range id" : "");
    }
    clear();
    if (!errors.empty()) {
        // The host keeps its pre-step state: imports are not written back.
        throw_error(ErrorCode::Kernel, "vulkan backend: device data error: {}", errors);
    }
    owner_->write_back_imports();
}

}  // namespace

std::unique_ptr<Backend> make_vulkan_backend(std::shared_ptr<vulkan::Context> ctx, const VulkanBackendOptions& options) {
    HALO_CHECK(ctx != nullptr, ErrorCode::Kernel, "make_vulkan_backend: null context");
    return std::make_unique<VulkanBackend>(std::move(ctx), options);
}

}  // namespace halo::backend
