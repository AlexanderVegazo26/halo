// HIP backend adapter (ADR-001 §5.1, §9 WS-BI-3): the Backend interface over halo::hip::Ops.
//
// Every op (1) checks the stream, the kernel choice and the (empty) status ref, (2) resolves
// its TensorRefs exactly as the CPU backend does (ownership, range, alignment, read-only
// outputs, the neutral aliasing rule), (3) checks this backend's operand limits
// (Error(Unsupported)), then (4) translates the operands byte for byte into hip::BufferViews
// and calls the hip::Ops method of the selected variant. Workspaces and status words the HIP
// kernels need are allocated per stream; data-error status words and argmax NaN words are
// evaluated right after the op in emulation and at Stream::wait() on a device.
//
// The operand resolution is a port of cpu_backend.cpp's Resolver (same checks, same messages)
// so both backends reject exactly the same calls; it addresses buffers through
// hip::Buffer::data() so it also works for device buffers that have no host address.

#include "halo/backend/hip_backend.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "halo/backends/hip/registry.h"
#include "halo/backends/hip/runtime.h"
#include "halo/core/error.h"
#include "halo/tensor/quant.h"

namespace halo::backend {

namespace {

constexpr std::uint64_t kF32 = 4;

// This backend's operand limits (docs/hip.md "Divergences from the CPU contract").
constexpr std::uint32_t kMaxDk = 128;
constexpr std::uint32_t kMaxChunk = 64;
constexpr std::uint32_t kMaxHeadDim = 256;
constexpr std::uint32_t kMaxConvK = 8;
constexpr std::uint32_t kMaxTopK = 1024;
constexpr std::uint32_t kMaxRopeDims = 128;
constexpr std::uint32_t kChunksPerGroup = 16;  // chunked GDN workspace: chunks per launch group

// ---------------------------------------------------------------------------------------
// Registry: explicit, append-only ids (ADR §5.6). Never renumber; retire an id by removing
// its row and never reusing the number. Grouped by op, then form (variants() returns runs).
// ---------------------------------------------------------------------------------------

struct Entry {
    std::uint32_t id;
    OpId op;
    std::string_view form;    // backend form ("recurrent" / "chunked" for GATED_DELTANET)
    std::string_view hip_op;  // hip registry operator id; empty = adapter-native (COPY)
    std::string_view name;    // hip registry variant name
};

constexpr std::uint32_t kGemmId = 8;
constexpr std::uint32_t kCopyId = 29;

constexpr std::array<Entry, 29> kTable{{
    {1, OpId::GetRows, "", "GET_ROWS", "get_rows_b256"},
    {2, OpId::RmsNorm, "", "RMS_NORM", "rms_norm_b128"},
    {3, OpId::RmsNorm, "", "RMS_NORM", "rms_norm_b32"},
    {4, OpId::AddRmsNorm, "", "ADD_RMS_NORM", "add_rms_norm_b128"},
    {5, OpId::Gemv, "", "QUANT_GEMV", "gemv_wave32_r4"},
    {6, OpId::Gemv, "", "QUANT_GEMV", "gemv_wave32_r8"},
    {7, OpId::Gemv, "", "QUANT_GEMV", "gemv_generic_b64"},
    {kGemmId, OpId::Gemv, "", "QUANT_GEMM", "gemm_t16x16_b256"},
    {9, OpId::GdnGates, "", "GDN_GATE", "gdn_gate_b64"},
    {10, OpId::Conv1dSilu, "", "CONV1D_SHORT", "conv1d_silu_b256"},
    {11, OpId::Conv1dSilu, "", "CONV1D_SHORT", "conv1d_silu_b64"},
    {12, OpId::GatedDeltaRule, "recurrent", "GATED_DELTANET", "gdn_recurrent_b128"},
    {13, OpId::GatedDeltaRule, "recurrent", "GATED_DELTANET", "gdn_recurrent_b64"},
    {14, OpId::GatedDeltaRule, "chunked", "GATED_DELTANET", "gdn_chunked_b64"},
    {15, OpId::GatedDeltaRule, "chunked", "GATED_DELTANET", "gdn_chunked_b32"},
    {16, OpId::GatedRmsNorm, "", "GATED_NORM", "gated_norm_b128"},
    {17, OpId::GatedRmsNorm, "", "GATED_NORM", "gated_norm_b32"},
    {18, OpId::PartialRope, "", "PARTIAL_ROPE", "rope_neox_b128"},
    {19, OpId::KvWrite, "", "KV_WRITE", "kv_write_b256"},
    {20, OpId::Attention, "", "ATTENTION", "attn_online_b128"},
    {21, OpId::Attention, "", "ATTENTION", "attn_online_b64"},
    {22, OpId::Attention, "", "ATTENTION", "attn_exact_b128"},
    {23, OpId::Swiglu, "", "SWIGLU", "swiglu_b256"},
    {24, OpId::MulSigmoid, "", "MUL_SIGMOID", "mul_sigmoid_b256"},
    {25, OpId::Add, "", "ADD", "add_b256"},
    {26, OpId::LmHead, "", "ARGMAX_FUSED", "argmax_b256"},  // stage 2; stage 1 = the Gemv variant (gemv.kernel)
    {27, OpId::Argmax, "", "ARGMAX_FUSED", "argmax_b256"},
    {28, OpId::TopK, "", "TOP_K", "topk_bitonic_b256"},
    {kCopyId, OpId::Copy, "", "", "copy_async"},
}};

constexpr std::array<VariantInfo, kTable.size()> kInfos = [] {
    std::array<VariantInfo, kTable.size()> a{};
    for (std::size_t i = 0; i < kTable.size(); ++i) a[i] = VariantInfo{kTable[i].id, kTable[i].name, kTable[i].op, kTable[i].form};
    return a;
}();

const Entry* find_entry(std::uint32_t id) noexcept {
    for (const Entry& e : kTable) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

/// `base` with the variant of (hip_op, form) replaced by `name`.
hip::OpsOptions with_variant(hip::OpsOptions o, std::string_view hip_op, std::string_view form, std::string_view name) {
    std::string* field = nullptr;
    if (hip_op == "GET_ROWS") field = &o.get_rows;
    else if (hip_op == "RMS_NORM") field = &o.rms_norm;
    else if (hip_op == "ADD_RMS_NORM") field = &o.add_rms_norm;
    else if (hip_op == "QUANT_GEMV") field = &o.gemv;
    else if (hip_op == "QUANT_GEMM") field = &o.gemm;
    else if (hip_op == "GDN_GATE") field = &o.gdn_gate;
    else if (hip_op == "CONV1D_SHORT") field = &o.conv1d;
    else if (hip_op == "GATED_DELTANET") field = form == "chunked" ? &o.gdn_chunked : &o.gdn_recurrent;
    else if (hip_op == "GATED_NORM") field = &o.gated_norm;
    else if (hip_op == "PARTIAL_ROPE") field = &o.rope;
    else if (hip_op == "KV_WRITE") field = &o.kv_write;
    else if (hip_op == "ATTENTION") field = &o.attention;
    else if (hip_op == "SWIGLU") field = &o.swiglu;
    else if (hip_op == "MUL_SIGMOID") field = &o.mul_sigmoid;
    else if (hip_op == "ADD") field = &o.add;
    else if (hip_op == "ARGMAX_FUSED") field = &o.argmax;
    else if (hip_op == "TOP_K") field = &o.topk;
    HALO_CHECK(field != nullptr, ErrorCode::Config, "hip backend: no options field for HIP operator {}", hip_op);
    *field = std::string(name);
    return o;
}

// ---------------------------------------------------------------------------------------
// Buffers and streams
// ---------------------------------------------------------------------------------------

class HipBuffer final : public Buffer {
public:
    HipBuffer(const Backend* owner, hip::Buffer buf, std::unique_ptr<std::byte[]> storage, std::uint64_t bytes, Tier tier,
              bool writable, bool host_addressable)
        : owner_(owner), storage_(std::move(storage)), buf_(std::move(buf)), bytes_(bytes), tier_(tier), writable_(writable),
          host_(host_addressable) {}

    [[nodiscard]] std::uint64_t bytes() const noexcept override { return bytes_; }
    [[nodiscard]] Tier tier() const noexcept override { return tier_; }
    [[nodiscard]] const Backend* backend() const noexcept override { return owner_; }
    [[nodiscard]] bool writable() const noexcept override { return writable_; }
    [[nodiscard]] const std::byte* host_data() const noexcept override {
        return host_ ? static_cast<const std::byte*>(buf_.data()) : nullptr;
    }
    [[nodiscard]] std::byte* host_ptr() const noexcept override {
        return host_ && writable_ ? static_cast<std::byte*>(buf_.data()) : nullptr;
    }
    [[nodiscard]] const hip::Buffer& hip() const noexcept { return buf_; }
    /// Address of byte 0 in the kernels' address space (host in emulation, device on a GPU).
    [[nodiscard]] std::uintptr_t address() const noexcept { return std::bit_cast<std::uintptr_t>(buf_.data()); }

private:
    const Backend* owner_;
    std::unique_ptr<std::byte[]> storage_;  // emulation allocations (buf_ wraps it)
    hip::Buffer buf_;
    std::uint64_t bytes_;
    Tier tier_;
    bool writable_;
    bool host_;
};

/// One deferred data-error check: a status word, or the NaN words of argmax results.
struct PendingCheck {
    std::string op;
    std::uint32_t owner = kBatchWide;
    hip::BufferView view{};
    std::uint32_t n_vec = 0;  // 0 = status word; else argmax results of n_vec vectors
};

class HipStream final : public Stream {
public:
    HipStream(const Backend* owner, std::shared_ptr<hip::Context> ctx, bool reverse, bool* poisoned)
        : owner_(owner), ctx_(std::move(ctx)), reverse_(reverse), poisoned_(poisoned) {
        if (ctx_) dev_.emplace(*ctx_);
    }
    ~HipStream() override { abort(); }
    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;

    [[nodiscard]] const Backend* owner() const noexcept { return owner_; }
    [[nodiscard]] bool emulation() const noexcept { return !dev_.has_value(); }
    [[nodiscard]] hip::Target target() const noexcept {
        return dev_ ? hip::Target::device(*dev_) : hip::Target::emulation(reverse_);
    }
    [[nodiscard]] const hip::Stream* device_stream() const noexcept { return dev_ ? &*dev_ : nullptr; }

    /// Scratch memory that lives until the op's checks ran (emulation) / wait() (device).
    hip::BufferView scratch(std::uint64_t bytes) {
        const std::uint64_t n = std::max<std::uint64_t>(bytes, kF32);
        auto s = std::make_unique<Scratch>();  // heap node: the returned view points at s->buf, which must not move
        if (dev_) {
            s->buf = hip::Buffer::allocate(*ctx_, n, hip::MemoryTier::Device);
        } else {
            HALO_CHECK(n <= std::numeric_limits<std::size_t>::max(), ErrorCode::Memory, "hip backend: {} scratch bytes", n);
            try {
                s->host = std::make_unique<std::byte[]>(static_cast<std::size_t>(n));
            } catch (const std::bad_alloc&) {
                throw_error(ErrorCode::Memory, "hip backend: cannot allocate {} scratch bytes", n);
            }
            std::memset(s->host.get(), 0xFF, static_cast<std::size_t>(n));  // device scratch is not zeroed either
            s->buf = hip::Buffer::wrap_host(s->host.get(), n);
        }
        scratch_.push_back(std::move(s));
        return hip::BufferView(scratch_.back()->buf, 0, n);
    }

    void expect_status(const char* op, std::uint32_t owner, hip::BufferView word) {
        pending_.push_back(PendingCheck{op, owner, word, 0});
    }
    void expect_argmax(const char* op, std::uint32_t owner, hip::BufferView result, std::uint32_t n_vec) {
        pending_.push_back(PendingCheck{op, owner, result, n_vec});
    }

    /// Called after every op: emulation evaluates its checks now (the CPU backend's timing).
    void op_done() {
        if (emulation()) evaluate();
    }

    void submit() override {}  // HIP work is enqueued as each op is called

    // Perf note (cpp23-efficiency-review.md §G): correctness of the async batching here depends
    // entirely on call-site discipline -- the device path only synchronizes and evaluates status
    // words here, in wait(), not per op. Callers must invoke wait() once per batch of enqueued ops
    // (e.g. once per decode tick), not once per op, or every op degenerates into a full
    // hipStreamSynchronize and the async pipeline this backend enqueues into is defeated.
    void wait() override {
        if (dev_) {
            try {
                dev_->synchronize();
            } catch (const Error& e) {
                // ADR §5.5: a lost device or a wild kernel poisons the backend.
                if (e.code() == ErrorCode::Device || e.code() == ErrorCode::Kernel) *poisoned_ = true;
                clear();
                throw;
            }
        }
        evaluate();
    }

    void abort() noexcept override {
        if (dev_) {
            try {
                dev_->synchronize();
            } catch (...) {  // NOLINT(bugprone-empty-catch): abort drains; the error was or will be reported by wait()
            }
        }
        clear();
    }

private:
    struct Scratch {
        std::unique_ptr<std::byte[]> host;
        hip::Buffer buf;
    };

    void clear() noexcept {
        pending_.clear();
        scratch_.clear();
    }

    /// Reads every pending check (the work has completed), clears them, then throws one
    /// Error(Kernel) naming every failed op.
    void evaluate() {
        std::string failures;
        const hip::Target t = target();
        for (const PendingCheck& c : pending_) {
            try {
                if (c.n_vec == 0) {
                    hip::Ops::check_status(t, c.view);
                } else {
                    static_cast<void>(hip::Ops::read_argmax(t, c.view, c.n_vec));
                }
            } catch (const Error& e) {
                if (e.code() != ErrorCode::Kernel) {
                    clear();
                    throw;
                }
                failures += std::format("{}{} ({}): {}", failures.empty() ? "" : "; ", c.op,
                                        c.owner == kBatchWide ? std::string("batch-wide") : std::format("sequence {}", c.owner),
                                        e.what());
            }
        }
        clear();
        HALO_CHECK(failures.empty(), ErrorCode::Kernel, "hip backend: data error: {}", failures);
    }

    const Backend* owner_;
    std::shared_ptr<hip::Context> ctx_;
    std::optional<hip::Stream> dev_;
    bool reverse_;
    bool* poisoned_;
    std::vector<PendingCheck> pending_;
    std::vector<std::unique_ptr<Scratch>> scratch_;
};

// ---------------------------------------------------------------------------------------
// Operand resolution (a port of cpu_backend.cpp's Resolver; see the file comment)
// ---------------------------------------------------------------------------------------

std::uint64_t mul_u64(std::uint64_t a, std::uint64_t b, const char* op, const char* what) {
    HALO_CHECK(b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b, ErrorCode::Kernel, "{}: {} size overflows", op,
               what);
    return a * b;
}

/// A validated operand: rows of row_bytes, `stride` bytes apart.
struct Operand {
    const HipBuffer* buf = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;    // the TensorRef's bytes (0 = to the end)
    std::uint64_t stride = 0;
    std::uint64_t extent = 0;   // bytes from the first to one past the last byte touched
    std::uintptr_t addr = 0;
    bool flat = false;          // passed to HIP as a dense (row_stride 0) range
    const char* name = "";

    /// The same bytes as a hip::BufferView: buffer, offset and bytes unchanged; the row stride
    /// is the resolved one (explicit, so HIP's own dense default never applies) except for
    /// flat operands (state, slots, ids, pool images), which HIP addresses densely.
    [[nodiscard]] hip::BufferView view() const {
        if (buf == nullptr) return {};
        return hip::BufferView(buf->hip(), offset, bytes, flat ? 0 : stride);
    }
    [[nodiscard]] bool empty() const noexcept { return buf == nullptr; }
};

enum class Access : std::uint8_t { Read, Write };

class Resolver {
public:
    Resolver(const Backend* self, const char* op) : self_(self), op_(op) {}

    Operand get(const TensorRef& ref, std::uint64_t rows, std::uint64_t row_bytes, std::uint64_t align, Access access,
                const char* name, bool flat = false) const {
        Operand o;
        o.name = name;
        o.flat = flat;
        o.stride = ref.row_stride != 0 ? ref.row_stride : row_bytes;
        if (rows == 0 || row_bytes == 0) {
            if (ref.empty()) return o;
        }
        HALO_CHECK(!ref.empty(), ErrorCode::Kernel, "{}: operand {} is empty", op_, name);
        const Buffer& b = *ref.buffer;
        HALO_CHECK(b.backend() == self_, ErrorCode::Kernel, "{}: operand {} belongs to another backend", op_, name);
        const auto* hb = dynamic_cast<const HipBuffer*>(&b);
        HALO_CHECK(hb != nullptr, ErrorCode::Kernel, "{}: operand {} is not a hip backend buffer", op_, name);
        HALO_CHECK(access == Access::Read || b.writable(), ErrorCode::Kernel, "{}: output {} is read-only", op_, name);
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
        o.buf = hb;
        o.offset = ref.offset;
        o.bytes = ref.bytes;
        o.addr = hb->address() + ref.offset;
        HALO_CHECK(o.addr % align == 0 && o.stride % align == 0, ErrorCode::Kernel,
                   "{}: operand {} is not {}-byte aligned (offset {}, stride {})", op_, name, align, ref.offset, o.stride);
        return o;
    }

    /// fp32 rows x cols.
    Operand f32(const TensorRef& ref, std::uint64_t rows, std::uint64_t cols, Access access, const char* name) const {
        return get(ref, rows, mul_u64(cols, kF32, op_, name), kF32, access, name);
    }
    /// A dense range of `bytes` bytes (flat operand).
    Operand flat(const TensorRef& ref, std::uint64_t bytes, std::uint64_t align, Access access, const char* name) const {
        return get(ref, 1, bytes, align, access, name, true);
    }

    void no_overlap(const Operand& out, const Operand& other, bool allow_identical = false) const {
        if (out.extent == 0 || other.extent == 0) return;
        const std::uintptr_t lo1 = out.addr;
        const std::uintptr_t lo2 = other.addr;
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

void check_status_ref(const StatusRef& s, OpId op) {
    HALO_CHECK(s.empty(), ErrorCode::Unsupported,
               "{}: status words are not implemented yet (ADR-001 WS-BI-2); the hip backend raises data errors itself",
               op_name(op));
}

void check_nonzero(std::uint64_t v, const char* op, const char* what) {
    HALO_CHECK(v > 0, ErrorCode::Kernel, "{}: {} is 0", op, what);
}

void check_limit(std::uint64_t v, std::uint32_t limit, const char* op, const char* what) {
    HALO_CHECK(v <= limit, ErrorCode::Unsupported, "{}: {} {} exceeds the hip backend's limit {}", op, what, v, limit);
}

std::uint64_t weight_row_bytes(DType t, std::uint32_t cols, const char* op) {
    HALO_CHECK(dequant_supported(t), ErrorCode::Unsupported, "{}: no dequantization for {}", op, traits(t).name);
    return tensor::row_bytes(t, cols);
}

hip::GdnHeadMapping hip_mapping(GdnHeadMapping m) noexcept {
    return m == GdnHeadMapping::Tiled ? hip::GdnHeadMapping::Tiled : hip::GdnHeadMapping::Grouped;
}

// ---------------------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------------------

class HipBackend final : public Backend {
public:
    explicit HipBackend(const HipBackendOptions& o)
        : opt_(o), defaults_(o.defaults) {
        // Every variant's Ops, built once (validates each name against the HIP registry).
        for (const Entry& e : kTable) {
            if (e.hip_op.empty()) continue;
            static_cast<void>(hip::find_variant(e.hip_op, e.name));  // Error(Config) if the registry lost it
            if (e.op == OpId::LmHead) continue;  // LM head Ops are per (gemv, argmax) pair below
            per_variant_.emplace(e.id, hip::Ops(with_variant(o.defaults, e.hip_op, e.form, e.name)));
        }
        for (const Entry& g : kTable) {
            if (g.op != OpId::Gemv || g.id == kGemmId) continue;
            for (const Entry& a : kTable) {
                if (a.op != OpId::LmHead) continue;
                const hip::OpsOptions oo = with_variant(with_variant(o.defaults, g.hip_op, g.form, g.name), a.hip_op, a.form, a.name);
                lm_ops_.emplace(std::pair{g.id, a.id}, hip::Ops(oo));
                lm_ops_.emplace(std::pair{g.id, 0u}, hip::Ops(with_variant(o.defaults, g.hip_op, g.form, g.name)));
                lm_ops_.emplace(std::pair{0u, a.id}, hip::Ops(with_variant(o.defaults, a.hip_op, a.form, a.name)));
            }
        }
        if (o.mode == HipMode::Device) {
            ctx_ = hip::Context::create(o.device_ordinal);  // Error(Device) without a usable device (D-001)
        }
    }

    [[nodiscard]] Kind kind() const noexcept override { return ctx_ ? Kind::Hip : Kind::HipEmulation; }

    [[nodiscard]] std::string describe() const override {
        const hip::OpsOptions& d = defaults_.options();
        const std::string profile = std::format("gemv {}, attention {}", d.gemv, d.attention);
        if (ctx_) {
            const hip::DeviceInfo& i = ctx_->info();
            return std::format("hip device {} ({}, {} CUs, driver {}, runtime {}; built for {}; {})", i.name, i.gcn_arch_full,
                               i.compute_units, i.driver_version, i.runtime_version, hip::compiled_offload_archs(), profile);
        }
        return std::format("hip-emulation (host emulation of the {} kernels, {} order; {})", hip::compiled_offload_archs(),
                           opt_.emulation_reverse_order ? "reverse" : "forward", profile);
    }

    [[nodiscard]] Limits limits() const noexcept override {
        return Limits{kMaxDk, kMaxChunk, kMaxHeadDim, kMaxConvK, kMaxTopK, kMaxRopeDims, true};
    }

    std::unique_ptr<Buffer> allocate(std::uint64_t bytes, Tier tier) override {
        check_alive("allocate");
        const std::uint64_t n = std::max<std::uint64_t>(bytes, 1);  // hip::Buffer needs a non-empty range
        if (ctx_) {
            const hip::MemoryTier t = tier == Tier::Vram ? hip::MemoryTier::Device : hip::MemoryTier::HostPinned;
            return std::make_unique<HipBuffer>(this, hip::Buffer::allocate(*ctx_, n, t), nullptr, bytes, tier, true, false);
        }
        HALO_CHECK(n <= std::numeric_limits<std::size_t>::max(), ErrorCode::Memory, "hip backend: {} bytes", bytes);
        std::unique_ptr<std::byte[]> storage;
        try {
            storage = std::make_unique<std::byte[]>(static_cast<std::size_t>(n));
        } catch (const std::bad_alloc&) {
            throw_error(ErrorCode::Memory, "hip backend: cannot allocate {} bytes", bytes);
        }
        if (opt_.emulation_poison_alloc) std::memset(storage.get(), 0xFF, static_cast<std::size_t>(n));
        hip::Buffer b = hip::Buffer::wrap_host(storage.get(), n);
        return std::make_unique<HipBuffer>(this, std::move(b), std::move(storage), bytes, tier, true, true);
    }

    std::unique_ptr<Buffer> import_host(std::span<std::byte> bytes) override {
        check_alive("import_host");
        HALO_CHECK(!ctx_, ErrorCode::Unsupported,
                   "hip backend: import_host is not supported on a device (ADR-001 §5.2: copy into a backend buffer)");
        return wrap(bytes.data(), bytes.size(), true);
    }

    std::unique_ptr<Buffer> import_host_readonly(std::span<const std::byte> bytes) override {
        check_alive("import_host_readonly");
        if (ctx_) {
            // D-017 / ADR §5.2 default until owner question Q3 is answered: weights are copied
            // into VRAM once at load. The copy is read-only for the ops, like the import.
            const std::uint64_t n = std::max<std::uint64_t>(bytes.size(), 1);
            hip::Buffer b = hip::Buffer::allocate(*ctx_, n, hip::MemoryTier::Device);
            if (!bytes.empty()) b.upload(bytes.data(), bytes.size());
            return std::make_unique<HipBuffer>(this, std::move(b), nullptr, bytes.size(), Tier::Vram, false, false);
        }
        // The kernels take non-const pointers; the Resolver rejects this buffer as an output
        // (writable() == false), so nothing ever writes through the cast.
        return wrap(const_cast<std::byte*>(bytes.data()), bytes.size(), false);  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    }

    std::unique_ptr<Stream> create_stream() override {
        check_alive("create_stream");
        return std::make_unique<HipStream>(this, ctx_, opt_.emulation_reverse_order, &poisoned_);
    }

    void upload(Stream& s, TensorRef dst, std::span<const std::byte> src) override {
        stream(s, "upload");
        const Resolver r(this, "upload");
        const Operand d = r.flat(dst, src.size(), 1, Access::Write, "dst");
        if (!src.empty()) d.buf->hip().upload(src.data(), src.size(), d.offset);
    }

    void download(Stream& s, TensorRef src, std::span<std::byte> dst) override {
        stream(s, "download");
        const Resolver r(this, "download");
        const Operand o = r.flat(src, dst.size(), 1, Access::Read, "src");
        if (!dst.empty()) o.buf->hip().download(dst.data(), dst.size(), o.offset);
    }

    [[nodiscard]] std::span<const VariantInfo> variants(OpId op, std::string_view form) const override {
        const auto i = static_cast<std::size_t>(op);
        HALO_CHECK(i < kOpCount, ErrorCode::Api, "variants: unknown op id {}", i);
        std::size_t first = kInfos.size(), last = 0;
        for (std::size_t k = 0; k < kInfos.size(); ++k) {
            if (kInfos[k].op != op || (!form.empty() && kInfos[k].form != form)) continue;
            first = std::min(first, k);
            last = k + 1;
        }
        if (first >= last) return {};
        return std::span<const VariantInfo>(kInfos).subspan(first, last - first);
    }

    // ---- ops ---------------------------------------------------------------------------

    void get_rows(Stream& s, const GetRowsArgs& a) override {
        HipStream& st = stream(s, "GET_ROWS");
        const hip::Ops& ops = select(OpId::GetRows, a.kernel);
        check_status_ref(a.status, OpId::GetRows);
        const Resolver r(this, "GET_ROWS");
        const std::uint64_t rb = weight_row_bytes(a.type, a.cols, r.op());
        const Operand table = r.get(a.table, a.n_rows, rb, a.type == DType::F32 ? kF32 : 1, Access::Read, "table");
        const Operand ids = r.flat(a.ids, mul_u64(a.n_ids, 4, r.op(), "ids"), 4, Access::Read, "ids");
        const Operand out = r.f32(a.out, a.n_ids, a.cols, Access::Write, "out");
        r.no_overlap(out, table);
        r.no_overlap(out, ids);
        if (a.n_ids == 0 || a.cols == 0) return;
        hip::GetRowsArgs h;
        h.wtype = a.type;
        h.w = table.view();
        h.n_rows = a.n_rows;
        h.cols = a.cols;
        h.ids = ids.view();
        h.n_ids = a.n_ids;
        h.out = out.view();
        h.status = st.scratch(4);
        ops.get_rows(st.target(), h);
        st.expect_status(r.op(), kBatchWide, h.status);
        st.op_done();
    }

    void rms_norm(Stream& s, const RmsNormArgs& a) override {
        HipStream& st = stream(s, "RMS_NORM");
        const hip::Ops& ops = select(OpId::RmsNorm, a.kernel);
        const Resolver r(this, "RMS_NORM");
        const Operand x = r.f32(a.x, a.rows, a.cols, Access::Read, "x");
        const Operand w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, w);
        if (a.rows == 0 || a.cols == 0) return;
        ops.rms_norm(st.target(), hip::RmsNormArgs{x.view(), w.view(), out.view(), a.rows, a.cols, a.eps});
        st.op_done();
    }

    void add_rms_norm(Stream& s, const AddRmsNormArgs& a) override {
        HipStream& st = stream(s, "ADD_RMS_NORM");
        const hip::Ops& ops = select(OpId::AddRmsNorm, a.kernel);
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
        if (a.rows == 0 || a.cols == 0) return;
        ops.add_rms_norm(st.target(), hip::AddRmsNormArgs{x.view(), y.view(), h.view(), w.view(), out.view(), a.rows, a.cols, a.eps});
        st.op_done();
    }

    void gemv(Stream& s, const GemvArgs& a) override {
        HipStream& st = stream(s, "MATMUL");
        const hip::Ops& ops = select(OpId::Gemv, a.kernel);
        const Resolver r(this, "MATMUL");
        const std::uint64_t rb = weight_row_bytes(a.wtype, a.cols, r.op());
        const Operand w = r.get(a.w, a.rows, rb, a.wtype == DType::F32 ? kF32 : 1, Access::Read, "w");
        const Operand x = r.f32(a.x, a.n_vec, a.cols, Access::Read, "x");
        const Operand y = r.f32(a.y, a.n_vec, a.rows, Access::Write, "y");
        r.no_overlap(y, x);
        r.no_overlap(y, w);
        if (a.n_vec == 0 || a.rows == 0) return;
        const hip::GemvArgs h{a.wtype, w.view(), x.view(), y.view(), a.rows, a.cols, a.n_vec};
        if (a.kernel.variant_id == kGemmId) {
            ops.gemm(st.target(), h);
        } else {
            ops.gemv(st.target(), h);
        }
        st.op_done();
    }

    void gdn_gates(Stream& s, const GdnGateArgs& a) override {
        HipStream& st = stream(s, "GDN_GATES");
        const hip::Ops& ops = select(OpId::GdnGates, a.kernel);
        const Resolver r(this, "GDN_GATES");
        const std::uint64_t T = a.n_tokens, nv = a.n_heads;
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
        if (T == 0 || nv == 0) return;
        // HIP names: b = the beta projection, a = the alpha projection, ssm_a = the model's a.
        hip::GdnGateArgs h;
        h.b = beta.view();
        h.a = alpha.view();
        h.dt_bias = dt.view();
        h.ssm_a = av.view();
        h.beta = bo.view();
        h.g = g.view();
        h.rows = a.n_tokens;
        h.n_v = a.n_heads;
        ops.gdn_gates(st.target(), h);
        st.op_done();
    }

    void conv1d_silu(Stream& s, const Conv1dArgs& a) override {
        HipStream& st = stream(s, "CONV1D_SHORT");
        const hip::Ops& ops = select(OpId::Conv1dSilu, a.kernel);
        const Resolver r(this, "CONV1D_SHORT");
        check_nonzero(a.kernel_size, r.op(), "kernel_size");
        const std::uint64_t T = a.n_tokens, C = a.channels, K = a.kernel_size;
        const Operand x = r.f32(a.x, T, C, Access::Read, "x");
        const Operand w = r.f32(a.weight, C, K, Access::Read, "weight");
        const Operand cst = r.f32(a.conv_state, K - 1, C, Access::Write, "conv_state");
        const Operand out = r.f32(a.out, T, C, Access::Write, "out");
        const std::uint64_t slot_floats = mul_u64(mul_u64(a.n_slots, K - 1, r.op(), "slots"), C, r.op(), "slots");
        const Operand sl = r.flat(a.state_slots, mul_u64(slot_floats, kF32, r.op(), "slots"), kF32, Access::Write, "state_slots");
        HALO_CHECK(cst.extent == 0 || cst.stride == C * kF32, ErrorCode::Kernel, "CONV1D_SHORT: conv_state must be dense");
        r.no_overlap(out, x);
        r.no_overlap(out, w);
        r.no_overlap(out, cst);
        r.no_overlap(out, sl);
        r.no_overlap(cst, x);
        r.no_overlap(cst, w);
        r.no_overlap(sl, x);
        r.no_overlap(sl, w);
        r.no_overlap(sl, cst);
        check_limit(K, kMaxConvK, r.op(), "kernel size");
        if (T == 0 || C == 0) return;
        hip::Conv1dArgs h;
        h.x = x.view();
        h.weight = w.view();
        h.conv_state = cst.view();
        h.out = out.view();
        h.state_slots = sl.view();
        h.n_tokens = a.n_tokens;
        h.channels = a.channels;
        h.kernel = a.kernel_size;
        h.n_slots = sl.empty() ? 0 : a.n_slots;
        ops.causal_conv1d_silu(st.target(), h);
        st.op_done();
    }

    void gated_delta_rule(Stream& s, const GdnArgs& a) override {
        HipStream& st = stream(s, "GATED_DELTANET");
        const bool chunked = a.form == GdnForm::Chunked;
        const hip::Ops& ops = select(OpId::GatedDeltaRule, a.kernel, chunked ? "chunked" : "recurrent");
        check_status_ref(a.status, OpId::GatedDeltaRule);
        const Resolver r(this, "GATED_DELTANET");
        const std::uint64_t T = a.n_tokens;
        const std::uint64_t qk_cols = static_cast<std::uint64_t>(a.n_k) * a.d_k;
        const std::uint64_t v_cols = static_cast<std::uint64_t>(a.n_v) * a.d_v;
        const std::uint64_t state_floats = mul_u64(mul_u64(a.n_v, a.d_k, r.op(), "state"), a.d_v, r.op(), "state");
        const std::uint64_t state_bytes = mul_u64(state_floats, kF32, r.op(), "state");
        const Operand q = r.f32(a.q, T, qk_cols, Access::Read, "q");
        const Operand k = r.f32(a.k, T, qk_cols, Access::Read, "k");
        const Operand v = r.f32(a.v, T, v_cols, Access::Read, "v");
        const Operand g = r.f32(a.g, T, a.n_v, Access::Read, "g");
        const Operand beta = r.f32(a.beta, T, a.n_v, Access::Read, "beta");
        const Operand sta = r.flat(a.state, state_bytes, kF32, Access::Write, "state");
        const Operand sl = r.flat(a.state_slots, mul_u64(a.n_slots, state_bytes, r.op(), "slots"), kF32, Access::Write, "state_slots");
        const Operand out = r.f32(a.out, T, v_cols, Access::Write, "out");
        for (const Operand* o : {&sta, &sl, &out}) {
            for (const Operand* i : {&q, &k, &v, &g, &beta}) r.no_overlap(*o, *i);
        }
        r.no_overlap(out, sta);
        r.no_overlap(out, sl);
        r.no_overlap(sl, sta);
        check_limit(a.d_k, kMaxDk, r.op(), "d_k");
        if (chunked) check_limit(a.chunk_size, kMaxChunk, r.op(), "chunk size");
        if (T == 0) return;
        hip::GdnArgs h;
        h.q = q.view();
        h.k = k.view();
        h.v = v.view();
        h.g = g.view();
        h.beta = beta.view();
        h.state = sta.view();  // in place (state_out = nullopt), as the CPU op
        h.state_slots = sl.view();
        h.out = out.view();
        h.n_k = a.n_k;
        h.n_v = a.n_v;
        h.d_k = a.d_k;
        h.d_v = a.d_v;
        h.n_tokens = a.n_tokens;
        h.n_slots = sl.empty() ? 0 : a.n_slots;
        h.mapping = hip_mapping(a.mapping);
        h.qk_l2norm = a.qk_l2norm;
        h.q_scale = a.q_scale;
        if (!chunked) {
            ops.gated_delta_rule_recurrent(st.target(), h);
            st.op_done();
            return;
        }
        check_nonzero(a.chunk_size, r.op(), "chunk size");
        const std::uint32_t n_chunks = static_cast<std::uint32_t>((T + a.chunk_size - 1) / a.chunk_size);
        hip::GdnChunkedArgs c;
        c.gdn = h;
        c.chunk_size = a.chunk_size;
        c.workspace = st.scratch(hip::gdn_chunked_workspace_bytes(h, a.chunk_size, std::min(n_chunks, kChunksPerGroup)));
        c.status = st.scratch(4);
        ops.gated_delta_rule_chunked(st.target(), c);
        st.expect_status(r.op(), kBatchWide, c.status);
        st.op_done();
    }

    void gated_rms_norm(Stream& s, const GatedNormArgs& a) override {
        HipStream& st = stream(s, "GATED_NORM");
        const hip::Ops& ops = select(OpId::GatedRmsNorm, a.kernel);
        const Resolver r(this, "GATED_NORM");
        const Operand x = r.f32(a.x, a.rows, a.cols, Access::Read, "x");
        const Operand z = r.f32(a.z, a.rows, a.cols, Access::Read, "z");
        const Operand w = r.f32(a.w, 1, a.cols, Access::Read, "w");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x);
        r.no_overlap(out, z);
        r.no_overlap(out, w);
        if (a.rows == 0 || a.cols == 0) return;
        ops.gated_rms_norm(st.target(), hip::GatedNormArgs{x.view(), z.view(), w.view(), out.view(), a.rows, a.cols, a.eps});
        st.op_done();
    }

    void partial_rope(Stream& s, const RopeArgs& a) override {
        HipStream& st = stream(s, "PARTIAL_ROPE");
        const hip::Ops& ops = select(OpId::PartialRope, a.kernel);
        const Resolver r(this, "PARTIAL_ROPE");
        check_nonzero(a.n_heads, r.op(), "n_heads");
        const std::uint64_t hs = a.head_stride == 0 ? a.head_dim : a.head_stride;
        HALO_CHECK(hs >= a.head_dim, ErrorCode::Kernel, "PARTIAL_ROPE: head_stride {} < head_dim {}", hs, a.head_dim);
        const std::uint64_t row_cols = (a.n_heads - 1) * hs + a.head_dim;
        const Operand x = r.f32(a.x, a.n_tokens, row_cols, Access::Write, "x");
        const Operand pos = r.flat(a.positions, mul_u64(a.n_tokens, 4, r.op(), "positions"), 4, Access::Read, "positions");
        r.no_overlap(x, pos);
        check_limit(a.rot_dims, kMaxRopeDims, r.op(), "rot_dims");
        if (a.n_tokens == 0) return;
        // HIP addresses the heads with head_stride natively (TD-9): no per-head decomposition.
        hip::RopeArgs h;
        h.x = x.view();
        h.positions = pos.view();
        h.n_tokens = a.n_tokens;
        h.n_heads = a.n_heads;
        h.head_dim = a.head_dim;
        h.rot_dims = a.rot_dims;
        h.theta = a.theta;
        h.head_stride = static_cast<std::uint32_t>(hs);
        ops.partial_rope_neox(st.target(), h);
        st.op_done();
    }

    /// Validates the pool image and the block table's shape (its entries live in backend
    /// memory, so the kernels check them: kStatusBadBlock).
    struct PoolView {
        Operand pool;
        Operand table;
    };
    PoolView pool_view(const Resolver& r, const TensorRef& pool, const TensorRef& table, std::uint32_t n_pool_blocks,
                       std::uint32_t n_layers, std::uint32_t layer, std::uint32_t block_tokens, std::uint64_t kv_dim,
                       std::uint32_t n_table, std::uint64_t rows_needed, Access access) const {
        check_nonzero(block_tokens, r.op(), "block_tokens");
        check_nonzero(n_layers, r.op(), "n_layers");
        HALO_CHECK(layer < n_layers, ErrorCode::Kernel, "{}: layer {} of {}", r.op(), layer, n_layers);
        PoolView p;
        const std::uint64_t block_floats =
            mul_u64(mul_u64(mul_u64(n_layers, 2, r.op(), "pool"), block_tokens, r.op(), "pool"), kv_dim, r.op(), "pool");
        p.pool = r.f32(pool, n_pool_blocks, block_floats, access, "kv_pool");
        HALO_CHECK(p.pool.extent == 0 || p.pool.stride == block_floats * kF32, ErrorCode::Kernel, "{}: kv_pool must be dense",
                   r.op());
        p.pool.flat = true;
        p.table = r.flat(table, mul_u64(n_table, 4, r.op(), "block_table"), 4, Access::Read, "block_table");
        const std::uint64_t needed_blocks = (rows_needed + block_tokens - 1) / block_tokens;
        HALO_CHECK(needed_blocks <= n_table, ErrorCode::Kernel, "{}: {} rows need {} blocks, table has {}", r.op(), rows_needed,
                   needed_blocks, n_table);
        return p;
    }

    void kv_write(Stream& s, const KvWriteArgs& a) override {
        HipStream& st = stream(s, "KV_WRITE");
        const hip::Ops& ops = select(OpId::KvWrite, a.kernel);
        check_status_ref(a.status, OpId::KvWrite);
        const Resolver r(this, "KV_WRITE");
        const std::uint64_t end = static_cast<std::uint64_t>(a.start) + a.n_tokens;
        const PoolView p = pool_view(r, a.kv_pool, a.block_table, a.n_pool_blocks, a.n_layers, a.layer, a.block_tokens, a.kv_dim,
                                     a.n_block_table, end, Access::Write);
        const Operand k = r.f32(a.k, a.n_tokens, a.kv_dim, Access::Read, "k");
        const Operand v = r.f32(a.v, a.n_tokens, a.kv_dim, Access::Read, "v");
        r.no_overlap(p.pool, k);
        r.no_overlap(p.pool, v);
        r.no_overlap(p.pool, p.table);
        if (a.n_tokens == 0 || a.kv_dim == 0) return;
        hip::KvWriteArgs h;
        h.kv_pool = p.pool.view();
        h.n_pool_blocks = a.n_pool_blocks;
        h.n_layers = a.n_layers;
        h.layer = a.layer;
        h.block_tokens = a.block_tokens;
        h.kv_dim = a.kv_dim;
        h.block_table = p.table.view();
        h.start = a.start;
        h.n_tokens = a.n_tokens;
        h.k = k.view();
        h.v = v.view();
        h.status = st.scratch(4);
        ops.kv_write(st.target(), h);
        st.expect_status(r.op(), kBatchWide, h.status);
        st.op_done();
    }

    void attention(Stream& s, const AttentionArgs& a) override {
        HipStream& st = stream(s, "ATTENTION");
        const hip::Ops& ops = select(OpId::Attention, a.kernel);
        check_status_ref(a.status, OpId::Attention);
        const Resolver r(this, "ATTENTION");
        check_nonzero(a.n_head, r.op(), "n_head");
        check_nonzero(a.n_kv_head, r.op(), "n_kv_head");
        HALO_CHECK(a.n_head % a.n_kv_head == 0, ErrorCode::Kernel, "ATTENTION: n_head {} not a multiple of n_kv_head {}",
                   a.n_head, a.n_kv_head);
        const std::uint64_t hd = a.head_dim, T = a.n_tokens;
        const std::uint64_t kvd = static_cast<std::uint64_t>(a.n_kv_head) * hd;
        const std::uint64_t qhs = a.q_head_stride == 0 ? hd : a.q_head_stride;
        HALO_CHECK(qhs >= hd, ErrorCode::Kernel, "ATTENTION: q_head_stride {} < head_dim {}", qhs, hd);
        const std::uint64_t q_cols = (a.n_head - 1) * qhs + hd;
        const std::uint64_t rows = static_cast<std::uint64_t>(a.q_offset) + T;
        const PoolView p = pool_view(r, a.kv_pool, a.block_table, a.n_pool_blocks, a.n_layers, a.layer, a.block_tokens, kvd,
                                     a.n_block_table, rows, Access::Read);
        const Operand q = r.f32(a.q, T, q_cols, Access::Read, "q");
        const Operand out = r.f32(a.out, T, static_cast<std::uint64_t>(a.n_head) * hd, Access::Write, "out");
        r.no_overlap(out, q);
        r.no_overlap(out, p.pool);
        r.no_overlap(out, p.table);
        check_limit(a.head_dim, kMaxHeadDim, r.op(), "head_dim");
        if (T == 0) return;
        hip::AttentionArgs h;
        h.q = q.view();
        h.q_head_stride = static_cast<std::uint32_t>(qhs);
        h.kv_pool = p.pool.view();
        h.n_pool_blocks = a.n_pool_blocks;
        h.n_layers = a.n_layers;
        h.layer = a.layer;
        h.block_tokens = a.block_tokens;
        h.block_table = p.table.view();
        h.n_head = a.n_head;
        h.n_kv_head = a.n_kv_head;
        h.head_dim = a.head_dim;
        h.n_tokens = a.n_tokens;
        h.q_offset = a.q_offset;
        h.scale = a.scale;
        h.out = out.view();
        const std::uint64_t ws = ops.attention_workspace_bytes(a.n_tokens, a.n_head, a.q_offset);
        if (ws > 0) h.workspace = st.scratch(ws);
        h.status = st.scratch(4);
        ops.attention(st.target(), h);
        st.expect_status(r.op(), kBatchWide, h.status);
        st.op_done();
    }

    void eltwise(Stream& s, const EltwiseArgs& a, OpId op, bool allow_out_eq_a) {
        const std::string_view name = op_name(op);
        HipStream& st = stream(s, name.data());
        const hip::Ops& ops = select(op, a.kernel);
        const Resolver r(this, name.data());
        const Operand x = r.f32(a.a, a.rows, a.cols, Access::Read, "a");
        const Operand y = r.f32(a.b, a.rows, a.cols, Access::Read, "b");
        const Operand out = r.f32(a.out, a.rows, a.cols, Access::Write, "out");
        r.no_overlap(out, x, allow_out_eq_a);
        r.no_overlap(out, y);
        if (a.rows == 0 || a.cols == 0) return;
        const hip::EltwiseArgs h{x.view(), y.view(), out.view(), a.rows, a.cols};
        switch (op) {
            case OpId::Swiglu: ops.swiglu(st.target(), h); break;
            case OpId::MulSigmoid: ops.mul_sigmoid(st.target(), h); break;
            default: ops.add(st.target(), h); break;
        }
        st.op_done();
    }
    void swiglu(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::Swiglu, false); }
    void mul_sigmoid(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::MulSigmoid, false); }
    void add(Stream& s, const EltwiseArgs& a) override { eltwise(s, a, OpId::Add, true); }

    void lm_head(Stream& s, const LmHeadArgs& a) override {
        HipStream& st = stream(s, "LM_HEAD");
        const hip::Ops& ops = select_lm_head(a.gemv.kernel, a.kernel);
        check_status_ref(a.status, OpId::LmHead);
        const Resolver r(this, "LM_HEAD");
        const GemvArgs& g = a.gemv;
        const std::uint64_t rb = weight_row_bytes(g.wtype, g.cols, r.op());
        const Operand w = r.get(g.w, g.rows, rb, g.wtype == DType::F32 ? kF32 : 1, Access::Read, "w");
        const Operand x = r.f32(g.x, g.n_vec, g.cols, Access::Read, "x");
        const bool want_logits = !g.y.empty();
        const Operand y = want_logits ? r.f32(g.y, g.n_vec, g.rows, Access::Write, "logits") : Operand{};
        const Operand res = r.flat(a.result, mul_u64(g.n_vec, kArgmaxResultBytes, r.op(), "result"), 4, Access::Write, "result");
        for (const Operand* o : {&y, &res}) {
            r.no_overlap(*o, x);
            r.no_overlap(*o, w);
        }
        r.no_overlap(res, y);
        if (g.n_vec == 0) return;
        hip::LmHeadArgs h;
        h.gemv = hip::GemvArgs{g.wtype, w.view(), x.view(), y.view(), g.rows, g.cols, g.n_vec};
        h.workspace = st.scratch(ops.lm_head_workspace_bytes(g.rows, g.n_vec));
        h.result = res.view();
        ops.lm_head_argmax(st.target(), h);
        st.expect_argmax(r.op(), kBatchWide, h.result, g.n_vec);
        st.op_done();
    }

    void argmax(Stream& s, const ArgmaxArgs& a) override {
        HipStream& st = stream(s, "ARGMAX_FUSED");
        const hip::Ops& ops = select(OpId::Argmax, a.kernel);
        check_status_ref(a.status, OpId::Argmax);
        const Resolver r(this, "ARGMAX_FUSED");
        const Operand l = r.f32(a.logits, a.n_vec, a.n, Access::Read, "logits");
        const Operand res = r.flat(a.result, mul_u64(a.n_vec, kArgmaxResultBytes, r.op(), "result"), 4, Access::Write, "result");
        r.no_overlap(res, l);
        if (a.n_vec == 0) return;
        hip::ArgmaxArgs h;
        h.logits = l.view();
        h.n = a.n;
        h.n_vec = a.n_vec;
        h.workspace = st.scratch(ops.argmax_workspace_bytes(a.n, a.n_vec));
        h.result = res.view();
        ops.argmax(st.target(), h);
        st.expect_argmax(r.op(), kBatchWide, h.result, a.n_vec);
        st.op_done();
    }

    void top_k(Stream& s, const TopKArgs& a) override {
        HipStream& st = stream(s, "TOP_K");
        const hip::Ops& ops = select(OpId::TopK, a.kernel);
        check_status_ref(a.status, OpId::TopK);
        const Resolver r(this, "TOP_K");
        const Operand l = r.f32(a.logits, a.n_vec, a.n, Access::Read, "logits");
        const Operand ids = r.get(a.ids, a.n_vec, mul_u64(a.k, 4, r.op(), "ids"), 4, Access::Write, "ids");
        const Operand vals = r.f32(a.values, a.n_vec, a.k, Access::Write, "values");
        r.no_overlap(ids, l);
        r.no_overlap(vals, l);
        r.no_overlap(ids, vals);
        check_limit(a.k, kMaxTopK, r.op(), "k");
        if (a.n_vec == 0) return;
        hip::TopKArgs h;
        h.logits = l.view();
        h.n = a.n;
        h.k = a.k;
        h.n_vec = a.n_vec;
        const std::uint64_t ws = hip::topk_workspace_bytes(a.n, a.k, a.n_vec);
        if (ws > 0) h.workspace = st.scratch(ws);
        h.ids = ids.view();
        h.values = vals.view();
        h.status = st.scratch(4);
        ops.top_k(st.target(), h);
        st.expect_status(r.op(), kBatchWide, h.status);
        st.op_done();
    }

    void copy(Stream& s, const CopyArgs& a) override {
        HipStream& st = stream(s, "COPY");
        HALO_CHECK(a.kernel.variant_id == 0 || a.kernel.variant_id == kCopyId, ErrorCode::Config,
                   "COPY: unknown kernel variant id {} on the hip backend", a.kernel.variant_id);
        const Resolver r(this, "COPY");
        const Operand src = r.flat(a.src, a.bytes, 1, Access::Read, "src");
        const Operand dst = r.flat(a.dst, a.bytes, 1, Access::Write, "dst");
        r.no_overlap(dst, src);
        if (a.bytes == 0) return;
        hip::copy_async(st.device_stream(), dst.buf->hip(), dst.offset, src.buf->hip(), src.offset, a.bytes);
        st.op_done();
    }

private:
    std::unique_ptr<Buffer> wrap(std::byte* data, std::size_t n, bool writable) {
        // hip::Buffer::wrap_host needs a non-empty range; an empty import is a 1-byte view of
        // a private byte that no op can reach (bytes() == 0).
        if (n == 0) {
            auto storage = std::make_unique<std::byte[]>(1);
            hip::Buffer b = hip::Buffer::wrap_host(storage.get(), 1);
            return std::make_unique<HipBuffer>(this, std::move(b), std::move(storage), 0, Tier::Host, writable, true);
        }
        return std::make_unique<HipBuffer>(this, hip::Buffer::wrap_host(data, n), nullptr, n, Tier::Host, writable, true);
    }

    void check_alive(const char* what) const {
        HALO_CHECK(!poisoned_, ErrorCode::Device, "hip backend: {} after a device failure (the backend is poisoned; recreate it)",
                   what);
    }

    HipStream& stream(Stream& s, const char* op) {
        check_alive(op);
        auto* hs = dynamic_cast<HipStream*>(&s);
        HALO_CHECK(hs != nullptr && hs->owner() == this, ErrorCode::Api, "{}: stream of another backend", op);
        return *hs;
    }

    /// The prebuilt Ops of `k` for `op` (0 = the configured defaults).
    const hip::Ops& select(OpId op, KernelChoice k, std::string_view form = {}) const {
        if (k.variant_id == 0) return defaults_;
        const Entry* e = find_entry(k.variant_id);
        HALO_CHECK(e != nullptr && e->op == op && (form.empty() || e->form == form), ErrorCode::Config,
                   "{}: unknown kernel variant id {} on the hip backend{}", op_name(op), k.variant_id,
                   form.empty() ? std::string() : std::format(" (form {})", form));
        return per_variant_.at(e->id);
    }

    const hip::Ops& select_lm_head(KernelChoice gemv, KernelChoice head) const {
        if (gemv.variant_id == 0 && head.variant_id == 0) return defaults_;
        const auto it = lm_ops_.find(std::pair{gemv.variant_id, head.variant_id});
        HALO_CHECK(it != lm_ops_.end(), ErrorCode::Config,
                   "LM_HEAD: unknown kernel variant ids (gemv {}, argmax {}) on the hip backend (the GEMM tile is not an "
                   "LM-head variant)",
                   gemv.variant_id, head.variant_id);
        return it->second;
    }

    HipBackendOptions opt_;
    hip::Ops defaults_;
    std::map<std::uint32_t, hip::Ops> per_variant_;
    std::map<std::pair<std::uint32_t, std::uint32_t>, hip::Ops> lm_ops_;
    std::shared_ptr<hip::Context> ctx_;
    bool poisoned_ = false;
};

}  // namespace

hip::OpsOptions hip_bitwise_defaults() {
    hip::OpsOptions o;
    o.gemv = "gemv_generic_b64";
    o.attention = "attn_exact_b128";
    return o;
}

std::unique_ptr<Backend> make_hip_backend(const HipBackendOptions& options) { return std::make_unique<HipBackend>(options); }

}  // namespace halo::backend
