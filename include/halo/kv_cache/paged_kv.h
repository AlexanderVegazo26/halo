#pragma once
// Paged KV cache for the full-attention layers (ARCHITECTURE "SequenceState", D-005, D-013).
//
// Storage: a KvPool owns a fixed number of fp32 blocks, allocated once at construction.
// A block holds `block_tokens` consecutive token rows for *every* layer of the pool:
//   block[layer][K|V][token][kv_dim]   (kv_dim = n_kv_head * head_dim floats)
// so one block table per sequence serves all layers. The trunk and the MTP layer use
// separate pools (different layer counts).
//
// Placement (ADR-001 §5.2, WS-BI-2 stage 2): the pool image lives in host memory until
// attach(backend) gives it a State-arena backend buffer — zero-copy over the host storage
// on the CPU / HIP-emulation backends (host accessors stay valid), device-resident on GPU
// backends. On a device-resident pool the host accessors throw Error(Unsupported) (the ADR's
// rule for a pool whose host_ptr() is null); read_all() works in both modes (tests).
//
// Sharing (prefix reuse): blocks are refcounted. SequenceKv::share_prefix() makes a new
// sequence reference another sequence's blocks; a write into a block with refcount > 1
// copies it first (copy-on-write) — a host memcpy on host pools, a synchronous device-side
// Backend::copy on device-resident pools (synchronous so reserve() keeps its strong
// guarantee: the block table changes only after the copy completed).
//
// Exhaustion: allocation from an empty pool throws halo::Error(Memory). Every mutating
// SequenceKv operation either completes or leaves the sequence unchanged (strong guarantee):
// reserve() allocates everything it needs before touching the block table.
//
// Storage format (KvLayout::type, backend::KvType): fp32 by default. F16 and Q8 (opt-in, engine
// env HALO_KV_FP16=1 / HALO_KV_TYPE=q8) are device-resident Vulkan formats: the host storage
// image is never allocated, the host accessors (k_rows / v_rows / write / keys / values) and
// read_all() throw Error(Unsupported), and attach() to a non-Vulkan backend throws
// Error(Config). The CPU reference always stores fp32.
//
// Segments (large contexts): a backend can cap one buffer (Vulkan/RADV: min(maxMemoryAllocationSize,
// maxStorageBufferRange), 4 GiB), and the attention / kv-write kernels bind ONE pool buffer per
// dispatch. Placement::Single (default; also every CPU / HIP pool) is one contiguous
// block-major image. Placement::PerLayer stores one buffer ("segment") per attention layer,
// each holding block[K|V][token][kv_dim] for that layer only (layer-major slabs); the caller
// binds layer_image(l) -- buffer + n_layers = 1, layer = 0 -- so the kernels need no change.
// A block id is the same in every segment, so one block table still serves all layers and a
// block never straddles buffers. Each segment is 1/n_layers of the pool, which is what lifts
// the cap: 128k tokens x 2 sequences of the 27B model is 32 GiB in fp32 but only 2 GiB per layer.
// KvPool::max_blocks() gives the largest pool a per-buffer cap admits (fp32, 4 GiB cap: 32768
// blocks = 512Ki tokens per layer buffer, so fp32 128k x 4 sequences, 8 blocks over once each
// sequence's COW / draft slack is counted, does not fit; fp16 does). The host
// accessors, read_all() (always the logical block-major image) and copy-on-write behave
// identically in both placements.
//
// KV bytes per token (exact, from KvLayout): n_layers x 2 x kv_row_bytes(type, kv_dim), with
// kv_dim = n_kv_head x head_dim. For the 27B model shape in memory::qwen38_27b_shape (16 attention
// layers, 4 KV heads x 256 = 1024; taken from the planner constants, not re-read from a GGUF):
//   fp32 128 KiB/token (2 MiB per 16-token block), fp16 64 KiB, q8 36 KiB.
// The MTP pool has one layer: 8 KiB/token in fp32. So 128k tokens is 16 GiB (fp32) per sequence;
// 4 GiB per layer holds 512Ki tokens of fp32 across all sequences (1Mi fp16, 1.8Mi q8).
//
// Thread-safety: KvPool's allocate/retain/release are internally synchronized, so
// sequences on different threads may share one pool. A SequenceKv is not thread-safe.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

#include "halo/backend/backend.h"
#include "halo/backends/cpu/views.h"

namespace halo::kv_cache {

using BlockId = std::uint32_t;

struct KvLayout {
    std::size_t n_layers = 0;      ///< attention layers stored in this pool
    std::size_t kv_dim = 0;        ///< floats per K (and per V) row = n_kv_head * head_dim
    std::size_t block_tokens = 16; ///< token rows per block
    backend::KvType type = backend::KvType::F32;  ///< element format of the pool image

    /// floats in one block (all layers, K and V) -- an element count, not a byte size for a
    /// non-F32 type. Throws Error(Config) on overflow.
    [[nodiscard]] std::size_t block_floats() const;
    /// bytes in one block in the `type` format. Throws Error(Config) on overflow or if kv_dim
    /// is not representable in `type`.
    [[nodiscard]] std::size_t block_bytes() const;
    /// bytes of one block's slice for a single layer (block_bytes() / n_layers): the unit of a
    /// Placement::PerLayer segment. Same errors as block_bytes().
    [[nodiscard]] std::size_t layer_block_bytes() const;
};

/// How the pool image is split into backend buffers (see "Segments" above).
enum class Placement : std::uint8_t {
    Single,    ///< one contiguous block-major buffer
    PerLayer,  ///< one buffer per attention layer (layer-major slabs)
};

[[nodiscard]] constexpr std::string_view to_string(Placement p) noexcept {
    return p == Placement::Single ? "single" : "per-layer";
}

class KvPool {
public:
    /// Allocates n_blocks blocks up front. Throws Error(Config) for a zero dimension or
    /// size overflow, Error(Memory) if the storage cannot be allocated.
    /// host_image = false skips the zero-filled host mirror of an fp32 pool: the pool is then
    /// device-resident only (as fp16/q8 always are) and attach() must get a non-zero-copy
    /// backend. A GPU backend should pass false, else a 44 GiB pool is first built in host RAM.
    KvPool(KvLayout layout, std::size_t n_blocks, Placement placement = Placement::Single, bool host_image = true);

    /// Largest block count whose biggest single buffer is <= cap_bytes (0 = no cap: the
    /// largest id-representable count). Single: cap / block_bytes; PerLayer: cap / layer_block_bytes.
    /// 0 when even one block does not fit. Throws Error(Config) for an invalid layout.
    [[nodiscard]] static std::size_t max_blocks(const KvLayout& layout, Placement placement, std::uint64_t cap_bytes);
    /// Bytes of the largest single buffer of a pool of n_blocks (the value a per-buffer cap limits).
    [[nodiscard]] static std::uint64_t segment_bytes_for(const KvLayout& layout, Placement placement, std::size_t n_blocks);

    [[nodiscard]] Placement placement() const noexcept { return placement_; }
    /// Backend buffers the image occupies: 1 (Single) or n_layers (PerLayer).
    [[nodiscard]] std::size_t segment_count() const noexcept { return placement_ == Placement::Single ? 1 : layout_.n_layers; }
    /// Bytes of one segment's buffer (all segments are equal-sized).
    [[nodiscard]] std::uint64_t segment_bytes() const { return segment_bytes_for(layout_, placement_, total_blocks()); }
    /// Bytes of the whole pool (all segments).
    [[nodiscard]] std::uint64_t total_bytes() const { return static_cast<std::uint64_t>(block_bytes_) * total_blocks(); }

    KvPool(const KvPool&) = delete;
    KvPool& operator=(const KvPool&) = delete;

    [[nodiscard]] const KvLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] std::size_t total_blocks() const noexcept { return refcount_.size(); }
    [[nodiscard]] std::size_t free_blocks() const;
    [[nodiscard]] std::size_t used_blocks() const { return total_blocks() - free_blocks(); }

    /// A block with refcount 1. Throws Error(Memory) when none is free.
    [[nodiscard]] BlockId allocate();
    /// n blocks, all or nothing. Throws Error(Memory) (allocating none) if fewer are free.
    [[nodiscard]] std::vector<BlockId> allocate_n(std::size_t n);
    void retain(BlockId id);
    void release(BlockId id) noexcept;
    [[nodiscard]] std::uint32_t refcount(BlockId id) const;

    [[nodiscard]] float* k_rows(BlockId id, std::size_t layer);
    [[nodiscard]] float* v_rows(BlockId id, std::size_t layer);
    [[nodiscard]] const float* k_rows(BlockId id, std::size_t layer) const;
    [[nodiscard]] const float* v_rows(BlockId id, std::size_t layer) const;

    // ---- backend attachment (ADR-001 §5.2: the pool image is a State-arena buffer) --------
    /// Gives the pool image a backend-owned image (idempotent for the same backend;
    /// Error(Api) for another). CPU / HIP emulation: zero-copy over the host storage. GPU
    /// backends: a device-resident buffer; the current host content (zeros, or test
    /// sentinels written so far) is uploaded, then the host storage is released.
    /// Error(Memory) on allocation failure.
    void attach(backend::Backend& be);
    [[nodiscard]] bool attached() const noexcept { return be_ != nullptr; }
    /// True when the host storage is the image (unattached, or a zero-copy attachment), so
    /// the host accessors are valid. False on a device-resident pool (accessors then throw
    /// Error(Unsupported), per ADR-001 §5.2).
    [[nodiscard]] bool host_coherent() const noexcept { return coherent_; }
    /// The whole pool image as a backend operand (attached only; Error(Api) otherwise, and
    /// Error(Api) for a PerLayer pool, which has no single image -- use layer_image()).
    [[nodiscard]] backend::TensorRef storage_ref() const;
    /// What a kv_write / attention call needs to address attention layer `layer`: the buffer
    /// holding it and the (n_layers, layer) pair to pass with it. Single: the whole image and
    /// (layout n_layers, layer); PerLayer: that layer's segment and (1, 0). Attached only.
    struct LayerImage {
        backend::TensorRef ref;
        std::uint32_t n_layers = 1;
        std::uint32_t layer = 0;
    };
    [[nodiscard]] LayerImage layer_image(std::size_t layer) const;
    /// A copy of the whole pool image (host pools: a host copy; device pools: a download),
    /// always in the logical block-major order block[layer][K|V][token][kv_dim] whatever the
    /// placement. For tests.
    [[nodiscard]] std::vector<float> read_all() const;
    /// Device-side copy of block `src` to block `dst`, synchronously (the COW of
    /// SequenceKv::reserve on a device-resident pool). Error(Api) on a host-coherent pool.
    void copy_block(BlockId dst, BlockId src);
    /// Host-side copy of block `src` to block `dst` (all layers; the COW of SequenceKv::reserve
    /// on a host-coherent pool). Error(Api) on a device-resident pool.
    void copy_block_host(BlockId dst, BlockId src);

private:
    [[nodiscard]] std::size_t offset(BlockId id, std::size_t layer, std::size_t kv) const noexcept;
    [[nodiscard]] float* host_storage(const char* what) const;  // Error(Unsupported) if device-resident
    [[nodiscard]] std::size_t image_bytes() const noexcept { return block_bytes_ * total_blocks(); }
    [[nodiscard]] std::size_t layer_image_bytes() const noexcept { return layer_block_bytes_ * total_blocks(); }

    KvLayout layout_;
    Placement placement_ = Placement::Single;
    std::size_t block_floats_ = 0;
    std::size_t block_bytes_ = 0;  // block_floats_ * 4 for F32
    std::size_t layer_block_floats_ = 0;  // 2 * block_tokens * kv_dim: one layer's slice of a block
    std::size_t layer_block_bytes_ = 0;   // block_bytes_ / n_layers
    std::size_t slab_floats_ = 0;         // PerLayer: layer_block_floats_ * total_blocks()
    std::unique_ptr<float[]> storage_;  // null on a device-resident pool; PerLayer: layer-major slabs
    backend::Backend* be_ = nullptr;
    std::unique_ptr<backend::Buffer> dev_;                  // Single
    std::vector<std::unique_ptr<backend::Buffer>> dev_layers_;  // PerLayer: one per layer
    bool coherent_ = true;
    bool host_dirty_ = false;  // host writes since construction (attach uploads them)
    mutable std::mutex mu_;
    std::vector<std::uint32_t> refcount_;  // 0 = free
    std::vector<BlockId> free_list_;
};

/// One sequence's view of a pool: block table + logical length.
class SequenceKv {
public:
    explicit SequenceKv(KvPool& pool) : pool_(&pool) {}
    ~SequenceKv();
    SequenceKv(const SequenceKv&) = delete;
    SequenceKv& operator=(const SequenceKv&) = delete;
    SequenceKv(SequenceKv&& other) noexcept;
    SequenceKv& operator=(SequenceKv&& other) noexcept;

    [[nodiscard]] KvPool& pool() const noexcept { return *pool_; }
    [[nodiscard]] std::size_t length() const noexcept { return length_; }
    /// Rows writable without allocation (length() + reserved rows).
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::span<const BlockId> blocks() const noexcept { return blocks_; }

    /// Makes rows [length, length + n) writable: allocates missing blocks and copies any
    /// shared block those rows fall into. All-or-nothing; throws Error(Memory).
    void reserve(std::size_t n);
    /// Writes one K and V row at row index `row` in [length, capacity) of `layer`.
    /// Requires reserve() to have covered it (Error(Kernel) otherwise).
    void write(std::size_t layer, std::size_t row, std::span<const float> k, std::span<const float> v);
    /// length += n; requires length + n <= capacity (Error(Kernel)).
    void commit(std::size_t n);
    /// length = min(length, n); releases blocks no longer needed. Never throws.
    void truncate(std::size_t n) noexcept;
    /// Drops everything.
    void clear() noexcept { truncate(0); }

    /// This (empty) sequence references the first n rows of `src` (same pool). Shared
    /// blocks are copied on the first write. Throws Error(Api) if this is not empty, the
    /// pools differ or n > src.length().
    void share_prefix(const SequenceKv& src, std::size_t n);

    /// Paged K / V views over rows [0, rows) of `layer` (rows <= capacity). `table` is
    /// filled with block pointers and must outlive the returned view.
    [[nodiscard]] cpu::PagedRows keys(std::size_t layer, std::size_t rows, std::vector<const float*>& table) const;
    [[nodiscard]] cpu::PagedRows values(std::size_t layer, std::size_t rows, std::vector<const float*>& table) const;

private:
    [[nodiscard]] cpu::PagedRows view(std::size_t layer, std::size_t rows, bool k, std::vector<const float*>& table) const;
    void release_all() noexcept;

    KvPool* pool_;
    std::vector<BlockId> blocks_;
    std::size_t length_ = 0;
};

}  // namespace halo::kv_cache
