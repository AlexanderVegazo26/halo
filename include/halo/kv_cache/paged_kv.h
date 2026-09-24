#pragma once
// Paged KV cache for the full-attention layers (ARCHITECTURE "SequenceState", D-005, D-013).
//
// Storage: a KvPool owns a fixed number of fp32 blocks, allocated once at construction.
// A block holds `block_tokens` consecutive token rows for *every* layer of the pool:
//   block[layer][K|V][token][kv_dim]   (kv_dim = n_kv_head * head_dim floats)
// so one block table per sequence serves all layers. The trunk and the MTP layer use
// separate pools (different layer counts).
//
// Sharing (prefix reuse): blocks are refcounted. SequenceKv::share_prefix() makes a new
// sequence reference another sequence's blocks; a write into a block with refcount > 1
// copies it first (copy-on-write), so a shared prefix is never modified.
//
// Exhaustion: allocation from an empty pool throws halo::Error(Memory). Every mutating
// SequenceKv operation either completes or leaves the sequence unchanged (strong guarantee):
// reserve() allocates everything it needs before touching the block table.
//
// fp16 storage (TRD §15 baseline) is not implemented yet: the CPU reference stores fp32.
//
// Thread-safety: KvPool's allocate/retain/release are internally synchronized, so
// sequences on different threads may share one pool. A SequenceKv is not thread-safe.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "halo/backends/cpu/views.h"

namespace halo::kv_cache {

using BlockId = std::uint32_t;

struct KvLayout {
    std::size_t n_layers = 0;      ///< attention layers stored in this pool
    std::size_t kv_dim = 0;        ///< floats per K (and per V) row = n_kv_head * head_dim
    std::size_t block_tokens = 16; ///< token rows per block

    /// floats in one block (all layers, K and V). Throws Error(Config) on overflow.
    [[nodiscard]] std::size_t block_floats() const;
    [[nodiscard]] std::size_t block_bytes() const { return block_floats() * sizeof(float); }
};

class KvPool {
public:
    /// Allocates n_blocks blocks up front. Throws Error(Config) for a zero dimension or
    /// size overflow, Error(Memory) if the storage cannot be allocated.
    KvPool(KvLayout layout, std::size_t n_blocks);

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

    [[nodiscard]] float* k_rows(BlockId id, std::size_t layer) noexcept;
    [[nodiscard]] float* v_rows(BlockId id, std::size_t layer) noexcept;
    [[nodiscard]] const float* k_rows(BlockId id, std::size_t layer) const noexcept;
    [[nodiscard]] const float* v_rows(BlockId id, std::size_t layer) const noexcept;

private:
    [[nodiscard]] std::size_t offset(BlockId id, std::size_t layer, std::size_t kv) const noexcept;

    KvLayout layout_;
    std::size_t block_floats_ = 0;
    std::unique_ptr<float[]> storage_;
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
