// Paged fp32 KV cache: fixed block pool, refcounted sharing, copy-on-write.

#include "halo/kv_cache/paged_kv.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "halo/core/error.h"

namespace halo::kv_cache {

namespace {

std::size_t mul_checked(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw_error(ErrorCode::Config, "KV cache size overflow computing {}", what);
    }
    return a * b;
}

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

}  // namespace

std::size_t KvLayout::block_floats() const {
    HALO_CHECK(n_layers > 0 && kv_dim > 0 && block_tokens > 0, ErrorCode::Config,
               "KvLayout: zero dimension (layers {}, kv_dim {}, block_tokens {})", n_layers, kv_dim, block_tokens);
    return mul_checked(mul_checked(mul_checked(n_layers, 2, "block"), block_tokens, "block"), kv_dim, "block");
}

KvPool::KvPool(KvLayout layout, std::size_t n_blocks) : layout_(layout), block_floats_(layout.block_floats()) {
    HALO_CHECK(n_blocks > 0, ErrorCode::Config, "KvPool: zero blocks");
    HALO_CHECK(n_blocks <= std::numeric_limits<BlockId>::max(), ErrorCode::Config, "KvPool: {} blocks exceed the id range",
               n_blocks);
    const std::size_t total = mul_checked(block_floats_, n_blocks, "pool");
    try {
        storage_ = std::make_unique_for_overwrite<float[]>(total);
    } catch (const std::bad_alloc&) {
        throw_error(ErrorCode::Memory, "KvPool: cannot allocate {} bytes for {} KV blocks", total * sizeof(float), n_blocks);
    }
    refcount_.assign(n_blocks, 0);
    free_list_.reserve(n_blocks);
    for (std::size_t i = n_blocks; i-- > 0;) free_list_.push_back(static_cast<BlockId>(i));  // pop_back yields 0 first
}

std::size_t KvPool::free_blocks() const {
    const std::lock_guard lock(mu_);
    return free_list_.size();
}

BlockId KvPool::allocate() { return allocate_n(1).front(); }

std::vector<BlockId> KvPool::allocate_n(std::size_t n) {
    const std::lock_guard lock(mu_);
    HALO_CHECK(n <= free_list_.size(), ErrorCode::Memory,
               "KV cache pool exhausted: need {} block(s) of {} bytes, {} of {} free", n, block_floats_ * sizeof(float),
               free_list_.size(), refcount_.size());
    std::vector<BlockId> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = free_list_.back();
        free_list_.pop_back();
        refcount_[out[i]] = 1;
    }
    return out;
}

void KvPool::retain(BlockId id) {
    const std::lock_guard lock(mu_);
    HALO_CHECK(id < refcount_.size() && refcount_[id] > 0, ErrorCode::Kernel, "KvPool: retain of free block {}", id);
    ++refcount_[id];
}

void KvPool::release(BlockId id) noexcept {
    const std::lock_guard lock(mu_);
    if (id >= refcount_.size() || refcount_[id] == 0) return;  // programming error; never double-free
    if (--refcount_[id] == 0) free_list_.push_back(id);
}

std::uint32_t KvPool::refcount(BlockId id) const {
    const std::lock_guard lock(mu_);
    HALO_CHECK(id < refcount_.size(), ErrorCode::Kernel, "KvPool: block {} out of range", id);
    return refcount_[id];
}

std::size_t KvPool::offset(BlockId id, std::size_t layer, std::size_t kv) const noexcept {
    return static_cast<std::size_t>(id) * block_floats_ + (layer * 2 + kv) * layout_.block_tokens * layout_.kv_dim;
}

float* KvPool::k_rows(BlockId id, std::size_t layer) noexcept { return storage_.get() + offset(id, layer, 0); }
float* KvPool::v_rows(BlockId id, std::size_t layer) noexcept { return storage_.get() + offset(id, layer, 1); }
const float* KvPool::k_rows(BlockId id, std::size_t layer) const noexcept { return storage_.get() + offset(id, layer, 0); }
const float* KvPool::v_rows(BlockId id, std::size_t layer) const noexcept { return storage_.get() + offset(id, layer, 1); }

// ---------------------------------------------------------------------------------------
// SequenceKv
// ---------------------------------------------------------------------------------------

SequenceKv::~SequenceKv() { release_all(); }

SequenceKv::SequenceKv(SequenceKv&& other) noexcept
    : pool_(other.pool_), blocks_(std::move(other.blocks_)), length_(std::exchange(other.length_, 0)) {
    other.blocks_.clear();
}

SequenceKv& SequenceKv::operator=(SequenceKv&& other) noexcept {
    if (this != &other) {
        release_all();
        pool_ = other.pool_;
        blocks_ = std::move(other.blocks_);
        other.blocks_.clear();
        length_ = std::exchange(other.length_, 0);
    }
    return *this;
}

void SequenceKv::release_all() noexcept {
    for (const BlockId b : blocks_) pool_->release(b);
    blocks_.clear();
    length_ = 0;
}

std::size_t SequenceKv::capacity() const noexcept { return blocks_.size() * pool_->layout().block_tokens; }

void SequenceKv::reserve(std::size_t n) {
    const std::size_t bt = pool_->layout().block_tokens;
    HALO_CHECK(n <= std::numeric_limits<std::size_t>::max() - length_, ErrorCode::Api, "SequenceKv: reserve overflow");
    const std::size_t end = length_ + n;
    const std::size_t need_blocks = ceil_div(end, bt);
    // Blocks that rows [length, end) touch and that are shared must be copied first.
    std::vector<std::size_t> cow;
    if (n > 0) {
        for (std::size_t b = length_ / bt; b < std::min(need_blocks, blocks_.size()); ++b) {
            if (pool_->refcount(blocks_[b]) > 1) cow.push_back(b);
        }
    }
    const std::size_t add = need_blocks > blocks_.size() ? need_blocks - blocks_.size() : 0;
    blocks_.reserve(need_blocks);                                       // may throw; nothing changed yet
    std::vector<BlockId> fresh = pool_->allocate_n(add + cow.size());  // throws before any change
    // --- no-throw from here on ---
    const std::size_t block_floats = pool_->layout().block_floats();
    std::size_t next = 0;
    for (const std::size_t b : cow) {
        const BlockId dst = fresh[next++];
        std::memcpy(pool_->k_rows(dst, 0), pool_->k_rows(blocks_[b], 0), block_floats * sizeof(float));
        pool_->release(blocks_[b]);
        blocks_[b] = dst;
    }
    while (next < fresh.size()) blocks_.push_back(fresh[next++]);
}

void SequenceKv::write(std::size_t layer, std::size_t row, std::span<const float> k, std::span<const float> v) {
    const KvLayout& l = pool_->layout();
    HALO_CHECK(layer < l.n_layers, ErrorCode::Kernel, "SequenceKv: layer {} of {}", layer, l.n_layers);
    HALO_CHECK(row >= length_ && row < capacity(), ErrorCode::Kernel, "SequenceKv: row {} outside reserved [{}, {})", row,
               length_, capacity());
    HALO_CHECK(k.size() == l.kv_dim && v.size() == l.kv_dim, ErrorCode::Kernel, "SequenceKv: row of {}/{} floats, kv_dim {}",
               k.size(), v.size(), l.kv_dim);
    const BlockId b = blocks_[row / l.block_tokens];
    const std::size_t off = (row % l.block_tokens) * l.kv_dim;
    std::memcpy(pool_->k_rows(b, layer) + off, k.data(), l.kv_dim * sizeof(float));
    std::memcpy(pool_->v_rows(b, layer) + off, v.data(), l.kv_dim * sizeof(float));
}

void SequenceKv::commit(std::size_t n) {
    HALO_CHECK(n <= capacity() - length_, ErrorCode::Kernel, "SequenceKv: commit {} rows past capacity {} (length {})", n,
               capacity(), length_);
    length_ += n;
}

void SequenceKv::truncate(std::size_t n) noexcept {
    const std::size_t bt = pool_->layout().block_tokens;
    length_ = std::min(length_, n);
    const std::size_t keep = ceil_div(length_, bt);
    while (blocks_.size() > keep) {
        pool_->release(blocks_.back());
        blocks_.pop_back();
    }
}

void SequenceKv::share_prefix(const SequenceKv& src, std::size_t n) {
    HALO_CHECK(blocks_.empty() && length_ == 0, ErrorCode::Api, "SequenceKv::share_prefix: target is not empty");
    HALO_CHECK(src.pool_ == pool_, ErrorCode::Api, "SequenceKv::share_prefix: different pools");
    HALO_CHECK(n <= src.length_, ErrorCode::Api, "SequenceKv::share_prefix: {} rows of a {}-row sequence", n, src.length_);
    const std::size_t nb = ceil_div(n, pool_->layout().block_tokens);
    blocks_.reserve(nb);  // may throw; nothing retained yet
    for (std::size_t b = 0; b < nb; ++b) {
        pool_->retain(src.blocks_[b]);
        blocks_.push_back(src.blocks_[b]);
    }
    length_ = n;
}

cpu::PagedRows SequenceKv::view(std::size_t layer, std::size_t rows, bool k, std::vector<const float*>& table) const {
    const KvLayout& l = pool_->layout();
    HALO_CHECK(layer < l.n_layers, ErrorCode::Kernel, "SequenceKv: layer {} of {}", layer, l.n_layers);
    HALO_CHECK(rows <= capacity(), ErrorCode::Kernel, "SequenceKv: view of {} rows, capacity {}", rows, capacity());
    table.resize(blocks_.size());
    const KvPool& pool = *pool_;
    for (std::size_t b = 0; b < blocks_.size(); ++b) table[b] = k ? pool.k_rows(blocks_[b], layer) : pool.v_rows(blocks_[b], layer);
    if (rows == 0) return cpu::PagedRows::contiguous(nullptr, 0, l.kv_dim, l.kv_dim);
    return cpu::PagedRows::paged(table, l.block_tokens, rows, l.kv_dim, l.kv_dim);
}

cpu::PagedRows SequenceKv::keys(std::size_t layer, std::size_t rows, std::vector<const float*>& table) const {
    return view(layer, rows, true, table);
}

cpu::PagedRows SequenceKv::values(std::size_t layer, std::size_t rows, std::vector<const float*>& table) const {
    return view(layer, rows, false, table);
}

}  // namespace halo::kv_cache
