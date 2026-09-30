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

std::size_t KvLayout::block_bytes() const {
    if (type == backend::KvType::F32) return block_floats() * sizeof(float);
    HALO_CHECK(n_layers > 0 && kv_dim > 0 && block_tokens > 0, ErrorCode::Config,
               "KvLayout: zero dimension (layers {}, kv_dim {}, block_tokens {})", n_layers, kv_dim, block_tokens);
    const std::size_t row = backend::kv_row_bytes(type, kv_dim);
    return mul_checked(mul_checked(mul_checked(n_layers, 2, "block"), block_tokens, "block"), row, "block");
}

std::size_t KvLayout::layer_block_bytes() const { return block_bytes() / n_layers; }  // block_bytes() checked n_layers > 0

std::size_t KvPool::max_blocks(const KvLayout& layout, Placement placement, std::uint64_t cap_bytes) {
    const std::uint64_t unit = placement == Placement::Single ? layout.block_bytes() : layout.layer_block_bytes();
    const std::uint64_t id_max = std::numeric_limits<BlockId>::max();
    if (cap_bytes == 0) return static_cast<std::size_t>(id_max);
    return static_cast<std::size_t>(std::min<std::uint64_t>(cap_bytes / unit, id_max));
}

std::uint64_t KvPool::segment_bytes_for(const KvLayout& layout, Placement placement, std::size_t n_blocks) {
    const std::uint64_t unit = placement == Placement::Single ? layout.block_bytes() : layout.layer_block_bytes();
    return mul_checked(unit, n_blocks, "segment");
}

KvPool::KvPool(KvLayout layout, std::size_t n_blocks, Placement placement)
    : layout_(layout),
      placement_(placement),
      block_floats_(layout.block_floats()),
      block_bytes_(layout.block_bytes()),
      layer_block_floats_(block_floats_ / layout.n_layers),
      layer_block_bytes_(layout.layer_block_bytes()) {
    HALO_CHECK(n_blocks > 0, ErrorCode::Config, "KvPool: zero blocks");
    HALO_CHECK(n_blocks <= std::numeric_limits<BlockId>::max(), ErrorCode::Config, "KvPool: {} blocks exceed the id range",
               n_blocks);
    const std::size_t total = mul_checked(block_floats_, n_blocks, "pool");
    (void)mul_checked(block_bytes_, n_blocks, "pool");
    slab_floats_ = mul_checked(layer_block_floats_, n_blocks, "pool");
    if (layout_.type == backend::KvType::F32) {
        try {
            // Zero-filled, as the backends' allocate() (the State-arena image starts zeroed on
            // every backend, so attach() uploads content only after real host writes).
            storage_ = std::make_unique<float[]>(total);
        } catch (const std::bad_alloc&) {
            throw_error(ErrorCode::Memory, "KvPool: cannot allocate {} bytes for {} KV blocks", total * sizeof(float), n_blocks);
        }
    } else {
        coherent_ = false;  // device-resident only: no host image exists (attach() allocates the device one)
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
               "KV cache pool exhausted: need {} block(s) of {} bytes, {} of {} free", n, block_bytes_,
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
    if (placement_ == Placement::PerLayer) {  // layer-major slabs: [layer][block][K|V][token][kv_dim]
        return layer * slab_floats_ + static_cast<std::size_t>(id) * layer_block_floats_ +
               kv * layout_.block_tokens * layout_.kv_dim;
    }
    return static_cast<std::size_t>(id) * block_floats_ + (layer * 2 + kv) * layout_.block_tokens * layout_.kv_dim;
}

float* KvPool::host_storage(const char* what) const {
    HALO_CHECK(coherent_, ErrorCode::Unsupported,
               "KvPool: {} on a device-resident pool (host storage was released at attach; use read_all())", what);
    return storage_.get();
}

float* KvPool::k_rows(BlockId id, std::size_t layer) {
    host_dirty_ = true;
    return host_storage("k_rows") + offset(id, layer, 0);
}
float* KvPool::v_rows(BlockId id, std::size_t layer) {
    host_dirty_ = true;
    return host_storage("v_rows") + offset(id, layer, 1);
}
const float* KvPool::k_rows(BlockId id, std::size_t layer) const { return host_storage("k_rows") + offset(id, layer, 0); }
const float* KvPool::v_rows(BlockId id, std::size_t layer) const { return host_storage("v_rows") + offset(id, layer, 1); }

// ---- backend attachment (ADR-001 §5.2) ----------------------------------------------------

void KvPool::attach(backend::Backend& be) {
    if (be_ == &be) return;
    HALO_CHECK(be_ == nullptr, ErrorCode::Api, "KvPool: already attached to another backend");
    HALO_CHECK(layout_.type == backend::KvType::F32 || be.kind() == backend::Kind::Vulkan, ErrorCode::Config,
               "KV cache type {} (HALO_KV_FP16 / HALO_KV_TYPE) is implemented only on the Vulkan backend; this backend is {} "
               "(unset the variable or use --backend vulkan)",
               backend::to_string(layout_.type), backend::to_string(be.kind()));
    const bool zero_copy = be.kind() == backend::Kind::Cpu || be.kind() == backend::Kind::HipEmulation;
    if (placement_ == Placement::PerLayer) {
        // One backend buffer per layer slab. Built into a local vector so a failure part-way
        // (allocation, import) leaves the pool unattached and attach() retryable.
        std::vector<std::unique_ptr<backend::Buffer>> layers;
        layers.reserve(layout_.n_layers);
        for (std::size_t l = 0; l < layout_.n_layers; ++l) {
            if (zero_copy) {
                float* slab = storage_.get() + l * slab_floats_;
                layers.push_back(be.import_host(std::as_writable_bytes(std::span(slab, slab_floats_))));
                HALO_CHECK(layers.back()->host_ptr() == static_cast<void*>(slab), ErrorCode::Backend,
                           "KvPool: a zero-copy backend returned an import that does not wrap KV segment {}", l);
            } else {
                layers.push_back(be.allocate(layer_image_bytes(), backend::Tier::Vram));
            }
        }
        if (!zero_copy && host_dirty_) {  // test sentinels written before attach carry over
            const auto s = be.create_stream();
            for (std::size_t l = 0; l < layout_.n_layers; ++l) {
                be.upload(*s, backend::TensorRef::of(*layers[l]),
                          std::as_bytes(std::span(storage_.get() + l * slab_floats_, slab_floats_)));
            }
            s->submit();
            s->wait();
        }
        dev_layers_ = std::move(layers);
    } else if (zero_copy) {
        dev_ = be.import_host(std::as_writable_bytes(std::span(storage_.get(), block_floats_ * total_blocks())));
        HALO_CHECK(dev_->host_ptr() == static_cast<void*>(storage_.get()), ErrorCode::Backend,
                   "KvPool: a zero-copy backend returned an import that does not wrap the pool storage");
    } else {
        // Device-resident (State arena). allocate() zero-fills on the cpu/vulkan backends;
        // host writes made so far (test sentinels) carry over.
        dev_ = be.allocate(image_bytes(), backend::Tier::Vram);
        be_ = &be;
        if (host_dirty_) {
            const auto s = be_->create_stream();
            be_->upload(*s, storage_ref(),
                        std::as_bytes(std::span(storage_.get(), block_floats_ * total_blocks())));
            s->submit();
            s->wait();
        }
    }
    be_ = &be;
    coherent_ = zero_copy;
    if (!zero_copy) storage_.reset();  // the host image is released (GBs); accessors throw
}

backend::TensorRef KvPool::storage_ref() const {
    HALO_CHECK(placement_ == Placement::Single, ErrorCode::Api,
               "KvPool: a per-layer pool has no single image (use layer_image())");
    HALO_CHECK(dev_ != nullptr, ErrorCode::Api, "KvPool: not attached to a backend");
    return backend::TensorRef::of(*dev_);
}

KvPool::LayerImage KvPool::layer_image(std::size_t layer) const {
    HALO_CHECK(layer < layout_.n_layers, ErrorCode::Kernel, "KvPool: layer {} of {}", layer, layout_.n_layers);
    if (placement_ == Placement::Single) {
        return {storage_ref(), static_cast<std::uint32_t>(layout_.n_layers), static_cast<std::uint32_t>(layer)};
    }
    HALO_CHECK(dev_layers_.size() == layout_.n_layers, ErrorCode::Api, "KvPool: not attached to a backend");
    return {backend::TensorRef::of(*dev_layers_[layer]), 1, 0};
}

std::vector<float> KvPool::read_all() const {
    HALO_CHECK(layout_.type == backend::KvType::F32, ErrorCode::Unsupported,
               "KvPool::read_all: the pool stores {} (not fp32) elements", backend::to_string(layout_.type));
    const std::size_t n = block_floats_ * total_blocks();
    if (placement_ == Placement::PerLayer) {
        // Re-interleave the layer slabs into the logical block-major image.
        std::vector<float> out(n);
        std::vector<float> tmp;
        for (std::size_t l = 0; l < layout_.n_layers; ++l) {
            const float* slab = nullptr;
            if (coherent_) {
                slab = storage_.get() + l * slab_floats_;
            } else {
                HALO_CHECK(dev_layers_.size() == layout_.n_layers, ErrorCode::Api, "KvPool: not attached to a backend");
                tmp.assign(slab_floats_, 0.0f);
                const auto s = be_->create_stream();
                be_->download(*s, backend::TensorRef::of(*dev_layers_[l]), std::as_writable_bytes(std::span(tmp)));
                s->submit();
                s->wait();
                slab = tmp.data();
            }
            for (std::size_t b = 0; b < total_blocks(); ++b) {
                std::memcpy(out.data() + b * block_floats_ + l * layer_block_floats_, slab + b * layer_block_floats_,
                            layer_block_floats_ * sizeof(float));
            }
        }
        return out;
    }
    if (coherent_) return {storage_.get(), storage_.get() + n};
    std::vector<float> out(n);
    const auto s = be_->create_stream();
    be_->download(*s, storage_ref(), std::as_writable_bytes(std::span(out)));
    s->submit();
    s->wait();
    return out;
}

void KvPool::copy_block(BlockId dst, BlockId src) {
    HALO_CHECK(!coherent_, ErrorCode::Api, "KvPool::copy_block is the device-resident COW path");
    // Synchronous so SequenceKv::reserve keeps its strong guarantee: the block table swap
    // happens only after the copy completed (ADR-001 §5.2: the host bookkeeping stays
    // strongly atomic). COW is rare (a write into a shared block).
    const auto s = be_->create_stream();
    if (placement_ == Placement::PerLayer) {
        const std::uint64_t bytes = layer_block_bytes_;  // the block's slice in each layer segment
        for (const auto& seg : dev_layers_) {
            be_->copy(*s,
                      backend::CopyArgs{{seg.get(), static_cast<std::uint64_t>(src) * bytes, bytes, 0},
                                        {seg.get(), static_cast<std::uint64_t>(dst) * bytes, bytes, 0},
                                        bytes,
                                        {}});
        }
    } else {
        const std::uint64_t bytes = block_bytes_;
        be_->copy(*s,
                  backend::CopyArgs{{dev_.get(), static_cast<std::uint64_t>(src) * bytes, bytes, 0},
                                    {dev_.get(), static_cast<std::uint64_t>(dst) * bytes, bytes, 0},
                                    bytes,
                                    {}});
    }
    s->submit();
    s->wait();
}

void KvPool::copy_block_host(BlockId dst, BlockId src) {
    HALO_CHECK(coherent_, ErrorCode::Api, "KvPool::copy_block_host is the host-coherent COW path");
    host_dirty_ = true;
    if (placement_ == Placement::PerLayer) {
        for (std::size_t l = 0; l < layout_.n_layers; ++l) {
            std::memcpy(storage_.get() + offset(dst, l, 0), storage_.get() + offset(src, l, 0),
                        layer_block_floats_ * sizeof(float));
        }
        return;
    }
    std::memcpy(storage_.get() + offset(dst, 0, 0), storage_.get() + offset(src, 0, 0), block_floats_ * sizeof(float));
}

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
    // --- host bookkeeping is no-throw from here on ---
    std::size_t next = 0;
    if (pool_->host_coherent()) {
        for (const std::size_t b : cow) {
            const BlockId dst = fresh[next++];
            pool_->copy_block_host(dst, blocks_[b]);  // placement-aware: a PerLayer block is not contiguous
            pool_->release(blocks_[b]);
            blocks_[b] = dst;
        }
    } else {
        // Device-resident pool (ADR-001 §5.2): a synchronous device-side copy per shared
        // block, so the block table changes only for copies that completed. A completed
        // swap is content-identical; a failed copy releases the not-yet-used fresh blocks.
        try {
            for (const std::size_t b : cow) {
                const BlockId dst = fresh[next++];
                pool_->copy_block(dst, blocks_[b]);
                pool_->release(blocks_[b]);
                blocks_[b] = dst;
            }
        } catch (...) {
            for (std::size_t i = next; i < fresh.size(); ++i) pool_->release(fresh[i]);
            throw;
        }
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
