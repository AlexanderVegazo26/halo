// GDN prefix checkpoint store (D-013).

#include "halo/state/checkpoint.h"

#include <algorithm>
#include <utility>

namespace halo::state {

bool CheckpointStore::insert(Checkpoint c) {
    const std::uint64_t need = c.bytes();
    if (c.tokens.empty() || need > budget_) return false;
    // Same prefix already stored: replace it.
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->tokens == c.tokens) {
            used_ -= it->bytes();
            entries_.erase(it);
            break;
        }
    }
    while (used_ + need > budget_ && !entries_.empty()) {
        used_ -= entries_.back().bytes();
        entries_.pop_back();
        ++evictions_;
    }
    used_ += need;
    entries_.push_front(std::move(c));
    return true;
}

const Checkpoint* CheckpointStore::find(std::span<const std::int32_t> query, std::size_t max_len) {
    return find(query, max_len, [](const Checkpoint&) { return true; });
}

std::size_t CheckpointStore::erase_owner(std::uint64_t owner) noexcept {
    std::size_t n = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->owner == owner) {
            used_ -= it->bytes();
            it = entries_.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

const Checkpoint* CheckpointStore::find(std::span<const std::int32_t> query, std::size_t max_len,
                                        const std::function<bool(const Checkpoint&)>& usable) {
    const std::size_t limit = std::min(max_len, query.size());
    auto best = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const std::size_t n = it->tokens.size();
        if (n > limit || (best != entries_.end() && n <= best->tokens.size()) || !usable(*it)) continue;
        if (std::equal(it->tokens.begin(), it->tokens.end(), query.begin())) best = it;
    }
    if (best == entries_.end()) return nullptr;
    entries_.splice(entries_.begin(), entries_, best);  // most recently used; iterators stay valid
    return &entries_.front();
}

void CheckpointStore::clear() noexcept {
    entries_.clear();
    used_ = 0;
}

}  // namespace halo::state
