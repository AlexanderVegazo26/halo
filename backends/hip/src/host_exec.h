#pragma once
// Host emulation of a kernel launch (see kernels/hd.h for the execution model). Runs every
// workgroup of the grid, and within a workgroup every thread of each phase, sequentially,
// in forward or reverse order. Registers and LDS start poisoned (all bytes 0xFF = NaN /
// 0xFFFFFFFF), so a read of never-written state or of a value another thread writes in the
// *same* phase changes the result between the two orders — the unit tests run both orders
// and require bit-identical results.
//
// This models the kernel's work decomposition, indexing and arithmetic. It does NOT model
// device scheduling, the memory model, wave-level behaviour or the device libm.
//
// Perf note (cpp23-efficiency-review.md §G): every block/thread here runs fully sequentially,
// so wall-clock timing of this path says nothing about real device performance -- HIP has not
// yet run on real ROCm hardware in this project, so this emulation is the only path anyone has
// timed so far. Do not treat a timing taken through this path as representative of the device.

#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "kernels/hd.h"

namespace halo::hip::emu {

template <class T>
void poison(T& v) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memset(static_cast<void*>(&v), 0xFF, sizeof(T));
}

template <class Regs>
class HostExec {
public:
    HostExec(unsigned block, bool reverse) : block_(block), reverse_(reverse), regs_(block) {
        for (Regs& r : regs_) poison(r);
    }
    [[nodiscard]] unsigned block_dim() const noexcept { return block_; }

    template <class F>
    void phase(F&& f) {
        if (reverse_) {
            for (unsigned t = block_; t-- > 0;) f(t, regs_[t]);
        } else {
            for (unsigned t = 0; t < block_; ++t) f(t, regs_[t]);
        }
    }
    static void atomic_or(std::uint32_t* p, std::uint32_t v) noexcept { *p |= v; }

private:
    unsigned block_;
    bool reverse_;
    std::vector<Regs> regs_;
};

/// Runs body(exec, shared, bx, by) for every block of the launch.
template <class Regs, class Shared, class Body>
void run_grid(const kern::Launch& l, bool reverse, Body&& body) {
    auto sh = std::make_unique<Shared>();
    const unsigned n = l.grid_x * l.grid_y;
    for (unsigned k = 0; k < n; ++k) {
        const unsigned b = reverse ? n - 1 - k : k;
        poison(*sh);
        HostExec<Regs> ex(l.block, reverse);
        body(ex, *sh, b % l.grid_x, b / l.grid_x);
    }
}

}  // namespace halo::hip::emu
