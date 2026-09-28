// CONV1D_SHORT: causal depthwise conv1d + SiLU with a rolling conv state.

#include <algorithm>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "kernel_common.h"

namespace halo::cpu {

void causal_conv1d_silu(ConstRows x, ConstRows weight, Rows conv_state, Rows out, ThreadPool* pool,
                        std::span<float> state_slots) {
    constexpr const char* kOp = "causal_conv1d_silu";
    const std::size_t n_tok = x.rows();
    const std::size_t n_ch = x.cols();
    const std::size_t k = weight.cols();
    HALO_CHECK(k >= 1, ErrorCode::Kernel, "{}: kernel size must be >= 1", kOp);
    HALO_CHECK(weight.rows() == n_ch, ErrorCode::Kernel, "{}: weight has {} rows, C = {}", kOp,
               weight.rows(), n_ch);
    HALO_CHECK(conv_state.rows() == k - 1 && (k == 1 || conv_state.cols() == n_ch), ErrorCode::Kernel,
               "{}: conv_state is {}x{}, expected {}x{}", kOp, conv_state.rows(), conv_state.cols(),
               k - 1, n_ch);
    detail::check_same_shape(x, out, kOp, "out");
    detail::check_alias_exact_or_disjoint(x, out, kOp, "x");
    detail::check_disjoint(conv_state, out, kOp, "conv_state");
    detail::check_disjoint(conv_state, x, kOp, "conv_state (as x)");
    detail::check_disjoint(weight, out, kOp, "weight");
    const std::size_t slot_n = (k - 1) * n_ch;  // floats per slot
    std::size_t n_slots = 0;
    if (!state_slots.empty()) {
        HALO_CHECK(slot_n > 0 && state_slots.size() % slot_n == 0, ErrorCode::Kernel,
                   "{}: state_slots has {} floats, not a multiple of (K-1)*C = {}", kOp, state_slots.size(), slot_n);
        n_slots = state_slots.size() / slot_n;
        const ConstRows sl(state_slots);
        for (const auto& [r, what] : {std::pair{x, "x"}, std::pair{weight, "weight"},
                                      std::pair{ConstRows(conv_state), "conv_state"}, std::pair{ConstRows(out), "out"}}) {
            HALO_CHECK(!detail::overlaps(r, sl), ErrorCode::Kernel, "{}: state_slots overlaps {}", kOp, what);
        }
    }

    // Channels are independent: split them into fixed blocks. Within a block, time runs
    // sequentially and the K-1 previous inputs live in a ring buffer, so `out` may alias
    // `x` (each input is copied into the ring before its output is written).
    constexpr std::size_t kBlock = 64;
    const std::size_t hist = k - 1;
    const std::size_t n_blocks = (n_ch + kBlock - 1) / kBlock;
    parallel_for(pool, n_blocks, [&](std::size_t b0, std::size_t b1) {
        std::vector<float> wt(k * kBlock);      // wt[j * kBlock + c]: tap j, channel c
        std::vector<float> ring(hist * kBlock);  // ring[slot * kBlock + c]
        std::array<float, kBlock> acc{};
        for (std::size_t blk = b0; blk < b1; ++blk) {
            const std::size_t c0 = blk * kBlock;
            const std::size_t nc = std::min(kBlock, n_ch - c0);
            for (std::size_t c = 0; c < nc; ++c) {
                for (std::size_t j = 0; j < k; ++j) wt[j * kBlock + c] = weight(c0 + c, j);
            }
            for (std::size_t s = 0; s < hist; ++s) {
                for (std::size_t c = 0; c < nc; ++c) ring[s * kBlock + c] = conv_state(s, c0 + c);
            }
            std::size_t oldest = 0;  // ring slot holding the oldest input
            for (std::size_t t = 0; t < n_tok; ++t) {
                const float* xr = x.row_ptr(t) + c0;
                float* yr = out.row_ptr(t) + c0;
                for (std::size_t c = 0; c < nc; ++c) acc[c] = 0.0f;
                for (std::size_t j = 0; j < hist; ++j) {
                    const float* wj = &wt[j * kBlock];
                    const float* hj = &ring[((oldest + j) % hist) * kBlock];
                    for (std::size_t c = 0; c < nc; ++c) acc[c] += wj[c] * hj[c];
                }
                const float* wl = &wt[hist * kBlock];
                for (std::size_t c = 0; c < nc; ++c) acc[c] += wl[c] * xr[c];
                if (hist > 0) {
                    float* slot = &ring[oldest * kBlock];
                    for (std::size_t c = 0; c < nc; ++c) slot[c] = xr[c];  // before writing out
                    oldest = (oldest + 1) % hist;
                }
                for (std::size_t c = 0; c < nc; ++c) yr[c] = detail::silu(acc[c]);
                const std::size_t slot = n_tok - 1 - t;  // slot 0 = state after the last row
                if (slot < n_slots) {
                    float* dst = state_slots.data() + slot * slot_n;
                    for (std::size_t s = 0; s < hist; ++s) {
                        const float* src = &ring[((oldest + s) % hist) * kBlock];
                        for (std::size_t c = 0; c < nc; ++c) dst[s * n_ch + c0 + c] = src[c];
                    }
                }
            }
            for (std::size_t s = 0; s < hist; ++s) {
                const float* src = &ring[((oldest + s) % hist) * kBlock];
                for (std::size_t c = 0; c < nc; ++c) conv_state(s, c0 + c) = src[c];
            }
        }
    });
}

void causal_conv1d_silu(ConstRows x, ConstRows weight, const StateRingView& ring, Rows out, ThreadPool* pool) {
    constexpr const char* kOp = "causal_conv1d_silu (ring)";
    const std::size_t n_tok = x.rows();
    const std::size_t n_ch = x.cols();
    const std::size_t k = weight.cols();
    HALO_CHECK(k >= 1, ErrorCode::Kernel, "{}: kernel size must be >= 1", kOp);
    HALO_CHECK(weight.rows() == n_ch, ErrorCode::Kernel, "{}: weight has {} rows, C = {}", kOp,
               weight.rows(), n_ch);
    detail::check_same_shape(x, out, kOp, "out");
    detail::check_alias_exact_or_disjoint(x, out, kOp, "x");
    detail::check_disjoint(weight, out, kOp, "weight");
    const std::size_t slot_n = (k - 1) * n_ch;  // floats per state
    HALO_CHECK(ring.P >= 2, ErrorCode::Kernel, "{}: state ring needs P >= 2, got {}", kOp, ring.P);
    HALO_CHECK(ring.live < ring.P, ErrorCode::Kernel, "{}: ring live {} outside P {}", kOp, ring.live, ring.P);
    HALO_CHECK(ring.slab.size() == detail::checked_mul(ring.P, slot_n, kOp), ErrorCode::Kernel,
               "{}: ring slab has {} floats, expected {} x (K-1)*C = {}", kOp, ring.slab.size(), ring.P, slot_n);
    HALO_CHECK(ring.n_slots <= ring.P - 1, ErrorCode::Kernel, "{}: {} ring slots requested, P - 1 = {}", kOp,
               ring.n_slots, ring.P - 1);
    {
        const ConstRows sl(ring.slab);
        for (const auto& [r, what] : {std::pair{x, "x"}, std::pair{weight, "weight"}, std::pair{ConstRows(out), "out"}}) {
            HALO_CHECK(!detail::overlaps(r, sl), ErrorCode::Kernel, "{}: ring slab overlaps {}", kOp, what);
        }
    }
    const std::size_t n_slots = std::min(n_tok, ring.n_slots);
    // Ring placement (ADR-001 §5.3): read slab[live]; final state -> slab[(live+1) mod P];
    // logical slot s -> slab[(live+1+s) mod P].
    const float* sin = ring.slab.data() + ring.live * slot_n;
    float* sout = ring.slab.data() + (ring.live + 1) % ring.P * slot_n;
    const auto slot_at = [&](std::size_t s) { return ring.slab.data() + (ring.live + 1 + s) % ring.P * slot_n; };

    // Channels are independent: split them into fixed blocks. Within a block, time runs
    // sequentially and the K-1 previous inputs live in a ring buffer, so `out` may alias
    // `x` (each input is copied into the ring before its output is written). Exactly the
    // in-place form's arithmetic; only the state placement differs.
    constexpr std::size_t kBlock = 64;
    const std::size_t hist = k - 1;
    const std::size_t n_blocks = (n_ch + kBlock - 1) / kBlock;
    parallel_for(pool, n_blocks, [&](std::size_t b0, std::size_t b1) {
        std::vector<float> wt(k * kBlock);       // wt[j * kBlock + c]: tap j, channel c
        std::vector<float> ringbuf(hist * kBlock);  // ringbuf[slot * kBlock + c]
        std::array<float, kBlock> acc{};
        for (std::size_t blk = b0; blk < b1; ++blk) {
            const std::size_t c0 = blk * kBlock;
            const std::size_t nc = std::min(kBlock, n_ch - c0);
            for (std::size_t c = 0; c < nc; ++c) {
                for (std::size_t j = 0; j < k; ++j) wt[j * kBlock + c] = weight(c0 + c, j);
            }
            for (std::size_t s = 0; s < hist; ++s) {
                for (std::size_t c = 0; c < nc; ++c) ringbuf[s * kBlock + c] = sin[s * n_ch + c0 + c];
            }
            std::size_t oldest = 0;  // ring slot holding the oldest input
            for (std::size_t t = 0; t < n_tok; ++t) {
                const float* xr = x.row_ptr(t) + c0;
                float* yr = out.row_ptr(t) + c0;
                for (std::size_t c = 0; c < nc; ++c) acc[c] = 0.0f;
                for (std::size_t j = 0; j < hist; ++j) {
                    const float* wj = &wt[j * kBlock];
                    const float* hj = &ringbuf[((oldest + j) % hist) * kBlock];
                    for (std::size_t c = 0; c < nc; ++c) acc[c] += wj[c] * hj[c];
                }
                const float* wl = &wt[hist * kBlock];
                for (std::size_t c = 0; c < nc; ++c) acc[c] += wl[c] * xr[c];
                if (hist > 0) {
                    float* slot = &ringbuf[oldest * kBlock];
                    for (std::size_t c = 0; c < nc; ++c) slot[c] = xr[c];  // before writing out
                    oldest = (oldest + 1) % hist;
                }
                for (std::size_t c = 0; c < nc; ++c) yr[c] = detail::silu(acc[c]);
                const std::size_t slot = n_tok - 1 - t;  // slot 0 = state after the last row
                if (slot < n_slots) {
                    float* dst = slot_at(slot);
                    for (std::size_t s = 0; s < hist; ++s) {
                        const float* src = &ringbuf[((oldest + s) % hist) * kBlock];
                        for (std::size_t c = 0; c < nc; ++c) dst[s * n_ch + c0 + c] = src[c];
                    }
                }
            }
            for (std::size_t s = 0; s < hist; ++s) {
                const float* src = &ringbuf[((oldest + s) % hist) * kBlock];
                for (std::size_t c = 0; c < nc; ++c) sout[s * n_ch + c0 + c] = src[c];
            }
        }
    });
}

}  // namespace halo::cpu
