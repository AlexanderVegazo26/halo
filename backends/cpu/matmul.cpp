// MATMUL, ARGMAX_FUSED, TOP_K.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "kernel_common.h"

namespace halo::cpu {

namespace {

// W rows per task: one materialized block is reused for every x row.
constexpr std::size_t kRowBlock = 16;

/// Strict total order used by argmax/top_k: larger value first, then lower index.
[[nodiscard]] bool better(float va, std::int32_t ia, float vb, std::int32_t ib) noexcept {
    return va > vb || (va == vb && ia < ib);
}

void check_no_nan(std::span<const float> row, const char* op) {
    for (std::size_t i = 0; i < row.size(); ++i) {
        HALO_CHECK(!std::isnan(row[i]), ErrorCode::Kernel, "{}: NaN at index {}", op, i);
    }
}

void check_index_range(std::size_t n, const char* op) {
    HALO_CHECK(n <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()), ErrorCode::Kernel,
               "{}: {} entries exceed int32 indices", op, n);
}

}  // namespace

void matmul(ConstRows x, const WeightMatrix& w, Rows y, ThreadPool* pool) {
    constexpr const char* kOp = "matmul";
    HALO_CHECK(x.cols() == w.cols(), ErrorCode::Kernel, "{}: x has {} cols, W is {}x{}", kOp, x.cols(), w.rows(),
               w.cols());
    HALO_CHECK(y.rows() == x.rows() && y.cols() == w.rows(), ErrorCode::Kernel, "{}: y is {}x{}, expected {}x{}",
               kOp, y.rows(), y.cols(), x.rows(), w.rows());
    detail::check_disjoint(x, y, kOp, "x");
    const std::size_t n_blocks = (w.rows() + kRowBlock - 1) / kRowBlock;
    const std::size_t kdim = x.cols();
    parallel_for(pool, n_blocks, [&](std::size_t b0, std::size_t b1) {
        std::vector<float> scratch;
        for (std::size_t blk = b0; blk < b1; ++blk) {
            const std::size_t n0 = blk * kRowBlock;
            const std::size_t nn = std::min(kRowBlock, w.rows() - n0);
            const ConstRows wb = w.row_block(n0, nn, scratch);
            for (std::size_t t = 0; t < x.rows(); ++t) {
                const float* xt = x.row_ptr(t);
                float* yt = y.row_ptr(t) + n0;
                for (std::size_t n = 0; n < nn; ++n) yt[n] = detail::dot(xt, wb.row_ptr(n), kdim);
            }
        }
    });
}

TopKEntry argmax(std::span<const float> logits) {
    HALO_CHECK(!logits.empty(), ErrorCode::Kernel, "argmax: empty row");
    check_index_range(logits.size(), "argmax");
    // Single pass: the NaN check and the max-finding scan used to be two separate O(n) passes.
    HALO_CHECK(!std::isnan(logits[0]), ErrorCode::Kernel, "argmax: NaN at index {}", 0);
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
        HALO_CHECK(!std::isnan(logits[i]), ErrorCode::Kernel, "argmax: NaN at index {}", i);
        if (logits[i] > logits[best]) best = i;  // strict: ties keep the lower index
    }
    return {static_cast<std::int32_t>(best), logits[best]};
}

std::vector<TopKEntry> top_k(std::span<const float> logits, std::size_t k) {
    HALO_CHECK(k >= 1 && k <= logits.size(), ErrorCode::Kernel, "top_k: k = {} for {} entries", k, logits.size());
    check_index_range(logits.size(), "top_k");
    check_no_nan(logits, "top_k");
    std::vector<std::int32_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    const auto kk = static_cast<std::ptrdiff_t>(k);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](std::int32_t a, std::int32_t b) {
        return better(logits[static_cast<std::size_t>(a)], a, logits[static_cast<std::size_t>(b)], b);
    });
    std::vector<TopKEntry> res(k);
    for (std::size_t i = 0; i < k; ++i) res[i] = {idx[i], logits[static_cast<std::size_t>(idx[i])]};
    return res;
}

TopKEntry matmul_argmax(std::span<const float> x, const WeightMatrix& w, ThreadPool* pool) {
    constexpr const char* kOp = "matmul_argmax";
    HALO_CHECK(x.size() == w.cols(), ErrorCode::Kernel, "{}: x has {} elements, W is {}x{}", kOp, x.size(),
               w.rows(), w.cols());
    HALO_CHECK(w.rows() > 0, ErrorCode::Kernel, "{}: W has no rows", kOp);
    check_index_range(w.rows(), kOp);
    const std::size_t n_blocks = (w.rows() + kRowBlock - 1) / kRowBlock;
    std::vector<TopKEntry> block_best(n_blocks);
    std::vector<unsigned char> block_nan(n_blocks, 0);
    parallel_for(pool, n_blocks, [&](std::size_t b0, std::size_t b1) {
        std::vector<float> scratch;
        for (std::size_t blk = b0; blk < b1; ++blk) {
            const std::size_t n0 = blk * kRowBlock;
            const std::size_t nn = std::min(kRowBlock, w.rows() - n0);
            const ConstRows wb = w.row_block(n0, nn, scratch);
            TopKEntry best;
            for (std::size_t n = 0; n < nn; ++n) {
                const float v = detail::dot(x.data(), wb.row_ptr(n), x.size());
                const auto idx = static_cast<std::int32_t>(n0 + n);
                if (std::isnan(v)) block_nan[blk] = 1;
                if (best.index < 0 || better(v, idx, best.value, best.index)) best = {idx, v};
            }
            block_best[blk] = best;
        }
    });
    TopKEntry best = block_best[0];
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        HALO_CHECK(block_nan[blk] == 0, ErrorCode::Kernel, "{}: NaN logit in rows [{}, {})", kOp, blk * kRowBlock,
                   std::min(w.rows(), (blk + 1) * kRowBlock));
        if (better(block_best[blk].value, block_best[blk].index, best.value, best.index)) best = block_best[blk];
    }
    return best;
}

}  // namespace halo::cpu
