#pragma once
// HALO CPU reference backend — weight matrix abstraction for y = W x (WS-D).
//
// A WeightMatrix is an N x K matrix (N output features, K input features, row n is the
// weights of output n — the GGUF / nn.Linear layout). Rows are either a float view, or
// produced on demand by a dequantization callback, so the CPU matmul never needs the
// whole matrix in float32. No GGUF knowledge lives here.

#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "halo/backends/cpu/views.h"
#include "halo/core/error.h"

namespace halo::cpu {

class WeightMatrix {
public:
    /// Writes rows [first_row, first_row + n_rows) as float32 into `out`
    /// (n_rows * cols elements, dense row-major). Must be safe to call concurrently from
    /// several threads with disjoint `out` buffers. May throw halo::Error.
    using RowDequantFn =
        std::function<void(std::size_t first_row, std::size_t n_rows, std::span<float> out)>;

    /// Non-owning view of float rows (N = w.rows(), K = w.cols()); w must outlive this.
    [[nodiscard]] static WeightMatrix dense(ConstRows w) {
        WeightMatrix m;
        m.dense_ = w;
        m.rows_ = w.rows();
        m.cols_ = w.cols();
        return m;
    }

    /// Rows materialized on demand by `fn`.
    [[nodiscard]] static WeightMatrix dequantized(std::size_t rows, std::size_t cols,
                                                  RowDequantFn fn) {
        HALO_CHECK(static_cast<bool>(fn), ErrorCode::Kernel, "WeightMatrix: empty dequant fn");
        WeightMatrix m;
        m.rows_ = rows;
        m.cols_ = cols;
        m.dequant_ = std::move(fn);
        return m;
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool is_dense() const noexcept { return !dequant_; }

    /// Rows [first, first + n) as a float view. Dense matrices return a view into the
    /// matrix; dequantized ones fill `scratch` (resized as needed) and view it.
    [[nodiscard]] ConstRows row_block(std::size_t first, std::size_t n,
                                      std::vector<float>& scratch) const {
        HALO_CHECK(first <= rows_ && n <= rows_ - first, ErrorCode::Kernel,
                   "WeightMatrix: rows [{}, {}) out of {}", first, first + n, rows_);
        if (!dequant_) {
            return {dense_.row_ptr(first), n, cols_, dense_.stride()};
        }
        scratch.resize(n * cols_);
        dequant_(first, n, std::span<float>(scratch));
        return {std::span<const float>(scratch), n, cols_};
    }

private:
    WeightMatrix() = default;
    ConstRows dense_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    RowDequantFn dequant_;
};

}  // namespace halo::cpu
