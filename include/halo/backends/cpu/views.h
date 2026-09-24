#pragma once
// HALO CPU reference backend — non-owning float32 views (WS-D).
//
// RowsView<T>: a 2-D row-major view with an explicit row stride (in elements), so an
// operator can read e.g. the q slice of a fused qkv row, or every head row of an
// interleaved [q | gate] projection, without copies.
//
// PagedRows: a read-only row view whose rows may live in separate fixed-size blocks
// (a paged KV block table). A contiguous buffer is the one-block special case.

#include <cstddef>
#include <span>
#include <type_traits>

#include "halo/core/error.h"

namespace halo::cpu {

template <class T>
class RowsView {
    static_assert(std::is_same_v<std::remove_const_t<T>, float>, "CPU reference views are float32");

public:
    constexpr RowsView() noexcept = default;

    /// rows x cols elements, row r starting at data + r * stride. Requires stride >= cols
    /// (when rows > 1) and a non-null data pointer when the view is non-empty.
    RowsView(T* data, std::size_t rows, std::size_t cols, std::size_t stride)
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {
        HALO_CHECK(rows <= 1 || stride >= cols, ErrorCode::Kernel,
                   "RowsView: stride {} < cols {}", stride, cols);
        HALO_CHECK(data != nullptr || rows == 0 || cols == 0, ErrorCode::Kernel,
                   "RowsView: null data for a {}x{} view", rows, cols);
    }

    /// Dense rows x cols view over a span of exactly rows * cols elements.
    RowsView(std::span<T> s, std::size_t rows, std::size_t cols)
        : RowsView(s.data(), rows, cols, cols) {
        HALO_CHECK(cols == 0 || rows <= s.size() / cols, ErrorCode::Kernel,
                   "RowsView: {}x{} does not fit span of {}", rows, cols, s.size());
        HALO_CHECK(rows * cols == s.size(), ErrorCode::Kernel,
                   "RowsView: span of {} is not {}x{}", s.size(), rows, cols);
    }

    /// A single row covering the whole span (implicit, for 1-D element-wise use).
    RowsView(std::span<T> s)  // NOLINT(google-explicit-constructor)
        : data_(s.data()), rows_(s.empty() ? 0 : 1), cols_(s.size()), stride_(s.size()) {}

    /// Mutable -> const conversion.
    template <class U>
        requires(std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>)
    RowsView(const RowsView<U>& o) noexcept  // NOLINT(google-explicit-constructor)
        : data_(o.data()), rows_(o.rows()), cols_(o.cols()), stride_(o.stride()) {}

    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
    [[nodiscard]] bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    /// Pointer to row r (unchecked).
    [[nodiscard]] T* row_ptr(std::size_t r) const noexcept { return data_ + r * stride_; }
    /// Row r as a span (unchecked).
    [[nodiscard]] std::span<T> row(std::size_t r) const noexcept { return {row_ptr(r), cols_}; }
    [[nodiscard]] T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * stride_ + c];
    }

    /// Address one past the last element touched by the view (for overlap checks).
    [[nodiscard]] const float* end_ptr() const noexcept {
        return empty() ? data_ : data_ + (rows_ - 1) * stride_ + cols_;
    }

private:
    T* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t stride_ = 0;
};

using Rows = RowsView<float>;
using ConstRows = RowsView<const float>;

/// Read-only rows spread over fixed-size blocks: row i lives at
/// blocks[i / block_rows] + (i % block_rows) * row_stride.
class PagedRows {
public:
    PagedRows() = default;

    /// One contiguous buffer: row i at data + i * row_stride.
    static PagedRows contiguous(const float* data, std::size_t rows, std::size_t cols,
                                std::size_t row_stride) {
        HALO_CHECK(rows <= 1 || row_stride >= cols, ErrorCode::Kernel,
                   "PagedRows: stride {} < cols {}", row_stride, cols);
        HALO_CHECK(data != nullptr || rows == 0, ErrorCode::Kernel, "PagedRows: null data");
        PagedRows p;
        p.base_ = data;
        p.rows_ = rows;
        p.cols_ = cols;
        p.stride_ = row_stride;
        p.block_rows_ = rows == 0 ? 1 : rows;
        return p;
    }

    /// Block table. `blocks` must outlive the view; every block needed for `rows` rows
    /// must be non-null and hold block_rows rows of stride row_stride.
    static PagedRows paged(std::span<const float* const> blocks, std::size_t block_rows,
                           std::size_t rows, std::size_t cols, std::size_t row_stride) {
        HALO_CHECK(block_rows > 0, ErrorCode::Kernel, "PagedRows: block_rows == 0");
        HALO_CHECK(block_rows <= 1 || row_stride >= cols, ErrorCode::Kernel,
                   "PagedRows: stride {} < cols {}", row_stride, cols);
        const std::size_t needed = (rows + block_rows - 1) / block_rows;
        HALO_CHECK(blocks.size() >= needed, ErrorCode::Kernel,
                   "PagedRows: {} rows need {} blocks of {}, table has {}", rows, needed,
                   block_rows, blocks.size());
        for (std::size_t b = 0; b < needed; ++b) {
            HALO_CHECK(blocks[b] != nullptr, ErrorCode::Kernel, "PagedRows: block {} is null", b);
        }
        PagedRows p;
        p.blocks_ = blocks;
        p.rows_ = rows;
        p.cols_ = cols;
        p.stride_ = row_stride;
        p.block_rows_ = block_rows;
        return p;
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

    /// Pointer to row i (unchecked).
    [[nodiscard]] const float* row_ptr(std::size_t i) const noexcept {
        if (blocks_.empty()) return base_ + i * stride_;
        return blocks_[i / block_rows_] + (i % block_rows_) * stride_;
    }

private:
    const float* base_ = nullptr;
    std::span<const float* const> blocks_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t stride_ = 0;
    std::size_t block_rows_ = 1;
};

}  // namespace halo::cpu
