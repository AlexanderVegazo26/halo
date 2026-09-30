// ATTENTION: causal GQA softmax attention over a (paged) K/V history.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "kernel_common.h"

namespace halo::cpu {

void attention_gqa(const AttentionDims& dims, ConstRows q, const PagedRows& k, const PagedRows& v,
                   std::size_t q_offset, float scale, Rows out, ThreadPool* pool) {
    constexpr const char* kOp = "attention_gqa";
    const std::size_t hd = dims.head_dim;
    HALO_CHECK(dims.n_head > 0 && dims.n_kv_head > 0 && hd > 0 && dims.n_head % dims.n_kv_head == 0,
               ErrorCode::Kernel, "{}: bad dims n_head {} n_kv_head {} head_dim {}", kOp, dims.n_head,
               dims.n_kv_head, hd);
    const std::size_t q_cols = detail::checked_mul(dims.n_head, hd, kOp);
    const std::size_t kv_cols = detail::checked_mul(dims.n_kv_head, hd, kOp);
    const std::size_t n_tok = q.rows();
    HALO_CHECK(q.cols() == q_cols, ErrorCode::Kernel, "{}: q has {} cols, expected {}", kOp, q.cols(), q_cols);
    HALO_CHECK(out.rows() == n_tok && out.cols() == q_cols, ErrorCode::Kernel, "{}: out is {}x{}, expected {}x{}",
               kOp, out.rows(), out.cols(), n_tok, q_cols);
    HALO_CHECK(k.cols() == kv_cols && v.cols() == kv_cols, ErrorCode::Kernel,
               "{}: k/v have {}/{} cols, expected {}", kOp, k.cols(), v.cols(), kv_cols);
    HALO_CHECK(k.rows() == v.rows(), ErrorCode::Kernel, "{}: k has {} rows, v has {}", kOp, k.rows(), v.rows());
    HALO_CHECK(q_offset <= k.rows() && n_tok <= k.rows() - q_offset, ErrorCode::Kernel,
               "{}: q_offset {} + T {} exceeds history {}", kOp, q_offset, n_tok, k.rows());
    detail::check_disjoint(q, out, kOp, "q");

    const std::size_t group = dims.n_head / dims.n_kv_head;
    const std::size_t n_items = n_tok * dims.n_head;
    parallel_for(pool, n_items, [&](std::size_t i0, std::size_t i1) {
        std::vector<float> p(q_offset + n_tok);
        for (std::size_t item = i0; item < i1; ++item) {
            const std::size_t t = item / dims.n_head;
            const std::size_t h = item % dims.n_head;
            const std::size_t kvh = h / group;
            const std::size_t n_keys = q_offset + t + 1;
            const float* qh = q.row_ptr(t) + h * hd;
            float m = -std::numeric_limits<float>::infinity();
            for (std::size_t s = 0; s < n_keys; ++s) {
                p[s] = detail::dot(qh, k.row_ptr(s) + kvh * hd, hd) * scale;
                m = std::max(m, p[s]);
            }
            float sum = 0.0f;
            for (std::size_t s = 0; s < n_keys; ++s) {
                p[s] = std::exp(p[s] - m);
                sum += p[s];
            }
            for (std::size_t s = 0; s < n_keys; ++s) p[s] = p[s] / sum;
            float* o = out.row_ptr(t) + h * hd;
            std::fill(o, o + hd, 0.0f);
            for (std::size_t s = 0; s < n_keys; ++s) {
                const float ps = p[s];
                const float* vs = v.row_ptr(s) + kvh * hd;
                for (std::size_t d = 0; d < hd; ++d) o[d] += ps * vs[d];
            }
        }
    });
}

// ---- tree attention -----------------------------------------------------------------------

void check_tree_parents(std::span<const std::int32_t> parents) {
    constexpr const char* kOp = "attention_gqa_tree";
    HALO_CHECK(!parents.empty() && parents.size() <= kMaxTreeRows, ErrorCode::Kernel, "{}: {} tree rows (1..{})", kOp,
               parents.size(), kMaxTreeRows);
    HALO_CHECK(parents[0] == -1, ErrorCode::Kernel, "{}: row 0 must be the root (parent -1), got {}", kOp, parents[0]);
    for (std::size_t i = 1; i < parents.size(); ++i) {
        HALO_CHECK(parents[i] >= 0 && static_cast<std::size_t>(parents[i]) < i, ErrorCode::Kernel,
                   "{}: row {} has parent {} (must be in [0, {}))", kOp, i, parents[i], i);
    }
}

std::size_t tree_depth(std::span<const std::int32_t> parents, std::size_t row) {
    std::size_t d = 0;
    for (std::int32_t p = parents[row]; p >= 0; p = parents[static_cast<std::size_t>(p)]) ++d;
    return d;
}

std::vector<std::size_t> tree_path(std::span<const std::int32_t> parents, std::size_t row) {
    std::vector<std::size_t> path(tree_depth(parents, row) + 1);
    std::size_t r = row;
    for (std::size_t i = path.size(); i-- > 0;) {
        path[i] = r;
        if (parents[r] >= 0) r = static_cast<std::size_t>(parents[r]);
    }
    return path;
}

bool is_chain_plus_root_leaf(std::span<const std::int32_t> parents) noexcept {
    const std::size_t n = parents.size();
    if (n < 3 || n > kMaxTreeRows || parents[0] != -1) return false;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (parents[i] != static_cast<std::int32_t>(i - 1)) return false;
    }
    return parents[n - 1] == 0;
}

std::vector<std::int32_t> chain_plus_root_leaf_parents(std::size_t n_rows) {
    HALO_CHECK(n_rows >= 3 && n_rows <= kMaxTreeRows, ErrorCode::Kernel, "chain_plus_root_leaf_parents: {} rows (3..{})",
               n_rows, kMaxTreeRows);
    std::vector<std::int32_t> p(n_rows);
    p[0] = -1;
    for (std::size_t i = 1; i + 1 < n_rows; ++i) p[i] = static_cast<std::int32_t>(i - 1);
    p[n_rows - 1] = 0;
    return p;
}

void attention_gqa_tree(const AttentionDims& dims, ConstRows q, const PagedRows& k, const PagedRows& v,
                        std::size_t q_offset, float scale, std::span<const std::int32_t> parents, Rows out,
                        ThreadPool* pool) {
    constexpr const char* kOp = "attention_gqa_tree";
    const std::size_t hd = dims.head_dim;
    HALO_CHECK(dims.n_head > 0 && dims.n_kv_head > 0 && hd > 0 && dims.n_head % dims.n_kv_head == 0,
               ErrorCode::Kernel, "{}: bad dims n_head {} n_kv_head {} head_dim {}", kOp, dims.n_head,
               dims.n_kv_head, hd);
    const std::size_t q_cols = detail::checked_mul(dims.n_head, hd, kOp);
    const std::size_t kv_cols = detail::checked_mul(dims.n_kv_head, hd, kOp);
    const std::size_t n_tok = q.rows();
    check_tree_parents(parents);
    HALO_CHECK(parents.size() == n_tok, ErrorCode::Kernel, "{}: {} parents for {} query rows", kOp, parents.size(), n_tok);
    HALO_CHECK(q.cols() == q_cols, ErrorCode::Kernel, "{}: q has {} cols, expected {}", kOp, q.cols(), q_cols);
    HALO_CHECK(out.rows() == n_tok && out.cols() == q_cols, ErrorCode::Kernel, "{}: out is {}x{}, expected {}x{}",
               kOp, out.rows(), out.cols(), n_tok, q_cols);
    HALO_CHECK(k.cols() == kv_cols && v.cols() == kv_cols, ErrorCode::Kernel,
               "{}: k/v have {}/{} cols, expected {}", kOp, k.cols(), v.cols(), kv_cols);
    HALO_CHECK(k.rows() == v.rows(), ErrorCode::Kernel, "{}: k has {} rows, v has {}", kOp, k.rows(), v.rows());
    HALO_CHECK(q_offset <= k.rows() && n_tok <= k.rows() - q_offset, ErrorCode::Kernel,
               "{}: q_offset {} + T {} exceeds history {}", kOp, q_offset, n_tok, k.rows());
    detail::check_disjoint(q, out, kOp, "q");

    const std::size_t group = dims.n_head / dims.n_kv_head;
    const std::size_t n_items = n_tok * dims.n_head;
    parallel_for(pool, n_items, [&](std::size_t i0, std::size_t i1) {
        std::vector<float> p(q_offset + n_tok);
        std::vector<std::size_t> phys(q_offset + n_tok);  // virtual key -> physical history row
        for (std::size_t item = i0; item < i1; ++item) {
            const std::size_t t = item / dims.n_head;
            const std::size_t h = item % dims.n_head;
            const std::size_t kvh = h / group;
            const std::vector<std::size_t> path = tree_path(parents, t);
            const std::size_t n_keys = q_offset + path.size();
            for (std::size_t s = 0; s < q_offset; ++s) phys[s] = s;
            for (std::size_t j = 0; j < path.size(); ++j) phys[q_offset + j] = q_offset + path[j];
            const float* qh = q.row_ptr(t) + h * hd;
            float m = -std::numeric_limits<float>::infinity();
            for (std::size_t s = 0; s < n_keys; ++s) {
                p[s] = detail::dot(qh, k.row_ptr(phys[s]) + kvh * hd, hd) * scale;
                m = std::max(m, p[s]);
            }
            float sum = 0.0f;
            for (std::size_t s = 0; s < n_keys; ++s) {
                p[s] = std::exp(p[s] - m);
                sum += p[s];
            }
            for (std::size_t s = 0; s < n_keys; ++s) p[s] = p[s] / sum;
            float* o = out.row_ptr(t) + h * hd;
            std::fill(o, o + hd, 0.0f);
            for (std::size_t s = 0; s < n_keys; ++s) {
                const float ps = p[s];
                const float* vs = v.row_ptr(phys[s]) + kvh * hd;
                for (std::size_t d = 0; d < hd; ++d) o[d] += ps * vs[d];
            }
        }
    });
}

}  // namespace halo::cpu
