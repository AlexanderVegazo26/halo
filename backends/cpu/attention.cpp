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

}  // namespace halo::cpu
