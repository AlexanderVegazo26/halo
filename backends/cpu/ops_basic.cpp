// Norms, RoPE, element-wise ops and softmax of the CPU reference backend.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "halo/backends/cpu/ops.h"
#include "kernel_common.h"

namespace halo::cpu {

using detail::check_alias_exact_or_disjoint;
using detail::check_same_shape;

// --- normalization ---------------------------------------------------------------------

void rms_norm(ConstRows x, std::span<const float> w, float eps, Rows out, ThreadPool* pool) {
    check_same_shape(x, out, "rms_norm", "out");
    HALO_CHECK(w.size() == x.cols(), ErrorCode::Kernel, "rms_norm: weight has {} elements, D = {}",
               w.size(), x.cols());
    check_alias_exact_or_disjoint(x, out, "rms_norm", "x");
    const std::size_t d = x.cols();
    parallel_for(pool, x.rows(), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* xr = x.row_ptr(r);
            float* yr = out.row_ptr(r);
            const float mean = detail::sum_squares(xr, d) / static_cast<float>(d);
            const float inv = 1.0f / std::sqrt(mean + eps);
            for (std::size_t i = 0; i < d; ++i) yr[i] = (xr[i] * inv) * w[i];
        }
    });
}

void gated_rms_norm(ConstRows x, std::span<const float> w, ConstRows z, float eps, Rows out,
                    ThreadPool* pool) {
    check_same_shape(x, out, "gated_rms_norm", "out");
    check_same_shape(x, z, "gated_rms_norm", "z");
    HALO_CHECK(w.size() == x.cols(), ErrorCode::Kernel,
               "gated_rms_norm: weight has {} elements, D = {}", w.size(), x.cols());
    check_alias_exact_or_disjoint(x, out, "gated_rms_norm", "x");
    check_alias_exact_or_disjoint(z, out, "gated_rms_norm", "z");
    const std::size_t d = x.cols();
    parallel_for(pool, x.rows(), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* xr = x.row_ptr(r);
            const float* zr = z.row_ptr(r);
            float* yr = out.row_ptr(r);
            const float mean = detail::sum_squares(xr, d) / static_cast<float>(d);
            const float inv = 1.0f / std::sqrt(mean + eps);
            // HF order: weight * (x * rsqrt) then * silu(gate).
            for (std::size_t i = 0; i < d; ++i) yr[i] = (w[i] * (xr[i] * inv)) * detail::silu(zr[i]);
        }
    });
}

void l2_norm_heads(ConstRows x, std::size_t n_heads, std::size_t head_dim, Rows out, float eps,
                   ThreadPool* pool) {
    HALO_CHECK(detail::checked_mul(n_heads, head_dim, "l2_norm_heads") == x.cols(), ErrorCode::Kernel,
               "l2_norm_heads: {} heads x {} != {} columns", n_heads, head_dim, x.cols());
    check_same_shape(x, out, "l2_norm_heads", "out");
    check_alias_exact_or_disjoint(x, out, "l2_norm_heads", "x");
    parallel_for(pool, x.rows(), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            for (std::size_t h = 0; h < n_heads; ++h) {
                const float* xh = x.row_ptr(r) + h * head_dim;
                float* yh = out.row_ptr(r) + h * head_dim;
                const float inv = 1.0f / std::sqrt(detail::sum_squares(xh, head_dim) + eps);
                for (std::size_t i = 0; i < head_dim; ++i) yh[i] = xh[i] * inv;
            }
        }
    });
}

// --- RoPE ------------------------------------------------------------------------------

std::vector<float> rope_inv_freq(std::size_t rot_dims, float theta) {
    HALO_CHECK(rot_dims > 0 && rot_dims % 2 == 0, ErrorCode::Kernel,
               "rope: rot_dims {} must be even and > 0", rot_dims);
    std::vector<float> inv(rot_dims / 2);
    for (std::size_t i = 0; i < inv.size(); ++i) {
        // transformers: 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), all fp32.
        const float e = static_cast<float>(2 * i) / static_cast<float>(rot_dims);
        inv[i] = 1.0f / std::pow(theta, e);
    }
    return inv;
}

void partial_rope_neox(Rows x, std::size_t n_heads, std::size_t head_dim,
                       std::span<const std::int32_t> positions, std::size_t rot_dims, float theta,
                       ThreadPool* pool) {
    HALO_CHECK(detail::checked_mul(n_heads, head_dim, "partial_rope_neox") == x.cols(),
               ErrorCode::Kernel, "partial_rope_neox: {} heads x {} != {} columns", n_heads, head_dim,
               x.cols());
    HALO_CHECK(positions.size() == x.rows(), ErrorCode::Kernel,
               "partial_rope_neox: {} positions for {} tokens", positions.size(), x.rows());
    HALO_CHECK(rot_dims <= head_dim, ErrorCode::Kernel, "partial_rope_neox: rot_dims {} > head_dim {}",
               rot_dims, head_dim);
    const std::vector<float> inv = rope_inv_freq(rot_dims, theta);
    const std::size_t half = rot_dims / 2;
    parallel_for(pool, x.rows(), [&](std::size_t t0, std::size_t t1) {
        std::vector<float> cs(half);
        std::vector<float> sn(half);
        for (std::size_t t = t0; t < t1; ++t) {
            const float pos = static_cast<float>(positions[t]);
            for (std::size_t i = 0; i < half; ++i) {
                const float angle = pos * inv[i];  // one fp32 multiply, as HF's fp32 matmul
                cs[i] = static_cast<float>(std::cos(static_cast<double>(angle)));
                sn[i] = static_cast<float>(std::sin(static_cast<double>(angle)));
            }
            for (std::size_t h = 0; h < n_heads; ++h) {
                float* xh = x.row_ptr(t) + h * head_dim;
                for (std::size_t i = 0; i < half; ++i) {
                    const float x1 = xh[i];
                    const float x2 = xh[i + half];
                    // HF: x*cos + rotate_half(x)*sin, rotate_half = cat(-x2, x1).
                    xh[i] = x1 * cs[i] + (-x2) * sn[i];
                    xh[i + half] = x2 * cs[i] + x1 * sn[i];
                }
            }
        }
    });
}

// --- element-wise ----------------------------------------------------------------------

namespace {

template <class F>
void unary(ConstRows x, Rows out, ThreadPool* pool, const char* op, F f) {
    check_same_shape(x, out, op, "out");
    check_alias_exact_or_disjoint(x, out, op, "x");
    const std::size_t d = x.cols();
    parallel_for(pool, x.rows(), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* xr = x.row_ptr(r);
            float* yr = out.row_ptr(r);
            for (std::size_t i = 0; i < d; ++i) yr[i] = f(xr[i]);
        }
    });
}

template <class F>
void binary(ConstRows a, ConstRows b, Rows out, ThreadPool* pool, const char* op, F f) {
    check_same_shape(a, b, op, "second operand");
    check_same_shape(a, out, op, "out");
    check_alias_exact_or_disjoint(a, out, op, "first operand");
    check_alias_exact_or_disjoint(b, out, op, "second operand");
    const std::size_t d = a.cols();
    parallel_for(pool, a.rows(), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* ar = a.row_ptr(r);
            const float* br = b.row_ptr(r);
            float* yr = out.row_ptr(r);
            for (std::size_t i = 0; i < d; ++i) yr[i] = f(ar[i], br[i]);
        }
    });
}

}  // namespace

void mul_sigmoid(ConstRows x, ConstRows gate, Rows out, ThreadPool* pool) {
    binary(x, gate, out, pool, "mul_sigmoid", [](float a, float g) { return a * detail::sigmoid(g); });
}
void swiglu(ConstRows gate, ConstRows up, Rows out, ThreadPool* pool) {
    binary(gate, up, out, pool, "swiglu", [](float g, float u) { return detail::silu(g) * u; });
}
void silu(ConstRows x, Rows out, ThreadPool* pool) {
    unary(x, out, pool, "silu", [](float v) { return detail::silu(v); });
}
void sigmoid(ConstRows x, Rows out, ThreadPool* pool) {
    unary(x, out, pool, "sigmoid", [](float v) { return detail::sigmoid(v); });
}
void softplus(ConstRows x, Rows out, ThreadPool* pool) {
    unary(x, out, pool, "softplus", [](float v) { return detail::softplus(v); });
}
void add(ConstRows a, ConstRows b, Rows out, ThreadPool* pool) {
    binary(a, b, out, pool, "add", [](float u, float v) { return u + v; });
}
void mul(ConstRows a, ConstRows b, Rows out, ThreadPool* pool) {
    binary(a, b, out, pool, "mul", [](float u, float v) { return u * v; });
}

void softmax_rows(ConstRows x, Rows out, ThreadPool* pool) {
    check_same_shape(x, out, "softmax_rows", "out");
    check_alias_exact_or_disjoint(x, out, "softmax_rows", "x");
    const std::size_t d = x.cols();
    parallel_for(pool, x.rows(), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* xr = x.row_ptr(r);
            float* yr = out.row_ptr(r);
            float m = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < d; ++i) m = std::max(m, xr[i]);
            float sum = 0.0f;
            for (std::size_t i = 0; i < d; ++i) {
                yr[i] = std::exp(xr[i] - m);
                sum += yr[i];
            }
            for (std::size_t i = 0; i < d; ++i) yr[i] = yr[i] / sum;
        }
    });
}

}  // namespace halo::cpu
