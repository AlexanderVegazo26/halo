#pragma once
// CONV1D_SHORT (causal depthwise conv1d + SiLU, D-004 item 2, D-012 slots) and
// RMS_NORM / GATED_NORM (D-004: plain x̂·w; ssm_norm with the silu(z) gate) kernel bodies.
// Same arithmetic, operation by operation, as backends/cpu/conv1d.cpp and
// backends/cpu/ops_basic.cpp. See hd.h for the execution model.

#include "hd.h"

namespace halo::hip::kern {

// =======================================================================================
// CONV1D_SHORT: grid ceil(C / block), one thread per channel, sequential over tokens.
// =======================================================================================

inline constexpr unsigned kConvMaxK = 8;

struct ConvParams {
    const float* x = nullptr;
    std::uint64_t x_stride = 0;
    const float* w = nullptr;       // [C, K]
    std::uint64_t w_stride = 0;
    float* state = nullptr;         // [K-1, C] in/out
    std::uint64_t state_stride = 0;
    float* out = nullptr;           // may equal x (exact alias)
    std::uint64_t out_stride = 0;
    float* slots = nullptr;         // n_slots * (K-1) * C, slot rows dense (stride C)
    unsigned n_slots = 0;
    unsigned n_tok = 0;
    unsigned channels = 0;
    unsigned k = 0;
};

struct ConvRegs {
    float w[kConvMaxK];
    float h[kConvMaxK - 1];  // previous inputs, oldest first
};
struct ConvShared {
    float unused;
};

inline Launch conv_launch(unsigned channels, unsigned block) {
    return Launch{(channels + block - 1) / block, 1, block};
}

template <class Exec>
HALO_HD void conv1d_silu_body(Exec& ex, ConvShared&, const ConvParams& p, unsigned bx, unsigned) {
    ex.phase([&](unsigned tid, ConvRegs& r) {
        const unsigned c = bx * ex.block_dim() + tid;
        if (c >= p.channels) return;
        const unsigned hist = p.k - 1u;
        const std::uint64_t slot_n = static_cast<std::uint64_t>(hist) * p.channels;
#pragma unroll
        for (unsigned j = 0; j < kConvMaxK; ++j) {
            if (j < p.k) r.w[j] = p.w[c * p.w_stride + j];
        }
#pragma unroll
        for (unsigned j = 0; j < kConvMaxK - 1; ++j) {
            if (j < hist) r.h[j] = p.state[j * p.state_stride + c];
        }
        for (unsigned t = 0; t < p.n_tok; ++t) {
            const float xv = p.x[t * p.x_stride + c];
            float acc = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < kConvMaxK - 1; ++j) {
                if (j < hist) acc = acc + r.w[j] * r.h[j];
            }
#pragma unroll
            for (unsigned j = 0; j < kConvMaxK; ++j) {
                if (j == hist) acc = acc + r.w[j] * xv;
            }
            // Shift the history (before the output is written: out may alias x).
#pragma unroll
            for (unsigned j = 0; j < kConvMaxK - 1; ++j) {
                if (j + 1u < hist) r.h[j] = r.h[j + 1];
                else if (j + 1u == hist) r.h[j] = xv;
            }
            p.out[t * p.out_stride + c] = siluf(acc);
            const unsigned slot = p.n_tok - 1u - t;
            if (slot < p.n_slots) {
                float* dst = p.slots + slot * slot_n + c;
#pragma unroll
                for (unsigned j = 0; j < kConvMaxK - 1; ++j) {
                    if (j < hist) dst[static_cast<std::uint64_t>(j) * p.channels] = r.h[j];
                }
            }
        }
#pragma unroll
        for (unsigned j = 0; j < kConvMaxK - 1; ++j) {
            if (j < hist) p.state[j * p.state_stride + c] = r.h[j];
        }
    });
}

// =======================================================================================
// RMS_NORM / GATED_NORM: one workgroup per row. Sum of squares with the CPU's 8-lane
// fixed order (threads 0..7 = lanes), combined by thread 0; then element-wise.
//   plain: y = (x * inv) * w                    (cpu::rms_norm)
//   gated: y = (w * (x * inv)) * silu(z)        (cpu::gated_rms_norm)
// =======================================================================================

inline constexpr unsigned kNormMinBlock = 8;

struct NormParams {
    const float* x = nullptr;
    std::uint64_t x_stride = 0;
    const float* z = nullptr;       // null = plain RMS_NORM
    std::uint64_t z_stride = 0;
    const float* w = nullptr;
    float* out = nullptr;           // may equal x or z (exact alias)
    std::uint64_t out_stride = 0;
    unsigned rows = 0;
    unsigned cols = 0;
    float eps = 0.0f;
};

struct NormRegs {
    float unused;
};
struct NormShared {
    float part[8];
    float inv;
};

inline Launch norm_launch(unsigned rows, unsigned block) { return Launch{rows, 1, block}; }

template <class Exec>
HALO_HD void norm_body(Exec& ex, NormShared& sh, const NormParams& p, unsigned bx, unsigned) {
    const float* xr = p.x + static_cast<std::uint64_t>(bx) * p.x_stride;
    const At xa{xr};
    ex.phase([&](unsigned tid, NormRegs&) {
        if (tid < 8u) sh.part[tid] = dot8_lane(xa, xa, p.cols, tid);
    });
    ex.phase([&](unsigned tid, NormRegs&) {
        if (tid != 0) return;
        const float mean = dot8_combine(sh.part, xa, xa, p.cols) / static_cast<float>(p.cols);
        sh.inv = 1.0f / hsqrt(mean + p.eps);
    });
    ex.phase([&](unsigned tid, NormRegs&) {
        float* yr = p.out + static_cast<std::uint64_t>(bx) * p.out_stride;
        const float inv = sh.inv;
        if (p.z == nullptr) {
            for (unsigned i = tid; i < p.cols; i += ex.block_dim()) yr[i] = (xr[i] * inv) * p.w[i];
        } else {
            const float* zr = p.z + static_cast<std::uint64_t>(bx) * p.z_stride;
            for (unsigned i = tid; i < p.cols; i += ex.block_dim()) {
                yr[i] = (p.w[i] * (xr[i] * inv)) * siluf(zr[i]);
            }
        }
    });
}

}  // namespace halo::hip::kern
