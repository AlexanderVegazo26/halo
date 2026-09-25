#pragma once
// Decode-step primitives: KV append into the halo::kv_cache pool layout, embedding row
// lookup (GET_ROWS) through wq_elem, and the fused residual add + RMSNorm. See hd.h for the
// execution model.

#include "conv_norm.h"
#include "gemv.h"
#include "hd.h"

namespace halo::hip::kern {

// ---- KV append ------------------------------------------------------------------------
// Writes n_tok K and V rows at history positions start .. start + n_tok - 1 of one layer,
// through the block table: exactly kv_cache::SequenceKv::write (memcpy of kv_dim floats to
// block table[s / block_tokens], row s % block_tokens). Pure copies (no arithmetic).

struct KvWriteParams {
    float* pool = nullptr;
    std::uint64_t block_floats = 0;
    std::uint64_t k_off = 0;  // (layer * 2 + 0) * block_tokens * kv_dim
    std::uint64_t v_off = 0;  // (layer * 2 + 1) * block_tokens * kv_dim
    unsigned block_tokens = 0;
    unsigned kv_dim = 0;
    unsigned n_pool_blocks = 0;
    const std::uint32_t* table = nullptr;
    unsigned start = 0;
    const float* k = nullptr;
    std::uint64_t k_stride = 0;  // elements between source rows
    const float* v = nullptr;
    std::uint64_t v_stride = 0;
    std::uint32_t* status = nullptr;
};

struct DecNoRegs {
    float unused;
};

inline Launch kv_write_launch(unsigned n_tok, unsigned kv_dim, unsigned block) {
    return Launch{n_tok, (kv_dim + block - 1) / block, block};
}

template <class Exec>
HALO_HD void kv_write_body(Exec& ex, DecNoRegs&, const KvWriteParams& p, unsigned bx, unsigned by) {
    ex.phase([&](unsigned tid, DecNoRegs&) {
        const unsigned c = by * ex.block_dim() + tid;
        if (c >= p.kv_dim) return;
        const unsigned s = p.start + bx;
        const std::uint32_t id = p.table[s / p.block_tokens];
        if (id >= p.n_pool_blocks) {
            ex.atomic_or(p.status, kStatusBadBlock);
            return;
        }
        const std::uint64_t row = static_cast<std::uint64_t>(id) * p.block_floats +
                                  static_cast<std::uint64_t>(s % p.block_tokens) * p.kv_dim + c;
        p.pool[row + p.k_off] = p.k[static_cast<std::uint64_t>(bx) * p.k_stride + c];
        p.pool[row + p.v_off] = p.v[static_cast<std::uint64_t>(bx) * p.v_stride + c];
    });
}

// ---- GET_ROWS (embedding lookup) ----------------------------------------------------------
// out[t][c] = dequantize_row(W[ids[t]])[c] — wq_elem is bit-identical to tensor::dequantize_row.

struct GetRowsParams {
    WType type = WType::F32;
    const std::uint8_t* w = nullptr;
    std::uint64_t w_stride = 0;  // bytes between table rows
    unsigned n_rows = 0;         // vocabulary
    unsigned cols = 0;
    const std::int32_t* ids = nullptr;
    float* out = nullptr;
    std::uint64_t out_stride = 0;
    std::uint32_t* status = nullptr;
};

inline Launch get_rows_launch(unsigned n_ids, unsigned cols, unsigned block) {
    return Launch{n_ids, (cols + block - 1) / block, block};
}

template <class Exec>
HALO_HD void get_rows_body(Exec& ex, DecNoRegs&, const GetRowsParams& p, unsigned bx, unsigned by) {
    ex.phase([&](unsigned tid, DecNoRegs&) {
        const unsigned c = by * ex.block_dim() + tid;
        if (c >= p.cols) return;
        const std::int32_t id = p.ids[bx];
        if (id < 0 || static_cast<unsigned>(id) >= p.n_rows) {
            ex.atomic_or(p.status, kStatusBadIndex);
            return;
        }
        const std::uint8_t* row = p.w + static_cast<std::uint64_t>(id) * p.w_stride;
        p.out[static_cast<std::uint64_t>(bx) * p.out_stride + c] = wq_elem(p.type, row, c);
    });
}

// ---- fused residual add + RMS_NORM -----------------------------------------------------
// h = a + b (written to h, which may alias a or b exactly), then y = rms_norm(h) * w with the
// CPU's 8-lane order: exactly cpu::add followed by cpu::rms_norm. The norm phases read h from
// memory after the barrier that ends the add phase (workgroup-visible), so the CPU's two-op
// composition and this kernel see the same rounded h.

struct AddNormParams {
    const float* a = nullptr;
    std::uint64_t a_stride = 0;
    const float* b = nullptr;
    std::uint64_t b_stride = 0;
    float* h = nullptr;
    std::uint64_t h_stride = 0;
    const float* w = nullptr;
    float* y = nullptr;
    std::uint64_t y_stride = 0;
    unsigned rows = 0;
    unsigned cols = 0;
    float eps = 0.0f;
};

inline Launch add_norm_launch(unsigned rows, unsigned block) { return Launch{rows, 1, block}; }

template <class Exec>
HALO_HD void add_norm_body(Exec& ex, NormShared& sh, const AddNormParams& p, unsigned bx, unsigned by) {
    const float* ar = p.a + static_cast<std::uint64_t>(bx) * p.a_stride;
    const float* br = p.b + static_cast<std::uint64_t>(bx) * p.b_stride;
    float* hr = p.h + static_cast<std::uint64_t>(bx) * p.h_stride;
    ex.phase([&](unsigned tid, NormRegs&) {
        for (unsigned i = tid; i < p.cols; i += ex.block_dim()) hr[i] = ar[i] + br[i];
    });
    NormParams np;
    np.x = p.h;
    np.x_stride = p.h_stride;
    np.w = p.w;
    np.out = p.y;
    np.out_stride = p.y_stride;
    np.rows = p.rows;
    np.cols = p.cols;
    np.eps = p.eps;
    norm_body(ex, sh, np, bx, by);
}

}  // namespace halo::hip::kern
