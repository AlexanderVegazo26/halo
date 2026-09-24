#pragma once
// HALO CPU reference operator set (WS-D; TRD §9 operator ids, DECISIONS.md D-003/D-004/D-016).
//
// This is the semantic definition of every HALO operator: GPU kernels are accepted only
// after differential tests against these functions (TRD §30). Correctness and clarity
// first; loops are written to auto-vectorize, and independent rows/heads run on a
// ThreadPool.
//
// Conventions shared by every operator
//   * float32 everywhere; no model or GGUF knowledge.
//   * Tensors are RowsView (row-major with an explicit row stride, in elements).
//     "token-major [T, H*D]" means row t holds H heads of D contiguous elements.
//   * Aliasing: an output may alias an input only where the contract says "may alias",
//     and then only *exactly* (same data pointer and stride). Any other overlap between
//     an output and another operand is rejected with halo::Error(Kernel).
//   * Shape mismatches throw halo::Error(ErrorCode::Kernel) before any output is written.
//   * `pool` may be null (single-threaded). Results are bit-identical for any pool size:
//     work is split only over independent outputs, and every output is produced by the
//     same fixed-order fp32 arithmetic (see ThreadPool). The CPU backend is compiled with
//     -ffp-contract=off so that HALO_NATIVE (FMA) cannot change reference results.
//   * Reductions (dot products, sums of squares) use 8 interleaved fp32 partial sums,
//     combined in a fixed pairwise order, plus a sequential tail. Fixed, but not the same
//     order as a naive loop or as torch — that is covered by the tested tolerances.

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "halo/backends/cpu/thread_pool.h"
#include "halo/backends/cpu/views.h"
#include "halo/backends/cpu/weight_matrix.h"

namespace halo::cpu {

// ---------------------------------------------------------------------------------------
// Operator registry entries (TRD §9): stable id + declared contract summary.
// ---------------------------------------------------------------------------------------
struct OpContract {
    std::string_view id;             // stable TRD §9 operator id
    std::string_view function;       // C++ entry point(s) in this header
    std::string_view state_effects;  // "none" or what persistent state is read/written
};

inline constexpr std::array<OpContract, 18> kOpContracts{{
    {"RMS_NORM", "rms_norm", "none"},
    {"GATED_NORM", "gated_rms_norm", "none"},
    {"L2_NORM", "l2_norm_heads", "none"},
    {"PARTIAL_ROPE", "partial_rope_neox", "none"},
    {"CONV1D_SHORT", "causal_conv1d_silu", "reads+writes conv_state [K-1, C]; optional per-row rollback slots"},
    {"GATED_DELTANET", "gated_delta_rule_recurrent / gated_delta_rule_chunked",
     "reads+writes recurrent state [n_v, d_k, d_v] fp32; optional per-row rollback slots"},
    {"ATTENTION", "attention_gqa", "reads K/V history (paged), writes none"},
    {"MUL_SIGMOID", "mul_sigmoid", "none"},
    {"SWIGLU", "swiglu", "none"},
    {"SILU", "silu", "none"},
    {"SIGMOID", "sigmoid", "none"},
    {"SOFTPLUS", "softplus", "none"},
    {"ADD", "add", "none"},
    {"MUL", "mul", "none"},
    {"SOFTMAX", "softmax_rows", "none"},
    {"MATMUL", "matmul", "none"},
    {"ARGMAX_FUSED", "argmax / matmul_argmax", "none"},
    {"TOP_K", "top_k", "none"},
}};

// ---------------------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------------------

/// RMS_NORM. out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, per row.
/// x, out: [R, D] (any row count, e.g. R = T or R = T*heads); w: [D].
/// Plain x̂·w: GGUF norm weights already contain the +1 of HF's zero-centred RMSNorm
/// (D-004). out may alias x.
void rms_norm(ConstRows x, std::span<const float> w, float eps, Rows out,
              ThreadPool* pool = nullptr);

/// GATED_NORM (Qwen3_5RMSNormGated). out[r] = (x̂[r] · w) * silu(z[r]) per row, where a
/// row is one value head (D = d_v). x, z, out: [R, D]; w: [D]. out may alias x or z.
void gated_rms_norm(ConstRows x, std::span<const float> w, ConstRows z, float eps, Rows out,
                    ThreadPool* pool = nullptr);

/// L2-normalize each head: y = x * rsqrt(sum(x^2) + eps) (transformers `l2norm`).
/// x, out: token-major [T, n_heads * head_dim]. out may alias x.
void l2_norm_heads(ConstRows x, std::size_t n_heads, std::size_t head_dim, Rows out,
                   float eps = 1e-6f, ThreadPool* pool = nullptr);

// ---------------------------------------------------------------------------------------
// Position encoding
// ---------------------------------------------------------------------------------------

/// PARTIAL_ROPE, NeoX (rotate-half) layout, in place on x: token-major
/// [T, n_heads * head_dim], positions: [T]. Only the first rot_dims of each head rotate
/// (pairs (i, i + rot_dims/2), i < rot_dims/2); the rest pass through. Angles are fp32,
/// computed exactly as transformers does: inv_freq[i] = 1 / theta^(float(2i)/rot_dims)
/// (fp32 pow + fp32 divide), angle = float(pos) * inv_freq[i] (one fp32 multiply);
/// cos/sin of that fp32 angle are then correctly rounded. rot_dims must be even and
/// <= head_dim.
void partial_rope_neox(Rows x, std::size_t n_heads, std::size_t head_dim,
                       std::span<const std::int32_t> positions, std::size_t rot_dims,
                       float theta, ThreadPool* pool = nullptr);

/// The fp32 inverse frequencies partial_rope_neox uses (rot_dims / 2 values).
[[nodiscard]] std::vector<float> rope_inv_freq(std::size_t rot_dims, float theta);

// ---------------------------------------------------------------------------------------
// GDN short convolution
// ---------------------------------------------------------------------------------------

/// CONV1D_SHORT: causal depthwise conv1d (no bias) followed by SiLU.
///   x:          [T, C]   input channels for T new tokens
///   weight:     [C, K]   tap j multiplies the input j - (K-1) steps back (tap K-1 = now)
///   conv_state: [K-1, C] in/out; the previous K-1 inputs, oldest row first. Zeros for a
///                        fresh sequence. On return holds the last K-1 inputs seen.
///   out:        [T, C]   may alias x.
/// out[t][c] = silu(sum_j weight[c][j] * xin[t - (K-1) + j][c]), where xin continues
/// conv_state. Processing T tokens at once is bit-identical to T calls with one token.
///
/// state_slots (optional, for speculative rollback): n_slots * (K-1) * C floats, slot s
/// laid out like conv_state. Slot s receives the conv state after row T-1-s (slot 0 = the
/// final state), for s < min(T, n_slots); slots s >= T are left untouched (caller-owned).
/// Must not overlap any other operand. (llama.cpp also writes slots s >= T, with the
/// pre-call state; HALO leaves them alone, matching its GDN kernel.)
void causal_conv1d_silu(ConstRows x, ConstRows weight, Rows conv_state, Rows out,
                        ThreadPool* pool = nullptr, std::span<float> state_slots = {});

// ---------------------------------------------------------------------------------------
// Gated DeltaNet (D-003, D-004 item 5-6)
// ---------------------------------------------------------------------------------------

/// How value head j selects its key/query head.
enum class GdnHeadMapping {
    Tiled,    // GGUF order (llama.cpp converter _reorder_v_heads): k head = j % n_k_heads
    Grouped,  // HF order (repeat_interleave): k head = j / (n_v_heads / n_k_heads)
};

struct GdnDims {
    std::size_t n_k_heads = 0;
    std::size_t n_v_heads = 0;  // multiple of n_k_heads
    std::size_t d_k = 0;
    std::size_t d_v = 0;
    GdnHeadMapping mapping = GdnHeadMapping::Tiled;
};

/// Key/query head used by value head j.
[[nodiscard]] std::size_t gdn_k_head(const GdnDims& dims, std::size_t v_head);

/// Per-token inputs. All token-major with T rows (row strides may differ, e.g. q/k/v can
/// be views into one fused conv-output row):
///   q, k: [T, n_k_heads * d_k]; v: [T, n_v_heads * d_v];
///   g:    [T, n_v_heads] log-decay (g = -exp(A_log) * softplus(a + dt_bias), <= 0);
///   beta: [T, n_v_heads] (already sigmoid-ed).
struct GdnInputs {
    ConstRows q;
    ConstRows k;
    ConstRows v;
    ConstRows g;
    ConstRows beta;
};

/// q/k preprocessing of GATED_DELTANET — the one input contract shared by every backend
/// (DECISIONS.md D-016; the Vulkan op's GdnDecodeArgs has the same two fields with the
/// same defaults). q and k are the raw per-head rows (after conv + SiLU). The kernel
/// applies exactly this and nothing else:
///   if qk_l2norm:  q <- q * rsqrt(sum(q^2) + 1e-6),  k <- k * rsqrt(sum(k^2) + 1e-6)
///   q <- q * q_scale   (after the optional normalization; k is never scaled)
/// q_scale = nullopt means 1/sqrt(d_k), the qwen35 value (D-004 item 3). Pass 1.0f for
/// "no scaling". The model forward should pass both fields explicitly:
/// GdnQkParams{.qk_l2norm = true, .q_scale = 1.0f / std::sqrt(128.0f)}.
struct GdnQkParams {
    bool qk_l2norm = true;
    std::optional<float> q_scale;  // nullopt = 1/sqrt(d_k)
};

/// The q_scale a GdnQkParams resolves to for head dim d_k.
[[nodiscard]] float gdn_q_scale(const GdnQkParams& qk, std::size_t d_k);

/// GATED_DELTANET, recurrent form (D-004 item 6), per value head j and token t:
///   q, k preprocessed per `qk` (GdnQkParams, D-016)
///   S <- S * exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S <- S + k delta^T;
///   o = S^T q.
/// state: [n_v_heads, d_k, d_v] fp32, d_v fastest (HF layout; NOTE ggml stores the
///        transpose, d_k fastest), in/out. Zero-fill for a fresh sequence.
/// out:   [T, n_v_heads * d_v]; must not overlap any input or the state.
/// Splitting a sequence into several calls is bit-identical to one call.
///
/// state_slots (optional; per-row state snapshots for MTP rollback, as llama.cpp's
/// ggml_gated_delta_net with K > 1): n_slots * (n_v_heads * d_k * d_v) floats, each slot
/// laid out like `state`. Slot s receives the state after row T-1-s (slot 0 = most recent
/// row = the final state) for s < min(T, n_slots); slots s >= T are left untouched
/// (caller-owned). Rolling back j rows = copying slot j into `state`. Slot s is
/// bit-identical to the final state of the same call run on rows [0, T-1-s] only.
/// Must not overlap any other operand.
void gated_delta_rule_recurrent(const GdnDims& dims, const GdnInputs& in,
                                std::span<float> state, Rows out, const GdnQkParams& qk = {},
                                ThreadPool* pool = nullptr, std::span<float> state_slots = {});

/// Source-compatible shorthand (pre-D-016 signature): GdnQkParams{qk_l2norm, 1/sqrt(d_k)}.
/// This is exactly what the old bool form computed, so existing callers keep their results.
/// A template on exactly `bool` so that a braced `{}` argument can only mean GdnQkParams{}
/// (a plain `bool` overload would win overload resolution for `{}` and mean false).
template <std::same_as<bool> Bool>
void gated_delta_rule_recurrent(const GdnDims& dims, const GdnInputs& in, std::span<float> state,
                                Rows out, Bool qk_l2norm, ThreadPool* pool = nullptr,
                                std::span<float> state_slots = {}) {
    gated_delta_rule_recurrent(dims, in, state, out, GdnQkParams{.qk_l2norm = qk_l2norm, .q_scale = std::nullopt}, pool,
                               state_slots);
}

/// GATED_DELTANET, chunked form: the same function as the recurrent form, evaluated
/// chunk_size tokens at a time exactly as transformers torch_chunk_gated_delta_rule
/// (UT transform by forward substitution, intra-chunk attention, one state update per
/// chunk). A final partial chunk is processed at its true length, which is exactly what
/// HF's zero padding computes (padded positions have k = v = beta = g = 0).
/// Numerically equivalent to the recurrent form within the tested tolerance (not
/// bitwise). Requires g <= 0 (HF contract; checked). Same state/out/state_slots contract;
/// a slot for a row inside a chunk is evaluated with the chunk's own closed form
/// S_a = S_prev * exp(gc_a) + sum_{b<=a} (k_b exp(gc_a - gc_b))^T v_new_b, so requesting
/// slots never changes `out` or `state`, and slot s is bit-identical to the final state of
/// a chunked call on rows [0, T-1-s] only (same chunk boundaries).
void gated_delta_rule_chunked(const GdnDims& dims, const GdnInputs& in, std::span<float> state,
                              Rows out, const GdnQkParams& qk = {}, std::size_t chunk_size = 64,
                              ThreadPool* pool = nullptr, std::span<float> state_slots = {});

/// Source-compatible shorthand (pre-D-016 signature): GdnQkParams{qk_l2norm, 1/sqrt(d_k)}.
template <std::same_as<bool> Bool>
void gated_delta_rule_chunked(const GdnDims& dims, const GdnInputs& in, std::span<float> state,
                              Rows out, Bool qk_l2norm, std::size_t chunk_size = 64,
                              ThreadPool* pool = nullptr, std::span<float> state_slots = {}) {
    gated_delta_rule_chunked(dims, in, state, out, GdnQkParams{.qk_l2norm = qk_l2norm, .q_scale = std::nullopt}, chunk_size,
                             pool, state_slots);
}

// ---------------------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------------------

struct AttentionDims {
    std::size_t n_head = 0;
    std::size_t n_kv_head = 0;  // n_head is a multiple; q head h uses kv head h / (n_head/n_kv)
    std::size_t head_dim = 0;
};

/// ATTENTION: causal GQA softmax attention over a (paged) K/V history.
///   q:    token-major [T, n_head * head_dim]
///   k, v: [S, n_kv_head * head_dim] (S = history length incl. the T new tokens)
///   q_offset: history index of query 0; query t attends keys 0 ..= q_offset + t.
///             Requires q_offset + T <= S.
///   out:  [T, n_head * head_dim]; must not overlap q (checked) or the K/V blocks (not
///         checked: a block table has no single address range).
/// scores = (q . k) * scale; p = exp(scores - max) / sum; out = sum_s p_s v_s — all fp32.
void attention_gqa(const AttentionDims& dims, ConstRows q, const PagedRows& k,
                   const PagedRows& v, std::size_t q_offset, float scale, Rows out,
                   ThreadPool* pool = nullptr);

// ---------------------------------------------------------------------------------------
// Element-wise (all operands the same [R, D] shape; out may alias any input)
// ---------------------------------------------------------------------------------------

/// out = x * sigmoid(gate)  (attention output gate, D-004).
void mul_sigmoid(ConstRows x, ConstRows gate, Rows out, ThreadPool* pool = nullptr);
/// out = silu(gate) * up.
void swiglu(ConstRows gate, ConstRows up, Rows out, ThreadPool* pool = nullptr);
/// silu(x) = x / (1 + exp(-x)).
void silu(ConstRows x, Rows out, ThreadPool* pool = nullptr);
/// sigmoid(x) = 1 / (1 + exp(-x)).
void sigmoid(ConstRows x, Rows out, ThreadPool* pool = nullptr);
/// torch softplus (beta 1, threshold 20): x > 20 ? x : log1p(exp(x)).
void softplus(ConstRows x, Rows out, ThreadPool* pool = nullptr);
void add(ConstRows a, ConstRows b, Rows out, ThreadPool* pool = nullptr);
void mul(ConstRows a, ConstRows b, Rows out, ThreadPool* pool = nullptr);
/// Row-wise stable softmax: exp(x - max) / sum. out may alias x.
void softmax_rows(ConstRows x, Rows out, ThreadPool* pool = nullptr);

// ---------------------------------------------------------------------------------------
// Matmul + logits reductions
// ---------------------------------------------------------------------------------------

/// MATMUL: y[T, N] = x[T, K] · Wᵀ (W is N x K). Parallel over blocks of W rows; each W
/// block is materialized once (dequant callback) and applied to every x row.
/// y must not overlap x.
void matmul(ConstRows x, const WeightMatrix& w, Rows y, ThreadPool* pool = nullptr);

struct TopKEntry {
    std::int32_t index = -1;
    float value = 0.0f;
    friend bool operator==(const TopKEntry&, const TopKEntry&) = default;
};

/// Index of the maximum; ties resolve to the lowest index. Throws Kernel on an empty
/// row or on any NaN (a NaN logit is a bug upstream and must not be sampled silently; every
/// backend raises Error(Kernel) on NaN, D-016).
[[nodiscard]] TopKEntry argmax(std::span<const float> logits);

/// The k largest entries, sorted by value descending, ties by lower index first.
/// Requires 1 <= k <= logits.size(); throws Kernel on NaN.
[[nodiscard]] std::vector<TopKEntry> top_k(std::span<const float> logits, std::size_t k);

/// ARGMAX_FUSED: argmax(x · Wᵀ) for one hidden row without materializing the logits
/// row. Identical result (index and value) to matmul followed by argmax.
[[nodiscard]] TopKEntry matmul_argmax(std::span<const float> x, const WeightMatrix& w,
                                      ThreadPool* pool = nullptr);

}  // namespace halo::cpu
