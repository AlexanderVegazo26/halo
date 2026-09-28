# HALO Vulkan backend (WS-F / WS-F2)

Status: a **correctness-first building-block set**, verified on lavapipe only (D-001). It covers RMS_NORM (incl. per-head q/k norm), the GEMV (decode and batched T > 1) for every UD weight type, the GATED_DELTANET recurrent decode (MTP-verify rows, rollback slots), ARGMAX and (V1, 2026-09-25) CONV1D+SiLU, GATED_NORM, PARTIAL_ROPE with head_stride, SWIGLU, MUL_SIGMOID, ADD, ADD+RMS_NORM, the fused GDN gates and (V2, 2026-09-26) GET_ROWS, KV write and paged GQA ATTENTION, and (V3) the batched GEMV, IQ4_NL / Q3_K / IQ3_S, the LM head (logits + argmax) and TOP_K, and (V4) the chunked GATED_DELTANET: the MVP op set of ADR-001 BI-5. It is **not yet a qwen35 forward**; "What the full forward still needs" below lists the missing operators.

Code: `backends/vulkan/`, `include/halo/backends/vulkan/`, tests in `tests/unit/vulkan/`.

## What was verified where (read this first)

The development host has no AMD GPU (DECISIONS D-001). Every Vulkan claim here was
established on **Mesa lavapipe** (CPU Vulkan 1.3, minStorageBufferOffsetAlignment 16)
under WSL2.

| Claim | Status |
|---|---|
| Shaders compile to SPIR-V and pass `spirv-val` (vulkan1.3) | **Verified** (every build) |
| Kernel results agree with `halo::cpu` / `halo::tensor` on the same inputs, within stated a-priori bounds | **Verified on lavapipe** |
| Host validation (shapes, views, overflow, byte-range aliasing) | **Verified** (device-independent host code, unit-tested) |
| Correct on RADV / gfx1151 (the target) | **Unverified.** Not run. Differences from lavapipe that could matter: subgroup sizes (unused), fp16 denormal handling in `unpackHalf2x16` (test weights use normal fp16 scales on purpose), `exp` / `inversesqrt` precision (the bounds assume the Vulkan spec's ULP limits, not lavapipe's), and a larger minStorageBufferOffsetAlignment |
| Correct on AMDVLK | **Unverified.** The ICD policy prefers RADV and warns about AMDVLK |
| Any performance, bandwidth or occupancy number | **Unmeasured.** lavapipe timings printed by tests are labelled "not a GPU performance number". The kernels use one workgroup per row or head with shared-memory tree reductions, and no tuning was done |

The tests stay green on a machine without a Vulkan device, because they skip with a
reason (`HALO_VK_CONTEXT_OR_SKIP`). On the dev host they run; they did not skip in any
reported run.

## How kernels are accepted

A kernel is accepted only by a **differential test against `halo::cpu` and
`halo::tensor::dequantize_row` on the same host arrays and weight bytes** (TRD §30, code
review M-2). Each assertion has the form:

    |vk − cpu| ≤ (bound_vk + bound_cpu) · scale

Here each bound is that backend's a-priori forward-error bound against exact arithmetic,
and fp64 is used only to compute `scale` (the Σ|terms| magnitudes). The bounds are
documented at the top of `test_vk_ops.cpp` and `test_vk_differential.cpp`.

Every new check was shown to fail on a targeted mutation (a shader, host or CPU change),
one mutation at a time. Exceptions, for the V1 layer ops:
- **`precise` (NoContraction) is not demonstrated.** Removing every `precise` from
  `rope_neox.comp` or `halo_math.glsl` leaves the tests green on lavapipe, because lavapipe did
  not fuse multiply-adds even when allowed. The bit-identical claims are verified on lavapipe
  only; whether RADV honours `precise` as required is unverified.
- **Equivalent mutant:** dropping the stride comparison from the exact-alias check
  (`same_range`) is not detectable, because an output and its input always have the same row
  count and row size, so equal first and last bytes already imply equal strides.

Exceptions for the V2 ops (GET_ROWS, KV write, ATTENTION):
- **`precise` again.** The bit-identity of GET_ROWS to `tensor::dequantize_row` also rests on
  `precise`, with the same lavapipe caveat.
- **Equivalent mutant:** the `id < 0` test in GET_ROWS is redundant, because `uint(id)` of a
  negative id is at least 2^31 and therefore always ≥ n_rows.
- **Not mutated:** turning a bounds check from `id >= n` into `id > n` for a block id or row id.
  The mutant kernel would read or write one block past the bound, which is undefined behaviour on
  the device rather than a clean failure. The tests do use the boundary values: a block id equal
  to n_pool_blocks, and a row id equal to n_rows.
- **Shared dequantization wiring.** A single mutation in each `common/dequant_<type>.glsl`
  (q8_0, q4_k, q5_k, q6_k, iq4_xs) turns both the GET_ROWS test and the matching `VkMatvec`
  tests red, which shows that one piece of code serves both ops.

For the V3 ops (batched GEMV, IQ4_NL / Q3_K / IQ3_S, LM head, TOP_K), every targeted mutation
went red. That includes the dequantization of the three new types (red in both `VkMatvec` and
GET_ROWS), the batched x / y strides and chunk offsets, the LM head's per-vector logits and
result offsets, and the TOP_K tie order, pad id, NaN flag, later-pass strides, chunk output
offsets and bitonic direction. Two mutations were not run:
- **Sharing the argmax scratch across the LM head's vectors.** This is harmless, because the
  vectors are serialised by barriers, so it is equivalent rather than a defect.
- **Dropping the 8-vector cap.** The kernel would index past its per-thread array, which is
  undefined behaviour on the device.

For the V4 chunked GATED_DELTANET, every targeted mutation went red. These cover:
- the head mapping, the cumulative gc, the forward substitution, the `exp(gc)` in k_cumdecay,
  the intra-chunk output term and its diagonal, and the state decay;
- the slot index, later chunks reading the input state, q_scale and k's L2 norm;
- the g pre-pass and the main kernel honouring it, an undersized per-head workspace, and the
  chunk-size limit.

Measured detection floor: a per-chunk bias of 1e-4 or 1e-3 in the state decay is caught by
the shape and slot tests. The fast-decay T = 4096 prefill does not catch it, because the decay
forgets the bias. The bound is loose: observed differences are at most 4.6e-3 of it.

**Caveat — the bounds are loose.** Observed |vk − cpu| is typically 1e-4 to 4e-2 of the
bound. The long-trajectory GDN test therefore adds a second, random-walk check at 2× CPU
Model G (see S-5 below). A small systematic drift passes the worst-case bound and fails
that check. The random-walk check is a model, not a proof.

## Operator inventory

| Op (TRD §9) | Vulkan | Shader(s) | Semantics | Tests |
|---|---|---|---|---|
| RMS_NORM | `Ops::rms_norm` | `norm/rms_norm` | `(x · rsqrt(mean x² + eps)) · w`, plain x̂·w (**D-004**). Same operation order as `cpu::rms_norm`. No in-place | `VkOps.RmsNormMatchesCpu`, `VkViews.RmsNormStridedViewsMatchCpuAndDense` |
| MATMUL / QUANT_GEMV (decode and batched, V3) | `Ops::matvec`, `Ops::gemv` (`GemvArgs`, n_vec rows) | `matmul/matvec_{f32,q8_0,q4_k,q5_k,q6_k,iq4_xs,iq4_nl,q3_k,iq3_s}` (body `common/matvec_quant_main.glsl`, dequantization `common/dequant_<type>.glsl`; q4_k/q5_k/q6_k/q3_k/iq4_xs/iq4_nl/iq3_s have their own llama-style SWAR bodies `common/matvec_<type>_main.glsl`, 16 (iq4_xs/iq4_nl/iq3_s: 8) threads/block at WG=64; only f32/q8_0 keep `common/matvec_quant_main.glsl`) | ggml block layouts, dequantization as `tensor::dequantize_row`, every type of the UD-Q4_K_XL pack (**D-007**, **D-014**). One workgroup per W row dequantizes each element once and applies it to up to 8 x vectors per dispatch (n_vec > 8 is split); a vector's result is bitwise the same as its own n_vec = 1 matvec | `VkMatvec.*` (9 types × shapes / workgroups / realistic scales), `VkHead.GemvBatched*` (9 types × n_vec 1, 3, 8, 11, 17, strided x / y), 2-D grid past 65 535 rows, `VkViews.Arena*` |
| GATED_DELTANET (recurrent; T = 1 decode, T > 1 MTP verify) | `Ops::gated_delta_rule_decode` | `linear_attn/gated_delta_rule_decode` | **D-004** item 6. Tiled head mapping `j % n_k`. q/k contract **D-016**: raw q/k, in-kernel per-head L2 (eps 1e-6) when `qk_l2norm`, then `q *= q_scale`. Rollback slots **D-012**. In place or disjoint state_out | `VkDiffGdn.*`, `VkGdn.*` |
| GATED_DELTANET chunked (prefill, V4) | `Ops::gated_delta_rule_chunked` (`GdnChunkedArgs`, `gdn_chunked_workspace_bytes`) | `linear_attn/gated_delta_rule_chunked`, `linear_attn/gdn_gcheck` | `cpu::gated_delta_rule_chunked` (HF torch_chunk_gated_delta_rule): same operands and D-016 / D-012 contract as the decode op, chunk_size 1..64, the CPU op's operation order step by step (UT forward substitution, intra-chunk attention, one state update per chunk; slots by the chunk's closed form). One workgroup per value head, chunks in order, per-head scratch in `workspace`. A g that is not <= 0 (NaN included) sets `k_status_positive_g` in a pre-pass and the op writes nothing. Bounded vs CPU (2 x Model G random-walk model), observed 9e-6 to 4.6e-3 of it | `VkChunkedGdn.*` |
| ARGMAX (building block of ARGMAX_FUSED) | `Ops::argmax` + `read_argmax` | `reduce/argmax_{partial,final}` | Lowest index on ties, -inf ordinary. **NaN → Error(Kernel)** through a NaN word in the 12-byte result (**D-016**, review S-3) | `VkArgmax.*`, `VkViews.ArgmaxOnSubViews*` |
| LM head (V3) | `Ops::lm_head` (`LmHeadArgs`) | gemv + `reduce/argmax_*` | `cpu::matmul` then `cpu::argmax` per vector (= `cpu::matmul_argmax`), composed on the device (no host round trip; not a fused kernel). Logits optional (`gemv.y`, else workspace). NaN → the vector's NaN word | `VkHead.LmHead*` |
| TOP_K (V3) | `Ops::top_k` (`TopKArgs`, `topk_workspace_bytes`) | `sample/topk` | `cpu::top_k` per vector: value descending, ties by lower index. Chunked bitonic selection (2048-entry chunks in 16 KiB shared memory, best k kept, repeated until one chunk per vector) → **bit-identical** ids and values. k ≤ 1024. NaN sets `k_status_nan` | `VkHead.TopK*` |
| CONV1D_SHORT (V1) | `Ops::causal_conv1d_silu` (`Conv1dArgs`) | `linear_attn/conv1d_silu` | `cpu::causal_conv1d_silu`: K ≤ 8, conv state in/out, rollback slots (**D-012**), out may alias x. Pre-activation accumulated in the CPU order, uncontracted → **bit-identical**; SiLU bounded. State and slots **bit-identical** | `VkLayer.Conv1d*` |
| GATED_NORM (V1) | `Ops::gated_rms_norm` (`GatedNormArgs`) | `norm/gated_rms_norm` | `cpu::gated_rms_norm`, HF order `(w·(x·inv))·silu(z)`; out may alias x or z | `VkLayer.GatedRmsNormMatchesCpu` |
| Per-head q/k RMS_NORM (V1) | `Ops::rms_norm` with a strided view | `norm/rms_norm` | Q halves read in place from the interleaved `[Q \| gate]` rows (rows = T·n_head, stride 512) | `VkLayer.PerHeadQkRmsNorm*` |
| PARTIAL_ROPE (V1) | `Ops::partial_rope_neox` (`RopeArgs`, `rope_cos_sin_table`) | `rope/rope_neox` | `cpu::partial_rope_neox` in place with **head_stride** (TD-9): Q rotated inside `[Q \| gate]` (head_dim 256, stride 512), gate halves untouched. cos/sin table built on the host exactly as the CPU op (Vulkan float sin/cos is only 2^-11 absolute) → **bit-identical** | `VkLayer.PartialRope*`, `VkLayer.RopeTable*` |
| SWIGLU, MUL_SIGMOID, ADD (V1) | `Ops::swiglu` / `mul_sigmoid` / `add` (`EltwiseArgs`) | `eltwise/eltwise` (spec constant) | `cpu::swiglu`, `cpu::mul_sigmoid` (gate may be the strided half of `[Q \| gate]`), `cpu::add` (**bit-identical**); out may alias a or b (the in-place residual h ≡ a, ADR-001 §5.2) | `VkLayer.SwigluMulSigmoidAdd*`, `VkLayer.EltwiseAliasingRules` |
| ADD + RMS_NORM (V1) | `Ops::add_rms_norm` (`AddRmsNormArgs`) | `norm/add_rms_norm` | = `cpu::add` then `cpu::rms_norm` (ADR-001 §5.4): h **bit-identical**, y bounded vs CPU and **bit-identical** to `Ops::rms_norm(h)`; h may alias a or b | `VkLayer.AddRmsNorm*` |
| GDN gates (V1) | `Ops::gdn_gates` (`GdnGateArgs`) | `eltwise/gdn_gates` | qwen35.cpp's sequence: beta = sigmoid(b); g = ssm_a · softplus(a + dt_bias), torch threshold 20, accurate log1p (series for e ≤ 0.5, because Vulkan's log is only 2^-21 absolute near 1); beta may alias b, g may alias a | `VkLayer.GdnGates*` |
| GET_ROWS (V2) | `Ops::get_rows` (`GetRowsArgs`) | `get_rows/get_rows_{f32,q8_0,q4_k,q5_k,q6_k,iq4_xs}` | `tensor::dequantize_row` per id, for every GEMV weight type. The element dequantization lives in `common/dequant_<type>.glsl`, shared with `matvec_<type>` (same code, so the GEMV and the lookup cannot drift), evaluated uncontracted → **bit-identical**. An id outside [0, n_rows) sets `k_status_bad_index` and leaves that row unwritten | `VkRows.*` |
| KV write (V2) | `Ops::kv_write` (`KvWriteArgs`) | `attention/kv_write` | `kv_cache::SequenceKv::write` into the pool layout `block[layer][K\|V][token][kv_dim]` through the block table → **byte-identical pool** (other layers and blocks untouched). A table entry ≥ n_pool_blocks sets `k_status_bad_block`; that row is not written | `VkKv.KvWrite*`, `VkKv.PoolLayoutIsTheDocumentedOne` |
| ATTENTION (V2) | `Ops::attention` (`AttentionArgs`) | `attention/attention` | `cpu::attention_gqa` over the paged pool: causal, GQA head h → h / (n_head / n_kv_head), q read in place with `q_head_stride` (512 for `[Q \| gate]`). Online softmax over key tiles of `attention_workgroup` keys; scores use the CPU's 8-lane dot order. Bounded vs CPU (derivation at the top of `test_vk_kv.cpp`), observed 0.0002–0.033 of the bound (loose: the score term assumes the dot may be contracted). Bad table entries are skipped (never read) and set `k_status_bad_block`. head_dim ≤ 256 and ≤ 4 × attention_workgroup | `VkKv.Attention*`, `VkKv.KvWriteThenAttention*` |

Activation bounds (V1 ops with exp/log): `|vk − cpu| ≤ f·u·|cpu| + FLT_MIN`, f derived from
Vulkan's precision rules (exp 3 + 2|x| ULP, division 2.5 ULP) plus the CPU's libm; the
per-op derivation is at the top of `test_vk_layer.cpp`. Observed: 0.26–0.42 of the bound
(conv, SwiGLU, sigmoid, gates), 0.10 (gated norm), 0.006 (add+rms_norm y).

Common to all ops:
- **Operands are `BufferView`s** (review S-1): a byte range of a Buffer with a row stride.
  - Each view is bound at its offset rounded down to the device alignment, with only its own extent, so views deep inside an arena past maxStorageBufferRange work (tested).
  - The remainder and the stride go to the shader as push constants.
  - fp32 views need 4-byte alignment. Quantized weights may start at any byte.
  - A fused qkv row is read as q / k / v views without copies (tested).
- **Aliasing** is a byte-range overlap check within one VkBuffer. Exceptions: the GDN state (input and output regions may be identical), and the V1 ops that say "may alias exactly" (an output naming exactly the same elements as one input, e.g. the residual h ≡ a); a partial overlap is always rejected.
- **Transfers:** `Buffer::upload` / `download` on non-host-visible memory go through the context's single reused staging buffer. It is bounded by `ContextOptions::staging_bytes` (default 16 MiB), and larger transfers are chunked (review N-1).
- **Determinism:** fixed-order reductions give identical results run to run and independent of where an operand lives. Tests check the view vs dense-copy results bitwise.

### GATED_DELTANET specifics
- **State layout** is `[n_v, d_k, d_v]` fp32, d_v fastest (HF layout, the same as the CPU op; ggml stores the transpose).
- **Chunked form (V4).** `Ops::gated_delta_rule_chunked` evaluates prefill 64 tokens at a time (T = 4096 tested against `cpu::gated_delta_rule_chunked`, both decay bands). It parallelises over value heads only, like the CPU op, so it is correct but not a tuned GPU prefill. The decode op still accepts T > 1 (MTP verify) and runs it serially.
- **Limits:** `d_k ≤ gdn_max_dk` (default 256; shared memory). `gdn_workgroup` can be any size (32, 96 and 100 tested).
- **S-5 evidence (lavapipe):**
  - **Long trajectory.** Heads (1 key / 2 value) at d_k = d_v = 128. One 4096-row dispatch, then 64 single-row dispatches in place, compared with CPU chunked prefill plus 64 recurrent steps.
    - Worst |vk − cpu| is 1.4e-6 / 1.6e-5 (fast / slow decay) of the linear bound, and 6.3e-5 / 7.0e-4 of the 2×-Model-G random-walk check.
    - The CPU-side test (4096 chunked + 64 recurrent vs fp64, same dims, Model and WeakDecay regimes) reaches ≤ 1.6e-3 of Model G.
  - **Detection floor (measured by mutation).**
    - A per-token decay bias of 1e-5 in the Vulkan shader is **not** detected (random-walk ratio 0.10). A bias of 1e-4 is detected, barely (ratio 1.002). The linear bound alone misses both (0.002 / 0.022).
    - On the CPU side, a 1e-4 decay bias in the recurrent decode is detected. In the chunked prefill, a 1e-4 per-chunk bias is not detected, and 1e-3 per chunk is (ratio 1.8 / 4.1).
    - The tolerances were fixed before measuring and were not tightened to fit the observations.
  - **In-place multi-row with slots.** T = 5 / K = 4, T = 3 / K = 6 and real dims T = 4 / K = 4. The state is in place at an odd arena offset, with slots in the same arena. It is compared with `cpu::gated_delta_rule_recurrent` with slots. Slots at or beyond T stay untouched, and slot 0 is bitwise the final state.

## What the full qwen35-on-Vulkan forward still needs

Per D-004, the layer ops, the existing Vulkan pieces and what is missing:

| Needed op | Where in the model | Vulkan status | CPU reference to diff against |
|---|---|---|---|
| **GQA attention over the paged KV cache** (24 Q / 4 KV heads, head_dim 256, scale 1/√256, causal) + KV write | full-attention layers (every 4th) | **Done (V2)**: `Ops::kv_write` + `Ops::attention` over the kv_cache pool layout. Limit: the pool is one descriptor (see Known limits) | `cpu::attention_gqa` with `PagedRows`, `SequenceKv::write` |
| **Causal depthwise conv1d + SiLU** (kernel 4, conv state of 3 rows, rollback slots per D-012) | GDN layers, over the 10240 qkv channels | **Done (V1).** The ADR-001 §5.3 state ring (out-of-place final state) is in with WS-BI-2 stage 1: `Conv1dArgs::ring` | `cpu::causal_conv1d_silu` |
| **Gated RMSNorm** `(x̂·w)·silu(z)` per value head (dim 128, `ssm_norm`) | GDN output | **Done (V1)** | `cpu::gated_rms_norm` |
| **Partial NeoX RoPE with a head stride** (first 64 of 256 dims, theta 1e7) on the interleaved `[Q, gate]` rows of `attn_q` (per head 256 Q then 256 gate) | full-attention Q and K | **Done (V1)**, in place with head_stride. The CPU reference is applied per head through a strided `Rows` view (bit-identical) | `cpu::partial_rope_neox` |
| **Per-head RMSNorm on Q and K** (`attn_q_norm`, `attn_k_norm`, dim 256) | full-attention layers | **Done (V1)**: rms_norm over a strided view, tested on the interleaved rows | `cpu::rms_norm` |
| **SwiGLU** `silu(gate) · up` | FFN (5120 → 17408 → 5120) | **Done (V1)** | `cpu::swiglu` |
| **Sigmoid output gate** `attn · sigmoid(gate)` | full-attention output | **Done (V1)** (gate read strided from `[Q \| gate]`) | `cpu::mul_sigmoid` |
| **L2 norm of q/k** | GDN | **Done in-kernel** (D-016) | — |
| **beta = sigmoid(b), g = ssm_a · softplus(a + dt_bias)** | GDN gates | **Done (V1)**, fused | `cpu::sigmoid`, `cpu::softplus` (qwen35.cpp sequence) |
| Residual adds, ADD + RMS_NORM | every layer | **Done (V1)** (h ≡ a in place) | `cpu::add`, `cpu::rms_norm` |
| **Batched / prefill matmul** (T > 1 rows) | prefill, MTP verify | **Done (V3)**: `Ops::gemv` reads W once per 8 rows (correctness-first; not a tiled GEMM) | `cpu::matmul` |
| **Chunked GATED_DELTANET** | prefill | **Done (V4)**: `Ops::gated_delta_rule_chunked` | `cpu::gated_delta_rule_chunked` |
| **IQ4_NL, Q3_K, IQ3_S matvec** | remaining UD types (D-014 priority list) | **Done (V3)**, GEMV and GET_ROWS | `tensor::dequantize_row` |
| **LM head: matvec + argmax, TOP_K** | head | **Done (V3)**: `Ops::lm_head` (gemv + argmax on the device), `Ops::top_k` | `cpu::matmul_argmax`, `cpu::top_k` |
| Token embedding lookup (`token_embd`, quantized rows) | input | **Done (V2)**: `Ops::get_rows` | `tensor::dequantize_row` |

After all of those, the forward still needs engine integration: model wiring, per-layer
buffer planning in the GPU pool (D-002), GDN checkpoint copies (D-013), and the RADV runs
that turn every "verified on lavapipe" above into a claim about the target.

## The Vulkan backend (halo::backend, V5)

`make_vulkan_backend(ctx)` (`src/backend/vulkan_adapter.h`, library
`halo_backend_vulkan_adapter`) implements `halo::backend::Backend` (ADR-001) over the ops
above, so the one `models::Qwen35` forward runs on Vulkan. The library is separate from
`halo_backend`, so nothing else links Vulkan. Its CMake target is created at the end of
configuration, because `backends/` is configured after `src/`.

- **Interface rules.** The adapter enforces the interface's neutral aliasing rule even where
  a Vulkan kernel accepts more. It also checks buffer ownership, logical ranges and alignment,
  and accepts KernelChoice 0 only. A non-empty StatusRef is `Error(Unsupported)`, as on the
  CPU backend (see Known limits).
- **Memory.** `allocate` returns zero-filled device memory. `import_host_readonly` makes a
  device copy at import time. `import_host` is a device mirror: the caller's bytes are copied
  in at import time and written back after every successful `Stream::wait()`. `abort()` never
  writes back. Uploads and downloads are staged copies recorded in stream order, and a download
  captures the buffer at its position in the stream.
- **Device data errors.** A bad row id or block id, a positive g, or a NaN in TOP_K is written
  to stream-owned status words and raised as `Error(Kernel)` by `Stream::wait()`. In that case
  wait() skips the import write-back, so the host keeps its pre-step state. A NaN logit in the
  LM head or ARGMAX arrives through the result's NaN word, and `decode_argmax` raises.
- **RoPE.** The cos/sin table is built on the host from the positions. The positions must
  therefore be a read-only host import, which is what the forward passes; device-resident
  positions are `Error(Unsupported)`.
- **Large weights.** gemv and get_rows process a W larger than one binding in row slabs. An
  example is the tiny F32 model's 254 MB embedding and LM head against lavapipe's 128 MiB
  maxStorageBufferRange. The split result is bitwise identical to the unsplit one (tested).
- **Tests.**
  - `tests/unit/backend/test_vulkan_backend.cpp` covers each op against the CPU backend, plus
    the interface rules, memory semantics and device errors.
  - `tests/unit/backend/test_vulkan_forward.cpp` is the ADR §6.3 differential: Qwen35 over
    Vulkan against Qwen35 over the CPU backend on the tiny models. It uses the §6.2 fixture
    dimensions and runs the Recurrent path, then the Chunked path.
  - Pass condition, fixed before the first run: relative L2 ≤ 1e-4 per compared tensor, and
    equal argmax tokens except for printed near-ties.
  - Observed worst relative L2: 2.5e-6 to 1.2e-5, with no near-ties.
  - tiny-f32, q8_0, q6_k and iq4_xs pass. tiny-q4_k_m (Q5_0) and tiny-q3_k_m (Q4_0) skip,
    because those types are not implemented on Vulkan.
- **Mutations.** One adapter mutation at a time, each red in the op tests and, where it
  changes values, in the forward differential:
  - swapped gdn_gates operands;
  - imports never written back;
  - device errors ignored;
  - the RoPE head stride dropped;
  - attention on layer 0;
  - KV write at start + 1;
  - swiglu accepting an alias;
  - abort executing the recording;
  - the chunked form ignored (red in the op test only);
  - the LM head result offset;
  - conv slots dropped.

  Two did not go red or were not demonstrable:
  - Removing the zero-fill stays green, because lavapipe hands out zeroed memory anyway, so
    this is not demonstrable here.
  - Ignoring the chunked form passes the forward differential, because the two forms agree
    well within tau there. The op test catches it.

  Detection floor of the forward differential, measured by scaling the rms_norm kernel's
  1/rms:
  - by 1 + 1.2e-7 (1 ulp) and 1 + 9.5e-7 (8 ulp): not detected (worst relative L2 1.0e-5
    and 2.8e-5);
  - by 1 + 7.6e-6 (64 ulp): detected (1.8e-4).

  ADR §6.3 asks for a 1-ulp-scale mutation to be detectable. With tau = 1e-4, fixed a priori,
  this test is about 64× coarser. Tightening tau would need a derived per-op bound for the
  whole forward, which does not exist yet.

## Known limits and notes

- **One `Ops` per thread.** An `Ops` builds pipelines lazily into an internal cache and is not thread-safe.
- **Transfers:** staged transfers are synchronous and serialized per context (one staging buffer). Buffers must not be in use by un-waited GPU work during upload/download.
- **Device status words (V2):** GET_ROWS, KV write and ATTENTION each take their own 4-byte `status` view. The op zeroes it with a recorded fill (`Stream::fill`) and the kernel ORs `k_status_bad_block` (4) / `k_status_bad_index` (8) into it (the halo::hip values, ADR-001 §5.5). Read it after the wait with `read_status` and pass it to `check_status`, which raises `Error(Kernel)`. Two calls must not share a word, because the second call's fill erases the first one's bits.
- **KV pool size (V2):** the whole pool is bound as one storage-buffer descriptor with 32-bit float indices. It must fit in maxStorageBufferRange and 2^32 floats, and a larger pool is rejected with `Error(Unsupported)`. At fp32 the 27B model's 16 attention layers take 128 KiB per token, so a 4 GiB pool holds about 32k tokens across all sequences. Lifting this needs per-layer pools or several descriptors, which is a planner and kv_cache decision.
- **Backend status words (V5):** per-sequence StatusRef owners (ADR §5.5) need the step status array of WS-BI-2, which the interface does not have yet. The Vulkan backend therefore reports device data errors batch-wide, at `wait()`.
- **Backend state ring (V5):** the GDN and conv state ring of ADR §5.3 is in (WS-BI-2 stage 1): `GdnDecodeArgs::ring` / `Conv1dArgs::ring` address one slab of P physical states by a `live` slot index; the model's per-sequence ring slabs are device-resident (`GdnState::attach`). KV pools are still host memory imported per forward (stage 2).
- **Backend weight types (V5):** Q4_0, Q5_0, Q5_1, F16 and BF16 matrices are `Error(Unsupported)`. The MTP pack uses Q4_0 (D-006).
- **Argmax results:** the result buffer must be at least 12 bytes. Read it only through `read_argmax` / `decode_argmax`, which raise on NaN.
- **fp32 remainders on low-alignment devices:** on a device with minStorageBufferOffsetAlignment ≤ 4, fp32 view remainders are always 0, so those shader paths are not exercised there. The fused-qkv test prints a NOTE in that case.
