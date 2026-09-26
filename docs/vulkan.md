# HALO Vulkan backend (WS-F / WS-F2)

Status: a **correctness-first building-block set**, verified on lavapipe only (D-001). It covers RMS_NORM (incl. per-head q/k norm), the quantized GEMV for every UD weight type but three, the GATED_DELTANET recurrent decode (MTP-verify rows, rollback slots), ARGMAX and (V1, 2026-09-25) CONV1D+SiLU, GATED_NORM, PARTIAL_ROPE with head_stride, SWIGLU, MUL_SIGMOID, ADD, ADD+RMS_NORM and the fused GDN gates. It is **not yet a qwen35 forward**; "What the full forward still needs" below lists the missing operators.

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

**Caveat — the bounds are loose.** Observed |vk − cpu| is typically 1e-4 to 4e-2 of the
bound. The long-trajectory GDN test therefore adds a second, random-walk check at 2× CPU
Model G (see S-5 below). A small systematic drift passes the worst-case bound and fails
that check. The random-walk check is a model, not a proof.

## Operator inventory

| Op (TRD §9) | Vulkan | Shader(s) | Semantics | Tests |
|---|---|---|---|---|
| RMS_NORM | `Ops::rms_norm` | `norm/rms_norm` | `(x · rsqrt(mean x² + eps)) · w`, plain x̂·w (**D-004**). Same operation order as `cpu::rms_norm`. No in-place | `VkOps.RmsNormMatchesCpu`, `VkViews.RmsNormStridedViewsMatchCpuAndDense` |
| MATMUL / QUANT_GEMV (decode, one x row) | `Ops::matvec` | `matmul/matvec_{f32,q8_0,q4_k,q5_k,q6_k,iq4_xs}` | ggml block layouts, dequantization as `tensor::dequantize_row`. Covers every type the UD pack needs except IQ4_NL, Q3_K and IQ3_S (**D-007**, **D-014**) | `VkMatvec.*` (6 types × shapes / workgroups / realistic scales), 2-D grid past 65 535 rows, `VkViews.Arena*` |
| GATED_DELTANET (recurrent; T = 1 decode, T > 1 MTP verify) | `Ops::gated_delta_rule_decode` | `linear_attn/gated_delta_rule_decode` | **D-004** item 6. Tiled head mapping `j % n_k`. q/k contract **D-016**: raw q/k, in-kernel per-head L2 (eps 1e-6) when `qk_l2norm`, then `q *= q_scale`. Rollback slots **D-012**. In place or disjoint state_out | `VkDiffGdn.*`, `VkGdn.*` |
| ARGMAX (building block of ARGMAX_FUSED) | `Ops::argmax` + `read_argmax` | `reduce/argmax_{partial,final}` | Lowest index on ties, -inf ordinary. **NaN → Error(Kernel)** through a NaN word in the 12-byte result (**D-016**, review S-3) | `VkArgmax.*`, `VkViews.ArgmaxOnSubViews*` |
| CONV1D_SHORT (V1) | `Ops::causal_conv1d_silu` (`Conv1dArgs`) | `linear_attn/conv1d_silu` | `cpu::causal_conv1d_silu`: K ≤ 8, conv state in/out, rollback slots (**D-012**), out may alias x. Pre-activation accumulated in the CPU order, uncontracted → **bit-identical**; SiLU bounded. State and slots **bit-identical** | `VkLayer.Conv1d*` |
| GATED_NORM (V1) | `Ops::gated_rms_norm` (`GatedNormArgs`) | `norm/gated_rms_norm` | `cpu::gated_rms_norm`, HF order `(w·(x·inv))·silu(z)`; out may alias x or z | `VkLayer.GatedRmsNormMatchesCpu` |
| Per-head q/k RMS_NORM (V1) | `Ops::rms_norm` with a strided view | `norm/rms_norm` | Q halves read in place from the interleaved `[Q \| gate]` rows (rows = T·n_head, stride 512) | `VkLayer.PerHeadQkRmsNorm*` |
| PARTIAL_ROPE (V1) | `Ops::partial_rope_neox` (`RopeArgs`, `rope_cos_sin_table`) | `rope/rope_neox` | `cpu::partial_rope_neox` in place with **head_stride** (TD-9): Q rotated inside `[Q \| gate]` (head_dim 256, stride 512), gate halves untouched. cos/sin table built on the host exactly as the CPU op (Vulkan float sin/cos is only 2^-11 absolute) → **bit-identical** | `VkLayer.PartialRope*`, `VkLayer.RopeTable*` |
| SWIGLU, MUL_SIGMOID, ADD (V1) | `Ops::swiglu` / `mul_sigmoid` / `add` (`EltwiseArgs`) | `eltwise/eltwise` (spec constant) | `cpu::swiglu`, `cpu::mul_sigmoid` (gate may be the strided half of `[Q \| gate]`), `cpu::add` (**bit-identical**); out may alias a or b (the in-place residual h ≡ a, ADR-001 §5.2) | `VkLayer.SwigluMulSigmoidAdd*`, `VkLayer.EltwiseAliasingRules` |
| ADD + RMS_NORM (V1) | `Ops::add_rms_norm` (`AddRmsNormArgs`) | `norm/add_rms_norm` | = `cpu::add` then `cpu::rms_norm` (ADR-001 §5.4): h **bit-identical**, y bounded vs CPU and **bit-identical** to `Ops::rms_norm(h)`; h may alias a or b | `VkLayer.AddRmsNorm*` |
| GDN gates (V1) | `Ops::gdn_gates` (`GdnGateArgs`) | `eltwise/gdn_gates` | qwen35.cpp's sequence: beta = sigmoid(b); g = ssm_a · softplus(a + dt_bias), torch threshold 20, accurate log1p (series for e ≤ 0.5, because Vulkan's log is only 2^-21 absolute near 1); beta may alias b, g may alias a | `VkLayer.GdnGates*` |

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
- **No chunked form.** A T-row call runs the recurrence serially per head, so prefill of 4096 tokens is one dispatch of 4096 sequential rows. It is correct (tested at T = 4096 + 64), but a GPU prefill needs a chunked kernel (see below).
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
| **GQA attention over the paged KV cache** (24 Q / 4 KV heads, head_dim 256, scale 1/√256, causal) | full-attention layers (every 4th) | **Missing.** Needs a paged K/V gather and online softmax. The HIP backend has one (docs/hip.md, M4) that can serve as the design reference | `cpu::attention_gqa` with `PagedRows` |
| **Causal depthwise conv1d + SiLU** (kernel 4, conv state of 3 rows, rollback slots per D-012) | GDN layers, over the 10240 qkv channels | **Done (V1).** The ADR-001 §5.3 state ring (out-of-place final state) is not in yet; it lands with WS-BI-2's ring fields | `cpu::causal_conv1d_silu` |
| **Gated RMSNorm** `(x̂·w)·silu(z)` per value head (dim 128, `ssm_norm`) | GDN output | **Done (V1)** | `cpu::gated_rms_norm` |
| **Partial NeoX RoPE with a head stride** (first 64 of 256 dims, theta 1e7) on the interleaved `[Q, gate]` rows of `attn_q` (per head 256 Q then 256 gate) | full-attention Q and K | **Done (V1)**, in place with head_stride. The CPU reference is applied per head through a strided `Rows` view (bit-identical) | `cpu::partial_rope_neox` |
| **Per-head RMSNorm on Q and K** (`attn_q_norm`, `attn_k_norm`, dim 256) | full-attention layers | **Done (V1)**: rms_norm over a strided view, tested on the interleaved rows | `cpu::rms_norm` |
| **SwiGLU** `silu(gate) · up` | FFN (5120 → 17408 → 5120) | **Done (V1)** | `cpu::swiglu` |
| **Sigmoid output gate** `attn · sigmoid(gate)` | full-attention output | **Done (V1)** (gate read strided from `[Q \| gate]`) | `cpu::mul_sigmoid` |
| **L2 norm of q/k** | GDN | **Done in-kernel** (D-016) | — |
| **beta = sigmoid(b), g = ssm_a · softplus(a + dt_bias)** | GDN gates | **Done (V1)**, fused | `cpu::sigmoid`, `cpu::softplus` (qwen35.cpp sequence) |
| Residual adds, ADD + RMS_NORM | every layer | **Done (V1)** (h ≡ a in place) | `cpu::add`, `cpu::rms_norm` |
| **Batched / prefill matmul** (T > 1 rows) | prefill, MTP verify | **Missing.** Only the one-row matvec exists; prefill would re-read W per row | `cpu::matmul` |
| **Chunked GATED_DELTANET** | prefill | **Missing.** The recurrent kernel is correct but serial in T | `cpu::gated_delta_rule_chunked` |
| **IQ4_NL, Q3_K, IQ3_S matvec** | remaining UD types (D-014 priority list) | **Missing** | `tensor::dequantize_row` |
| **LM head: fused matvec + argmax, TOP_K** | head | argmax building block **done**; fusion and TOP_K **missing** | `cpu::matmul_argmax`, `cpu::top_k` |
| Token embedding lookup (`token_embd`, quantized rows) | input | **Missing.** Needs a dequantize-row kernel | `tensor::dequantize_row` |

After all of those, the forward still needs engine integration: model wiring, per-layer
buffer planning in the GPU pool (D-002), GDN checkpoint copies (D-013), and the RADV runs
that turn every "verified on lavapipe" above into a claim about the target.

## Known limits and notes

- **One `Ops` per thread.** An `Ops` builds pipelines lazily into an internal cache and is not thread-safe.
- **Transfers:** staged transfers are synchronous and serialized per context (one staging buffer). Buffers must not be in use by un-waited GPU work during upload/download.
- **Argmax results:** the result buffer must be at least 12 bytes. Read it only through `read_argmax` / `decode_argmax`, which raise on NaN.
- **fp32 remainders on low-alignment devices:** on a device with minStorageBufferOffsetAlignment ≤ 4, fp32 view remainders are always 0, so those shader paths are not exercised there. The fused-qkv test prints a NOTE in that case.
