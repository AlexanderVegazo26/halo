# HALO — Technical Requirements Document (TRD)

**Product:** HALO
**Version:** 1.2 — supersedes v1.1
**Date:** 2026-09-22

---

## Changelog v1.1 → v1.2 (external review)

1. Fixed inverted roofline arithmetic in §2.3 (naive full-weight read ≈ 66 ms/token ≈ 15 tok/s, not 15 ms ≈ 66 tok/s); added the empirical contradiction (verified baselines exceed the naive roofline) and made the effective roofline a Phase-0 measurement that gates PR-004.
2. Concrete DeltaNet state math in §2.2 (48 layers × 16 QK-heads × 128 × 128 × 2 B ≈ 24 MiB/sequence FP16 — carveout-fitting).
3. AMDVLK policy: RADV remains default; AMDVLK is detected, warned, and autotuned as a candidate (community decode wins; P1) rather than hard-blocked (§3.2, §20, §53, §59).
4. ROCm: pin a Phase-0-verified exact build; noted field report of 7.2.2 stable compute-visibility regression on gfx1151 (§3.3, §59).
5. Memory model: added the access-processor axis (CPU-issued ≈ half GPU-issued bandwidth via GMI link) to §13.
6. Load path: mmap-on no longer assumed; Phase-0 load-path experiment (§67 v0.1/v0.4).
7. GDN reframed as green-field — no accelerated AMD baseline exists; elevated in §68 sequence; Phase-0/1 timeboxes assume no kernel to benchmark against.
8. KV prefix caching vs DeltaNet state-checkpoint caching split: state checkpoints are an R&D spike with always-recompute fallback (§12.2, §67).
9. Reasoning-budget safety requirement added to API (§25).
10. §69 north-star efficiency metric redefined against the *measured* roofline.
**Primary target:** AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151) / 128 GB unified LPDDR5x
**Reference model:** Qwen/Qwen3.8-27B (mandatory v1 validation target) — see §2
**Primary implementation:** C++20/23 + Vulkan/RADV and HIP (co-primary backends)
**Research tooling:** Python (never in the hot inference path)

---

## Changelog v1.0 → v1.1

1. Added **§2 Reference Model Technical Specification** (hybrid GDN + attention architecture, MTP, 248K vocab).
2. Added **linear-attention operator set and DeltaNet state manager** throughout (§9, §12, §15, §16).
3. **MTP speculative decoding** specified as v1 draft strategy (§21).
4. Backend reality: **Vulkan/RADV co-primary**; HIP restricted to custom kernels with **no dependency on rocBLAS/hipBLASLt tuned libraries** (§3, §17, §18).
5. Added **memory-tier model** (carveout / GTT / pinned) with per-tier bandwidth microbenchmarks (§13, §14).
6. Added **tokenizer, chat-template engine, Anthropic-compatible API, structured output** (§10, §34, §35).
7. Added **LM-head quantization and fused greedy argmax** kernel requirements (§19).
8. Continuous batching and prefix caching moved earlier in the roadmap (§78).
9. Fixed broken citation artifacts; added References appendix (§82).
10. Repository layout extended: `src/tokenizer/`, `src/template/`, `src/state/`, `models/qwen3_8/`, `backends/*/kernels/linear_attn/` (§4).
11. Deployment requirements: kernel ≥ 6.19, Phase-0-pinned ROCm build, RADV-default driver policy with AMDVLK autotune-candidate, IOMMU, BIOS UMA FB / GTT sizing (§59).

---

# 1. Technical Summary

HALO is a modular inference runtime composed of:

1. model ingestion (GGUF pack + MTP file, v1),
2. graph representation (hybrid linear-attention + transformer),
3. tensor/runtime abstractions,
4. tokenizer + chat-template engine,
5. execution planner,
6. tier-aware memory manager,
7. KV-cache manager (16 attention layers),
8. DeltaNet recurrent-state manager (48 linear-attention layers),
9. CPU backend (reference/fallback),
10. HIP backend (custom kernels),
11. Vulkan/RADV backend (co-primary),
12. kernel registry,
13. autotuner,
14. MTP speculative decoder,
15. scheduler with prefix cache,
16. sampler,
17. benchmark/profiler,
18. OpenAI- and Anthropic-compatible API server.

The architecture allows low-level specialization without coupling model logic to a single GPU backend. The v1 specialization target is Qwen3.8-27B.

---

# 2. Reference Model Technical Specification

Canonical v1 input: **ggml-org GGUF pack, Q4_K_M (~17 GB), plus the separate MTP head file**. Direct safetensors ingestion is a later phase. The loader MUST isolate GGUF specifics (§30).

## 2.1 Decoder

| Property | Value |
|---|---|
| Layers | 64 = 16 × [3 × GDN·FFN + 1 × Attn·FFN] |
| GDN layers | 48 — Gated DeltaNet linear attention; state = per-layer recurrent matrix; short causal conv on Q/K; gated norm; chunked delta-rule recurrence |
| Attention layers | 16 — GQA 24 Q / 4 KV heads, head_dim 256, **partial RoPE (64 of 256 dims)** |
| FFN | gated (SwiGLU-family), per architecture metadata |
| Vocabulary | 248,320; **untied** embeddings; LM head ≈ 1.27B params |
| Context | 256K native; YaRN to 1M |
| MTP | trained multi-token-prediction head(s) as separate GGUF tensor file |
| Vision tower | 27-layer ViT — **detected, reported, rejected in v1** with a clean error |

## 2.2 Cache arithmetic (drives the memory planner)

- KV per token (FP16): `16 layers × 4 KV heads × 256 dim × 2 (K,V) × 2 B` = **64 KiB/token** → 32K ≈ 2 GB; 256K ≈ 16 GB.
- DeltaNet state per sequence: GDN layers use 16 QK-heads at head_dim 128 with 128×128 state per head → `48 layers × 16 × 128 × 128 × 2 B` ≈ **24 MiB/sequence FP16** (plus small conv/gating buffers). This fits the 0.5–2 GB carveout even for many concurrent sequences — state should live in TIER_CARVEOUT until measured otherwise.
- LM head per decode step: FP16 2.54 GB → ~10 ms at 256 GB/s; Q4 ≈ 0.72 GB → ~2.8 ms. **LM head precision is a first-order decode decision.**

## 2.3 Roofline (drives everything)

Decode, Q4_K_M, batch 1: ~17 GB weights + ~0.7–2.5 GB LM head + state/KV per token. Naive nameplate math: ~17 GB ÷ ~256 GB/s ≈ **66 ms/token ≈ 15 tok/s**. However, verified community baselines on this chip already exceed that naive figure (~20.4 tok/s), so the effective roofline is higher — via effective bandwidth above nameplate, cache reuse on hot tensors, and/or fewer bytes touched per token than the full weight set. **Phase 0 MUST therefore measure, on the actual unit: (a) per-tier effective bandwidth, (b) actual bytes-touched-per-token for Qwen3.8-27B decode, and (c) the implied true roofline — before PR-004 targets are treated as final.** All decode kernel work is budgeted against the measured roofline, not the nameplate one. Prefill is compute-bound: 40 CU RDNA 3.5 FP16 peak on the order of 55–60 TFLOPS; expect 30–45% sustained MFU on GEMM-heavy prompt processing without tuned vendor libraries (none exist for gfx1151).

---

# 3. Language and Toolchain Requirements

## 3.1 C++

C++20 minimum, C++23 selectively, Clang/LLVM. C++ owns runtime, scheduling, tensors, memory, graph execution, sampling, speculation, server, profiling interfaces.

## 3.2 Vulkan (co-primary backend)

Vulkan 1.3 compute; **RADV (Mesa) is the default and reference driver**. Policy: detect the active ICD at startup; emit a prominent warning for AMDVLK (it can silently hijack the ICD and has regressed prefill in field reports; it is also discontinued). Per P1, HALO does not hard-block AMDVLK — the autotuner may benchmark it as a candidate driver, since community data shows it winning decode on some quant/model combos; the default and CI reference remain RADV. Shader pipeline: GLSL/HLSL → glslang/glslc → SPIR-V → Vulkan runtime. Shaders compiled reproducibly and embedded/cached.

## 3.3 HIP (co-primary backend, custom kernels only)

HIP/C++ compiles for `gfx1151`; where toolchain support requires it, the `gfx11-generic` ISA target is acceptable (record which was used in the profile key). **HALO SHALL NOT depend on rocBLAS/hipBLASLt/MIOpen tuned kernels for gfx1151** — they ship no tuned Tensile libraries for this target (the `TensileLibrary_lazy_gfx1151.dat` warning is the canonical symptom) and gfx1151 is not on AMD's official ROCm support matrix. These libraries may be used, if present, as optional references only. HALO's GEMM/GEMV/GDN kernels are its own.

Requirements: kernel ≥ 6.19, and a **Phase-0-verified exact ROCm build** (not a blanket "7.2+"): a field report indicates ROCm 7.2.2 stable has a compute-visibility bug that blocks GPU compute entirely on gfx1151, fixed only in a later nightly — so the deployment checklist pins the exact validated build and the profile key records it. The build MUST fail loudly if it would silently compile for a generic target when `gfx1151` is available.

## 3.4 Python

Never in the hot path. Benchmark orchestration, statistics, visualization, model inspection, autotuner orchestration, regression analysis, experiment generation.

---

# 4. Repository Architecture

```text
halo/
├── CMakeLists.txt
├── cmake/
├── include/halo/
├── src/
│   ├── core/            # errors, logging, config, versioning
│   ├── runtime/         # sessions, execution engine
│   ├── graph/           # graph IR, fusion (later phases)
│   ├── tensor/
│   ├── memory/          # tier-aware allocator + planner
│   ├── kv_cache/        # paged KV (attention layers)
│   ├── state/           # DeltaNet recurrent-state manager
│   ├── scheduler/       # batching, prefix cache, priorities
│   ├── sampling/
│   ├── speculative/     # MTP + generic draft abstraction
│   ├── tokenizer/       # BPE (tiktoken-compat), tokenizer.json
│   ├── template/        # Jinja chat templates, tool-call rendering
│   ├── model/           # GGUF loader, normalized model
│   ├── profiling/
│   └── api/             # OpenAI + Anthropic endpoints, SSE
├── backends/
│   ├── cpu/             # reference + AVX2/AVX-512 kernels
│   ├── hip/
│   │   ├── kernels/     # gemm, gemv, linear_attn, norm, rope, quant...
│   │   └── attention/
│   └── vulkan/
│       ├── shaders/     # matmul, quant, linear_attn, norm, rope...
│       └── pipelines/
├── models/
│   ├── qwen3_8/         # reference model implementation
│   ├── common/
│   ├── llama/
│   └── mistral/
├── compiler/            # later phases (HALO IR → MLIR if justified)
├── autotune/
├── tools/               # halo, halo-server, halo-bench, halo-tune, halo-inspect
├── python/              # experiments, analysis, plotting, regression
├── tests/               # unit, integration, correctness, performance, fuzz
└── docs/
```

---

# 5. Core Architecture

```text
API (OpenAI/Anthropic)
 ↓
Request Scheduler (batching, prefix cache)
 ↓
Execution Planner
 ↓
Graph Runtime (GDN + attention operators)
 ↓
Operator Layer
 ↓
Backend Abstraction
 ↓
CPU / HIP / Vulkan(RADV)
 ↓
Tiered unified memory → Hardware
```

No upper layer may invoke HIP or Vulkan APIs except through backend interfaces.

---

# 6. Core Interfaces

```cpp
class Backend {
public:
    virtual ~Backend() = default;
    virtual DeviceInfo device_info() const = 0;
    virtual MemoryTierInfo memory_tiers() const = 0;        // carveout / gtt / pinned
    virtual Buffer allocate(size_t bytes, MemoryTier tier) = 0;
    virtual void copy(const Tensor& src, Tensor& dst, Stream) = 0;
    virtual void synchronize(Stream) = 0;
    virtual void execute(const KernelInvocation&) = 0;
};
```

---

# 7. Tensor System

```cpp
struct TensorDesc {
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    Layout layout;
    Device device;
    MemoryTier tier;
};
```

Data types: FP32, FP16, BF16, FP8 (experimental), INT8, and GGML block quants (Q4_K, Q5_K, Q6_K, Q8_0 families). Support contiguous/strided tensors, views, aliases, device and tier ownership, asynchronous transfers.

---

# 8. Graph Representation

Nodes: tensors, operators, edges with shapes, dtypes, devices, tiers, layouts, dependencies. The v1 graph MUST express the hybrid pattern directly:

```text
Embedding
 ↓
× 16 blocks of:
 ├─ 3 × [ GDN layer (conv1d-short → gated norm → chunked delta rule → gated out ) → FFN ]
 └─ 1 × [ RMSNorm → QKV (partial RoPE) → GQA attention (paged KV) → out proj → residual ] → FFN
 ↓
Final norm → LM head (quantized) → logits
```

---

# 9. Operator Registry

Stable operator IDs:

```text
RMS_NORM
GATED_NORM                         # GDN gating normalization
MATMUL / QUANT_MATMUL / QUANT_GEMV
LOGITS_MATMUL                      # LM head — quantized variants mandatory
PARTIAL_ROPE                       # 64 of 256 dims
ATTENTION                          # GQA, paged KV
CONV1D_SHORT                       # GDN short causal conv (Q/K)
GATED_DELTANET                     # chunked delta-rule recurrence, state R/W
LINEAR_ATTENTION_STATE_CHECKPOINT  # state snapshot/restore for prefix reuse
SWIGLU / SILU_MUL
SOFTMAX
ADD / MUL / CAST / RESHAPE / TRANSPOSE
TOP_K / TOP_P / MIN_P / SAMPLER
ARGMAX_FUSED                       # fused greedy path over logits
MTP_DRAFT / MTP_VERIFY             # speculative decode ops
```

Each operator declares: supported dtypes, layouts, backends, shape constraints, workspace requirements, determinism characteristics, and state side-effects (for GDN: read/write/shape of recurrent state).

---

# 10. Tokenizer and Template Engine

- tiktoken-compatible BPE over the 248K vocab; HF `tokenizer.json` ingestion; byte-level fallback; special-token handling; `added_tokens` (tool-call markup) support.
- Jinja2-compatible chat-template renderer (subset sufficient for Qwen3.x templates, including tool-call variants); shared by `/v1/chat/completions`, `/v1/messages`, and `/apply-template`.
- Template rendering must be deterministic and unit-tested against golden transcripts; template bugs are API-level bugs.

---

# 11. Execution Planner

Inputs: `ModelGraph`, `HardwareProfile` (incl. per-tier bandwidth), `RuntimeConfig`, `RequestProfile`. Output: `ExecutionPlan`:

```cpp
struct ExecutionPlan {
    Backend backend;
    std::vector<KernelSelection> kernels;
    MemoryPlan memory;          // per-tier placement
    KVPlan kv;                  // paged, precision
    StatePlan deltanet;         // recurrent state placement/format
    BatchPlan batch;
    SpeculativePlan speculative; // MTP config + measured acceptance gate
};
```

---

# 12. State Management: KV and DeltaNet

## 12.1 Paged KV (16 attention layers)

Block-based arena; logical position → physical block mapping; allocation, deallocation, prefix sharing (refcounted), eviction, compaction where beneficial. Prefix-cache hit rate is a first-class metric (agents replay long system prompts).

## 12.2 DeltaNet recurrent state

- Per-sequence state allocation across 48 layers; state lives in the fastest feasible memory tier.
- Chunked-delta-rule kernels read/write state each step; state layout is kernel-contract (versioned).
- **State checkpoints** at block boundaries would enable prefix reuse for linear-attention layers. Status: **R&D spike** — unsolved elsewhere at time of writing ("prompt caching is disabled by the architecture" in existing ROCm forks); every checkpoint proposal must be measured against always-recompute, and always-recompute is the guaranteed fallback. This is explicitly distinct from KV prefix caching (16 attention layers), which is a committed v1 deliverable.
- State eviction under memory pressure follows the same policy machinery as KV.
- Correctness invariant: cancellation or eviction at ANY point must leave state consistent (differential tests, §30).

---

# 13. Memory Architecture

## 13.1 Tiers

Physical memory is shared, but bandwidth varies along **two axes** — where the bytes live, and **which processor issues the access**. Field reports indicate CPU-issued bandwidth is roughly half of GPU-issued bandwidth on this SoC (GMI link design), independent of tier; this is a first-order constraint on CPU/GPU-split experiments and on placement of any tensor the CPU touches in the hot loop.

```text
TIER_CARVEOUT   — BIOS UMA frame buffer (0.5–2 GB); small, fastest
TIER_GTT        — GPU-accessible system memory (~120 GB class); the workhorse tier
TIER_PINNED     — pinned host allocations for staging / CPU tensors
TIER_HOST       — ordinary pageable host memory
```

## 13.2 Tier discovery and microbenchmarks

At startup HALO SHALL: read carveout size (sysfs/amdgpu), GTT size and limits, total/available memory; then run a short per-tier bandwidth microbenchmark (read, write, mixed; record in the hardware profile). These numbers feed the cost model (§55) and planner directly. GPU-visible-vs-reported-VRAM confusion (rocm-smi reports) MUST be normalized in `device_info()`.

## 13.3 Placement policy

- DeltaNet state: TIER_CARVEOUT if it fits, else TIER_GTT.
- Weights: TIER_GTT (bulk), hot small tensors (norms, embeddings slices) may pin to TIER_CARVEOUT.
- KV: TIER_GTT.
- Staging buffers: TIER_PINNED, recycled.
- Greedy logits: stay on GPU (fused argmax); sampled logits: quantized transfer if needed.

---

# 14. Memory Planner

Estimates per tier: weights + KV + DeltaNet state + activations + workspace + runtime + safety reserve, before execution; refuses overcommit with a typed `MEMORY_ERROR` and a readable budget table. Initial policy: `available × configurable safety factor = HALO budget`, with per-tier sub-budgets. Weight residency states: RESIDENT / PREFETCHING / EVICTED / PINNED / READ_MOSTLY.

---

# 15. KV Cache Quantization

FP16/BF16 baseline; INT8 experimental; lower precision only with validation against perplexity, output divergence, latency, and memory reduction on agentic workloads (not synthetic prompts alone). Given 64 KiB/token FP16, KV quantization is an optimization, not a necessity, for this model.

---

# 16. DeltaNet State Quantization

FP16/BF16 baseline. INT8 state experiments gated on measured divergence — recurrent error compounds over long sequences, so validation requires LONG-CONTEXT perplexity, not just short prompts.

---

# 17. HIP Backend

Components: `hip_context`, `hip_memory` (tier-aware), `hip_stream`, `hip_event`, `hip_kernel_registry`, `hip_graph` (GPU-side graph capture for decode-step replay where profitable), `hip_profiler`. Build target `gfx1151` (or `gfx11-generic` with explicit logging). No rocBLAS/hipBLASLt dependency (§3.3). llama.cpp's HIP build remains a useful reference for ROCm integration mechanics, not for math libraries.

---

# 18. HIP Kernel Priorities (v1.1 reorder)

1. **Gated DeltaNet chunked delta rule** (state R/W, conv1d-short, gated norm) — the moat.
2. **Quantized GEMV** (decode path, Q4_K/Q6_K/Q8_0) and quantized GEMM (prefill).
3. **LM head: quantized logits matmul + fused argmax** (§19).
4. RMSNorm, partial RoPE, SwiGLU.
5. GQA attention over paged KV (decode + prefill variants).
6. Softmax, reductions, KV-cache ops.
7. MTP draft/verify orchestration kernels.

---

# 19. LM Head Requirements (v1.1 addition)

- LM head runs quantized by default (Q4_K-class; FP16 only as a measured opt-in). Rationale: FP16 head ≈ 2.54 GB/token-read ≈ 10 ms at nameplate bandwidth — a large fraction of the naive 66 ms/token budget (and worse relative to any measured-effective budget); Q4 ≈ 0.72 GB ≈ 2.8 ms.
- Greedy decoding: fused on-GPU argmax; logits never round-trip to CPU.
- Sampling path: top-k pre-filter on GPU (e.g., k ≤ 1024 candidates), transfer only candidate logits + indices for CPU sampling — reduces per-step transfer from ~1 MB (FP16 248K) to a few KB.
- Partial-vocab / two-stage logits experiments logged as open questions.

---

# 20. Vulkan/RADV Backend

Must support: device discovery (with ICD identification and RADV enforcement policy), buffers, descriptor sets, command buffers, synchronization, compute pipelines, shader compilation and caching, timestamp queries. Compute-shader source organization mirrors the HIP kernel tree, including `shaders/linear_attn/`. RADV is the default/reference driver; benchmark reports record the driver (RADV Mesa version or AMDVLK, if autotuned in) in the profile key.

---

# 21. Speculative Decoding (MTP-first)

```text
Request
 ↓
MTP draft (k tokens; tree or linear — measured)
 ↓
Target verification (batched positions, single weight pass)
 ↓
Accepted prefix
 ↓
KV + DeltaNet state commit (atomic w.r.t. cancellation)
```

Requirements:

- Draft source: the model's **trained MTP head** (separate GGUF file), loaded once, sharing the memory plan.
- Metrics per request and aggregate: draft tokens, accepted tokens, acceptance rate, target evaluations, effective tokens/evaluation, net tok/s.
- **Profit gate:** speculation auto-disables when measured net speedup ≤ overhead for the active workload class; re-evaluates periodically.
- Generic draft abstraction (`model:<path>`, n-gram) exists behind the same interface for later phases.

---

# 22. Scheduler

Single request → continuous batching → dynamic batching; priorities; cancellation; backpressure. Objectives: latency, throughput, fairness; policy selectable at runtime. **Prefix cache** integrated at the scheduler layer (not the KV layer) so hit/miss and eviction are scheduling decisions. v1 sequencing: batch-1 → prefix cache → continuous batching (see §78).

## Request state machine

```text
QUEUED → PREFILL → DECODING → STREAMING → COMPLETED
ANY STATE → FAILED | CANCELLED
```

---

# 23. Sampling Engine

Greedy fast path: fused GPU argmax (§19). Sampling path: GPU top-k pre-filter → CPU temperature/top-p/min-p/typical/penalties with deterministic seeds. CPU sampling is the v1 default; GPU sampling only if profiling proves a win (with 248K logits, the transfer is the cost — hence §19's pre-filter).

---

# 24. Model Format

## v1

GGUF ingestion (ggml-org Qwen3.8-27B pack, Q4_K_M canonical) **plus the MTP head file**. Isolated parser:

```text
ModelLoader → NormalizedModel → HALO Graph
```

GGUF metadata handling must cover hybrid models: `layer_types`/`full_attention_interval`, GDN parameters, partial-RoPE dims, untied embeddings, MTP file linkage. Missing/unknown hybrid metadata is a typed `MODEL_ERROR`, never a silent mis-parse.

## Native format (future)

`.halo` — metadata, weights, quantization metadata, graph, kernel hints, hardware profiles. Not before profiling proves value.

---

# 25. API Server

C++ HTTP stack: Boost.Beast or a lightweight production-grade alternative. HTTP/1.1, SSE, JSON, keep-alive, connection cancellation, configurable limits.

Endpoints (v1):

```text
GET  /health
GET  /v1/models
GET  /metrics
POST /v1/chat/completions      # OpenAI
POST /v1/completions           # OpenAI
POST /v1/messages              # Anthropic-compatible
POST /tokenize
POST /apply-template
```

v1 feature requirements: SSE streaming; tool-call rendering via the template engine; `reasoning_effort` passthrough (model thinks by default; `preserve_thinking` honored); **reasoning-budget safety**: reasoning and output token usage tracked and reported separately; safe defaults reserving output headroom when `max_tokens` is small (reasoning can otherwise silently consume the whole budget and return empty content); `reasoning_effort: "none"` accepted gracefully; clear diagnostic when content is empty due to budget exhaustion; **JSON-Schema structured output** (constrained decoding over the 248K vocab — implement via token-mask, not grammar loops in the hot path); structured errors.

---

# 26. CLI Architecture

Executables: `halo`, `halo-server`, `halo-bench`, `halo-tune`, `halo-inspect`. All call the same runtime library; no duplicated inference logic.

---

# 27. Profiling Architecture

Layers: application → operator → kernel → GPU hardware. Sources: high-resolution CPU timers, GPU timestamp queries (both backends), ROCm profiling facilities where available, Vulkan timestamps, Linux telemetry (perf, sysfs, sensors). Profile records include power mode, clocks, temperature — every benchmark artifact carries the full hardware-state header.

---

# 28. Benchmark Harness

Records (JSON, one per run, plus rolled-up statistics):

```json
{
  "model": "Qwen/Qwen3.8-27B",
  "model_hash": "...",
  "pack": "ggml-org Qwen3.8-27B-GGUF Q4_K_M + MTP",
  "backend": "vulkan|hip",
  "driver": "radv <mesa-version>",
  "gpu": "gfx1151",
  "quantization": "Q4_K_M",
  "lm_head": "Q4_K",
  "context": 32768,
  "batch": 1,
  "concurrency": 4,
  "ttft_ms": 0, "ttft_cached_ms": 0,
  "prompt_tps": 0, "decode_tps": 0, "decode_effective_tps": 0,
  "mtp_acceptance": 0.0, "prefix_cache_hit_rate": 0.0,
  "memory_by_tier_gb": {}, "load_time_s": 0,
  "temperature_c": 0, "clocks_mhz": {}, "power_mode": "..."
}
```

Benchmark suites: micro (per-kernel), model (PR-004 matrix), system (4/8 concurrent agents, cancellation storms), regression (nightly vs. baseline database).

---

# 29. Correctness Testing

Every optimized operator: reference implementation, randomized + edge-case inputs, relative and absolute error checks; operator-specific tolerances for quantized ops. **GDN operators additionally require:** state-continuity tests over long random sequences (chunk-boundary alignment), state checkpoint/restore equality tests, and cancellation-mid-state consistency tests.

---

# 30. Differential Testing

For a given input and state:

```text
CPU reference → expected output
HIP           → candidate output
Vulkan/RADV   → candidate output
```

Compare within defined numerical tolerance. GDN operators additionally compare recurrent state trajectories over long sequences (not just outputs). Differential tests run in CI on the GPU runner.

# 31. Fuzz Testing

Fuzz: tensor dimensions, strides, quantization blocks, context lengths, batch sizes, malformed model metadata (especially hybrid `layer_types`), API JSON, cancellation timing, prefix-cache eviction timing, MTP draft-depth extremes.

# 32. Memory Safety

RAII; smart pointers where ownership is shared; explicit ownership for GPU resources; no raw owning pointers; bounds validation in debug/reference paths; sanitizer coverage on CPU code. GPU memory wrappers guarantee release on exceptions. Tier-aware allocators must fail loudly on tier confusion (e.g., assuming carveout capacity that does not exist).

# 33. Determinism

Deterministic mode where practical (fixed seeds, fixed reduction orders, no atomics-based nondeterministic reductions); may sacrifice performance. Config: `runtime.deterministic: false`. Deterministic mode is required for the correctness CI lane.

# 34. Performance Counters

```text
tokens_generated  kernel_calls  kernel_time  GPU_time  CPU_time
memory_allocations_by_tier  KV_hits  KV_misses
prefix_cache_hits  prefix_cache_misses
deltanet_state_checkpoints  state_evictions
speculative_attempts  speculative_accepts  mtp_acceptance_rate
tier_bytes_read  tier_bytes_written  effective_bandwidth_per_tier
```

# 35. Public API Stability

Minimal public C++ API: `HaloRuntime`, `Model`, `Session`, `Request`, `Response`, `Tensor`, `Backend`. Internal classes must not become public accidentally; symbol visibility is enforced in the build.

# 36. Threading Model

```text
API threads → Request scheduler → Runtime workers
            → CPU execution threads → GPU submission thread(s)
```

No uncontrolled thread creation; thread counts derive from the topology probe and are pinned where profitable (Zen 5 core pairs, GPU submit thread isolated from sampler threads).

# 37. Synchronization

CPU mutexes, condition variables, atomics; GPU events, streams, fences. GPU synchronization is explicit; **no global device synchronization in normal request execution** (fatal for decode latency). GDN state commit and MTP verification use fine-grained event chains.

# 38. CUDA Compatibility

CUDA is not a v1 target. The backend interface keeps future CUDA support possible without contaminating HIP-specific code.

# 39. NPU

The XDNA 2 NPU is not a v1 inference backend. Community evidence on the EVO-X2 shows an NPU sidecar adds only ~3% main-workload latency versus ~69% for an iGPU sidecar — so NPU exploration (embedding, reranking, drafting, preprocessing) stays a strictly benchmarked side quest; any NPU integration must beat CPU/GPU alternatives in §50 methodology.

# 40. Acceptance Criteria — Runtime

v1 runtime passes when: supported models load; prompt processing works; streaming decode works; cancellation works at every request state; memory errors are handled with typed diagnostics; CPU fallback works; HIP execution works; **Vulkan/RADV execution works**; GDN state remains consistent under cancellation/eviction (differential tests, §30).

# 41. Build System

CMake options:

```text
-DHALO_BUILD_HIP=ON
-DHALO_BUILD_VULKAN=ON
-DHALO_BUILD_TESTS=ON
-DHALO_BUILD_BENCHMARKS=ON
-DHALO_BUILD_SERVER=ON
-DHALO_ENABLE_LTO=ON
-DHALO_ENABLE_ASAN=OFF
```

HIP target detection: prefer `gfx1151`; fall back to `gfx11-generic` with a logged warning; never silently build generic. Vulkan: require glslang/glslc at build time; embed SPIR-V with hashes for cache validation.

---

# 42. Compiler Configuration

Release: `-O3`, LTO where stable, `-g`, native CPU tuning only for local builds. HIP: per §3.3. Debug: `-O0/-Og -g` + sanitizers where compatible. No aggressive flags that compromise numerical correctness without benchmark evidence.

---

# 43. Dependency Policy

Prefer: C++ standard library, CMake, LLVM/Clang, HIP/ROCm (custom kernels only), Vulkan + RADV headers, SQLite, one JSON library, one HTTP library, one Jinja-subset template library. No large frameworks in the hot path. No Python in serving processes.

---

# 44. Runtime Configuration Resolution

```text
CLI → Environment → Config file → Autotuner profile → Hardware defaults
```

Explicit user configuration overrides automatic tuning except where safety requires otherwise.

---

# 45. Error Model

```text
MODEL_ERROR  MEMORY_ERROR  BACKEND_ERROR  KERNEL_ERROR
CONFIG_ERROR  API_ERROR  DEVICE_ERROR  UNSUPPORTED_ERROR
```

Never silently fall back when doing so materially alters requested behavior; emit a typed diagnostic.

---

# 46. Logging

TRACE / DEBUG / INFO / WARN / ERROR / FATAL; production default INFO; structured JSON logs in benchmark mode.

---

# 47. Observability API

`/metrics` (Prometheus format v1), `/health`, `/v1/models`.

---

# 48. Resource Governance

Maximum model memory, context, batch, concurrent requests, request timeout, queue length. One request must never destabilize the system.

---

# 49. Thermal and Power Governance

- Benchmark runner pins the EVO-X2 performance/cTDP mode and records it; comparisons across power modes are invalid and rejected by the harness.
- Records: temperature, power state, clocks (before/after each suite).
- Reports distinguish cold and steady-state performance.
- Long sessions interleave cool-downs; thermal drift beyond a threshold invalidates the run.

---

# 50. Performance Methodology

Environment prep → cold start → warm-up → steady state → measurement → cool-down → repeat. Never compare a cold HALO run against a warmed-up baseline. Micro: ≥ 5 warm-up + 20 measured. End-to-end: ≥ 3 independent repetitions. Statistics: min/max/mean/median/stddev/p50/p95/p99 stored per run; unstable candidates rejected by the autotuner.

---

# 51. Baseline Integration

Adapters for llama.cpp (Vulkan/RADV and HIP), llama-server **with MTP enabled**, and Ollama. Each run records: binary version, commit hash, backend/driver versions, full command line, model hash, environment, power mode, results. This is the continuously reproducible baseline database all PR-004 claims are judged against.

---

# 52. llama.cpp Relationship

Reference implementation, performance baseline, model-format knowledge source, reusable algorithms where licensing permits, external benchmark target. HALO does not rebuild existing behavior without a measurable reason. Note: llama.cpp's Vulkan/RADV path is currently the strongest general-purpose decode engine on this chip; "beat it" means beat its measured numbers, not its architecture.

---

# 53. Backend Strategy and AMD-Specific Direction

```text
HALO Runtime
 ├── Vulkan/RADV backend   — co-primary; robust, currently best-measured decode
 ├── HIP backend           — co-primary; custom kernels only (no vendor math libs)
 └── Future AMD-native     — abstraction boundary preserved
```

Rationale (measured on Strix Halo, to be re-verified in Phase 0): ROCm gfx1151 is unofficial; vendor math libraries lack gfx1151 tuning; Vulkan/RADV leads on decode today for stock configs (while tuned ROCmFP4+MTP community builds currently hold the absolute decode record — Phase 0 re-measures all of this). HIP's value is HALO's own kernels plus GPU graph replay — where HALO can surpass RADV, it must prove it per §50 methodology. Driver policy: RADV default; AMDVLK detected/warned/autotuned as candidate (§3.2).

---

# 54. Compiler / IR Roadmap

Phase 1: Model → HALO Graph → Backend.
Phase 2: Model → HALO IR → optimization passes → Backend.
Phase 3: HALO IR → MLIR → target lowering → HIP/Vulkan — **only if Phase-2 profiling justifies it.**

---

# 55. Cost Model

`T = max(T_compute, T_memory) + T_sync + T_launch`, with `T_memory = BytesMoved / EffectiveBandwidth(tier)` using **measured per-tier bandwidth from the startup microbenchmarks** (§13.2), and `T_compute` calibrated against HALO's own kernels (no vendor-library anchors). Decode-first workloads on this APU are expected to be memory-bound; the model must *verify* bandwidth-boundness per kernel (occupancy/stall counters) before applying bandwidth-based optimization.

---

# 56. Kernel Selection Strategy

```text
1. Exact profile match (incl. driver/RADV/ROCm versions)
2. Compatible cached profile (version-tolerant match, flagged)
3. Heuristic selection (cost model, §55)
4. Microbenchmark on demand
5. Persist result
```

Autotuning never runs per-request.

---

# 57. Profile Versioning

Profile key: HALO_VERSION, MODEL_HASH, PACK_ID (GGUF + MTP file hashes), GPU_DEVICE, GPU_ARCH, DRIVER_VERSION (RADV Mesa or ROCm), ROCM_VERSION, VULKAN_VERSION, KERNEL_VERSION, OS, POWER_MODE, ISA_TARGET (gfx1151 vs gfx11-generic).

---

# 58. Autotuner

Strategies: EXHAUSTIVE, RANDOM, GRID, HEURISTIC (initial default: heuristic pre-filter → measured selection), BAYESIAN (future). Candidate dimensions per operator family; per §50 statistics; reject unstable candidates. Database: SQLite v1 — tables `hardware_profile`, `tier_bandwidth`, `model_profile`, `operator_profile`, `kernel_candidate`, `benchmark_run`, `winning_configuration`.

---

# 59. Deployment

```text
Ubuntu Linux (kernel ≥ 6.19)
 → RADV (Mesa, current stable) / Phase-0-pinned ROCm (HIP path only)
 → HALO (native install first-class; containers optional)
 → OpenAI- / Anthropic-compatible API
```

Deployment checklist (validated by `halo devices --verify`):

- BIOS UMA frame buffer at minimum (0.5–2 GB); performance mode pinned for benchmarks.
- `amdgpu` GTT sized to the ~120 GB class via documented kernel parameters.
- **IOMMU disabled** (~6% memory-read improvement); warn if enabled.
- RADV active (default); AMDVLK absent or explicitly autotuned — warn otherwise.
- ROCm: exact Phase-0-verified build for the HIP path (see §3.3 — do not assume "7.2+"); record versions in every profile.

---

# 60. CI/CD

Stages: format → compile → unit tests → integration tests → CPU correctness → GPU correctness (self-hosted gfx1151 runner; GPU CI MUST run on real hardware — no simulation) → benchmark smoke vs. regression database → package. Performance CI tracks regressions against the baseline database rather than absolute thresholds; nightly full PR-004 matrix on the pinned runner with pinned power mode.

---

# 62. Acceptance Criteria — Vulkan/RADV

- RADV-default driver policy enforced (AMDVLK detected/warned/autotuned as candidate, §3.2) and reported.
- Device discovery and tier probing succeed.
- Compute pipelines work; core reference operators execute.
- Correctness passes (§29–§31); GDN shaders included.
- Benchmark comparison vs. HIP and llama.cpp baselines runs reproducibly.

# 63. Acceptance Criteria — HIP

- `gfx1151` (or explicitly logged `gfx11-generic`) build succeeds.
- Device discovery succeeds; tier probing succeeds.
- GPU allocations succeed on GTT and carveout.
- Core operators execute; **custom GDN and GEMV kernels exist** (no vendor math-library dependency).
- Reference-vs-HIP correctness passes.
- Benchmark suite executes reproducibly.

# 64. Acceptance Criteria — Autotuner

`halo tune <model>` must: inspect hardware (incl. tier bandwidths) → inspect model → identify tunable operators → generate candidates → benchmark per §50 statistics → select winners → persist versioned profile → have the runtime consume the profile on next launch.

# 65. Acceptance Criteria — API

An OpenAI-compatible client and an Anthropic-Messages-compatible client must each be able to: list models; submit a completion; stream tokens; use tool calls; request structured output; cancel; receive structured errors. `/tokenize` and `/apply-template` round-trip golden transcripts.

# 66. Minimum Viable Product (v1 runtime)

```text
C++ runtime
+ GGUF pack loader (Q4_K_M) + MTP file
+ tokenizer + Jinja chat templates
+ CPU reference backend
+ Vulkan/RADV backend (full operator set incl. GDN)
+ HIP backend (GDN + GEMV + LM head as first custom kernels)
+ paged KV + DeltaNet state manager (single request)
+ MTP speculative decoding with profit gate
+ greedy/temperature/top-p/min-p sampling + fused greedy argmax
+ OpenAI + Anthropic chat APIs + /tokenize + /apply-template
+ prefix cache (single-request prefix reuse at minimum)
+ benchmark harness with baseline adapters
+ persistent autotuned profiles
```

No MLIR. No continuous batching (phase 2). No vision. No NPU. No GUI.

# 67. Version Roadmap

## v0.1 — Baseline and hardware truth

Hardware inventory + tier microbenchmarks (including the access-processor axis, §13.1); **measured decode roofline** (per-tier effective bandwidth × bytes-touched-per-token — gates PR-004); **load-path experiment** (mmap vs buffered per backend; do not assume mmap-on); **ROCm build verification** (exact validated version, §3.3); tuned ROCmFP4+MTP community build reproduced in-harness; deployment checklist applied; llama.cpp Vulkan/HIP/MTP + Ollama + ROCmFP4 baseline database; reference workloads frozen.

## v0.2 — Runtime skeleton

C++ runtime; GGUF+MTP loader; tokenizer/templates; tensor/graph; CPU reference; benchmark harness.

## v0.3 — First GPU paths

Vulkan/RADV operator set including **GDN chunked delta rule**; HIP backend with GDN + quantized GEMV; correctness gates green.

## v0.4 — State, memory, speculation

Paged KV; DeltaNet state manager; tier-aware memory planner; **KV prefix cache** (committed); DeltaNet state-checkpoint caching (R&D spike, always-recompute fallback — elsewhere unsolved); **MTP speculative decoding**; LM-head quantization + fused argmax.

## v0.5 — Autotuner + profiles

Kernel registry; microbenchmarks; heuristic autotuning; SQLite profile database; load-time memory plan + fast load path chosen by the Phase-0 load experiment (not assumed mmap).

## v0.6 — Concurrency

Continuous batching; dynamic batching; cancellation storms; 4–8 concurrent agent benchmarks.

## v0.7 — HIP push

GPU graph replay for decode; HIP vs. RADV kernel shootout per operator family; per-operator backend mixing if measurable.

## v0.8 — Generalization

Llama/Mistral-class transformers via shared graph; additional quants; generic drafts; graph fusion; HALO IR.

## v1.0 — Production-quality runtime

All PR-004 targets measured; API surface complete; CI regression database live.

# 68. Recommended Initial Engineering Sequence

Do not start by writing dozens of kernels.

```text
 1. Apply deployment checklist; capture hardware + tier bandwidths (both axes, §13.1)
 2. Measure the real decode roofline (effective bandwidth × bytes-touched-per-token)
 3. Run the load-path experiment (mmap vs buffered, per backend)
 4. Verify and pin the exact ROCm build (§3.3)
 5. Benchmark llama.cpp Vulkan/RADV
 6. Benchmark llama.cpp HIP
 7. Benchmark llama-server with MTP
 8. Benchmark the tuned ROCmFP4+MTP community build (current record holder)
 9. Benchmark Ollama
10. Capture hardware telemetry under all of the above
11. Establish reference workloads (incl. 4-agent concurrency)
12. Implement HALO tensor abstraction
13. Implement CPU reference (transformer + GDN)
14. Implement Vulkan/RADV GDN kernel — validate  [CRITICAL PATH: green-field,
    no accelerated AMD GDN baseline exists anywhere; success criteria are
    CPU-fallback correctness + measured speedup, not beating a bad kernel]
15. Implement one quantized GEMV + the LM-head path — validate
16. Benchmark against steps 5–9
17. Add profiler + tier telemetry
18. Add kernel registry + autotuner
19. Add paged KV + state manager + KV prefix cache
20. Add MTP speculation with profit gate
21. Add HIP custom kernels; per-operator backend shootout
22. Add scheduler + continuous batching
23. Only then: generalization, fusion, IR
```

This ordering minimizes the risk of spending months optimizing the wrong bottleneck.

# 69. Engineering North Star

The runtime should ultimately make decisions like:

```text
Model: Qwen3.8-27B (Q4_K_M + MTP pack)
Hardware: gfx1151, 128 GB, tiers measured: carveout 1 GB / GTT 118 GB
Context: 32K; Concurrency: 4

Planner:
  Decode backend: vulkan-radv        (hip wins only for GEMM prefill)
  LM head: Q4_K, fused argmax        (greedy path, 0 MB logits transfer)
  DeltaNet state: TIER_CARVEOUT→GTT  (state 38 MB, fits carveout)
  KV: FP16 paged, 16 layers          (2 GB @ 32K)
  Prefix cache: on                   (agent system prompts, hit rate 0.83)
  Speculation: MTP k=4               (measured acceptance 0.71, net +38%)
  Continuous batch: 4

Reason:
  profile_match=true  power_mode=pinned
  decode_efficiency=0.68 of measured_roofline   # Phase-0 measured, not nameplate
  mtp_profit=true     prefix_hit_rate=0.83
```

The user should not need to know why every low-level decision was made, but HALO should always be able to explain it.

# 70. Final Technical Principle

HALO is a **hardware-aware, model-specialized execution system**. The central loop:

```text
MODEL → ANALYZE → PLAN → EXECUTE → MEASURE → LEARN → REPLAN
```

The most important engineering assets:

1. a correct runtime (reference + optimized, differentially tested),
2. an accurate profiler (tier bandwidths, power state, clocks),
3. a strong benchmark corpus with honest baselines,
4. **efficient GDN linear-attention kernels** (Vulkan + HIP),
5. a tier-aware memory planner and state manager,
6. an effective autotuner with versioned profiles,
7. an agent-facing API surface (OpenAI + Anthropic + templates + structured output).

Everything else supports those assets.

---

# 71. References

1. Qwen/Qwen3.8-27B — model card, config, architecture. <https://huggingface.co/Qwen/Qwen3.8-27B>
2. ggml-org Qwen3.8-27B GGUF pack (Q4_K_M sizes, MTP head file). <https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF>
3. incoai/splash — Apple-Silicon reference implementation and benchmarks. <https://github.com/incoai/splash>
4. AMD ROCm documentation and support matrix (gfx1151 status; gfx11-generic target). <https://rocm.docs.amd.com/>
5. llama.cpp — HIP and Vulkan backends. <https://github.com/ggml-org/llama.cpp>
6. Strix Halo (Ryzen AI Max+ 395) local-inference guides — BIOS UMA FB, GTT sizing, IOMMU effect, RADV policy, measured baselines. Various community write-ups, September 2026.
