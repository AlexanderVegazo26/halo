# HALO — Product Requirements Document (PRD)

**Product:** HALO
**Working name:** Hardware-Aware Local Optimization / Inference Engine
**Target hardware:** GMKtec EVO-X2-class system with AMD Ryzen AI Max+ 395, Radeon 8060S, 128 GB unified LPDDR5x memory
**Reference model:** Qwen/Qwen3.8-27B (mandatory v1 validation target)
**Primary platform:** Linux (native; Windows/WSL2 explicitly out of scope for v1)
**Document status:** v1.1 — supersedes Draft v1.0
**Date:** 2026-09-22

---

## Changelog v1.0 → v1.1

1. Added **§2 Reference Model Specification** — the product is now explicitly built around Qwen3.8-27B.
2. Added **roofline/performance budget** (§10) with hardware-relative targets and measured Strix Halo baselines.
3. Added **hybrid linear-attention (Gated DeltaNet) architecture** as a v1 requirement, including recurrent-state management and MTP-based speculative decoding.
4. Corrected the HIP-primary assumption: **Vulkan/RADV is co-primary**; HIP strategy is custom kernels, not ROCm math libraries (§4, §22).
5. Added **unified-memory tier model** (carveout / GTT / pinned) and a deployment checklist.
6. Added **tokenizer, chat-template, Anthropic-compatible API, structured output** requirements for the coding-agent use case.
7. Elevated **continuous batching and prefix caching** into the MVP-adjacent phases.
8. Fixed broken citation artifacts; added a real References appendix.
9. Resolved the model-format open question: canonical v1 input is the ggml-org GGUF pack (Q4_K_M) plus the MTP head file.
10. Vision (VLM) support declared an explicit v1 non-goal, with a roadmap phase.

---

# 1. Executive Summary

HALO is a hardware-specialized, **model-specialized** local LLM inference platform designed around two fixed points:

1. **One machine:** the GMKtec EVO-X2 — AMD Ryzen AI Max+ 395 (16 Zen 5 cores / 32 threads), Radeon 8060S (40 CU, RDNA 3.5, ROCm target `gfx1151`), 128 GB LPDDR5x-8000 shared memory, ~256 GB/s memory bandwidth, 120 W shared APU TDP.
2. **One model:** Qwen/Qwen3.8-27B — a 27B-parameter Apache-2.0 hybrid model (48 Gated DeltaNet linear-attention layers + 16 full-attention GQA layers), trained multi-token-prediction (MTP) heads, a 248K-token vocabulary, native 256K context, and a native vision tower.

HALO is intentionally not a generic llama.cpp replacement, and it is not "splash, but on AMD" in the naive sense: splash achieves 74 tok/s on an Apple M5 Pro whose memory subsystem is *faster* than this APU. HALO's bar is to **match splash's bandwidth efficiency** and to **beat the best existing local baseline on this hardware by a measured, reproducible margin** (see §10).

The strategic principle:

> **Python is the laboratory. C++ is the engine. HIP and Vulkan/RADV are the performance paths. The model is the blueprint.**

Splash's lesson is that the win comes from being built *around* the model — per-model kernels, a trained draft, and a startup memory plan. HALO generalizes that lesson: the engine is engineered for Qwen3.8-27B first, and the generic machinery (graph, autotuner, profiles) is what lets the specialization generalize later.

---

# 2. Reference Model Specification (v1 mandatory target)

## 2.1 Identity

- **Model:** Qwen/Qwen3.8-27B (Hugging Face), released August 2026, Apache 2.0.
- **Class:** dense hybrid linear-attention VLM; **v1.0 of HALO serves it text-only** (vision is a roadmap phase, §16).
- **Canonical v1 input format:** ggml-org GGUF pack, `Q4_K_M` (~17 GB) **plus the separate MTP head file** from the same pack. Direct safetensors ingestion is a later phase.

## 2.2 Decoder architecture

| Property | Value |
|---|---|
| Total decoder layers | 64, repeating pattern `16 × (3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN))` |
| Gated DeltaNet (linear attention) layers | 48 |
| Full-attention layers | 16 (GQA: 24 Q heads, 4 KV heads, head_dim 256, partial RoPE on 64 dims) |
| Recurrent cache | DeltaNet layer state (chunked delta-rule recurrence), per-sequence |
| KV cache | only the 16 attention layers: **~64 KiB/token FP16** (unusually cheap) |
| Context | 256K native; 1M with YaRN |
| Vocabulary | 248,320 tokens, **untied** input/output embeddings |
| LM head | ~1.27B parameters — 2.5 GB FP16, ~0.7 GB quantized; a first-order decode bottleneck (§10) |
| MTP | trained multi-token-prediction head(s), shipped as a separate GGUF file — HALO's v1 draft model |
| Vision tower | 27-layer ViT (images + video); **v1 non-goal** |

## 2.3 Why this matters architecturally

- Only 25% of the layers use the conventional attention/KV machinery the v1.0 documents described. **48 of 64 layers require Gated DeltaNet kernels** (chunked delta rule, short causal conv on Q/K, gated normalization, recurrent-state read/write). This is HALO's primary kernel moat: general-purpose engines' GDN support on this hardware is immature.
- KV memory pressure is low (256K context ≈ 16 GB FP16 KV); the hard problems are **weight-bandwidth-bound decode**, the **LM head**, and **GDN kernel efficiency** — not KV capacity.
- Speculative decoding does not need a separately trained draft model: **the MTP head ships with the model**. MTP is the v1 speculation strategy.

## 2.4 Model discovery additions (extends FR-002)

HALO SHALL detect, from GGUF metadata: `layer_types` / hybrid layout, `full_attention_interval`, linear-attention (GDN) parameters, MTP head presence, vision tower presence, YaRN configuration, and untied-embedding models with oversized vocabularies.

---

# 3. Problem Statement

Large local models on unified-memory APUs create a different optimization problem from discrete-GPU inference.

The Radeon 8060S has no large isolated VRAM pool: a small BIOS-configured carveout (0.5–2 GB) plus a large GTT region (~120 GB, kernel-parameter sized) both map onto the same 128 GB LPDDR5x at ~256 GB/s shared with the CPU. GPU and CPU literally compete for the same bytes-per-second. Every allocation class — carveout, GTT, pinned host memory — has measurably different effective bandwidth, and that difference is a first-order performance input, not an implementation detail.

For decode, the critical path is dominated by:

```text
Weight reads (Q4_K_M ≈ 17 GB per token)
+ LM head read (~0.7–2.5 GB per token depending on precision)
+ DeltaNet state read/write
+ KV read/write (small for this model)
+ kernel efficiency and launch overhead
+ CPU/GPU coordination
```

HALO must optimize the entire data path, on the actual machine, measured — not assumed.

---

# 4. Goals

## G1 — Hardware- and model-specialized inference

A runtime optimized specifically for (a) Ryzen AI Max+/Radeon 8060S-class hardware and (b) Qwen3.8-27B as the reference model. Other Llama/Mistral-class models are supported through the same graph/operator machinery but are not co-equal design targets.

## G2 — Competitive token generation, measured against the right baselines

Improve TTFT and decode tok/s over the best reasonable local configurations on the same machine, measured on the same power state:

- llama.cpp **Vulkan/RADV** — currently the fastest general-purpose decode path on this chip (measured ≈ 20.4 tok/s decode, ≈ 292 tok/s prefill on Qwen3.8-27B Q4_K_M).
- llama.cpp HIP (ROCm 7.2 via `gfx11-generic`; note the missing gfx1151-tuned rocBLAS/hipBLASLt libraries).
- **llama-server with MTP enabled** — the strongest existing configuration on this hardware (MTP roughly doubles dense 27B decode on Strix Halo in community measurements).
- Ollama (the de-facto user default on this hardware).
- CPU-only reference.

The absolute bar is defined in §10 in hardware-relative terms. "Faster than splash on an M5 Pro" is **not** a valid v1 bar.

## G3 — Large-model support

Exploit unified memory tiers and (measured, optional) CPU/GPU partitioning to run models whose weights exceed discrete-GPU VRAM. Default execution is 100% GPU; CPU offload is an autotuned experiment, not a headline feature (CPU and GPU share the same 256 GB/s).

## G4 — Automatic configuration

HALO determines: backend (HIP vs Vulkan/RADV), memory tier placement, quantization, DeltaNet state and KV representation, batch size, context strategy, kernel variant, MTP speculative configuration, draft token count. The user should never need to know; HALO should always be able to explain (plan with reasons, §24).

## G5 — Model portability (secondary to G1)

Support multiple model families through the shared graph/operator model, including hybrid linear-attention architectures. Llama/Mistral-style transformers are validation targets, not the priority.

## G6 — Agent-facing API

OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`) **and Anthropic Messages-compatible** (`/v1/messages`) endpoints, plus `/tokenize` and `/apply-template`, tool-call rendering, `reasoning_effort` passthrough (the model thinks by default), and JSON-Schema structured output. Target consumers are coding agents (Claude Code-style clients, OpenAI SDKs), which is also splash's actual user base.

## G7 — Benchmark-driven engineering

Every major optimization must be measurable through reproducible benchmarks with pinned power state (§9, TRD §49).

## G8 — Dual GPU backend

HIP (custom kernels) and Vulkan/RADV are **co-primary** backends. HIP is not assumed superior: on this chip, ROCm gfx1151 support is unofficial (not on AMD's support matrix as of early 2026), math libraries ship no tuned gfx1151 kernels, and measured decode favors Vulkan/RADV today. HIP's role is HALO's *custom* kernel path; Vulkan/RADV is the robust fallback and the current evidence-based default for decode.

---

# 5. Non-Goals (v1.0)

1. Training or fine-tuning foundation models.
2. Replacing ROCm or Vulkan.
3. Replacing a general-purpose compiler.
4. Every model architecture — v1 is Qwen3.8-27B plus Llama/Mistral-class transformers through the shared IR.
5. Every GPU vendor. AMD-only; CUDA is a future possibility via the backend interface, not a goal.
6. GUI before the engine is stable.
7. Datacenter-GPU-first optimization.
8. **NPU (XDNA 2) inference backend.** Community evidence on this exact box shows an NPU sidecar adds only ~3% main-workload latency versus ~69% for an iGPU sidecar — NPU exploration remains a benchmarked side quest only.
9. **Vision/multimodal inference** (the Qwen3.8-27B ViT tower). Explicit v1 non-goal; text in/text out.
10. Any performance claim without benchmark evidence against §9's methodology.
11. Windows and WSL2 as v1 platforms (WSL2 GPU memory semantics are hostile to this design; revisit post-v1).

---

# 6. Target Users

## Primary

### Local AI enthusiast / developer

Large local models, high throughput, long context, easy API access, minimal manual tuning. Installs on an EVO-X2 and gets the tuned configuration out of the box (persistent profile, no knobs).

### AI engineer / coding-agent user

Runs multiple agent clients concurrently; needs Anthropic- and OpenAI-compatible endpoints, structured output, tool calls, prefix caching across repeated agent system prompts, and predictable concurrency behavior.

### Runtime/kernel researcher

Custom kernels (especially GDN/linear-attention), autotuning, tensor-level instrumentation, memory-tier telemetry, reproducible performance data.

## Secondary

### Local inference server operator

OpenAI/Anthropic endpoints, concurrent requests, monitoring (`/metrics`), predictable resource utilization, resource governance.

---

# 7. Product Principles

- **P1 — Measure, don't guess.** No optimization ships without a benchmark; no claim ships without §9 methodology.
- **P2 — Hardware awareness is first-class.** Tier-level memory bandwidth, carveout/GTT topology, power state — all detected and modeled.
- **P3 — Model specialization is the strategy.** The engine is built around Qwen3.8-27B; generality is derived, not primary.
- **P4 — Memory movement matters as much as arithmetic.** A mathematically optimal kernel loses if its memory behavior is worse; the APU makes this literal.
- **P5 — Specialize aggressively, preserve fallbacks.** Fast paths alongside correct reference paths; typed diagnostics instead of silent fallback.
- **P6 — Deterministic where possible.** Reproducible benchmarks; deterministic mode may sacrifice performance.
- **P7 — Optimize for the real machine.** The actual EVO-X2, its BIOS settings, kernel parameters, and power modes — not an abstract Radeon GPU.

---

# 8. Functional Requirements

## FR-001 — Hardware discovery

HALO SHALL detect: CPU model/topology/SIMD; GPU model, architecture, `gfx1151` target; **memory tiers** — carveout size (via `amdgpu` reporting / sysfs), GTT size and limits, total/available unified memory; measured per-tier effective bandwidth (startup microbenchmark); OS, kernel version (≥ 6.19 required for ROCm 7.2 on gfx1151), ROCm version, Vulkan driver and ICD (RADV vs AMDVLK — AMDVLK must be flagged), CPU SIMD, temperature and power telemetry.

## FR-002 — Model discovery

Architecture; hybrid layer layout (`layer_types`, `full_attention_interval`); parameter count; per-layer-type counts; hidden/intermediate dims; attention heads, KV heads, head_dim, partial-RoPE dims; GDN parameters (state size, conv width, chunk size); vocabulary size and embedding tying; context length and YaRN config; MTP head presence; vision tower presence (report as unsupported-in-v1 with a clean error); tensor types; quantization; weight size; per-layer weight sizes.

## FR-003 — Model compatibility report

Before loading, expose: model, architecture (including hybrid breakdown), quantization, estimated memory by tier, estimated KV and DeltaNet-state memory, supported/partially-supported/unsupported, recommended configuration, warnings (e.g., AMDVLK detected, IOMMU enabled, kernel < 6.19, missing MTP file).

## FR-004 — Memory planning

Estimate weights + KV + DeltaNet state + activations + workspace + runtime + safety margin **per memory tier** before execution; produce a startup budget breakdown and report load time. Never blindly consume all system memory (configurable safety factor).

## FR-005 — Execution planning

Select: backend; memory-tier placement per tensor class; kernel implementation; tensor precision; KV format; DeltaNet state format; batch size; MTP speculation (on/off, draft tokens, verification depth) **based on measured acceptance rate**; CPU/GPU split (default 100% GPU, autotuned as an experiment).

## FR-006 — Runtime inference

Prompt ingestion, prefill, decode, sampling, stop sequences, streaming (SSE), cancellation, multiple concurrent requests (continuous batching), configurable context, configurable batch.

## FR-007 — Sampling

Greedy, temperature, top-k, top-p, min-p, repetition penalty, seed, typical sampling. **Greedy fast path:** fused on-GPU argmax over the 248K-vocab logits, skipping the GPU→CPU logits round-trip.

## FR-008 — State and KV cache management

- **Paged KV allocation** for the 16 attention layers: reusable blocks, prefix reuse (headline metric — splash's cached-TTFT advantage is 6–7×), quantized KV (FP16 baseline, INT8 experimental, validated against perplexity/divergence), memory-pressure detection, eviction, per-request ownership.
- **DeltaNet recurrent-state manager**: per-sequence state allocation, chunked-state checkpointing for prefix reuse where beneficial, state quantization experiments.

## FR-009 — Speculative decoding (MTP-first)

v1 strategy is the model's **trained MTP head** as draft:

```text
MTP draft (k tokens, tree or linear)
      ↓
Target verification (batched over draft positions)
      ↓
Accepted tokens
      ↓
KV + DeltaNet state commit
```

The runtime SHALL measure acceptance rate, target evaluations, effective tokens/evaluation, and net tok/s, and **disable speculation automatically when measured speedup ≤ overhead**. A generic draft-model abstraction remains for later (n-gram, self-speculative, smaller dense drafts).

## FR-010 — Kernel registry

Each kernel exposes: operator, dtypes, tensor dimensions, backend, architecture, tile config, required alignment, memory tier assumptions, estimated bandwidth, measured latency.

## FR-011 — Autotuning

Benchmark candidate kernels per (hardware tier, model shape, quantization) and persist winners. Autotuning SHALL never make the model unusable (validated fallbacks).

## FR-012 — Benchmarking

TTFT (cold and prefix-cached), prompt tok/s, decode tok/s, effective decode tok/s with MTP, end-to-end latency, load time, per-tier peak memory, bandwidth utilization, GPU/CPU utilization, temperature, power, clocks, MTP acceptance rate, **prefix-cache hit rate**, batch scaling curve (1/2/4/8 concurrent).

## FR-013 — API

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions          (OpenAI)
POST /v1/completions               (OpenAI)
POST /v1/messages                  (Anthropic-compatible)
POST /v1/responses                 [phase 2]
POST /tokenize
POST /apply-template
POST /v1/embeddings                [future]
GET  /metrics
```

Streaming via SSE; structured output (JSON Schema) in v1; tool-call rendering per model template; `reasoning_effort` passthrough.

## FR-014 — Tokenizer and chat templates

tiktoken-compatible BPE for the 248K vocab, with HF `tokenizer.json` ingestion; Jinja chat-template rendering (including tool-call variants) shared by the API and `/apply-template`. Tokenizer and template logic must never be ad-hoc per-endpoint.

---

# 9. Performance Requirements

Performance targets are engineering targets, not guarantees. All measurements follow PR-003 methodology with **pinned EVO-X2 performance mode** and recorded clocks.

## PR-001 — Baseline suite

Compare against, on the same machine, same model file, same power state:

1. llama.cpp Vulkan/RADV
2. llama.cpp HIP (ROCm 7.2, gfx11-generic)
3. llama-server with MTP enabled (best-tuned)
4. Ollama (default user configuration)
5. CPU-only reference
6. HALO Vulkan/RADV
7. HALO HIP

## PR-002 — Reporting

Every optimization reports: baseline, candidate, absolute/percentage improvement, variance, hardware state (temperature, clocks, power mode), software versions, model hash, configuration, memory tier placement.

## PR-003 — Methodology

No claim from a single run. Minimum: 5 warm-up iterations; 20 measured iterations for microbenchmarks; 3 independent end-to-end repetitions; cold vs. steady-state distinguished; power mode pinned and recorded.

## PR-004 — SUPERSEDED by DECISIONS.md D-011/D-014 (owner-resolved 2026-09-24)

**This section's original arithmetic was inverted** (the true naive figure is ≈ 66 ms/token ≈
15 tok/s, not 15 ms/token ≈ 66 tok/s) and its absolute targets are physically unreachable on
this platform (≥ 45 tok/s batch-1 decode would need ≥ 742 GB/s of memory bandwidth against a
≈ 256 GB/s machine — see D-011's per-step byte-traffic model, which gives a decode ceiling of
**≈ 15.2 tok/s** at S=1, n=0, η=1). The owner resolved this in D-014: success is defined by
**efficiency NFRs**, not the absolute tok/s numbers below, which are retained only as the
aspirations they were originally written as:

| Metric | Original target (superseded, aspirational only) | D-014 replacement NFR |
|---|---|---|
| Decode, Q4_K_M, batch 1, 4K ctx | ~~≥ 45 tok/s~~ | decode bandwidth efficiency η ≥ 0.80 of measured bandwidth (S=1, n=0, 4K context) |
| Decode with MTP, batch 1 | ~~≥ 70 tok/s effective~~ | MTP net speedup ≥ 1.5× at measured acceptance; auto-disabled below 1.05× |
| Prefill, 512-token prompt | ≥ 400 tok/s (unaffected by the roofline error; still a target) | — |
| Cached TTFT, 32K prefix replay | ≤ 300 ms | cached TTFT ≤ 300 ms for a 24K re-submit and for 32K multi-turn replay at ≥ 80% hits |
| Aggregate decode, 4 concurrent agents | ~~≥ 120 tok/s~~ | above llama.cpp's same-file 22.7 tok/s at S=4 (stretch ≥ 40) |
| Load time, Q4_K_M + MTP | ≤ 30 s (unaffected; still a target) | — |
| 256K context | serve within memory budget (unaffected; still a target) | — |
| Rollback cost | (not in the original) | ≤ (K−1)·0.157 GB extra, no extra weight passes |
| Checkpoints | (not in the original) | GPU pool only |

Every efficiency NFR above is unmeasured until TRD §68 Phase 0 (roofline, load-path,
per-tier bandwidth) runs on the EVO-X2. Targets are revisited each phase against measured
baselines; failing a target is a planning input, not a silent miss.

---

# 10. Competitive Analysis (v1.1 addition)

**Splash (incoai/splash, Apple Silicon)** — the reference for what "built around the model" achieves: Qwen3.8-27B at ~74 tok/s decode on a 48 GB M5 Pro (~300+ GB/s class memory), 363 tok/s prefill, 123–282 ms cached TTFT at 32K, ~170 tok/s aggregate at 4 concurrent agents. HALO does not chase these absolutes — the EVO-X2 has less memory bandwidth. HALO chases **splash's efficiency** (≈ 0.25 tok/s per GB/s ⇒ ≈ 60+ tok/s equivalent on 256 GB/s) and a **≥ 2× margin over the best local baseline on this machine**.

**Local baselines on this machine (community-measured, to be re-verified in Phase 0):**

| Engine | Config | Decode | Prefill |
|---|---|---|---|
| llama.cpp | Vulkan/RADV, Q4_K_M | ≈ 20.4 tok/s | ≈ 292 tok/s |
| llama.cpp | HIP (ROCm 7.2) | slower than Vulkan in tg128 class tests | competitive in prompt processing |
| llama-server + MTP | MTP on | ≈ 1.8–2× over no-MTP (Q8_0 dense 27B: 7.6 → 14.7 tok/s) | — |
| Ollama | defaults | ≈ llama.cpp backend levels | — |

**Where HALO wins:** GDN linear-attention kernel quality (immature in general engines on this hardware), MTP integration depth, LM-head quantization (248K vocab FP16 head ≈ 10 ms/token of the ~15 ms budget — unacceptable), memory-tier-aware placement, prefix caching, and agent-facing API surface.

---

# 11. Reliability Requirements

- RR-001 — Failed optimized kernel falls back to known-correct implementation with a typed diagnostic (never silent when behavior materially changes).
- RR-002 — Autotuning never makes a model unusable.
- RR-003 — Persistent profiles are versioned (HALO version, model hash, GPU/driver/ROCm/Vulkan/kernel/OS versions).
- RR-004 — Corrupt profiles ignored safely; heuristics take over.
- RR-005 — API server recovers from individual request failures without process termination.
- RR-006 — GDN state and KV integrity validated under cancellation and eviction races (differential tests, TRD §30).

---

# 12. Security Requirements

Local service by default: bind localhost, optional API key, TLS via deployment layer, configurable origins, request-size limits, model path validation, no arbitrary shell execution or file access via API, per-request resource limits, queue caps.

---

# 13. Observability

Runtime: requests/s, active requests, queue depth, tok/s, TTFT, generation latency, **prefix-cache hit rate**, MTP acceptance rate. Hardware: GPU/CPU utilization, per-tier memory usage, memory pressure, temperature, power, clocks. Kernel: invocation count, latency p50/p95/p99, bytes moved, effective bandwidth. All available as structured logs (JSON in benchmark mode) and `/metrics` (Prometheus format in v1).

---

# 14. CLI

```bash
halo run qwen3.8-27b-q4km.gguf --mtp mtp.gguf
halo benchmark <model>            # full §9 suite vs baselines
halo tune <model>                 # autotune + persist profile
halo inspect <model>              # FR-003 compatibility report
halo devices                      # hardware + tier + bandwidth report
halo profile <model>              # layered profiling
halo serve --model <model> --port 8080
```

---

# 15. Configuration

```yaml
runtime:
  backend: auto                  # hip | vulkan | auto (evidence-based default)
  device: auto
  threads: auto
  deterministic: false

memory:
  tier_policy: auto              # carveout | gtt | pinned placement per tensor class
  gpu_budget: auto
  kv_cache: auto                 # paged; FP16 baseline, INT8 experimental
  deltanet_state: auto
  kv_precision: auto

scheduler:
  mode: throughput
  max_batch_size: auto
  prefix_cache: true

speculative:
  enabled: auto                  # MTP by default; auto-disabled if not profitable
  draft: mtp                     # mtp | model:<path> | ngram
  max_draft_tokens: 8

autotune:
  enabled: true
  warmup_runs: 5
  measurement_runs: 20

server:
  host: 127.0.0.1
  port: 8080
  anthropic_compat: true
  structured_output: true
```

Explicit user configuration overrides autotuner output except where safety requires otherwise.

---

# 16. Product Roadmap

## Phase 0 — Baseline and hardware truth (weeks 1–3)

Hardware inventory incl. per-tier bandwidth microbenchmarks; BIOS/kernel-parameter checklist applied (§22); llama.cpp Vulkan + HIP + MTP + Ollama benchmark harness; measured-baseline database; reference workloads (agentic chat, long-doc QA, code editing, 4-concurrent agents); Qwen3.8-27B GGUF + MTP canonical pack frozen with hashes.

## Phase 1 — HALO Runtime (MVP)

C++ runtime; GGUF loader (isolated parser → normalized model → graph); tensor abstraction; **tokenizer + Jinja templates**; CPU reference backend; **Vulkan/RADV backend**; GDN + attention operator set; MTP speculative decoding; OpenAI + Anthropic API server; benchmark harness; persistent profiles. Text-only, single-model, batch 1.

## Phase 2 — Memory and State Engine

Tier-aware memory planner; paged KV (16 attention layers); **DeltaNet state manager**; prefix caching; weight residency policy; load-time memory plan; load-time optimization (zero-copy mmap).

## Phase 3 — Kernel Engine and HIP

Custom **GDN chunked-delta-rule kernels** (both Vulkan compute and HIP); quantized GEMM/GEMV families; RMSNorm/gated-norm, partial RoPE, conv1d-short; **LM-head quantization + fused greedy argmax**; kernel registry; microbenchmarks; autotuning; HIP backend reaches parity with Vulkan, then pushes past it where custom kernels win.

## Phase 4 — Concurrency

Continuous batching; dynamic batching; priority and cancellation; backpressure; 4–8 concurrent agent workloads as first-class benchmarks.

## Phase 5 — Generalization

Llama/Mistral-class transformers via shared IR; additional quant families (incl. IQ-class where licensing permits); generic draft models; graph fusion; HALO IR; eventual MLIR (only if profiling justifies).

## Phase 6 — Vision (post-v1)

ViT tower ingestion; image/video preprocessing pipeline; multimodal API (OpenAI `image_url`, Anthropic `image` blocks). Only after text-path performance targets are met.

---

# 17. Quantization Roadmap

GGML K-quant naming throughout (since GGUF ingestion is v1): Q4_K_M (canonical), Q5_K_M, Q6_K, Q8_0, IQ-class experiments where licensing permits. FP16/BF16 for sensitive paths (partial-RoPE dims, gated-norm internals) as measured. **LM head must be quantized by default** (FP16 head ≈ 2.5 GB/token-read is disqualifying). The autotuner measures quality proxy, memory, bandwidth, kernel efficiency, tok/s — and exposes the tradeoff. Note the measured Strix Halo evidence that Q6_K can be both more accurate *and* faster than Q4 on coding workloads; lowest-bit is not assumed fastest.

---

# 18. Benchmark Matrix

### Models

Qwen3.8-27B (mandatory), a small dense (Qwen3-4B class), a Llama/Mistral-class transformer (generality check), a MoE model (phase 5).

### Quantization

Q4_K_M (canonical), Q5_K_M, Q6_K, Q8_0.

### Context

4K, 32K, 128K, **256K** (native), 1M-with-YaRN as an experiment.

### Modes

Prompt processing; single-token decode; **MTP speculative decode**; continuous batching at 1/2/4/8; prefix-cached replay at 32K; cancellation storm; load time.

---

# 19. Success Criteria

HALO v1 is successful when:

1. It loads and serves Qwen3.8-27B (Q4_K_M + MTP) reliably with streaming, cancellation, and structured errors.
2. It characterizes the target hardware including per-tier memory bandwidth.
3. It selects execution configurations automatically and can explain each decision.
4. It has a reproducible benchmark suite against all PR-001 baselines.
5. It produces persistent tuned profiles.
6. It exposes OpenAI + Anthropic compatible APIs with tokenize/apply-template and structured output.
7. It meets at least the decode and MTP-effective targets of PR-004 on the reference machine — or publishes honest variance-attributed shortfalls with an analysis.
8. Correctness holds across optimized and reference kernels (differential tests for GDN state and KV under concurrency).
9. Prefix caching demonstrates a measurable cached-TTFT win on agentic workloads.

---

# 20. Product Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **gfx1151 is not officially ROCm-supported**; library kernels absent | High | Vulkan/RADV co-primary; HIP restricted to HALO custom kernels; ROCm version pinning; kernel ≥ 6.19 requirement |
| ROCm upgrades break compatibility (not in-place supported) | High | Versioned profiles; Vulkan fallback; CI matrix on pinned ROCm |
| **GDN kernel correctness** (recurrent state, chunking) | Critical | Reference implementations; differential tests; fuzzing of state checkpoints |
| Unified-memory bandwidth bottleneck | High | Tier-aware planner; roofline-first kernel design; §10 budgets |
| LM head dominates decode | High | Quantized head; fused argmax; partial-vocab experiments |
| MTP acceptance lower than expected on real workloads | Medium | Auto-disable logic; n-gram/generic drafts as fallback; honest reporting |
| Thermal throttling on 120 W shared APU | Medium | Pin power mode in benchmarks; cold vs steady-state reporting; thermal-aware scheduling later |
| Model compatibility explosion | High | Stable graph/operator IR; Qwen3.8-27B as sole v1 target |
| Autotuning takes too long | Medium | Persistent profiles; heuristic pre-filtering; representative workloads |
| Driver regressions (RADV/AMDVLK/ROCm) | Medium | Driver policy enforcement (RADV only); versioned compatibility matrix; AMDVLK detection warning |
| Excessive scope | High | §16 phasing; vision and extra architectures gated behind text-path targets |

---

# 21. Competitive Positioning

> **A hardware-aware, model-specialized inference runtime engineered to extract maximum practical inference performance from AMD unified-memory AI PCs — starting with Qwen3.8-27B on the Ryzen AI Max+ 395.**

The optimization loop:

```text
Observe → Analyze → Generate → Benchmark → Select → Persist → Execute
```

Differentiators vs. llama.cpp/Ollama on this hardware: GDN kernel quality, MTP depth, LM-head treatment, tier-aware memory planning, prefix caching, agent-facing API breadth — each individually measured (G2, P1), not claimed.

---

# 22. Deployment Requirements (v1.1 addition)

HALO ships with and validates against a deployment checklist for the EVO-X2:

- BIOS: UMA frame buffer at minimum (0.5–2 GB); performance mode selected and pinned for benchmarks.
- Kernel ≥ 6.19; `amdgpu` GTT sized appropriately (~120 GB class) via kernel parameters; documented, not guessed.
- **IOMMU disabled** (≈ 6% memory-read improvement on this platform) — HALO warns if enabled.
- Vulkan: **RADV only**; AMDVLK must be absent (can silently hijack the ICD and severely regress prefill); HALO detects and warns.
- ROCm 7.2+ for the HIP path; `HSA_OVERRIDE_GFX_VERSION` history documented; HALO never requires it silently.
- Native installation first-class; containers optional (driver/GPU integration is performance-critical).

---

# 23. Open Questions (updated)

1. Which GDN chunked-delta-rule tiling wins on gfx1151 / RADV — and does it beat CPU-side state math at small batch?
2. INT8 KV and INT8/state-quantized DeltaNet: measured perplexity/divergence tradeoff on agentic workloads?
3. Optimal MTP draft depth k per workload class; tree vs. linear verification on this GPU?
4. How much of the 248K LM head can be FP8/Q4 before logit quality measurably degrades (perplexity + downstream task)?
5. Does HIP custom GDN beat RADV compute for decode once tiers are tuned — and by enough to justify ROCm's unsupported status?
6. CPU/GPU split: at what layer count does bandwidth contention make offload counterproductive? (Default 100% GPU until measured otherwise.)
7. Can prefix caching extend to DeltaNet state checkpoints at 32K+ without measurable TTFT regression?
8. Continuous batching with heterogeneous GDN state sizes: measured scheduler overhead at 8 concurrent?

(Resolved since v1.0: model format → GGUF pack + MTP file; draft strategy → MTP head; backends → HIP custom-kernel path + Vulkan/RADV co-primary; vision → post-v1 phase.)

---

# 24. Architectural North Star

```text
                ┌──────────────────────────┐
                │   OpenAI / Anthropic API │
                │  + tokenize + templates  │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │    Request Scheduler     │
                │  (batching, prefix cache)│
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │    Execution Planner     │
                └──────┬───────┬───────────┘
                       │       │
              ┌────────▼─┐  ┌──▼───────────┐
              │  Memory  │  │ MTP Speculative
              │  Planner │  │   Decoder     │
              └────┬─────┘  └───┬───────────┘
                   │            │
        ┌──────────▼────────────▼─────────┐
        │  Graph Runtime (hybrid: GDN +   │
        │  attention operators)           │
        └──────────┬──────────────────────┘
                   │
        ┌──────────▼───────────┐
        │    Kernel Registry   │
        └──┬─────────┬─────────┘
           │         │
      ┌────▼───┐ ┌───▼─────┐
      │ HIP    │ │ Vulkan  │
      │ custom │ │ /RADV   │
      └──┬─────┘ └───┬─────┘
         └──────┬────┘
                ▼
   Tiered unified memory → Radeon 8060S
```

---

# 25. References

1. Qwen/Qwen3.8-27B model card and config (HF) — hybrid layout, MTP, vision tower, vocab, context. <https://huggingface.co/Qwen/Qwen3.8-27B>
2. incoai/splash — README, benchmarks, architecture notes. <https://github.com/incoai/splash>
3. AMD ROCm support matrix (gfx1151 not listed; gfx11-generic ISA target). <https://rocm.docs.amd.com/>
4. Strix Halo local-inference guides — GTT sizing, BIOS UMA FB, IOMMU, RADV policy, measured llama.cpp/Ollama numbers on Ryzen AI Max+ 395. <https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF> community guides and Strix Halo inference write-ups.
5. llama.cpp — HIP and Vulkan backends (reference implementation and baseline). <https://github.com/ggml-org/llama.cpp>
6. Qwen3.8-27B GGUF pack (ggml-org) — Q4_K_M sizes, MTP head file. <https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF>
