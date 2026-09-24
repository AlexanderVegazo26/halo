# HALO v0.2 — Code Review, commits 1aa7f30..a0cdf7f (WS-A, B, C, D, F)

Reviewer: sdlc-suite:code-reviewer (static review; nothing was executed). Date: 2026-09-24.
Tier 3: this range defines the model semantics and the public operator contracts. The reviewer
has no write tools, so the orchestrator saved this file from the reviewer's report; the
findings are unedited.

## Summary

- The CPU reference, model loader, dequant, tokenizer and template are strong. Their goldens
  are anchored to independent sources: HF transformers functions, gguf-py on real blocks, HF
  tokenizers and jinja2, and real GGUF headers.
- The core math traced matches D-004, HF and llama.cpp (GDN recurrent and chunked, tiled head
  mapping `j % n_k` per `qwen35.cpp:440-441`, conv state layout, layer kinds, MTP fallbacks).
- CPU and Vulkan agree on the GDN state layout `[n_v, d_k, d_v]` (d_v fastest) and on the slot
  contract: slot s = state after row T-1-s; slots s ≥ T untouched.
- The ThreadPool has no race or deadlock by reading. The GGUF parser bounds counts, checks
  overflow, alignment and overlap, and has a fuzz test.
- **The problem is the CPU/Vulkan seam.**

**Verdict:** request changes for the Vulkan backend's acceptance (M-1, M-2). The CPU, model,
tensor, tokenizer, template, hardware and memory modules are approvable, with the Should Fix
items tracked.

**Findings:** Must Fix 2 · Should Fix 7 · Nit 2.

## Must Fix

**M-1 — CPU and Vulkan GDN kernels expect different q inputs.**
- The CPU kernel always scales q by 1/sqrt(d_k), even when `qk_l2norm=false`
  (`backends/cpu/gated_delta_rule.cpp:72-76`).
- The Vulkan kernel expects q already normalized and scaled, and does neither itself
  (`include/halo/backends/vulkan/ops.h:50-52`).
- Yet the Vulkan header claims its conventions are shared with the CPU op.
- Consequence: shared input buffers give q scaled twice on one backend, or not scaled on the
  other.

→ Resolved by **D-016**: explicit `qk_l2norm` + `q_scale` parameters on both backends.

**M-2 — Vulkan kernels are validated against a private oracle, not `halo::cpu` / `halo::tensor`.**
- The oracle is `tests/unit/vulkan/reference.h`: a third copy of the dequant code plus its own
  fp64 `gdn_step`.
- TRD §30 and ARCHITECTURE's acceptance gate are therefore unmet.

→ Fix: link `halo_backend_cpu` and `halo_tensor` into `test_vulkan`; run differential tests on
the same input buffers; delete the private dequant port.

## Should Fix

**S-1 — Vulkan ops accept only whole, dense buffers at offset 0** (`ops.cpp:28-43`).
Consequences:
- a fused qkv row cannot be split without copies;
- weights cannot share an arena buffer;
- the aliasing check compares buffer handles, not byte ranges.

→ `BufferView{buffer, offset, bytes, stride}` for every op; range-overlap checks.

**S-2 — CPU views can't express interleaved [Q | gate] for token-major head ops.**
- `overlaps()` compares address ranges, so it rejects element-disjoint interleaved views.

→ Either add per-head strides, or add a de-interleave op and document it.

**S-3 — ARGMAX handles NaN differently on the two backends.**
- CPU throws on NaN.
- Vulkan ignores NaN and returns `0xFFFFFFFF` only if every element is NaN.

→ Resolved by D-016: both backends raise.

**S-4 — Missing adapters for the forward pass.**
- `WeightRef` (ggml `ne={K,N}`) → `WeightMatrix` (N×K plus a dequant callback).
- F16/BF16 vector weights → float.
- `TokenizerMetadata` → `VocabSpec` (the int64/int32 types don't line up).

→ One tested adapter layer.

**S-5 — TRD §30 long GDN trajectories are untested.**
- The longest CPU run is T=150.
- Vulkan is tested for 20 steps.
- The Vulkan in-place mode is tested only with T=1.

→ Add T ≥ 4096 chunked prefill + recurrent decode vs fp64, and the same trajectory as a
CPU-vs-Vulkan differential.

**S-6 — No Vulkan Q5_K matvec**, although D-014 lists Q5_K as a first-priority type.

**S-7 — Planner and ARCHITECTURE still call GDN prefix checkpoints a disabled spike.**
- The planner has `enabled=false`, which D-013 supersedes.

## Nits

**N-1** — Staging upload allocates a full-size staging buffer on every call (1 GB for the LM
head). → Use a bounded, reused staging ring.

**N-2** — The Vulkan backend includes `halo/tensor/dtype.h` without linking `halo_tensor`.

## Needs runtime verification (qa-engineer)

1. Chunked vs recurrent GDN divergence after a 4k–24k-token prefill, at real dims.
2. The Vulkan in-place T > 1 path with slots.
3. RADV-specific behaviour:
   - fp16 subnormal scales;
   - FMA contraction;
   - `maxStorageBufferRange` for the 1.04 GB LM head binding.
4. The M-1 contract, once fixed, run through both backends on shared buffers.
