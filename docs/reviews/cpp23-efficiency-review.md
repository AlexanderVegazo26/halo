# C++23 Adversarial Efficiency Review

**Repository:** HALO — C++23 LLM inference runtime (Qwen3.8 on AMD "Strix Halo" / EVO-X2: Zen 5 CPU + RDNA-class iGPU, unified LPDDR5X memory, no discrete VRAM)
**Date:** 2026-09-26
**Scope:** Repository-wide performance/efficiency review per `cpp-bp.md`. Independent of `ADVERSARIAL_REVIEW.md` (correctness/security) — no overlap intended; a handful of items are noted as "known, out of scope" where the two reviews touch the same file for different reasons.
**Method:** Six parallel read-only review slices (Vulkan backend, HIP backend, CPU kernels + thread pool, decode-tick/engine hot path, API/tokenizer/template, build system + profiling harness), each required to read every file in its slice in full and to separate OBSERVED fact from INFERRED hypothesis from REQUIRES MEASUREMENT. Synthesized by the orchestrator. **No files were modified.**
**Hardware constraint:** This review ran on a dev host with no AMD GPU (D-001). No profiling or benchmarking was performed. Every impact estimate below is either a stated algorithmic-complexity argument or is explicitly marked `Impact: Unknown — requires benchmark/profiling`. No benchmark numbers were invented anywhere in this document.

---

## A. Executive Summary

HALO's C++ is, on the whole, disciplined: RAII is used correctly throughout the GPU backends (move-only wrappers, fence-guarded destructors), `std::span` is already idiomatic at buffer boundaries, per-thread scratch vectors are hoisted out of hot loops almost everywhere they matter, and the thread pool avoids the worst false-sharing patterns. The review did not find a single P0 correctness bug (no data race, no use-after-free, no UB) — that finding was in the prior `ADVERSARIAL_REVIEW.md` and is out of scope here. But two facts change the whole shape of this review's conclusions:

1. **This is a unified-memory APU, not a CPU-bound or discrete-GPU workload**, so the review actively looked for "optimizing the wrong resource" — and found it in one specific place: the Vulkan backend still designs its memory-type selection and staging path as if it were talking to a discrete GPU across PCIe (§E.2, §C-row 3), forgoing the zero-copy opportunity the actual hardware offers. Every other subsystem's cost model (CPU-bound vs memory-bound) is genuinely ambiguous without measurement, and the review says so explicitly rather than assuming CPU-bound is the default lens.

2. **The single most severe, unconditional, unmeasured-but-provable finding is algorithmic, not micro-architectural**: `Qwen35::forward()` runs an O(n²) nested loop over every active sequence in the batch, on every single decode tick, to check for KV/GDN-state aliasing (§C-row 1, finding F6 from the decode-tick slice). This is exactly the kind of finding cpp-bp.md §27 asks to be weighted above micro-optimizations — it is a certain complexity fact, independent of any profiler.

3. **The build has real, unclaimed infrastructure sitting idle**: a full statistics/warmup/stability-classification profiling harness (`src/profiling/**`) and a real host memory-bandwidth prober already exist and already feed the autotuner's cost model — but PGO is entirely unwired despite that harness being exactly the training-workload generator PGO needs, and the *only* microbenchmark in the repo (`bench_gdn.cpp`) benchmarks the CPU reference oracle, which is deliberately de-optimized (`-ffp-contract=off`) for bit-exactness against GPU kernels — i.e., the one number anyone could currently produce from this repo would be actively misleading about real performance (§C-row 4, §I).

**Is the code appropriately optimized for C++23?** Partially. `std::span` is used well; `std::expected`, `std::mdspan`, and concepts-based dispatch are all realistic, moderate-effort upgrades that were not found (idiom opportunities are noted per-slice below, all P3, none urgent). No C++23 feature gap is blocking correctness or causing a measurable regression on its own.

**Is the architecture CPU-bound, memory-bandwidth-bound, dispatch-bound, or synchronization-bound?** The honest answer is: **it is not known, and the repo currently cannot find out**, because there is no benchmark for matmul, attention, the Vulkan path, or the HIP path — only for one GDN kernel on the reference CPU oracle. The structural evidence leans toward **dispatch-bound and allocation-bound on the GPU host side** (many small Vulkan dispatches per layer, each paying descriptor-set allocation + a fresh scratch-buffer `vkAllocateMemory`, plus a coarse whole-pipeline barrier between every pair of them) rather than compute-bound — but this is an inference from code shape, not a measurement, and is called out as such throughout.

---

## B. Architecture Map

```
HTTP request
  │
  ▼
src/api/server.cpp, requests.cpp, admission.cpp   (parse, validate, admit)
  │
  ▼
src/api/prompt.cpp (build_chat_prompt: template render + escape + tokenize)
  │            └─ src/template/chat_template.cpp (Jinja, parsed once at load)
  │            └─ src/tokenizer/{tokenizer,pretokenize,unicode}.cpp
  ▼
src/runtime/engine.cpp (CpuEngine::tick — the batched decode step, runs once per generated token across the whole batch)
  │
  ▼
src/speculative/speculative.cpp (draft / verify / commit)
  │
  ▼
src/models/qwen35.cpp (forward / mtp_forward — per-layer graph glue)
  │
  ├──▶ src/kv_cache/paged_kv.cpp        (KV storage, page tables)
  ├──▶ src/sampling/sampler.cpp          (top-k/top-p/typical/penalties)
  │
  ▼
src/backend/{cpu,vulkan,hip}_adapter.cpp  (backend-agnostic op dispatch)
  │
  ├──▶ backends/cpu/**      (reference float32 kernels, bit-exact oracle)
  ├──▶ backends/vulkan/**   (production iGPU path: buffer/stream/kernel + GLSL/SPIR-V shaders)
  └──▶ backends/hip/**      (compile-checked only; never run on real ROCm hardware yet)

Supporting, off the hot path:
  src/memory/planner.cpp    — runs once at Engine construction only
  src/profiling/**          — offline/on-demand measurement harness
  src/autotune/**           — offline kernel-parameter search, feeds a SQLite profile DB
  src/hardware/**           — host memory-bandwidth probe, feeds autotune's cost model
  cmake/**, CMakeLists.txt  — compiler flags, LTO/march/sanitizer toggles
```

Data flow for one generated token: `engine.tick()` → `speculative::step()` (draft + verify, 1–9 model forward calls depending on MTP depth) → `qwen35::forward()`/`mtp_forward()` per call → per-layer: norm → attention (KV read/write) → GDN/conv where applicable → MLP → (trunk only) LM head → argmax/sampler → detokenize → SSE emit. The backend adapter (CPU/Vulkan/HIP) is selected once at startup and every op in that per-layer chain routes through it.

---

## C. Top Performance Findings

| Priority | Location | Problem | Why It Matters | Recommended Change | Expected Impact | Confidence |
|---|---|---|---|---|---|---|
| **P0** | `src/models/qwen35.cpp:686-735,704-707` (`Qwen35::forward`), `:827-829` (`mtp_forward`) | Unconditional O(n²) nested loop over every active sequence in the batch, checking KV/GDN pointer aliasing, on **every decode tick** | This is the single hottest loop in the runtime (runs every token, every tick); a certain algorithmic-complexity fact (cpp-bp.md §27 weights this above any micro-optimization), scales quadratically with `max_sequences` — exactly the concurrency scaling this project is built for | Replace the nested loop with a single O(n) (or O(n log n)) pass using a hash-set/sorted-vector duplicate check on `kv`/`gdn` pointers | Algorithmic: turns O(n²) into O(n). Wall-clock magnitude: Unknown — requires benchmark, but the growth curve itself is provable from the code, not a guess | High (complexity fact) / Unknown (wall-clock) |
| **P1** | `backends/cpu/matmul.cpp:37-59` + `include/halo/backends/cpu/weight_matrix.h:54-64` | The "dequant reused across x.rows()" optimization gives zero benefit during single-token decode (`x.rows()==1`), so every quantized weight matrix is fully re-dequantized from scratch on every decode-step matmul call, in addition to the multiply-add itself — roughly doubling per-token FLOP-adjacent work for every quantized layer | Runs every projection (QKV, gate/up/down, output), every layer, every decode token — the CPU reference/fallback path's single largest per-token cost | Route the decode (`rows==1`) case through `quant.cpp`'s existing fused `vec_dot_row()` (already implements dequant-then-dot in a cache-resident stack buffer) instead of materializing a full float row block | Algorithmic: removes a full O(N·K) unpack pass. Wall-clock: Unknown — requires benchmark | Medium (structural fact, magnitude unmeasured) |
| **P1** | `backends/vulkan/src/stream.cpp:140-199` (`Stream::dispatch`) + `src/backend/vulkan_adapter.cpp:114-130` (`scratch`/`stage`/`add_download`) | Every op dispatch does 2 heap allocations + a fresh `VkDescriptorSet` pool-allocation + a full `vkUpdateDescriptorSets`; separately, every workspace/staging need calls `hv::Buffer::create` — a fresh `vkAllocateMemory` — per op, per layer, per token, with no pooling/reuse until the next `wait()`'s `clear()` | ~10-20 dispatches per transformer layer × num_layers × every token; `vkAllocateMemory` and descriptor-set churn are among the most expensive Vulkan driver calls, and this is the Vulkan path's dominant per-token host-side overhead | Adopt `VK_KHR_push_descriptor`/descriptor-update templates to skip pool allocation+update entirely; give `VkStream` a small fixed set of pre-sized, reused scratch/staging buffers reset (not destroyed) per step | Algorithmic: removes O(dispatches-per-token) allocations entirely. Wall-clock: Unknown — requires benchmark on real hardware | Medium (call-count fact certain; dominance over GPU compute unmeasured) |
| **P1** | `backends/cpu/thread_pool.cpp:51-93` | Every kernel dispatch (dozens per token) pays a full mutex-lock + `notify_all` + OS thread-wake round trip, even for tiny per-call workloads (decode with few parts) | Paid on literally every op in the CPU reference path, every token; OS wake latency is real (µs-scale) and this is the dispatch mechanism, not the compute, for the whole reference backend | Spin briefly on an atomic generation counter before falling back to condvar wait; cap wakeups to the number of parts actually needed instead of `notify_all()` unconditionally | Algorithmic: eliminates wasted wake/sleep cycles for threads with zero work. Wall-clock: Unknown — requires benchmark | Medium |
| **P1** | `src/sampling/sampler.cpp:117-219,311-330` (`chain::draw`, `apply_top_p`, `apply_typical`) | `draw()` unconditionally sorts the full candidate set (up to full ~248K-row vocab when top_k is disabled) purely to consume it in id order for a single weighted draw — an O(V log V) sort for an O(V) task; `apply_top_p`/`apply_typical` also fully sort before any top_k-style bound is applied | Runs once per non-greedy sampled token — likely the common case in production serving (greedy is rarely the default); this is a certain complexity fact, not a guess, since the sort's output order is provably discarded except to sum weights | Drop the unconditional sort in `draw()` (a single pass computing the CDF suffices for a weighted draw); apply top_k unconditionally before top_p/typical with a sane default cap so later stages operate on a bounded candidate count (standard llama.cpp-style chain ordering) | Algorithmic: O(V log V) → O(V) (or O(k log k) after bounding). Wall-clock: Unknown — requires benchmark, but V≈248K is not a small constant | Medium-High (complexity fact certain) |
| **P1** | `src/api/prompt.cpp:244-270` (`build_chat_prompt`, M10 checkpoint hints) | For every chat request past 8 messages (i.e., effectively every real multi-turn conversation), the code re-renders the Jinja template AND re-tokenizes up to 8 conversation prefixes non-incrementally, on top of the main render+tokenize | Adds up to ~8× the primary per-request render+tokenize cost, unconditionally, to every multi-turn request | Render prefixes incrementally, or derive checkpoint hints from token counts already produced elsewhere, or make `kMaxHintBoundaries` configurable/smaller | Algorithmic: 8× → close to 1× extra work. Wall-clock: Unknown — requires benchmark under a long-conversation workload | Medium (complexity fact certain; per-request not per-token, so lower overall weight than per-tick findings) |
| **P1** | `src/kv_cache/paged_kv.cpp:197-206` (`SequenceKv::view`) | Rebuilds a `std::vector<const float*>` block-pointer table from scratch on every call — O(context_length / block_tokens) — even though the underlying block list (`blocks_`) only changes via explicit `reserve()`/`truncate()`, not on every read | Called per sequence per attention layer per tick; this is the one place in the decode-tick slice where per-tick cost provably scales with context length across all layers — the classic "hidden O(n) that becomes O(n²) in aggregate" shape cpp-bp.md flags as high priority | Cache the pointer table on `SequenceKv`, invalidated only when `blocks_` actually changes (a dirty flag or generation counter), instead of rebuilding unconditionally | Algorithmic: O(context/block) redundant work per layer per tick → O(1) amortized. Wall-clock: Unknown — requires benchmark, magnitude grows with context length | Medium-High (complexity fact certain, grows with the one dimension — context length — this project explicitly cares about scaling) |
| **P1** | `-ffp-contract=off` applied uniformly to `halo_models`, `halo_speculative`, `halo_tensor`, `halo_backend` (per-subsystem CMakeLists), not just the CPU reference oracle it was intended for | FMA fusion is disabled project-wide for bit-exactness reasons that only apply to the CPU reference backend's role as a differential-testing oracle; Zen 5 has 2 FMA ports, so this is potentially leaving real throughput on the table in modules that don't need bit-exactness | If these modules never do their own float math outside the abstract op interface, disabling FMA there is pure cost with no correctness benefit | Audit `src/tensor`, `src/models`, `src/speculative`, `src/backend` for any direct floating-point arithmetic outside calls into `backend_cpu`; remove `-ffp-contract=off` from targets that don't need the oracle guarantee | Algorithmic: restores FMA fusion where safe. Wall-clock: Unknown — requires benchmark; FMA removal classically costs one add instruction per fused op on throughput-bound loops | Medium (mechanism certain, applicability to these specific modules requires code audit + measurement) |
| **P2** | `backends/hip/src/hip_adapter.cpp:197-214` (`HipStream::scratch`) | Fresh `hipMalloc`/`hipHostMalloc` per op invocation needing scratch (status words, GDN/attention/argmax/top_k workspace), every call, every decode step | `hipMalloc` is a synchronizing driver call; several of the affected ops (GET_ROWS, KV_WRITE, ATTENTION, ARGMAX) plausibly run every decode step | Pool/reuse scratch allocations per `HipStream` (already scoped to stream lifetime); sizes are deterministic per op signature | Algorithmic: removes per-call alloc/free churn. Wall-clock: Unknown — HIP has never run on real ROCm hardware in this project yet, so this is speculative even relative to other findings | Low-Medium (HIP path is compile-checked only; real-hardware relevance unconfirmed) |
| **P2** | `backends/vulkan/src/stream.cpp:95-105,189-236` (`barrier_if_needed`) | A coarse whole-pipeline memory barrier (`SHADER_WRITE\|TRANSFER_WRITE → SHADER_READ\|WRITE\|TRANSFER_READ\|WRITE` across `COMPUTE_SHADER\|TRANSFER`) fires unconditionally between every pair of dispatches, with no per-buffer dependency analysis, even when consecutive ops touch entirely disjoint buffers | Prevents any command-level overlap between back-to-back dispatches; happens dozens of times per token | Track write-sets per dispatch (the aliasing-check code already computes disjoint byte ranges) and only barrier when the next op's bound buffers actually intersect the previous op's writes | Algorithmic: fewer inserted barriers. Wall-clock: Unknown — depends on whether RDNA hides this via wave occupancy | Low-Medium |
| **P2** | Root `CMakeLists.txt:17` (`HALO_ENABLE_LTO` OFF by default) | No documented rationale for LTO being off; architecture is static-lib-per-module, so cross-module inlining is blocked at every archive boundary without it | Affects every cross-module call in the decode hot path where caller and callee live in different static libs | Enable LTO by default for Release/RelWithDebInfo now that module boundaries are stable, or document why not | Wall-clock: Unknown — requires an A/B build comparison (e.g. `bench_gdn.cpp` with/without LTO) | Low (plausible, unmeasured) |
| **P2** | Repo has exactly one microbenchmark (`backends/cpu/bench/bench_gdn.cpp`), and it benchmarks the deliberately de-optimized CPU reference oracle, not the production Vulkan/HIP paths | Any performance conclusion drawn from this repo's only benchmark would be about the wrong code path entirely — the file's own header comment already disclaims this | Add benchmark targets for Vulkan and (when hardware is available) HIP, mirroring `bench_gdn.cpp`'s structure, covering matmul/attention/GDN on the actual production path | Not a runtime-cost finding — an evidence-availability finding: right now, no number this repo could produce would answer cpp-bp.md's central question | High (architectural fact, not a hypothesis) |
| **P3** | `backends/cpu/kernel_common.h:16-27` (`detail::dot`) and callers throughout `ops_basic.cpp`/`matmul.cpp` | No `__restrict` (or C++23 equivalent) on pointer parameters, even though runtime `check_disjoint` calls already establish non-aliasing | Compiler cannot statically prove non-aliasing → can block auto-vectorization; `dot()` has the broadest blast radius of any single finding since matmul/attention/norm all route through it | Add `__restrict` now that runtime checks already guarantee the property; or adopt `std::mdspan` with a non-aliasing accessor policy as a self-documenting C++23 alternative | Wall-clock: Unknown — requires compiler-output inspection or benchmark | Low-Medium |

*(Full per-slice finding lists, including all P2/P3 items not promoted to this table, are preserved in the per-slice reports referenced in §D–§G below; this table surfaces only what rose to P0/P1/highest-signal P2 after cross-slice dedup.)*

---

## D. C++23 Findings

| Current approach | C++23 opportunity | Why it matters | Example strategy | Expected effect |
|---|---|---|---|---|
| `detail::dot()` and friends take raw `float*`/`const float*` with no aliasing annotation, despite runtime-verified non-aliasing | `__restrict` (compiler extension, not strictly C++23) now, or `std::mdspan` with a custom non-aliasing accessor policy as the standardized alternative | Removes a real vectorization blocker in the single most call-frequency-critical primitive in the CPU backend | Add `__restrict`/`__restrict` (MSVC) immediately; consider `mdspan` migration for `RowsView`/`ConstRows` as a larger follow-up | Unknown — requires benchmark/compiler-output check |
| Argument-validation checks (`ops_internal.h::operand`, `Resolver::get`) throw on the (rare) failure path | `std::expected<T, Error>` for hot-path validation | Would avoid C++ exception-handling overhead on the failure path — but since these checks are expected to never fire in steady state, this is a style preference, not a perf fix (exceptions on the happy path cost nothing in Itanium ABI) | N/A — noted as low priority | None expected on steady-state throughput |
| Hand-rolled 2D views (`RowsView`/`ConstRows`, `Operand`/`Ref`/`PoolRefs` structs) with explicit stride/offset math | `std::mdspan` for multi-dimensional tensor views | Self-documenting layout + could carry a non-aliasing accessor, addressing the `__restrict` gap in a standard way | Migrate incrementally, starting with the CPU backend's row-block views | Unknown — structural/style benefit, not proven perf |
| `WeightMatrix`'s dense/dequantized dual-mode dispatch uses `std::function` | Concept/template-based per-DType static dispatch | Removes one indirect call per row-block in the matmul hot path | Template `WeightMatrix` on a dtype-tag, dispatch via `if constexpr`/concepts | Likely negligible — `std::function` overhead is small relative to the O(N·K) work it wraps; low priority |
| Function-pointer based dispatch in HIP's `ops.cpp` (`dispatch<P>`) | *(no change recommended)* | This is already the correct zero-overhead choice over `std::function`/lambdas for a hot dispatch path — flagged as a **positive pattern**, not a finding | — | — |
| GLSL shaders use `precise` throughout `attention.comp`/`eltwise.comp` | *(no C++23-analogous change — GLSL, not C++, but flagged for awareness)* | Disables FMA-contraction/reassociation, likely a deliberate bit-exactness tradeoff vs the CPU reference (comments say so explicitly) | Only revisit if bit-exact GPU/CPU parity is negotiable | Unknown, and possibly an accepted tradeoff, not a defect |

No case was found where blindly replacing existing code with a C++23 feature would be a net win without measurement; all items above are flagged as opportunities, consistent with cpp-bp.md §25's instruction not to recommend features for novelty.

---

## E. Memory Findings

**Allocations found on the per-token/per-tick hot path:**
1. `Tick::seqs.assign(n, {})` / `Tick::out.assign(n, {})` in `speculative::draft()` — destroys and rebuilds every internal per-sequence vector every tick, rather than resizing-and-clearing in place (P2, decode-tick slice F1).
2. `Qwen35::Impl::head()` allocates a fresh byte buffer and a fresh `ArgmaxResult` vector on every call — called once per trunk forward and once per MTP forward per tick (P2, F9).
3. `Qwen35::Impl::make_acts()` allocates ~20 backend buffers fresh on every forward call (trunk + every MTP draft depth) — the single largest source of repeated per-tick allocation in the model's forward path if the backend's `allocate()` isn't itself pooled (P2, F10).
4. Vulkan's `scratch()`/`stage()`/`add_download()` — a fresh `vkAllocateMemory` per op needing workspace/staging, every layer, every token (P1, promoted to §C).
5. HIP's `HipStream::scratch()` — analogous fresh `hipMalloc` per op call (P2, but HIP is compile-checked-only, unconfirmed on real hardware).

**Ownership/ zero-copy:**
- Given no discrete VRAM on the EVO-X2, the Vulkan backend's memory-type preference order (`device_info.cpp::choose_memory_type`) explicitly prefers a non-host-visible `DeviceLocal` type ahead of a combined `DEVICE_LOCAL|HOST_VISIBLE` type when both exist — forgoing a likely zero-copy (mapped, no staging round-trip) path that a unified-memory APU may actually offer (P2, §C-row "Vulkan barrier" area; detailed in the Vulkan slice's finding #5). This is flagged as **Requires-measurement on real hardware to confirm** whether the EVO-X2 exposes such a heap, but the code's current preference order forecloses the option regardless.
- RAII/ownership discipline is otherwise clean: no use-after-free, no missing fence-waits before buffer destruction, in any of the three backends.

**Cache/data-layout:**
- `backends/cpu/matmul.cpp`'s dequantization path materializes full float rows into heap scratch instead of using the already-implemented fused, cache-resident `vec_dot_row()` in `quant.cpp` — a real duplication where the hot path uses the less cache-friendly of two existing strategies (§C-row 2).
- `ggml_tables.h`'s lookup tables (≤2KB) are small enough to be L1-resident; no cache concerns found there.
- Unicode classification table (`unicode_data.inc`) is a flat, O(1)-lookup byte array — no cache concerns.

**Positive findings (explicitly checked, no defect):** per-thread scratch vectors in `gated_delta_rule.cpp`/`attention.cpp`/`ops_basic.cpp`'s `parallel_for` lambdas are allocated once per thread-task, not once per token — the correct pattern, confirmed by reading the loop nesting.

---

## F. CPU Findings

- **Hot loops:** `matmul`, `attention_gqa`, `gated_delta_rule_{recurrent,chunked}`, `causal_conv1d_silu`, `rms_norm`/`gated_rms_norm`/`l2_norm_heads`, `partial_rope_neox` are confirmed per-token, per-layer hot paths (CPU slice's hot-path list, §below).
- **Vectorization blocker:** missing `__restrict` on `detail::dot()` and its callers (§C, §D).
- **Double-precision transcendentals on a per-token path:** `partial_rope_neox` promotes to `double` for `cos`/`sin` inside the RoPE hot loop — likely an intentional HF-parity tradeoff (comment says so), flagged for awareness (P3), not urged as a blind change.
- **Serial pre-pass ahead of parallel work:** `gated_delta_rule_chunked`'s NaN/validity check over the full `g` matrix runs single-threaded before the parallelized main computation, during which all other cores are idle (P3) — small relative to the O(n_tok·n_v_heads·d_k·d_v) main computation, but structurally wasteful.
- **Algorithmic:** the decode-tick's O(n²) sequence-aliasing check (§C-row 1) is the standout algorithmic finding of the entire review.
- **Branch/redundant work:** `matmul.cpp`'s `argmax()` does two separate O(V) passes (NaN check, then max-finding) where one would do (P3, likely negligible next to the surrounding matmuls).
- **Runtime modulo in a small, bounded ring buffer:** `conv1d.cpp`'s causal ring-buffer index computes `% hist` per tap per token where `hist` is tiny and bounded — a manual wrap would be cheaper (P3).

---

## G. Concurrency Findings

- **Thread pool dispatch overhead (P1, §C):** every kernel dispatch pays a full mutex + `condition_variable_any`-with-`stop_token` wake round trip, even for small jobs; workers that wake to discover `index >= job.parts` do so anyway (`notify_all()` unconditional). This is the CPU backend's single highest-priority structural concern precisely because it multiplies across the highest call-frequency path in the runtime.
- **Vulkan submission model (P2):** one command buffer accumulates an entire per-token forward pass, then the host blocks fully on a fence before starting the next token's recording — no CPU/GPU overlap across steps, no double-buffering of command-buffer recording.
- **Vulkan barrier granularity (P2, §C):** unconditional whole-pipeline barrier between every pair of dispatches, no per-buffer dependency tracking, even between provably disjoint ops.
- **HIP stream/wait coupling (P0-labeled-as-design-risk in the HIP slice, not a code defect):** correctness of async batching depends entirely on caller discipline (how often `wait()` is invoked) — nothing in the HIP backend itself enforces or documents the expected granularity, so a caller invoking `wait()` per-op instead of per-tick would silently destroy the async pipeline. Recommended: assert/document the expected `wait()` granularity.
- **HIP host-emulation path (structural risk, not a defect):** `emu::run_grid`/`HostExec::phase` are fully serial nested loops with zero parallelism — an intentional, documented correctness reference, but since HIP has never run on real ROCm hardware in this project, this emulation path is the *only* one that has ever executed; any performance intuition formed from running it would be very likely wrong. This is a project-risk flag, not a code fix.
- **No false sharing found** in the CPU thread pool's shared counters/queue metadata during this review (flagged as checked, not found).

---

## H. Build / Compiler Findings

- **Current compiler/standard:** C++23 required, no GNU extensions; default build type `RelWithDebInfo`.
- **LTO:** `HALO_ENABLE_LTO` OFF by default, no documented rationale (§C-row).
- **PGO:** entirely absent from the build, despite the profiling harness (`src/profiling/**`) already providing exactly the curated, warmup-disciplined training workloads PGO needs (§C-row, §I) — flagged P1 as a real, low-effort opportunity given the infrastructure already exists.
- **Architecture tuning:** `HALO_MARCH`/`HALO_NATIVE` exist and are the correct mechanism, but no CMake code names `znver5` (the actual EVO-X2 target) explicitly, and `-march=native` on that hardware is never verified to be accepted by the pinned toolchain — a silent fallback/failure risk, not currently guarded against.
- **`-ffp-contract=off`** is applied to more modules than the bit-exactness rationale (CPU reference oracle for differential testing) actually requires — likely leaving real FMA throughput on the table in `halo_models`/`halo_speculative`/`halo_tensor`/`halo_backend` if they don't do their own float math outside the abstract op interface (§C-row, needs a code audit to confirm applicability).
- **Warnings (`-Wall -Wextra -Wpedantic ... -Werror`):** no evidence this set masks or blocks any optimization-friendly pattern; a documented per-build escape hatch (`HALO_WERROR=OFF`) already exists for toolchain drift. No finding here.
- **Visibility:** hidden by default, correctly configured; doesn't substitute for LTO's cross-TU inlining but isn't a defect on its own.
- **The build's only benchmark measures the wrong path** (§C-row, §I) — this is as much a build/tooling finding as a benchmarking one, since it means the build produces no artifact capable of validating any of the above tuning decisions on the production (Vulkan/HIP) path.

---

## I. Benchmarking Plan

| Benchmark | Input | Metric | Expected bottleneck | Tool | Baseline | Optimization experiment | Success criteria |
|---|---|---|---|---|---|---|---|
| Vulkan matmul/attention/GDN op bench (**does not exist — must be created**) | Representative Qwen3.8 layer shapes, decode (rows=1) and prefill (rows>1) | Wall-clock per dispatch, dispatch count, host CPU time in `Stream::dispatch` | Dispatch/allocation overhead (per §C findings) vs actual GPU compute | Custom bench mirroring `bench_gdn.cpp`'s structure, run on real EVO-X2 | None yet — first measurement | Dispatch overhead measured; is it <5%, ~50%, or dominant, of total per-token time? |
| HIP op bench (**does not exist — must be created, only meaningful once real ROCm hardware is available**) | Same op set as above | Same metrics | Unknown — HIP has never run on real hardware | `rocprofv2` | None | Establish whether HIP is even a viable path before investing further optimization there |
| Full decode-tick end-to-end (tokens/sec, TTFT) | `bench/workloads/agent_replay.json`, `concurrency_4agents.json`, `long_context.json` via `src/profiling/suite.cpp` (harness exists) | tokens/sec, time-to-first-token, p50/p95/p99 latency | Ambiguous — this is exactly what needs measuring first | The existing `src/profiling` suite | Confirm it's run as a repeatable regression gate, not only ad hoc | Establishes the actual current baseline the rest of this review's findings should be weighed against |
| CPU-backend matmul/attention microbench (extend existing pattern) | Same shapes as `bench_gdn.cpp` | ns/call, allocations/call | Dequant duplication (§C-row 2), thread-pool dispatch overhead (§C-row) | Extend `bench_gdn.cpp`'s pattern to `matmul.cpp`/`attention.cpp` | None | Confirms whether the dequant-per-decode-token finding is measurable |
| Device (GPU-side) memory bandwidth probe (**does not exist**) | A compute-shader read/write/copy sweep, analogous to `hardware/bandwidth.cpp`'s host-side prober | GB/s for Vram/Gtt/Pinned tiers | Fills a real gap: only host bandwidth is currently measured and fed to the autotune cost model | Custom Vulkan compute shader | None | Autotune's cost model gets real GPU-side bandwidth input instead of none |
| LTO A/B comparison | `bench_gdn.cpp` (until a better bench exists) built with `-DHALO_ENABLE_LTO=ON` vs `OFF` | ns/call | Cross-module inlining (§C-row, §H) | Standard CMake rebuild + bench | LTO OFF (current default) | Measurable ns/call delta, or none — either result resolves the open question |

**Existing bench honesty note:** `bench_gdn.cpp` already self-disclaims ("NOT a HALO performance claim... typically the dev laptop / WSL2") and this review agrees with that disclaimer — it should not be treated as informative about production (Vulkan/HIP) performance, and no number from it should be extrapolated to a broader claim.

---

## J. Profiling Plan

All commands below target AMD hardware (Zen 5 + RDNA-class iGPU), not generic/NVIDIA-flavored tooling, and are meant to be run once real EVO-X2 access is available:

```sh
# CPU hot-path instruction/cache profiling (works today, even on the WSL2 dev host)
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,\
L1-dcache-load-misses,LLC-load-misses -- ./build/backends/cpu/bench/halo_bench_cpu_gdn 8 512
perf record -F 999 -g -- ./build/backends/cpu/bench/halo_bench_cpu_gdn 8 512 && perf report

# AMD-specific uncore/L3/DRAM-controller events (enumerate exact event names on the real EVO-X2 first —
# they vary by microarchitecture revision):
perf list | grep -i amd_l3
perf list | grep -i amd_df
perf stat -e amd_l3/l3_request_g1.caching_l3_cache_accesses/ -- <bench>

# Allocation profiling (nothing in src/profiling currently intercepts malloc — see §H/§C)
heaptrack ./build/backends/cpu/bench/halo_bench_cpu_gdn 8 512
heaptrack_gui heaptrack.halo_bench_cpu_gdn.<pid>.zst

# Independent cross-check of hardware/bandwidth.cpp's own host-bandwidth measurement
mlc            # Intel Memory Latency Checker — cross-vendor, useful as an independent sanity check
stream         # STREAM benchmark, compare against tools/halo/tune.cpp's in-repo measurement

# Vulkan path (once a Vulkan bench target exists — currently none does)
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./build/.../vulkan_bench   # catch API-misuse-driven stalls
# RenderDoc: single-frame/single-dispatch capture-and-replay of backends/vulkan/shaders/{matmul,attention}/*
#   to inspect real shader occupancy and register/LDS usage on the RDNA iGPU
# RGP (Radeon GPU Profiler), if available for this RDNA-class iGPU: shader-level occupancy, wavefront stalls
#   — this is the AMD-native equivalent of Nsight Compute

# ROCm/HIP path (only meaningful once real ROCm hardware is confirmed available — currently unconfirmed)
rocprofv2 --hip-trace --hsa-trace -o hip_profile ./build/backends/hip/bench/<target-once-it-exists>
rocprofv2 --pmc SQ_INSTS_VALU,TCC_HIT,TCC_MISS ...   # ALU utilization, L2/TCC cache hit-rate
```

**Priority order for profiling investment:** (1) full decode-tick end-to-end via the existing `src/profiling` suite, to establish an actual baseline before anything else is measured in isolation; (2) Vulkan dispatch-overhead microbench, since §C's Vulkan findings are the largest unmeasured P1 cluster; (3) CPU matmul dequant A/B, since it's the cheapest to measure and has a concrete, bounded fix ready.

---

## K. Recommended Refactoring Roadmap

**Phase 1 — Correctness / Safety:** None required from this review — no correctness defects were found (that's the other review's domain). Only process-level flags: document the HIP `wait()` granularity contract (§G), and add a guard preventing perf claims sourced from HIP's host-emulation path (§G) from being treated as representative of real hardware.

**Phase 2 — Algorithmic Improvements** *(cpp-bp.md: prioritize above micro-optimization)*:
1. Fix `Qwen35::forward`/`mtp_forward`'s O(n²) sequence-aliasing check → O(n) (§C-row 1, **P0**).
2. Fix `SequenceKv::view()`'s per-call, context-length-scaling pointer-table rebuild → cache with invalidation (§C-row, **P1**).
3. Fix `Sampler::chain::draw()`'s unconditional O(V log V) sort → O(V) weighted draw (§C-row, **P1**).
4. Fix `build_chat_prompt`'s 8× non-incremental prefix re-render/re-tokenize (§C-row, **P1**).

**Phase 3 — Memory Optimization:**
1. Route CPU decode-time matmul through the existing fused `vec_dot_row()` instead of materializing float rows (§C-row, **P1**).
2. Pool/reuse Vulkan scratch/staging buffers instead of allocating fresh per op (§C-row, **P1**).
3. Pool/reuse HIP scratch allocations, once real-hardware relevance is confirmed (**P2**).
4. Hoist `Qwen35::Impl::head()`'s and `make_acts()`'s per-call allocations into reusable scratch state (**P2**).
5. Give `Tick`/`speculative::draft()` a `reset(n)` that resizes-in-place instead of `assign(n, {})` (**P2**).

**Phase 4 — CPU Optimization:**
1. Add `__restrict` to `detail::dot()` and its hot-path callers (§C-row, **P3**, broad blast radius but individually low-confidence magnitude).
2. Fold `argmax()`'s two O(V) passes into one; parallelize or fold `gated_delta_rule_chunked`'s serial validation pre-pass into the parallel loop (**P3**).

**Phase 5 — Concurrency Optimization:**
1. Thread pool: spin-then-block wait, cap wakeups to parts actually needed (§C-row, **P1**).
2. Vulkan: buffer-scoped barrier tracking instead of unconditional whole-pipeline barriers; consider double-buffered command-buffer recording across decode steps (**P2** each).

**Phase 6 — Compiler / Build Optimization:**
1. Resolve the LTO-off default: either enable it or document why not (§C-row, **P2**).
2. Audit and narrow `-ffp-contract=off`'s scope to only the modules that need bit-exactness (§C-row, **P1**).
3. Wire PGO using the already-existing profiling-suite workloads as the training corpus (§H/§I, **P1** — high-leverage because the hard infrastructure already exists).
4. Add explicit `znver5` tuning path + a compiler-support check for `-march=native` on the real target (**P2**).
5. Add missing benchmarks: Vulkan op bench, HIP op bench, GPU-side bandwidth probe (§I, **P0-as-evidence-gap**, since nothing downstream can be validated without these).

**Phase 7 — Micro-Optimization:** Not recommended yet — Phases 2–5 contain unresolved, higher-confidence findings; per cpp-bp.md §K, Phase 7 work should wait until those are addressed and measured.

---

## TOP 10 ACTIONS

1. **`src/models/qwen35.cpp:704-707,827-829`** (`Qwen35::forward`/`mtp_forward`) — Current: O(n²) nested loop checking KV/GDN pointer aliasing across the batch, every tick. Problem: unconditional quadratic-in-batch-size cost on the single hottest per-tick loop. Proposed: single-pass insertion into a hash-set (or sort-and-adjacent-check) for O(n)/O(n log n) duplicate detection. Why faster: removes a certain quadratic growth curve. Risk: low (behavior-preserving refactor of a validation check). Benchmark: time `forward()` at increasing `max_sequences` before/after. Implement now or after profiling: **now** — the complexity fact alone justifies it regardless of measurement.

2. **`src/sampling/sampler.cpp:202-219`** (`chain::draw`) — Current: unconditional full sort of the candidate set before a weighted draw. Problem: O(V log V) for an O(V) task, V≈248K. Proposed: single-pass CDF walk without sorting; apply top_k as a default bound before top_p/typical. Why faster: removes the sort entirely for the common non-greedy-sampling case. Risk: low-medium (must preserve RNG-draw semantics/determinism guarantees — verify against existing sampler tests). Benchmark: per-token sampler time under temperature-only config. Implement now or after profiling: **now** for the algorithmic fix; validate magnitude with a microbench alongside.

3. **`src/kv_cache/paged_kv.cpp:197-206`** (`SequenceKv::view`) — Current: rebuilds the block-pointer table from `blocks_` on every call. Problem: O(context_length/block_tokens) redundant work per layer per tick, the one place cost provably grows with context length. Proposed: cache the table, invalidate only when `blocks_` changes. Why faster: removes O(n) redundant reconstruction most calls don't need. Risk: low (cache-invalidation correctness needs a test covering block growth/eviction). Benchmark: attention-layer time at long context lengths before/after. Implement now or after profiling: **now**, given it's the only context-length-scaling finding in this slice.

4. **`backends/cpu/matmul.cpp:37-59`** — Current: full float-row dequant materialization on every decode-step matmul. Problem: doubles per-token FLOP-adjacent work for quantized layers during single-token decode. Proposed: route the `rows==1` case through `quant.cpp`'s existing `vec_dot_row()`. Why faster: removes a full read/write pass over each dequantized row, uses the already-implemented cache-resident fused path. Risk: low (the fused function already exists and is presumably tested elsewhere). Benchmark: `bench_gdn.cpp`-style extension for matmul, decode-shape only. Implement now or after profiling: **measure first** — confirm the fused path's behavior matches exactly before switching the hot path onto it.

5. **`src/api/prompt.cpp:244-270`** (M10 checkpoint hints) — Current: up to 8 non-incremental prefix re-renders + re-tokenizations per multi-turn request. Problem: up to 8× the primary render/tokenize cost, unconditionally. Proposed: incremental prefix rendering, or derive hints from already-computed token counts, or reduce/configure `kMaxHintBoundaries`. Why faster: removes redundant O(document length) work × 8. Risk: medium (checkpoint-hint semantics must be preserved exactly for whatever consumes them). Benchmark: request latency at increasing message counts, before/after. Implement now or after profiling: **measure first** — confirm actual wall-clock share before committing to a specific redesign.

6. **`backends/vulkan/src/stream.cpp:140-199`** (`Stream::dispatch`) — Current: fresh descriptor-set pool allocation + `vkUpdateDescriptorSets` every dispatch. Problem: real per-dispatch driver overhead, tens of times per token. Proposed: `VK_KHR_push_descriptor`/descriptor-update templates. Why faster: eliminates pool allocation and the update call per dispatch. Risk: medium (requires the extension to be available/enabled; needs real-hardware validation). Benchmark: per-dispatch host CPU time before/after, on real EVO-X2. Implement now or after profiling: **measure first** — no GPU available on this review's host to confirm the extension's availability/behavior.

7. **`src/backend/vulkan_adapter.cpp:114-130`** (`scratch`/`stage`/`add_download`) — Current: fresh `vkAllocateMemory` per op needing workspace/staging. Problem: one of the most expensive Vulkan driver calls, paid per op per layer per token. Proposed: pre-sized, reused scratch/staging buffer pool, reset (not destroyed) per step. Why faster: removes allocation/deallocation churn entirely for the steady-state case. Risk: medium (sizing the pool to the worst-case workspace need without over-provisioning). Benchmark: per-token host-side Vulkan CPU time before/after. Implement now or after profiling: **measure first**.

8. **`backends/cpu/thread_pool.cpp:51-93`** — Current: full mutex+condvar wake round trip per dispatch, `notify_all()` regardless of parts needed. Problem: OS wake latency paid on every one of dozens of dispatches per token. Proposed: spin-then-block wait; cap wakeups to parts actually needed. Why faster: removes wasted wake/sleep cycles. Risk: low-medium (spin duration needs tuning to avoid burning CPU on genuinely long waits). Benchmark: dispatch-to-first-work latency before/after, across a range of job sizes. Implement now or after profiling: **measure first** — the mechanism is a stated systems fact, but spin-tuning needs real hardware.

9. **CMake: PGO wiring** — Current: no `-fprofile-generate`/`-fprofile-use` anywhere; `src/profiling`'s workload/warmup/stability infrastructure already exists and is unused for this purpose. Problem: paying for -O2/-O3 without the additional inlining/branch-layout/hot-cold-splitting benefit PGO would add. Proposed: `HALO_ENABLE_PGO` option + a two-phase build script (instrument → run `bench_gdn` + `bench/workloads/*.json` → merge profile → rebuild `-fprofile-use`). Why faster: PGO's benefit is well-established broadly; this project already has the hard part (a representative, stability-checked training corpus) built. Risk: low (additive build option, doesn't change default behavior). Benchmark: A/B build comparison. Implement now or after profiling: **measure first** (an A/B build comparison is itself the validation step).

10. **CMake: `-ffp-contract=off` scope audit** — Current: applied to `halo_models`/`halo_speculative`/`halo_tensor`/`halo_backend`, not just the CPU reference oracle. Problem: potentially disabling FMA fusion (2 ports on Zen 5) in modules that don't need bit-exactness. Proposed: audit for direct float math outside the abstract op interface in those modules; narrow the flag's scope. Why faster: restores FMA fusion where safe. Risk: low (additive scoping change, easy to revert per-target). Benchmark: A/B on any affected hot loop found during the audit. Implement now or after profiling: **measure first** — requires the code audit before any flag change, then a benchmark to confirm.

---

## Implementation Candidates

**SAFE TO IMPLEMENT NOW** *(behavior-preserving algorithmic fixes with clear, bounded risk; correctness can be verified by existing/extended unit tests without needing real GPU hardware)*:
- Fix the O(n²) sequence-aliasing check in `qwen35.cpp` (Top 10 #1).
- Cache `SequenceKv::view()`'s block-pointer table with invalidation (Top 10 #3).
- Give `Tick`/`speculative::draft()` a `reset(n)` instead of `assign(n, {})` (§K Phase 3).
- Add `__restrict` to `detail::dot()` and its hot-path callers (§K Phase 4) — purely additive, no behavior change.
- Fold `argmax()`'s two O(V) passes into one (§K Phase 4).
- Document the HIP `wait()` granularity contract; guard against treating HIP-emulation timings as real-hardware-representative (§K Phase 1).
- Document (or fix) the `HALO_ENABLE_LTO` default's rationale (§K Phase 6, item 1) — at minimum, the documentation half is safe now.

**MEASURE BEFORE IMPLEMENTING** *(real fixes exist, but magnitude, extension-availability, or exact semantics need real-hardware/profiling confirmation first)*:
- Route CPU decode matmul through `vec_dot_row()` (Top 10 #4).
- Rework `build_chat_prompt`'s checkpoint-hint loop (Top 10 #5).
- Vulkan `VK_KHR_push_descriptor` adoption (Top 10 #6).
- Vulkan scratch/staging buffer pooling (Top 10 #7).
- Thread-pool spin-then-block tuning (Top 10 #8).
- PGO wiring (Top 10 #9) — the A/B build comparison IS the measurement step.
- `-ffp-contract=off` scope narrowing (Top 10 #10) — requires a code audit before any change.
- Fix `Sampler::chain::draw()`'s unconditional sort — the algorithmic direction is clear (Top 10 #2), but changing sampling code requires care around determinism/reproducibility guarantees that existing tests may depend on; verify those first.
- Vulkan memory-type preference (zero-copy on unified memory) — requires confirming the EVO-X2 actually exposes a combined `DEVICE_LOCAL|HOST_VISIBLE` heap before changing the preference order.

**ARCHITECTURAL CHANGE — DISCUSS FIRST:**
- Adding Vulkan and HIP benchmark suites (§I) — not risky, but represents new ongoing infrastructure/maintenance surface, worth agreeing on structure/ownership before building.
- Adding a GPU-side memory-bandwidth probe feeding the autotune cost model (§I) — touches the autotune/hardware-discovery contract; worth a short design note given `MemoryTier` already declares tiers this would newly populate.
- Double-buffering Vulkan command-buffer recording across decode steps (§K Phase 5) — a real architectural change to the stream/submission model, not a local fix; needs its own design discussion given fence/descriptor-pool lifetime implications.
- Znver5-specific tuning path in the build (§K Phase 6, item 4) — small in isolation, but touches the cross-compilation/native-build split documented in `docs/evox2.md`; coordinate with whoever owns that build script.

---

## Final Adversarial Pass (§49)

**"If I were trying to make this application substantially faster without changing its external behavior, what have I NOT investigated yet?"**

1. **The decode tick's actual wall-clock breakdown is unknown.** Every finding in this review is a structural/algorithmic argument; none is backed by a profile showing where time in one real decode tick actually goes. Without that, it's possible the O(n²) sequence check (§C-row 1) is genuinely dominant, or genuinely noise next to matmul FLOPs — the review cannot tell, and says so throughout. This is the single biggest gap, and §I's first-priority benchmark (full decode-tick end-to-end) exists specifically to close it.

2. **Whether the Vulkan path is even the one actually serving production traffic wasn't verified in this review.** `HALO_BUILD_VULKAN` defaults ON and `HALO_BUILD_HIP` defaults OFF per the build-system slice, suggesting Vulkan is the intended primary path — but this review didn't trace the runtime's backend-selection logic (which adapter gets chosen at startup, and under what config) to confirm that assumption. If HIP were somehow the default in some deployment, this review's weighting (heavy Vulkan focus, lighter HIP treatment given its unconfirmed-on-real-hardware status) could be wrong for that deployment.

3. **Model-loading/weight-conversion cost was not reviewed at all.** This review's six slices covered the decode/inference hot path exhaustively but never looked at `src/model/{gguf,mapped_file,model}.cpp` beyond confirming (in the decode-tick slice) that they're load-time-only and thus out of scope for per-token hot-path priority. If cold-start/model-load latency matters to this project's actual use case (e.g., frequent process restarts, multi-model serving), that's an entirely unreviewed dimension.

4. **Cross-request/multi-tenant memory pressure wasn't modeled.** The review found several small per-tick/per-request allocations (§E), but didn't examine what happens under the concurrency workload (`concurrency_4agents.json` exists exactly for this) — whether allocator contention across concurrent sequences compounds any single-sequence finding into something worse at scale. The build/profiling slice noted the profiling harness has no allocation-hook or `perf_event_open` integration (§H/§I) — this is a real blind spot the harness itself can't currently see into.

5. **The GPU shader source itself (beyond `attention.comp`/`eltwise.comp`/a couple of matvec shaders) was only sampled, not fully read.** The Vulkan slice explicitly reviewed "representative shaders," not the complete shader directory. GEMM/GEMV tiling strategy, workgroup-size choices across the *full* shader set, and shared-memory (LDS) usage patterns in the less-sampled shaders remain a gap — the P2 finding about single-workgroup dispatch during decode (rows=1) is confirmed structurally, but whether other shaders have their own occupancy issues wasn't exhaustively checked.

6. **No investigation of whether inter-request batching (increasing effective `rows` per matmul call during concurrent decode) could turn the "decode dequant doubling" finding (§C-row 2) moot.** If the engine already batches multiple concurrent sequences' single-token steps into one `matmul()` call with `rows > 1`, the "reuse across x rows" optimization the CPU slice found ineffective for `rows==1` might already be effective in practice under real concurrent load. This review's decode-tick slice discusses per-sequence forward calls but didn't fully resolve whether trunk `forward()` is called once per tick for the *whole batch* (making rows = batch size, moot-ing this finding under concurrency) or per-sequence. **This is the one place two slices' findings may be in tension and deserves a direct code check before acting on Top 10 #4.**

**Answer:** the review is structurally thorough but evidence-starved by construction (no GPU on this host) — the single highest-leverage next step is not another round of code reading, it's standing up the full-decode-tick benchmark (§I, priority 1) on real EVO-X2 hardware, because that one measurement would resolve items 1, 2, and 6 above simultaneously and would tell this review's own priority ordering whether it's actually right.
