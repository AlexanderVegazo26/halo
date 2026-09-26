# EVO-X2 resume context — adversarial review remediation

This records the exact state of the 2026-09-26 adversarial review's remediation
(`ADVERSARIAL_REVIEW.md`) as of the last commit pushed from the dev host (no AMD GPU, D-001).
Everything below that needed real hardware, an offline reference run, or a cross-backend
change too risky to make blind was left for whoever has the EVO-X2 in hand. Read this before
starting new work so you don't re-discover the same blockers.

**Repo:** `git@github.com:AlexanderVegazo26/halo.git`, branch `main`.
**Last dev-host commit at hand-off:** `6e1a0c5` ("api: I1 install an error logger...").

## What's already fixed and pushed (don't redo this)

H2, M1, M3, M4, M6, M7, M8, M9, L1, L2, L3, L4, L5, L6, L7, L8, L9, I1, plus the S-46
symlinked-profile-db fix and the Vulkan backend adapter WIP (WS-BI-5), all verified passing
in WSL (CPU reference + Vulkan-on-lavapipe correctness only — no GPU was ever involved).
`git log --oneline` from `2835e11` onward has the individual commits and their verification
notes. The README and `docs/` were also audited/rewritten for staleness; no code content there.

## What's left, and why it needed to wait for you

### H1 — full per-sequence isolation (HIGH, foundation now landed, engine-side fix still open)
M1 (commit-after-head reorder in `qwen35.cpp`) is done: a NaN logit or allocation failure in
the LM head no longer leaves KV/GDN state partially advanced.

Beyond that, the **wire-format half of H1 is now also done**: `backend::decode_argmax`
(`src/backend/backend.cpp`) and the CPU backend's `lm_head()` (`src/backend/cpu_backend.cpp`)
no longer throw for the whole batch on a NaN logit — a NaN now poisons only its own row
(`ArgmaxResult{-1, NaN}`), and the other rows in the same batched call decode normally. This
was safe to do without a device: `qwen35.cpp`'s `head()` is backend-agnostic and only ever
consumes the shared `backend::decode_argmax`, and Vulkan/HIP's own kernels already report a
NaN per-row at the shader/kernel level (each GPU invocation only ever knows about its own row
— they structurally couldn't throw for a whole batch). So neither `backends/vulkan/`,
`backends/hip/`, nor their own `halo::vulkan::decode_argmax` / `halo::hip::decode_argmax`
helpers needed to change; the two `throw`s that mattered were both in code shared across all
three backends' adapter path, and both are now fixed. Verified: `test_backend` (12/12,
including a new regression proving row 1's NaN doesn't disturb rows 0/2's results),
`test_models` (25/25), `test_speculative` (18/18), `test_runtime` (32/32),
`test_vulkan_backend` (10/10), `test_vulkan_forward` (8/8, 2 pre-existing unrelated skips).

**What's still open, and why it stopped there:** `speculative.cpp`'s `verify()` currently
converts a poisoned row back into a **thrown exception** at the point it would otherwise
become token id `-1` (`HALO_CHECK(std::none_of(o.targets...))`), which still lets
`CpuEngine::run()`'s tick-wide catch call `fail_all` — i.e. **the user-visible failure domain
is unchanged from before this fix**: one bad sequence still fails every sequence in the tick.
This was a deliberate, correctness-first stopping point: finishing per-sequence isolation
needs `StepOutput`/`Tick::Seq` (`include/halo/speculative/speculative.h`) to carry a
per-sequence error/poisoned flag through `verify()`/`commit()`, and `engine.cpp` to read it
and call `retire(..., FinishReason::Error, ...)` for just that one `Active` instead of
`fail_all` for the whole tick — a real, non-trivial change to the rollback/commit invariants
this codebase is careful about, not a place to rush.

**Remaining plan:**
1. Add an error/poisoned signal to `StepOutput` (or a parallel per-sequence status the caller
   already has access to after `verify()`).
2. In `speculative.cpp::verify()`, replace the `HALO_CHECK(std::none_of(...))` guard with
   setting that signal instead of throwing.
3. In `engine.cpp`, after calling `verify()`, iterate sequences and call
   `retire(a, FinishReason::Error, ...)` for any poisoned one, then continue committing the
   rest of the tick normally instead of the tick-wide catch firing.
4. Do the same for `bad_alloc` in the head-section allocations (M1's other named producer) —
   that one isn't a per-row wire signal, so it needs its own handling (likely: catch it at the
   same point, treat as a poisoned row for every sequence sharing that batched call, since a
   host allocation failure isn't attributable to one row the way a NaN is).
5. Add a fault hook that injects a NaN mid-forward (today's four hooks fire before-step/
   output/retire, never inside the forward — this is why the test suite still can't exercise
   any of this end-to-end).
6. Re-run `test_models`, `test_runtime`, `test_speculative` after each step.

### M2 — fused-argmax vocab clamp (MEDIUM, partially fixed)
The sampler-side clamp (unstructured sampling excludes padded LM-head rows) is done. The
fused-greedy-argmax path (`qwen35.cpp:head()` calling `be->lm_head`) still computes argmax
over the full padded vocab in the backend kernel itself, which needs the same kind of
Backend-interface change as H1 (a vocab-limit parameter threaded through `GemvArgs`/
`LmHeadArgs` and all three kernels) — bundle it with the H1 work above since it touches the
same call sites.

### M5 — un-circularize the quantized-forward differential (MEDIUM)
Needs one offline run: dequantize the canonical pack with `gguf-py`, feed the f32 result
through `transformers`, and save that as a new golden. This can't happen on the dev host (no
real 27B weights there) or, probably, on the EVO-X2 either unless it also has the Python/
transformers toolchain and enough time for a 27B forward pass. If you have that, the target
is `tests/unit/models/test_qwen35_golden.cpp:623,712`
(`BitIdenticalToF32ForwardOverDequantizedWeights`) — its reference path currently shares the
implementation's own `dequantize_row`, so a systematic dequant bug would pass silently.

### M10 — checkpoint_hints producer + planner/engine reconciliation (MEDIUM)
Two independent pieces:
1. **Producer**: `ChatPrompt` needs to report message-boundary token offsets so
   `build_chat_prompt` (`src/api/prompt.cpp`) can populate `GenerateRequest::checkpoint_hints`.
   This needs changes in `src/template/chat_template.cpp`, which another workstream's own
   handoff notes (`docs/dev/handoff/prompts/resume_I.txt`) said not to touch while its own
   work was in flight. Check whether that constraint still applies before touching it.
2. **Reconciliation**: on a host-only machine, the planner reports prefix checkpoints as
   disabled (`src/memory/planner.cpp:352-361`) while the engine silently budgets them on the
   host anyway (`src/runtime/engine.cpp:302-316`), so `halo inspect`'s memory report
   understates the real footprint. On the EVO-X2 a GPU tier will actually be discovered, which
   may make this whole code path moot (checkpoints would then genuinely live in the GPU pool,
   matching D-013) — re-check whether this finding still applies before fixing it.

### M11 — real hardware validation
Everything in `docs/evox2.md` that hasn't run yet: the smoke harness beyond 3×32-token
prompts, chunked-prefill at scale, KV paging under real pressure, prefix cache at scale,
non-greedy sampling, structured output, concurrent sequences, and the full API surface, all
at 27B. This is the actual point of you having the hardware — just run it and see what
breaks.

### Infrastructure items that were always going to wait for real hardware
- **TRD §68 Phase 0** — the roofline/bandwidth measurement everything else in DECISIONS.md
  (D-011, D-014) is provisional on. Run this first; it changes what "good" looks like for
  everything downstream.
- **HIP validation** — flip from emulation to a real device. Re-check `kDevRel` (currently a
  guess, `hip_test_util.h:94-98`) against measured tolerances, and re-audit
  `test_hip_backend.cpp:1046`'s negative tests, which self-disable once a device is present —
  make sure they're actually exercising something once that happens.
- **Vulkan validation against RADV** — everything so far only ran against lavapipe (a
  software/CPU Vulkan implementation used for correctness, never performance). Re-run the
  Vulkan suites against the real RADV driver.
- **GPU autotune profiles** — `halo tune` only searches CPU tunables today; bridge the GPU
  variant search once there's a real device to search against.
- **BI-6** — wire `CpuEngine` (or a generalized successor) to actually construct `Qwen35`
  over a `vulkan`/`hip` `Backend&` instead of hard-rejecting anything but `cpu`/`auto`
  (`engine.cpp:227-228`). Nothing above matters for real serving until this lands.

## Suggested order

1. BI-6 engine wiring first — nothing else is real until the engine can actually dispatch to
   a GPU backend.
2. TRD §68 Phase 0 roofline measurement — everything performance-related is guesswork until
   this exists.
3. H1 + M2 together (same call sites, same three-backend contract change) — do this once,
   with the real Vulkan/HIP devices available to test against, not blind.
4. Real Vulkan (RADV) and HIP (ROCm) validation passes.
5. M11's serving-surface validation at scale.
6. M5 (offline transformers run) and M10 (message-boundary hints / planner reconciliation) as
   time allows — both are real but neither blocks the others.

## How to verify anything you fix

Same convention used throughout this remediation: build a scoped subset with
`-DHALO_ONLY="a;b;c"` (see `CMakeLists.txt:24-27` for the module-name list), run the specific
test binaries touched, and only then commit by explicit path — never `git add -A` in this
tree (`memory.md`/`halo-orchestration-state.md` explains why: this has been a shared,
multi-agent working tree, and a blind `add -A` has swept in another agent's half-finished
work before).
