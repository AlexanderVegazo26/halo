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

### H1 — full per-sequence isolation (HIGH, NaN-logit path now fixed end to end; bad_alloc still open)
M1 (commit-after-head reorder in `qwen35.cpp`) is done: a NaN logit or allocation failure in
the LM head no longer leaves KV/GDN state partially advanced.

`backend::decode_argmax` (`src/backend/backend.cpp`) and the CPU backend's `lm_head()`
(`src/backend/cpu_backend.cpp`) no longer throw for the whole batch on a NaN logit — a NaN
poisons only its own row (`ArgmaxResult{-1, NaN}`), and the other rows in the same batched
call decode normally. This was safe to do without a device: `qwen35.cpp`'s `head()` is
backend-agnostic and only ever consumes the shared `backend::decode_argmax`, and Vulkan/HIP's
own kernels already report a NaN per-row at the shader/kernel level (each GPU invocation only
ever knows about its own row — they structurally couldn't throw for a whole batch). So neither
`backends/vulkan/`, `backends/hip/`, nor their own `halo::vulkan::decode_argmax` /
`halo::hip::decode_argmax` helpers needed to change.

**The engine-side half is now also done, for the NaN-logit producer specifically.**
`speculative.cpp`'s `verify()` no longer converts a poisoned row into a thrown exception; a
poisoned row's `-1` flows through `StepOutput::targets`/`tokens` untouched (that function runs
once for every sequence in the batch, so throwing there would still fail all of them
together). Instead, `CpuEngine::tick()`'s existing per-sequence output-phase isolation (review
R-1: "a failure from here on ends only this sequence") now explicitly checks for a poisoned
token id *before* it is ever fed forward, sampled around, or detokenized, and throws there —
inside the per-sequence `try` that already resets just that one slot and marks just that one
`Active` as `FinishReason::Error`. So a NaN logit in one sequence's row now fails only that
sequence; every other sequence in the same tick commits normally. Verified: `test_backend`
(12/12, including a new regression proving row 1's NaN doesn't disturb rows 0/2's results),
`test_models` (25/25), `test_speculative` (18/18), `test_runtime` (32/32, including the
pre-existing `Faults.OutputPhaseFailureOfOneSequenceSparesTheOthersInTheTick`, which is the
generic form of the isolation this now relies on), `test_vulkan_backend` (10/10),
`test_vulkan_forward` (8/8, 2 pre-existing unrelated skips).

**What's still open:** `Error(Memory)`/`std::bad_alloc` from the head-section allocations
(M1's other named producer) is not a per-row wire signal the way NaN is — it's a C++ exception
from a shared allocation for the whole batched call, not attributable to one row. It still
escapes `spec_->step()` uncaught (the tick's outer catch only handles `halo::Error` with
`ErrorCode::Memory`, and re-throws everything else, including `std::bad_alloc`), and still
hits `CpuEngine::run()`'s tick-wide `fail_all`. There is also no fault hook that can inject a
real NaN mid-forward to exercise this end-to-end automatically (today's four hooks fire
before-step/output/retire, never inside the forward) — the verification above is by
construction (backend-level NaN injection + the already-tested generic isolation mechanism),
not a full engine-level "two real sequences, one goes NaN, watch the other survive" test.

**Remaining plan:**
1. Give `bad_alloc` the same treatment: catch it where it occurs and decide whether it's
   attributable to one sequence or must poison every sequence in that batched call (probably
   the latter, since the allocation is shared) — either way, convert it into the same kind of
   per-sequence-checkable signal instead of letting it reach `fail_all`.
2. Add a real mid-forward fault hook (a hook the CPU backend's `lm_head()` can call to inject
   a NaN into one specific row) and use it to write the true end-to-end regression test: two
   concurrent sequences, one is made to go NaN, assert the other completes normally and the
   first retires as `FinishReason::Error`.
3. Re-run `test_models`, `test_runtime`, `test_speculative` after each step.

### M2 — fused-argmax vocab clamp (MEDIUM, now fully fixed on CPU)
Both halves are done. The sampler-side clamp (unstructured sampling excludes padded LM-head
rows) landed earlier. `LmHeadArgs` (`include/halo/backend/backend.h`) now has a `valid_rows`
field (0 = no clamp, the pre-M2 default); the CPU backend's `lm_head()` excludes rows at/after
it from the argmax while still reporting every row's raw value in the full logits output;
`Qwen35Options::valid_vocab` threads the tokenizer's real vocab size in from `engine.cpp`
(built before the model now, since the model needs this value and the tokenizer only depends
on GGUF metadata, not the model). Turned out this needed none of the cross-backend risk H1's
plan originally worried about — `GemvArgs`/`LmHeadArgs` in `include/halo/backend/backend.h`
are a *different type* from Vulkan's and HIP's own internal `GemvArgs`/`LmHeadArgs` (their
adapters translate between them), so adding a field with a zero-default only had to be
threaded through the CPU implementation to take effect; Vulkan/HIP's adapters simply don't
set it yet, meaning they're unaffected and still unclamped on GPU (their own follow-up, not
blocked on anything -- the same `valid_rows` field is already there for them to use whenever
their adapters are updated to read it). Verified: `test_backend` 13/13 (new regression:
a deliberately larger padded-row logit never wins argmax, and is still visible in the full
logits row), `test_models` 25/25, `test_speculative` 18/18, `test_runtime` 32/32.

### M5 — un-circularize the quantized-forward differential (MEDIUM)
Needs one offline run: dequantize the canonical pack with `gguf-py`, feed the f32 result
through `transformers`, and save that as a new golden. This can't happen on the dev host (no
real 27B weights there) or, probably, on the EVO-X2 either unless it also has the Python/
transformers toolchain and enough time for a 27B forward pass. If you have that, the target
is `tests/unit/models/test_qwen35_golden.cpp:623,712`
(`BitIdenticalToF32ForwardOverDequantizedWeights`) — its reference path currently shares the
implementation's own `dequantize_row`, so a systematic dequant bug would pass silently.

### M10 — checkpoint_hints producer (DONE) + planner/engine reconciliation (MEDIUM, still open)
**Producer: done.** `ChatPrompt` (`include/halo/api/prompt.h`) now carries `checkpoint_hints`;
`build_chat_prompt` (`src/api/prompt.cpp`) populates it by rendering each of the last 8
message-boundary prefixes (no generation prompt) and recording each one's token count, bounded
so the cost is O(8 * document length) regardless of conversation length (this codebase
exercises multi-thousand-message conversations in its own tests, so an unbounded O(messages)
version would have been a real, self-inflicted performance regression -- an earlier version of
this doc flagged that as the reason to hold off; it turned out bounding it was straightforward
once actually attempted). Some templates (Qwen's included) reject a prefix that has no user
message yet ("No user query found in messages"); that's caught per-boundary and the boundary
is just skipped, since a hint is optional metadata, not something worth failing the request
over. `server.cpp`'s `make_spec` now threads `prompt.checkpoint_hints` into
`GenerateRequest::checkpoint_hints`. Verified: `test_api` 90/90 (new regression test asserting
exact boundary token counts, monotonicity, and the 8-hint bound on a 12-message conversation,
plus confirming the real Qwen template's two existing injection-resistance tests still pass
now that prefix rendering happens on every multi-message chat request), `test_cli` 28/28,
`test_runtime` 32/32.

**Correction to an earlier version of this doc**: the producer was previously described as
blocked by `docs/dev/handoff/prompts/resume_I.txt` saying "do not edit src/template." That was
a misreading on my part -- re-read in full, that note is a narrow, already-resolved
instruction about one specific compile error in a since-completed, since-committed workstream
(WS-I), not a standing prohibition on this file. Worth noting in case the same misreading
happens again with some other historical handoff note: check what a constraint actually says
and whether its context has since resolved, rather than treating a quoted fragment as a
standing rule.

**Reconciliation (still open)**: on a host-only machine, the planner reports prefix
checkpoints as disabled (`src/memory/planner.cpp:352-361`) while the engine silently budgets
them on the host anyway (`src/runtime/engine.cpp:302-316`), so `halo inspect`'s memory report
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
   a GPU backend. While you're in there: have it also set `valid_vocab` (M2) and check
   whether Vulkan's/HIP's adapters need `valid_rows` wired through to their own kernels for
   parity with the CPU path (see M2 above — it's a small, already-proven-safe addition now).
2. TRD §68 Phase 0 roofline measurement — everything performance-related is guesswork until
   this exists.
3. H1's remaining `bad_alloc` producer, plus a real mid-forward fault hook so this whole area
   has an actual end-to-end regression test instead of the current layered verification (see
   H1 above for exactly what's left).
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
