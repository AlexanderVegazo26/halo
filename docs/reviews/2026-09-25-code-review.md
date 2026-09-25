# HALO v0.2 — Code Review, commits ca68dc9..HEAD (3c4c71c)

Reviewer: sdlc-suite:code-reviewer. This is a static review: nothing was built or run. Date: 2026-09-25.
- Tier 3: the range adds the Engine, speculative decoding, D-012 rollback, the D-013 prefix cache, the public HTTP API, and GPU backends.
- Only the committed range was reviewed. For files that have uncommitted changes (backends/hip/*, the hip tests, tools/halo/cli.cpp, src/profiling/suite.cpp) the HEAD version was read with `git show HEAD:`.
- The reviewer has no write tools, so the orchestrator saved this file from the reviewer's report.

## Summary

| Area | Verdict | Findings |
|---|---|---|
| runtime / speculative / state / kv_cache / models | Correct on the traced paths; failure blast radius and worker robustness need work | R-1, R-2, R-5, N-1, N-2, N-4 |
| api + tools/halo | Solid limits and parsing; cancellation and error mapping have gaps | R-3, R-4, N-3 |
| sampling | Approvable; tests use an independent validator | — |
| profiling + autotune | Approvable; the tuner has no consumer | N-5 |
| vulkan + hip | Approvable; tests are real differentials against halo::cpu | N-6 (doc/contract) |

**Traced and found correct:**
- **Speculative commit keeps the MTP invariant.** `mtp_kv.length + queue + 1 == L` and `last_hidden == h_{L-1}` hold for all four cases (decode flushed or not, first or later prefill chunk; speculative.cpp:329-341).
- **GDN slot selection matches D-012.** The kept slot is `slot = (k+1) - n_keep`, i.e. slot 0 is the most recent state (gdn_state.cpp:83-99).
- **`SequenceKv::reserve` is strongly atomic.** It allocates everything it needs, including copy-on-write blocks, before changing anything (paged_kv.cpp:128-153).
- **`forward` is atomic for validation errors and KV exhaustion.** It validates first, then reserves (qwen35.cpp:483-539).
- **Prefix restore is consistent.** It shares only up to a checkpoint whose owner entry is still cached, and restores MTP KV P-1 rows only when the owner has them (engine.cpp:503-530).
- **Lock discipline is sound.** No lock is ever held while acquiring another (`mu_`, `r->mu`, `vocab_mu_`), so there is no lock-ordering deadlock. Request fields are published under `mu_` before the worker reads them.

**Verdict:** approve with changes. There is no Blocker. R-1 to R-4 should be fixed before the Engine is served to more than one client at a time.

**Requirement tracing:** traced against ARCHITECTURE (Engine/Scheduler/Commit-rollback), D-005/D-012/D-013/D-014/D-016, the `engine.h` threading and error contract, and docs/api.md "Resource governance / Cancellation".

## Should-fix

### R-1 — One failing sequence fails the whole batch, including errors after a consistent commit
- **Severity:** Should-fix. **Confidence:** High.
- **Where:** src/runtime/engine.cpp:447-459; throw sites inside `tick()` at 735 (`checkpoint` → `GdnState::snapshot`), 739/761 (`Sampler::sample`), 740/762 (`accept`), 769 (bookkeeping `HALO_CHECK`); src/models/qwen35.cpp:361 (NaN check over the batched head).
- **Evidence:**
  - The `catch` in `run()` calls `slots_[a.slot]->reset()` and `retire(..., Error)` for **every** sequence in `active_`, whatever the cause and whichever sequence raised it.
  - The output loop (726-772) runs **after** `spec_->step()` committed every sequence. At that point each state is consistent (speculative.h "after verify(): ... consistent state").
- **Failure scenarios:**
  - (a) A host allocation failure while taking a best-effort D-013 checkpoint (a 150 MiB snapshot per checkpoint on the 27B model, D-003) fails all concurrent requests. D-013 says always-recompute is the fallback.
  - (b) A sampler error for one structured-output or sampled request fails all co-batched greedy requests.
  - (c) TD-2 is recorded as "the sequence must be reset()". In practice one NaN logit in any sequence's row resets **every** sequence in the tick, and their prefix-cache checkpoints are erased too (retire with Error → `erase_owner`). **TD-2 is worse than recorded.**
- **Suggested fix:**
  - Split tick failures into two classes:
    - failures before or inside `step()`, where the current "fail all" is correct;
    - per-sequence failures in the output phase, where only that sequence is retired with Error.
  - Make `checkpoint()` best-effort: catch Error(Memory), log, and skip the checkpoint.
  - Record the batch-wide reset in TD-2.

### R-2 — The worker is `noexcept` but calls a throwing `retire()` outside any try, so the server can terminate
- **Severity:** Should-fix. **Confidence:** High on the path. Low on likelihood, which depends on allocator behaviour.
- **Where:** engine.cpp:413 (`void run() noexcept`); unguarded `retire` calls at 440 (cancel), 462 (finished), 456 (inside the catch handler) and 477 (shutdown).
- **Evidence:**
  - `retire` with prefix cache on builds `c.gdn = seq.gdn.snapshot()` (799-801). That throws Error(Memory) on `bad_alloc` (gdn_state.cpp:113-117).
  - It also allocates `CacheEntry` / `make_unique`.
  - Its `flush_mtp` guard catches only `const Error&` (795). A `std::bad_alloc` from `mtp_forward`'s vectors escapes.
  - Any exception leaving `run()` calls `std::terminate`.
- **Failure scenario:** under memory pressure (the 32 GiB OS pool on the EVO-X2, D-002), a request finishes or is cancelled. The retirement snapshot fails, and the whole `halo serve` process aborts, taking every client with it. This contradicts the tick handler's "keep serving" comment and RR-005 crash containment.
- **Suggested fix:**
  - Make the cache part of `retire` best-effort: wrap it in `try` / `catch (...)`, and on failure `erase_owner` and skip caching.
  - Always complete the slot-free and done-delivery part.
  - Also catch `std::exception` in the flush guard.

### R-3 — No cancellation before the first token: disconnect, `request_timeout` and `stop()` can't reach queued or prefilling requests
- **Severity:** Should-fix. **Confidence:** High.
- **Where:**
  - src/api/generation.cpp:138-186: shutdown, deadline, disconnect and nesting are all checked only inside the `TokenCallback`.
  - engine.cpp:438-445: `cancel` is polled only for `active_` sequences, and it can only be set from a callback (384).
  - A request in `pending_` (197) has no cancel path at all.
  - tools/halo/config.cpp:313: `max_concurrent` can exceed `runtime.parallel`.
  - src/api/server.cpp:933-941: `stop()` joins the listener. httplib then drains workers that are still inside `engine.generate`.
- **Failure scenarios:**
  - (a) A client sends a near-context prompt (32K tokens) and disconnects immediately. The engine runs the whole chunked prefill (many ticks, each sharing weight passes with the other streams) before the first callback notices.
  - (b) With `--max-concurrent 8 --parallel 4`, four admitted requests sit in the engine's `pending_`. They have passed `queue_timeout`, and `request_timeout` does not apply to them. They wait until a slot frees, however long that takes.
  - (c) SIGTERM during (a) or (b): `ApiServer::stop()` blocks until those requests are admitted, prefilled and emit a token. A second signal `_Exit`s (main.cpp:46).
- **Docs claim otherwise:**
  - docs/api.md:121: "Wall-clock limit for one generation".
  - docs/api.md:138: "`ApiServer::stop()` cancels in-flight generations".
- **Suggested fix:**
  - Add an engine-side cancel path that doesn't need a token. For example, a `GenerateRequest` field that is an `std::atomic<bool>*` or `std::stop_token` plus a deadline. The worker checks it for both `pending_` and prefilling sequences at every tick.
  - Wire the API's `stopping`, deadline and `sink.alive()` polling to it.
  - Clamp `max_concurrent` to `max_sequences`, or document that the extra requests queue inside the engine.

### R-4 — Invalid or unsupported client JSON schemas return 500 "generation failed" instead of 400
- **Severity:** Should-fix. **Confidence:** High.
- **Where:**
  - src/api/requests.cpp:169-178 and 463-479: the schema is only `dump`ed, never compiled.
  - src/api/generation.cpp:193-203: `call` catches every exception and sets `StopCause::Error`, `public_error = "generation failed"`.
  - src/api/server.cpp:499-503 (`stream_error` → `ErrorKind::Server`) and 527/572 (thrown as a 500).
- **Evidence:**
  - `engine.h` says `generate()` throws Error(Api) for an invalid SamplingParams / JSON schema, and Error(Unsupported) for an unsupported construct. The grammar enforces these (e.g. json_grammar.cpp:285 `unsupported("nesting deeper than 256")`).
  - `classify_exception` (json_util.cpp:122-134) would map these to 400 / unsupported, but it never sees them.
- **Failure scenario:**
  - A client sends `response_format.json_schema` with an unsupported keyword and gets HTTP 500 `server_error`. The error is also logged as `HALO_ERROR "generation failed"`.
  - When streaming, the client gets 200 plus an SSE error event after the role chunk.
  - A 5xx tells clients and SDKs to retry.
- **Test gap:** tests/unit/api/fake_engine.cpp never throws Api or Unsupported (it only throws `Backend` at :195), so no API test can catch this.
- **Suggested fix:**
  - In `run_generation`, keep the typed exception thrown by `generate()` before any token (e.g. rethrow it, or store an `ApiErrorInfo` from `classify_exception`).
  - Alternatively, compile the grammar during request parsing.
  - Add a fake-engine mode that throws Error(Api) and Error(Unsupported), and assert 400.

### R-5 — Engine failure paths are untested, and some tests only check call counts or self-consistency
- **Severity:** Should-fix. **Confidence:** High.
- **Where:** tests/unit/runtime/test_engine.cpp (no test for the areas below); test_speculative.cpp:387; test_qwen35_golden.cpp:449-450; engine test "EveryTickIsOneTrunkPassAndRollbackNeverAddsOne" (:439).
- **Evidence:**
  - (a) No test covers:
    - a tick failure with ≥ 2 active sequences (R-1);
    - destroying the Engine while requests are pending or active: the shutdown path at engine.cpp:469-488 has never run in a test;
    - a failing retirement snapshot (R-2);
    - cancelling during prefill or queueing (R-3).
  - (b) test_engine.cpp:519-521 says real MTP drafts are never accepted on the tiny model. Greedy output is invariant to draft content (speculative.h), and the draft-chain tests compare the model's own `mtp_forward` with itself (test_qwen35_golden.cpp:450: "No golden exists for the draft chain"). **Consequence: a wrong depth ≥ 2 MTP input (e.g. the trunk hidden instead of the MTP's own post-norm hidden) passes every test and only shows up as lower acceptance on the target.** This is the owed D-005 golden; it is still open.
  - (c) `trunk_passes` and `weight_passes` are counts of `forward` / `mtp_forward` calls, because each sets `cost.weight_passes = 1` unconditionally (qwen35.cpp:540, and the same in `mtp_forward`). `measured_weight_bytes` is reported but no test asserts it.
- **Suggested fix:**
  - Add engine tests for (a) using a failure-injection seam, e.g. an `on_tick` that throws, or a tiny `checkpoint_budget_bytes`.
  - Treat the llama.cpp draft-chain golden as a gate on enabling MTP by default.
  - Assert `measured_weight_bytes` against `CostModel` per tick.

## Nits

- **N-1 — A swallowed MTP flush failure lets the MTP queue grow without bound.**
  - Location: speculative.cpp:399-405. Confidence: Medium.
  - When `flush_mtp` throws Memory and the profit gate is off (k = 0, so no depth-1 call resets MTP), the queue keeps growing by n_embd floats per token (20 KB on the 27B model). The warning is logged every tick.
  - `ensure_capacity` (engine.cpp:584-587) evicts the cache for the MTP pool, so this is unlikely by default.
  - Fix: on a failed flush, drop that sequence's MTP (`mtp_kv.reset()`, clear the queue), as the engine's fallback does.
- **N-2 — The Memory retry loop can skip the MTP fallback when many cache entries exist.**
  - Location: engine.cpp:696-699. Confidence: High.
  - Each retry evicts one cache entry and counts as an attempt. With more than 9 cached entries (`prefix_cache_entries` ≥ 9), `attempt > 8` rethrows before the k = 0 fallback is tried, and R-1 then fails every sequence.
  - Fix: evict until the cache is empty without counting attempts, then fall back.
- **N-3 — The reasoning-budget continuation can exceed the context.**
  - Location: generation.cpp:219-226. Confidence: Medium. The token count of `close_reasoning` is unverified.
  - `r2.prompt = prompt + observed + close_reasoning`. With `max_tokens` clamped to `ctx - n` and an explicit budget of `max-1` (or a tiny `max_tokens` where reserve < the close length), `r2.prompt.size() >= ctx`. `generate` then throws Error(Api), which becomes a 500 (R-4).
  - Fix: skip the continuation (finish with Length) when `r2.prompt.size() + 1 > ctx`.
- **N-4 — `TickInfo` pass counters are call counts.**
  - See R-5(c). The comment "cost-model weight bytes" on `last_tick_predicted_bytes` is accurate, but "measured" weight passes is not.
- **N-5 — Autotune has no consumer.**
  - `git grep autotune HEAD -- src/runtime src/api src/speculative src/models tools/halo` is empty. "Tuning never runs per request" holds only trivially, and tuned winners are never used.
  - Fix: track it as debt until the runtime reads `winning_configuration` at plan time.
- **N-6 — D-016 wording and code differ on default scaling.**
  - All three backends default `q_scale = nullopt` → 1/sqrt(d_k): cpu/ops.h `GdnQkParams`, vulkan/ops.h:137, hip/ops.h:106 (HEAD). D-016 says "neither backend scales implicitly".
  - The behaviour is consistent across backends and qwen35.cpp:240-241 passes it explicitly, so there is no bug.
  - Fix: either amend D-016 to name the default, or remove the default so callers must be explicit.

## Needs runtime verification (qa-engineer)

1. R-2: confirm that a failing snapshot at retirement terminates the process (ASan, or an allocation-failure injection).
2. R-3: time from disconnect to engine stop for a 32K prompt; `stop()` latency with requests waiting in the engine queue.
3. Structured-output mask cost: `compute_trie_mask` traverses the vocabulary trie on the worker thread for every structured sequence on a cache miss. Measure the added tick latency on the 248K vocabulary for co-batched greedy streams (head-of-line blocking).
4. Vulkan `isnan` in argmax_partial.comp on RADV (driver fast-math), and the HIP kernels on gfx1151 (D-001: emulation only).

## Verified fixed from the 2026-09-24 review

- **M-1:** fixed. `GdnQkParams{qk_l2norm, q_scale}` is on CPU (ops.h, gated_delta_rule.cpp `gdn_q_scale`), Vulkan (ops.cpp:242-243) and HIP (`resolve_q_scale`). The model passes it explicitly (qwen35.cpp:240-241). The Vulkan tests `NoL2NormAppliesExactlyTheGivenQScale` and `NonFiniteQScaleIsRejectedLikeCpu` cover it.
- **M-2:** fixed. test_vulkan links `halo_backend_cpu` and `halo_tensor` (tests/unit/vulkan/CMakeLists.txt). reference.h keeps only data generation plus `halo::tensor::dequantize_row`, and the comparisons are against `hc::matmul` / `hc::gated_delta_rule_*` on the same arrays. fp64 is used only for bound scales (allowed by D-016). HIP tests do the same, in both emulation orders (hip_test_util.h).
- **S-1:** fixed. `BufferView` operands; `FusedQkvRowReadThroughViewsWithoutCopies` has guard checks and requires bitwise equality between view and dense runs. The overlap-check internals were not read line by line.
- **S-2:** addressed with a de-interleave copy in the model (qwen35.cpp:193-194). The in-kernel stride is deferred.
- **S-3:** fixed. The Vulkan NaN word is checked on the host (backends/vulkan/src/ops.cpp:127), tested by `VkArgmax.EdgeValuesInfNanAndAllEqual` (test_vk_ops.cpp:814). HIP has the same check (`decode_argmax`, hip ops.cpp:554 at HEAD).
- **S-4:** fixed. src/models/adapters.cpp, with tests/unit/models/test_adapters.cpp (dequant for every type, vector/embedding, VocabSpec vs the tokenizer golden).
- **S-5:** fixed. `LongTrajectoryPrefillThenDecodeMatchesCpu` (T = 4096 + 64, with a second random-walk bound because the linear bound is about 0.08×scale) and `InPlaceMultiRowWithSlotsMatchesCpu`.
- **S-6:** fixed. `matvec_q5_k.comp` (plus IQ4_XS).
- **S-7:** fixed. planner.h:151 `enabled = true`; ARCHITECTURE updated (8705e46).
- **N-1:** fixed. A bounded, reused, chunked staging buffer under `transfer_mutex_` (context.cpp `staged_upload` / `staged_download`).
- **N-2:** fixed. `halo_tensor` is PUBLIC on `halo_backend_vulkan`.

## Not reviewed

- Uncommitted working-tree changes (other agents' work in progress).
- src/profiling suite.cpp, record.cpp, workload.cpp, hw_state.cpp, counters.cpp, timer.cpp: only stats.cpp was read closely.
- src/autotune: db.cpp schema/migrations, lookup.cpp, profile_key.cpp, cost_model.cpp, cpu_ops.cpp. Only tuner.cpp, sqlite.cpp and `persist_tune` were read.
- src/sampling json_schema.cpp and the json_grammar.cpp internals: only the nesting and depth bounds were checked.
- Vulkan and HIP kernel math line by line: the shaders, and hip kernels/gdn.h, head.h, attention.h, gemv.h. hip runtime.cpp and registry.cpp.
- tools/halo bench.cpp, args.cpp, model_io.cpp; src/api prompt.cpp and most of json_util.cpp.
- The factual claims in docs/benchmarks.md and docs/vulkan.md.
- Everything that needs execution: no test counts or ASan results were observed.
