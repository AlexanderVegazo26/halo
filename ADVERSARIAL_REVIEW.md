# Adversarial Repository Review

**Repository:** HALO — C++23 LLM inference runtime (Qwen3.8-27B on AMD Strix Halo)
**Branch/commit:** `halo-v0.2` @ `27813ad`, plus 34 files of uncommitted in-flight work (reviewed as both states where it matters)
**Date:** 2026-09-26
**Method:** full reconstruction from PRD/TRD/DECISIONS/ADRs first, then five parallel adversarial workstreams (HTTP API surface, hostile-input parsers, engine correctness, build/supply chain, test suite), each required to disprove its own candidates against the code before reporting. No repository files were modified during the review. Every finding below carries file:line evidence that was independently re-verified by the reviewer unless marked otherwise.

---

## Executive Summary

HALO is in better adversarial shape than most codebases at this stage: the untrusted-input parsers (GGUF, tokenizer, minja templates) survived a genuinely hostile review with only one UB defect and one defense-in-depth gap; the HTTP application layer's claimed security fixes were re-verified and are real; and the numerics core (CPU forward vs. independent transformers goldens) is honestly tested with documented tolerances. The project's own review trail (`docs/reviews/`, `docs/security-hardening.md`) is unusually candid, and this review confirmed most of its claims rather than relitigating them.

That said, the review found **two High-severity defects that the existing review trail missed or left open**, and a set of contract-level problems:

| Severity | Count | What they are |
|---|---|---|
| CRITICAL | 0 | — |
| HIGH | 2 | In-step forward errors kill **every** concurrent sequence (contradicts owner decision D-017 and PRD §48); shutdown-time use-after-free between `ApiServer` destruction and httplib's un-joined worker pool (SIGTERM during traffic → crash) |
| MEDIUM | 11 | Memory-retry atomicity violation in the engine's error-recovery path; padded LM-head rows are live sampling candidates (latent on the real 27B packs); slowloris pool-exhaustion fix (S-38) exists only uncommitted; PRD roofline arithmetic is inverted and its PR-004 targets are physically unreachable (never corrected in the PRD); circular differential test on the quantized forward; no CI despite TRD §60 mandating it; llama.cpp "pin" is defeatable; packaging modes ship untracked secrets; recursive JSON escape runs before its depth guard; planner vs. engine disagree on checkpoint memory; 27B validation is 3 prompts × 32 tokens, manual, harness not in the repo |
| LOW | 9 | Signed-overflow UB in GGUF `rows()`; negative temperature silently means greedy; `default_max_tokens` not cross-checked against the cap; dead KV-only cache entries; O(cache × prompt) admission scan; profit-gate speedup accounting bias; S-39/S-45 races fixed only uncommitted; stale docs/comments; script-hardening items |
| INFORMATIONAL | 5 | Silent SSE exception swallowing; double grammar compile per request; `/tokenize` response amplification; no slowloris defense on non-Linux builds; accepted SIGBUS risk (S-5) |

**Most important risks, ranked:**

1. **Availability under the exact workload HALO is built for.** The engine's batched tick fails *all* sequences when any one sequence produces an in-step error (NaN logit in the fused-greedy path, allocation failure inside the forward). PRD §48 says "one request must never destabilize the system"; owner decision D-017 already committed to per-sequence isolation; `TECH_DEBT.md` TD-2 admits it is not done. Until ADR-001 WS-BI-2/BI-6 land, a single pathological request — or plain memory pressure — kills every concurrent agent stream.
2. **The most common maintenance operation crashes the server.** `halo serve` destroys `ApiServer` while httplib worker threads can still be executing handlers that touch already-destroyed members (verified member declaration order + `stop()` semantics + the CLI destroy path). SIGTERM with any in-flight request is the trigger; a stalled SSE reader makes the window ~60 s wide.
3. **The headline agentic feature (prefix caching, D-013/D-014) works but not the way the decisions describe, and its NFR is unvalidated.** Message-boundary `checkpoint_hints` have no producer anywhere in the serving path — D-013's core mechanism silently does not exist in production code. The planner reports checkpoints as disabled on CPU while the engine allocates a host checkpoint budget outside the plan, so `halo inspect`'s memory report understates runtime footprint. And the D-014 "cached TTFT ≤ 300 ms at ≥ 80% hits" NFR has never been measured against real multi-turn agent traffic.
4. **Performance targets in the PRD are physically impossible and the PRD was never corrected.** PRD §9 contains roofline arithmetic the TRD's own changelog calls "inverted" (15 ms/token ≈ 66 tok/s — the real naive figure is 66 ms ≈ 15 tok/s) and targets (≥ 45 tok/s decode) that `DECISIONS.md` D-011/D-014 compute would need ≥ 742 GB/s on a 256 GB/s machine. DECISIONS.md overrides, but the PRD is still the cited spec in the README.
5. **Security fixes live only in the uncommitted tree.** S-38 (rolling-reconnect pool exhaustion, *reproduced* by the prior review), S-39 (guard phase-loss race that kills long non-streaming generations), and S-45 (fd-reuse TOCTOU) are all fixed in the working tree but not committed. HEAD — the only state a fresh clone, package, or bisect sees — still carries them. There is no CI to catch this class of drift.

**Major testing gaps:** the shipped server binary is never executed as a process and spoken to over HTTP; cancellation storms run only against a fake engine with a manual clock; mid-forward error injection is architecturally impossible (the four fault hooks fire around, never inside, the forward); ~110 of 545 tests silently skip without the dev-host `~/halo-ref` fixture directory; the quantized-forward differential is circular (reference dequantizer = implementation dequantizer); HIP device tolerances are an unmeasured guess whose negative tests self-disable when a device appears.

**Bottom line:** this is *not* "production ready" and the project itself does not claim to be (hardware verification pending is stated honestly). The code is code-complete on the CPU path with real strengths; the two High findings and the D-017 isolation gap should gate any v0.2-serving claim beyond loopback toy deployments.

---

## System Understanding

**What it is:** a single-model (Qwen3.8-27B, hybrid 48×Gated-DeltaNet + 16×GQA-attention), single-machine (EVO-X2, gfx1151, 128 GB unified LPDDR5x) LLM inference runtime with an agent-facing HTTP API. C++23, clang, CMake/Ninja. CPU reference backend is the semantic definition of every operator; Vulkan and HIP backends must differentially match it.

**Architecture (reconstructed from code, not just docs):**

```
tools/halo CLI ── api (cpp-httplib, thread pool, ConnectionGuard, Admission)
                     └── runtime: CpuEngine (ONE worker thread)
                             ├── scheduler: admit → tick (single batched forward
                             │   over ALL active sequences) → retire
                             ├── speculative: MTP draft/verify + ProfitGate
                             ├── sampling: per-request Sampler (greedy fused argmax
                             │   or CPU chain penalties→temp→top_k→typical→top_p→min_p)
                             ├── kv_cache: paged KV, CoW on prefix share
                             ├── state: GDN fp32 recurrent state, K-slot rollback ring (D-012)
                             └── models/qwen35: one forward graph over a Backend&
model (GGUF, mmap) · tokenizer (byte-level BPE, trie added-tokens) · template (patched minja)
backends: cpu (reference) | vulkan (partial, lavapipe-verified) | hip (emulation-only)
```

**Trust boundaries:**
1. **HTTP clients → API.** Loopback-only by default (D-017, enforced and re-verified). Optional API key. The pre-auth surface (headers only) is defended by `ConnectionGuard` (Linux-only).
2. **Model files → parsers.** GGUF/tokenizer/template are treated as hostile (WS-L) and are genuinely hardened: budget-charged metadata parsing, trie tokenizer with caps, pinned+patched+fail-closed minja with render budgets.
3. **Engine worker → everything.** Single worker thread owns all mutable engine state; API threads touch only `Request` state under its mutex. Verified no shared-state race.
4. **Build → machine.** Four hash-pinned FetchContent deps; a SHA-256-verified patch pipeline for minja; packaging scripts that run on the operator's machine, some with root.

**External dependencies:** nlohmann/json, cpp-httplib 0.57.1, minja (pinned commit + patch), googletest — all hash-verified; SQLite (system); Python dev tools fetching reference data from HuggingFace `resolve/main` (mutable, unpinned — dev-only). Model files from HF (hostile-input threat model applies).

**Key assumptions the system relies on (and this review's verdict on each):**
- *The CPU backend is the semantic oracle* — supported by independent transformers goldens (with the circular-quant caveat, Finding M5).
- *The tiny model exercises the same code paths as the 27B* — **partially unsupported**: MTP real acceptance is 0 on the tiny model (accept paths oracle-forced); padded-vocab LM-head behavior differs; chunked-prefill at scale untested.
- *MTP greedy output equals non-MTP greedy output* — verified on 96 positions at 27B, 0 divergences; narrow.
- *The loopback-only posture makes pre-auth DoS a Medium* — accepted by D-017; S-38 shows the residual is real at HEAD.
- *Hardware will arrive before the performance claims need defending* — Phase 0 (TRD §68 steps 1–11) is entirely unexecuted; every efficiency NFR is unmeasured.

---

## Requirements Compliance

Key: ✅ implemented / 🟡 partial / ❌ violated / ⚠️ contradictory docs / ⬜ missing / 🔬 untested-at-scale

| Requirement | Status | Evidence | Risk |
|---|---|---|---|
| FR-002/003 model discovery + compatibility report | ✅ (CPU) | `halo inspect` exercised on real 27B headers (`smoke-27b.md` §1) | none on CPU |
| FR-004 memory planning, "never blindly consume all memory" | 🟡 | planner is overflow-safe (`planner.cpp:17-39`), but engine allocates a **host checkpoint budget outside the plan** (`engine.cpp:302-316`) that `halo inspect` never reports — Finding M10 | Medium |
| FR-006 streaming, cancellation, concurrent requests | 🟡 | works; cancel checked every tick in every state (`engine.cpp:501-504,544,579`); but in-step errors kill all sequences — Finding H1 | High |
| FR-007 sampling (greedy fused argmax, top-k/p, min-p, penalties, seed) | ✅ | `sampler.cpp:46-59` validation is strict; chain order verified correct | Low residual (Finding L2 negative-temperature) |
| FR-008 paged KV + prefix reuse | 🟡 | paged KV verified correct (CoW, clamping); prefix cache works engine-level (`test_engine.cpp:340-389`); `checkpoint_hints` have **no producer** in serving; GPU-tier policy untested | Medium (M10) |
| FR-009 MTP + profit gate (auto-disable ≤ overhead) | 🟡 | `ProfitGate` implemented (`speculative.cpp:21-48`); real acceptance only verified at 27B on 3×32 greedy tokens; accept-path correctness oracle-only on tiny model | Medium |
| FR-011 autotuning never makes model unusable; RR-003/004 versioned, corrupt-safe profiles | 🟡 | SQLite profiles work for **CPU tunables only**; GPU variants unbridged (memory.md) | Low |
| FR-013 API surface | ✅ | all documented endpoints exist; F-1 fixed honestly with a warning | — |
| FR-014 tokenizer/templates | ✅ | HF-token-for-token match; hostile-template bounded with demonstrated-red evidence | — |
| PR-004 performance targets | ⚠️ | PRD arithmetic inverted vs TRD §2.3/D-011; targets physically unreachable (need ≥742 GB/s); D-014 replaced them but PRD never revised — Finding M4 | Medium (contract) |
| RR-001 optimized-kernel fallback with typed diagnostic | 🔬 | n/a on dev host (no GPU kernels runnable) | — |
| RR-005 API recovers from individual request failures | ❌ | in-step forward error → `fail_all` kills every active sequence (`engine.cpp:529-536,592-601`) — Finding H1 | High |
| RR-006 state/KV integrity under cancellation & eviction | ✅ | rollback equivalence tested; slot indexing verified correct; "verify+rollback == decode" invariant per tick (`engine.cpp:918`) | — |
| §12 security (loopback, request limits, path validation, no shell/file access via API) | ✅ | re-verified: no request field names a path; body caps enforced incl. chunked; auth before body | — |
| §48 "one request must never destabilize the system" | ❌ | same as H1; also any `bad_alloc` inside the forward hits `catch(...)` → `fail_all` | High |
| TRD §60 CI (format→compile→tests→GPU correctness→bench→package) | ⬜ | no CI exists (no `.github/`, no pipeline files) — Finding M6 | Medium |
| TRD §31 fuzzing (API JSON, cancellation timing, eviction timing, draft-depth extremes) | 🟡 | GGUF parser fuzzed (no oracle); no API-level fuzzer beyond one "never crashes" test; cancellation/eviction timing not fuzzed | Medium |

---

## Critical Findings

None. No remote code execution, no memory-safety vulnerability reachable from a hostile model file or HTTP request was found that survived disproof. The two High findings are availability/reliability defects, not exploitable compromises.

---

## High Findings

### [HIGH] H1 — One sequence's in-step error kills every concurrent sequence (D-017 violated)

**Location:** `src/runtime/engine.cpp:529-536` (`fail_all`), `:592-601` (tick-wide catch); `src/models/qwen35.cpp:771-792` (KV commit + `mark_slots_written` *before* the LM-head section); `src/speculative/speculative.cpp:256-259`; `docs/dev/TECH_DEBT.md:8` (TD-2)

**Category:** Reliability / requirements violation

**Description:** `CpuEngine::run()` wraps each `tick()` in a catch that invokes `fail_all`, which resets the slot and retires **every** active sequence with an error:

```cpp
// engine.cpp:592-601
} catch (const std::exception& e) {
    log_error("tick failed", e.what());
    fail_all(e.what());            // resets slots_, retires ALL active requests
} catch (...) {
    fail_all("internal error");    // std::bad_alloc lands here
}
```

An exception escapes `tick()` whenever the error happens *inside* the batched forward, after state has been modified. Two concrete producers:

1. **NaN in the fused-greedy path.** The trunk forward commits KV and marks GDN slots written before `head()` computes the argmax, which raises `Error(Kernel)` on NaN (D-016). The speculator's verify catch only restores the MTP KV (`speculative.cpp:256-259`); `speculative.h:32-35` admits the trunk state is then "unspecified: reset() them". The exception propagates to `fail_all`.
2. **Allocation failure inside the forward.** `head()` resizes output vectors and allocates through the backend (`qwen35.cpp:782-787`, `cpu_backend.cpp:243-248`); a `std::bad_alloc` there is not a `halo::Error`, skips any per-sequence handling, and hits `catch (...)` → `fail_all`.

The output phase *is* isolated per sequence (`engine.cpp:924-932`); only in-step errors are tick-wide.

**Expected behavior (per the project's own contract):** D-017 (DECISIONS.md): "a data error in one sequence (NaN logit, sampler or grammar error) fails only that sequence; the rest of its batch commits." PRD §48: "One request must never destabilize the system." RR-005: "API server recovers from individual request failures without process termination."

**Actual behavior:** every active sequence — including healthy unrelated agents mid-generation — is terminated with "tick failed" / "internal error".

**Attack / failure scenario:** Four coding-agent clients are streaming from one `halo serve`. Memory pressure on the 32 GiB OS-visible host (D-002 topology) makes one allocation inside a batched step fail, or one prompt drives the Q6_K LM head to a NaN row. All four streams die simultaneously. No attacker privileges are needed for the memory-pressure variant; it is an expected operating condition on this hardware.

**Impact:** total loss of in-flight work for all tenants on a single-tenant-class event; directly contradicts a committed owner decision and two PRD requirements. Not a crash — the process survives — which is why the test suite (which never injects mid-forward errors; the four `CpuEngineOptions::faults` hooks fire only before-step/output/retire) gives false confidence here.

**Evidence:** quoted above; TD-2 in `docs/dev/TECH_DEBT.md:8` documents the same behavior ("in-step data errors (NaN in the LM head) still reset every sequence in the tick") and defers the fix to ADR-001 WS-BI-2/BI-6, which are not landed.

**Reproduction:** not runnable on the dev host without a model that emits NaN; labeled mechanism-verified, scenario-conditional. The control flow is unambiguous from the code.

**Recommended fix:** commit-after-status ordering (ADR-001 WS-BI-2): compute logits/argmax before committing KV/GDN state, so a data error leaves pre-tick state intact and the per-sequence output-phase isolation covers it. This must land together with the Finding M1 reorder (same code region).

---

### [HIGH] H2 — Shutdown-time use-after-free: `ApiServer` is destroyed while httplib workers still run handlers

**Location:** `src/api/server.cpp:386-390` (member declaration order), `:1061-1071` (`ApiServer::stop()`), `:909-914` (worker-thread logger touching `guard`/`metrics`), `tools/halo/commands.cpp:560-572` (destroy path); httplib 0.57.1 `Server::stop()` semantics (verified against upstream source by the API workstream)

**Category:** Reliability (memory safety at the process level)

**Description:** `Impl` declares members in this order:

```cpp
httplib::Server svr;      // :386 — destroyed LAST
Admission admission;      // :387
Admission utility;        // :388
ConnectionGuard guard;    // :389 — destroyed before svr's worker pool is joined
Metrics metrics;          // :390 — destroyed before svr's worker pool is joined
```

C++ destroys members in reverse declaration order: `metrics`, then `guard`, then `svr` — and `~httplib::Server` joins the worker thread pool only in its own destructor. In httplib 0.57.1, `Server::stop()` merely closes the listening socket; it does **not** shut down or join workers (verified against the upstream source). `ApiServer::stop()` joins only the *listen* thread (`server.cpp:1068-1069`). Worker threads touch `guard` and `metrics` on every in-flight request: the per-request logger calls `metrics.count_request` + `guard.request_done` (`server.cpp:909-914`), `guard.handler_started` runs in routing (`server.cpp:409`), and `StreamPhase`'s destructor calls `guard.request_done` (`server.cpp:340-345`).

`cmd_serve` destroys the `ApiServer` immediately after `stop()` returns (`commands.cpp:565-572`); SIGTERM/SIGINT drive exactly this path (`tools/halo/main.cpp:64-72`).

**Attack / failure scenario:** An operator restarts `halo serve` (routine maintenance). One client has an SSE stream whose socket stalled: generation is cancelled quickly, but the handler thread is blocked in `DataSink::write` for up to `write_timeout` (60 s, `server.h:85`). `stop()` returns; `~ApiServer` begins destroying `metrics` and `guard` microseconds later; the still-blocked worker's next log line or `StreamPhase` destruction calls into freed objects. Undefined behavior: crash (SIGSEGV), or heap corruption. Even without a stalled reader, any engine tick in progress (up to seconds at 27B CPU decode) leaves a real window.

**Expected behavior:** after `stop()`, no thread references server members; destruction is safe.

**Actual behavior:** destruction races un-joined workers that hold pointers into the dying object.

**Impact:** process crash or heap corruption on the most common operational action, with client traffic in flight. Not exploitable for privilege escalation, but it converts graceful-shutdown semantics into a coin flip. Absent from both security reviews (they covered the guard's *internal* races, not object lifetime vs. the pool).

**Evidence:** member order and `stop()` quoted above (independently re-read by this reviewer); httplib `stop()`/`~ThreadPool` semantics verified against httplib 0.57.1 source by the API workstream.

**Reproduction:** start `halo serve`, begin a streaming request with a client that stops reading, send SIGTERM. Window is up to 60 s (write timeout). Labeled mechanism-verified; the crash itself is UB and may manifest silently.

**Recommended fix:** in `ApiServer::stop()`, after `svr.stop()`, call `svr.wait_until_gracefully_stopped()` (stop + `thread_pool_->shutdown()`) before returning; alternatively declare `guard`/`metrics`/`admission` *before* `svr` so the pool is joined while they live. Add a regression test that stops the server with a handler blocked on a full socket.

---

## Medium Findings

### [MEDIUM] M1 — The engine's Memory-error retry is built on a false "errors are atomic" assumption

**Location:** `src/runtime/engine.cpp:838-858`; `src/models/qwen35.cpp:771-787` (trunk), `:856-873` (MTP); `src/speculative/speculative.cpp:74-77`

**Category:** Correctness (fault-handling path)

**Description:** On `Error(Memory)` from a speculative step the engine evicts one prefix-cache entry and retries the whole step, on the documented assumption "Pre-tick state is intact (Memory errors are atomic)" (`engine.cpp:840`). That assumption is false: the trunk forward commits KV and marks GDN slots written **before** the head section, and the head section then allocates (`o.hidden.resize`, `o.argmax.resize`, `st.alloc` — `qwen35.cpp:782-787`; `cpu_backend.cpp:243-248` throws `Error(Memory)`). A Memory error from those allocations escapes `step()` after state advanced; the retry re-verifies the old pending token at a shifted position (RoPE position `len0 + n_keep` instead of `len0`). The fed-vs-length invariant (`engine.cpp:918`) eventually kills the sequence — converting a *designed-for recoverable* condition (evict cache, retry) into a sequence failure, and burning cache entries per retry.

A latent trap compounds it: `std::bad_alloc` from the un-mapped vector growth paths (`mtp_queue_hidden.insert`, `speculative.cpp:74-77`) is **not** converted to `Error(Memory)` (unlike `paged_kv.cpp:42`, `gdn_state.cpp:43,115`, `cpu_backend.cpp:247`), so it skips the retry and lands in `fail_all` (H1) — killing every sequence. If someone later "fixes" that inconsistency by mapping `bad_alloc` to `Error(Memory)` (the pattern used everywhere else) without reordering the commit, the retry path will silently corrupt sequences.

**Impact:** reliability under memory pressure — a normal operating condition on a 128 GB unified-memory box serving a 17 GB model with a 32 GiB OS pool. No test can inject this: the four engine fault hooks fire only before-step/output/retire; no hook fires mid-forward.

**Recommended fix:** move KV commit + slot marking after the head section (same reorder as H1), or scope the retry to pre-commit failures only. Add a fault hook that throws inside the head allocations.

---

### [MEDIUM] M2 — Padded LM-head rows are live sampling candidates in unstructured generation

**Location:** `src/sampling/sampler.cpp:246-248` (devs' own comment), `:300-316`; `src/models/qwen35.cpp:486-517` (fused argmax)

**Category:** Correctness (latent until the real packs run)

**Description:** The GGUF LM head is padded to 248,320 rows while the tokenizer has 248,077 entries. The code comment says "Ids past the tokenizer are never allowed under structured output" — true there (`mask_has` bounds-checks, `sampler.cpp:32-38`), but **unstructured** sampling admits every non-`-inf` logit in `[0, 248320)`, and the fused greedy argmax has no tokenizer-vocab clamp either. If pad rows carry near-zero logits, they win the argmax whenever all real logits are negative; if sampled, `emit()` → `token_to_piece` → `check_id` throws `Error(Api)` ("token id … outside vocabulary", `tokenizer.cpp:179-182,752-754`) and the sequence dies mid-generation.

**Impact:** spurious mid-generation failures on any pack whose head is zero/untrained-padded past the tokenizer vocab. Unverifiable on the dev host (only `*.header.gguf` exists — the real weights have never been read); the tiny model has no pad rows, so no test can catch it. The 27B smoke (96 greedy tokens) did not trip it, which is evidence but not proof for 256K contexts.

**Recommended fix:** clamp unstructured candidates to `min(model_vocab, tokenizer_vocab)` (or subtract a large constant from pad-row logits) exactly as structured output does.

---

### [MEDIUM] M3 — S-38 (reproduced slowloris pool exhaustion) is fixed only in the uncommitted tree

**Location:** `src/api/conn_guard.cpp:161-185` (per-peer cap + overload shedding exist in the working tree); committed state at `e6ce83e` carries the S-38 finding (`docs/reviews/2026-09-26-security-review-api-m5-evox2.md:43-71`, reproduced: 100 dribbling connections made `/health` 1/79 available)

**Category:** Security/availability — process failure

**Description:** the 2026-09-26 security review reproduced rolling-reconnect pre-auth pool exhaustion against the committed code and graded it Medium under D-017's loopback-only scope. The recommended fix (per-peer header-phase cap of 8, overload shedding, header clock starting on first byte) is implemented and regression-tested (`tests/unit/api/test_api_guard.cpp`) — **but only as uncommitted modifications** (`git status`: `M src/api/conn_guard.cpp`, `M src/api/server.cpp`, new `test_api_guard.cpp` staged). The same applies to S-39 (guard phase-loss race that can cut off a long non-streaming generation at the header timeout; fix at `conn_guard.cpp:132-135`) and S-45 (fd-reuse TOCTOU; fix at `conn_guard.cpp:232-255`). HEAD still carries all three.

**Impact:** any fresh clone, package build from HEAD, or bisect lands on the vulnerable state. D-017's loopback-only grading keeps this Medium rather than High, but the fix's existence in the tree creates a false sense of closure (the 2026-09-26 review documents it as still-open).

**Recommended fix:** commit the WS-I M6 work; until then, treat S-38/S-39/S-45 as open in release notes.

---

### [MEDIUM] M4 — PRD roofline arithmetic is inverted and PR-004 targets are physically unreachable; the PRD was never corrected

**Location:** `docs/HALO_PRD_v1.1.md:309` ("~17 GB ÷ ~256 GB/s ≈ 15 ms/token ≈ 66 tok/s"); `docs/HALO_TRD_v1.1.md:96` (changelog: "Fixed inverted roofline arithmetic… naive full-weight read ≈ 66 ms/token ≈ 15 tok/s"); `docs/DECISIONS.md:194-203` (D-011: ceiling ≈ 15.2 tok/s; "PR-004's batch-1 ≥ 45 tok/s is physically unreachable — it would need ≥ 742 GB/s")

**Category:** Requirements contradiction

**Description:** The PRD still states a theoretical ceiling of 66 tok/s and targets ≥ 45 tok/s decode / ≥ 70 tok/s MTP-effective / ≥ 120 tok/s 4-agent. The TRD's own v1.2 changelog calls this arithmetic inverted; D-011 computes the naive ceiling at ~15.2 tok/s and D-014 (owner-resolved) redefined success as efficiency NFRs. But DECISIONS.md overrides "until the PRD/TRD are revised" — and the PRD was not revised. Anyone planning against the PRD (the README's cited spec) derives budgets that exceed the memory subsystem's nameplate bandwidth by ~3–5×.

**Impact:** mis-planning, mis-communication, and credibility risk; Phase-0 roofline measurement (the TRD's own gate) can be quietly skipped when the written targets already look "achieved" on a spreadsheet.

**Recommended fix:** revise PRD §9/§10 to the D-011 ceiling model and D-014 NFRs; mark PR-004 as superseded rather than "recorded as aspirations" in a non-normative file.

---

### [MEDIUM] M5 — The quantized-forward differential test is circular

**Location:** `tests/unit/models/test_qwen35_golden.cpp:623,712` (`BitIdenticalToF32ForwardOverDequantizedWeights` builds its reference with `halo::tensor::dequantize_row` — the same routine the quantized forward uses); `:577-583` (the file itself admits the a-priori tolerance model "is refuted by the measurement")

**Category:** Testing — false confidence

**Description:** the strongest claim backing "the CPU backend matches the transformers goldens" for the canonical quantized packs is a bitwise test whose reference path shares the implementation's dequantizer. A `dequantize_row` bug cancels out. The independent link to transformers exists only for the f32 path and by composition. The dequant kernels are differentially tested against `gguf-py` on real blocks (`D-007`), which softens this — but the *forward-over-quantized-weights vs transformers* numerics are effectively unasserted, and the project knows it (the quoted admission).

**Impact:** a systematic dequant-scale bug (exactly the class of bug that produces plausible-but-wrong tokens at 27B) would pass the suite.

**Recommended fix:** generate a transformers golden over dequantized-to-f32 weights (one offline run with `gguf-py` dequant + transformers) so the differential is independent end to end.

---

### [MEDIUM] M6 — No CI exists, while the project's own process relies on manually-verified commits

**Location:** no `.github/`, `.gitlab-ci.yml`, or any pipeline file (verified repo-wide); `memory.md:16-31` describes an orchestrator-plus-agents workflow where "agents never commit" and the orchestrator verifies by hand; TRD §60 mandates CI stages

**Category:** Process / requirements gap

**Description:** every protection this review verified — hash-pinned deps, the minja fail-closed patch, the 545-test suite, ASan runs — executes only when a human remembers to run it. The working tree currently contains 34 modified/untracked files across at least five workstreams, including staged security fixes (M3), uncommitted Vulkan ops, and docs describing in-flight state; the monthly agent spend-limit interruptions documented in `memory.md:33-35` make half-applied states a *recurring* condition. With no CI, nothing detects a commit that sweeps another agent's half-applied edit, a test that silently skips without fixtures, or a security fix that never lands.

**Impact:** the most likely root cause of a future bad release is process, not code. The "545 tests, 543 pass" claim is additionally a dev-host artifact: ~110 of those tests skip without the `~/halo-ref` fixture directory (64 `GTEST_SKIP` sites in 36 files, including nearly all forward/engine/model coverage), so the number is not reproducible on a fresh clone.

**Recommended fix:** minimal CI (build + ctest with fixtures cached, ASan lane) even before the GPU runner exists; a fixture-packaging step so skips become failures when fixtures are absent.

---

### [MEDIUM] M7 — `build-llamacpp.sh`'s pinned-commit enforcement is theater on the failure path

**Location:** `scripts/build-llamacpp.sh:8-9`

```bash
git fetch -q --depth 1 origin bd4f514db... || true
git checkout -q bd4f514db... || true
```

**Category:** Supply chain (dev tooling)

**Description:** both the fetch and the checkout swallow failure. If the fetch fails (throttling, network), the checkout is a no-op and the script builds whatever HEAD is — on a fresh clone, current llama.cpp master: arbitrary unreviewed upstream code. This is not decorative: the header states this build is the *reference implementation* and the source of `llama-quantize` used for golden and baseline comparison (D-009's numbers, PR-001 baselines). A silent version skew both executes unpinned code on the bench host and invalidates every baseline computed against it, with no signal.

**Disproof attempt:** the pin holds when the fetch succeeds (full 40-hex sha); the script is dev-invoked, not part of CMake/kit. But the documented invariant ("at the commit the EVO-X2 runs") is simply false on the failure path.

**Recommended fix:** drop `|| true` from the checkout; assert `git rev-parse HEAD` equals the pin after checkout.

---

### [MEDIUM] M8 — `package.sh --source worktree|copy` packages untracked files — including secrets — into the shipped kit

**Location:** `.gitignore:1-8` (no `*.db`, `*.log`, `.env`, `*.pem`, `*.key`, `native/`); `scripts/package.sh:148-151` (worktree mode stages `git ls-files -co --exclude-standard`; copy mode tars with only build/dist exclusions and **ignores .gitignore entirely**); `package.sh:247` (staged tree becomes `kit/source/halo-src-*.tar.gz` inside the distributed kit)

**Category:** Supply chain / secrets exposure

**Description:** `worktree` mode ships every untracked, non-ignored file; `copy` mode ships everything except four hard-coded exclusions (a `*.gguf` is gitignored — invisible to `git status` — but copy mode tar's it anyway). The whole staged tree is SHA256SUMS-verified and tarred into the field kit: a stray `.env`, private key, autotune DB, or 15 GB model in the repo directory is integrity-*endorsed* into `dist/halo-evox2-…-dirty.tar.gz` (a non-head `-dirty` kit already exists in `dist/`, proving these modes are used). Additionally, `DIRTY_PATHS` (untracked *filenames*) is embedded verbatim into `SOURCE_INFO`/`BUILDINFO` — reconnaissance value even without contents.

**Disproof attempt:** default `--source head` uses `git archive HEAD` (tracked files only) and is safe; overlays are path-validated (`package.sh:154`). The exposure requires opting into the loose modes — but those modes exist precisely for "ship what I'm looking at" and nothing warns about the git-ignore boundary.

**Recommended fix:** extend `.gitignore` (`*.db *.sqlite* *.log .env *.pem *.key native/`); make `copy` honor it (`rsync --filter=':- .gitignore'` or `tar --exclude-from`); refuse to package when the staged tree contains files matching a secret-name pattern.

---

### [MEDIUM] M9 — `escape_json` recurses over message JSON *before* the depth check that claims to protect it

**Location:** `src/api/prompt.cpp:188-206` (recursive `Escaper::escape_json` called at line 200, before `tmpl.apply` at line 206); the depth check lives inside render (`src/template/chat_template.cpp:478-484`); `include/halo/template/chat_template.h:475-477` documents the render-side check as necessary *because* "copying and converting the inputs below recurse once per nesting level"

**Category:** Security (defense-in-depth gap)

**Description:** the S-2 render-side depth check was added as defense in depth behind the API's body-parse cap — but the API's prompt builder *already* recursively walks and deep-copies the same JSON in `escape_json` before render is ever reached, with no depth guard (`json_nesting_depth` is exported at `output_parser.h:51` but never called from `prompt.cpp`). HTTP traffic is protected today only by `ServerConfig::max_json_depth = 64` at body-parse time; the documented second layer does not cover the first recursive pass. `build_chat_prompt` is also a public library entry point (`include/halo/api/prompt.h`), so non-HTTP embedders have no protection at all.

**Impact:** stack-overflow DoS for any caller that skips the body-parse cap; a false statement in the security model ("defence in depth").

**Recommended fix:** call `json_nesting_depth(messages/tools, kMaxJsonDepth)` at the top of `build_chat_prompt` (iterative walk, cheap), before escaping.

---

### [MEDIUM] M10 — The prefix-cache machinery works, but D-013's message-boundary hints have no producer, and the planner contradicts the engine on checkpoint memory

**Location:** `include/halo/runtime/engine.h:72` (`checkpoint_hints`) — **zero references in `src/api/`** (grep-verified; only engine + tests); `src/runtime/engine.cpp:672-695` (hints consumed); `src/memory/planner.cpp:347-354` (checkpoints "GPU-tier only", config error otherwise) vs `src/runtime/engine.cpp:302-316` (engine then budgets checkpoints on the **host**, "outside the plan", when no GPU tier exists)

**Category:** Requirements gap (D-013) / correctness of reporting (FR-004)

**Description:** three linked problems in the headline agentic feature:

1. **No producer.** D-013 commits to checkpoints "at the end of the system/tool preamble, at user-message starts… The template returns message-boundary offsets." The engine consumes `checkpoint_hints` and the renderer has the message offsets — but nothing in the serving path passes them. In production, only spacing-based (`ckpt_spacing_`) and N−tail checkpoints ever exist, so multi-turn hit quality is far below what D-013 describes. The engine-level hint behavior is tested only by manual injection (`test_engine.cpp:376-389`).
2. **Planner/engine contradiction.** On a host-only machine the planner reports prefix checkpoints as disabled/config-error (GPU pool only, per D-013), while the engine silently allocates a host checkpoint budget "outside the plan" (`engine.cpp:309-314`). `halo inspect`'s memory report therefore understates the runtime footprint by the checkpoint budget — FR-004's "startup budget breakdown" is wrong exactly where D-002's 96 GiB/32 GiB topology makes memory accounting matter most. (Cross-check at `engine.cpp:317-327` only fires when `max_memory_bytes` is set.)
3. **D-013's own text is violated.** "Checkpoints use a planner-owned budget in the GPU pool, not the 32 GiB OS pool" — the engine's host fallback does precisely that on the actual target topology.

**Impact:** (1) the D-014 NFR "cached TTFT ≤ 300 ms for 24K re-submit and 32K multi-turn replay at ≥ 80% hits" cannot be met as specified and has never been measured against real agent traffic; (2) operators get a wrong memory plan.

**Recommended fix:** wire `ChatPrompt`'s message-boundary offsets into `GenerateRequest::checkpoint_hints` at the API/engine boundary; make the planner own the host fallback budget so inspect and engine agree; reconcile the code with D-013 (or amend D-013 explicitly).

---

### [MEDIUM] M11 — The only real-scale validation is 3 prompts × 32 greedy tokens, manual, with the harness outside the repo

**Location:** `docs/dev/smoke-27b.md` (verdict PASS; scope: "prompts are short (at most 23 tokens)", "the numbers were measured on a shared dev host"; harness "in the session scratchpad"); `memory.md:88` (the real run is an MVP exit criterion)

**Category:** Testing — the gap between "code-complete" and "correct at scale"

**Description:** the single execution of HALO against the real 17.5 GB UD-Q4_K_XL pack covered three short prompts, 32 greedy tokens each, MTP on/off equivalence, on a CPU. It found 96/96 token agreement with llama.cpp (genuinely valuable, including margins down to 0.033 nats). It did not touch: any context beyond 55 tokens, chunked-prefill at scale, KV paging under pressure, prefix cache at scale, sampling (non-greedy) paths, structured output, concurrent sequences at 27B, or any API surface. The smoke doc itself lists these under "Not verified". Meanwhile the exit criterion in `memory.md` §3a requires this run for MVP — a criterion met by the narrowest possible reading.

**Impact:** every behavior that scales with context length, batching, or sampling is validated only on a 256-hidden/4096-ctx tiny model whose real-MTP acceptance is 0. The first realistic agent workload on the EVO-X2 will be the first test of most of the serving surface.

**Recommended fix:** promote the smoke harness into `scripts/` (or `tests/`) and add: one 4–8K-context prompt, one seeded sampling run, one 2–4 concurrent-sequence run, one long-document prefix-reuse case — all at 27B, all on the CPU path that exists today.

---

## Low Findings

### [LOW] L1 — Signed-overflow UB in `GgufTensorInfo::rows()` reachable from a hostile file
**Location:** `include/halo/model/gguf.h:93` — `return ne[1] * ne[2] * ne[3];` called at `src/model/gguf.cpp:418`.
If `ne[0] == 0`, the full-product overflow check passes trivially, and `ne = {0, 2^40, 2^16, 2^7}` yields `2^63` in `rows()` — UB in a parser whose header promises "nothing in this parser relies on the file being well formed for memory safety". Fails closed in practice on x86-64 (wrap → huge u64 → `checked_mul` throws or a bogus 0-byte tensor; no OOB found). Fix: compute in `uint64_t` or reject zero dims. Severity limited by the typed-error layers behind it.

### [LOW] L2 — Negative `temperature` silently means greedy
**Location:** `include/halo/sampling/sampling.h:28` (`greedy() = temperature <= 0.0f`); `src/sampling/sampler.cpp:46-59` validates finiteness but not negativity. `-1.0` decodes greedily instead of 400. Adjacent: tiny positive temperatures (1e-40) divide logits to ±inf and degrade to lowest-id argmax among +inf ties (`sampler.cpp:111-114,209-217`) — no crash, surprising semantics.

### [LOW] L3 — `default_max_tokens` is not cross-checked against `max_tokens_cap`
**Location:** `src/api/server.cpp:984-985` (both > 0, no ordering check); `tools/halo/config.cpp:109-112`. An operator who lowers the cap but leaves the default finds requests *without* `max_tokens` generating up to the old higher budget — the cap's intent silently bypassed on the default path.

### [LOW] L4 — KV-only cache entries (checkpoint insert failed) can never be matched and hold pool blocks hostage
**Location:** `src/runtime/engine.cpp:971` (comment admits it); `state/checkpoint.cpp:12` (oversized checkpoint rejected). Such a `CacheEntry` can never satisfy `admit` (which requires a live checkpoint, `engine.cpp:642-664`), so it occupies KV blocks that `ensure_capacity` can only reclaim by evicting *other* entries. Bounded by `cache_cap_` count eviction — no leak, but under pool pressure the engine may evict good entries while dead weight remains.

### [LOW] L5 — Prefix-cache admission is O(cache_entries × prompt_len)
**Location:** `src/runtime/engine.cpp:637-639` (LCP scan over every cache entry per admission), plus `state/checkpoint.cpp:49-61` (linear find). With dozens of cached 24K-token agent prompts, every new request scans all of them; `CheckpointStore::insert` is likewise O(entries). Correctness verified (exact token comparison); this is a scaling cliff for the exact agentic workload the cache exists for.

### [LOW] L6 — Profit-gate speedup accounting is optimistic on short generations
**Location:** `src/speculative/speculative.cpp:353` records `accepted + 1` before the `max_tokens`/stop cap (`speculative.h:104-106` documents it). `GateMode::Auto` disable decisions are biased toward keeping speculation on for workloads that mostly hit token caps. By design, but the bias direction is worth noting.

### [LOW] L7 — S-39 and S-45 are also fixed only uncommitted
Same state as M3: the `touched < t_snap` erase condition (`conn_guard.cpp:135`) and the `dup()`-based verified shutdown (`conn_guard.cpp:232-255`) exist solely in the working tree. At HEAD the phase-loss race (can kill a long non-streaming generation at the header deadline) and the fd-reuse TOCTOU remain.

### [LOW] L8 — Documentation drift: stale cancellation text and a stale F-1 test comment
`docs/api.md:203-216,181` still claims "Nothing can reach a request that is still prefilling… This needs an engine-side cancel path, which WS-G is adding" — that path landed (`engine.cpp:501-504,544,579`); the docs understate a shipped security control. `tests/unit/integration/test_integration.cpp:427-428` still says the F-1 test "fails until F-1 is fixed" though the skip was removed and the fix landed in `e6ce83e`.

### [LOW] L9 — Script-hardening residuals
`scripts/package.sh:92-107` takes `git_sha` from `SOURCE_INFO` without format validation (flows into paths and a sed replacement; bounded because a crafted `SOURCE_INFO` implies the attacker already supplied the source tree). `scripts/wsl-run.sh:5-7` interpolates `$1` into a log path without validation (script already runs arbitrary `"$@"` as root, so impact is hardening-only). `python/tools/fetch_*.py` fetch reference goldens from HF `resolve/main` with no hash pinning (mutable upstream → poisoned goldens; defended at consumption by byte-comparison for quant blocks, dev-only).

---

## Informational Findings

- **I1 — SSE content-provider exceptions are swallowed silently.** In httplib 0.57.1 only `routing()` is exception-wrapped; the chunked content provider (HALO's stream lambda, `server.cpp:600-622,656-671`) runs outside it. A stray exception there reaches the connection-level `catch(...)` but HALO installs no error logger, so the drop leaves no trace. No crash; an observability hole. Fix: `svr.set_error_logger(...)`.
- **I2 — Structured-output grammars compile twice per request, pre-admission.** `checked_schema` compiles at parse time (`requests.cpp:134-147`, before `admit()` at `server.cpp:572`); the engine compiles again. Worst hostile schema ≈ 0.5 s each per the 2026-09-25 review — up to ~1 s of pre-admission CPU per structured request (bounded by the 64 KiB schema cap). The double-compile is intentional (fail before streaming); the placement is TD-6-adjacent.
- **I3 — `/tokenize` response amplification.** 4 MiB of content (the cap) → ~4M token ids → a ~30 MB JSON response, ×2 concurrent + 8 queued utility slots. Bounded, but ~8× input-to-memory; never stress-tested.
- **I4 — Non-Linux builds have zero slowloris defense.** All of `ConnectionGuard` is `#if defined(__linux__)` (`conn_guard.cpp:38,50,189,257`); documented in `conn_guard.h:44-45` and `api.md:130`. Re-assert before any non-Linux serving claim.
- **I5 — S-5 SIGBUS on concurrently-truncated mmap is real and accepted.** `src/model/mapped_file.cpp:38-46` trusts `fstat` size; no lease, lock, or handler. Documented as out of scope (`security-hardening.md:429-437`) with mitigation assigned to the download path. Worth re-accepting explicitly before remote model management exists.

---

## Security Review

**Attack surface.** A loopback-by-default HTTP API (OpenAI + Anthropic + utilities) in front of a single-tenant inference engine; three hostile-input parsers (GGUF, tokenizer.json/GGUF vocab, embedded Jinja template); CLI/config surface; packaging scripts; dev-tooling fetchers. There is no multi-user authorization model — the security posture is "local process or explicitly-proxied remote", which D-017 states honestly.

**What was re-verified as solid (with the disproof noted):**
- **Loopback-only bind** is enforced in `validate_server_config` (`server.cpp:956-989`) before the model loads; `is_loopback_host` (`:998-1016`) is strict (rejects `127.1`, `127.0.0.2`); the only override is the explicit `allow_remote` flag.
- **Auth ordering**: auth and Host/Origin checks run in the pre-routing handler *before* the body is read (httplib `routing()` order verified against upstream source). API-key compare is constant-time.
- **Request parsing**: depth cap enforced inside the SAX callback before DOM materialization (`json_util.cpp:178-214`); duplicate keys rejected; invalid UTF-8 and `1e999` → 400; everything lands in a catch chain (handler-level, httplib exception handler, connection-level `catch(...)`); no uncaught-exception crash path found.
- **Input bounds**: `max_tokens ∈ [1, min(cap, INT32_MAX)]`; all floats `std::isfinite`-checked; stops ≤ 16 × ≤ 256 B; schema ≤ 64 KiB; grammar-compiled at parse (worst hostile ≈ 0.5 s); no unbounded request value found.
- **Special-token injection** (S-1): closed on every chat route, re-confirmed by code reading (escaper `prompt.cpp:174-244`).
- **Path traversal**: no request field reaches a filesystem path; model path is operator-side only.
- **Secrets**: none in git history (blob sweep; largest file ever committed is 104 KB); `HALO_LOG_FILE` hardening verified by tests (symlink/ELOOP, FIFO, hardlink, owner refusal — `test_log_env.cpp:218-271`); env redaction in the field kit is name+value-pattern based with fail-closed exclusion (residuals in the 2026-09-26 review, S-44).
- **Dependency supply chain**: all four FetchContent deps hash-verified by CMake before extraction; the minja patch is SHA-256-checked against a constant in `HaloDeps.cmake`, idempotent, fail-closed, and uncompilable if unpatched (`#error` on `MINJA_HALO_LIMITS`); no `GIT_REPOSITORY`/`file(DOWNLOAD)`/`ExternalProject` anywhere else.
- **Template sandbox**: pinned minja exposes no `fromjson`, no include/import, no host objects (no `__globals__`-class escape exists — there are no Python objects); render budgets are thread-local, non-zero by default at both construction sites.
- **Parser memory safety**: budget-charged GGUF metadata (1 GiB, stored-size charging), dense-id tokenizer validation before id-sized allocation, trie-based added-token matching — no memory-safety bug reachable from a hostile file survived disproof (the one UB found is L1).

**Residual security items**: S-38/M3 (uncommitted fix), M7, M8, M9, I4, I5; plus the accepted D-017 posture that pre-auth slow-header defense is Medium on loopback and *requires* a proxy for `allow_remote` — the 2026-09-26 review's rolling-reconnect evidence stands.

**Weakest boundaries:** (1) the engine's single-worker trust assumption — one bad sequence's in-step error kills the batch (H1); (2) process lifecycle (H2); (3) the packaging scripts' loose modes (M8) on a machine where the operator runs as root.

---

## AI / Agent Red-Team Review

HALO is not itself an agent; it is the **inference backend that coding agents trust**. The adversarial question inverts: *the model file is hostile input, the model's output is attacker-influenced content, and the clients are agents that will act on whatever HALO returns.*

| Attack | Possible? | Protection | Evidence | Risk |
|---|---|---|---|---|
| Prompt injection via model file (chat template) | Yes — template ships in GGUF | Pinned+patched minja; render budgets (steps/depth/output/alloc); demonstrated-red test table | `security-hardening.md:20-201` | Low (bounded to ~4 s CPU) |
| Prompt injection via chat `messages` content | Yes — untrusted content rendered through the template | Same budgets; special-token escaper (S-1); JSON depth caps | `prompt.cpp:174-244` | Low |
| Special-token smuggling (`<|im_start|>` in user text) | Was possible; closed | Escaper on every chat route; re-verified | S-1 review, 2026-09-25 | Low |
| Tool-call argument injection (model emits hostile args) | Yes — model output parsed into tool calls | Output parser bounded (depth 64, `parse_json_bounded`); deep args stay raw strings | `output_parser.cpp:147-159` | Low |
| Grammar/structured-output DoS (hostile schema) | Bounded | 64 KiB cap; parse-time compile ≈ 0.5 s worst; admission before render | `requests.cpp:80-86,134-147` | Low (but pre-admission CPU, I2) |
| Model-file memory corruption | Not found | Budget-charged parser; `checked_mul` everywhere; fuzzed (no oracle) | L1 is the only residual | Low |
| Cross-request state contamination (prefix cache shares KV/GDN across requests) | Would be catastrophic; **not possible as coded** | Sharing requires exact token-prefix match + owner liveness + checkpoint token comparison | `checkpoint.cpp:49-61`, `engine.cpp:642-664` | — |
| Agent-loop / cost amplification against HALO | Via API: max_tokens cap, queue caps, request timeout | `requests.cpp:88-104`; admission; engine deadlines | verified | Low |
| Data exfiltration via API | No request field reads files; error messages sanitized (512 B cap) | `server.cpp:556-562,873`; `json_util.cpp:122-139` | H1 (message detail leak) is a residual hypothesis — engine `Error(Api)` texts (token counts, no paths) may reach clients as 400 detail | Low |
| Malicious *server operator* shipping a tampered kit | Yes — build-side | Kit tarball SHA256SUMs; anchor ownership/mode checks; source overlays path-validated | `package.sh`, `common.py:346-380` | M7/M8 residuals |

**Explicit answer to the mandated question** — "if the LLM is malicious/compromised/wrong, what technically prevents damage?": model output can at worst (a) produce wrong tokens to an agent client (outside HALO's control), (b) burn bounded compute (token caps, budgets), or (c) kill its own sequence via a sampled out-of-vocab pad id (M2 — visible failure, no corruption). It **cannot** corrupt other sequences' state (per-slot isolation verified), read files, escape the template sandbox, or inject tokens into the text stream (S-1 escaper). The one genuine multi-tenant hole is H1: a model-induced NaN kills *other* sequences as collateral.

---

## Reliability Review

- **Failure handling**: typed errors, per-sequence output-phase isolation, checkpoint best-effort with always-recompute fallback (D-013) — good. The two structural holes are H1 (in-step errors are tick-wide) and M1 (the retry path assumes atomicity the forward doesn't provide).
- **Retries**: exactly one retry path (Memory → evict + retry), and it is unsafe as shown in M1. No other retries; no retry storms.
- **Timeouts**: header 10 s / body 60 s per-connection deadlines (Linux-only), request timeout, queue timeout, SSE write timeout 60 s, per-request stop-token watcher at 100 ms. Coherent.
- **Idempotency**: requests carry no idempotency keys; duplicate submissions are safe (prefix cache makes a re-submit cheap) but produce duplicate generations — acceptable for this API shape.
- **State recovery**: rollback slots verified correct (slot indexing, `commit_rows_kept`, conv offsets); "verify + rollback to r == decode r tokens" is a tested invariant; the fed-vs-KV-length invariant runs every tick and converts state corruption into sequence failure.
- **Process crash**: SIGTERM/SIGINT → graceful stop, except H2's UAF makes "graceful" unreliable with in-flight traffic.
- **Dependency failures**: the only runtime network dependency is none — fully self-contained serving. (A virtue.)
- **Data consistency**: prefix-cache sharing requires exact token match; eviction under memory pressure is count-bounded; dead KV-only entries (L4) are the one wart.

---

## Performance Review

- **Algorithmic**: prefix-cache admission O(cache × prompt) (L5) is the one scaling cliff that matters for the agentic target workload; `CheckpointStore::insert` O(entries) dup-scan. Everything in the hot decode path is linear in weights (as it must be).
- **Known unmeasured costs** (TECH_DEBT): TD-1 (K× state traffic per verify vs. designed K−1 — *confirmed real*, test asserts the wasteful 4× for K=4), TD-3/4/7/8/10/11 (HIP kernel efficiency, all "unmeasured"), TD-7 (full bitonic top-k over 248K logits).
- **Profit-gate accounting bias** (L6) can keep speculation enabled where it loses.
- **CPU/GPU contention** is unanalyzed by construction on the dev host (D-001); the entire TRD §55 cost model runs on microbenchmarks that have never executed on the target.
- **Pre-admission CPU** (I2 grammar compile; template render on the HTTP worker — TD-6's bounded-pool fix is still open per TECH_DEBT.md:12) is a per-request latency tax under concurrency.
- **No performance claim in the repo should be read as a HALO number** — this is stated (D-001, smoke doc) and this review found no violation of that discipline.

---

## Testing Review

**Adequately tested (independent references, honest tolerances):** CPU forward vs. transformers float32 goldens (`test_qwen35_golden.cpp`, a-priori tolerances with near-tie softening documented at `:4-22,203-209`); tokenizer/template vs. HF (token-for-token; hostile-template death-test harness with CPU-time budgets); MTP draft-chain vs. llama.cpp recordings; conn_guard regressions incl. a genuine dup2 descriptor-reuse race test (`test_api_guard.cpp`); rollback equivalence; KV CoW; sampler chains; `HALO_LOG_FILE` hardening.

**Weak / false-confidence (evidence in each finding above):**
- M5: circular quant differential (`dequantize_row` = reference = implementation).
- `MtpInPiecesEqualsOneShotAndDraftChainIsSelfConsistent` (`test_qwen35_golden.cpp:449-454`) — self-consistency where a golden is owed.
- **Oracle-driven accept paths**: every MTP accept/reject test forces drafts (`test_speculative.cpp`); real acceptance at engine level is exercised only by the manual 27B smoke (M11).
- **Timing-premise tests**: `test_api_guard.cpp:348,356,414-440` and `test_api_security.cpp:635,666` sleep fixed durations then assert load-generation premises on a host documented to run at load average 12–35 — flake factories with no CI to catch drift.
- **Hang-not-fail busy-waits**: `test_engine_faults.cpp:211,333,337`; `test_engine.cpp:213` — no deadlines; a regression consumes the 1500 s ctest timeout instead of failing.
- **HIP device tolerance is a guess** (`kDevRel = 1e-5f`, "unmeasured, to be confirmed on the EVO-X2", `hip_test_util.h:94-98`), and the no-device negative tests **skip when a device appears** (`test_hip_backend.cpp:1046`) — they self-disable exactly where they matter.

**Not tested at all:** the shipped server binary as a process (every API test is in-process over a fake or real engine); cancellation storms against the real engine/server (only a ManualClock fake); mid-forward error injection (architecturally impossible with today's four hooks — which is why H1/M1 are invisible to the suite); long-context/RoPE boundaries (all engine tests ≤150-token prompts, ctx ≤1024); malformed-but-parseable GGUF reaching engine creation (fuzz asserts only "no crash + some typed error", no oracle); structured output + MTP under the API beyond one case; D-013 message-boundary hints in serving (no producer, M10).

**The "545 tests, 543 pass, 2 skipped" claim** is a dev-host-only artifact: 64 `GTEST_SKIP` sites in 36 files gate ~110 tests on the `~/halo-ref` fixture directory — roughly a fifth of the suite, including nearly all forward/engine/model coverage — and all Vulkan suites additionally skip without an ICD. On a fresh clone the number is far lower and the skips are silent.

---

## Documentation / Implementation Drift

| Document says | Code does | Verdict |
|---|---|---|
| PRD §9: ceiling 66 tok/s, targets ≥45/≥70/≥120 tok/s | TRD/D-011: 15.2 tok/s ceiling; targets need ≥742 GB/s | ⚠️ contradiction (M4) |
| `docs/api.md:203-216`: "Nothing can reach a request that is still prefilling… engine-side cancel path, which WS-G is adding" | Cancel checked every tick in every state (`engine.cpp:501-504,544,579`) | stale (L8) |
| `docs/security-hardening.md`: S-2 render-side depth check as defence in depth | `escape_json` recurses before that check (`prompt.cpp:200` vs `chat_template.cpp:478`) | ⚠️ gap (M9) |
| D-013: checkpoints at message boundaries via template offsets; GPU-pool budget only | No hint producer; host fallback budget outside the plan | ⚠️ gap (M10) |
| `test_integration.cpp:427-428`: F-1 "fails until fixed" | Fixed in `e6ce83e`; test runs | stale (L8) |
| 2026-09-26 review: S-38 open, S-39/S-45 open | Fixed in uncommitted tree | ⚠️ state confusion (M3/L7) |
| `memory.md`: "Prefix cache Done — checkpoint_hints has no producer yet" | Accurate; engine reuse works on CPU (`test_engine.cpp:340-389`) | ✅ accurate (and this review confirms the engine-side works) |
| Smoke doc: "planner notes say prefix checkpoints disabled (no GPU tier)" | Engine budgets them on host anyway (`engine.cpp:309-314`) | ⚠️ planner/engine contradiction (M10) |
| README: "545 tests, 543 pass, 2 skipped" | ~110 tests fixture-gated; not reproducible fresh | ⚠️ (M6) |

---

## Attack Scenarios

### Scenario 1 — SIGTERM during active traffic (H2)
**Trigger:** operator restarts `halo serve`; a client holds an SSE stream with a stalled reader.
**Expected:** in-flight generations cancel; the process exits cleanly.
**Observed (mechanism-verified):** `stop()` closes the listener and joins only the listen thread; the blocked worker (up to the 60 s write timeout) keeps running; `~ApiServer` destroys `metrics` and `guard` before httplib's pool is joined (`server.cpp:386-390`, `commands.cpp:565-572`). The worker's logger / `~StreamPhase` then touches freed objects.
**Impact:** crash/UB on the most common ops action.
**Mitigation:** `wait_until_gracefully_stopped()` in `stop()`, or reorder members.

### Scenario 2 — One pathological request kills every agent stream (H1)
**Trigger:** four concurrent clients; one request drives the Q6_K LM head to a NaN row, or host memory pressure makes a mid-forward allocation fail.
**Expected (D-017):** the offending sequence fails; the other three commit.
**Observed:** `Error(Kernel)`/NaN escapes the tick → `fail_all` retires all four; `bad_alloc` does the same via `catch (...)`.
**Impact:** total in-flight work loss; PRD §48/RR-005 violated.
**Mitigation:** commit-after-status ordering (ADR-001 WS-BI-2) + mid-forward fault injection tests.

### Scenario 3 — Rolling-reconnect slowloris (S-38, at HEAD)
**Trigger:** any local process (100 sockets, dribbled headers, reconnect on close) against a HEAD build.
**Expected:** `/health` and the API stay available.
**Observed (reproduced by the 2026-09-26 review):** `/health` 1/79 available for 40 s; 94,558 attacker reconnects. The per-peer cap + shedding that fix this exist only uncommitted.
**Impact:** full DoS of a local service by an unprivileged user. Graded Medium per D-017 loopback scope.
**Mitigation:** commit WS-I M6; per-peer caps; proxy for any `allow_remote`.

### Scenario 4 — Memory pressure during MTP decode (M1)
**Trigger:** 27B model, MTP on, OS pool nearly exhausted; a head-section allocation throws `Error(Memory)` (or `bad_alloc`).
**Expected:** evict a cache entry, retry, continue.
**Observed:** for `Error(Memory)`, the retry runs with KV already advanced → wrong RoPE positions → the fed/length invariant kills the sequence; for `bad_alloc`, `fail_all` kills all sequences.
**Impact:** the designed-for recovery path converts into failures exactly under the pressure it was built for.
**Mitigation:** move commit after head; add a mid-head fault hook.

### Scenario 5 — Greedy generation dies mid-stream on the real pack (M2)
**Trigger:** 27B pack whose LM head is zero/untrained-padded past token 248,077; a decode step where all real logits are negative (long tail contexts make this plausible).
**Expected:** the best real token is emitted.
**Observed:** the fused argmax (or sampler) selects a pad-row id; `emit` → `check_id` throws; the sequence fails mid-generation.
**Impact:** spurious request failures, worst exactly where logits are flat. Unverifiable until real weights are read — which is why it must be fixed before, not after, the EVO-X2 runs.
**Mitigation:** clamp candidates to the tokenizer vocab.

### Scenario 6 — Kit packaging leaks the operator's environment (M8)
**Trigger:** developer with a `.env`/key/DB in the repo dir runs `package.sh --source worktree` (or `copy`).
**Expected:** the kit contains the source tree and nothing else.
**Observed:** untracked non-ignored files (worktree mode) or *everything* except four exclusions (copy mode) are staged, tarred, SHA256SUMS-signed, and shipped; untracked filenames land in BUILDINFO.
**Impact:** integrity-endorsed exfiltration of whatever sits in the repo directory — to a machine the field kit then runs as root.
**Mitigation:** gitignore coverage; make `copy` honor it; secret-pattern refusal.

### Scenario 7 — In-place model update while serving (I5/S-5)
**Trigger:** an operator (or a download script without atomic rename) overwrites the GGUF that a running `halo serve` has mmap'd.
**Expected:** graceful error.
**Observed:** `SIGBUS` on the next tensor read; process death. Accepted/documented, but nothing technical prevents the operator mistake.
**Mitigation:** advisory lock (`flock`) on the model file at load; document the atomic-rename requirement at every download path.

---

## Recommended Remediation Plan

### Immediate (before any serving claim beyond loopback dev use)
1. **H2:** `svr.wait_until_gracefully_stopped()` in `ApiServer::stop()` (or member reorder). One line + a regression test with a blocked socket. Tradeoff: stop() latency grows up to the write timeout — acceptable, and configurable.
2. **H1/M1 (plan, don't rush):** schedule the commit-after-status reorder as the *first* ADR-001 WS-BI-2 task, and add a mid-forward fault hook to the engine's fault-injection surface. Doing H1 without M1's reorder recreates the retry-corruption trap. Tradeoff: commit-after-status may cost an extra state copy on the GPU path — measure on the EVO-X2, but correctness first.
3. **M3/L7:** commit the staged WS-I M6 conn_guard work (it is already written, tested, and documented) so HEAD stops carrying reproduced vulnerabilities.
4. **M2:** clamp unstructured sampling/argmax to the tokenizer vocab — a few lines, removes a latent whole-request failure class before first hardware run.
5. **M4:** revise PRD §9–§10 to the D-011/D-014 model so nobody plans against impossible numbers.

### Short term (next development cycle)
6. **M9:** depth-check in `build_chat_prompt` (closes the documented-but-false defense-in-depth claim).
7. **M7/M8:** llama.cpp pin enforcement; packaging gitignore/secret refusal. Both are small script changes with outsized blast-radius reduction.
8. **M10:** wire message-boundary `checkpoint_hints` through the API/engine boundary; reconcile planner vs. engine checkpoint budgets (one owner: the planner).
9. **M5:** independent transformers golden over dequantized weights (one offline run) to un-circularize the quant differential.
10. **Test hardening:** deadlines on busy-waits; replace sleep-premise tests with synchronization; promote the 27B smoke harness into the repo and extend it per M11; make fixture absence fail loudly in CI (M6).
11. **L1–L6, L8–L9:** the small correctness/robustness items (u64 `rows()`, negative temperature, cap cross-check, dead-entry reclaim, admission indexing, gate accounting, doc sync).

### Long term (architectural / with hardware)
12. **M6/CI:** stand up the minimal CI lane now (CPU build + fixture-cached ctest + ASan); add the self-hosted gfx1151 runner per TRD §60 once the EVO-X2 is stable. The GPU correctness lane must not become the *first* automated gate after months of manual verification.
13. **HIP/Vulkan validation debt:** replace the guessed `kDevRel` with measured tolerances; fix the no-device tests that self-disable when a device appears; execute the lavapipe→RADV and emulation→device differentials.
14. **Performance truth:** execute TRD §68 Phase 0 (roofline, load-path, per-tier bandwidth) before any further kernel optimization is budgeted — the TD-3..TD-11 items are all "unmeasured" and should be ranked by the measured roofline, not by intuition.
15. **TD-1:** slot-0 skip / pointer-swap for the state ring when the `cpu/ops.h` contract change lands with the GPU backends (already planned; keep it paired).

**Engineering-tradeoff notes:** the strongest recommendation patterns here are (a) *fix ordering* — H1, M1, TD-1, and the fault hooks all touch the same commit/rollback region and must land together or not at all; (b) *verification before optimization* — every perf item is cheaper to rank after Phase 0 than before; (c) *process over code* — M6 (CI) has the highest expected value per dollar of anything in this plan, because every other protection in this repository currently executes only when a human remembers it.

---

## Final Assessment

1. **Most dangerous failure modes:** (a) one sequence's in-step error killing every concurrent stream (H1) — the exact multi-agent workload HALO targets; (b) shutdown-time UAF crashing the process on SIGTERM with traffic (H2); (c) the Memory-retry path corrupting position state under memory pressure (M1); (d) a latent padded-vocab argmax killing generations on the real packs (M2).
2. **Assumptions the system relies on:** CPU-as-oracle (solid vs transformers, circular for quant — M5); tiny-model-exercises-27B-paths (weak — M11, oracle drafts, no padded head); loopback posture (honest, D-017); human-in-the-loop verification (no CI — M6); hardware arriving before performance claims (true so far).
3. **Unsupported assumptions:** that the documented S-2 defense-in-depth covers the escaping pass (M9); that PRD targets are meaningful (M4); that `halo inspect` reflects runtime memory (M10); that "545/543" means something off the dev host (M6); that HEAD carries the security fixes (M3).
4. **Requirements not fully implemented:** D-017 per-sequence isolation (H1); D-013 message-boundary checkpoints (M10); TRD §60 CI (M6); FR-004 accurate budget reporting (M10); PRD §9 correctness (M4). All others checked are implemented as specified.
5. **Weakest security boundaries:** the single-worker engine's tick-wide failure domain; process lifecycle vs. the HTTP thread pool; the packaging scripts' loose modes on a root-operated machine.
6. **Tests providing insufficient confidence:** the quantized-forward differential (circular); all MTP accept paths on the tiny model (oracle-forced); the API suites (in-process, fake-engine, no binary-level test); the HIP device suites (guessed tolerance, self-disabling negatives); every timing-premise guard test.
7. **Largest production incident potential:** restarting the server under load (H2) is the most *probable*; a memory-pressure event during a busy agent session (H1+M1) is the most *damaging*; shipping a kit built from the working tree (M8) is the most *embarrassing*.
8. **What an adversarial user/agent could exploit:** on HEAD, local pre-auth DoS via rolling reconnects (S-38); via the API, only bounded resource consumption — the input-validation and template-sandbox layers held. A hostile model file gets bounded CPU, nothing more (L1 excepted).
9. **Validate before any production deployment:** the H2 shutdown path under a stalled-reader test; H1 isolation with a mid-forward fault hook; M2 against the real pack's head rows; the full prefix-cache path against real multi-turn agent traffic (D-014 NFRs); the packaged kit from a clean clone (M6/M8); and, on hardware, Phase 0 before any performance statement.

**This repository does not claim to be production-ready, and this review does not grant it.** What the review *does* establish: the code is unusually honest about its own gaps, the security engineering that exists is real and verified (not checkbox), the parsers are genuinely hardened, and the defects found are concentrated in fault-path ordering, process lifecycle, and process discipline — all fixable with the plan above, and all cheaper to fix before the hardware arrives than after.
