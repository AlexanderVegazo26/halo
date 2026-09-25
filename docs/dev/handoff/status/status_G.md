# WS-G status (attempt #3)
- M1 2589a6f, M2 71cd610, M3 3c4c71c, M4 committed. CURRENT: M5 (review 526c4cc: R-1, R-2, R-3 cancel path in engine.h, R-5(a), N-1, N-2, N-4).
  M5 (next, after the go-ahead): review 526c4cc findings R-1, R-2, R-3, R-5(a), N-1, N-2, N-4.
- M4 scope: src/runtime (+profile.{h,cpp}), tests/unit/integration (NEW), additive src/models Qwen35Options{gdn_chunk}
  (my module, outside the stated M4 scope: flagged). Do NOT edit src/api or tools/halo (WS-I).
- SCRATCHPAD IS SHARED: only g_* tooling. Private tree /root/wsg-src = HEAD (g_head.rev) + my scope (g_snapsrc.sh, g_sync.sh).
  Build dirs wiped at HEAD 8235c3b. g_ti1.sh <filter|ALL> runs the integration binary; g_mrun4.sh / g_mut_m4*.sh mutations.
- Release (HEAD 8235c3b + scope): 533 tests, 530 pass, 2 fail, 1 skip:
    Api.JsonSchemaWithThinkingOnReturnsTheJsonAsContent = finding F-1 (src/api), intentionally left red;
    TokenizerLimits.SharedPrefixVocabularyEncodesInBoundedTime = WS-L wall-clock flake (1.93 s vs 1.0 s under -j; passes alone).
- Integration: 12 tests (11 pass + F-1). Mutations I1-I11, I5b red (I5 and I11 needed stronger checks; fixed).
- Findings: F-1 structured output + thinking (server.cpp:460/510, requests.cpp:504); F-2 bench concurrency_context not settable;
  F-3 bench decode needs --mtp-draft 0 (documented by the harness note); F-4 tokenizer timing flake; seam: WS-I tune.cpp key recipe.
- Snapshot: scratchpad/g_m4_snapshot.tgz.
- FINAL M4 (private tree HEAD 8235c3b + scope): release 534 tests, 532 pass, 1 fail (F-1, intentional), 1 skip.
  ASan: full suite earlier 0 sanitizer reports (only F-1 failed); integration re-run after the profile-test split 12/13 (F-1), 0 reports,
  slowest mine 391 s. Mutations I1-I13 (I13 equivalent) in g_mut_m4*.out. Snapshot refreshed.
- M5 (2026-09-25 ~07:00): engine.h GenerateRequest::{cancel (std::stop_token), deadline}; cpu_engine.h FaultInjection + N-4 docs;
  engine.cpp R-1 per-seq output isolation + best-effort checkpoint, R-2 noexcept retire/cache split, R-3 polling (pending+active),
  N-2 uncounted evictions; speculative.cpp N-1 drop MTP on failed flush (+Metrics::mtp_dropped); tests/unit/runtime/test_engine_faults.cpp
  (10 tests), N-4 byte check in EveryTick, Spec.FailedMtpFlush test. Mutations E1-E11 red (E4 = terminate, E8 = hang/timeout).
  Release (HEAD 7cf4238 + scope, integration from HEAD): 555 tests, 553 pass, 0 fail, 2 skip. ASan running -> g_asan_m5.out.
  NOTE: WS-I is fixing F-1 in the shared tree and removed the F-1 skip there: tests/unit/integration is no longer synced
  (g_sync.sh), private tree uses HEAD's copy (g_headint.sh). Snapshot g_m5_snapshot.tgz.
- M5 follow-ups done: generate() notify inside mu_ (destruction-safe); drain loop honours stop token/deadline (no buffered
  delivery); GenerateResult::deadline_expired; queued test polls. E12 red, E13 green alone (two independent paths), E13b red.
  D-017 / ADR-001 read: NaN-in-head isolation needs commit-after-status (WS-BI-2 forward, WS-BI-6 engine) -> noted, not forced.
  OWNERSHIP: src/models, include/halo/models, tests/unit/models now belong to WS-BI-1 (M5 did not touch them; verified clean).
  Release final: 556 tests, 554 pass, 0 fail, 2 skip. Final ASan running -> g_asan_m5b.out / .sum. Then REPORT + STOP.
