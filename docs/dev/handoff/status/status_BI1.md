# WS-BI-1 status

Start commit (pinned for the private tree): 9babebe80ea15cdc0879a6c37f3ba88bf48d786b

## Milestones
- M1 (in progress): halo_backend interface + CPU adapter + tests/unit/backend. No models change.
- M2 (drafted in scratchpad bi1_m2/, not built yet): Qwen35 over Backend&, legacy copy,
  bitwise gate test, mutations, ASan, full ctest.

## Baseline (measured)
- Private tree /root/bi1-base = git archive 9babebe; build dir /root/halo-build-bi1-base.
- HALO_ONLY=tensor;model;cpu;tokenizer;template;memory;hardware;kv_cache;state;models;speculative;sampling;runtime;autotune;profiling
- ctest: 455 total, 454 passed, 0 failed, 1 skipped (Bandwidth.ConformantDefaultsDevHost:
  "set HALO_BANDWIDTH_FULL=1 ..."), EXIT=0. Names: scratchpad/bi1_base_tests.txt.

## M1 evidence so far
- Private tree /root/bi1-src (9babebe + bi1_work/ via rsync -c), build /root/halo-build-bi1,
  same HALO_ONLY + backend: 466 tests = 455 + 11 new CpuBackendTest; first run 2 red (test
  bug: strided rope/attention refs lacked the explicit row stride), fixed; 11/11 green.
- Mutations: bi1_mut_m1.txt via bi1_mut.sh -> bi1_mut_m1.out (running).

## Working files
- bi1_work/ = files to overlay (M1); bi1_m2/ = M2 drafts.
- Scripts: bi1_base.sh, bi1_sync.sh, bi1_build.sh, bi1_quick.sh, bi1_mut.sh, bi1_list.sh, bi1_w.ps1.

## M1 result (2026-09-25)
- Full ctest private tree: 466 total, 465 passed, 0 failed, 1 skipped (same Bandwidth skip),
  EXIT=0. Name diff vs baseline (normalized, GetParam pointers stripped): exactly +11
  CpuBackendTest.*, nothing removed (bi1_base_tests.norm vs bi1_m1_tests.norm).
- ASan+UBSan (HALO_ONLY=tensor;cpu;backend, /root/halo-build-bi1m1-asan): 27/27 passed,
  0 "runtime error" lines in a direct test_backend run.
- Mutations: 19 distinct valid mutations, one at a time, all RED (bi1_mut_m1.out, _m1b.out,
  _m1c.out; M20 of the first list was a setup error, never applied); 3 first
  attempts were compile errors (unused var) and were redone as valid mutations; the first
  alignment mutation survived because the test's 2-byte offset also tripped the range
  check (test fixed: spare float) and the harness had not synced it (harness fixed).
- M1 files copied to the shared tree (compile-clean snapshot).

## M1 final (after advisor review)
- TensorRef::shifted now throws Kernel on over-shift of a bounded view (no longer noexcept).
- New test ValidationRejectsBadShapesAndLayouts + 15 more mutations (bi1_mut_m1d.out), all RED.
  Total valid mutations: 34, all RED.
- Full ctest: 467 total, 466 passed, 0 failed, 1 skipped (Bandwidth), EXIT=0; +12 names, -0.
- ASan: /root/halo-build-bi1m1-asan (20 -fsanitize=address rules in build.ninja), 28/28,
  direct test_backend 12/12, 0 runtime errors.
- bi1_w.ps1 drops dash args -> use bi1_w2.ps1 (verified forwarding).
- Shared tree: M1 files copied, cmp-identical to bi1_work.
- STOPPED at M1 boundary; M2 drafts in bi1_m2/ (unbuilt).
