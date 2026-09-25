# WS-J status (attempt #3 agent)
Build: scratchpad/build_J.sh [--asan] (WSL; HALO_BUILD_DIR=/root/halo-build-wsj; log /root/wsj-build.log)
## Milestone 1 (profiling) — COMPLETE, awaiting orchestrator commit (do NOT start M3 until told)
- Baseline at resume: configure FAILED (tests/unit/profiling/CMakeLists listed 6 nonexistent test sources). Library sources (predecessor) complete, never compiled.
- Changes by this agent: suite.cpp before-snapshot now captured at construction (was stamped at finish); captured_at injectable (hw_state make/capture_hardware_state param + EnvironmentOptions::utc_now); MTP decode derives tau = gen/(gen-accepted) for efficiency, non-MTP modes force n=0,w_mtp=0; workload.cpp <format> include.
- GREEN: all 6 profiling test files written; ctest 162 total (52 profiling at end), 0 failed, 1 skip (Bandwidth.ConformantDefaultsDevHost, not mine). Fixed strict-uint parsing bugs in workload.cpp/record.cpp found by the tests.
- Demonstrate-fail DONE: 23/23 mutations red (scratchpad/mutations_M1.log + mutations_M1_rerun.log); size-bound test gap found+fixed. docs/benchmarks.md (M1) DONE.
- Final: release ctest 162/162 (52 profiling, 0 fail, 1 skip not mine); ASan ctest 162/162 same. Next when told: Milestone 3 (autotune + SQLite), then M2.
- Milestones 2/3 not started (fixtures for M2 already captured in tests/unit/profiling/fixtures/).
- Round 2 (after advisor review): concurrency records no longer get efficiency (prefill-inclusive rate); docs give per-mode metric definitions; 18 more mutations, 41/41 total RED (mutations_M1_round2.log). Verified pristine (check_restored.py). Final release ctest 162/162 + profiling binary 52 passed exit 0; ASan rerun. M1 report sent to orchestrator; WAIT for instruction before M3.
## Milestone 1 COMMITTED as f0e9b91 (llama fixtures held back for M2).
## Milestone 3 (autotune + SQLite) — IN PROGRESS (started after orchestrator go-ahead)
- M3 design (advisor-reviewed): halo_autotune (core+profiling+hardware+SQLite; DB, key, lookup, cost model, tuner) + halo_autotune_cpu (CPU TunableOps, guarded by halo_selected(cpu)). ProfileLookup read-only, in-memory snapshot, no TunableOp/Clock in its API. Typed ProfileDbError(kind). Key: must-match MODEL_HASH PACK_ID GPU_DEVICE GPU_ARCH POWER_MODE ISA_TARGET; tolerant others.
- M3 code + tests written (first draft); first compile next.
- M3 first green: ctest 210/210 (autotune 45 + autotune_cpu 3), 1 skip not mine. Next: no-cpu build check, demonstrate-fail, docs, ASan.
- M3 docs written; demonstrate-fail running: scratchpad/mutate_M3.py -> mutations_M3.log (harness restores files; verify with check_M3.py)
- M3 demonstrate-fail: 33/34 RED (mutations_M3.log + _rerun.log); matmul bit-identity gate failure branch unreachable via API (reported). bayesian test gap fixed. Release ctest 210/210 (autotune 45 + cpu 3). no-cpu build OK. ASan run done (see report).
- Tooling renamed to j_* (j_build_J.sh, j_mutate_M3.py + j_mutations_M3_list.py, j_check_M3.py, j_check_restored.py, j_run_at.sh, j_nocpu.sh; logs j_mutations_*.log). Fixed after advisor: winning_configuration UNIQUE now includes backend (+test+mutation), Busy test added. 36/36 M3 mutations checked (35 red; #33 matmul gate unreachable).
- M3 FINAL: release 212/212 (autotune 47, autotune_cpu 3), no-cpu 164/164, ASan 212/212 0 sanitizer reports. Report sent; WAIT for orchestrator before M2.
## M3 COMMITTED 3672113. ## Milestone 2 (baseline adapters) — IN PROGRESS
- M2 facts: llama-bench is at /root/llama-build-wsj/bin (predecessor build, commit bd4f514, CPU-only); /root/llama.cpp/build/bin has llama-server but NO llama-bench (brief premise wrong). Backend names: Vulkan="Vulkan", HIP="ROCm", CPU="CPU". --version prints to stderr. -fa takes on|off|auto.
- INCIDENT: `ollama.exe list` on the Windows host auto-started the Ollama app+server (was not running); I stopped the 3 processes I started (PIDs 25608,24032,10712). Ollama 0.17.6 is installed on Windows with llama3.2 pulled; not in WSL. Ollama fixture will be hand-written from API docs.
- M2 written so far: artifact_builder.h (refactor), sha256.{h,cpp}, subprocess.{h,cpp}, http_client.{h,cpp}. Next: baseline.{h,cpp}, autotune baseline_store, tests, fixtures (ollama hand-written).
- M2 tests written (sha256, subprocess, http, baseline, integration, autotune baseline_store); first build next
- M2 green: ctest 247/247 (profiling 85 incl. 2 real llama.cpp integration tests; autotune 49; cpu 3). Running j_mutate_M2.py -> j_mutations_M2.log, then M1 #9,16,17 re-run (ArtifactBuilder moved to src/profiling/artifact_builder.h).
- M2 mutations: 23/26 red first pass (#22 segfault=red), #4 #15 #25 fixed to compile, rerunning -> j_mutations_M2_rerun.log. M1 #9,16,17 red after ArtifactBuilder move.
- M2 FINAL: 26/26 M2 mutations red; release 247/247 (profiling 85 incl. real llama-bench + llama-server MTP integration, autotune 49, cpu 3), no-cpu 199/199, ASan 247/247, 0 sanitizer reports. No stale processes. Report sent; M2 boundary.
- M2 FINAL (after pack_hash + docs): release 246 pass/1 skip, ASan+nocpu rerun below; 27/27 M2 mutations red. Report sent.
## M2 COMMITTED 8b40f46. ## M4 (security fixes S-29..S-37) IN PROGRESS. RULE: never launch programs on the Windows host (WSL only).
- M4 fixes implemented (subprocess rewrite: Child/waitid WNOWAIT, redaction, closefrom, log 0600 NOFOLLOW; http content_type; CLOEXEC reads; baseline port probe+ownership+argv checks). Builds. Testing next.
- M4 tests green (profiling 96). Running j_mutate_M4.py -> j_mutations_M4.log
- M4: added fake_server helper + S-29 layer-2 test. Rerunning M4 mutations 0 1 11 14 15 -> j_mutations_M4_rerun.log
- M4 FINAL: release 263 pass/1 skip (264), ASan same, no-cpu 211 (210+1 skip); 19 M4 mutations (17 red, #11 and #15 green: see report). Orphan from mutation #17 found+killed; helper grandchild now alarm(30). Report sent; stop.
