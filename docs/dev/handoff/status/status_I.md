# WS-I status (attempt #3)
Milestones: M1 = API server (src/api, include/halo/api) + tests/unit/api green+ASan+demo-fail.
            M2 = CLI tools/halo + tests/unit/cli.  M3 = docs/api.md + final report.
Current: M1. Inventory done: src/api/* and include/halo/api/* complete (no truncation), predecessor's
ed1-ed4 A-1..A-12 edits all applied and halo_api built (BUILD_EXIT=0 in pred transcript).
tests/unit/api has fake_engine.{h,cpp} only (no CMakeLists, no tests yet).
Next: build baseline in /root/halo-build-wsi, then write tests/unit/api/CMakeLists.txt + test_api_*.cpp.
- Baseline measured: 151 tests, 150 pass, 1 skip (Bandwidth.ConformantDefaultsDevHost) in /root/halo-build-wsi.
- Wrote tests/unit/api/{CMakeLists.txt,api_test_util.h,test_api_units.cpp,test_api_server.cpp,test_api_security.cpp,fixtures/*}; fixed fake_engine designated-init warning. First full compile pending.
- Build/test script: scratchpad/bt.sh (FILTER=, SUFFIX=-asan, EXTRA=).
- test_api: 64/64 green (release). Product fixes found by tests: is_loopback_host rejected 127.0.0.1 (group count bug) -> rewritten; parse_strict let nlohmann out_of_range (1e999) escape as 500 -> now 400 (found by ApiFuzz).
- Next: full ctest (build.sh), ASan (build.sh --asan), demonstrate-fail mutations (scratchpad/mut.sh), then M1 report.
- Full ctest (release, /root/halo-build-wsi): 215 total, 214 pass, 0 fail, 1 skip (Bandwidth). Running ASan next (bi.sh --asan).
- ASan ctest (/root/halo-build-wsi-asan): 215 total, 214 pass, 0 fail, 1 skip. GREEN.
- Demonstrate-fail: running scratchpad/mut.py mutations_m1.json (results -> mut_results.json, log mut.log). Harness restores + hash-checks each file; if interrupted, check src/api files for mutations (grep "if (false)").
- Demonstrate-fail DONE: 40 mutation runs (mut_m1a.log, mut_b.log, mut_c.log): all red; M05 = SIGSEGV 139 (S-2 real crash w/o cap). M26 (write-only) survived -> redundant alive() check; M26b (both) red. Tree verified clean (hash restore).
- Test fix: j["warnings"] on const json -> j.value(...) (crashed under M28).
- docs/api.md written (folded into M1). Next: final release + ASan ctest, then M1 report. M2 = CLI.
- FINAL M1 (after all fixes): release ctest 215 total/214 pass/0 fail/1 skip; ASan 215/214/0/1. test_api = 64 tests. No mutation left (pattern check over all 3 mutation files: bad=0), no WSL build/test procs running.
- M1 REPORTED to orchestrator; waiting for go-ahead on M2 (CLI tools/halo + tests/unit/cli).
## M2 (started; M1 committed 763892d)
Scope: tools/halo CLI (inspect, devices --root, serve, run, tokenize, template, bench->profiling::run_suite guarded if(TARGET halo_profiling), tune stub), tests/unit/cli; plus replace const-json operator[] in tests/unit/api with checked access.
halo_runtime does not exist yet -> serve/run print "runtime not built" (exit 2?).
- SCRATCHPAD IS SHARED with other WS agents: use only i_* tooling from now (i_bi.sh, i_bt.sh, i_mut.py). M1 mutation provenance verified by content: mut.py (12:03:15) is my JSON driver; mut_m1a/b/c logs have 36/6/4 "=== M.." blocks with my ids and only test_api test names (ApiJson, PromptFixture, ...).
- M2 files so far: tools/halo/{cli.h,args.h,args.cpp,config.h,config.cpp,model_io.h,model_io.cpp}. Next: commands.cpp, bench.cpp, main.cpp, CMakeLists, tests/unit/cli.
- Shared tree had others' in-flight breakage (engine.h 0x97 byte; src/template minja patch). Now building from PRIVATE tree: i_sync.sh (git archive HEAD + my paths) -> /root/wsi-src, build dir /root/halo-build-wsip via i_bp.sh.
- M2 code written: tools/halo/{CMakeLists.txt,cli.h,cli.cpp,args.*,config.*,model_io.*,commands.*,bench.cpp,main.cpp}, tests/unit/cli/{CMakeLists.txt,test_cli.cpp}. First private run: 286 tests, 285 pass, 1 skip (before the at() rewrite).
- (2) done: i_checked_at.py rewrote read subscripts in tests/unit/api/test_api_{server,security,units}.cpp to .at() (backups scratchpad/i_bak/). Needs rebuild+run.
- Next: rebuild, ASan, demonstrate-fail for CLI (i_mut.py with private tree?), smoke run of the real halo binary, docs, report.
- M2 demo-fail: i_mut_m2.json (22 entries: C01..C18 CLI, X01 checked-access proof, M01/M05/M26b M1 re-runs) via i_pmut.py IN WSL on /root/wsi-src. Part 1 (C01-C10 all red) in i_mut_m2_part1.log; the run was killed by my `timeout 598` mid-C11 (private tree only; restored by i_sync.sh). Part 2 (C11..M26b) running -> i_mut_m2b.log. Do NOT use `timeout` for long runs.
- Tests tightened (profiling::to_string mode names, runtime-not-built text for bench model); docs/api.md "Running the server" section added.
- M2 DONE: private tree HEAD 3672113 + WS-I paths: release 286/285/0/1skip, ASan 286/285/0/1skip. Mutations: 22 run, all red after fixing C18 test (initially survived). Smoke of real binary ok. M2 REPORTED; wait for go-ahead.
## M3 (final; M2 committed 6de1366)
(1) tools/halo/tune.cpp: `halo tune` -> halo_autotune (+ halo_autotune_cpu CpuMatmulTunable/CpuGdnChunkedTunable), default exhaustive w/o calibration, --host-label/--power-mode required, --list/--show via ProfileLookup read-only + find() per op. Guard if(TARGET halo_autotune). Tests in test_cli.cpp. Private tree HALO_ONLY must add autotune (autotune_cpu needs cpu).
(2) docs/cli.md.
- M3: tools/halo/{tune.cpp,hashing.h,hashing.cpp} written (own SHA-256 since WS-J sha256.h is uncommitted); CMake guards autotune/autotune_cpu; cli.cpp dispatch; tests CliHashing + CliTune x2 pass (private tree, HALO_ONLY adds autotune). Next: docs/cli.md, full ctest+ASan, tune mutations (i_mut_m3.json), final report.
- M3 mutations T01..T12 all red (i_mut_m3.log; T03 aborted -6). docs/cli.md written. Next: final release + ASan in private tree, report.
- M3 DONE: private tree (release HEAD c555eb4, ASan HEAD 3064931) 339 total/338 pass/0 fail/1 skip both. test_cli=22. Final report sent.
## M4 (review fixes R-4, N-3, R-3 docs/clamp; M3 committed f2f8b4b)
Edits: src/api/{generation.h,generation.cpp,server.cpp,requests.cpp,CMakeLists.txt}, tests/unit/api/{fake_engine.h,fake_engine.cpp,test_api_security.cpp,test_api_server.cpp}, tools/halo/commands.cpp, tests/unit/cli/test_cli.cpp, docs/api.md, docs/cli.md.
Private tree HALO_ONLY now includes sampling..runtime (real engine linked). Next: build, fix, mutations i_mut_m4.json, report.
- M4 code+tests green (test_api 67, test_cli 22). Mutations R01-R09 all red (R04b replaced build-failing R04; R09 needed a message assertion). Running final release+ASan.
- M4 DONE: release 517/516/0/1skip (HEAD e2fbe3b); ASan 524/522/1 fail (WS-J Subprocess.TimeoutTerminatesAndEscalatesToSigkill, passes 3/3 isolated)/1 skip (HEAD 8235c3b). M4 reported. M5 = security review api findings (queued).
## M5 (M4 committed 40309f8). Order: (3) security S-13,S-16,S-17..S-22 now; (0) F-1 and (1) key contract WAIT for coordinator msg "WS-G M4 committed"; (2) F-2/F-3.
S-13 design: Linux connection watchdog (scan /proc/self/fd sockets on our port, phase tracked by peer addr:port via pre-routing/logger/guarded hooks; shutdown() conns stuck in header>header_timeout or body>body_timeout). httplib has no total header deadline; process_and_close_socket is private.
- M5(3) code written for S-13(conn_guard),S-16,S-17,S-18,S-19,S-20,S-21(fsutil),S-22 + tests; building.
- M5 (0) F-1 fix (requests.cpp structured_output_disables_thinking + integration skip removed), (1) tune via runtime profile.h, hashing.* deleted, serve/run/bench pass power_mode/profile_db/isa_target to engine, (2) F-2 --concurrency-context, F-3 bench drafting off w/o mtp_decode. Building.
- docs api.md/cli.md updated for M5. Next: full ctest, mutations i_mut_m5.json, ASan, report.
- M5 mutations running detached in WSL: /root/halo-logs/wsi-mut7.log (i_mut_m5.json, 28). Check ALL_DONE + TREE_RESTORED_OK_AFTER.
- M5 mutations: 28 + 3 reruns (F1c2,S16a2,S13d w/ timing) all red; TREE_RESTORED_OK. Running final release+ASan.
- ASan found my S-13 watchdog cutting long SSE streams (httplib logger fires before chunked provider). Fix i_ed24.py (StreamPhase) + test ApiLimits.LongStreamsOutliveTheReadDeadlines. Also D-017: loopback-only, need explicit opt-in flag for any non-loopback bind.
- i_sync.sh fixed: rsync without -t (changed files get a fresh mtime; git-archive commit times made ninja miss rebuilds -> stale test_profiling link errors).
- StreamPhase+logger skip: LongStreams test red before fix (observed), keep-alive test catches SP1b; SP3 alone survives (redundant with StreamPhase ctor). D-017 done (allow_remote). Release 568/567/0/1 @fb9173c. Running final ASan + release.
