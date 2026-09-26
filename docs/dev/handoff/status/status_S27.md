# WS-S27 status (live)
- [running] download UD-Q4_K_XL -> /root/models (curl -C -, log /root/s27_dl.log). HF size 17559178144, LFS sha256 3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e
- [done] build HEAD 75799b1 in /root/s27-src -> /root/halo-build-s27 (Release, EXIT=0); halo 0.2.0 components api/runtime/profiling/autotune yes
- [done] inspect on header-only file: census matches summary; planner REFUSED ctx4096x1 by 0.23 GiB (budget = 0.9 x MemAvailable while WS-G ASan runs)
- [done] scratchpad harness s27_margins (/root/halo-build-s27h, tree /root/s27h-src = HEAD + tests/unit/s27) validated on tiny goldens 12/12
- Note: brief names python/tools/fetch_reference.py; actual files are fetch_gguf_headers.py + fetch_hf_files.py (tests/CMakeLists.txt also names the missing file)
