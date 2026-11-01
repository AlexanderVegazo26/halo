# Load tests (k6)

k6 scripts for HALO's HTTP API. They only ever target `halo_load_server`, a test binary that
`load/run.py` starts on an ephemeral 127.0.0.1 port: the real `ApiServer` (admission, connection
guard, request parsing, chat templating, SSE streaming, metrics) over the deterministic
`FakeEngine` from `tests/unit/api`, so no model, GPU or network is involved. Never point them at
a real deployment.

| Script | Where it runs | What it checks |
|---|---|---|
| `smoke.js` | ctest `load_smoke` (so `scripts/test-agent` and both CI jobs, incl. ASan) | 4 VUs, 20 s over every route; no failed requests, every check passes, p95 < 1 s (generous, holds under ASan) |
| `load.js` | on demand | ramps to 16 VUs, soaks 2 min, spikes to 48 (past `max_concurrent` 4 + `max_queue` 16): overload is shed with 429/503 + `Retry-After`, never an error; admitted requests p95 < 3 s |

```bash
cmake --build build --target halo_load_server
python3 load/run.py --server build/tests/unit/api/halo_load_server load/smoke.js
python3 load/run.py --server build/tests/unit/api/halo_load_server --server-arg=--token-delay-ms=2 load/load.js
```

`halo_load_server` takes `--concurrent=N`, `--queue=N` and `--token-delay-ms=N`. `run.py` exits 0
pass, 1 fail, 77 when `k6` is not on PATH (ctest reports SKIPPED); on failure it prints each
crossed threshold and failed check with the rerun command. The k6 summary JSON goes to
`$HALO_LOAD_REPORT_DIR` (`.test-report/load/` under `scripts/test-agent`) or `./load-report/`.
The server must also exit 0 on SIGTERM, so under ASan a leak or memory error at shutdown fails
the run.

Baseline (2026-10-31, RelWithDebInfo, 32 threads): smoke ~840 req/s, 0 % failed, p95 41 ms (the
streaming route, ~40 ms per stream), p99 42 ms.
