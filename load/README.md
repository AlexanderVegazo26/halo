# Load tests (k6)

k6 scripts for HALO's HTTP API. They only ever target `halo_load_server`, a test binary that
`load/run.py` starts on an ephemeral 127.0.0.1 port: the real `ApiServer` (admission, connection
guard, request parsing, chat templating, SSE streaming, metrics) over the deterministic
`FakeEngine` from `tests/unit/api`, so no model, GPU or network is involved. Never point them at
a real deployment.

| Script | Where it runs | What it checks |
|---|---|---|
| `smoke.js` | ctest `load_smoke` (so `scripts/test-agent` and the release, ASan and TSan CI jobs) | 4 VUs, 20 s over every route; no failed requests, every check passes, p95 < 25 ms, p99 < 50 ms, streamed chat p95 < 30 ms (x20 on sanitizer/Debug builds) |
| `load.js` | on demand | ramps to 16 VUs, soaks 2 min, spikes to 48 (past `max_concurrent` 4 + `max_queue` 16): overload is shed with 429/503 + `Retry-After`, never an error; admitted requests p95 < 700 ms, p99 < 1 s; >= 25 served/s; the spike is shed (x scale) |

```bash
cmake --build build --target halo_load_server
python3 load/run.py --server build/tests/unit/api/halo_load_server load/smoke.js
python3 load/run.py --server build-asan/tests/unit/api/halo_load_server --budget-scale=20 load/smoke.js
python3 load/run.py --server build/tests/unit/api/halo_load_server --server-arg=--token-delay-ms=2 load/load.js
```

`halo_load_server` takes `--concurrent=N`, `--queue=N` and `--token-delay-ms=N`. `run.py` exits 0
pass, 1 fail, 77 when `k6` is not on PATH (ctest reports SKIPPED); on failure it prints each
crossed threshold and failed check with the rerun command. The k6 summary JSON goes to
`$HALO_LOAD_REPORT_DIR` (`.test-report/load/` under `scripts/test-agent`) or `./load-report/`.
The server must also exit 0 on SIGTERM, so under ASan a leak or memory error at shutdown fails
the run.

Latency budgets are ~10x what the GitHub-hosted runner measures on an optimized build, so the
release lane catches an order-of-magnitude slowdown (e.g. the ~41 ms Nagle stall per streamed reply
fixed in 9814e0d fails it). `--budget-scale` multiplies them (`HALO_LOAD_BUDGET_SCALE` in the
script); `tests/unit/api/CMakeLists.txt` passes 20 for ASan, TSan and Debug/unconfigured builds,
1 otherwise.

Baseline (2026-11-06, smoke.js, 4 VUs x 20 s):

| Build | Host | req/s | p95 | p99 | chat_stream p95 |
|---|---|---|---|---|---|
| RelWithDebInfo | GitHub ubuntu-24.04 | ~4,400 | 2.5 ms | 3.8 ms | 3.7 ms |
| Debug + ASan | GitHub ubuntu-24.04 | ~290 | 54 ms | 72 ms | 76 ms |
| Debug + TSan | GitHub ubuntu-24.04 | ~470 | 31 ms | 45 ms | 47 ms |
| RelWithDebInfo | dev host, 32 threads | ~10,300 | 1.0 ms | 1.3 ms | 1.3 ms |
| Debug + ASan | dev host, 32 threads | ~1,000 | 18 ms | 21 ms | 21 ms |

(2026-10-31, before the Nagle fix: ~840 req/s, p95 41 ms on the dev host, release.)

load.js (2026-11-06, dev host, RelWithDebInfo, `--token-delay-ms=2`, 4 min): ~15,400 req/s, almost
all shed (429/503 with `Retry-After`) during the 48-VU spike, 0 failed; served replies min 91 ms
(one generation), median 369 ms, p95 466 ms, p99 468 ms. That is the admission bound, not host
speed: an admitted request waits behind at most 16 / 4 = 4 generations. With the queue unbounded
(`--queue=1000`) or half the slots (`--concurrent=2`) the budgets fail; the old p95 < 3 s did not
catch the unbounded queue.
