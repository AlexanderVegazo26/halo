# Security review: WS-J baseline adapters (subprocess, HTTP client, SHA-256) (2026-09-25)

- **Scope:** WS-J's committed `8b40f46`, reviewed and built as part of `adb3c46`:
  - `src/profiling/{subprocess,http_client,sha256,baseline}.cpp` and `include/halo/profiling/{subprocess,http_client,sha256,baseline}.h`;
  - `src/autotune/baseline_store.cpp`.
- **Why this is in scope:** it meets three security-engineer triggers: subprocess execution, a new HTTP client, and file reads and writes at caller-chosen paths.
- **Mode:** review only. No source file was changed.
- **Tier:** 2, not 3:
  - No network listener is added.
  - The HTTP client only connects out, and only to loopback.
  - No `halo` CLI command drives these adapters yet. `grep` of `tools/` at `adb3c46` finds no caller, so the inputs come from the calling code or config (the operator).
  - The untrusted inputs are the output of child processes (`llama-bench`, `llama-server`) and HTTP responses from local servers (llama-server, Ollama).
- **Numbering:** findings continue after S-28 in `2026-09-25-security-review-api.md`.

## Method and baseline

- **Build:** `git archive adb3c46` into `/root/halo-build-secapi2` (clang 18, RelWithDebInfo, WSL). `test_profiling` passes 85/85 and `test_autotune` passes 49/49 (measured).
- **Harness:** a PoC harness links the built `halo_*` static libraries directly and calls the public API: `run_process`, `recorded_environment`, `http_request`, `sha256_file` and `run_llama_server`. It ran against:
  - one-shot fake HTTP servers on `127.0.0.1`;
  - the real llama.cpp `llama-server` (`/root/llama.cpp/build/bin`, commit `bd4f514`) with the tiny GGUF.
- **Location:** the harness (`poc.cpp`) and its outputs are in the reviewer's session scratchpad, which is temporary. The key inputs are inlined in each finding.
- **Evidence labels:** **Verified** means reproduced, with the output observed. **Believed** means code reading only.

## Summary

The primitives are carefully built:
- **Subprocesses:** argv only, with no shell and no PATH search. The environment is explicit by default. Both pipes are drained concurrently and capped. Each child runs in its own process group, and SIGTERM is followed by SIGKILL on timeout.
- **HTTP client:** only the literal strings `127.0.0.1` and `::1` are accepted as the host, with no name resolution and no redirect following. Response parsing is strictly bounded.
- **SHA-256:** correct.

I found **no memory-safety issue, no path to a remote host, and no shell injection.** The findings concern trust in *who answered*, error-path status reporting, secrets in recorded artifacts, and signal delivery after a child is reaped.

| Gate reading (per the §4 mapping) | Findings |
|---|---|
| Blocking (Critical/High) | none |
| Should-fix (Medium) | none |
| Low | S-29, S-30, S-31, S-32 |
| Informational | S-33, S-34, S-35, S-36, S-37 |

**Counts:** Critical 0, High 0, Medium 0, Low 4, Informational 5.

## Findings

| ID | Sev | Component | Evidence | Impact | Mitigation |
|---|---|---|---|---|---|
| S-29 | Low | `baseline.cpp` `run_llama_server` (readiness probe and `/completion`) | **Verified** | Whatever already listens on `c.port` is taken to be the spawned llama-server. A stale server, or a process run by another local user, supplies forged baseline numbers under the real binary's version and commit, and no note flags it. | Before spawning, check that the port is free (a bind probe), and fail if the port is taken. After `/health` returns 200, confirm that the listening socket belongs to the child: match `/proc/<child>/fd` socket inodes against `/proc/net/tcp`. Treat the child exiting during the run as a failure (`running()` is only checked before readiness). |
| S-30 | Low | `subprocess.cpp` `try_reap()` / `reap()` | **Verified** | When `waitpid` fails (ECHILD, for example because the embedding process sets `SIGCHLD` to `SIG_IGN`, or another thread reaps with `waitpid(-1)`), status 0 is reported. A child that exited with 7 comes back as `exit_code=0, ok()=true`. | Return a distinct "status unknown" on `waitpid` errors, and make `ok()` false for it. Optionally assert at startup that `SIGCHLD` is not ignored. |
| S-31 | Low | `subprocess.cpp` `recorded_environment()`, used by `baseline.cpp` `make_invocation()` | **Verified** (the function); persisting to the artifact and DB **Believed** (code path) | Inherited `LLAMA_*` variables are recorded, and `LLAMA_API_KEY` is llama-server's API-key variable (`common/arg.cpp:3497` at `bd4f514`). Every explicitly passed variable is recorded verbatim. The secrets end up in `halo.bench.record/1` artifacts and in the profile DB (`store_baseline`). | Redact values whose names match `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*` or `*AUTH*`, and store `"<redacted>"`. Or record names only, with values for an explicit allowlist. |
| S-32 | Low | `subprocess.cpp` `kill_group()` after reap (`run_process`, `terminate_group`, `ChildProcess::running`) | **Believed** | Once the child has been reaped, its PID and PGID may be reused. `kill(-pid, SIGKILL)` can then hit an unrelated process group, and the ESRCH fallback `kill(pid, SIGKILL)` can hit an unrelated process owned by the same user (any process, when run as root). The race window is small and needs PID wrap-around, so this was not reproduced. | Signal the group **before** reaping: use `waitid(P_PID, pid, …, WEXITED \| WNOWAIT)`, then `kill(-pid, SIGKILL)`, then `waitpid`. Or use `pidfd_open` / `pidfd_send_signal`. Drop the `kill(pid)` fallback once the child is known to have exited. |
| S-33 | Info | `subprocess.cpp` `spawn()` (posix_spawn with no close-range) | **Verified** | Parent file descriptors without `O_CLOEXEC` are inherited by the benchmarked tools. A plain `open()` fd was visible in `/proc/self/fd` of the child. HALO's own pipes and HTTP sockets use CLOEXEC (code; SQLite's fds are believed to as well); an embedding program's may not. | Add `posix_spawn_file_actions_addclosefrom_np(&fa, 3)` (glibc 2.34 or later). |
| S-34 | Info | `subprocess.cpp` process-group teardown | **Verified** | A grandchild that calls `setsid` escapes the group kill: `setsid sleep 301` survived a timeout, while the in-group `sleep 300` and `sleep 302` were killed. This is inherent to process groups; llama.cpp tools do not do this. | Document it. For a hard guarantee, use `PR_SET_CHILD_SUBREAPER` plus a sweep, or a transient cgroup. |
| S-35 | Info | `http_client.cpp` request head | **Verified** | `HttpRequest::content_type` is written into the request unchecked. A value containing `\r\nX-Injected: 1` injected a header line. Every current caller uses the constant default, so this is not reachable today. `path` and `host` are validated. | Reject CR, LF and NUL in `content_type`, as is already done for `path`. |
| S-36 | Info | `baseline.cpp` model path and argv construction | **Believed** (code, plus llama.cpp source at `bd4f514`) | The hash is not guaranteed to describe the model that was benchmarked: (1) `llama-bench` splits `-m` on `,` (`llama-bench.cpp:557–562`), so a path containing a comma is hashed as one file but benchmarked as several models; (2) `extra_args` can add `-m` (another model) or, for llama-server, `--host 0.0.0.0` after HALO's `--host 127.0.0.1`, exposing llama-server beyond loopback with no key; (3) the file is hashed, then re-opened by the tool (a TOCTOU, as in S-21); (4) `sha256_file` blocks forever on a FIFO or char device given as the model. All of these are operator or config inputs. | Reject `,` in `model` for llama-bench, and reject `-m`, `--model`, `--host` and `--port` in `extra_args`. Require `is_regular_file(model)`. Note in the artifact that the hash was taken before the run. |
| S-37 | Info | `subprocess.cpp` `ChildProcess` `log_path` | **Believed** (code) | The log is opened with `O_CREAT \| O_TRUNC`, mode `0644`, and follows symlinks: the same class as S-21. It is also world-readable, and llama-server logs can include prompts. | Use `O_NOFOLLOW`, mode `0600`, and create it in a private directory. |

### S-29 evidence: port hijack. Verified

A fake HTTP server was bound on `127.0.0.1:18096` first. It answers `/health` with 200, and `/completion` with forged timings: `prompt_per_second` 99999, `predicted_per_second` 88888, `draft_n` = `draft_n_accepted` = 100. Then `run_llama_server` ran with the real `llama-server`, `port = 18096`, `mtp = true` and 2 repetitions:

```
artifact valid=0 records=2
  record mode=mtp_completion prompt_tps=99999 decode_effective_tps=88888 mtp_acceptance=1 version=0.5.0-dev (build 1, commit bd4f514)
  record mode=mtp_completion prompt_tps=99999 decode_effective_tps=88888 mtp_acceptance=1 version=0.5.0-dev (build 1, commit bd4f514)
(no note mentions the port or the server)
```

- **What happened:** the real llama-server could not bind the port, and its log was empty when it was torn down. The records carry the impostor's numbers under the real binary's version string.
- **Why the artifact was invalid:** `valid=0` is believed to come from the 2 repetitions being below the TRD §50 minimum, not from detecting the problem. With a conformant repetition count, `store_baseline` would store these numbers as the latest baseline.
- **Severity:** Low. It needs a process already on the port (a local attacker or a stale server), and the impact is the integrity of benchmark data, not code execution.

### S-30 evidence: exit status lost. Verified

```
SIGCHLD=SIG_IGN: exit_code=0 signaled=0 ok()=1      # child ran: /bin/sh -c "exit 7"
SIGCHLD=SIG_DFL: exit_code=7 ok()=0
```

## Verified-good (no finding)

- **SHA-256 (the correctness evidence the brief asked for):**
  - `sha256_file` was compared with coreutils `sha256sum` on 15 inputs: empty, `abc`, the NIST 448-bit message, random files of 1, 3, 55, 56, 57, 63, 64, 65, 119, 120 and 121 bytes (every padding boundary), and 1,000,000 bytes.
  - Each input was hashed at chunk sizes 1, 7, 63, 64, 65 and 4 MiB. **All 90 digests match.**
  - The tiny GGUF gives `d01fff3e5c9555f1…` from both tools.
  - `make_pack_id` refuses input that is not 64 lowercase hex characters (code, and `Sha256.PackIdDefinition`).
- **Subprocess:**
  - `argv[0]` must contain `/` (no PATH search), NUL bytes are refused, there is no shell, stdin is `/dev/null`, and the signal mask and dispositions are reset (code, and the tests `ArgumentsAreNotShellParsed` and `InvalidProgramsAreTypedErrors`).
  - The environment is explicit unless `inherit_env` is set.
  - Pipe caps work: 200 MB on each of stdout and stderr kept 1 MiB each (both truncated flags set), with the exit code preserved, 0.16 s wall and 6 MB max RSS.
  - On timeout, the group gets SIGTERM and the in-group grandchildren are gone (`sleep 300` and `sleep 302`: 0 left).
- **HTTP client:**
  - **Host:** these hosts were refused before any socket was opened: `localhost`, `127.0.0.2`, `127.1`, `0x7f000001`, `2130706433`, `::ffff:127.0.0.1`, `[::1]`, `127.0.0.1 ` (trailing space), `0.0.0.0`, `10.0.0.1`. The address comes from constants, not from the string.
  - **Redirects:** a 302 with `Location: http://10.0.0.1/` is returned as status 302 and not followed. The adapters treat any non-200 as a failure.
  - **Content-Length:** a length above the cap, an overflow, `-1` and `+5` are all typed errors.
  - **Chunked:** a 17-hex-digit size, 2^64-1, a chunk sum above the cap, and a 2 KiB size line without CRLF are all typed errors. A 100 KiB header block is refused at 64 KiB. A read-until-close body above the cap is refused.
  - **Slow drip:** bounded by the request deadline (3.0 s against a 3 s timeout).
  - **Many tiny chunks:** 20,000 1-byte chunks took 0.23 s and 80,000 took 0.92 s, which is linear in practice. `decode_chunked` restarts from the beginning after each read, so the worst case is quadratic in the number of reads, but it is bounded by the deadline and `max_body_bytes` (believed; not a finding).
- **Output parsers:** `parse_llama_bench_json`, `parse_llama_server_completion` and `parse_ollama_generate` check the type and range of every field used, with caps on tests (4096) and samples (100,000) (code, plus `test_baseline`).
- **Store:** `store_baseline` refuses invalid artifacts unless `allow_invalid` is set. SQLite writes use bound parameters (`sqlite3_bind_*`). The one `std::format`-built statement (`SELECT count(*) FROM {}`) takes an internal table name.

## What I could not test

- **Ollama:** there is no Ollama install, so `run_ollama` against a real server was not tested. Its parser was reviewed in code only.
- **The S-32 PID-reuse race:** it needs PID wrap-around, so it was not reproduced.
- **Persistence of S-31 secrets** into an artifact file and the DB: this is from the code path (`make_invocation`, then `Invocation.environment`, then the record JSON). I did not run a full adapter with a secret set.
- **Vulkan and HIP llama.cpp builds:** only the CPU llama-server was used.
- **Mutation testing** of WS-J's tests: the commit claims "27/27 mutations red", which I did not re-check.

## Risk acceptance

Nothing has been accepted. None of these findings blocks. S-29 and S-31 should be fixed before baselines are stored from shared or multi-user hosts; that decision belongs to the owner.
