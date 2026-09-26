# Security review: WS-I M5 (API/CLI hardening) and WS-X (EVO-X2 field kit), 2026-09-26

- **Scope:** two commits on `halo-v0.2`.
  - `e6ce83e` (WS-I M5): `src/api/conn_guard.{h,cpp}`, `src/api/server.cpp`, `src/api/generation.cpp`, `src/api/requests.cpp`, `include/halo/api/server.h`, `tools/halo/fsutil.{h,cpp}`, `tools/halo/{commands,config,tune,bench}.cpp`.
  - `60fa5c2` (WS-X): `src/core/log.cpp`, `scripts/package.sh`, `scripts/evox2/*.sh`, `scripts/evox2/lib/halokit/*.py`.
- **Mode:** review only. No source file was changed. This document is the only file written to the repository.
- **Tier:** 3. The commits touch the pre-auth network surface, file writes to user-chosen paths, and scripts that the owner runs on the target machine, some of them with `sudo`.
- **Context:** `docs/reviews/2026-09-25-security-review-api.md` (S-13 to S-28), `docs/reviews/2026-09-25-security-review-bench.md` (S-29 to S-37), and `docs/DECISIONS.md` D-017. New finding IDs start at **S-38**.
- **Evidence labels:**
  - **Reproduced:** run in WSL (Ubuntu 24.04, as root) and the output observed.
  - **Traced:** followed through the code, including the httplib 0.57.1 source where relevant. Not executed.
  - **Plausible:** the mechanism is sound, but the triggering conditions were not built.

## Method

- **Runtime checks:** used the WSL build `/root/halo-build-bi/tools/halo/halo`, built from `/root/bi-src`. After normalizing CRLF line endings, the following files in that source tree are byte-identical to this tree:
  - `src/api/{server,conn_guard,generation,requests}.cpp`
  - `include/halo/api/server.h`
  - `tools/halo/{config,commands}.cpp`

  The model was `/root/halo-ref/tiny/tiny-q4_0.gguf` on the CPU backend, bound to loopback port 18931.
- **Logging:** `src/core/log.cpp` from this tree was compiled into a small harness (g++ `-std=c++23`, nlohmann/json taken from the build's `_deps`) to test `HALO_LOG_FILE`.
- **httplib behaviour:** checked against `/root/halo-build-orch5/_deps/httplib-src/httplib.h` (v0.57.1):
  - `routing()` runs `pre_routing_handler_` before the body is read.
  - `req.is_connection_closed` and `DataSink::is_writable` both call `is_socket_alive()`, which does `select` with a zero timeout and then a `MSG_PEEK` recv, so neither blocks.
  - The logger runs in `write_response` before the body or content provider is written.
  - `remote_addr` comes from `getnameinfo(NI_NUMERICHOST)`. HALO never calls `set_trusted_proxies`, so it is always the TCP peer.
- **Scripts:** all scratch scripts are prefixed `sec_` in the session scratchpad, outside the repository.

## Summary and readiness verdict

| Class | Findings |
|---|---|
| Blocking (Critical/High) | none |
| Should-fix (Medium) | **S-38**: the S-13 pre-auth pool exhaustion still works with rolling reconnects (reproduced). D-017 already grades S-13 Medium and non-blocking for loopback-only v0.2. This finding confirms the residual and does not change that grade. |
| Low | S-39 (conn_guard phase-loss race), S-40 (package.sh anchor `mkdir -p` race), S-41 (collect scrub and anonymize gaps), S-42 (`HALO_LOG_FILE` chmods and appends to any existing file) |
| Info | S-43 to S-47 |

**Release gate.** For the loopback-only v0.2 scope in D-017 there is no blocking finding. A deployment with `allow_remote` depends entirely on the reverse proxy that D-017 requires, because S-38 shows the in-process deadline does not stop a multi-connection attacker.

## Findings

### S-38 Medium: pre-auth HTTP pool exhaustion still works with rolling reconnects (S-13 residual). Reproduced

- **Where:**
  - `src/api/conn_guard.cpp:111-159` (`scan`): a per-connection deadline only.
  - `src/api/server.cpp:778-782`: pool size `max_concurrent + max_queue + utility_concurrency + utility_queue + 4`, queue 64.
- **Issue:** the guard bounds how long **one** connection can take to send its headers (10 s by default). It does not bound how many connections one peer holds. An attacker opens more connections than there are workers plus queue slots, dribbles header bytes on each, and reopens each one as soon as the guard closes it.
  - Workers stay pinned.
  - httplib drops new connections once its 64-slot queue is full (`ResourceExhaustion`, then `shutdown` and `close`).
  - The guard also starts the header clock at accept time. A legitimate connection that waits in httplib's queue behind the attacker is therefore shut down after 10 s even though its headers are already buffered.
  - Authentication runs in the pre-routing handler, after the headers, so an API key does not help.
- **Evidence** (`sec_s13.py`, default server settings with `--parallel 1`):

  ```
  A: single dribbling connection closed after 10.227385919999506 s     <- the S-13 deadline works for one connection
  baseline health: (200, 0.0029)
  B: N=100 attackers, 40.0s: /health ok 1/79, errors {'RemoteDisconnected': 78}, attacker reconnects 94558
  after attack health: (200, 0.0011)
  halo_api_connection_deadline_closes_total 381
  ```

- **Exploit:** any local process, including an unprivileged user on a shared host, makes `/health` and every API route unavailable for as long as it keeps about 100 sockets open. With `allow_remote` and no proxy, any network peer can do the same.
- **Browser vector:** a web page cannot mount this attack. Browsers cannot dribble header bytes, and the Origin check in the pre-routing handler rejects cross-origin requests before the body is read.
- **Fix (in the guard, where the data already is):**
  - Count connections that are in `Phase::Header` (and `Body`) per peer address. When a peer holds more than N (for example 8, configurable), shut down its oldest header-phase connections immediately.
  - Add a global load-shedding rule: when the number of header-phase connections exceeds `threads - max_concurrent`, shorten their deadline (for example to 1 s) or shut down the oldest.
  - Start the header clock when a worker picks the connection up, not at accept time, or exempt connections that have unread bytes buffered.
  - Keep D-017's proxy requirement for `allow_remote`.
  - Add a regression test that runs a rolling-reconnect attacker and asserts `/health` stays at 90% or more available.

### S-39 Low: a phase update can be lost in `ConnectionGuard::scan`, which cuts off a long non-streaming request at `header_timeout`. Traced, not reproduced

- **Where:** `src/api/conn_guard.cpp:113-138` together with `set_phase` (`:51-56`).
- **Issue:** `scan()` snapshots `/proc/self/fd` without the lock (step 1), then takes the lock and erases every map entry whose key was not in the snapshot (`:138`). Consider this sequence:
  1. A connection is accepted after `readdir` has passed its fd number.
  2. `headers_done` and `handler_started` run before the scanner takes `mu_`.
  3. The entry that `set_phase` created, now in `Phase::Handler`, is erased as "not seen".
  4. On the next scan the socket appears and is re-added as `Phase::Header` with `since = now`.

  A non-streaming `/v1/chat/completions` or `/v1/completions` whose generation lasts longer than `header_timeout` then gets `shutdown(SHUT_RDWR)` in the middle of the handler. The client sees a reset, and the disconnect watcher cancels the generation.

  Streaming requests recover, because `StreamPhase` sets the phase again after the handler returns. Non-streaming requests have no further `set_phase` call until `request_done`.
- **Likelihood:** the window is the step-1 syscall time (two syscalls per fd) divided by the 200 ms interval. It is small, but it hits real traffic at random, and slow non-streaming generations on the 27B model are the normal case.
- **Fix:** record `t_snap = steady_clock::now()` before `readdir`, and erase only entries where `!seen && since < t_snap`. Alternatively, erase an entry only after it has been missing from two consecutive scans. A unit test can inject the interleaving through a scan hook.

### S-40 Low: `package.sh` builds into an anchor directory that another local user can own. Race window, primitive reproduced

- **Where:** `scripts/package.sh:119-131`.

  ```bash
  if [[ -e "$ANCHOR" || -L "$ANCHOR" ]]; then ... rm -rf -- "$ANCHOR"; fi
  mkdir -p -m 0700 "$ANCHOR"
  ```

- **Issue:** the anchor is the predictable path `/tmp/<prefix>-<sha12>-anchor`, and the sha12 is the public HEAD. The ownership check (`-O`) runs only if the directory already exists. After it, the script runs `rm -rf` and then `mkdir -p`.
  - `mkdir -p` succeeds silently on an existing directory owned by someone else, and the `-m 0700` is applied only when it creates the directory.
  - Reproduced: a `nobody`-owned `0777` directory stayed `drwxrwxrwx nobody` after `mkdir -p -m 0700`, and then received root's `.halo-kit-staging`.
  - A local attacker looping on `mkdir` wins the gap between `rm -rf` finishing and `mkdir -p`, or the gap between the existence check and `mkdir -p` on a first build.
- **Impact:** the attacker's directory then holds `src/` and `build/`. They can modify sources between staging and `cmake`. That gives code execution as the builder (CMake runs code at configure time) and a tampered kit that the owner later runs on the EVO-X2.
  - This applies to any multi-user build host, including the EVO-X2 itself through `build-native.sh`, which uses `--source copy`.
  - On the single-user WSL dev host there is no attacker, hence Low.
  - `fs.protected_symlinks` blocks the symlink variant but not this one.
- **Fix:** create the anchor with `mkdir -m 0700 "$ANCHOR"` (no `-p`) and fail if it exists. Afterwards re-check `[[ -O "$ANCHOR" && ! -L "$ANCHOR" ]]` and that the mode is `0700` before staging.
- **Target side:** `setup_anchor` (`scripts/evox2/lib/halokit/common.py:330-340`) already uses `mkdir` without `parents` and refuses a directory it does not own, which is good. It does not check the mode, though. An anchor owned by the right user but group- or world-writable lets another user swap the `build` symlink, and `tests.py run_binary` executes `anchor/build/...`. Add `st.st_mode & 0o022 == 0`.

### S-41 Low: the bundle scrub silently skips files, and `--anonymize` does not do what the docs say. Traced

- **Where:**
  - `scripts/evox2/lib/halokit/collect.py:722-740` (`scrub`) and `:876-900` (replacements and tar).
  - `docs/evox2.md:174` ("replaces the hostname, user name and home directory **everywhere in the bundle**").
- **Issues:**
  1. `scrub` skips any file over 64 MiB, and any file with a NUL byte in its first 8 KiB (`:728`, `:731`). It records nothing and does not warn, and `tarfile.add(root)` then packages those files unscrubbed. Two cases are likely to hit this:
     - The trace-level JSON logs (`model/*.log.jsonl`) from a 27B run can exceed 64 MiB.
     - UTF-16 or binary tool output trips the NUL check.

     Secrets from the environment and the hostname, user and home directory (with `--anonymize`) then ship in the `.tar.gz`, even though the run printed "scrubbed N file(s)".
  2. The user name is replaced only when it is at least 3 characters (`:886`). A 2-character login is never anonymized.
  3. The replacement is a plain substring replace, so a short login corrupts the data. A user named `amd`, `gpu` or `drm` turns `amdgpu` into `<user>gpu` across every file, which also breaks the checklist that parses those files.
  4. The user and home directory come from `$USER` and `~` (`:790-791`). The run book suggests running as root when `sudo -v` is not an option. Under `sudo`, `USER` is `root` and `HOME` is `/root`, so the invoking user's name and `/home/<name>/...` paths in `--kit`, `--model`, `--llama-dir`, the manifest `args` and every `command.txt` are not anonymized.
- **Fix:**
  - Stream-scrub large files, or refuse to write the tarball and list the file when any file was skipped, so the bundle is left for manual review, as the scrub-error path already does.
  - Treat NUL-containing files explicitly (scrub them as bytes).
  - Replace only on word boundaries (`(?<![A-Za-z0-9_])user(?![A-Za-z0-9_])`), and anonymize names of any length on boundaries.
  - Include `SUDO_USER` and its home directory.
  - Change the docs to "text files up to 64 MiB", or remove the limit.

### S-42 Low: `HALO_LOG_FILE` chmods and appends to any existing regular file, including files the process does not own. Reproduced

- **Where:** `src/core/log.cpp:60-75`. The code comment reads `(void)::fchmod(fd, 0600); // tighten a pre-existing file; fails harmlessly if not ours`.
- **Issue:** as root, `fchmod` does not fail on a file root does not own. Any existing regular file named by `HALO_LOG_FILE` has its mode changed to `0600` and gets log lines appended. Hard links are accepted as well.
- **Evidence:**

  ```
  echo precious > shared.txt; chmod 0644 shared.txt; chown nobody shared.txt
  HALO_LOG_FILE=$W/shared.txt ./h
  -rw------- 1 nobody root 40 ... shared.txt
  precious
  [WARN] sec: hello from harness
  ```

  A second test confirmed that a hard link (`orig` and `hl` sharing one inode) is appended to.
- **Impact:** operator error rather than an attack, because setting the environment variable already implies control. The field kit, though, encourages running as root and sets `HALO_LOG_FILE` itself (`collect.py:416,422,432`).
  - A mistyped `HALO_LOG_FILE=/etc/…` in a root session silently makes a system file `0600` and appends JSON lines to it.
  - For example, `/etc/passwd` becoming `0600` breaks name resolution for every non-root user.
- **Fix:** after `fstat`, refuse unless `st.st_uid == geteuid()` and `st.st_nlink == 1`. Call `fchmod` only when the process created the file (open once with `O_EXCL`, and fall back to a plain open when the file exists) or when it owns the file and its mode is wider than `0600`.

### S-43 Info: `HALO_LOG_FILE` opens device nodes before the type check, and `O_NOCTTY` is missing. Traced

- **Where:** `src/core/log.cpp:60-67`.
- **Observed behaviour:**
  - A FIFO with no reader fails with `ENXIO` and does not hang.
  - A FIFO with a reader is refused as not a regular file.
  - `/dev/null` is refused.
  - A symlink is refused with `ELOOP`.
- **Issue:** `open(O_WRONLY|O_NONBLOCK)` has already reached the driver by the time `fstat` rejects the file. Some drivers act on open: `/dev/watchdog` arms on open and may reboot the machine after close, and a serial tty raises modem lines. Without `O_NOCTTY`, a session-leader daemon that opens a tty takes it as its controlling terminal.
- **Fix:** add `O_NOCTTY`. Run `stat()` or `lstat()` first and skip anything that is not a regular file (keep the post-open `fstat` for the race), or open with `O_PATH`, check, and reopen through `/proc/self/fd/N`.

### S-44 Info: environment redaction works by variable name only. Traced

- **Where:** `scripts/evox2/lib/halokit/common.py:24` (`SECRET_NAME_RE`) and `:46-56`; `collect.py` `env_fn` (writes `system/NN-environment/env.txt`).
- **Issue:** values that carry credentials under names outside `KEY|TOKEN|SECRET|PASSW|AUTH|CREDENTIAL|COOKIE` are written in full. Examples are `http_proxy` / `https_proxy=http://user:pass@proxy:3128`, `DATABASE_URL`, `*_DSN`, and `*_PAT`.
  - `secret_values()` scrubs only the values of secret-named variables.
  - Under `--anonymize`, `SSH_CONNECTION` and `SSH_CLIENT` (client and server IP addresses) remain in the bundle.
- **Fix:** also redact any value matching `[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s]+@`, and add `PAT|DSN|PRIVATE` to the name pattern. Under `--anonymize`, drop `SSH_*`, `DISPLAY`, `XDG_SESSION_*` and `MAIL`.

### S-45 Info: `ConnectionGuard` has a check-then-shutdown TOCTOU on a raw fd number. Traced

- **Where:** `src/api/conn_guard.cpp:148-152`.
- **Issue:** between `peer_of(fd)` and `shutdown(fd)`, the descriptor can be closed by the httplib worker and the number reused.
  - A wrong-connection kill needs the reused fd to be a socket on the server port with the same peer address and port, which means the same client's next connection. It is not practically reachable.
  - Any other reuse lands on a socket that `shutdown` would affect only in a microsecond window. A non-socket fd gets `ENOTSOCK`.
- **Fix (removes the race completely):** in step 1, `dup()` each matching fd and keep the duplicate. Check and shut down through the duplicate (it refers to the same open file description and cannot be reused), and close duplicates at the end of the scan. `pidfd_getfd` is an alternative.

### S-46 Info: `fsutil` and `tune` residuals after the S-21 fix. Traced

- **Where:** `tools/halo/fsutil.cpp:36,52,57`; `tools/halo/tune.cpp:147-151,223,299`.
- **Issues:**
  1. The replaced file always gets `0644 & ~umask`. An existing `0600` bench artifact or tune report becomes world-readable when it is rewritten. Keep the original's mode and owner with `fstat` plus `fchmod` on the temporary file when the target exists.
  2. `if (::fsync(fd) != 0 || ::close(fd) != 0)` skips `close` when `fsync` fails, which leaks the fd. It only matters in an embedding process.
  3. `refuse_symlink(s.db)` followed by `ProfileDb::open` (`sqlite3_open_v2`, `src/autotune/sqlite.cpp:94-96`) is a TOCTOU window. Pass `SQLITE_OPEN_NOFOLLOW` (SQLite 3.31 or later), which makes the `lstat` pre-check unnecessary.
  4. `file_identity` compares dev, ino, size and mtime. An in-place rewrite that keeps the size and restores the mtime with `utimensat` passes the check. Add `st_ctim`, which user space cannot set.

  None of these is reachable without write access to the target directory or file.

### S-47 Info: the collect tarball is opened by a predictable name without `O_EXCL`. Traced

- **Where:** `collect.py:897-900`, `tarfile.open(tgz, "w:gz")`.
- **Issue:** the tarball name is `halo-diag-<host>-<UTC second>.tar.gz` inside `--out`, and it is opened by following symlinks and truncating. In `/tmp`, `fs.protected_symlinks=1` blocks this. In a shared, non-sticky `--out` directory, another user can redirect the write.
- **Fix:** open the file with `os.open(tgz, O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, 0o600)` and hand the file object to `tarfile.open(fileobj=…, mode="w:gz")`.

## Status of earlier findings

| ID | Status | Evidence |
|---|---|---|
| S-13 | **Partially fixed.** A single stalled connection is now closed after about 10.2 s, and non-loopback binds need `allow_remote` (D-017). The multi-connection form remains, see **S-38**. | Reproduced |
| S-16 | **Fixed.** `chat()` and `completion()` call `admit()` before `build_chat_prompt` and `encode` (`server.cpp:567`, `:624`). `/tokenize` and `/apply-template` run behind the `utility` admission (2 running, 8 queued, then 429). The HTTP pool is sized for both. Body parsing still happens before admission, which is bounded by `max_body_bytes`. | Traced (the 429 timing was not re-measured) |
| S-17 | **Fixed.** `check_schema_size` applies to the whole tool object (OpenAI `requests.cpp:306`, Anthropic `:615`). Tool strings count toward `max_content_bytes` on chat, apply-template and Anthropic. `max_messages` is re-checked after `tool_result` expansion. | Traced |
| S-18 | **Fixed.** An empty environment value counts as unset (`config.cpp:232`). An empty config-file `api_key` is a Config error. `--api-key ''` is a usage error. `--print-config` shows `<empty>` or `<redacted>`. `log.cpp` applies the same empty-means-unset rule. | Reproduced: `HALO_API_KEY=` with a file key gives source `file:`; `--api-key ''` gives "must not be empty" |
| S-19 | **Fixed.** `cors_origins '*'` without a key is refused. | Reproduced: CONFIG_ERROR |
| S-20 | **Fixed.** An unauthenticated non-loopback bind requires `allowed_hosts`. | Reproduced: CONFIG_ERROR with `--allow-remote --allow-unauthenticated-remote` |
| S-21 | **Fixed for the reported symptoms.** `tune --report`, `tune` DB output and `bench --out` go through a temporary file (`O_EXCL|O_NOFOLLOW`), then `fsync` and `rename`. A symlink target is refused, and `rename` would replace the link rather than follow it. Write failures now throw. The model is identity-checked across the hash and the header read. Residuals are in **S-46**. | Traced |
| S-22 | **Fixed (exposed).** The setting is available as `--no-completions-parse-special`, `HALO_COMPLETIONS_PARSE_SPECIAL` and `server.completions_parse_special`. The default is still `true`; the review offered exposing it or changing the default, and exposing it was chosen. | Reproduced: `--print-config` shows source `cli` |
| S-15 (context) | **Wired.** Every engine call carries a `std::stop_token` and the deadline. A per-request watcher polls the stopping flag and `peer_connected()` every 100 ms. The watcher is declared after `cancel_source` and `why`, so it is joined before they are destroyed. The peer probes are the non-blocking `is_socket_alive` (checked in httplib), so the watcher cannot stall the handler. | Traced |
| S-32, S-33, S-37 (bench, observed in passing) | `subprocess.cpp` now uses `WNOWAIT` (`:192`), `addclosefrom_np(3)` (`:142`), and `O_NOFOLLOW` with mode `0600` for the child log (`:441-444`). These changes are not part of the two commits under review and were not re-verified in depth. | Traced |

## Verified-good (no finding)

- **D-017 gate.** `validate_server_config` is enforced both in the `ApiServer` constructor and in `cmd_serve`, before the engine loads. `--host 0.0.0.0` gives: `CONFIG_ERROR: ... v0.2 serving is loopback-only (D-017)` (reproduced). `is_loopback_host` accepts only `localhost`, `::1` and strict dotted-quad `127.a.b.c` addresses.
- **Auth runs before the body is read** (httplib `routing()` order). An unauthenticated client on a keyed server can hold a worker only during the header phase.
- **Streaming responses are exempt from the read deadline, and streams are not cut off.** `StreamPhase` runs for the whole content provider. The logger skips `request_done` when the response has `X-Accel-Buffering`, and `set_sse_headers` is immediately followed by `set_chunked_content_provider` with no throwing call between them.
- **`log.cpp` file handling.** Symlinks are refused (`ELOOP`, reproduced). A FIFO with no reader returns `ENXIO`, so there is no hang (reproduced). FIFOs with a reader and `/dev/null` are refused (reproduced). A new file is created `0600` (reproduced). `O_CLOEXEC` is set, and `O_NONBLOCK` is cleared after the type check.
- **`collect.py`:**
  - It sets `umask 077`.
  - The bundle directory is created fresh (`mkdir` without `exist_ok`).
  - Every subprocess gets an argv list (no `shell=True`), and `--bench-args` / `--llama-args` go through `str.split()`, not a shell.
  - Every step has a timeout with a process-group SIGTERM, then SIGKILL.
  - `sudo` is used only as `sudo -n dmesg [-T]`: non-interactive, only `dmesg`, and only the GPU lines are saved.
  - Steps run with `start_new_session=True`, so the binaries under test have no controlling terminal and cannot reuse a tty-scoped sudo ticket from `sudo -v`. This is inferred from sudo's default `timestamp_type=tty` on Ubuntu and was not tested, because WSL here runs as root.
- **No archive extraction in the Python kit** (there is no `extractall`). `build-native.sh` extracts only the kit's own source tarball with GNU tar, which by default strips a leading `/` and refuses `..` members.
- **`setup_anchor` on the target** refuses a symlinked or foreign-owned anchor, and its `mkdir` without `parents` fails if another user wins the race (see S-40 for the mode check).

## What I could not test

- **The S-39 interleaving.** The race window is roughly the scan's syscall time divided by the 200 ms interval. Triggering it deterministically needs a hook in `scan()`, and this review was read-only.
- **The sudo tty-ticket behaviour** under `start_new_session` (WSL runs as root here).
- **A real EVO-X2 or ROCm host**, and the size of a 27B trace log (which is what makes S-41 item 1 likely).
- **S-38 against a server with an API key.** The code order (auth in the pre-routing handler, after the headers) makes the result the same, but only the keyless loopback default was run.

## Risk acceptance

None was requested. S-38 falls under the owner's D-017 decision: Medium, non-blocking for loopback-only v0.2, with a reverse proxy required for `allow_remote`. This review does not accept any risk on the owner's behalf. The S-38 evidence shows that the in-process deadline alone does not make a non-proxied `allow_remote` deployment safe.
