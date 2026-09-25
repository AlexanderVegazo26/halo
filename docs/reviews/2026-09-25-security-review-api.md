# Security review: HTTP API, CLI and engine driving (2026-09-25)

- **Scope:** committed code at `6a54d3f` on `halo-v0.2`:
  - `src/api/**` and `include/halo/api/**`: the cpp-httplib server, OpenAI and Anthropic routes, SSE, auth, admission, and the special-token protection;
  - `tools/halo/**`: config resolution, `serve` bind and auth rules, `tune` and `bench` file writes, model hashing;
  - how the API drives `include/halo/runtime/engine.h` and `src/runtime/engine.cpp`.

  The working tree has uncommitted edits by other agents (`src/template`, `src/profiling`, `tests/unit/integration`), so everything here was reviewed and built from `git archive 6a54d3f`, not from the working tree.
- **Mode:** review only. No source file was changed. This document is the only file created.
- **Tier:** 3. This is new network exposure plus authentication, and the server is the trust boundary between HTTP clients and the model.
- **Inputs:** `docs/reviews/2026-09-24-security-review.md` (S-1..S-12, A-1..A-12), `docs/api.md`, `docs/cli.md`, `docs/dev/TECH_DEBT.md`, and the WS-I M1 commit message (`763892d`), which carries the implementer's status table (A-5 partial, A-8 not done, A-10 partial, S-3 regression owed by `src/template`).
- **Numbering:** the previous review already uses S-12 (licence notices), so new findings start at **S-13** to avoid a collision.
- **Evidence labels:** **Verified** means reproduced against a running `halo serve` or binary, with the output observed. **Believed** means code reading only.

## Method and baseline

- **Build:** HEAD exported to `/root/halo-src-secapi` and built in WSL (clang 18, RelWithDebInfo) at `/root/halo-build-secapi`, with Vulkan off. `halo version` reports `api yes, runtime yes, profiling yes, autotune yes (cpu ops)`.
- **Baseline at HEAD (measured, not quoted):**
  - `test_api` 64/64 passed, `test_cli` 22/22 passed.
  - ASan+UBSan build (`/root/halo-build-secapi-asan`): `test_api` 64/64 passed, with no sanitizer report. That run includes `ApiFuzz.RequestToPromptPathNeverCrashes`.
- **Live tests:**
  - `halo serve` ran on `127.0.0.1:18080` with the **real CPU engine** and `/root/halo-ref/tiny/tiny-q8_0.gguf`. That file has the real Qwen tokenizer and chat template: `<|im_start|>` = 248045, `<|im_end|>` = 248046, `<tool_call>` = 248058, `<tool_response>` = 248066.
  - Settings: `HALO_API_KEY=secretkey123`, `--parallel 2 --max-queue 2`, so the HTTP pool is 2 + 2 + 4 = 8 threads.
  - Hostile requests went through a raw-socket Python client, so malformed framing reached the server unmodified. Exit codes were captured directly.
- **Harness location:** the harness and its outputs are in the reviewer's session scratchpad, which is temporary. The key inputs are inlined in each finding so the cases can be rebuilt.

## Summary and readiness verdict

**The application-layer defences hold up well:**
- **Special-token injection (S-1):** closed on every chat route. OpenAI chat, Anthropic Messages and `/apply-template` were tested over HTTP, covering system, developer, user and tool content, content-part concatenation, tool descriptions and schema keys, and assistant arguments and reasoning.
- **Parse-time JSON depth cap (S-2):** holds.
- **Template kwargs allowlist (S-8):** holds.
- **Admission:** 429 and 503 work.
- **Cancellation on disconnect:** works after the first token.
- **Other checks:** Host, Origin, Content-Type and compressed-body checks all behave as documented. Request smuggling vectors (TE+CL, a double CL, a non-final `chunked`) get 400. Chunked bodies respect the size cap.

**The real risks are below the application layer, and in the gaps the implementer already flagged:**

1. **Connection exhaustion (S-13):**
   - The HTTP worker pool is fixed and small, and httplib has no header-read deadline. **Eight slow TCP connections with no API key made `/health` unreachable for more than 40 s** on the test config. The default config needs 24.
   - This is **High for any non-loopback deployment**, which is exactly the one the API key is meant to make safe.
2. **Hostile model template (S-14, the concrete form of S-3 through the A-5 and A-8 gaps):**
   - With a model whose template is a 54-byte recursive macro, **one authenticated chat request killed the server process (SIGSEGV, rc 139)**.
   - A nested-loop template pinned a worker for more than 25 s despite `--request-timeout 5`, and it made SIGTERM shutdown hang for more than 8 minutes.
3. **Uncancellable prefill (S-15):** a client that sends a long prompt and closes the socket still gets the whole prefill computed. Disconnect, deadline and shutdown are checked only when a token is delivered.

| Gate reading (per the §4 mapping) | Findings |
|---|---|
| **Blocking** (Critical/High, Must-Fix-equivalent) for a non-loopback deployment | S-13 |
| Should-fix before serving untrusted models or untrusted networks | S-14, S-15, S-16 |
| Low / hardening | S-17, S-18, S-19, S-20, S-21, S-22, S-24 |
| Informational | S-23, S-25, S-26, S-27, S-28 |

**Counts:** Critical 0, High 1, Medium 3, Low 7, Informational 5.

For a **loopback-only** deployment, where only local users can connect, S-13 falls to Medium and nothing blocks. Choosing that deployment posture is a decision for the owner or `release-manager`; this review does not make it.

## Threat model (STRIDE, condensed)

**Assets:**
- server availability (one process serves every client);
- the integrity of role separation and of the operator's system prompt;
- the API key;
- operator files written by the CLI;
- other clients' prompts (through the shared prefix cache).

**Actors:**
- an unauthenticated network peer (non-loopback bind);
- a local user or a browser page reaching the loopback bind (CSRF, DNS rebinding);
- an authenticated client;
- the author of a malicious model file;
- another local user who shares a writable directory with the operator.

| Boundary | S | T | R | I | D | E |
|---|---|---|---|---|---|---|
| TCP to httplib (before auth) | — | smuggling: refused (verified) | — | — | **pool exhaustion, S-13** | — |
| HTTP request to auth, Host and Origin | key check sound; rebinding when unauthenticated remote, S-20; `*` CORS, S-19 | — | no request log (only metrics) | error text generic (verified) | — | — |
| JSON to render to tokenize | role forgery closed (verified); assistant markup by design, S-23; completions by design, S-22 | — | — | — | pre-admission CPU, S-16; cap gaps, S-17 | — |
| Engine (prefill and decode) | — | — | — | prefix-cache probing, S-27 | **uncancellable prefill, S-15** | — |
| Model file to template | — | — | — | — | **one request kills the process, S-14** | none found |
| CLI to filesystem and env | empty env var drops the key, S-18 | symlink follow, S-21 | — | argv key, S-28 | — | — |

## Findings

| ID | Sev | Component | Evidence | Impact | Mitigation |
|---|---|---|---|---|---|
| S-13 | **High** (non-loopback) / Medium (loopback) | `src/api/server.cpp` `setup()` with httplib 0.57.1 | **Verified** | Any peer that can reach the port, **with no API key**, makes the whole API unavailable, including `/health`, using `http_threads` connections (default 24). | Front non-loopback binds with a reverse proxy that has header and body timeouts and per-IP connection caps, and document it as required. Add a total header-read deadline, which httplib lacks. Decouple admission waiters from HTTP threads. Cap connections per peer. |
| S-14 | Medium | `src/api/server.cpp` `chat()`/`apply_template()` render on the HTTP thread, plus `src/template` (at HEAD) | **Verified** | A malicious GGUF template kills the process on the first request (SIGSEGV), or pins workers beyond `request_timeout`, and blocks graceful shutdown. | Land the WS-L template limits (depth, iterations, output) and the S-3 regression tests. Render on a bounded pool with a known stack and a time budget. Do the A-8 template SHA-256 allowlist or warning. |
| S-15 | Medium | `src/api/generation.cpp` token callback; `src/runtime/engine.cpp` `generate()` | **Verified** (tiny model); magnitude on 27B **Believed** | A client that disconnects, and the deadline and shutdown, are ignored until the first token. Fire-and-forget long prompts burn the prefill and hold slots at almost no cost to the attacker. | Poll the cancel condition (client alive, deadline, stopping) between prefill chunks. Give `generate()` a cancel token or deadline, or let the caller wake periodically and set `Request::cancel`. |
| S-16 | Medium | `src/api/server.cpp` `chat()` order (parse, render, tokenize, then `admit()`); `/tokenize` and `/apply-template` not admitted | **Verified** (cost), **Believed** (order, from code) | Admission does not bound CPU. A request that ends in 429 still pays up to about 1.8 s of render and tokenize on this host, and the utility routes are never queued. | Admit, or take a cheap pre-admission token, **before** rendering. Put `/tokenize` and `/apply-template` behind a small separate semaphore. This is TD-6 / A-5. |
| S-17 | Low | `src/api/requests.cpp` | **Verified** | The caps are not uniform. Tool descriptions are outside `max_content_bytes`: a 7 MiB description was tokenized into 1,359,538 tokens. Anthropic `tool_result` blocks expand past `max_messages`: 6,000 blocks reached the template. The body cap still bounds both. | Count tool strings in `max_content_bytes`, and apply `max_tool_schema_bytes` to the whole tool object. Check `max_messages` after Anthropic expansion. |
| S-18 | Low | `tools/halo/config.cpp` `server_config()`, `to_json()` | **Verified** | `HALO_API_KEY=` (empty) silently overrides a config-file `api_key`. The loopback server then runs **without auth**, while `--print-config` still shows `"<redacted>"`. | Treat an empty env value as unset, or fail with a config error. Show `"<empty>"` or `null` rather than `<redacted>` for an empty secret. |
| S-19 | Low | `src/api/server.cpp` `origin_allowed()` | **Verified** | `--cors-origins '*'` with no key lets any web page drive the model and read the responses. It starts without a warning. | Refuse `*` unless an API key is set, or at least warn. |
| S-20 | Low | `src/api/server.cpp` `Impl()` (`check_host` off for a non-loopback bind with empty `allowed_hosts`) | **Verified** | With `--allow-unauthenticated-remote`, DNS-rebinding pages can read `GET /v1/models` and `/metrics`. POSTs are stopped only by the Origin check. | Keep the Host check on when unauthenticated, by requiring `allowed_hosts` together with `allow_unauthenticated_remote`. |
| S-21 | Low | `tools/halo/tune.cpp` (`--report`, `--db`), `tools/halo/bench.cpp` (`--out`) | **Verified** (same user); cross-user in sticky `/tmp` **blocked** by `fs.protected_symlinks=1` (verified); non-sticky shared directory **Believed** | Output paths follow symlinks and truncate in place. `tune --list --report <unwritable>` exits 0 with no error. The model is hashed and then re-opened, a TOCTOU between the key hash and the header shapes (believed). | Write to a temp file with `O_NOFOLLOW` / `O_EXCL` and rename it into place. Check the stream state after writing. Take the hash and the header from one open file descriptor. |
| S-22 | Low | `include/halo/api/server.h` `completions_parse_special`; `tools/halo/config.cpp` | **Verified** | `/v1/completions` turns `<|im_start|>` into control tokens by default: 5 tokens instead of 15. The documented opt-out has **no flag, env var or file key**, so a `halo serve` operator cannot use it. | Expose `server.completions_parse_special`, or default it to false for servers with an API key. |
| S-23 | Info | `src/api/prompt.cpp` (assistant history escapes Control tokens only) | **Verified** | UserDefined markup (`<tool_call>`, `<think>`) in assistant history becomes real tokens. This is by design, but it matters when an application relays untrusted text as assistant turns. | Document it for integrators. Offer a strict mode. |
| S-24 | Low | `src/api/generation.cpp` `call()` catch-all | **Verified** | Client-caused `Error(Api/Unsupported)` from `generate()`, for example `maxLength: 1e6`, becomes **HTTP 500** "generation failed" plus two ERROR log lines. Clients get no reason, and anyone can generate ERROR noise. | Rethrow, or map `Api`/`Unsupported` from `generate()` to 400 through `classify_exception`, and log it at INFO. |
| S-25 | Info | httplib header handling | **Verified** | Duplicate `Host` headers: the first wins, with no 400. `Content-Encoding: gzip, identity` or `x-gzip` passes through undecoded to the JSON parser. There is no bypass, because no decompressor is compiled in. | Optionally reject duplicate Host headers, and reject any coding list that contains a known coding. |
| S-26 | Info | `src/api/json_util.cpp` `random_hex` | **Believed** | Response and tool-call ids come from `mt19937_64` with a single 32-bit `random_device` seed per thread, so they are predictable. They are not secrets, and the escape nonce uses `random_device` directly and is checked against the input. | None needed. Use `random_device` if the ids ever carry authority. |
| S-27 | Info | `src/runtime/engine.cpp` prefix cache; `usage.cached_tokens` | **Believed** | All clients share one cache, and the cache-hit length is reported. Clients sharing one server or key can probe each other's prompt prefixes. Checkpoints were disabled on this host, so this was not tested. | Partition the cache per key, or hide `cached_tokens`, when the server has multiple tenants. |
| S-28 | Info | `tools/halo/config.cpp` | **Verified** | An `--api-key` value is readable in `/proc/<pid>/cmdline` (documented). A config file holding `api_key` is not permission-checked (it was `0644` in the test). The key never appeared in `--print-config` or in the server log (both checked). | Warn when a config file containing `api_key` is group- or world-readable. |

### S-13 High: pre-auth connection exhaustion of the HTTP pool. Verified

- **Mechanism:**
  - `setup()` installs `httplib::ThreadPool(n, n, 64)` with `n = max_concurrent + max_queue + 4`.
  - httplib serves each connection on one worker thread. `set_read_timeout(60 s)` bounds each `recv`, but nothing bounds the total time spent reading headers.
  - A client that sends one header line every few seconds therefore holds a worker indefinitely. Once all workers are held, up to 64 further connections queue, and after that httplib drops new ones.
  - Auth runs in the pre-routing handler, *after* the headers have been read, so the key does not help.
- **Evidence** (pool = 8; 8 sockets send `GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n`, then `X-a: b\r\n` every 3 s):

  ```
  [legit GET /health while 8 slow conns held] status=-1 after 20.6s
  [second legit GET /health, 40s budget] status=-1 after 40.6s
  [after release] status=200 after 0.0s
  [legit while 8 idle conns] status=200 after 4.6s     # idle sockets: 5 s keep-alive wait each
  ```

- **Aggravating factors:**
  - Queued generations also occupy HTTP threads while they wait in `admit()`.
  - `--max-queue` accepts up to 65,536, which would ask httplib for about 65,540 threads (believed; not tested).
- **Mitigation:** see the table. The cheapest effective control is a reverse proxy in front of every non-loopback bind, with `client_header_timeout`, `client_body_timeout` and `limit_conn` in nginx terms. `docs/api.md` currently says only "No TLS: terminate TLS in a reverse proxy". It should say that the proxy is also the availability control.

### S-14 Medium: a hostile chat template takes the server down through one request. Verified

This is S-3 from the previous review, made concrete by the API's A-5 and A-8 gaps. The test models were copies of the tiny GGUF with the template replaced (`gguf.scripts.gguf_new_metadata --chat-template`). Each was served with `--request-timeout 5`, and each received one authenticated `POST /v1/chat/completions`.

```
template {% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}
  one chat request -> curl rc=52 (empty reply) after 5.8 s; SERVER PROCESS DIED rc=139
template {% for i in range(99999) %}{% for j in range(99999) %}x{% endfor %}{% endfor %}
  one chat request -> no reply within 25 s (request_timeout=5 does not cover rendering)
  SIGTERM -> "stopping the server", still running 493 s later; a second SIGTERM exits (rc 143)
```

- **Precondition:** the operator loads a malicious or compromised model. `request_timeout` starts only when generation starts, so it never covers rendering.
- **The "partial" status is confirmed:**
  - Rendering is not isolated (A-5).
  - There is no template allowlist (A-8).
  - HEAD has no template limits, so the S-3 regression test is still owed by `src/template`. The uncommitted `tests/unit/template/test_template_limits.cpp` in the working tree was not reviewed.

### S-15 Medium: prefill is not cancellable. Verified

- **Mechanism:**
  - `run_generation()` checks the client, `request_timeout` and shutdown only inside the `TokenCallback`.
  - `CpuEngine::generate()` invokes that callback only when a token event arrives, and `Request::cancel` is set only when the callback returns false.
  - The whole prompt is prefilled (in `prefill_chunk` ticks) before the first token, whatever happens to the client.
- **Evidence** (a 3,852-token prompt, `max_tokens` 1; the full request took 4.57 s on the tiny model):

  ```
  fire-and-close (send request, close socket immediately):
  0.1 s after the close:  halo_api_active_requests=1, halo_engine_active_sequences=1
  6 s later:              halo_engine_tokens_generated_total 648 -> 649; halo_api_cancelled_total 2 -> 3
  ```

  The proof that prefill ran is the engine token counter. `CpuEngine::emit()` produces a token only after the last prefill chunk (`want_output`), so its increment shows the full prompt was prefilled after the client had gone. The slot also stayed active after the close. `halo_api_prompt_tokens_total` also rose by 3,853, but that is **not** evidence: `run_generation()` counts `req.prompt.size()` before the engine runs. The cancellation is recorded only after the first sampled token.
- **Believed extrapolation:** on the 27B model, a prefill at `ctx` 32768 costs many seconds to minutes. N such requests, each abandoned instantly, hold every generation slot for that long.

### S-16 Medium: admission does not bound pre-generation CPU. Verified cost; order from code

- **Order:** `chat()` runs `parse_request_body`, then `parse_*`, then `build_chat_prompt` (template render plus tokenize), then `make_spec`, and only then `admit()`.
- **Measured costs:**
  - `/v1/chat/completions` with a user message of about 4 MiB: 400 `context_length_exceeded` after 0.94 s.
  - With a 7 MiB tool description: 1.76 s.
  - `/tokenize` on 4 MiB: 0.25 s, and it returns a 4.5 MB body.
- **Consequence:** these costs are paid in full even when the request then gets 429, and `/tokenize` and `/apply-template` are never admitted. One authenticated client can keep every HTTP worker CPU-bound. `docs/api.md` ("Known gaps") and TD-6 state this in part, but they do not say that 429 gives no protection against it.

### S-18 Low: an empty `HALO_API_KEY` disables a key set in the config file. Verified

```
cfg.json: {"server":{"api_key":"filekey-XYZ"}}
HALO_API_KEY= halo serve M --config cfg.json --print-config
  "server.api_key": {"source": "env:HALO_API_KEY", "value": "<redacted>"}
HALO_API_KEY= halo serve M --config cfg.json ...   ->  "halo: serving Hf on http://127.0.0.1:18083"  (no "(API key required)")
GET /v1/models without key -> 200
```

A non-loopback bind in the same situation fails closed with `CONFIG_ERROR` (verified), so the exposure is loopback only. A typical trigger is a service unit or container that passes `HALO_API_KEY=` through with an empty value.

## Verified-good (no finding)

- **Host (loopback bind):** these got 403:
  - `evil.example`, `localhost.`, `127.0.0.2`, `127.0.0.1.nip.io`;
  - an empty Host, a missing Host (HTTP/1.1 and HTTP/1.0);
  - an absolute-form URI with a foreign authority (it gets 401, and never reaches `/health`).

  `LOCALHOST`, `[::1]` and `127.0.0.1` are accepted.
- **Origin:** a foreign origin, `null` and even the server's own origin get 403 on every route, including `OPTIONS` and `/health`.
- **Auth:**
  - These got 401: a missing key, a wrong key, a key with an extra or missing suffix character, `Basic`, two spaces after `Bearer`, and an empty `x-api-key`.
  - These were accepted: lowercase `bearer`, `x-api-key`, and a trailing space (httplib trims header values).
  - Only exactly `/health` (including `?x=1` and `/%68ealth`) skips auth. `//health`, `/health/`, `/HEALTH`, `/health%00`, `/health/../v1/models` and unknown routes all need the key.
  - `constant_time_equal` loops over the configured key's length only, so only length can leak (code).
  - An unauthenticated 7 MiB body is refused with 401 in 0.16 s.
- **Content-Type:** only `application/json`, with any case and parameters, is accepted. `text/plain`, forms, multipart, `application/jsonx`, `application/json, text/plain` and an empty value get 415.
- **Compressed bodies and framing:**
  - `gzip`, `GZIP`, `deflate`, `br` and `zstd` get 415. Other codings pass through undecoded and then fail JSON parsing (S-25).
  - Content-Length 9 MiB and chunked 9 MiB get 413.
  - These got 400: TE+CL, `TE: gzip, chunked`, `TE: chunked, identity` with CL, a CL of 20 digits, CL `-1`, and two CL headers.
- **JSON:**
  - Depth 62, 63 and 64 are accepted. Depth 100 and 100,000 get 400, and so does depth 100,000 inside a `function.arguments` string.
  - Duplicate keys, trailing data, invalid UTF-8 and `1e999` get 400.
  - `max_tokens` values of 0, -1, 1.5, 1e30, 2^64-1, 32,769, `"10"` and `true` get 400. `budget_tokens` 2^64-1 gets 400.
  - The server stayed up throughout.
- **Special tokens (A-1):** in every case over `/apply-template` with `tokenize:true`, the count of control and markup tokens equalled the benign baseline. The cases were:
  - user, system and developer content;
  - content parts `"<|im_"` + `"end|>"` (joined before escaping);
  - a tool message carrying `</tool_response><tool_call>…` plus the payload;
  - a tool description, and a schema property key `<|im_end|>`;
  - assistant control text, argument keys and values, and `reasoning_content` (`</think>` plus the payload).

  More checks:
  - **Anthropic route:** `input_tokens` = 85, which equals the escaped count, against 65 unescaped.
  - **OpenAI chat:** `prompt_tokens` = 85.
  - **Near-literals:** 12 variants were tried, among them case changes, zero-width characters, NUL, fullwidth characters, soft hyphen and partial literals. The tokenizer turned none of them into an added token that the escaper missed.
  - **CLI:** `halo template --tokenize` also keeps the payload as text, emitting 3 `<|im_start|>` tokens, all from the template.
- **Errors and logs:**
  - A template `raise_exception` gets 400 with the message.
  - ESC bytes in a 404 path are shown as `?`.
  - A role containing ESC gets a generic 400.
  - Internal failures return generic 500 text.
- **Admission and cancellation:**
  - With 2 active slots and a queue of 2, six concurrent streams gave 2 immediate, 2 after about 20 s, and 2 × 429.
  - Streaming and non-streaming disconnects each incremented `halo_api_cancelled_total` and freed the slot within about 1.5 s.
  - `--request-timeout 2` ended a 3000-token request at 2.22 s with `finish_reason: length` and a warning.
- **Structured output:** `maxLength` 1e6 and `maxItems` 1e6 are rejected by the grammar compiler; the status code is wrong, see S-24. A nested `{1,1000}` pattern, a recursive `anyOf`, and an enum of 20,000 values each compiled in about 0.5 s.
- **CLI config:**
  - These were rejected: a non-loopback host (`0.0.0.0`, and even `127.1`) without a key, from a flag, the env or the config file.
  - Env parsing is strict: `8080x`, ` 8080`, `-1`, `65536`, `TRUE`, `a,,b`, `2^64`, `0x10` and fullwidth digits all fail with `CONFIG_ERROR`.
  - These fail cleanly: a FIFO, `/dev/zero` and `/proc/self/environ` given as the config file.
  - The key is redacted in `--print-config`, and it did not appear in the server log.
- **Tune:**
  - The model hash prefix `d01fff3e5c95` matches `sha256sum`.
  - `--list` on a missing DB fails and creates nothing (read-only).
  - `--db` pointing through a symlink to a non-SQLite file fails with "file is not a database".

## Per-requirement verdicts (A-1..A-12)

| Req | Implementer's claim | Verdict | Evidence and notes |
|---|---|---|---|
| A-1 Tokenization boundary | done | **Met** | Verified over HTTP for every chat route and client-string location listed above. Residual items are by design: assistant-history markup (S-23), `/v1/completions` (S-22), and the "no partial-literal completion" assumption, which is believed to hold for the Qwen templates. |
| A-2 Request JSON limits | done | **Partial** | Body cap before parsing: verified, including chunked. Depth 64: verified. `max_messages`, `max_tools` and schema size: verified or from code. **But** tool descriptions are outside the content cap, and Anthropic expansion bypasses `max_messages` (S-17). |
| A-3 Token and time limits | done | **Partial** | Prompt must be below context: verified. `max_tokens` cap: verified. Wall-clock deadline: verified (2.22 s). Disconnect cancels: verified. **But** none of these is enforced during render, queue wait or prefill (S-15, S-14). |
| A-4 Kwargs allowlist | done | **Met** | `messages`, `add_generation_prompt`, `bos_token`, `tools` and badly typed values get 400. Verified. |
| A-5 Render isolation and queue cap | partial | **Partial (confirmed)** | The queue cap works: 429 with `Retry-After`, and 503 on timeout. Verified. There is no render isolation: one request kills the process with a hostile template (S-14), and pre-admission CPU is unbounded (S-16). |
| A-6 Network exposure | done | **Met, with Low residuals** | Loopback by default. Host validated on loopback. Origin denied by default. JSON-only POST. Key compared in constant time. Key required for a non-loopback bind. All verified. Residuals: S-18, S-19, S-20. S-13 is a transport issue, filed under A-10. |
| A-7 Model paths and downloads | — (not in the table) | **Met by absence / N/A** | No request field names a path (code: no route reads one). The v0.2 tree has no download code (grep). Model provenance is limited to the model id in the log, with no hash. The download requirements stay open for when downloads are added. |
| A-8 Untrusted-model posture | not done | **Not met (confirmed)** | There is no template allowlist or warning. The model is loaded before `bind()` (code), which is good, but loading does not validate that the template is safe to render (S-14). |
| A-9 Error and log hygiene | done | **Met** | Generic 5xx text, sanitized log lines and 404 bodies, and no `what()` from the exception handler: verified or from code. Parse errors echo only the client's own bytes. Residual: S-24 maps client errors to 500. |
| A-10 httplib configuration | partial | **Partial (confirmed)** | Timeouts, a bounded pool, compile-time header limits (100 × 8 KiB), no decompression (415) and SSE disconnect handling are all present. **But** the bounded pool can be exhausted before auth, and there is no total header or body read deadline (S-13, High). |
| A-11 Output path | done | **Met (believed for nesting)** | `max_output_nesting` (256) guards the tool-argument parser. That is from code plus `ApiCancel.DeeplyNestedModelOutputStopsGeneration`; it was not forced live. Output is bounded by the `max_tokens` cap (verified). |
| A-12 Tests | S-3 owed | **Partial (confirmed)** | The fuzz test passes under ASan+UBSan at HEAD (64/64, verified). Regression tests exist for S-1 (`ApiInjection.*`, `PromptReal.*`), S-2 (`ApiLimits.DeeplyNestedJson…`) and S-8 (`ApiLimits.TemplateKwargs…`). S-3 has no regression test at HEAD. I did **not** mutation-test these tests (I never saw one go red). |

**Summary:** the implementer's four stated gaps (A-5 partial, A-8 not done, A-10 partial, S-3 owed) are all **confirmed**. Three more claimed as done are **partial**:
- A-2 (S-17);
- A-3 (S-15);
- A-12 (no evidence that the tests can fail).

A-10's gap is larger than "fixed transport limits": it is S-13.

### Documentation claims refuted

- **`include/halo/api/server.h`** says "one failing request never takes the process down", and **`docs/api.md`**, under "Crash containment (RR-005)", says the same. **Refuted by S-14:** with a hostile template, one request ends the process with SIGSEGV. Handlers catch exceptions, but they cannot catch a stack overflow.
- **`docs/api.md`, "Known gaps"** says a hostile template "can still crash or hang a worker". **This understates it:** the crash takes down the whole process, not one worker, and a hung render also blocks SIGTERM shutdown (S-14).
- **`docs/api.md`, "Compressed bodies"** says a body with a known compression coding gets 415. **Partly refuted:** `gzip, identity`, `identity, gzip` and `x-gzip` pass through undecoded (S-25). This is harmless, because nothing is decompressed.
- **`docs/api.md`, "Resource governance"** says `/tokenize` and `/apply-template` are bounded by the HTTP pool. That is true, but the pool can be exhausted before auth (S-13), and 429 gives no protection against pre-admission CPU (S-16).
- **`docs/api.md`, "Special-token handling"** says client strings cannot inject control tokens. **Confirmed** for chat routes. It does not hold for `/v1/completions` by design, and the documented opt-out cannot be reached from `halo serve` (S-22).
- **`docs/cli.md`, "Strict parsing"** is **confirmed** for env and CLI values. However, a duplicate key in the config file is accepted silently (last value wins), and an empty env value silently overrides a file value (S-18).

## What I could not test

- **Scale:** the real 27B model, and GPU backends (Vulkan was off, HIP needs ROCm). All engine behaviour was measured on the CPU reference with the tiny model. The prefill-cost magnitude in S-15 is extrapolated.
- **The prefix-cache side channel (S-27):** prefix checkpoints were disabled on this host (no GPU tier), so this rests on code reading.
- **A slow SSE reader** (a client that stops reading without closing) holding a slot until `write_timeout` or `request_timeout`: believed from code, not run.
- **`--max-queue 65536`** producing a very large thread pool at startup: believed.
- **Cross-user symlink attacks in a shared non-sticky directory:** my test user could not traverse `/root`. The sticky `/tmp` case was verified blocked by the kernel.
- **Timing measurements of the key comparison:** code review only.
- **The output-nesting guard live:** the tiny model could not be steered into emitting deep brackets.
- **Other hostile model files:** the malicious-model DoS was tested with two templates only. Tokenizer and GGUF amplification (S-4, S-6, S-7) were not re-run through the server.
- **Mutation testing** of the API security tests, to show they can go red.
- **The working tree:** the uncommitted `src/template` limits, `src/profiling` HTTP client and `tests/unit/integration` were out of scope by instruction.

## Risk acceptance

Nothing has been accepted. S-13 blocks non-loopback deployment unless the owner or `release-manager` records one of two things here:
- a decision that v0.2 is loopback-only;
- a decision that non-loopback deployments must sit behind a proxy with connection and header timeouts.

S-14 inherits the S-3 rationale question from the previous review: "models are operator-selected", weakened by Hugging Face being an untrusted source. That decision belongs to the owner too.

## Re-test at adb3c46 (2026-09-25)

The original findings above are unchanged. This section records re-tests of S-14 and S-16 against `adb3c46`, which contains `09b7c22` (the pinned minja limits patch) and `8235c3b` (the tokenizer S-6/S-7 fixes).

- **Build:** `git archive adb3c46` into a fresh build directory, `/root/halo-build-secapi2`. The minja patch was applied at FetchContent: the `MINJA_HALO_LIMITS` marker is present in `_deps/minja-src`. The old `/root/halo-build-secapi` tree has no marker.
- **Baseline at this commit (measured):** `test_template` 40/40, `test_api` 64/64, `test_profiling` 85/85, `test_autotune` 49/49, all passed.
- **Setup:** the same method as before. The tiny GGUF was re-templated with `gguf_new_metadata --chat-template`, then served with `halo serve --parallel 2 --max-queue 2 --request-timeout 5` and an API key.
- **Timings:** these are from a quiet host (load about 5). A first run at load average about 90 (other agents building) showed the same verdicts, with roughly 3× the times.

### S-14: mitigated at adb3c46 (Verified); the residual is Low and folded into S-16

| Template | At 6a54d3f | At adb3c46: one chat request | Server after | SIGTERM |
|---|---|---|---|---|
| `{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}` | **process died, SIGSEGV (rc 139)** | 400 "template render depth limit (1024) exceeded", 0.06 s | alive; `/health` 200 | exits 0 in 0.11 s |
| nested `range(99999)` loops | no reply in 25 s; shutdown hung more than 8 min | 400 "loop iteration limit (1000000) exceeded", 1.1–1.8 s | alive | exits 0 in 0.28 s |
| `'x' * 1000000` printed 99,999 times | (OOM class, previous review) | 400 "output size limit (67108864) exceeded", 0.36–0.40 s | alive | exits 0 in 0.42 s |
| `{{ 'x' * 100000000 }}` | 100 MB output | 400 "value size limit (33554432) exceeded", 0.07–0.09 s | alive | exits 0 in 0.34 s |
| exponential macro `f(40)` | not run | 400 "step limit (4000000) exceeded", 1.9–3.5 s | alive | exits 0 in 0.25 s |
| 5000 parentheses | parse crash (previous review) | `halo serve` refuses to start: `CONFIG_ERROR: … Template nesting too deep (limit 256)` | — | — |
| 100 KB comment | parse crash (previous review) | `halo serve` refuses to start: `CONFIG_ERROR: chat template is 100011 bytes (limit 65536)` | — | — |

For each template, four concurrent hostile requests finished in 0.08–2.6 s, and `/health` answered 200 afterwards.

**Verdict:**
- **The S-14 failure modes are gone.** The process no longer dies, render no longer hangs, and shutdown is no longer blocked. The `server.h` claim "one failing request never takes the process down" now holds for the template paths tested.
- **What remains is bounded, not closed.** Each hostile-template request still costs up to the step budget, about 1–3.5 s of CPU here and about 4 s by `docs/security-hardening.md`. That cost is paid on an HTTP worker **before admission**. This is the S-16 mechanism, and it is still open (below).
- **S-14 is re-graded from Medium to Low** (malicious-model precondition, bounded cost).
- **Requirement status:**
  - A-5 stays **Partial**: there is a render budget, but no separate bounded pool, and render still happens before admission.
  - A-8 stays **Not met**: there is still no template SHA-256 allowlist or warning.
  - A-12's S-3 regression tests now exist (`test_template_limits.cpp` via `test_template`, 40/40). I did not mutation-test them.

### S-16: unchanged at adb3c46 (Verified); stays Medium

Both builds were measured back to back in one session with the same requests. The table shows three repetitions each.

| Request | 6a54d3f | adb3c46 |
|---|---|---|
| chat with a 7 MiB tool description (400 `context_length_exceeded`, 1,359,538 tokens) | 1.01–1.04 s | 1.08–1.16 s |
| chat with about 4 MiB of user content (400, 776,758 tokens) | 0.61–0.66 s | 0.61–0.67 s |
| `/tokenize` on 4 MiB | 0.14–0.17 s | 0.15–0.16 s |
| `/apply-template` on 4 MiB with `tokenize:true` | 0.74–0.81 s | 0.74–0.82 s |

- **Unchanged within noise.** The tokenizer trie (`8235c3b`) targets hostile vocabularies (S-6); it does not change the cost of benign text, and it was not expected to.
- **Still open:** rendering and tokenizing still run before `admit()`, and `/tokenize` and `/apply-template` are still not admitted.
- **The mitigation stands as written:** admit, or take a cheap pre-admission token, before rendering.

### Updated counts after the re-test

- **Counts:** High 1 (S-13), Medium 2 (S-15, S-16), Low 8 (S-14 re-graded, plus S-17 to S-22 and S-24), Informational 5.
- **Gate:** S-13 still blocks non-loopback deployment. With a loopback-only posture, nothing blocks.
