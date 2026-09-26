# HALO HTTP API (v0.2)

HALO serves one model over HTTP/1.1. It speaks the OpenAI Chat Completions and Completions
formats and the Anthropic Messages format, and it has a few utility routes. The server is
`halo::api::ApiServer` (`include/halo/api/server.h`), built on cpp-httplib. `halo serve` starts it (see "Running the server").

## Running the server

```
halo serve --model model.gguf [--mtp mtp.gguf] [--host 127.0.0.1] [--port 8080] [--ctx N]
           [--parallel N] [--backend auto|cpu|vulkan|hip] [--cors-origins a,b]
           [--allowed-hosts h1,h2] [--served-model-name NAME] [--max-queue N]
           [--max-concurrent N] [--max-body-bytes N] [--max-tokens-cap N]
           [--default-max-tokens N] [--request-timeout S] [--allow-unauthenticated-remote]
           [--config file.json] [--print-config]
```

- **Where settings come from** (TRD §44): the command line, then `HALO_*` environment
  variables (for example `HALO_PORT` or `HALO_API_KEY`; `halo serve --help` lists every
  one), then a JSON config file given by `--config` or `HALO_CONFIG`, then the built-in
  defaults. The config file has `model`, `runtime` and `server` sections. It is parsed
  strictly: an unknown setting or a wrong type is an error. **YAML is not supported in
  v0.2.**
- **API key.** Pass it as `HALO_API_KEY` rather than `--api-key`: other local users can read
  a process's command line.
- **Check the configuration.** `--print-config` prints the resolved configuration, with the
  source of every value and the API key redacted, and exits without loading the model.
- **Unsafe binds fail early.** A non-loopback `--host` with no API key is refused before the
  model is loaded.
- **Stopping.** SIGINT or SIGTERM stops the server gracefully: in-flight generations are
  cancelled and queued requests get 503. SIGPIPE is ignored.
- **Engine.** The server needs the inference runtime (`halo_runtime`). A build without it
  prints "runtime not built" and exits 2. `halo version` lists the components a build
  contains.

Everything below describes what the code in `src/api/` does. Tests with a scripted fake
engine (`tests/unit/api/`) check this behaviour, but nothing in this document has been
checked against the hosted OpenAI or Anthropic services. The response-shape fixtures in
`tests/unit/api/fixtures/` were written by hand from the public API references.

## Endpoints

| Method and path | Purpose | API key required |
|---|---|---|
| `GET /health` | Liveness: `{"status":"ok"}`, or 503 `{"status":"stopping"}` during shutdown | no |
| `GET /v1/models` | The served model, as one list entry | yes, if a key is configured |
| `GET /metrics` | Prometheus text exposition, format 0.0.4 | yes, if a key is configured |
| `POST /v1/chat/completions` | OpenAI chat, non-streaming or SSE | yes, if a key is configured |
| `POST /v1/completions` | OpenAI legacy completions on a raw prompt | yes, if a key is configured |
| `POST /v1/messages` | Anthropic Messages, non-streaming or SSE | yes, if a key is configured |
| `POST /tokenize` | Tokenize a string | yes, if a key is configured |
| `POST /apply-template` | Render (and optionally tokenize) a chat with the model's template | yes, if a key is configured |
| `OPTIONS *` | CORS preflight, answered only for configured origins | no |

Any other route returns 404 in the OpenAI error format. The error format is Anthropic's for
`/v1/messages*` and OpenAI's for everything else.

## Security defaults

These implement PRD §12 and the security reviews of 2026-09-24 and 2026-09-25
(`docs/reviews/`).

- **v0.2 serving is loopback-only (DECISIONS.md D-017).** The default bind is `127.0.0.1`.
  - **Opt-in.** Binding any non-loopback address fails with `CONFIG_ERROR` unless
    `allow_remote` is set (`halo serve --allow-remote`), and the server then logs a warning.
  - **API key.** Without an API key, a non-loopback bind additionally needs
    `allow_unauthenticated_remote`. It also needs a non-empty `allowed_hosts`, so that the
    DNS-rebinding Host check stays on (review S-20).
- **A reverse proxy is required for every non-loopback deployment** (D-017). It must enforce
  per-client connection limits and header and body timeouts: in nginx terms, `limit_conn`,
  `client_header_timeout` and `client_body_timeout`. It also terminates TLS.
  - HALO's own read deadlines, per-peer cap and overload shedding (below) keep `/health`
    and complete requests served under slow-header floods, but they cannot tell apart
    clients behind one address, and they act once per scan (200 ms).
  - Without the proxy, a peer that holds more connections than `http_threads` plus
    `http_queue` still gets new connections refused at accept, for everyone.
- **API key.** An empty key is a `CONFIG_ERROR`, never "no key". The CLI treats an empty
  `HALO_API_KEY=` as unset (review S-18). When `api_key` is set, every route except
  `GET /health` and CORS preflight
  requires `Authorization: Bearer <key>` or `x-api-key: <key>`. The `bearer` scheme is
  matched case-insensitively. The comparison runs in constant time: its duration depends only
  on the lengths. A missing or wrong key gets 401 plus `WWW-Authenticate: Bearer`.
- **Host header (DNS rebinding).** On a loopback bind, only `localhost`, `127.0.0.1`,
  `[::1]`, the bind address and `allowed_hosts` are accepted as the Host (the port is
  ignored). Any other Host gets 403. On a non-loopback bind with an empty `allowed_hosts`,
  the Host is not checked, and the API key is then mandatory (see above).
- **Origin (CSRF).** A request whose `Origin` header is not in `cors_origins` gets 403 on
  every route, before any handler runs. `cors_origins` holds exact origins or `"*"`. It is
  empty by default, which means no browser origin is allowed. `"*"` requires an API key
  (`CONFIG_ERROR` otherwise, review S-19), because it would let any web page use the model. For an allowed origin the
  server returns `Access-Control-Allow-Origin: <origin>` and `Vary: Origin`.
- **Content type.** A POST body must be `application/json`, otherwise 415. This blocks
  HTML-form CSRF, which sends `text/plain` or `application/x-www-form-urlencoded` without a
  preflight.
- **Compressed bodies.** The server is built without decompression.
  - A request whose `Content-Encoding` is exactly one known coding (`gzip`, `deflate`, `br`
    or `zstd`) gets 415.
  - Other values pass through undecoded and the body is parsed as JSON. That covers
    unrecognized codings (`x-gzip`) and lists (`gzip, identity`). A body that really is
    compressed then fails as malformed JSON (review S-25). Nothing is ever decompressed.
- **Read deadlines and load shedding (reviews S-13, S-38).** Slow-header clients need no
  API key (auth runs after the headers are read), so a watchdog keeps them from holding
  every HTTP worker. It scans the server's sockets every 200 ms (`src/api/conn_guard.h`):
  - **Header deadline.** A complete header block must arrive within `header_timeout`
    (default 10 s). The clock starts when a worker has read the first byte of the request,
    or at the previous response on a keep-alive connection. A connection waiting in the
    queue with its request already sent has no clock running, so it is not closed while it
    waits. A connection that sends nothing is timed from accept.
  - **Body deadline.** A complete body must arrive within `body_timeout` (default 60 s)
    of the headers.
  - **Per-peer cap.** One peer address may hold at most `max_header_connections_per_peer`
    (default 8) connections that are stalled before their first request: still reading
    headers, never served, and without a complete header block in the receive buffer. The
    oldest beyond the cap are closed.
  - **Overload shedding.** When a complete request has waited a whole scan interval for a
    worker, the pool is saturated. Every connection still reading headers that is older
    than `header_shed_grace` (default 1 s) and is not a complete buffered request is then
    closed: dribbling, silent and idle keep-alive connections alike. Complete requests are
    never shed, and handlers (generation, streaming) are never touched.
  - Closed connections are counted in `halo_api_connection_deadline_closes_total` (all
    reasons) and `halo_api_connection_closes_total{reason}` (`deadline`, `peer_limit`,
    `overload`).
  - Measured with the review's S-38 repro (WSL, tiny model, `--parallel 1`, so 31 HTTP
    threads and a 256-slot queue; 100 dribblers that reconnect as soon as they are closed,
    `/health` probed every 0.5 s with a 3 s timeout for 40 s): `/health` answered 100% of
    probes with the dribblers on the client's own address (median 0.11 s), on one other
    address (0.11 s), and on 100 different addresses (0.71 s). Before this change it was 0%
    in all three cases. With 400 dribblers, more than threads plus queue, it answered 31%:
    new connections are then refused at accept, which only the reverse proxy can prevent.
  - Enforcement is Linux-only.
  - Residual: a request that sends its headers promptly and then dribbles its body holds a
    worker for up to `body_timeout`; overload shedding does not cover the body phase.
- **Body size.** `max_body_bytes` (default 8 MiB) is enforced by httplib before the body is
  parsed. A larger body gets 413.
- **Strict JSON.** The parser accepts RFC 8259 only: no comments, no trailing data, and valid
  UTF-8. Duplicate object keys are rejected. The top level must be an object. A number that
  overflows a double (for example `1e999`) is a 400.
  - Nesting is capped at parse time, inside the parser callback, before any recursive copy,
    dump or template conversion exists (review S-2).
  - With the default `max_json_depth` of 64, a value may sit at most 64 containers deep,
    counting the top-level object. An *empty* innermost container is reported at its parent's
    depth, so it may be the 65th level.
  - The same cap applies to JSON embedded in strings (OpenAI `tool_calls[].function.arguments`).
- **Size caps:**
  - `max_messages` (4096);
  - `max_tools` (256);
  - `max_tool_schema_bytes` for each whole tool definition, description included, and for
    each structured-output schema (64 KiB);
  - `max_content_bytes`, the total bytes of every string in the conversation **and the tool
    definitions** (4 MiB; 413; review S-17);
  - Anthropic `tool_result` blocks, each of which becomes a template message, count toward
    `max_messages`;
  - `max_stop_sequences` (16, each at most 256 bytes);
  - `max_tokens_cap` (32768; a larger request is a 400).
- **No file paths.** No request field names a file. The model is chosen by the operator at
  startup.
- **Special tokens.** Client strings cannot inject control tokens. See "Special-token
  handling" below.
- **Errors.** Internal failures return generic messages such as "internal error" or
  "generation failed". The full text, sanitized, goes to the log only. Template
  `raise_exception` text and request-validation messages are returned, sanitized and
  truncated to 512 bytes. Log lines strip C0/C1 control characters, so model-supplied or
  client-supplied strings cannot inject terminal escapes.
- **Crash containment (RR-005).** Every handler converts exceptions into the native error
  body. httplib's exception handler never exposes `what()`.
  - A stack overflow cannot be caught. The bounds that prevent one are the JSON depth caps and
    the template limits in `src/template` (render depth, loop iterations, output size;
    re-tested in the 2026-09-25 security review, S-14).

## Resource governance

`http_queue`, `max_header_connections_per_peer` and `header_shed_grace` are `ServerConfig`
fields only for now: `halo serve` has no flag, environment variable or config-file key for
them yet, so the CLI always uses the defaults.

| Setting (`ServerConfig`) | Default | Effect |
|---|---|---|
| `max_concurrent` | 4 | Generations holding a slot at once. |
| `max_queue` | 16 | Generations waiting for a slot. When the queue is full: **429** plus `Retry-After: 1`. |
| `queue_timeout` | 30 s | A queued request that waits longer gets **503** plus `Retry-After: 1`. |
| `request_timeout` | 600 s | Wall-clock limit for one generation, checked each time the engine delivers a token. The generation then ends like `max_tokens`, with a warning. Time spent in engine prefill before the first token is not interrupted (see "Cancellation"). |
| `default_max_tokens` | 8192 | Used when a request gives none. It is clamped to the context. |
| `max_tokens_cap` | 32768 | A request asking for more gets 400. |
| `reasoning_output_reserve` | 512 | Output headroom kept free of reasoning. See "Reasoning". |
| `max_output_nesting` | 256 | Unmatched `[`/`{` allowed in model output (review S-9). |
| `utility_concurrency` / `utility_queue` | 2 / 8 | Admission for `/tokenize` and `/apply-template`. Beyond it: 429 (review S-16). |
| `header_timeout` / `body_timeout` | 10 s / 60 s | Per-connection read deadlines (reviews S-13, S-38). 0 = unlimited. |
| `max_header_connections_per_peer` | 8 | Connections one address may hold before its first request (review S-38). 0 = no cap. |
| `header_shed_grace` | 1 s | Overload shedding of header-phase connections older than this (review S-38). 0 = off. |
| `http_threads` | 0 (`max_concurrent + max_queue + utility_concurrency + utility_queue + 4`) | Size of the HTTP worker pool. `max_queue` and `utility_queue` are capped at 4096. |
| `http_queue` | 256 | Connections accepted and waiting for an HTTP worker (1-65536). One more is closed at accept. Every queued connection holds a descriptor; the server warns at start when `http_threads + http_queue` comes within 64 of the open-file limit. |
| `read_timeout` / `write_timeout` / `keep_alive_timeout` | 60 s / 60 s / 5 s | Socket timeouts. At most 100 requests per keep-alive connection. |

**Admission comes before the expensive work (review S-16).**
- A chat or completion request is parsed and validated, which is cheap and size-capped, and
  then admitted. Only an admitted request renders the template and tokenizes. So a request
  refused with 429 has not paid for rendering, which can take about a second for a
  multi-MiB request.
- `/tokenize` and `/apply-template` pass their own small admission (`utility_concurrency`,
  `utility_queue`).
- `/metrics`, `/health` and `/v1/models` are not queued.

**Cancellation.** The API can stop an engine request only from its token callback, which
the engine calls once per generated token.
- **Client disconnect.** A streaming client that disconnects is detected on the next write
  that fails. The token callback then returns false and the engine stops that request. A
  non-streaming client that disconnects is detected by polling the socket once per token.
- **Shutdown.** `ApiServer::stop()` cancels generations that are producing tokens. Requests
  waiting in the API's admission queue get 503.
- **Limitation: before the first token.**
  - Nothing can reach a request that is still prefilling (a long prompt) or waiting for an
    engine sequence. Disconnect, `request_timeout` and `stop()` all take effect at its first
    token, so `stop()` may wait for such requests to finish prefill.
  - This needs an engine-side cancel path, which WS-G is adding (review R-3).
  - Until then, `halo serve` caps `--max-concurrent` at `--parallel`, so that requests queue
    in the API, where `queue_timeout` and shutdown apply, rather than inside the engine.

## Chat request mapping

### OpenAI `POST /v1/chat/completions`

| Field | Support |
|---|---|
| `model` | Accepted and ignored. The response reports the served name. |
| `messages` | Roles `system`, `developer` (rendered as `system`), `user`, `assistant`, `tool`. Content is a string, null, or an array of `text` / `refusal` parts. Parts of type `image_url`, `input_audio`, `file` or `image` get 400 `unsupported_parameter`. An assistant message may carry `reasoning_content` and `tool_calls`; `arguments` is a JSON string, which is parsed under the depth cap, or an object. A tool message may carry `tool_call_id`. |
| `tools` | `type: "function"` only, with `name`, `description` and `parameters`. Names are 1–128 characters from `[A-Za-z0-9_.:-]`, because they are rendered into template markup. |
| `tool_choice` | `auto` (the default) and `none`. `required` and a named function get 400 `unsupported_parameter`. |
| `parallel_tool_calls` | Accepted, not enforced. |
| `reasoning_effort` | `none` turns thinking off. `minimal`/`low` map to `low`, `medium` to `medium`, and `high`/`xhigh`/`max` to `xhigh` (the Qwen3.8 template values). Anything else gets 400. |
| `chat_template_kwargs` | Allowlist: `enable_thinking`, `preserve_thinking`, `reasoning_effort`. Any other key gets 400. That includes `messages`, `tools`, `bos_token` and `add_generation_prompt` (review S-8, A-4). |
| `temperature` (0–2), `top_p`, `top_k`, `min_p`, `typical_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty` (±2), `seed` | Passed to `SamplingParams`. |
| `max_completion_tokens`, or else `max_tokens` | Must be at least 1 and at most `max_tokens_cap`. If it does not fit in the context it is clamped, with a warning. |
| `stop` | A string or an array. Stop sequences are matched on content, never on reasoning, and are excluded from the output. |
| `response_format` | `text`; `json_object` sets `SamplingParams::json_object`; `json_schema` sets `SamplingParams::json_schema` to the serialized `json_schema.schema`. The schema is capped at `max_tool_schema_bytes` and compiled when the request is parsed, so an invalid schema gets 400 `invalid_request_error` and an unsupported construct gets 400 `unsupported_parameter`, before any streaming starts. Constrained decoding is done by the engine. |
| `stream`, `stream_options.include_usage` | SSE; see below. |
| `n` > 1, `logprobs: true`, `top_logprobs`, non-empty `logit_bias` | 400 `unsupported_parameter`. |
| Other fields (`user`, `metadata`, `store`, …) | Ignored. |

**Response** (`chat.completion`):
- `choices[0].message` has `role`, `content` (null when only tool calls were produced),
  `reasoning_content` (when non-empty) and `tool_calls` (`id`, `type`, and `function.name` /
  `function.arguments` as a JSON string).
- `finish_reason` is `stop`, `length` or `tool_calls`.
- `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`.
- A non-standard `warnings` array appears only when there is something to report: a
  reasoning budget close, a clamped `max_tokens`, a time limit, or empty content.

**Streaming:**
- The first `chat.completion.chunk` has `delta: {"role":"assistant","content":""}`.
- Then come `delta.reasoning_content` and `delta.content` chunks. Each tool call is one chunk
  with `delta.tool_calls: [{index, id, type, function: {name, arguments}}]`, sent complete
  once its `</tool_call>` is parsed.
- A final chunk has an empty delta and `finish_reason` (plus `warnings`).
- With `include_usage`, a chunk with `choices: []` and `usage` follows.
- The stream ends with `data: [DONE]`.
- An error after the headers were sent arrives as a `data:` event holding the OpenAI error
  object, followed by `[DONE]`.

### Anthropic `POST /v1/messages`

| Field | Support |
|---|---|
| `model` | Accepted and ignored. |
| `max_tokens` | Required, at most `max_tokens_cap`. |
| `system` | A string, or an array of text blocks. |
| `messages` | Roles `user` and `assistant`. Content is a string or blocks. Users: `text`, and `tool_result` (`tool_use_id`; string or text-block content; `is_error` prefixes `Error: `). Assistants: `text`, `thinking` (rendered as `reasoning_content`), `redacted_thinking` (dropped) and `tool_use`. `image` and `document` get 400. A final assistant message (prefill) gets 400 (unsupported). |
| `tools` | Custom tools: `name`, `description`, `input_schema`. Server tools (any other `type`) get 400. |
| `tool_choice` | `auto` and `none`. `any` and `tool` get 400 (unsupported). `disable_parallel_tool_use` is accepted, not enforced. |
| `thinking` | `enabled` with `budget_tokens` < `max_tokens` (an explicit reasoning budget); `disabled` (thinking off); `adaptive` (the default behaviour). |
| `output_config.effort` | `low`, `medium`, `high`, `xhigh` or `max`, mapped as for OpenAI. `none` gets 400. |
| `output_config.format`, or the deprecated `output_format` | `{"type":"json_schema","schema":{…}}` goes to `SamplingParams::json_schema`. |
| `temperature` (0–1), `top_p`, `top_k` | Passed to `SamplingParams`. |
| `stop_sequences` | As OpenAI `stop`. |
| `stream` | SSE. |
| Other fields (`metadata`, `service_tier`, …) | Ignored. The `anthropic-version` / `anthropic-beta` headers are accepted and ignored. |

**Response** (`message`):
- `content` blocks come in this order: `thinking` (with `signature: ""`), `text`, then
  `tool_use` (`id` `toolu_…`, `name`, `input` object).
- `stop_reason` is `end_turn`, `max_tokens`, `stop_sequence` or `tool_use`. `stop_sequence`
  is set when one matched.
- `usage` has `input_tokens` (the prompt minus cached tokens), `output_tokens`,
  `cache_read_input_tokens` and `cache_creation_input_tokens` (always 0).

**Streaming:** the events are
- `message_start`;
- then, for each block, `content_block_start`, `content_block_delta` and
  `content_block_stop`. The deltas are `thinking_delta`, `text_delta`, or one complete
  `input_json_delta` per tool call;
- then `message_delta`, carrying `stop_reason`, `stop_sequence` and `usage.output_tokens`;
- then `message_stop`.

Each SSE `event:` name equals the `type` in its data. An error after the headers were sent
arrives as `event: error`.

### OpenAI `POST /v1/completions`

`prompt` is a string, or a one-element array of strings. Batches and token arrays get 400.
The other fields are as for chat: sampling, `max_tokens`, `stop`, `stream` and
`stream_options`. `echo`, `suffix`, `best_of` > 1 and `logprobs` get 400. The output is raw
text: no reasoning, tool or template parsing. By default the prompt is tokenized with special
tokens parsed (`completions_parse_special = true`), because the client wrote the whole
prompt. This is a deliberate exception to the chat routes' injection protection. To treat
the prompt as plain text, set it to false: `halo serve --no-completions-parse-special`,
`HALO_COMPLETIONS_PARSE_SPECIAL=false`, or `server.completions_parse_special` in the config
file (review S-22).

### `POST /tokenize` and `POST /apply-template`

- **`/tokenize`:** takes `{"content": str, "parse_special": false, "with_pieces": false}` and
  returns `{"tokens": [id, …]}`, or `[{"id", "piece"}, …]` with `with_pieces`.
- **`/apply-template`:** takes OpenAI-form `messages` and `tools`, plus
  `add_generation_prompt` (default true), `tokenize` (default false), `reasoning_effort` and
  `chat_template_kwargs`, validated as for chat. It returns
  `{"prompt": str, "tokens": [...]?, "neutralized_literals": n}`. `tokens` are exactly what a
  chat request would feed the model, including the special-token protection.

### `GET /v1/models` and `GET /metrics`

`/v1/models` returns
`{"object":"list","data":[{id, object:"model", type:"model", created:0, created_at, owned_by:"halo", display_name, architecture, context_length, has_mtp}], has_more:false, first_id, last_id}`.
It is a superset of the OpenAI and Anthropic list shapes.

`/metrics` (PRD §13) exposes:

- **API side:**
  - `halo_api_requests_total{route,status}`
  - `halo_api_rejections_total{reason}`, where reason is one of `queue_full`,
    `queue_timeout`, `unauthorized`, `host`, `origin`, `content_type` or `too_large`
  - `halo_api_active_requests`
  - `halo_api_queued_requests`
  - `halo_api_prompt_tokens_total`
  - `halo_api_completion_tokens_total`
  - `halo_api_reasoning_tokens_total`
  - `halo_api_cancelled_total`
  - `halo_api_reasoning_budget_closes_total`
  - `halo_api_last_decode_tokens_per_second`
  - the histograms `halo_api_time_to_first_token_seconds` and
    `halo_api_generation_duration_seconds`
  - `halo_api_connection_deadline_closes_total` and
    `halo_api_connection_closes_total{reason}` (`deadline`, `peer_limit`, `overload`), see
    "Read deadlines and load shedding"
- **Engine side**, taken from `Engine::stats()` at scrape time:
  - `halo_engine_active_sequences`
  - `halo_engine_queued_requests`
  - `halo_engine_tokens_generated_total`
  - `halo_engine_prefix_cache_{hits,misses}_total`
  - `halo_engine_prefix_cache_hit_ratio`
  - `halo_engine_speculative_{attempts,accepts}_total`
  - `halo_engine_speculative_acceptance_ratio`

Hardware and kernel metrics (PRD §13) are not exposed in v0.2.

## Reasoning (TRD §25)

- **Default.** Thinking is on by default: the template opens `<think>`. Output is split into
  reasoning and content by `chat::OutputParser`. `reasoning_effort: "none"`,
  `chat_template_kwargs.enable_thinking: false` and Anthropic `thinking.type: "disabled"` all
  render with `enable_thinking=false`.
- **Budget.** Reasoning may use `reasoning_budget` tokens. For Anthropic this is
  `thinking.budget_tokens`. Otherwise it is `max_tokens - min(reasoning_output_reserve,
  max_tokens / 2)`, so even a small `max_tokens` leaves at least half of it for the answer.
  When the budget runs out while the model is still thinking:
  1. HALO cancels that engine call.
  2. It appends `\n</think>\n\n` to the context.
  3. It continues in a second engine call with the remaining tokens.
  4. It adds a warning and counts `halo_api_reasoning_budget_closes_total`.
- **Accounting.**
  - `reasoning_tokens` counts tokens inside the think block, including the `</think>` token.
  - `completion_tokens` counts every token the engine produced, over both calls. It includes
    the EOS token.
- **Empty content.** If generation ends at the length or time limit with no content and no
  tool calls, but reasoning tokens were produced, the response carries the warning "no
  content was produced: reasoning used N of the M max_tokens; …".

### Structured output and thinking (finding F-1)

A request with `response_format` `json_schema` or `json_object`, or with Anthropic
`output_config.format`, is rendered with **thinking off** (`enable_thinking=false`, the
template's non-thinking form, which ends with `<think>\n\n</think>\n\n`).
- **The problem.** With thinking on, the generation prompt ends inside `<think>\n`. The
  grammar constrains the output from its first token and does not allow `</think>`. So the
  schema-valid JSON would be returned as `reasoning_content`, and `content` would be empty.
- **Why not pre-fill `</think>` instead.** The alternative was to prefill
  `\n</think>\n\n` before constrained decoding. It was not chosen:
  - Thinking off uses the template's own documented mode, so the prompt equals HF
    `apply_chat_template(..., enable_thinking=False)`, and the prefix cache sees the same
    prefix as any other non-thinking request.
  - A prefill would add a second, HALO-specific prompt shape and extra tokens outside the
    template.
- **Warnings.** An explicit thinking request that is overridden this way (`reasoning_effort`,
  `enable_thinking: true`, or Anthropic `thinking.type: "enabled"`) gets the warning
  "structured output … disables thinking".

## Special-token handling (review S-1, A-1)

The template renders client strings into one prompt, and that prompt must be tokenized with
special tokens parsed. Encoding it as one string would turn a literal `<|im_end|>` in a user
message into the real control token. HALO therefore escapes, renders, and then splits at the
escapes:

1. **Escape.** Added-token literals are replaced by an alphanumeric placeholder that carries
   a per-request random nonce. The call is refused if the nonce already occurs in the input.
   - Control tokens are replaced in every client string, including tool definitions.
   - UserDefined markup (`<think>`, `<tool_call>`, `<tool_response>`, FIM markers) is replaced
     in system, user and tool strings and in tool definitions, but **not** in assistant
     history. Assistant history is the model's own earlier output, and the template itself
     parses `</think>` there.
2. **Render.** The model's template renders the escaped messages.
3. **Split.** The rendered text is split at the placeholders. Template text is tokenized with
   specials parsed. Each placeholder becomes its literal tokenized as plain text: no added
   token can come out, even for UserDefined tokens that match without `parse_special`.

When no client string contains a literal, which is the normal case, the tokens equal
`encode(rendered, parse_special=true)`, i.e. HF `apply_chat_template` plus tokenize.

**Residual assumption:** template text next to a client string does not complete a partial
literal that the string starts or ends with. The Qwen3.8 templates follow content with
`<|im_end|>`, `\n</tool_response>` or `\n`, and none of these can.

Tests: `PromptFixture.*` and `PromptReal.*` (the real Qwen tokenizer and template), plus
`ApiInjection.*` over HTTP.

## Error format

| HTTP | Kind | OpenAI `error.type` (`code`) | Anthropic `error.type` |
|---|---|---|---|
| 400 | invalid request | `invalid_request_error` | `invalid_request_error` |
| 400 | unsupported feature | `invalid_request_error` (`unsupported_parameter`) | `invalid_request_error` |
| 401 | missing or bad key | `invalid_request_error` (`invalid_api_key`) | `authentication_error` |
| 403 | Host / Origin refused | `invalid_request_error` | `permission_error` |
| 404 | no route | `invalid_request_error` | `not_found_error` |
| 413 | body or content too large | `invalid_request_error` (`request_too_large`) | `request_too_large` |
| 415 | not JSON / compressed | `invalid_request_error` (`unsupported_media_type`) | `invalid_request_error` |
| 429 | queue full | `rate_limit_error` (`rate_limit_exceeded`) | `rate_limit_error` |
| 500 | internal / engine failure | `server_error` | `api_error` |
| 503 | queue timeout / shutting down / out of memory | `server_error` (`overloaded`) | `overloaded_error` |

- **OpenAI body:** `{"error":{"message","type","param","code"}}`. `param` names the offending
  field, for example `messages[1].tool_calls[0].function.arguments`.
- **Anthropic body:** `{"type":"error","error":{"type","message"}}`. The field name is
  prefixed to the message.
- A context overflow is a 400 with `code: "context_length_exceeded"`.

## Deviations from OpenAI and Anthropic

- Unknown request fields are ignored. OpenAI rejects some of them.
- There is one choice (`n = 1`), no logprobs, and no `logit_bias`.
- Tool choice cannot force a tool: OpenAI `required` or a named function, Anthropic `any` or
  `tool`. Parallel-tool-call flags are not enforced.
- Streamed tool calls arrive as one complete delta per call, not as incremental argument
  fragments.
- Thinking blocks carry an empty `signature`. `redacted_thinking` in history is dropped.
- Anthropic `cache_creation_input_tokens` is always 0. `cache_read_input_tokens` and OpenAI
  `cached_tokens` report the engine's prefix-cache hit.
- The input is text only: no images, audio or documents.
- There is no response prefill (a final assistant message).
- `completion_tokens` / `output_tokens` include the EOS token and every reasoning token.
- The non-standard `warnings` field is added when there is something to report.
- `/v1/completions` parses special tokens in the prompt by default (see above).
- Rate limiting is by queue depth, not per client or per key.

## Known gaps (v0.2)

- **Rendering has no separate pool (A-5, partial).** Rendering and tokenizing now run after
  admission, on the admitted request's HTTP thread. The template's own step, depth and
  output limits bound them (`src/template`, review S-14 re-test), but they run on no
  dedicated pool with its own stack.
- **Structured output.** Schemas are compiled at parse time (see the OpenAI and Anthropic
  tables), but the constrained decoding itself is done by the engine (WS-G/WS-H).
- **Engine rejections.** When the engine rejects a request with a typed Api or Unsupported
  error before producing a token, a non-streaming request gets a 400 in the API's format
  (`invalid_request_error`, or `unsupported_parameter` for OpenAI). A streaming request has
  already sent `200` and its first event, so it receives the same typed error object as an
  error event. Schema errors never reach this path, because they are caught at parse time.
- **Cancellation before the first token.** See "Cancellation" (review R-3).
- **Transport limits are fixed.** Header count and size are httplib's compile-time defaults:
  100 headers, 8 KiB per header line, 8 KiB request URI.
- **No TLS.** Terminate TLS in a reverse proxy (PRD §12, "TLS via deployment layer").
