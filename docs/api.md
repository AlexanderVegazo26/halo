# HALO HTTP API (v0.2)

HALO serves one model over HTTP/1.1. It speaks the OpenAI Chat Completions and Completions
formats and the Anthropic Messages format, and it has a few utility routes. The server is
`halo::api::ApiServer` (`include/halo/api/server.h`), built on cpp-httplib. The `halo serve`
CLI command that will start it is not written yet (planned for the next WS-I milestone).

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

These implement PRD §12 and the security review of 2026-09-24 (`docs/reviews/`).

- **Bind address.** The default is `127.0.0.1`. Binding a non-loopback address without an
  API key fails with `CONFIG_ERROR`, unless `allow_unauthenticated_remote` is set; the server
  then logs a warning.
- **API key.** When `api_key` is set, every route except `GET /health` and CORS preflight
  requires `Authorization: Bearer <key>` or `x-api-key: <key>`. The `bearer` scheme is
  matched case-insensitively. The comparison runs in constant time: its duration depends only
  on the lengths. A missing or wrong key gets 401 plus `WWW-Authenticate: Bearer`.
- **Host header (DNS rebinding).** On a loopback bind, only `localhost`, `127.0.0.1`,
  `[::1]`, the bind address and `allowed_hosts` are accepted as the Host (the port is
  ignored). Any other Host gets 403. On a non-loopback bind with an empty `allowed_hosts`,
  the Host is not checked, and the API key is then mandatory (see above).
- **Origin (CSRF).** A request whose `Origin` header is not in `cors_origins` gets 403 on
  every route, before any handler runs. `cors_origins` holds exact origins or `"*"`. It is
  empty by default, which means no browser origin is allowed. For an allowed origin the
  server returns `Access-Control-Allow-Origin: <origin>` and `Vary: Origin`.
- **Content type.** A POST body must be `application/json`, otherwise 415. This blocks
  HTML-form CSRF, which sends `text/plain` or `application/x-www-form-urlencoded` without a
  preflight.
- **Compressed bodies.** The server is built without decompression, so a request body with a
  known compression coding (`gzip`, `deflate`, `br` or `zstd`) also gets 415. httplib passes
  an unrecognized `Content-Encoding` value through unchanged, and the body is then parsed as
  JSON.
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
  - `max_tool_schema_bytes` per tool schema (64 KiB);
  - `max_content_bytes`, the total bytes of every string in the conversation (4 MiB; 413);
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

## Resource governance

| Setting (`ServerConfig`) | Default | Effect |
|---|---|---|
| `max_concurrent` | 4 | Generations holding a slot at once. |
| `max_queue` | 16 | Generations waiting for a slot. When the queue is full: **429** plus `Retry-After: 1`. |
| `queue_timeout` | 30 s | A queued request that waits longer gets **503** plus `Retry-After: 1`. |
| `request_timeout` | 600 s | Wall-clock limit for one generation. It ends like `max_tokens`, with a warning. |
| `default_max_tokens` | 8192 | Used when a request gives none. It is clamped to the context. |
| `max_tokens_cap` | 32768 | A request asking for more gets 400. |
| `reasoning_output_reserve` | 512 | Output headroom kept free of reasoning. See "Reasoning". |
| `max_output_nesting` | 256 | Unmatched `[`/`{` allowed in model output (review S-9). |
| `http_threads` | 0 (`max_concurrent + max_queue + 4`) | Size of the HTTP worker pool. |
| `read_timeout` / `write_timeout` / `keep_alive_timeout` | 60 s / 60 s / 5 s | Socket timeouts. At most 100 requests per keep-alive connection. |

The admission cap applies only to generation routes. `/tokenize`, `/apply-template`,
`/metrics`, `/health` and `/v1/models` are not queued. They run on the HTTP pool, which
bounds them. The prompt is rendered and tokenized **before** admission, on the HTTP worker
thread (see "Known gaps").

**Cancellation.**
- **Client disconnect.** A streaming client that disconnects is detected on the next write
  that fails. The token callback then returns false and the engine stops that request. A
  non-streaming client that disconnects is detected by polling the socket once per token.
- **Shutdown.** `ApiServer::stop()` cancels in-flight generations. Queued requests get 503.

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
| `response_format` | `text`; `json_object` sets `SamplingParams::json_object`; `json_schema` sets `SamplingParams::json_schema` to the serialized `json_schema.schema`. Enforcement (constrained decoding) belongs to the engine. |
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
prompt. This is a deliberate exception to the chat routes' injection protection; set it to
false to treat the prompt as plain text.

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

- **Rendering is not isolated (A-5, partial).** Rendering and tokenizing run on the HTTP
  worker thread before admission. They are bounded by the body and content caps but have no
  separate stack-size or time budget. A hostile *model* template (review S-3) can still
  crash or hang a worker. The fix belongs in `src/template`.
- **Structured output is not enforced here.** `json_schema` / `json_object` are passed
  through in `SamplingParams`. The engine (WS-G/WS-H) is responsible for constrained
  decoding.
- **Transport limits are fixed.** Header count and size are httplib's compile-time defaults:
  100 headers, 8 KiB per header line, 8 KiB request URI.
- **No TLS.** Terminate TLS in a reverse proxy (PRD §12, "TLS via deployment layer").
