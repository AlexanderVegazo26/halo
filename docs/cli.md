# The `halo` command-line tool (v0.2)

`halo` is a single binary with subcommands (PRD §14, TRD §26). Its logic lives in the
`halo_cli` library under `tools/halo/`, and `tests/unit/cli/` drives that library
in-process. `halo <command> --help` prints the options of any command; hidden test-only
options are not listed.

| Command | What it does | Needs a model file | Needs the runtime |
|---|---|---|---|
| `inspect` | FR-003 compatibility report and a memory estimate | yes; a header-only GGUF is enough | no |
| `devices` | hardware discovery; `--verify` runs the deployment checklist | no | no |
| `tokenize` | tokenizes text with the model's tokenizer | yes; header-only is enough | no |
| `template` | renders a chat with the model's chat template | yes; header-only is enough | no |
| `run` | one generation, streamed to stdout | yes | **yes** |
| `serve` | OpenAI / Anthropic HTTP API (`docs/api.md`) | yes | **yes** (and `halo_api`) |
| `bench micro` | CPU reference kernel micro-benchmarks | no | no (`halo_profiling` + CPU backend) |
| `bench model`, `bench system` | model and agent suites (`docs/benchmarks.md`) | yes | **yes** |
| `tune` | autotunes kernels into the profile database; `--list` shows the lookup | yes; header-only is enough | yes, for the profile-key functions (and `halo_autotune`) |
| `version` | prints the version and the optional components in this build | no | no |

"The runtime" means `halo_runtime`, the inference engine (`runtime::create_engine`). The
CLI detects each optional library at configure time with `if(TARGET ...)`. Those
libraries are `halo_runtime`, `halo_api`, `halo_profiling`, `halo_backend_cpu`,
`halo_autotune` and `halo_autotune_cpu`. A command whose library is missing prints why and
exits 2. For example, `run` in a build without the runtime prints:

```
halo: runtime not built: this halo binary does not include halo_runtime (the inference engine), so it cannot load a model for generation
```

`halo version` lists which of these libraries a build contains.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | the command ran and failed: a typed error such as `MODEL_ERROR` or `CONFIG_ERROR`, a failed `devices --verify` check, a `bench` artifact with a `FAILED:` note, a `tune` that found no winner for any operator, or a generation that ended with an engine error |
| 2 | usage error (unknown command or option, missing or invalid value), or a feature or library that is not part of this build |
| 128 + n | ended by signal n outside `serve` (SIGINT gives 130) |

Errors are printed to stderr as `halo <command>: <message>`. Results go to stdout.

## Configuration (TRD §44)

The commands that load a model (`run`, `serve`, `bench model`, `bench system`) resolve
their settings in this order:

1. command-line flags;
2. `HALO_*` environment variables;
3. a JSON config file given by `--config FILE` or `HALO_CONFIG`;
4. built-in defaults.

`halo serve --print-config` prints every resolved value together with its source (`cli`,
`env:HALO_X`, `file:<path>` or `default`). The API key is shown as `<redacted>`.

| Setting (file key) | Flag | Environment | Default |
|---|---|---|---|
| `model.path` | `--model` or the positional argument | `HALO_MODEL` | none (required) |
| `model.mtp` | `--mtp` | `HALO_MTP` | none |
| `runtime.backend` | `--backend` | `HALO_BACKEND` | `auto` |
| `runtime.threads` | `--threads` | `HALO_THREADS` | 0 (auto) |
| `runtime.ctx` | `--ctx` | `HALO_CTX` | 32768 |
| `runtime.parallel` | `--parallel` | `HALO_PARALLEL` | 4 |
| `runtime.mtp_draft` | `--mtp-draft` (0 turns MTP off) | `HALO_MTP_DRAFT` | 2 |
| `runtime.prefix_cache` | `--no-prefix-cache` | `HALO_PREFIX_CACHE` | true |
| `runtime.power_mode` | `--power-mode` | `HALO_POWER_MODE` | none; set, the engine applies tuned profiles (TRD §64) |
| `runtime.profile_db` | `--profile-db` | `HALO_PROFILE_DB` | with a power mode: `halo tune`'s default DB |
| `runtime.isa_target` | `--isa-target` | `HALO_ISA_TARGET` | the CPU ISA of this process (must match `halo tune`) |
| `server.host` | `--host` | `HALO_HOST` | `127.0.0.1` |
| `server.port` | `--port` | `HALO_PORT` | 8080 |
| `server.api_key` | `--api-key` | `HALO_API_KEY` | none |
| `server.cors_origins` | `--cors-origins a,b` | `HALO_CORS_ORIGINS` | none |
| `server.allowed_hosts` | `--allowed-hosts` | `HALO_ALLOWED_HOSTS` | none |
| `server.served_model_name` | `--served-model-name` | `HALO_SERVED_MODEL_NAME` | the engine's model id |
| `server.max_concurrent` | `--max-concurrent` | `HALO_MAX_CONCURRENT` | `runtime.parallel` (`serve` caps it there) |
| `server.max_queue` | `--max-queue` | `HALO_MAX_QUEUE` | 16 |
| `server.max_body_bytes` | `--max-body-bytes` | `HALO_MAX_BODY_BYTES` | 8 MiB |
| `server.max_tokens_cap` | `--max-tokens-cap` | `HALO_MAX_TOKENS_CAP` | 32768 |
| `server.default_max_tokens` | `--default-max-tokens` | `HALO_DEFAULT_MAX_TOKENS` | 8192 |
| `server.request_timeout_s` | `--request-timeout` | `HALO_REQUEST_TIMEOUT` | 600 |
| `server.allow_remote` | `--allow-remote` | `HALO_ALLOW_REMOTE` | false (D-017: loopback-only unless set; a reverse proxy is required) |
| `server.allow_unauthenticated_remote` | `--allow-unauthenticated-remote` | `HALO_ALLOW_UNAUTHENTICATED_REMOTE` | false (needs `allowed_hosts`) |
| `server.utility_concurrency` / `server.utility_queue` | `--utility-concurrency` / `--utility-queue` | `HALO_UTILITY_CONCURRENCY` / `HALO_UTILITY_QUEUE` | 2 / 8 |
| `server.header_timeout_s` / `server.body_timeout_s` | `--header-timeout` / `--body-timeout` | `HALO_HEADER_TIMEOUT` / `HALO_BODY_TIMEOUT` | 10 / 60 |
| `server.completions_parse_special` | `--no-completions-parse-special` | `HALO_COMPLETIONS_PARSE_SPECIAL` | true |

Example config file:

```json
{"model":   {"path": "/models/qwen3.8-27b-ud-q4_k_xl.gguf"},
 "runtime": {"backend": "vulkan", "ctx": 65536, "parallel": 4},
 "server":  {"port": 8080, "cors_origins": ["http://localhost:3000"]}}
```

- **Strict parsing.** An unknown section or key, a wrongly typed value, or a malformed
  number in the environment is a `CONFIG_ERROR` (exit 1). A malformed value on the command
  line is a usage error (exit 2). A typo never falls back to a default.
- **Formats.** The file must be JSON of at most 1 MiB. **YAML is not supported in v0.2.**
- **Booleans.** Environment booleans accept `1/0/true/false/yes/no/on/off`. Boolean flags
  state the non-default value, for example `--no-prefix-cache`.
- **Empty environment values count as unset** (review S-18). `HALO_API_KEY=` therefore
  cannot silently replace a key from the config file. An empty `--api-key` or an empty
  `api_key` in the file is an error. `--print-config` shows a secret as `<redacted>`, or as
  `<empty>` when its value is empty.
- **Loopback-only serving (D-017).** A non-loopback `--host` needs `--allow-remote`, and
  such a deployment requires a reverse proxy that enforces connection limits and
  header/body timeouts (`docs/api.md`, "Security defaults").
- **Safe values only.** `serve` refuses, before loading the model:
  - `--cors-origins '*'` without an API key (review S-19);
  - `--allow-unauthenticated-remote` without `--allowed-hosts` (review S-20).
- **API key.** Prefer `HALO_API_KEY` to `--api-key`: other local users can read a process's
  command line.

## Commands

### `halo inspect <model.gguf> [--mtp F] [--json] [--ctx N] [--parallel N] [--mtp-draft N]`

- **Report.** Opens the file header-only; tensor data is never mapped. It prints the hybrid
  layout, hyperparameters, weight bytes by class and dtype, the state size per token and
  per sequence, the MTP source (`embedded`, `separate_file` or `none`), the tokenizer
  summary, and the loader's warnings.
- **Memory estimate.** Runs the memory planner (`memory::plan_memory`) against the tiers
  that hardware discovery finds. It prints the placement table and the largest context
  that fits, for `--ctx` (default `min(32768, trained context)`) × `--parallel` (default 1),
  with the MTP draft depth.
- **Refusals.** A plan that does not fit is printed as refused. The command still exits 0,
  because that is an answer, not a failure.
- **`--json`** prints `{"report": …, "memory": {request, topology, plan,
  max_context_for_budget, notes}}`.
- **Errors.** An unreadable or malformed file is `MODEL_ERROR` or `IO_ERROR` (exit 1).

### `halo devices [--verify] [--json]`

- **Output.** Prints the CPU, host memory, the AMD GPUs (gfx target, CUs, VRAM, GTT), the
  memory topology (D-002), OS and kernel, ROCm, the Vulkan driver manifests, the IOMMU
  state, and warnings.
- **`--verify`** runs the TRD §59 deployment checklist and prints `ok`, `warn`, `fail` or
  `unknown` for each check. It exits **1 if any check fails**.
- **`--json`** prints `{"hardware": …, "checklist": […], "overall": …}`.

### `halo tokenize <model.gguf> [--text T | --file F] [--parse-special] [--pieces] [--json]`

- **Input.** With neither `--text` nor `--file`, the text is read from stdin. `--file -`
  also means stdin.
- **Output.** Space-separated ids by default. `--pieces` prints one `id<TAB>"piece"` line
  per token. `--json` prints `{"tokens": […], "count": n}`.
- **Special tokens.** By default special-token text stays text. `--parse-special` turns it
  into control tokens.

### `halo template <model.gguf> --messages F [--tools F] [--no-generation-prompt] [--no-think] [--reasoning-effort E] [--tokenize] [--json]`

- **Input.** `--messages` names a JSON file (`-` means stdin). It holds a messages array,
  or an object `{"messages": […], "tools": […]}`. `--tools` names a file with an
  OpenAI-style tools array.
- **Output.** Prints the rendered prompt. `--tokenize` also prints the prompt tokens.
- **Same construction as the server.** The prompt is built exactly as the API's chat
  routes build it, so client text such as a literal `<|im_end|>` stays text
  (`docs/api.md`, "Special-token handling"). This holds when `halo_api` is part of the
  build; otherwise the prompt is tokenized as one string with special tokens parsed.

### `halo run <model.gguf> -p PROMPT [options]`

- **Output.** Generates one reply and streams its content to stdout. Reasoning is hidden
  unless `--show-reasoning` is given, in which case it goes to stderr. Tool calls are
  printed as `[tool call] name {args}`.
- **Options:**
  - `-s/--system` sets the system message.
  - `-n/--max-tokens` sets the generation length (default 512).
  - `--temperature` (default 0.7; 0 is greedy), `--top-p`, `--top-k` and `--seed` control
    sampling.
  - `--no-think` turns thinking off.
  - `--raw` skips the chat template: the prompt is tokenized as written, with special
    tokens parsed.
  - The runtime settings in the configuration table above also apply.
- **Summary.** A one-line summary goes to stderr: tokens, time, decode tok/s, and the
  finish reason.
- **Runtime.** `run` needs the runtime.

### `halo serve [--model M] [options]`

- **What it does.** Starts the HTTP API; see `docs/api.md`, "Running the server".
- **Early refusal.** These are refused before the model is loaded:
  - a non-loopback `--host` without `--allow-remote` (D-017);
  - a non-loopback `--host` without an API key, unless `--allow-unauthenticated-remote` and
    `--allowed-hosts` are both given.
- **Concurrency cap.** `--max-concurrent` is capped at `--parallel`, and the cap is printed
  on stderr. Requests beyond the engine's sequences then wait in the API queue, where
  `queue_timeout` and shutdown apply (`docs/api.md`, "Cancellation").
- **Signals.** SIGINT or SIGTERM stops the server gracefully. SIGPIPE is ignored.
- **`--print-config`** prints the resolved configuration and exits. It does not need the
  runtime.
- **Runtime.** `serve` needs the runtime and `halo_api`.

### `halo bench micro|model|system [options]`

These commands run the `halo_profiling` suites (`docs/benchmarks.md`). Every suite
requires `--host-label`, because D-001 says every record must record where it was
measured. The artifact JSON goes to `--out` (`-` means stdout, the default), and a
one-line verdict goes to stderr.

| Suite | Runs | Suite-specific options |
|---|---|---|
| `micro` | RMS_NORM, SWIGLU, SOFTMAX and GATED_DELTANET decode on the CPU reference kernels at Qwen3.8-27B shapes | `--iterations`, `--filter REGEX`, `--threads`, `--list` |
| `model` | the PRD §18 / PR-004 matrix over the engine | `--modes`, `--contexts`, `--concurrency`, `--concurrency-context` (default 4096), `--prompt-tokens`, `--decode-tokens`, and the D-011 efficiency inputs (`--bandwidth-gbps`, `--w-trunk-gb`, `--w-mtp-gb`, `--w-head-gb`, `--draft`) |
| `system` | N agents replaying a frozen workload | `--workload FILE` (required), `--agents` |

- **Common options.** `--power-mode`, `--thermal-threshold-c`, `--warmup`, `--repetitions`
  and `--allow-nonconformant`. The last is for smoke runs only: a suite configured below the
  TRD §50 minimum is otherwise a `CONFIG_ERROR`.
- **Runtime.** `model` and `system` need the runtime; `micro` does not.
- **MTP and "decode" (F-3).** One engine serves a whole `bench model` run. When `--modes`
  does not include `mtp_decode`, MTP drafting is turned off unless `--mtp-draft`,
  `HALO_MTP_DRAFT` or the config file sets it, and stderr says so. That keeps plain-decode
  numbers from being MTP numbers. When a run includes both `decode` and `mtp_decode`, its
  `decode` records are taken with MTP enabled: run them separately for plain decode.
- **Tuned profiles.** With `--power-mode` the engine applies tuned profiles
  (`--profile-db`), exactly as `serve` and `run` do.
- **Output file.** `--out` never follows a symlink. The artifact is written to a temporary
  file and renamed into place, and a failed write is an error (review S-21).

### `halo tune --model M --host-label L --power-mode P [options]`

Autotunes kernels into the SQLite profile database (TRD §58, §64). The steps are:

1. Build the TRD §57 profile key with `runtime::engine_profile_key`:
   - the hardware state;
   - the SHA-256 of the trunk, and `PACK_ID` over the trunk and MTP hashes;
   - the power mode and the ISA.
2. Build the operator keys with `runtime::matmul_op_key` / `gdn_op_key`.
3. Measure host memory bandwidth and record it (skipped with `--no-bandwidth`).
4. Create the TunableOps at exactly those keys, and check that their keys match.
5. Tune, and persist the winners.

**These are the functions the Engine itself uses** to look winners up
(`include/halo/runtime/profile.h`), so a tuned winner is found at TRD §56 step 1 (exact) on
the engine's next launch. A test checks exactly that. For the same reason `tune` needs
`halo_runtime`, and `--gdn-tokens` accepts only 512, the prefill length the engine looks the
GDN tunable up at.

**Operators and database:**
- **Operators.** v0.2 tunes the CPU reference TunableOps from `halo_autotune_cpu`:
  - `MATMUL` (FFN projection `1 × n_embd → n_ff`, over thread counts);
  - `GATED_DELTANET` (the chunked form, over chunk size × thread count).
  - `--backend` other than `cpu` is a usage error: the GPU TunableOps are not wired yet.
- **Database.** `--db`, else `HALO_PROFILE_DB`, else `$XDG_CACHE_HOME/halo/profiles.db`,
  else `~/.cache/halo/profiles.db`. Missing directories are created.
  - This is also the database `serve`, `run` and `bench` apply when given `--power-mode`.
  - A symlinked `--db` or `--report` path is refused (review S-21).
  - `--report` is written through a temporary file, and a failed write is exit 1, for
    `--list` too.
  - If the model file changes between hashing and reading its header, `tune` fails rather
    than mixing two files.
- **Required flags.** `--model`, `--host-label` and `--power-mode`.
  - The power mode is a must-match field of the key. A profile tuned under another mode is
    never selected.
  - `--isa-target` defaults to this process's CPU ISA (`x86-64-avx512`, `x86-64-avx2` or
    `x86-64`). `docs/benchmarks.md` lists it as a required flag; the CLI supplies this
    default instead. A non-default value must also be given to `serve`/`run`
    (`--isa-target`), or the engine's key differs.

**Strategy:**
- **Default.** Without `--strategy` the tuner is **exhaustive**, because no cost-model
  calibration is available. The output says so:
  `strategy: exhaustive (no cost-model calibration available; …)`.
- **Heuristic.** With `--gflops-per-thread X`, a per-thread GFLOP/s calibration measured
  with a HALO kernel, the default becomes `heuristic`. The measured host bandwidth and X
  then form the cost model.
- **Others.** `--strategy grid|random` are available. `bayesian` is not implemented.
- **Candidate lists.** `--ops`, `--threads` and `--chunks` choose what is tuned. The
  defaults are powers of two up to the core count, and chunks 16, 32, 64 and 128. A value
  the engine cannot run is refused, because a winner outside the engine's candidate list is
  never applied.
- **Measurement.** `--warmup` (default 5), `--iterations` (default 20) and `--max-cv`
  (default 0.05) are the TRD §50 methodology and the stability gate.

**Output and exit code:**
- `--report FILE` writes the `halo.tune.report/1` JSON.
- The exit code is 0 when at least one operator got a winner, and 1 otherwise.

**`halo tune --list` (or `--show`):**
- **What it does.** Opens the database **read-only** (a missing database is an error; none
  is created). It builds the engine's key, op keys and candidate lists, with no measurement
  buffers, and prints what the runtime lookup (TRD §56, `autotune::select_kernel`) chooses
  for each operator.
- **Per operator it prints:**
  - the lookup step: `exact`, `compatible` (with the fields that differ) or `none`;
  - the candidate and the stored median;
  - the reasons stored profiles were rejected (for example a `POWER_MODE` mismatch).
- **`--report FILE`** writes the same information as JSON.

## Signals

- **SIGPIPE** is ignored. A reader going away surfaces as a write error instead of killing
  the process.
- **SIGINT and SIGTERM** are handled by one thread:
  - while `serve` runs, they stop the server gracefully: in-flight generations are
    cancelled and queued requests get 503;
  - at any other time the process exits with 128 + the signal number.

## Known limits (v0.2)

- **Commands not implemented.** `halo profile` (PRD §14) is not implemented. TRD §26's
  separate `halo-server`, `halo-bench`, `halo-tune` and `halo-inspect` executables are
  subcommands of the one `halo` binary.
- **YAML configuration** is not supported.
- **Runtime and tuning:**
  - `run`, `serve` and `bench model|system` run on the real engine; the end-to-end tests are
    in `tests/unit/integration` (WS-G).
  - `tune` tunes the CPU reference kernels only.
- **Duplicated code:** `tokenize` and `template` convert the GGUF vocabulary with a local
  copy of `models::vocab_spec`, so they do not link the model forward. The two must be kept
  in step.
- **Prefill cancellation.** Cancellation before a request's first token waits for the
  engine's cancel field (R-3 / S-15, `docs/api.md` "Cancellation").
