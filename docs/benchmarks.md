# HALO benchmarks: harness, records, workloads

Status: WS-J Milestone 1 (the `halo_profiling` library) and Milestone 3 (the autotuner
and profile database, `halo_autotune`; see the last section). Baseline adapters (TRD §51,
Milestone 2) are not described here yet.

> **D-001.** A number produced on the development host (WSL, CPU reference kernels, or a
> fake engine in unit tests) exercises the harness only. It is never a HALO performance
> claim. Every record carries `host_label`. Only records captured on the target unit
> (GMKtec EVO-X2, gfx1151) with a pinned power mode and a valid thermal verdict can back
> a claim.

## Library entry points

Header: `include/halo/profiling/suite.h`. The CLI (`halo bench`, owned by WS-I) wires these
entry points; the profiling library links neither the runtime nor the tokenizer.

| Entry point | What it runs |
|---|---|
| `run_micro_suite(MicroSuiteConfig, Clock)` | Per-kernel callbacks, ≥ 5 warm-up + ≥ 20 measured iterations |
| `run_model_suite(Engine&, ModelSuiteConfig, Clock)` | The PR-004 / PRD §18 matrix over `runtime::Engine` |
| `run_system_suite(Engine&, SystemSuiteConfig, Clock)` | N concurrent agents replaying a workload over `runtime::Engine` |
| `run_suite(SuiteRequest, Clock)` | Dispatches to one of the three (the `halo bench` entry point) |

Every suite returns a `SuiteArtifact` (JSON schema `halo.bench.artifact/1`) with:

- `records`: one `halo.bench.record/1` per run, warm-up (`phase: "cold"`) and measured
  (`phase: "steady"`).
- `summaries`: `halo.bench.summary/1` roll-ups per configuration group.
- `hardware_before` / `hardware_after`: the hardware-state snapshots.
- `thermal`, `power_mode_pinned`, `conformant`, `valid`, and `notes` (skips and
  `FAILED:` entries).

### Intended `halo bench` flags (for WS-I to wire)

```
halo bench micro  [--warmup N=5] [--iterations N=20] [--filter REGEX]
halo bench model  --model PATH [--mtp PATH] [--backend cpu|vulkan|hip]
                  [--modes load_time,prompt,decode,mtp_decode,concurrency,prefix_replay,cancellation_storm]
                  [--contexts 4096,32768,131072,262144] [--concurrency 1,2,4,8]
                  [--prompt-tokens 512] [--decode-tokens 128] [--repetitions 3] [--warmup 1]
                  [--bandwidth-gbps X --w-trunk-gb X [--w-mtp-gb X --w-head-gb X --draft N]]
system flags:     halo bench system --model PATH --workload bench/workloads/concurrency_4agents.json
                  [--agents 4] [--repetitions 3]
common:           --host-label LABEL (required) --power-mode LABEL (EVO-X2 BIOS/EC mode)
                  --out FILE.json [--thermal-threshold-c 5] [--allow-nonconformant]
```

`--power-mode` supplies the platform performance (cTDP) mode, which sysfs does not expose.
Without it the power mode is `unknown|…` and the artifact is invalid for comparison.

## Methodology (TRD §50, §49)

1. The before snapshot is taken when the suite starts (hardware discovery, `captured_at`
   in UTC).
2. Warm-up runs are recorded as `cold`. They form their own summary group and are counted
   as `warmup_count` of the steady group. They never mix into steady statistics.
3. Measured repetitions are recorded as `steady`.
4. The after snapshot is taken when the suite ends.
5. The verdicts are computed:
   - **Thermal:** the run is invalid when |T_after − T_before| > 5 °C (the policy default,
     configurable), or when either temperature is missing.
   - **Power mode:** the run is invalid when the before and after modes differ, or when
     either mode has an `unknown` part. Unknown-vs-unknown is also rejected.
   - **Conformance:** every steady group must meet its minimum. Micro needs ≥ 20 measured
     and ≥ 5 warm-ups; end-to-end needs ≥ 3 repetitions. A suite configured below the
     minimum is refused with `Error(Config)` unless `allow_nonconformant` is set, which is
     for tests only. Any engine failure makes the artifact nonconformant.
6. `valid = thermal.valid && power_mode_pinned.comparable && conformant`.

Timings come from the harness's injected `Clock` (steady_clock in production). The
engine's self-reported numbers are kept under `extra.engine_*` for cross-checking only.
**The metric definitions differ per mode**; compare a metric only within one mode:

| Mode | `ttft_ms` | `decode_tps` / `decode_effective_tps` | `efficiency` |
|---|---|---|---|
| `prompt`, `decode`, `mtp_decode`, `prefix_replay` | first token callback − call start | (generated − 1) / (last callback − first callback); `mtp_decode` reports it as `decode_effective_tps` | D-011 model, per record (see below); not set for `prompt` / `prefix_replay` |
| `concurrency` | p95 over the N requests | aggregate generated tokens / wall time of the whole batch, **including prefill** (`extra.aggregate_includes_prefill`) | **null** (a prefill-inclusive rate would understate η) |
| `system` (`agents`) | p95 over all turns | aggregate generated tokens / wall time, **including prefill and think time** | null |

`prompt_tps` = prompt_tokens / ttft; it includes the first decode step (stated, not hidden).
`ttft_cached_ms` is the replay TTFT (`prefix_replay`) or the mean TTFT of turns ≥ 1 (`system`).

### Statistics conventions (`stats.h`)

| Statistic | Convention |
|---|---|
| mean | arithmetic mean |
| stddev | **sample** (n − 1 divisor), 0 for n = 1 |
| median, p50, p95, p99 | **linear interpolation, Hyndman–Fan type 7**, the same as `numpy.percentile` default and Excel `PERCENTILE.INC` |
| cv | stddev / \|mean\|; +inf when mean = 0 and stddev > 0, serialized as `null` |

For even n, the median is the mean of the two middle values. Note that
`halo::hardware::Stats::p95` uses nearest-rank; the two p95 values are not comparable.

**Instability classifier.** `classify_stability` reports:

- `TooFewSamples` when n < the minimum.
- `Unstable` when cv > 0.05. This threshold is a HALO policy default, not a measured
  property of any machine.
- `Stable` otherwise.

The autotuner reuses the same classifier.

## The benchmark record (`record.h`, schema `halo.bench.record/1`)

**TRD §28 fields**, all present under their exact TRD names:

- model, model_hash, pack, backend, driver, gpu, quantization, lm_head
- context, batch, concurrency
- ttft_ms, ttft_cached_ms, prompt_tps, decode_tps, decode_effective_tps
- mtp_acceptance, prefix_cache_hit_rate, memory_by_tier_gb, load_time_s
- temperature_c, clocks_mhz, power_mode

**A metric that was not measured is `null`, never 0.**

**HALO additions:**

- schema, suite, mode, engine
- pack_hash, mtp_hash
- repetition, phase, host_label
- measured_bandwidth_gbps, predicted_bytes_per_token, efficiency
- hardware_before, hardware_after
- invocation (binary, version, commit, argv, environment)
- extra (suite-specific numbers)

**Efficiency (D-011 / D-014).** `DecodeByteModel` implements

    B_step = W_trunk + (n+1)·W_mtp + n·W_head + S·[(1+K)·0.157 GB + ctx·64 KiB]
    η = measured_tps · (B_step / (S·τ)) / BW

Defaults and conventions:

- K defaults to n + 1 (D-012).
- W_trunk includes the LM head.
- GB = 1e9 bytes.
- With W_trunk = 16.48, S = 1, n = 0 and 256 GB/s, the model reproduces D-011's ceiling
  of about 15.24 tok/s and η ≈ 0.787 at 12.0 tok/s.

The model suite applies the byte model per record:

- **Every mode except `mtp_decode`:** n = 0 and W_mtp = 0.
- **`mtp_decode`:** n is the configured draft length. τ is derived from the engine's
  counters as generated / (generated − accepted_draft_tokens), which assumes each
  verification step emits accepted + 1 tokens. It is recorded as `extra.tau_observed`.
  This step semantics is an assumption about the engine; it has only been exercised
  against a fake engine, not against the real HALO engine or llama-server.

`predicted_bytes_per_token` is B_step / (S·τ).

## Reference workloads (`bench/workloads/`, schema `halo.bench.workload/1`)

| File | id | Kind | Shape |
|---|---|---|---|
| `agent_replay.json` | `agent_replay_v1` | agent_replay | system prompt + 4 tools + 6-turn coding-agent replay |
| `long_context.json` | `long_context_v1` | long_context | one agent per PRD §18 depth (4K/32K/128K/256K); synthetic document bodies |
| `concurrency_4agents.json` | `concurrency_4agents_v1` | concurrency | 4 agents sharing one system prompt, 3 turns each, think time between turns |

**Provenance.** The text in these files is original, written for HALO by WS-J; there is no
third-party or user data. Where real text is not needed, the files use
`{"synthetic_tokens": N, "seed": S}` segments. Their token ids come from SplitMix64 over
the seed, reduced into the vocabulary with a 64×64 → high-64 multiply. They are
deterministic and portable, but they are **not natural text**: use them for throughput
and memory measurements, never for quality measurements.

**The files are frozen.** Any edit changes every token count, so add a new file with a new
`id` instead. A unit test loads every file in the directory.

**System-suite replay.** Each agent replays its script as a growing conversation. Turn k's
prompt is turn k−1's prompt, then the tokens generated for it, then the tokens the encoder
adds for turn k.

**Encoders.** Tokenization needs the model's tokenizer and chat template. The CLI supplies a
`WorkloadEncoder` built from the Engine. `synthetic_encoder(vocab)` is the
tokenizer-free stand-in: a text message becomes ceil(bytes/4) seeded tokens.

**Bounds.** Workload files are untrusted input and are rejected with `Error(Config)` when
they exceed any of:

- 16 MiB of file size
- 1 Mi synthetic tokens per segment
- 1024 messages
- 64 agents
- max_tokens outside [1, 65536]
- think time over 600 s

## Performance counters (`counters.h`, TRD §34)

`PerfCounters` holds the TRD §34 set. The derived values are computed from totals, never
stored:

- `mtp_acceptance_rate`
- `effective_bandwidth_per_tier`, as bytes / window in GB/s

`to_prometheus()` emits text exposition format 0.0.4:

- The `halo_` prefix is on every metric.
- Counters end in `_total`.
- Times are in seconds.
- Per-tier series carry `tier="VRAM|GTT|PINNED|HOST"`.
- A value that cannot be derived is `NaN`.

`from_engine_stats()` fills the counters the Engine contract exposes.

## Autotuner and profile database (TRD §55–§58, §64)

Two libraries:

- **`halo_autotune`**: the profile key, the SQLite database, the runtime lookup, the cost model and the tuner.
  - It depends on core, profiling, hardware and SQLite, and **not** on any backend.
  - The runtime links this library.
- **`halo_autotune_cpu`**: TunableOps over the CPU reference kernels.
  - It links `halo_backend_cpu`.
  - It is built only when the `cpu` backend is selected.

### Profile key (TRD §57), `profile_key.h`

`make_profile_key(HardwareState, model_hash, pack_id, isa_target)` builds the key from an
M1 hardware-state snapshot. The tuner and the runtime must both use it.

| Field | Class | Source |
|---|---|---|
| HALO_VERSION | version-tolerant | CMake `PROJECT_VERSION` |
| MODEL_HASH | **must match** | caller (SHA-256 of the trunk GGUF) |
| PACK_ID | **must match** | caller (hash over trunk + MTP file hashes) |
| GPU_DEVICE | **must match** | PCI `vendor:device`, or `cpu` for a CPU-only key |
| GPU_ARCH | **must match** | KFD `gfx_target`; empty for a CPU-only key |
| DRIVER_VERSION | version-tolerant | `driver_version`, else `mesa_version`; neither is discovered today, so this field is empty |
| ROCM_VERSION | version-tolerant | `/opt/rocm/.info/version` |
| VULKAN_VERSION | version-tolerant | the RADV ICD `api_version` |
| KERNEL_VERSION | version-tolerant | `osrelease` |
| OS | version-tolerant | os-release `PRETTY_NAME` |
| POWER_MODE | **must match** | the M1 power-mode string |
| ISA_TARGET | **must match** | caller (`gfx1151`, `gfx11-generic`, or a CPU ISA label) |

- **Exact match:** every field is equal.
- **Compatible match:** only version-tolerant fields differ. The match is flagged with the
  list of differing fields (TRD §56 step 2).
- **Rejected:** any must-match field differs, including the power mode.

The power-mode rule matches M1 `check_comparable`:

- The platform (BIOS/EC) part must be known.
- The GPU parts must be known whenever the key names a GPU.
- A CPU-only key, such as the WSL dev host with no amdgpu sysfs, may carry unknown GPU parts.

### Database (TRD §58), `db.h`

Schema v1 has exactly these tables:

- `hardware_profile`
- `tier_bandwidth`
- `model_profile`
- `operator_profile`
- `kernel_candidate`
- `benchmark_run`
- `winning_configuration`
- `schema_version`

The file header carries `application_id` 0x48414C4F ("HALO").

- **Writes:** values are only ever bound to prepared statements. Writes run in BEGIN
  IMMEDIATE transactions with RAII rollback.
- **Connection settings:** WAL journal, a 5 s busy timeout, and foreign keys on.
- **Open checks:** `PRAGMA quick_check` runs on every open.
- **Creation:** a read-write open of a missing or empty file creates v1.
- **Forward migration:** `OpenOptions::migrations` is the hook. Each step runs in one
  transaction together with its version bump. A migration that throws leaves the old
  version in place.

Every failure throws `ProfileDbError`, which derives from `halo::Error`. Its `kind()` is one of:

| Kind | When |
|---|---|
| `corrupt` | the file is not SQLite, or is damaged |
| `not_halo_db` | the file is valid SQLite but not a HALO database, or is empty on a read-only open |
| `foreign_version` | the schema is newer than this build, or older on a read-only open |
| `busy` | the database stays locked past the busy timeout |
| `io` | a read-only open of a missing file, or a write through a read-only handle |
| `constraint` | a constraint rejected the write |
| `internal` | any other SQLite error |

`winning_configuration` keeps one winner per (hardware profile, model profile, operator,
backend), so tuning Vulkan and HIP for the same operator keeps both winners.

`benchmark_run` holds two kinds of rows:

- tuning measurements (`kind = 'tune'`)
- M2 baseline and bench records (`kind = 'baseline' | 'bench'`), stored with their full
  `halo.bench.record/1` JSON

`latest_run(kind, backend, pack, context, mode)` answers "latest baseline for (backend,
pack, context, mode)". Timestamps are UTC ISO-8601 at second resolution; ties go to the
highest row id.

Concurrency:

- A `ProfileDb` is not thread-safe; use one per thread.
- Several processes may share the file, relying on SQLite locking and the busy timeout.
- A read-only open needs a writable directory, because SQLite creates a `-shm` file.

### Kernel selection (TRD §56) and "never per-request"

`ProfileLookup::open(path)` is the runtime entry point.

- It opens the database read-only and copies every winning configuration into memory.
- After that no SQL runs. `find()` and `select_kernel()` are const and thread-safe.
- Nothing in `lookup.h` accepts a TunableOp, a Clock or a writable database, so the runtime
  cannot measure or persist.

Every answer says which §56 step produced it:

| Answer | Step |
|---|---|
| `exact` | step 1 |
| `compatible` | step 2, with the differing fields |
| `heuristic` | step 3: cost model only, nothing measured |
| `none` | no answer; run `halo tune` |

A stored winner that is not among the current candidates is skipped with a note (for
example, a winner that uses more threads than are now available).

Steps 4 (microbenchmark) and 5 (persist) exist only in `autotune::tune()`.

### Cost model (TRD §55), `cost_model.h`

`T = max(T_compute, T_memory) + T_sync + T_launch`, where:

- **`T_memory`** = bytes / tier bandwidth. The bandwidth is the **median** read bandwidth
  from the measured `TierBandwidth`, scaled by min(1, parallelism / measured threads).
- **`T_compute`** = flops / (GFLOP/s per thread × parallelism). The per-thread rate comes
  from `calibrate_gflops_per_thread` on a HALO kernel measurement; there is no built-in
  constant.

The `memory_bound` flag is the model's own estimate. The CPU path has no occupancy or stall
counters to verify it, which TRD §55 would require.

### Tuner (TRD §58), `tuner.h`

A TunableOp provides:

- `key`
- `backend`
- `candidates`
- `run`, which may return a device-measured time
- optionally `validate` (a correctness gate) and `cost` (inputs for the cost model)

| Strategy | What it measures |
|---|---|
| EXHAUSTIVE | every candidate |
| RANDOM | a seeded draw without replacement (SplitMix64 partial Fisher–Yates, not `std::shuffle`) |
| GRID | per dimension, every `grid_stride`-th sorted value plus the last |
| HEURISTIC (default) | the `heuristic_keep` candidates the cost model predicts best |
| BAYESIAN | nothing: returns `Error(Unsupported)` |

The default strategy is HEURISTIC, which needs a `CostModel` (measured tier bandwidth plus a
calibrated GFLOP/s per thread); without one `tune()` fails with `Error(Config)`. A caller with
no calibration should pass `--strategy exhaustive`.

How a winner is chosen:

- Each candidate gets the TRD §50 methodology: 5 warm-up runs plus 20 measured. Fewer runs
  need `allow_nonconformant`.
- Unstable candidates (M1 classifier, default cv ≤ 0.05) are rejected.
- Invalid candidates (failed `validate`) are rejected and never timed.
- The winner has the smallest **median** in ns. Ties go to the smallest canonical candidate
  string (`name=value;…`).
- If no candidate is eligible, the runs are still recorded and the existing winner stays in
  place.

The CPU TunableOps:

- **`CpuMatmulTunable`** tunes the thread count. `validate` requires output bit-identical to
  the 1-thread result.
- **`CpuGdnChunkedTunable`** tunes chunk size × thread count. `validate` requires output
  within a tolerance of the recurrent form.

### Intended `halo tune` flags (for WS-I to wire)

```
halo tune --model PATH [--mtp PATH] --power-mode LABEL --isa-target gfx1151|gfx11-generic|<cpu-isa>
          [--db PATH (default ~/.cache/halo/profiles.db)] [--backend cpu|vulkan|hip]
          [--strategy heuristic|exhaustive|grid|random] [--seed N] [--random-budget N]
          [--grid-stride N] [--heuristic-keep N] [--warmup 5] [--iterations 20]
          [--max-cv 0.05] [--ops MATMUL,GATED_DELTANET,...] [--report FILE.json]
```

The CLI flow (TRD §64):

1. Call `capture_hardware_state`.
2. Measure the tier bandwidths and store each with `record_tier_bandwidth`.
3. Inspect the model and compute `MODEL_HASH` and `PACK_ID`.
4. Build the key with `make_profile_key`.
5. Build TunableOps for the model's operator shapes.
6. Call `tune()` and write the `TuneReport` JSON.

At start-up the runtime calls `ProfileLookup::open`, then `select_kernel` for each operator.
