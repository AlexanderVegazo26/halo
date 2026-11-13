# HALO

HALO is a C++23 inference runtime built for one model on one machine: **Qwen3.8-27B**, a
hybrid 48×Gated-DeltaNet + 16×GQA-attention model, running on the **GMKtec EVO-X2** (AMD
Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GiB unified LPDDR5x). It is not a general
multi-model server — every design choice, from the memory planner's tier discovery to the
kernel set, is made against that single machine and that single architecture.

The engine speaks an agent-facing HTTP API (OpenAI Chat Completions/Completions and
Anthropic Messages shapes), and its own CLI for local inspection, generation, benchmarking
and autotuning.

## Download

Prebuilt Linux x86-64 binaries are attached to each release on the
[Releases page](https://github.com/AlexanderVegazo26/halo/releases). Each release has two
archives, because the Vulkan build will not start on a machine without the Vulkan loader:

| File | Backends | Shared libraries it needs |
|---|---|---|
| `halo-<version>-linux-x86_64-vulkan.tar.gz` | CPU and Vulkan | `libvulkan.so.1` plus a Vulkan driver (Mesa RADV on the EVO-X2), `libsqlite3.so.0`, glibc, libstdc++ |
| `halo-<version>-linux-x86_64-cpu.tar.gz` | CPU only | `libsqlite3.so.0`, glibc, libstdc++ |

Both are built on Ubuntu 24.04 with clang and `-march=x86-64-v3`: they need a CPU with AVX2
and a distribution whose glibc and libstdc++ are at least as new as Ubuntu 24.04's. An archive
unpacks to `bin/halo`, `share/halo/workloads/` (the reference benchmark workloads),
`licenses/` (the notices of the third-party code compiled in) and `BUILD-INFO.txt` (the commit
and the exact shared libraries the binary loads). A `.sha256` file sits beside each archive:

```bash
sha256sum -c halo-<version>-linux-x86_64-vulkan.tar.gz.sha256
tar -xzf halo-<version>-linux-x86_64-vulkan.tar.gz
./halo-<version>-linux-x86_64-vulkan/bin/halo version
```

There is no Windows or macOS build. The sources use POSIX APIs and have no port, so on Windows
the Linux binary runs under WSL. The HIP backend is not in the release binaries because it
needs ROCm at build time; build from source for it (see "Building").

### Getting the model

The weights are not distributed with HALO. It loads a GGUF of Qwen3.8-27B that you download
yourself; the files this repository's own tooling and measurements refer to are
`Qwen3.8-27B-Q4_K_M.gguf` and `mtp-Qwen3.8-27B-Q4_0.gguf` from `ggml-org/Qwen3.8-27B-GGUF` and
`Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF` on Hugging Face (the URLs are in
`python/tools/fetch_gguf_headers.py`). No checksum for them is pinned in this repository. A
separate MTP head file is passed with `--mtp`.

### First run

```bash
halo inspect model.gguf                      # header report and memory plan; maps no weights
halo run model.gguf -p "Hello" -n 64 --ctx 4096 --parallel 1
halo serve --model model.gguf --port 8080    # loopback only, no API key by default
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

`inspect` prints the memory plan for the `--ctx` and `--parallel` you give it. The defaults
(`--ctx 32768 --parallel 4`) plan for far more memory than `--ctx 4096 --parallel 1`, so start
small. The backend defaults to `auto`, which today always means the CPU backend; pass
`--backend vulkan` to run on the GPU (see "Where the GPU work stands").

## What it does

- **Loads GGUF packs directly**, mmap'd, with a byte-level BPE tokenizer and a patched Jinja
  chat-template engine, both matched token-for-token against Hugging Face's own tokenizer
  and template output.
- **Runs the forward pass over a pluggable `Backend` interface** (ADR-001): one CPU
  reference implementation that is the semantic source of truth, plus Vulkan and HIP
  backends being brought up against it operator by operator, each checked bit-for-bit or
  within a stated tolerance against the CPU path before it is trusted.
- **Serves concurrent sequences from a single batched engine loop**: paged KV cache with
  copy-on-write prefix sharing, a rollback-slot scheme for the DeltaNet recurrent state, and
  a prefix cache that lets a returning multi-turn conversation skip re-running its prefill.
- **Speculates with the model's own MTP head** (multi-token prediction), verified against
  plain greedy decoding token-for-token, with a profit gate that turns speculation off when
  it stops paying for itself.
- **Samples and constrains generation**: the standard penalty/temperature/top-k/top-p/min-p/
  typical chain, plus JSON-Schema-constrained structured output via a compiled token grammar.
- **Tunes its own kernels**: a SQLite-backed autotuner searches kernel variants per operator
  shape and persists the winner, keyed to the exact hardware/driver/model combination that
  produced it.

## What it deliberately does not do

HALO does not try to be a general-purpose serving framework. There is no model-swapping, no
multi-tenant scheduling policy, no plugin system for third-party kernels. The HTTP API binds
to loopback by default and requires an explicit opt-in plus an API key to listen elsewhere.
The CPU backend is what the engine uses unless told otherwise: `--backend auto` (the default)
never selects a GPU backend, because there is no measured default policy yet. `--backend vulkan`
and `--backend hip` are explicit opt-ins (see "Where the GPU work stands" below).

## Layout

| Path | What lives there |
|---|---|
| `include/halo/`, `src/` | tensor/dtype handling, the GGUF model loader, tokenizer, chat template engine, hardware discovery, the memory planner, paged KV cache, DeltaNet state, the `models::Qwen35` forward graph, speculative decoding, sampling, the runtime `Engine`, the HTTP API, benchmarking/profiling, the autotuner |
| `backends/cpu`, `backends/vulkan`, `backends/hip` | the operator backends behind the `Backend` interface |
| `tools/halo` | the `halo` CLI — `inspect`, `devices`, `tokenize`, `template`, `run`, `serve`, `bench`, `tune` (`docs/cli.md`) |
| `load/` | k6 load tests of the HTTP API against a local test server (`load/README.md`) |
| `tests/unit` | GoogleTest suites, mostly differential: every backend and every quantization path is checked against the CPU reference or an independent transformers golden |
| `tests/fuzz` | libFuzzer targets + seed corpus (GGUF, API requests, model output), replayed by every build (`tests/fuzz/README.md`) |
| `scripts/` | build scripts, the EVO-X2 field-kit packager, and the pinned llama.cpp reference build |
| `docs/` | the product/technical specs, engineering decisions, architecture decision records, security hardening notes, and per-subsystem references |

## Building

Linux only. Requirements: clang with C++23 support, CMake ≥ 3.25, Ninja, SQLite3 development
headers, and network access at configure time for the pinned third-party sources. The Vulkan
backend is **on by default** (`HALO_BUILD_VULKAN=ON`) and needs the Vulkan headers and loader
plus `glslc`; configure fails without them, so either install them (on Ubuntu 24.04:
`libvulkan-dev glslc spirv-tools`) or pass `-DHALO_BUILD_VULKAN=OFF`. HIP is off by default and
needs ROCm.
Third-party sources (nlohmann_json, minja, cpp-httplib, googletest) are fetched by CMake with
pinned SHA-256 hashes — nothing is vendored, nothing is fetched unpinned.

```bash
scripts/build.sh                       # build in $HOME/halo-build (or $HALO_BUILD_DIR) + the full test suite
scripts/build.sh -DHALO_BUILD_VULKAN=OFF   # the same, without the Vulkan backend
scripts/build.sh --asan                # ASan/UBSan build + tests
HALO_BUILD_DIR=~/halo-build-tsan scripts/build.sh -DHALO_ENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug   # ThreadSanitizer
scripts/build.sh -DHALO_BUILD_HIP=ON   # also compile the HIP backend (needs ROCm)
```

`scripts/build.sh` passes no build type, so the project default applies: `RelWithDebInfo`.
The binary is `tools/halo/halo` under the build
directory; `cmake --install <build> --prefix <dir>` installs it as `<dir>/bin/halo` with the
benchmark workloads. The release archives are built by `.github/workflows/release.yml` with
`-DCMAKE_BUILD_TYPE=Release -DHALO_MARCH=x86-64-v3 -DHALO_BUILD_TESTS=OFF -DHALO_BUILD_BENCHMARKS=OFF`.

`scripts/test-agent [-- ctest args]` builds (configuring a CPU-only `build/` if there is none)
and runs the suite like CI does, then writes `.test-report/summary.md`: each failure with its
`file:line`, the GoogleTest expected-vs-actual message, and the command that reruns only that
test (plus `results.json`, `junit.xml` and the full `run.log`). CI uploads it as `test-report`.
CI also runs the suite under ASan + UBSan + LSan (`sanitizers` job, report `test-report-asan`);
locally, configure `build-asan` with `-DHALO_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug` and run
`scripts/test-agent --build-dir=build-asan`. A sanitizer failure is reported with its kind and
the first in-repo frame of each stack (the access, the allocation, the leak).
A third CI job (`tsan`, report `test-report-tsan`) runs it under ThreadSanitizer
(`-DHALO_ENABLE_TSAN=ON`, a separate build dir: TSan and ASan cannot be combined). `scripts/test-agent`
sets `TSAN_OPTIONS` to stop on the first race and apply `cmake/tsan.supp`, the reviewed false
positives (one comment each: fix a real race, don't list it).

The suite includes a k6 load smoke (`load_smoke`: `load/smoke.js` against `halo_load_server`,
the real HTTP API over a scripted fake engine, ~20 s; skipped where `k6` is not installed). The
heavier ramp/soak/spike run is opt-in, see `load/README.md`.

Fuzzing (`tests/fuzz/README.md`): libFuzzer targets for the GGUF parser, API request bodies and
model output. Every build replays their seed corpus (`fuzz_replay_*`); a fourth CI job (`fuzz`,
report `test-report-fuzz`) builds them with `-DHALO_BUILD_FUZZERS=ON -DHALO_ENABLE_ASAN=ON` and
runs a fixed-seed campaign per target (`fuzz_smoke_*`, ~2 min).

`-DHALO_ONLY="model;sampling;api"` (a semicolon list of module/backend/test-directory names)
builds only the named subset — useful when iterating on one part of the tree without paying
for a full rebuild.

## Using the CLI

```bash
halo inspect model.gguf --ctx 32768 --parallel 4   # header report + memory plan, no weights mapped
halo run model.gguf -p "explain gated delta rule"  # one-shot generation to stdout
halo serve --model model.gguf --port 8080          # HTTP API, loopback-only unless configured otherwise
halo bench model model.gguf --host-label evox2-1    # the PRD's performance matrix over the real engine
halo tune --model model.gguf --host-label evox2-1 --power-mode performance   # search kernel variants, persist the winners
halo version                                       # version and which components this build has
```

Full command reference, every flag, and the environment-variable/config-file precedence
rules: `docs/cli.md`. Full HTTP surface, request/response shapes, and the security model
behind loopback-by-default: `docs/api.md`.

## Where the GPU work stands

The engine runs the CPU reference backend unless a GPU backend is named: `--backend auto`
never selects one, because there is no measured default policy yet (`src/runtime/engine.cpp`,
`make_gpu_backend`). ADR-001 stages GPU enablement so each backend's kernels are proven
against the CPU path before they are trusted:

- the CPU backend is the complete, tested reference implementation every other backend is
  checked against;
- the Vulkan backend is wired into the engine behind `--backend vulkan` and has run the full
  model on the EVO-X2 (below). Its operators are verified against the CPU path on lavapipe, a
  software Vulkan implementation used for correctness, not performance;
- the HIP backend is reachable with `--backend hip` in a build configured with
  `-DHALO_BUILD_HIP=ON`, but its device path is not covered by tests yet;
- neither GPU backend is compiled or tested by `ci.yml`, which configures with both off. The
  release workflow compiles the Vulkan backend and runs no GPU test.

Measured on the EVO-X2 on 2026-09-27 with `Qwen3.8-27B-UD-Q4_K_XL.gguf`, greedy decoding, one
short prompt (`process.md` section 1 has the commands and the full table):

| Engine | Decode tok/s |
|---|---|
| halo, Vulkan backend, `--mtp-draft 2` (the default), `--ctx 4096 --parallel 1` | 4.04 |
| halo, Vulkan backend, `--mtp-draft 0` | 2.24 |
| llama.cpp, Vulkan build on the same driver stack | 12.3 |
| llama.cpp, HIP build | 12.0 |

So HALO's Vulkan path currently decodes at about a third of llama.cpp's speed on the same
machine, and the default `--ctx 32768` did not run on the Vulkan backend in that session
(`process.md` section 12.6). `docs/evox2.md` is the run book for running on the target
hardware and bringing results back.

## Troubleshooting

- **Configure fails with "HALO_BUILD_VULKAN=ON requires glslc"** — install `glslc` and the
  Vulkan headers, or pass `-DHALO_BUILD_VULKAN=OFF`.
- **`error while loading shared libraries: libvulkan.so.1`** — the `-vulkan` archive needs the
  Vulkan loader; install it or use the `-cpu` archive.
- **`backend 'vulkan' is not available in this build`** — the binary was built without that
  backend (`halo version` lists the components it has).
- **Out of memory at start** — the KV pools are sized from `--ctx` and `--parallel`; lower
  them, and check `halo inspect model.gguf --ctx N --parallel M` first.
- **`halo serve` refuses a non-loopback `--host`** — that needs `--allow-remote` and an API key
  (`docs/cli.md`, `docs/api.md`).

## License

This repository has no license file yet, so no license is granted for the HALO sources. The
release archives include the license texts of the third-party libraries compiled into the
binary (nlohmann_json, minja, cpp-httplib) under `licenses/`.

## Specs and decisions

`docs/HALO_PRD_v1.1.md` and `docs/HALO_TRD_v1.1.md` are the product and technical
specifications. `docs/DECISIONS.md` records every point where measurement on real hardware
or real model files corrected an assumption in those specs — where the two disagree,
`DECISIONS.md` wins. `docs/adr/` holds the standalone architecture decision records (starting
with ADR-001, the backend interface). `docs/security-hardening.md` documents the threat model
and the hardening applied to every untrusted-input surface (model files, the HTTP API,
packaging). `docs/reviews/` holds point-in-time adversarial and security reviews, kept as a
historical record rather than updated in place.
