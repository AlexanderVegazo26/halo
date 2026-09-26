# HALO

HALO is a C++23 inference runtime built for one model on one machine: **Qwen3.8-27B**, a
hybrid 48×Gated-DeltaNet + 16×GQA-attention model, running on the **GMKtec EVO-X2** (AMD
Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GiB unified LPDDR5x). It is not a general
multi-model server — every design choice, from the memory planner's tier discovery to the
kernel set, is made against that single machine and that single architecture.

The engine speaks an agent-facing HTTP API (OpenAI Chat Completions/Completions and
Anthropic Messages shapes), and its own CLI for local inspection, generation, benchmarking
and autotuning.

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
The CPU backend is the only one wired into the serving engine today — Vulkan and HIP exist as
independently-verified operator implementations, not yet as something `halo serve` will
actually route a request through (see "Where the GPU work stands" below).

## Layout

| Path | What lives there |
|---|---|
| `include/halo/`, `src/` | tensor/dtype handling, the GGUF model loader, tokenizer, chat template engine, hardware discovery, the memory planner, paged KV cache, DeltaNet state, the `models::Qwen35` forward graph, speculative decoding, sampling, the runtime `Engine`, the HTTP API, benchmarking/profiling, the autotuner |
| `backends/cpu`, `backends/vulkan`, `backends/hip` | the operator backends behind the `Backend` interface |
| `tools/halo` | the `halo` CLI — `inspect`, `devices`, `tokenize`, `template`, `run`, `serve`, `bench`, `tune` (`docs/cli.md`) |
| `tests/unit` | GoogleTest suites, mostly differential: every backend and every quantization path is checked against the CPU reference or an independent transformers golden |
| `scripts/` | build scripts, the EVO-X2 field-kit packager, and the pinned llama.cpp reference build |
| `docs/` | the product/technical specs, engineering decisions, architecture decision records, security hardening notes, and per-subsystem references |

## Building

Requirements: clang with C++23 support, CMake ≥ 3.28, Ninja, SQLite3 development headers.
Add the Vulkan SDK (headers + `glslc`) to build the Vulkan backend, and ROCm to build HIP.
Third-party sources (nlohmann_json, minja, cpp-httplib, googletest) are fetched by CMake with
pinned SHA-256 hashes — nothing is vendored, nothing is fetched unpinned.

```bash
scripts/build.sh                       # release build + the full test suite
scripts/build.sh --asan                # ASan/UBSan build + tests
scripts/build.sh -DHALO_BUILD_HIP=ON   # also compile the HIP backend (needs ROCm)
```

`-DHALO_ONLY="model;sampling;api"` (a semicolon list of module/backend/test-directory names)
builds only the named subset — useful when iterating on one part of the tree without paying
for a full rebuild.

## Using the CLI

```bash
halo inspect model.gguf --ctx 32768 --parallel 4   # header report + memory plan, no weights mapped
halo run model.gguf -p "explain gated delta rule"  # one-shot generation to stdout
halo serve --model model.gguf --port 8080          # HTTP API, loopback-only unless configured otherwise
halo bench model --host-label evox2-1 ...          # the PRD's performance matrix over the real engine
halo tune model.gguf                                # search kernel variants, persist the winners
```

Full command reference, every flag, and the environment-variable/config-file precedence
rules: `docs/cli.md`. Full HTTP surface, request/response shapes, and the security model
behind loopback-by-default: `docs/api.md`.

## Where the GPU work stands

The engine only executes the CPU reference backend today — that is a deliberate gate, not a
missing feature. ADR-001 splits GPU enablement into staged milestones so each backend's
kernels are proven correct against the CPU path in isolation before the engine is allowed to
route a real request through them:

- the CPU backend is the complete, tested reference implementation every other backend is
  checked against;
- the Vulkan backend has a growing set of operators verified against it on lavapipe (a
  software Vulkan implementation used for correctness, not performance) and is being
  extended toward full forward-pass coverage;
- the HIP backend has every decode operator implemented and differentially checked against
  the CPU path in software emulation — it has not yet run on an actual AMD GPU, because
  development happens on a machine without one;
- wiring the runtime `Engine` to actually dispatch to a GPU backend, and then validating all
  of the above against gfx1151 hardware itself, are the remaining milestones.

Nothing in this repository states a performance number measured on the target GPU, because
none has been measured yet. `docs/evox2.md` is the run book for the first time any of this
GPU code executes on real hardware, and how to bring the results back for analysis.

## Specs and decisions

`docs/HALO_PRD_v1.1.md` and `docs/HALO_TRD_v1.1.md` are the product and technical
specifications. `docs/DECISIONS.md` records every point where measurement on real hardware
or real model files corrected an assumption in those specs — where the two disagree,
`DECISIONS.md` wins. `docs/adr/` holds the standalone architecture decision records (starting
with ADR-001, the backend interface). `docs/security-hardening.md` documents the threat model
and the hardening applied to every untrusted-input surface (model files, the HTTP API,
packaging). `docs/reviews/` holds point-in-time adversarial and security reviews, kept as a
historical record rather than updated in place.
