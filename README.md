# HALO

HALO is a C++23 LLM inference runtime specialized for **Qwen3.8-27B** on **AMD Strix Halo**: the GMKtec EVO-X2 with a Radeon 8060S (gfx1151) and 128 GiB of unified LPDDR5.

- **Goal:** decode efficiency close to the memory-bandwidth roofline, using HALO's own kernels (Vulkan/RADV and HIP), MTP speculative decoding, and an agent-oriented API server.
- **Specs:** [`docs/HALO_PRD_v1.1.md`](docs/HALO_PRD_v1.1.md) and [`docs/HALO_TRD_v1.1.md`](docs/HALO_TRD_v1.1.md). Decisions in [`docs/DECISIONS.md`](docs/DECISIONS.md) override both.

## Status (2026-09-25): code-complete on the CPU path; GPU path in progress; hardware verification pending

HALO has been developed on a machine **without an AMD GPU** (decision D-001).
- Everything marked "tested" below was tested there: on the CPU, in HIP host emulation, or with Vulkan on the lavapipe software rasterizer.
- **Nothing has run on the Radeon 8060S yet**, and no HALO performance number exists.

| Area | Status |
|---|---|
| GGUF loader (Q4_K_M, UD-Q4_K_XL, separate MTP file), all pack quant types | Done. The real 27B file has not been loaded yet (a smoke run is in progress) |
| Tokenizer (byte-level BPE) and Jinja chat templates | Done; matches Hugging Face token for token; hardened against hostile model files |
| CPU reference backend and qwen35 forward (GDN + attention + MTP) | Done; matches the transformers goldens on a tiny model |
| Paged KV cache, DeltaNet state with rollback slots, prefix cache | Done |
| MTP speculative decoding with profit gate | Done; greedy output with MTP on equals output with it off |
| Sampling, fused greedy argmax, JSON-schema structured output | Done |
| OpenAI + Anthropic HTTP API, `/tokenize`, `/apply-template` | Done; security fixes in progress; **loopback-only for v0.2** |
| Benchmark harness, llama.cpp and Ollama baseline adapters | Done |
| Autotuner and SQLite profile database | Done for CPU kernels; GPU variants not yet bridged |
| HIP backend | Every decode op implemented and matching the CPU in emulation; **never executed on a GPU** |
| Vulkan backend | Partial (core ops); the rest of the op set is in progress |
| Model forward on a GPU | In progress: ADR-001 backend interface (workstreams BI-1..BI-6) |

**What runs today:** `halo run`, `halo serve` and `halo bench` work end to end **on the CPU engine**. The test suite: 545 tests, 543 pass, and 2 are skipped with stated reasons.

Detailed status and next steps: [`memory.md`](memory.md) (the resume checkpoint), [`docs/dev/TECH_DEBT.md`](docs/dev/TECH_DEBT.md) and [`docs/reviews/`](docs/reviews/).

## Layout
| Path | What |
|---|---|
| `include/halo/`, `src/` | Core modules: tensor, model (GGUF), tokenizer, template, hardware, memory planner, kv_cache, state, models (qwen35), speculative, sampling, runtime (Engine), api, profiling, autotune |
| `backends/cpu`, `backends/vulkan`, `backends/hip` | Operator backends |
| `tools/halo` | The `halo` CLI: `inspect`, `devices`, `tokenize`, `template`, `run`, `serve`, `bench`, `tune` ([`docs/cli.md`](docs/cli.md)) |
| `tests/unit` | GoogleTest suites (differential tests against the CPU reference and goldens) |
| `scripts/` | Build, packaging and EVO-X2 tooling |
| `docs/` | Specs, decisions, ADRs, reviews, per-component docs (`api.md`, `benchmarks.md`, `hip.md`, `vulkan.md`, `security-hardening.md`) |

## Building (Linux / WSL)
Requirements:
- clang (C++23), CMake ≥ 3.28, Ninja;
- Vulkan SDK headers plus glslc;
- SQLite3;
- ROCm (for `-DHALO_BUILD_HIP=ON`).

Third-party sources (nlohmann_json, minja, cpp-httplib, googletest) are fetched with pinned SHA-256 hashes.

```bash
scripts/build.sh                      # Release build + all tests
scripts/build.sh --asan               # ASan/UBSan build + tests
scripts/build.sh -DHALO_BUILD_HIP=ON  # also compile the HIP backend for gfx1151
```

## Running on the EVO-X2
The step-by-step guide will live in [`docs/evox2.md`](docs/evox2.md). It is being written together with the field kit: a packaged build, a native-build fallback, and a log collector that produces a diagnostics bundle to bring back for analysis. This section will then summarize those steps.
