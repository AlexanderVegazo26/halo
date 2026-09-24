# Strix Halo + Qwen3.8-27B Local Engine Report

Machine: `NucBox_EVO-X2` (GMKtec)  
Date: 2026-09-23  
Goal: size a Splash-style, model-specific local inference engine around the hardware and `Qwen3.8-27B`.

---

## 1. Hardware

### CPU / APU

| Item | Value |
|---|---|
| Model | AMD Ryzen AI MAX+ 395 ("Strix Halo") |
| Cores / Threads | 16 cores / 32 threads |
| Max CPU clock | 5.19 GHz |
| ISA | x86-64, AVX-512F/VL/BW/DQ/IFMA/VBMI/VNNI/BF16, AVX-VNNI, AVX2, FMA |
| L1d / L1i / L2 / L3 | 768 KiB / 512 KiB / 16 MiB / 64 MiB |
| Platform | GMK NucBox EVO-X2 |

### Integrated GPU

| Item | Value |
|---|---|
| Name | AMD Radeon 8060S Graphics |
| GFX target | `gfx1151` |
| Architecture | RDNA 3.5 |
| Compute Units | 40 CUs |
| Max clock | 2.9 GHz |
| Wavefront | 32 (wave32) |
| Fast FP16 | yes |
| Memory interface | 256-bit LPDDR5, unified with CPU |
| ROCm/HIP runtime | HIP 7.15.26333, ROCm runtime 1.21 |

### System memory — important caveat

The machine physically has **128 GiB of LPDDR5**, but it is split between the GPU and the OS:

| Pool | Size | Seen by |
|---|---:|---|
| Total physical RAM | ~128 GiB | firmware / unified memory fabric |
| GPU-visible (ROCm VRAM / HMM) | ~96 GiB | `rocm-smi`, HIP allocations |
| OS/CPU-visible RAM | ~32 GiB | `free`, Linux kernel |

Kernel evidence:

```
amdgpu: RAM width 256bits LPDDR5
amdgpu 0000:c5:00.0:  98304M of VRAM memory ready
amdgpu: HMM registered 98304MB device memory
```

**Engine implication:** budget against the **96 GiB GPU pool**, not the 32 GiB the OS reports. The CPU pool is only relevant for host-side code, CPU fallback, and other system services.

### Storage

| Item | Value |
|---|---|
| Disk | Lexar SSD NQ790 2 TB (NVMe, Shenzhen Longsys/Lexar NM790 controller) |
| Root filesystem size | 1.8 TiB / 104 GiB used |

### OS / drivers

| Item | Value |
|---|---|
| OS | Ubuntu 26.04.1 LTS (Resolute Raccoon) |
| Kernel | 7.0.0-31-generic |
| GPU driver | `amdgpu` |
| ROCm / HIP | HIP 7.15.26333 / ROCm runtime 1.21 |

---

## 2. Software stack already present

### llama.cpp build

Repository: `/home/alexander-vegazo/llama.cpp`

```
version: 0.5.0-dev (build 11151, commit bd4f514db)
built with GNU 15.2.0 for Linux x86_64
```

CMake choices (from `build/CMakeCache.txt`):

- `CMAKE_BUILD_TYPE=Release`
- `GGML_HIP=ON`
- `GGML_HIP_NO_VMM=ON`
- `GGML_HIP_MMQ_MFMA=ON`
- `GGML_HIP_GRAPHS=ON`
- `GGML_CPU=ON`, `GGML_NATIVE=ON`
- `GGML_LLAMAFILE=ON`
- Backends off: CUDA, Vulkan, OpenCL, SYCL, RPC

Built binaries include `llama-cli`, `llama-server`, and the HIP backend shared library `libggml-hip.so`.

### Other services

The host is already running an inference-related stack in containers / on localhost:

- `ollama serve` on `127.0.0.1:11434`
- Open WebUI (uvicorn) on `127.0.0.1:8080` in a container namespace
- LiteLLM on `127.0.0.1:4000`
- A root-owned `/app/llama-server` container already serving the same model at `-ngl 999 --ctx-size 32768`

These were not used for the benchmarks below.

---

## 3. Model

File: `/home/alexander-vegazo/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`  
Size: 17 GiB

### Architecture from GGUF metadata

| Field | Value |
|---|---|
| `general.architecture` | `qwen35` (Qwen3.8 family) |
| `general.name` | `Qwen3.8-27B` |
| `general.size_label` | `27B` |
| `general.file_type` | 15 (`Q4_K - Medium`) |
| `qwen35.block_count` | 65 (64 main layers + 1 native MTP / `nextn` draft layer) |
| `qwen35.context_length` | 262144 |
| `qwen35.embedding_length` | 5120 |
| `qwen35.feed_forward_length` | 17408 |
| `qwen35.attention.head_count` | 24 |
| `qwen35.attention.head_count_kv` | 4 (GQA) |
| `qwen35.attention.key_length` | 256 |
| `qwen35.attention.value_length` | 256 |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.nextn_predict_layers` | 1 |
| `tokenizer.ggml.model` | `gpt2` |
| Vocabulary size | 248320 tokens |

### Hybrid layout

`full_attention_interval = 4` means the 64 main layers are grouped as:

- 48 Gated DeltaNet (linear/recurrent attention) layers
- 16 full self-attention layers

Only the 16 full-attention layers store a KV cache. The Gated DeltaNet layers carry a recurrent state instead.

### Quantization layout

- Most weights: `Q4_K` / `Q5_K`
- `token_embd.weight`: `Q4_K`
- `output.weight` (lm_head): `Q6_K`
- Norms: `F32`
- The `blk.64.*` / `nextn` MTP drafter head is present in the file.

### KV-cache cost

Because only 16 layers keep KV, the cache is cheap:

| KV dtype | Per token | 32k context | 131k context | 262k context |
|---|---:|---:|---:|---:|
| f16 | ~64 KiB | ~2.0 GiB | ~8.4 GiB | ~16.8 GiB |
| q8_0 | ~32 KiB | ~1.0 GiB | ~4.2 GiB | ~8.4 GiB |
| q4_0 | ~16 KiB | ~0.5 GiB | ~2.1 GiB | ~4.2 GiB |

---

## 4. Memory budget

Working set for `Qwen3.8-27B UD-Q4_K_XL` at 131k context, f16 KV, 8 slots, unified KV pool:

| Component | Size |
|---|---:|
| Weights | ~17 GiB |
| Native MTP draft context | ~2.1 GiB |
| Compute buffers + GDN recurrent state | ~1.8–2.5 GiB |
| KV cache (131k, f16, 16 layers) | ~8.4 GiB |
| Measured GPU resident during benchmark | ~50.6 GiB |
| **Headroom remaining** | **~45 GiB** |

Conclusion: on the 96 GiB GPU pool this configuration is comfortable. You can:

- Run 8+ slots at 131k context with f16 KV.
- Load a larger quant (UD-Q6_K is ~20 GiB).
- Add a second small model or embedding model.
- Extend context toward the advertised 262k (in practice, the model tends to emit EOS beyond ~130k, so 131k is the safe cap today).

---

## 5. Benchmarks

### 5.1 Quick CLI sanity check

```bash
./build/bin/llama-cli -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 999 -c 8192 -n 50 \
  -p "Explain speculative decoding in one sentence." \
  --no-display-prompt --simple-io
```

Result:

```
Prompt:   148.2 t/s
Generation: 12.0 t/s
```

### 5.2 llama-server config used for HTTP benchmarks

```bash
./build/bin/llama-server \
  -m /home/alexander-vegazo/models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 999 \
  --ctx-size 131072 \
  --parallel 8 \
  --kv-unified \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -b 512 -ub 256 \
  --reasoning off \
  --host 127.0.0.1 --port 8084
```

### 5.3 Methodology

- Prompts built from `/usr/share/common-licenses/GPL-3`, tokenized, repeated to the target length, and detokenized.
- `max_tokens=128` for each generation.
- Streaming OpenAI-compatible `/v1/chat/completions` endpoint.
- TTFT = time from request to first streamed token.
- Cold-prefill rows used a *different* warmup prompt of the same length so graphs were compiled but the cache was cold.

### 5.4 Cold prefill + single-stream decode

| Prompt | Cold TTFT | Prefill | Decode |
|---|---:|---:|---:|
| 1k | 0.15 s | 6,963 tok/s | 18.9 tok/s |
| 4k | 0.15 s | 26,825 tok/s | 21.3 tok/s |
| 12k | 28.5 s | 432 tok/s | 18.7 tok/s |
| 24k | 73.5 s | 334 tok/s | 21.5 tok/s |

The 1k/4k rows are fast mostly because the server had already compiled graphs during warmup. The realistic long-prompt floor is **~330–430 tok/s** cold prefill. Decode is stable around **18–21 tok/s** regardless of context length.

### 5.5 Cache reuse

| 24k prompt | TTFT |
|---|---:|
| Cold | 73.5 s |
| Cached | 0.22 s |
| **Speedup** | **331×** |

### 5.6 Concurrent short-prompt decode

512-token prompt, 128-token decode per stream:

| Streams | Wall time | Aggregate | Per-stream |
|---|---:|---:|---:|
| 4 | 22.5 s | 22.7 tok/s | 5.7 tok/s |
| 8 | 33.9 s | 30.2 tok/s | 3.8 tok/s |

Batching raises aggregate throughput from ~20 tok/s single-stream to ~30 tok/s, but each individual stream slows as the memory bus saturates.

### 5.7 GPU resident memory during run

```
VRAM Total:    96 GiB
VRAM Used:     50.6 GiB  (≈ 49%)
```

---

## 6. Analysis: what limits performance

1. **Capacity is not the limit.** 96 GiB of GPU-visible memory is ample for this 17 GiB model plus large f16 KV caches.
2. **Memory bandwidth is the limit.** Single-stream decode is ~20 tok/s and aggregate decode saturates near 30 tok/s. Strix Halo’s LPDDR5 delivers roughly 256 GB/s, and the model reads ~17 GiB of weights per decode step.
3. **Prefill is fast but not free.** A cold 24k prompt takes ~73 s. A cached 24k prompt takes ~0.2 s. Cache reuse dominates the agent experience.
4. **The native MTP drafter works.** With `--spec-type draft-mtp --spec-draft-n-max 2`, single-stream decode is ~20 tok/s. Going deeper than 2 under real sampling loses acceptance.
5. **Continuous batching behaves as expected.** More concurrent streams raise aggregate throughput but lower per-stream speed.

### Comparison with published Strix Halo numbers

A similar Strix Halo laptop (Ryzen AI Max+ PRO 395, Radeon 8060S, 128 GB, Lemonade ROCm build) was reported at roughly [12–14 tok/s decode and 205–230 tok/s prefill](https://akougkas.io/journal/2026/serving-qwen38-27b-on-a-radeon-ai-pro-r9700/) for the same model family. This build did better on the tested run: ~20 tok/s decode and >300 tok/s cold prefill, likely because of a newer llama.cpp revision plus f16 KV + unified cache.

---

## 7. Recommendations for a Splash-like engine

### Short-term: best llama.cpp config

```bash
./build/bin/llama-server \
  -m /home/alexander-vegazo/models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 999 \
  --ctx-size 131072 \
  --parallel 8 \
  --kv-unified \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -b 512 -ub 256 \
  --reasoning off
```

Use fewer slots if individual stream latency matters; use more if total throughput matters.

### Medium-term: engine features worth building

1. **Paged KV cache + prefix caching.** The 331× cached-turn speedup is the biggest user-visible win.
2. **Gated DeltaNet state checkpoints.** Cache reuse only works on the full-attention layers unless the recurrent GDN state is also snapshotted at prefix boundaries.
3. **Shape-specialized HIP kernels.** Compile/fuse kernels for the exact model shapes: 5120 hidden, 17408 FFN, 24/4 heads, 256 head dim, 248320 vocab, 16 full-attn + 48 GDN layers.
4. **Batched speculative decode.** Fuse MTP drafting, verification, acceptance, and state updates into one decode unit.
5. **Memory budgeter.** Since the model sizes are known, compute the safe budget at startup (weights + draft + buffers + KV per token × max context) and expose only `--max-memory` / `--max-context` ceilings.
6. **Scheduler that balances prefill and decode.** New requests should start quickly; running streams should keep decoding.

### Experiments to try next

- Test UD-Q6_K (~20 GiB) for quality, since memory headroom is large.
- Try `-b 2048 -ub 512` to see if the gfx1151 ROCm micro-batch crash still occurs on this build.
- Benchmark 8 concurrent long-context streams to find the real slot/context ceiling.
- Compare native MTP against a separately trained DFlash2 draft on this hardware.

---

## 8. References

- [Inco AI — Splash: A Local Engine Built Around the Model](https://inco.ai/blog/splash/)
- [Serving Qwen3.8-27B on a Radeon AI PRO R9700 — akougkas.io](https://akougkas.io/journal/2026/serving-qwen38-27b-on-a-radeon-ai-pro-r9700/)
- llama.cpp repository: `/home/alexander-vegazo/llama.cpp`, commit `bd4f514db`, build 11151
- Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17 GiB, `qwen35` architecture
