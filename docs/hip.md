# HALO HIP backend (WS-K)

Status: **milestones M1–M6 complete**.
- M1: the host runtime and the GATED_DELTANET family.
- M2: quantized GEMV.
- M3: the LM head with fused argmax, TOP_K, RMS_NORM, PARTIAL_ROPE, SWIGLU and MUL_SIGMOID.
- M4: the PARTIAL_ROPE head stride (TD-9) and GQA ATTENTION over the paged KV cache.
- M5: GEMV for the D-014 second-tier types (IQ4_XS, IQ4_NL, Q3_K, IQ3_S). With them, every
  weight type of the canonical UD-Q4_K_XL pack has a HIP GEMV.
- M6: decode primitives.
  - KV write into the kv_cache pool;
  - GET_ROWS (embedding lookup) for every GEMV type;
  - ADD, and the fused ADD + RMS_NORM;
  - Q4_0 GEMV for the ggml-org MTP pack (D-006).

  See "One qwen35 decode step" for the per-op status.

What remains is listed under "Not done".

HALO's own kernels only. No rocBLAS, hipBLASLt or MIOpen (TRD §3.3); nothing here links a
vendor math library.

## What was verified where (read this first)

The development host has no AMD GPU (DECISIONS D-001). Each claim below is labelled with
how it was established.

| Claim | How it was established |
|---|---|
| Device code compiles for gfx1151 | **Compiled.** Every build with `-DHALO_BUILD_HIP=ON` (ROCm 7.1.0, amdclang 20). |
| Kernel math and indexing match `halo::cpu` bit for bit | **Host emulation.** See the next paragraph. |
| Kernels run correctly on a device | **Unverified.** Device tests exist and skip on the dev host with "no HIP device (dev host, D-001)". |
| Performance, occupancy and bandwidth | **Unmeasured.** The figures below are the compiler's static estimates. |

**Host emulation.** Each kernel body is written once, as a `__host__ __device__` template
in `backends/hip/kernels/*.h`, against a small execution model (`kernels/hd.h`):
- a body is one workgroup;
- `ex.phase(f)` runs `f(tid, regs)` for every thread and then acts as a barrier.

The device runs the body through `DeviceExec`: `f(threadIdx.x, r)` followed by
`__syncthreads()`. The host runs it through `HostExec`, which loops over the threads.
So both sides share the same source for:
- the tiling and the thread-to-data mapping;
- the per-element expressions and the fp32 summation order;
- the rounding, since everything is built with `-ffp-contract=off`, as `halo::cpu` is.

The unit tests run the emulation twice, once in forward and once in reverse
thread/workgroup order. Registers and LDS start poisoned with NaN. Both runs must equal
`halo::cpu` bit for bit on the same input buffers (TRD §30, D-016, review M-2).

**What the emulation does NOT verify:**
- device scheduling and the memory model (barrier placement is checked only in the
  sense that a missing barrier makes the two orders or the poison disagree);
- wave-level behaviour;
- the device libm: ocml `expf`, which is not guaranteed correctly rounded;
- register allocation and occupancy;
- anything about speed.

## Toolchain

- Dev host: ROCm **7.1.0** (`/opt/rocm`, HIP 7.1.25424, amdclang 20). CMake's first-class
  HIP language.
- Target EVO-X2: **HIP 7.15.26333 / ROCm runtime 1.21**
  (docs/strix-halo-qwen38-engine-report.md).
- The version difference is unresolved. The fatbin built here targets `gfx1151` and has
  not been loaded by the 7.15 runtime. The first EVO-X2 run is also the first
  compatibility check.
- `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1151`.
- A generic ISA (`gfx11-generic`) needs an explicit `-DHALO_HIP_ALLOW_GENERIC=ON`. Without
  it the configure step fails (TRD §3.3 "fail loudly"). With it, the build prints a
  warning, `Context::create` logs the choice, and `compiled_for_generic_isa()` reports it.
- With a gfx1151 build, `Context::create` raises `Error(Device)` if the device reports
  another architecture.
- Host code is built with the full HALO warning set and `-Werror`. HIP headers come in as
  system includes of the imported `hip::host` target. The `.hip` TU is also built with
  HALO's warnings.
- **ASan builds** (`scripts/build.sh --asan`):
  - the host side of every TU is sanitized;
  - device-side sanitizers need an `xnack+` target, which gfx1151 is not, so the device
    library adds `-fno-gpu-sanitize`. Device code is therefore not sanitized.

Build (inside WSL, as the brief describes):

    HALO_BUILD_DIR=/root/halo-build-wsk bash scripts/build.sh "-DHALO_BUILD_HIP=ON" "-DHALO_ONLY=tensor;cpu;hip;kv_cache"

(`kv_cache` is needed only for `test_hip_attention`, which is skipped at configure time without it.)

The default build (`HALO_BUILD_HIP=OFF`) contains no HIP target and no HIP test.

## Layout

| Path | Contents |
|---|---|
| `include/halo/backends/hip/runtime.h` | `probe()`, `Context`, `DeviceInfo`, `Buffer` + `MemoryTier`, `Stream`, `Event`, `error_code_for` / `check` |
| `include/halo/backends/hip/registry.h` | kernel variant registry (`kernel_variants`, `find_variant`, `default_variant`) |
| `include/halo/backends/hip/ops.h` | `BufferView`, `Target` (device stream or host emulation), `Ops`, operator argument structs |
| `backends/hip/kernels/hd.h` | execution model, host/device math shims, the CPU's fixed-order 8-lane dot |
| `backends/hip/kernels/gdn.h` | GATED_DELTANET recurrent + chunked (K0/K1/K2) bodies |
| `backends/hip/kernels/conv_norm.h` | CONV1D_SHORT, RMS/GATED_NORM bodies |
| `backends/hip/kernels/gemv.h` | QUANT_GEMV bodies (generic + wave), per-element dequant matching `tensor::dequantize_row`, and the LM head's argmax epilogue |
| `backends/hip/kernels/head.h` | ARGMAX (partial + reduce), TOP_K (bitonic), PARTIAL_ROPE, SWIGLU / MUL_SIGMOID bodies |
| `backends/hip/kernels/attention.h` | ATTENTION bodies: block-table check, exact and online (running-max) variants |
| `backends/hip/src/launch_kernels.hip` | `__global__` wrappers + launchers (the only device TU) |
| `backends/hip/src/emulate.cpp`, `host_exec.h` | host emulation |
| `backends/hip/src/ops.cpp` | validation (before anything runs), parameter building, dispatch |
| `backends/hip/src/runtime.cpp`, `registry.cpp` | runtime, variant table |

## Host runtime (TRD §17)

- **`probe()`** never throws. On the dev host it reports:

      hipErrorNoDevice (100): no ROCm-capable device is detected

- **`Context::create(ordinal)`** reads these device properties: name, `gcnArchName`
  (arch and features), CU count, wave size, clock, memory, shared memory per block,
  integrated, managed memory, and the driver and runtime versions.
- **`Buffer::allocate(ctx, bytes, tier)`** allocates in one of these tiers:

  | Tier | HIP call | On Strix Halo |
  |---|---|---|
  | `Device` | `hipMalloc` | carveout |
  | `HostPinned` | `hipHostMalloc(Mapped)` + device pointer | GTT |
  | `Managed` | `hipMallocManaged` | — |

  An allocation failure raises `Error(Memory)`. `Buffer::wrap_host` wraps host memory for
  the emulation target and never reaches the device.
- **`Stream`** uses the default (blocking) flags, so it is ordered with the synchronous
  `hipMemcpy` of `Buffer::upload`/`download` on the null stream; `Ops::check_status`
  synchronizes the target stream before reading. **`Event`** provides `record`, `synchronize` and
  `elapsed_ms`.
- **Error translation** (`error_code_for`) maps a `hipError_t` to a `halo::ErrorCode`:

  | HIP errors | ErrorCode |
  |---|---|
  | out of memory | Memory |
  | no device, invalid device, initialization, insufficient driver, shared-object init | Device |
  | launch failure, out of resources, illegal address, invalid configuration, invalid device function, no binary for GPU, invalid kernel file | Kernel |
  | not supported | Unsupported |
  | anything else | Backend |

  The tests check this table using the named enumerators.
- **Kernel registry** (`registry.h`): each entry records op id, form, variant name,
  device kernels, block size, work mapping and the DECISIONS it implements. `Ops` selects
  variants by name. An unknown name, or one of the wrong form, raises `Error(Config)`.
- **Not in M1:** `hip_graph` (graph capture for decode replay), `hip_profiler`, and tier
  probing microbenchmarks. The hardware module owns tier discovery (D-002).

## Operators

The contracts are the CPU reference's (`include/halo/backends/cpu/ops.h`). The operands are
`BufferView{buffer, offset, bytes, row_stride}`, the Vulkan backend's concept (review S-1):
- fp32, with 4-byte-aligned offsets and strides;
- aliasing is checked on address ranges, and outputs may alias only exactly (same address
  and stride), where the contract allows it;
- every check runs before any kernel is enqueued.

| Op (TRD §9) | Kernel(s) | Variants (default first) | DECISIONS | Contract notes |
|---|---|---|---|---|
| GATED_DELTANET, recurrent | `k_gdn_recurrent` | `gdn_recurrent_b128`, `gdn_recurrent_b64` | D-003, D-004 (5, 6), D-012, D-016 | Grid (n_v, ⌈d_v/block⌉). One thread per value column, which keeps S[:, c] in registers. Per token: thread 0 computes the q/k L2 norms, then all threads write normalized q (× q_scale) and k to LDS, then each column runs the delta rule. Tiled (`j % n_k`) or grouped head mapping. Slots s < min(T, n_slots) are written, slots s ≥ T are left untouched. In place or `state_out`. |
| GATED_DELTANET, chunked | `k_gdn_check_g` → per group (`k_gdn_chunk_intra`, `k_gdn_chunk_state`) | `gdn_chunked_b64`, `gdn_chunked_b32` | same | Same math as `cpu::gated_delta_rule_chunked`: UT forward substitution, intra-chunk attention, one state update per chunk, and slots from the chunk's closed form. Workspace records hold q, k, k_cumdecay, new_values/v_new, attn, decay and gc. The op runs as many chunks per group as fit. Group 0 reads `state`, later groups read `state_out`. |
| CONV1D_SHORT | `k_conv1d_silu` | `conv1d_silu_b256`, `conv1d_silu_b64` | D-004 (2), D-012 | One thread per channel. The K−1 history lives in registers (K ≤ 8). Fused SiLU. out may alias x. Slots. |
| GATED_NORM | `k_norm` (gate operand set) | `gated_norm_b128`, `gated_norm_b32` | D-004 (7) | One workgroup per row. Lanes 0..7 compute the CPU's 8 interleaved partial sums, thread 0 combines them, then the element-wise `(w·(x·inv))·silu(z)`. out may alias x or z. |
| QUANT_GEMV / MATMUL (decode) | `k_gemv_wave`, `k_gemv_generic` | `gemv_wave32_r4`, `gemv_wave32_r8`, `gemv_generic_b64` | D-007, D-014 | `y[t][n] = Σ_i x[t][i]·W[n][i]` for F32, F16, Q8_0, Q4_K, Q5_K, Q6_K and the second tier IQ4_XS, IQ4_NL, Q3_K, IQ3_S, in ggml block layouts, T = n_vec vectors (decode T = 1, MTP verify T > 1). Weight rows may start at any byte and have any byte stride (F32 needs 4-byte alignment). Other types raise `Error(Unsupported)`. See "GEMV reduction orders" below. |
| LOGITS_MATMUL + ARGMAX_FUSED (LM head) | the QUANT_GEMV kernel with its argmax epilogue → `k_argmax_reduce` | GEMV variant (`OpsOptions::gemv`) + `argmax_b256` | D-007, D-016, TRD §19 | `cpu::matmul_argmax` per vector (T = n_vec). Each GEMV workgroup ranks its own rows as soon as their logits exist, so the logits never have to be written (they are written only if `gemv.y` is set). Stage 2 is one workgroup per vector. Ties go to the lowest index; −inf is an ordinary value. A NaN sets the result's NaN word, and `Ops::read_argmax` raises `Error(Kernel)`. Result layout `{index, value bits, nan}` is the Vulkan backend's. |
| ARGMAX_FUSED (over logits) | `k_argmax_partial` → `k_argmax_reduce` | `argmax_b256` | D-016 | `cpu::argmax` per vector. 16 logits per thread, then a fixed LDS tree. |
| TOP_K | `k_topk` (repeated rounds) | `topk_bitonic_b256` | TRD §19, D-016 | `cpu::top_k` per vector: the k largest, sorted descending, ties by lower index; 1 ≤ k ≤ min(n, 1024). Each round bitonic-sorts 2048-candidate chunks in LDS (16 KiB) and keeps k of each, until one chunk remains. For 248,320 logits that is 3 rounds at k = 40 and 8 rounds at k = 1024. NaN sets `kStatusNaN`, and `check_status` raises `Error(Kernel)`. Only k ids and values leave the device. |
| RMS_NORM | `k_norm` (no gate) | `rms_norm_b128`, `rms_norm_b32` | D-004 | `cpu::rms_norm`: `(x·inv)·w`, the CPU's 8-lane sum order. Per-head norms are rows = T·heads. out may alias x. |
| PARTIAL_ROPE | `k_rope` | `rope_neox_b128` | D-004 | `cpu::partial_rope_neox` in place. `head_stride` (TD-9, default head_dim) addresses head h at h·head_stride, so RoPE runs in place on qwen35's interleaved [Q | gate] attn_q row (head_stride = 512), leaving the gate halves untouched; pass the full row as the view's row_stride. inv_freq is computed on the host with the CPU's expression and passed in the kernel arguments. The angle is one fp32 multiply; cos/sin are evaluated in double of the fp32 angle. rot_dims ≤ 128 (backend limit; qwen35 uses 64). |
| SWIGLU / MUL_SIGMOID | `k_eltwise` | `swiglu_b256`, `mul_sigmoid_b256` | D-004 | `silu(gate)·up` and `x·sigmoid(gate)`, with the CPU's expressions. out may alias either input. |
| ATTENTION (paged GQA) | `k_attn_check` → `k_attn_online` or `k_attn_exact` | `attn_online_b128`, `attn_online_b64`, `attn_exact_b128` | D-004 | `cpu::attention_gqa`: causal (query t sees rows 0 ..= q_offset + t); q head h uses KV head h / (n_head / n_kv_head); scores are `(q·k)·scale`. Reads the halo::kv_cache pool byte for byte: block[layer][K\|V][token][kv_dim] plus the sequence's uint32 block table (fragmented tables and a partial last block are fine). `q_head_stride` reads Q in place from the interleaved attn_q row. Block ids ≥ n_pool_blocks are flagged by `k_attn_check` (`kStatusBadBlock` → `check_status` raises `Error(Kernel)`) and never dereferenced; `out` is then undefined. See "Attention accumulation order" below. head_dim ≤ 256. |

### GEMV reduction orders and the tolerance

Dequantization (`kernels/gemv.h` `wq_elem`) reproduces `src/tensor/quant.cpp` operation by
operation, including the fp16 → fp32 conversion. Every dequantized weight is therefore
bit-identical to `halo::tensor::dequantize_row`. The variants differ only in how they sum:

| Variant | Summation | Accepted by |
|---|---|---|
| `gemv_generic_b64` | 8 lanes per row in `cpu::detail::dot`'s order: lane l sums i ≡ l (mod 8) in increasing i, then `((p0+p4)+(p1+p5))+((p2+p6)+(p3+p7))` + tail | **bitwise** equality with `dequantize_row` + `cpu::matmul` |
| `gemv_wave32_r4` (default), `gemv_wave32_r8` | one wave32 per row; lane l owns the 8-element groups l, l+32, … in increasing i (one scale lookup per group, contiguous loads), then a fixed LDS tree 16, 8, 4, 2, 1 | the summation bound below |

**The bound (derived, not tuned).**
- Both sides form every product `x_i·w_i` from bit-identical operands, so the products are
  identical and the results differ only by summation rounding.
- A sum whose longest addition chain has depth d satisfies `|ŝ − s| ≤ d·u·Σ|t_i|` to first
  order, with u = 2⁻²⁴.
- Hence `|y_hip − y_cpu| ≤ (d_cpu + d_wave)·u·Σ_i|x_i·w_i|·(1 + 10⁻³)`, where:
  - `d_cpu = K/8 + 3 + K mod 8 + 1`;
  - `d_wave = ⌈K/32⌉ + 5`;
  - `Σ|x_i·w_i|` is computed in double from the dequantized weights.
- Observed on the test data: at most **0.022** of the bound (emulation).

The bound is loose for large K (about a quarter of an average term at K = 5120), so it
cannot see a single wrong weight there. Dequantization errors are caught by the generic
variant's bitwise test, which shares `wq_elem` with the wave variants. The bound catches
reduction bugs in the wave kernels (a dropped tree level, a wrong group stride; see the
demonstrate-fail record in the M2 report).

### Weight types of the canonical pack (M5)

These are the types in the canonical pack, unsloth UD-Q4_K_XL (D-014).
- **Source:** `/root/halo-ref/unsloth-ud-q4kxl.summary.json`, produced by
  `python/tools/fetch_reference.py` from the real GGUF header (866 tensors). It is
  consistent with D-007, and the byte total matches its 17.56 GB.
- **How the table was made:** the byte counts are the ggml block geometry × element counts.
  I did not re-read the header with halo's own `model::GgufFile`.

| Type | Tensors | GB | Share | HIP GEMV |
|---|---|---|---|---|
| Q5_K | 191 | 7.926 | 45.2 % | M2 |
| IQ4_XS | 70 | 3.125 | 17.8 % | **M5** |
| Q4_K | 69 | 3.060 | 17.4 % | M2 |
| Q6_K | 56 | 2.862 | 16.3 % (incl. the LM head) | M2 |
| IQ4_NL | 6 | 0.280 | 1.6 % | **M5** |
| Q8_0 | 110 | 0.131 | 0.7 % | M2 |
| Q3_K | 3 | 0.115 | 0.7 % | **M5** |
| IQ3_S | 1 | 0.038 | 0.2 % | **M5** |
| F32 | 360 | 0.011 | 0.1 % (norms, ssm_a, …) | M2 |

- **Dequantization:** the M5 types dequantize element by element in `wq_elem`, reproducing
  `src/tensor/quant.cpp` (`deq_iq4_xs`, `deq_iq4_nl`, `deq_q3_k`, `deq_iq3_s`) operation
  by operation.
- **Lookup tables:** `kvalues_iq4nl`, `iq3s_grid` and `kmask_iq2xs` are halo::tensor's own
  `src/tensor/ggml_tables.h`, included by the kernels. There is one table source for the
  CPU reference and the device, and no private copy (review M-2). The tables are
  `constexpr std::array`. They appear as data objects in the gfx1151 code object:
  `llvm-readelf -s` on the unbundled `.hip_fatbin` shows `iq3s_grid` (2048 B),
  `kvalues_iq4nl` (16 B) and `kmask_iq2xs` (8 B). That the device reads them correctly is
  unverified until the kernels run on hardware.
- **Acceptance:** as for M2. The generic variant is bitwise equal to `dequantize_row` +
  `cpu::matmul`; the wave variants are within the summation bound.

### Attention accumulation order and the tolerance

Both variants use one workgroup per (query row, query head). Both are numerically stable:
every exponent is `x − m ≤ 0`, with m either the exact maximum or the running maximum.

**`attn_exact_b128`** runs the CPU's order.
1. Scores: 8-lane dot × scale, in the order of `cpu::detail::dot`.
2. m = max over all scores.
3. `p_s = exp(x_s − m)`.
4. One thread forms the sequential sum `Σ_s p_s` in ascending s.
5. `p_s /= sum`.
6. `o[d] = Σ_s p_s·v_s[d]`, sequential in ascending s.

It is bit-identical to `cpu::attention_gqa` in emulation. The serial sum makes it slow on
purpose: it is the exactness reference.

**`attn_online_b*`** (default) makes one pass over key tiles of `block` keys.
1. Each thread scores one key of the tile.
2. `m' = max(m, max_tile)` and `α = exp(m − m')`.
3. `p_j = exp(x_j − m')`.
4. `l = l·α + Σ_j p_j`, ascending j, on one thread.
5. `acc[d] = acc[d]·α + Σ_j p_j·v_j[d]`, ascending j.
6. At the end, `o = acc / l`.

This is deterministic but rounds differently from the CPU. It is accepted by a first-order
bound per output element, with u = 2⁻²⁴:

    |o_hip − o_cpu| ≤ u·(4N + 3·n_tiles + 4R + 16)·Σ_s w_s·|v_s[d]|

- N is the number of keys and R = max_s (m − x_s) is the score range.
- The terms cover:
  - each side's `exp`;
  - the rounding of the exponent argument;
  - the α rescale chain;
  - the normalizing sums;
  - the weighted accumulation.
- Weights and R come from the CPU-side inputs, in double.
- Observed on the test data: at most **0.085** of the bound.
- Tile-rescale bugs (a dropped α on l or on acc) show up only with more than one tile. The
  1001-key case covers that, and both mutations went red there.

### D-016 (the GDN q/k contract)

- q and k arrive raw, as they come out of conv + SiLU.
- If `qk_l2norm` is set, the kernel computes `q·rsqrt(Σq² + 1e-6)`, and the same for k.
- Then `q *= q_scale`. When `q_scale` is nullopt it is 1/√d_k, and it must be finite.
- Nothing else is applied.

### Chunked-form data error

The CPU raises `Error(Kernel)` before writing when some g is not ≤ 0 (NaN included). The
HIP op handles this with a status word:
1. The op zeroes the status word at the start of every call, so a stale bit cannot turn
   later calls into no-ops. This is tested.
2. `k_gdn_check_g` runs over all T rows and sets `kStatusPositiveG` if any g is not ≤ 0.
3. The intra and state kernels then return immediately, so out, state, state_out and the
   slots keep their contents. This is tested bit for bit.
4. `Ops::check_status` raises `Error(Kernel)` after the stream has completed.

### Divergences from the CPU contract

Each of these is rejected with `Error(Kernel)` and tested:

| Limit | HIP backend | CPU |
|---|---|---|
| `d_k` | ≤ 128 (register-resident state column) | any |
| `chunk_size` | ≤ 64 (LDS tiles) | up to 1024 |
| Conv kernel size | ≤ 8 | any |
| TOP_K k | ≤ 1024 (TRD §19's pre-filter size) | ≤ n |
| RoPE rot_dims | ≤ 128 | ≤ head_dim |
| Attention head_dim | ≤ 256 | any |

Two additions go beyond the CPU contract: RoPE `head_stride` and attention `q_head_stride`
(TD-9). The CPU views cannot express the interleaved layout (review S-2).

In place is also different:
- The CPU GDN op is always in place.
- The HIP op also accepts a disjoint `state_out`. The input state is then left unchanged,
  which is tested.

## Tests (tests/unit/hip)

- **`test_hip_runtime`** covers probe, the no-device `Context::create` → `Error(Device)`,
  error translation, compiled arch = gfx1151, host `Buffer` range checks, and the
  registry and variant validation.
  - Device-only (skipped here): properties, allocation of every tier with round trip and
    oversize → `Error(Memory)`, stream/event timing, and rejection of the wrong memory
    kind per target.
- **`test_hip_gdn`**. Emulation cases each run in forward and reverse order and are
  compared bitwise against `halo::cpu`:
  - **Recurrent:** real dims 16/48/128/128 at T = 1, and T = 5 with 3 slots,
    out-of-place and a fused qkv view. Also:
    - T = 3 with 5 slots (slots s ≥ T stay untouched);
    - odd dims d_k 20 / d_v 150, grouped mapping, no L2 norm, q_scale 0.37, 3 column
      blocks;
    - d_v 100 at block 64;
    - d_k 7 (dot tail only).
  - **Chunked:** real dims at T = 70 (a partial chunk). Also:
    - chunk 16 with slots straddling a chunk boundary;
    - a 2-chunk workspace (4 groups) with slots straddling a group boundary, both in place
      and out of place;
    - odd dims with chunk 8 and 1-chunk groups;
    - T = 1;
    - chunk 5 with d_k 7.
  - **Bad g:** positive g and NaN g → `Error(Kernel)`, with every output untouched.
  - **Status reuse:** a status word left set by a failed call is cleared by the next call.
  - **Long trajectory** (TRD §29/§30, review S-5): T = 4096 through the chunked form
    (64 chunk boundaries, 16-chunk groups) and 4096 recurrent steps; outputs, final state
    and slots are bit-identical.
  - **Rejections:** d_k 129, chunk 65, partial state/state_out overlap, out overlapping v,
    misaligned offset, too-small view, slot count without a slot buffer.
  - **Conv:** qwen35 decode with C = 10240. Also MTP verify in place with 4 slots, K = 2
    with a strided x, K = 1, and K = 8.
  - **Gated norm:** 48 heads × 128. Also cols 37 with out = x at block 32, out = z with a
    strided view, and cols 5.
- **`test_hip_gemv`** runs the emulation in both orders against `tensor::dequantize_row`
  followed by `cpu::matmul` on the same bytes. There are 22 shapes: 2 per type for all eleven
  types (Q4_0 added in M6), each through all three variants.
  - **Weights:** random quants with finite f16 scales. Every row includes one subnormal
    f16 scale (or subnormal element for F16 and F32), which exercises the fp16 subnormal
    path.
  - **Shape 1:** 37 rows, K = 512 (K = 100 for F16/F32, which exercises the dot tail).
  - **Shape 2:** 13 rows, K = 5120 (the qwen35 hidden size), T = 3 vectors, row stride
    padded by 3 bytes and the first row at byte offset 1. For F32 the padding and offset
    are 4 bytes, keeping 4-byte alignment. x and y are strided views (x padded by 5
    floats, y by 3), and the y padding must keep its sentinel.
  - **Rejections:** an unsupported type (Q4_0) raises `Error(Unsupported)`; cols not a
    multiple of the block size, y overlapping w, a view one block short, and misaligned
    F32 weights each raise `Error(Kernel)`.
- **`test_hip_head`** runs the emulation in both orders against `halo::cpu` / `halo::tensor`.
  - **LM head:** 6 cases × the 3 GEMV variants:
    - Q6_K (the real LM-head type, D-007): 1003 rows × 512, T = 2;
    - Q4_K without writing logits;
    - F16 at T = 3;
    - an exact tie (the winning row duplicated at a higher index);
    - a NaN weight, which gives `Error(Kernel)` on both sides.
    - the production shape: F32, 248,320 rows (D-008) × 16, T = 2, so stage 2 reduces
      about 62K partials per vector.
  - **LM-head acceptance by GEMV variant:**
    - Generic GEMV: index and value must be bitwise equal to `cpu::matmul_argmax`, and the
      written logits must equal `cpu::matmul`.
    - Wave GEMV: the index must be equal, and the value within the M2 summation bound.
      The test first asserts that the CPU's top-2 gap exceeds twice the worst bound, so
      the index is decided by the data, not by rounding.
  - **Argmax** (bitwise): the 248,320 vocabulary at T = 3 with strided rows, a tied
    maximum, all −inf (index 0), NaN → `Error(Kernel)`, and n = 1.
  - **Top-k** (ids and values bitwise vs `cpu::top_k`):
    - vocabulary with k = 40 at T = 2, k = 1024 with heavy exact ties (3 rounds), and
      k = 1;
    - n = 2049 with k = 1024 (just over one chunk);
    - k = n = 100;
    - all −inf;
    - NaN → `Error(Kernel)`.

    Rejections: k > n, k = 0, ids overlapping values.
  - **RMS norm, RoPE, SwiGLU, MUL_SIGMOID** (bitwise):
    - RMS norm: per-head norm 48 × 256 in place, 3 × 5120 strided, cols 37 at block 32;
    - RoPE: 24 and 4 heads × 256 with rot 64 and θ = 10⁷, rot 128 = head_dim;
      positions up to 262,143;
    - SwiGLU at the FFN width 17408, and MUL_SIGMOID at 6144; both also with aliasing and
      ±90 inputs.

    RoPE rejects rot 130 and odd rot.
- **`test_hip_attention`** builds its K/V history in a real `halo::kv_cache::KvPool`. Two
  sequences grow in alternation, so each block table is fragmented. It uses layer 1 of 2,
  and every case ends in a partially filled block. The CPU reads `SequenceKv::keys()` /
  `values()`; the HIP op reads a byte copy of the pool and the block ids.
  - **Cases,** each through the 3 variants: exact must be bitwise, online must be within
    the bound.
    - qwen35 decode: 24/4 heads, hd 256, 41 rows = 3 blocks;
    - MTP verify at T = 4, causal, with Q read in place from interleaved [Q | gate];
    - hd 20 with 6/2 heads, block 5, history = T;
    - 1001 keys: 8 tiles of 128 or 16 of 64.
  - **Errors and rejections:**
    - a bad block id raises `Error(Kernel)`;
    - rejections: head_dim 288, n_head not a multiple of n_kv_head, a table too short,
      a layer outside the pool, out overlapping the pool.
  - Device runs use the same derived bound rather than the fixed 2e-5 tolerance. In
    emulation the worst |d| is 6e-8 and the worst |d|/bound is 0.085, so the bound is
    generally stricter than 2e-5 here. Its exp term assumes ≤ 2 ulp per side, to be
    calibrated on the EVO-X2.
  - This test needs `kv_cache` in `HALO_ONLY`. Without it, CMake prints a warning and the
    test is not built.
- **RoPE head stride (TD-9, in `test_hip_head`):** 24 heads × [Q 256 | gate 256] at T = 5
  with rot 64 and θ = 10⁷. Q must be bitwise equal to `cpu::partial_rope_neox` on a
  de-interleaved copy, and the gate halves must be untouched. `head_stride < head_dim` is
  rejected.
- **`test_hip_decode`** (M6) runs both emulation orders, all bitwise.
  - **GET_ROWS:** all 11 GEMV types against `tensor::dequantize_row`, with first, last and
    repeated ids, plus the pack's Q4_K `token_embd` at 5120 columns. A bad id raises
    `Error(Kernel)`.
  - **ADD + RMS_NORM:** against `cpu::add` then `cpu::rms_norm`, at 1 × 5120 with h
    aliasing a, 3 × 5120, and 4 × 37 with h aliasing b. The same inputs check plain ADD
    against `cpu::add`.
- **`test_hip_kv`** (M6; needs `kv_cache`): writes into a real KvPool with a fragmented
  table, layer 1 of 2, and every float of the pool pre-filled. After the write, the
  **whole pool** must be byte-identical to `SequenceKv::write` on the same inputs. Cases:
  - 13 + 5 rows across the 16-row block boundary;
  - a decode step into a partial block;
  - a 12-row prefill over 3 blocks of 5.

  A bad block id raises `Error(Kernel)`.
- **Device tests.** Each scenario above also exists as a `…Device…` test. They skip on
  the dev host.
  - GDN, conv and norm compare against `halo::cpu` within `|err| ≤ 2e-5 + 1e-5·|ref|`.
    That tolerance is an unmeasured choice, to be confirmed on the EVO-X2.
  - ARGMAX and TOP_K must match exactly on the device too (pure comparisons), and so must
    the LM head with the generic GEMV. RMS norm, RoPE (device double `cos`/`sin`) and the
    element-wise ops use the tolerance above.
  - GEMV uses the same criteria as the emulation. The generic variant must be bitwise
    equal, because it uses only IEEE fp32 multiply and add. A device mismatch there (for
    example from denormal flushing) is a finding to investigate, not a tolerance to widen.

### Running the device tests on the EVO-X2

    cmake -S . -B build-hip -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
          -DHALO_BUILD_HIP=ON -DHALO_ONLY="tensor;cpu;hip;kv_cache"
    cmake --build build-hip && ctest --test-dir build-hip -R Hip --output-on-failure

`ROCM_PATH` must be `/opt/rocm`, or set `CMAKE_HIP_COMPILER`. Every `…Device…` test must
report Passed, not Skipped. A skip there means the runtime saw no device.

## Build-time resource usage (compiler estimates, `-Rpass-analysis=kernel-resource-usage`)

**Unmeasured on a device.** These are static numbers from amdclang 20 for gfx1151 (wave32)
at `-O2` (RelWithDebInfo).

| Kernel | VGPRs | Scratch B/lane | LDS B/block | Occupancy est. (waves/SIMD) |
|---|---|---|---|---|
| `k_gdn_recurrent` | 167 | 0 | 1040 | 9 |
| `k_gdn_check_g` | 12 | 0 | 0 | 16 |
| `k_gdn_chunk_intra` | 26 | 0 | 17408 | 14 |
| `k_gdn_chunk_state` | 20 | 0 | 32768 | 2 |
| `k_conv1d_silu` | 40 | 0 | 0 | 16 |
| `k_norm` | 15 | 0 | 36 | 16 |
| `k_gemv_wave` | 35 | 0 | 1152 | 16 |
| `k_gemv_generic` | 33 | 0 | 1152 | 16 |
| `k_argmax_partial` | 10 | 0 | 3072 | 16 |
| `k_argmax_reduce` | 10 | 0 | 3072 | 16 |
| `k_topk` | 11 | 0 | 16384 | 16 |
| `k_rope` | 43 | 0 | 512 | 16 |
| `k_eltwise` | 11 | 0 | 0 | 16 |
| `k_attn_check` | 3 | 0 | 0 | 16 |
| `k_attn_exact` | 27 | 0 | 2056 | 16 |
| `k_attn_online` | 28 | 0 | 10252 | 16 |
| `k_kv_write` | 7 | 0 | 0 | — |
| `k_get_rows` | 17 | 0 | 0 | — |
| `k_add_norm` | 11 | 0 | 36 | — |

The online kernel's per-thread accumulator array (8 floats × 256 threads = 8 KiB) was
placed in LDS by the compiler (promote-alloca) rather than in registers. That is why it
uses 10 KiB of LDS and not 2 KiB.

Two findings from getting these numbers (measured at build time):

**1. The first register-resident versions spilled.**
- The first recurrent kernel guarded every `i < d_k` step, and the compiler CSE'd or
  hoisted the 128 LDS loads of k/q next to the 128-register state column. Its first build
  reported 824 B/lane of scratch; stand-alone variants reached 256 VGPRs plus scratch.
- The fix has two parts:
  - pad the column to 128 with zeros, so compute is unguarded and only loads and stores
    are guarded. Padded rows add exactly ±0 to every sum for finite values, so results
    are unchanged; the tests at d_k 7 and 20 confirm it;
  - put a compiler-only memory fence (`sched_fence`) every 16 steps.

**2. The register-resident chunk-state kernel still spilled** (ScratchSize 2744 B/lane).
It now keeps each thread's state column in LDS; see "Known performance debt".

## Known performance debt

Each item is a deliberate correctness-first choice. None is measured; the cost is unknown
until the EVO-X2 runs.

1. **Chunked GDN state kernel keeps S in LDS, not registers.** (Recorded by the
   orchestrator.)
   - Each thread's state column sits in a thread-private 32 KiB LDS tile, with the block
     ≤ 64. The register-resident version spilled 2744 B/lane.
   - Cost: an estimated 2 waves/SIMD, and S is reloaded from LDS in every loop.
   - Affects prefill throughput. It is the first thing to tune on hardware, with
     cooperative LDS tiling of the chunk update.
2. **GEMV decodes each element's block header and scale again.** `wq_elem` decodes the
   f16 d/dmin and the 6-bit scale per element.
   - The wave variant's 8-element groups let the compiler share some of that work.
   - A tuned path would decode once per 32-element sub-block and use packed loads.
   - The generic variant exists for exactness and is not meant to be fast.
3. **No wave64 variant.** RDNA3.5 wave32 vs wave64 was not compared. Only wave32 variants
   are registered.
4. **TOP_K does a full bitonic sort per chunk.** At k = 1024 a 248,320-logit vector takes 8
   rounds (3 at k = 40). Each chunk uses O(C log² C) compare-exchanges per
   2048-element chunk, and all k survivors are kept every round.
   - A radix-select or threshold pre-pass would do less work.
   - Its cost against the ~1 GB LM-head read is expected to be small, but is unmeasured.
5. **The argmax reduce is one workgroup per vector.** The partials number
   ⌈rows / rows-per-workgroup⌉, about 62K for the 248,320-row head with `gemv_wave32_r4`.
   - One workgroup reads all 62K × 16 B. That is fine for correctness but serial.
   - A second partial stage would parallelize it.
6. **Attention decode is one workgroup per (token, head) with no split-K.**
   - For decode that is 24 workgroups over the whole history.
   - V rows are re-addressed for every output dimension.
   - A flash-decoding split over key ranges, with a combine step, and LDS-staged K/V tiles
     are the tuning path.
   - The exact variant is serial by design and is not for production.

## M6 decode primitives

| Op | Kernel | Variant | Contract | Accepted by |
|---|---|---|---|---|
| KV write | `k_kv_write` | `kv_write_b256` | `kv_cache::SequenceKv::write`: K and V rows at positions start … start+T−1 of `layer`, through the block table, into the same pool layout attention reads. Blocks must be exclusively owned; copy-on-write stays in `SequenceKv::reserve`. A bad block id sets `kStatusBadBlock` and that row is not written. | the **whole pool** byte-identical to `SequenceKv::write` on the same pool |
| GET_ROWS | `k_get_rows` | `get_rows_b256` | `out[t] = dequantize_row(W[ids[t]])` for every GEMV weight type; the pack's `token_embd` is Q4_K. An id outside [0, n_rows) sets `kStatusBadIndex`. | bitwise vs `tensor::dequantize_row` (pure dequant, so bitwise on the device too) |
| ADD | `k_eltwise` | `add_b256` | `cpu::add`; out may alias either input | bitwise |
| ADD + RMS_NORM | `k_add_norm` | `add_rms_norm_b128` | `h = a + b` (h may alias a or b), then `y = rms_norm(h)·w`, where the norm phases read the rounded h after a barrier | bitwise vs `cpu::add` then `cpu::rms_norm` |

**Q4_0 decision.**
- D-006 requires both MTP packagings to load, and D-014 keeps "ggml-org Q4_K_M plus a
  separate MTP file" supported and tested.
- That pack's `blk.64` weights are Q4_0 (D-007). So the runtime does load a Q4_0 file,
  and Q4_0 GEMV is in scope.
- halo::tensor dequantizes Q4_0 (`deq_q4_0`; D-007's list), so there was no blocker.
- `wq_elem` reproduces its `fl(q − 8) · d` exactly. GEMV and GET_ROWS accept Q4_0.

## One qwen35 decode step: ops and HIP status

These are the ops of the D-004/D-005 layer semantics at T = 1 (and T = K for MTP verify),
with the HIP op that covers each. "Emulated" means compiled for gfx1151 and verified in
emulation against halo::cpu / halo::tensor, but never executed on a device. That applies
to every row marked covered.

| Step | Op(s) | HIP status |
|---|---|---|
| token embedding | GET_ROWS (Q4_K) | covered, M6 |
| per layer: `attn_norm` | RMS_NORM, or the fused ADD + RMS_NORM with the previous residual | covered, M3 / M6 |
| **GDN layer** (48 per step): `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha` projections | QUANT_GEMV (Q5_K, Q4_K, IQ4_NL, Q8_0, …) | covered, M2 / M5 |
| conv1d over qkv + SiLU, conv state + slots | CONV1D_SHORT | covered, M1 |
| `beta = sigmoid(b)` | SIGMOID | **missing** (48 floats per token) |
| `g = ssm_a · softplus(a + dt_bias)` | ADD, SOFTPLUS, MUL (ssm_a per head) | ADD covered; **SOFTPLUS and MUL missing** (48 floats) |
| q/k L2 norm, q scale, delta rule, state + slots | GATED_DELTANET recurrent (D-016 in-kernel) | covered, M1 |
| `o = rmsnorm(o; ssm_norm) · silu(z)` | GATED_NORM | covered, M1 |
| `ssm_out` | QUANT_GEMV | covered |
| residual | ADD, or fused into the next ADD + RMS_NORM | covered, M6 |
| **Attention layer** (16 per step): `attn_q` (interleaved [Q \| gate]), `attn_k`, `attn_v` | QUANT_GEMV | covered |
| q/k per-head norm | RMS_NORM with per-head row views (row stride 512 on the interleaved q) | covered, M3 |
| partial RoPE, 64 of 256 dims, θ 10⁷ | PARTIAL_ROPE (head_stride 512 on q) | covered, M3 / M4 |
| append K/V | KV write | covered, M6 |
| GQA attention over the paged cache | ATTENTION (q_head_stride 512) | covered, M4 |
| `attn · sigmoid(gate)` | MUL_SIGMOID with row views (gate stride 512) | covered, M3 |
| `attn_output`, residual | QUANT_GEMV, ADD | covered |
| **FFN:** `post_attention_norm` | fused ADD + RMS_NORM | covered, M6 |
| `ffn_gate`, `ffn_up`, SwiGLU, `ffn_down`, residual | QUANT_GEMV (IQ4_XS, Q4_K, Q5_K, Q3_K, IQ3_S, …), SWIGLU, ADD | covered, M2 / M3 / M5 |
| **Head:** `output_norm` | RMS_NORM | covered |
| greedy | LM head GEMV (Q6_K) + fused argmax | covered, M3 |
| sampling | TOP_K (k ≤ 1024), then the CPU sampler | covered, M3 |
| **MTP (blk.64, D-005):** `enorm(embed)` and `hnorm(h)` into the two halves of one buffer | GET_ROWS + RMS_NORM into offset views (no CONCAT op is needed) | covered |
| `eh_proj`, then one attention block + FFN, `shared_head_norm`, the shared LM head | the rows above; Q4_0 for the ggml-org MTP file | covered (Q4_0: M6) |
| **Prefill** (T ≫ 1) | chunked GDN (M1); GEMV with n_vec = T; attention with T > 1 | functional; **no GEMM or prefill-attention kernel** (performance) |

The decode step is covered except for three trivial element-wise ops on 48-float vectors
(SIGMOID, SOFTPLUS, MUL). A fused GDN-gate kernel would be the natural shape for them.
Wiring these ops into a model forward or a `Backend` implementation is deliberately left
out of `backends/hip`. It is to be designed once, by one owner, after the Engine lands.

## TRD §63 acceptance status

| Item | Status |
|---|---|
| gfx1151 (or logged gfx11-generic) build succeeds | **Met on the dev host (compile).** gfx1151 by default. A generic ISA needs explicit opt-in and is logged. |
| Device discovery succeeds; tier probing succeeds | **Discovery implemented, unverified on a device.** On the dev host it fails cleanly with `hipErrorNoDevice`. Tier probing belongs to the hardware module and is not part of this backend. |
| GPU allocations succeed on GTT and carveout | **Implemented** (`HostPinned` = GTT, `Device` = carveout). The test exists but is unverified (skipped). |
| Core operators execute; custom GDN and GEMV kernels exist | **The kernels exist** and are compiled and emulation-verified, but have not executed on a device. They cover: GDN (recurrent and chunked), conv1d, gated and plain RMS norm, GEMV (F32, F16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_NL, IQ4_XS), GET_ROWS, KV write, ADD and ADD + RMS_NORM, LM head with fused argmax, argmax, top-k, RoPE (with head stride), SwiGLU, sigmoid gate, and GQA attention over the paged KV cache. |
| Reference-vs-HIP correctness passes | **Emulation passes.** It is bit for bit everywhere except two variants that sum in a different order, each within its own derived bound: the wave-GEMV logits (whose argmax index still equals the CPU's) and online attention. **Device: unverified.** |
| Benchmark suite executes reproducibly | **Not started.** |

## Not done (end of the WS-K plan)

- **Device execution of anything.** D-001. The first EVO-X2 run is also the first
  ROCm 7.1 → HIP 7.15 compatibility check.
- **GEMV:**
  - a prefill GEMM;
  - a wave64 comparison;
  - the remaining D-007 dequant types (Q4_1, Q5_0, Q5_1, Q2_K, BF16). None is in either pack.
    Q4_0 was added in M6.
- **Attention and kernels:**
  - a prefill (T ≫ 1) attention variant;
  - a split-K decode variant (debt 6);
  - the KV-cache write and append kernels, and fp16 KV storage (TRD §15; the CPU pool is fp32).
- **Runtime pieces:**
  - `hip_graph`;
  - the profiler;
  - the benchmark suite (TRD §63);
  - a `Backend`-interface adapter wiring these ops into `runtime` (other workstreams'
    modules).
- **Performance work:** see "Known performance debt".
