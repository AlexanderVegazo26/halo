# HALO HIP backend (WS-K)

Status: **milestone M1 of 3** (host runtime + GATED_DELTANET family). M2 (quantized GEMV)
and M3 (LM head, RMSNorm / RoPE / SwiGLU / sigmoid gate) are not built yet; see
"Not done" below.

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

    HALO_BUILD_DIR=/root/halo-build-wsk bash scripts/build.sh "-DHALO_BUILD_HIP=ON" "-DHALO_ONLY=tensor;cpu;hip"

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

## Operators (M1)

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
- **Device tests.** Each scenario above also exists as a `…Device…` test. It compares
  against `halo::cpu` within `|err| ≤ 2e-5 + 1e-5·|ref|`, which is an unmeasured choice
  to be confirmed on the EVO-X2. They skip on the dev host.

### Running the device tests on the EVO-X2

    cmake -S . -B build-hip -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
          -DHALO_BUILD_HIP=ON -DHALO_ONLY="tensor;cpu;hip"
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
- It now keeps each thread's state column in a thread-private LDS tile instead: 32 KiB,
  with the block ≤ 64.
- That costs occupancy (an estimated 2 waves/SIMD) and reloads S from LDS in every loop.
- It is the first thing to tune on hardware, and is recorded as performance debt.

## TRD §63 acceptance status

| Item | Status |
|---|---|
| gfx1151 (or logged gfx11-generic) build succeeds | **Met on the dev host (compile).** gfx1151 by default. A generic ISA needs explicit opt-in and is logged. |
| Device discovery succeeds; tier probing succeeds | **Discovery implemented, unverified on a device.** On the dev host it fails cleanly with `hipErrorNoDevice`. Tier probing belongs to the hardware module and is not part of this backend. |
| GPU allocations succeed on GTT and carveout | **Implemented** (`HostPinned` = GTT, `Device` = carveout). The test exists but is unverified (skipped). |
| Core operators execute; custom GDN and GEMV kernels exist | **GDN kernels exist**: compiled, emulation-verified, not executed. **GEMV: not yet (M2).** |
| Reference-vs-HIP correctness passes | **Emulation: passes bit for bit. Device: unverified.** |
| Benchmark suite executes reproducibly | **Not started.** |

## Not done (M1 scope boundary)

- **M2:** quantized GEMV (Q4_K, Q5_K, Q6_K, Q8_0, F16, F32) plus a tuned variant.
- **M3:**
  - LM head: logits GEMV, fused argmax with NaN → `Error(Kernel)`, and top-k ≤ 1024;
  - plain RMS_NORM exposed as an op (the kernel body already supports it);
  - partial NeoX RoPE, SwiGLU, sigmoid gate.
- **Later:** attention over paged KV, `hip_graph`, the profiler, a tuned chunked GDN
  (cooperative LDS tiling) and a register-resident chunk state.
