# ADR-001 — The Backend interface: one qwen35 forward over CPU, HIP and Vulkan

| | |
|---|---|
| Status | **Proposed** (2026-09-25). Needs the owner's answers in §8 before WS-BI-2 starts. |
| Decider | Project owner. Author: solution-architect. |
| Tier | 3 (structural: a cross-cutting contract that is expensive to reverse once three backends build against it). |
| Supersedes on acceptance | The `Backend` block in `docs/dev/ARCHITECTURE.md` ("Key interfaces → Backend", including the contract "`forward` must equal composing `run_op`"). It also resolves the deferred item "variants()/run_op() interface" in `include/halo/models/qwen35.h`. |
| Amends on acceptance | TD-1 (closed by WS-BI-2) and TD-2 (closed for GDN/KV by WS-BI-2). D-012's memory statement is clarified in §5.3. The owner should add a DECISIONS.md pointer (D-017 → this ADR). This author could not edit that file. |
| Evidence base | Only headers, sources and docs at HEAD of `halo-v0.2` (7cf4238) were read. Nothing was built or run for this ADR. No number here is a measurement (D-001). |

Epistemic labels used below:
- **[Fact]**: read from the code or docs, with a citation.
- **[Required]**: forced by a decision, constraint or correctness.
- **[Consensus]**: strong industry consensus.
- **[Convention]**: matches existing HALO patterns.
- **[Preference]**: the author's judgement, open to override.
- **A-n**: a numbered assumption.

---

## 1. Context

### 1.1 What exists

- **CPU reference (frozen contract).** `include/halo/backends/cpu/ops.h` holds free functions over `RowsView<float>` / `PagedRows` / `WeightMatrix`.
  - fp32 only, fixed-order reductions, bit-identical for any thread count, `-ffp-contract=off`.
  - This is the semantic definition of every op (TRD §30, D-016). [Fact]
- **HIP backend.** `include/halo/backends/hip/ops.h` has an `Ops` class of typed methods, one `XxxArgs` struct per op.
  - Operands are `BufferView{buffer, offset, bytes, row_stride}`.
  - Each op runs on a `Target`: either a device `Stream` or the host emulation, which is bit-identical to `halo::cpu` for the documented variants.
  - Device-detected data errors are written to 32-bit status words. `check_status` / `read_argmax` turn them into `Error(Kernel)`.
  - A static kernel registry (`registry.h`) names variants per op. `Ops` picks one per op at construction (`OpsOptions`).
  - Every op of one qwen35 decode step is covered, in emulation only (docs/hip.md, "One qwen35 decode step"). [Fact]
- **Vulkan backend.** `include/halo/backends/vulkan/ops.h` has the same `BufferView` concept. It records into a `Stream` (command buffer + fence) and submits explicitly.
  - Only RMS_NORM, matvec (6 types), recurrent GDN (with slots) and ARGMAX exist.
  - `rms_norm` rejects in-place.
  - It has no named variant registry: workgroup sizes are spec constants.
  - Device loss surfaces as `Error(Backend)` from `Stream::wait`.
  - docs/vulkan.md lists 14 missing ops. [Fact]
- **Model forward.** `src/models/qwen35.cpp` calls `halo::cpu` directly. It is batch-shaped (`SeqStep` / `MtpStep`), validates everything and reserves KV before touching state, then runs the trunk.
  - It commits KV and marks slots **before** the LM head. The head then raises `Error(Kernel)` on NaN. This ordering is TD-2.
  - It de-interleaves `[Q | gate]` by copy, because CPU views cannot express a head stride (review S-2). [Fact]
- **State.**
  - `KvPool` owns host `float[]` blocks.
  - `GdnState` owns a host live state plus `max_slots` slots per layer.
  - `commit_rows_kept` copies the chosen slot into the live state.
  - The slot contract writes slot 0, which duplicates the in-place final state.
  - Checkpoints are host `std::vector<float>` snapshots. [Fact]
- **Autotune.** `TunableOp` / `Candidate` hold named **int64** parameters. `select_kernel` implements TRD §56 steps 1–3.
  - The engine consumes CPU `threads` and GDN `chunk` winners at creation (8065734). [Fact]

### 1.2 The problem

The same qwen35 forward has to run on three backends. Two of them are asynchronous devices, the CPU must stay a bit-exact oracle, and decode is memory-bound. The TD-1 slot waste and the TD-2 non-atomic failure must not be carried into the GPU backends.

### 1.3 Parallel work in the tree

At the time of writing, other agents are editing `src/runtime`, `src/api`, `tools/halo` and `src/profiling`. The sequencing in §9 is built around that. [Fact: brief + git status]

---

## 2. Decision drivers

| # | Driver | Source | Class |
|---|---|---|---|
| DR-1 | Decode is memory-bound. The ceiling is `tok/s ≤ S·τ·η·BW / B_step`. The NFR is η ≥ 0.80 at S=1, n=0, 4K context. Everything not reading weights, launch gaps included, spends η. | D-011, D-014 | Required |
| DR-2 | The CPU reference stays the bit-exact oracle. GPU kernels are accepted only by differential tests against `halo::cpu` on the same buffers. | TRD §30, D-016 | Required |
| DR-3 | One model-math definition. The D-016 class of bug, where backends disagree on what an op means, must be structurally prevented. | D-016, review M-1 | Required |
| DR-4 | Kernel selection is an input from the planner/autotuner, never backend-private state. Variants are enumerable. | TRD §11/§56/§58/§64; review 2026-09-23 §2a | Required |
| DR-5 | Rollback: select a slot, with no restore copy and no redundant slot. Extra traffic ≤ (K−1)·0.157 GB per verify, and no extra weight passes. | D-012, D-014, TD-1 | Required |
| DR-6 | Failure atomicity. An op or step failure must not leave a committed-but-corrupt sequence. Also, one sequence's data error should not fail co-batched sequences. | TD-2, review 2026-09-25 R-1 | Required |
| DR-7 | Weights are mmapped once. Placement is tier-discovery-driven: 96 GiB carveout on the target, with GTT pages counted against the 32 GiB OS pool. | D-002, planner.h | Required |
| DR-8 | The migration must keep the existing tests and goldens passing unchanged. The exceptions are the deliberate contract tightenings listed in §6.4. | brief | Required |
| DR-9 | HIP graphs and Vulkan command-buffer reuse must stay possible without redesigning the model code. | TRD §17, §37 | Required (possibility), Preference (timing) |
| DR-10 | No silent fallback when behaviour would change. Errors are typed. | TRD §45 | Required |
| DR-11 | Simplicity. No IR/compiler layer before profiling justifies it. | TRD §54 phase 1 | Required |

### Assumptions

- **A-1.** The GPU cost of the model's work is dominated by GEMV weight reads. Per-op launch/dispatch gaps are the main *interface-shaped* overhead. *Risk if wrong:* the granularity trade-off in §4 shifts toward fusion earlier.
- **A-2.** The host emulation of HIP is bit-identical to `halo::cpu` for `gemv_generic_b64`, `attn_exact_b128`, and every op docs/hip.md marks bitwise. The claim is documented and emulation-tested [Fact], but not re-run here. *Risk if wrong:* the WS-BI-3 bitwise forward test fails for a reason outside the adapter.
- **A-3.** Per-(token, head) computations of RoPE, per-head RMSNorm, attention and `mul_sigmoid` depend only on that head's data. Running them per head, with strided views, is therefore bit-identical to the dense-copy path the forward uses today. **Believed** from the op definitions. WS-BI-1's gate test verifies it.
- **A-4.** On Strix Halo, GTT / `HostPinned` buffers are host-mapped. Host mapping of carveout (VRAM) buffers through a Vulkan host-visible device-local heap, with a 96 GiB carveout, is **unverified**. The design therefore never *requires* host mapping of a GPU buffer.
- **A-5.** A launch on gfx1151 costs a per-kernel host time t_launch plus a GPU-side inter-kernel gap. Both are **unmeasured** (D-001).

---

## 3. Options considered

Operation count per decode step, derived from D-004/D-005 and the docs/hip.md op table at S=1, T=1 (the residual add is fused into the next ADD+RMS_NORM):

| Layer type | Ops per layer |
|---|---|
| GDN layer | add_rms_norm, 4 GEMV (qkv, gate, beta, alpha), conv1d, gdn_gates, GATED_DELTANET, gated_norm, GEMV out, add_rms_norm, 2 GEMV (ffn gate/up), SwiGLU, GEMV down = **16** |
| Attention layer | add_rms_norm, 3 GEMV, 2 RMS_NORM (q/k heads), 2 RoPE, kv_write, attention, mul_sigmoid, GEMV o, add_rms_norm, 4 FFN = **18** |

Total: 48·16 + 16·18 + embedding + final norm + LM head (2) ≈ **1 060 kernel launches per trunk step**. Per-sequence ops (conv, GDN, attention, kv_write) scale with S. Each MTP draft depth adds about 20.

Step time at the D-011 ceiling, **computed, not measured**:

    T_step ≈ (16.48 + 1.04) GB / 256 GB/s ≈ 68 ms

Launch share of the step:

    launch_share = N_ops · t_gap / T_step

Here t_gap is the non-overlapped per-kernel cost. It is unknown (A-5). The η ≥ 0.80 NFR leaves 20% of T_step for everything that is not weight streaming. Launches would eat **half** of that budget, about 6.8 ms, at t_gap ≈ **6.4 µs**, and all of it at about 12.9 µs. Whether real gaps are above or below this is exactly what the first EVO-X2 run must measure. This ADR takes no position on it.

### Option A — Per-op dispatch through a typed Backend interface, asynchronous enqueue, one submit per step

One `Qwen35::forward` is written against an abstract `Backend` whose methods are the TRD §9 ops (typed args structs over a neutral `TensorRef`).
- Each call **enqueues** onto a backend `Stream`: HIP launches onto a `hipStream_t`, Vulkan records into a command buffer, and the CPU executes synchronously.
- A step ends with one `submit` and **one host sync** where the host needs results (argmax words + status words).

| | |
|---|---|
| **+** | One model definition (DR-3). |
| **+** | The oracle is exact: the CPU adapter calls exactly today's `halo::cpu` functions (DR-2). |
| **+** | Per-op differential tests and autotuning come for free: every op is individually addressable (DR-4). |
| **+** | Maps 1:1 onto HIP's `Ops` and Vulkan's `Ops`, which already use this shape [Fact]. |
| **+** | A virtual call costs nanoseconds. That is noise next to a kernel launch. The real cost is launches, which are the same under Option C unless C fuses. |
| **−** | ~1 060 launches per step, unfused except where the interface defines fused ops. |
| **−** | Graph replay needs extra work (see Option B). |

### Option B — Recorded op graph (an IR), compiled per backend and replayed

The forward emits a graph of nodes once per shape bucket. The backend compiles it into a HIP graph or a pre-recorded Vulkan command buffer and replays it each step, patching dynamic parameters.

| | |
|---|---|
| **+** | Lowest launch overhead (DR-1). |
| **+** | Enables whole-graph fusion passes later (TRD §54 phase 2). |
| **−** | Needs an IR, a node/tensor lifetime model and a parameter-patching scheme now, before any hardware measurement says launches matter (DR-11). |
| **−** | Graph replay needs *stable addresses*. Today's per-step dynamic values (`q_offset`, KV `start`, block tables, GDN slot targets) are kernel scalars or change addresses every step [Fact: `AttentionArgs.q_offset`, `KvWriteArgs.start`]. That forces "indirect operand" kernel variants across HIP and Vulkan. |
| **−** | The CPU oracle becomes a graph interpreter, one more layer between the model and `halo::cpu`. |

### Option C — Each backend implements the whole layer step (or the whole `forward`)

`Backend::forward(StepPlan, steps, result)` is implemented three times. This is ARCHITECTURE.md's current sketch, used as the primary path.

| | |
|---|---|
| **+** | Maximum freedom per backend to fuse and schedule. The fewest launches possible. |
| **−** | Three copies of the model math. That is exactly the failure D-016 records (DR-3). |
| **−** | The differential test can only compare whole forwards, so a divergence is found at the logits rather than at the op. |
| **−** | Every model change (MTP draft chain, R-5b, fp16 KV) lands three times. |
| **−** | Per-op backend mixing (TRD v0.7) has no seam (review §2a). |

### Option D — Type-erased single entry point (`run_op(const OpInvocation&)`) instead of typed methods

This is a variant of A: one virtual taking a `std::variant` of all args structs.

| | |
|---|---|
| **+** | Uniform for the tuner, tracing and mixing. |
| **−** | Loses compile-time checking of the forward's calls. |
| **−** | Every backend re-implements the dispatch `switch`. |

It is kept as a thin non-virtual helper over A (§5.1), not as the primary API.

### Comparison

| Criterion | A | B | C |
|---|---|---|---|
| One model definition (DR-3) | yes | yes | **no** |
| Bit-exact CPU oracle, op-level localisation (DR-2) | yes | yes (interpreter) | whole-forward only |
| Launch overhead (DR-1) | ~1 060 launches per step. Graph replay can be added later (§5.7) | lowest | lowest possible |
| Complexity now (DR-11) | low: adapters over existing `Ops` | high: IR, patching, indirect kernels | high ×3 |
| Graph/command-buffer reuse possible (DR-9) | yes, additive, backend-internal | built in | yes |
| Autotune seam (DR-4) | per op, natural | per node | backend-private |

---

## 4. Decision

**Option A**: a typed, per-op, **asynchronous-enqueue** `halo::backend::Backend` interface. The shared `Qwen35::forward` / `mtp_forward` is composed over it. [Preference, grounded in DR-1..DR-11]

The four properties that make Option A cheap on a memory-bound decode:
1. **Ops enqueue and never synchronise.** A step does one submit and syncs only at the host sync points listed in §5.5.
2. **Fused ops are first-class interface ops.** Each has a normative CPU reference *composition*, so fusion never forks the model math. Examples: ADD+RMS_NORM, GDN gates, LM-head GEMV+argmax.
3. **Dynamic per-step values travel in device buffers** wherever an op already allows it. Every new op contract must prefer that. This keeps HIP stream capture and Vulkan command-buffer reuse an additive, backend-internal optimisation (§5.7), to be decided on EVO-X2 measurements.
4. **Kernel selection is an argument**, a `KernelChoice` per op call taken from a per-backend `KernelPlan`. It is never backend state.

The backend-level `forward` virtual in ARCHITECTURE.md is **dropped**. There is exactly one forward. The ARCHITECTURE.md contract "forward == composed run_op" therefore holds by construction and needs no test. How the four properties of the 2026-09-23 review §2a survive:

| Review §2a property | How it holds here |
|---|---|
| Selection is an input, never backend-private | `KernelChoice` in every op call, from `KernelPlan` (§5.6) |
| `run_op` exists from day one | Every typed op *is* a run_op. `run_op(Stream&, const OpInvocation&)` is a non-virtual visitor for tests, the tuner and future mixing |
| `forward` is batch-shaped | `Qwen35::forward(span<const SeqStep>)` is unchanged |
| `n_state_slots` per sequence | `SeqStep::n_state_slots` is unchanged, and the slot targets become a ring (§5.3) |

A per-backend fused fast path (for example a whole-layer HIP kernel) stays possible later as an **additive** virtual whose default implementation is the composed ops. It is accepted only by the same differential test.

---

## 5. Design

### 5.1 Interface shape (illustrative sketch; names are proposals)

```cpp
// include/halo/backend/backend.h  (new module halo_backend; depends on core, tensor only)
namespace halo::backend {

enum class Kind : std::uint8_t { Cpu, Hip, HipEmulation, Vulkan };   // HipEmulation != Hip (§5.6)
enum class Tier : std::uint8_t { Vram, Gtt, Host };                    // planner's MemoryTier mapping

class Buffer {                        // owned via unique_ptr from Backend::allocate / import_host
 public:
  virtual ~Buffer() = default;
  virtual std::uint64_t bytes() const noexcept = 0;
  virtual Tier tier() const noexcept = 0;
  virtual std::byte* host_ptr() const noexcept = 0;   // non-null ONLY if host-addressable (A-4)
};

struct TensorRef {                    // = hip::BufferView / vulkan::BufferView, backend-neutral
  const Buffer* buffer = nullptr;
  std::uint64_t offset = 0, bytes = 0, row_stride = 0;   // bytes; 0 = dense / to end
};

struct StatusRef { std::uint32_t word = 0; std::uint32_t owner = kBatchWide; }; // §5.5
struct KernelChoice { std::uint32_t variant_id = 0; };   // 0 = backend default (§5.6)

struct Limits {                       // checked by Qwen35 at construction -> Error(Unsupported)
  std::uint32_t max_gdn_dk, max_gdn_chunk, max_head_dim, max_conv_k, max_top_k, max_rope_dims;
  bool gdn_chunked;                   // Vulkan: false until its chunked kernel lands
};

// Ring of P physical GDN (or conv) states per layer, one integer of per-sequence state (§5.3).
struct StateRing {
  TensorRef slab;                     // P states back to back for this layer
  std::uint32_t P = 0, live = 0;      // read from slab[live]
  std::uint32_t n_slots = 0;          // final state (slot 0) ALWAYS -> slab[(live+1) % P];
                                      // rollback slots 1..min(T,n_slots)-1 -> slab[(live+1+s) % P];
                                      // requires max(1, min(T, n_slots)) <= P-1 (T itself is unbounded)
};

// Args structs: HIP's (hip/ops.h) generalised over TensorRef + KernelChoice + StatusRef.
struct GemvArgs;  struct LmHeadArgs;  struct TopKArgs;  struct ArgmaxArgs;
struct RmsNormArgs;  struct AddRmsNormArgs;  struct GatedNormArgs;  struct RopeArgs;
struct GetRowsArgs;  struct Conv1dArgs;      struct GdnGateArgs;    struct GdnArgs;  // form: Recurrent|Chunked
struct KvWriteArgs;  struct AttentionArgs;   struct EltwiseArgs;    struct CopyArgs;

class Stream {                        // one per engine worker; single-threaded object
 public:
  virtual ~Stream() = default;
  virtual void submit() = 0;          // CPU: no-op
  virtual void wait() = 0;            // throws the translated device error (§5.5)
  virtual void abort() noexcept = 0;  // drain (HIP) / discard recording (Vulkan); see §5.5
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual Kind kind() const noexcept = 0;
  virtual std::string describe() const = 0;                 // device, driver, ISA -> profile key
  virtual Limits limits() const noexcept = 0;

  // ---- memory ------------------------------------------------------------------------
  virtual std::unique_ptr<Buffer> allocate(std::uint64_t bytes, Tier) = 0;       // Error(Memory)
  virtual std::unique_ptr<Buffer> import_host(std::span<std::byte>) = 0;         // CPU: zero-copy;
                                                                                  // GPU: Unsupported or copy (§5.2)
  virtual std::unique_ptr<Stream> create_stream() = 0;
  virtual void upload(Stream&, TensorRef dst, std::span<const std::byte>) = 0;   // ordered in stream
  virtual void download(Stream&, TensorRef src, std::span<std::byte>) = 0;       // valid after wait()

  // ---- registry (§5.6) ---------------------------------------------------------------
  virtual std::span<const VariantInfo> variants(OpId, std::string_view form = {}) const = 0;

  // ---- ops: validate on host (throw before enqueuing), then enqueue ------------------
  virtual void get_rows(Stream&, const GetRowsArgs&) = 0;
  virtual void rms_norm(Stream&, const RmsNormArgs&) = 0;
  virtual void add_rms_norm(Stream&, const AddRmsNormArgs&) = 0;   // CPU ref = cpu::add; cpu::rms_norm
  virtual void gemv(Stream&, const GemvArgs&) = 0;                 // T = n_vec rows (decode/verify/prefill)
  virtual void gdn_gates(Stream&, const GdnGateArgs&) = 0;         // CPU ref = qwen35.cpp's exact sequence
  virtual void conv1d_silu(Stream&, const Conv1dArgs&) = 0;        // StateRing
  virtual void gated_delta_rule(Stream&, const GdnArgs&) = 0;      // StateRing, D-016 fields explicit
  virtual void gated_rms_norm(Stream&, const GatedNormArgs&) = 0;
  virtual void partial_rope(Stream&, const RopeArgs&) = 0;         // head_stride (TD-9)
  virtual void kv_write(Stream&, const KvWriteArgs&) = 0;
  virtual void attention(Stream&, const AttentionArgs&) = 0;       // q_head_stride
  virtual void swiglu(Stream&, const EltwiseArgs&) = 0;
  virtual void mul_sigmoid(Stream&, const EltwiseArgs&) = 0;
  virtual void add(Stream&, const EltwiseArgs&) = 0;
  virtual void lm_head(Stream&, const LmHeadArgs&) = 0;            // GEMV + fused argmax, optional logits
  virtual void argmax(Stream&, const ArgmaxArgs&) = 0;
  virtual void top_k(Stream&, const TopKArgs&) = 0;
  virtual void copy(Stream&, const CopyArgs&) = 0;                 // checkpoints, KV COW, concat halves

  // Non-virtual: std::visit over OpInvocation = std::variant<GemvArgs, ...> -> typed method.
  void run_op(Stream&, const OpInvocation&);
};

std::unique_ptr<Backend> make_cpu_backend(cpu::ThreadPool*);                      // WS-BI-1
std::unique_ptr<Backend> make_hip_backend(const HipBackendOptions&);              // WS-BI-3 (emulation|device)
std::unique_ptr<Backend> make_vulkan_backend(std::shared_ptr<vulkan::Context>);   // WS-BI-5
}  // namespace halo::backend
```

Notes on the sketch:
- **Typed virtuals.** They are compile-time checked. The CPU adapter is about 20 small functions calling `halo::cpu`. The HIP adapter translates `TensorRef` → `hip::BufferView` and `KernelChoice` → a cached `hip::Ops` per variant. `hip::Ops` is immutable and holds only block sizes [Fact], so one instance per variant is cheap and `backends/hip` needs no change. [Preference]
- **Layering.** `models/qwen35` depends on `halo_backend` (interface) only, never on `backends/hip` or `backends/vulkan` (TRD §5). [Required]
- **Threading.** A GPU backend and its `Stream` are driven by one thread, the engine worker. The `qwen35.h` claim that forward may run concurrently on disjoint sequences stays true for the **CPU backend only**. The header comment must say so. [Required: hip::Stream and vulkan::Ops/Stream are single-thread objects, a Fact]

### 5.2 Memory and tensor ownership

**Ownership rule.** Every byte an op touches lives in a `backend::Buffer` owned (or imported) by the one `Backend` the model runs on. The Backend outlives all its buffers; this is the Vulkan `shared_ptr<Context>` pattern [Convention]. There are four arenas, sized by the memory planner (`memory::MemoryPlan`) and allocated through the backend:

| Arena | Contents | Lifetime | Placement (D-002 policy, planner decides) |
|---|---|---|---|
| **Weights** | Every GGUF tensor the forward reads, plus the small fp32 vectors (norms, `dt_bias`, `ssm_a`, conv taps) that today's `make_vec` dequantizes once | Engine | CPU: `import_host` over the mmapped GGUF bytes, zero-copy, byte-identical to today. GPU: copied once at load into VRAM when it fits, else GTT (§8 Q3) |
| **State** | KV pools (trunk, MTP), GDN rings, prefix checkpoints (D-013) | Engine | GPU pool only for checkpoints (D-013). GDN first, then KV, as planner.h already orders |
| **Step** | Activations, workspace (chunked GDN, argmax partials, top-k), per-step uploads (tokens, positions, block tables), status words, result words | One step; bump allocator reset after `wait()` | Fastest tier. The planner's `activations_override` / `workspace_override` get the measured high-water mark |
| **Host** | Results read back (argmax, top-k candidates, hidden rows for MTP), sampler state | Host | Host |

**How a `TensorRef` maps onto cpu views** (CPU adapter). [Required for DR-2]
- **Plain operands.** `host_ptr() + offset` gives the data pointer, with `row_stride / 4` as the element stride. The adapter builds `cpu::RowsView` from these after checking fp32 alignment. This is lossless: `RowsView` already has an explicit stride.
- **Head-strided operands** (RoPE and attention on the interleaved `[Q | gate]` row, and the `mul_sigmoid` gate). The adapter decomposes into one call per head over a `(T, head_dim, row_stride)` view:
  - `partial_rope_neox` with `n_heads = 1`;
  - `attention_gqa` with `{1, 1, hd}` dims, over a `PagedRows` block table offset by `kv_head · head_dim`.
  - This removes the forward's de-interleave copy without changing the frozen CPU contract. Bit-identity to today's path is **A-3**, and WS-BI-1's gate test proves it before the old path is deleted.
- **Per-head RMSNorm on q** needs no decomposition. With rows = T·n_head and row stride 512 floats, row r lands on the Q of token r/n_head, head r%n_head [Fact: docs/hip.md op table].
- **Weights.** The adapter builds a `cpu::WeightMatrix::dequantized` whose callback calls `tensor::dequantize_row` over the imported bytes. That is the same function over the same bytes as today's `models::weight_matrix` [Fact], so matmul results are unchanged.
- **Paged KV.** The adapter rebuilds `cpu::PagedRows` from the pool buffer and the host block table, exactly as `SequenceKv::keys()` does now.

**Neutral aliasing rule = the intersection of the three backends.** [Required]
- **Exactly two in-place cases are allowed:**
  - the residual accumulate `h = a + b` with `h ≡ a` (ADD, ADD+RMS_NORM);
  - PARTIAL_ROPE on `x`, which is in place by contract on CPU and HIP [Fact] and has no output operand.
  - GDN/conv state is out-of-place by construction (§5.3).
- **The forward writes every other output into a distinct step-arena buffer.** That includes q/k norm, conv output, gated_norm, SwiGLU and mul_sigmoid, all of which today's forward runs in place [Fact]. These ops are pure functions, so the values are identical. The only cost is a few activation-sized buffers.
- **Consequence for Vulkan.** It must support `h ≡ a` for ADD / ADD+RMS_NORM, and in-place RoPE (V2). It does not need in-place `rms_norm`, which it rejects today [Fact].
- **Operand limits** are the per-backend `Limits`:
  - HIP: d_k ≤ 128, chunk ≤ 64, head_dim ≤ 256, conv K ≤ 8, top-k ≤ 1024 [Fact].
  - `Qwen35` checks them at construction and at `KernelPlan` build, and raises `Error(Unsupported)` early rather than at op N.
  - The 27B model and the tiny model both fit HIP's limits. The tiny model is d_k 32, head_dim 64, conv 4, chunk 64 [Fact: `make_tiny_model.py`].

**KV pool.** `KvPool`'s storage becomes a State-arena `Buffer`. The layout `block[layer][K|V][token][kv_dim]` fp32 is unchanged, and HIP attention/kv_write already read it byte for byte [Fact].
- Block tables stay host-owned and are uploaded into the step arena per step, as uint32.
- Copy-on-write in `SequenceKv::reserve` enqueues `Backend::copy` as the step's first operation, in stream order, instead of `memcpy`. Its host bookkeeping stays strongly atomic.
- The host accessors `write`/`keys`/`values` remain for host-addressable pools, which is every CPU test. They throw `Error(Unsupported)` on a pool whose `host_ptr()` is null.

**Weights on GPU.** The default is **copy into the planner's tier at load** [Preference]. It is simple and it is the only option whose correctness is independent of A-4.
- Zero-copy import of the mmap into GTT would save a ~17 GB copy and 17 GB of VRAM. Its bandwidth, and whether `hipHostRegister` works on file-backed mappings, are **unverified**. It is an EVO-X2 measurement and an owner decision (§8 Q3).
- After a GPU upload, the mmap pages are not touched again. Whether to `madvise` them away to protect the 32 GiB OS pool is left to the implementer, and must be measured.

### 5.3 The D-012 slot and rollback contract on GPU backends: the TD-1 fix

**Mechanism: a state ring with out-of-place final state.** This applies per GDN layer, and the conv state uses the same ring.
- Each sequence holds P physical states in a slab, plus **one integer `live`** that is shared by all layers of the sequence.
- A call over T rows reads `slab[live]`.
  - It **always** writes the final state (logical slot 0) to `slab[(live+1) mod P]`. That includes plain decode and prefill chunks, where n_slots = 0.
  - Rollback slots s = 1 .. min(T, n_slots)−1 go to `slab[(live+1+s) mod P]`.
  - The constraint is **max(1, min(T, n_slots)) ≤ P−1**. T itself is unbounded, so a 64-row prefill chunk with P = 2 is fine.
- The input `slab[live]` is **never written**.

**Who advances `live`.** Today decode and prefill are in place, so they need no commit. Under the ring, every forward must advance `live`, or the state never moves.
- The forward records `base = live` for each sequence, and leaves `live` unchanged while the step is in flight.
- After a clean status (§5.5, step 4) the forward sets `live = (base + 1) mod P`, which keeps all rows. This is the whole commit for decode and prefill.
- For a verify, `commit_rows_kept(T, m)` then sets `live = (base + 1 + (T − m)) mod P`, from the **recorded base**. That is **one integer**: no copy, no device work.
- No forward may start on that sequence between the verify and its commit. `GdnState` enforces this: a forward on a sequence with an uncommitted verify is `Error(Api)`, as the slot bookkeeping does today.
- This is what keeps the existing "verify + rollback to r == decode r tokens" tests meaningful. Under the ring they compare `slab[live]` after both paths.

**Failure.** `live` is not changed, so the sequence is bit-for-bit at its pre-step state (DR-6).

**Accounting.** K = n_draft+1 rows per verify, and P = K+1.
- **Traffic per verify:** read 1 + write final 1 + write (K−1) slots. The extra over a plain decode is **(K−1)·state**, which meets DR-5.
  - Today it is K·state for the slots plus 2·state for the restore copy on partial accept [Fact: `test_speculative.cpp` asserts 4·state at K=4; `commit_bytes` = 2·state].
- **Memory:** P = K+1 physical states. That is **the same memory as today** (live + `max_slots` = n_draft+1) [Fact: gdn_state.h, engine.cpp `max_draft_ + 1`].
- **The one planner change:** with MTP off (K=1) the ring still needs P = 2. Today `gdn_rollback_copies` defaults to 0 when MTP is disabled [Fact: planner.h], so it must become max(1, draft+1). At S=4 on the 27B model that is +4 × ~150 MiB when MTP is off. That buys GDN failure atomicity on every decode step.
- **Why a ring index and not a pointer table:** it is one integer per sequence, the kernel's address arithmetic is trivial, and a future graph-capture variant can read `live` from a device word (§5.7). [Preference]
- **The rejected minimal-memory alternative:** in-place final state with P = K. It saves one state per sequence but loses GDN atomicity, so any failure means `reset()`. It remains available as a planner "memory-tight" mode if the owner wants it (§8 Q2).

**What each backend must add.** This is additive, not adapter-only. All three kernels write slots back-to-back from slot 0 and take one contiguous slots region [Fact].

| Backend | Additive change |
|---|---|
| CPU (frozen `ops.h`) | New overloads `gated_delta_rule_recurrent(dims, in, const StateRingView&, out, qk, pool)`, the same for `_chunked` and `causal_conv1d_silu`. `StateRingView = {span<float> slab, P, live, n_slots}`. The **existing overloads stay unchanged**, so today's callers and goldens are unaffected. This follows the D-016 bool-shim precedent: add, keep the old form, deprecate later. Normative: the ring overload is bit-identical to the old form given the same input state. |
| HIP | `GdnArgs` / `GdnChunkedArgs` / `Conv1dArgs` gain optional `ring {P, live}` fields. When they are set, `state`/`state_out`/`state_slots` are derived from one slab view. `state_out` already exists [Fact], so the kernel change is only the slot address computation. |
| Vulkan | `GdnDecodeArgs` gains the same field. `state_out` exists [Fact]. Conv1d is new work anyway. |

**Why this is correct.** `m ≥ 1` always holds: the fed token x was already emitted [Fact: speculative.h]. The pre-row-0 state is therefore never needed, and `live` itself is never a rollback target.

**KV and MTP KV rollback** stay host-only. The forward commits the logical length **after** the step's status check (§5.5), `truncate` is unchanged, and the MTP KV rolls back past `pos_max` as today.

**Checkpoints (D-013).** A snapshot enqueues `copy(slab[live] → checkpoint buffer)` for every layer. Checkpoint buffers come from the planner's GPU-pool budget, and are host-addressable on the CPU backend. A restore copies into `slab[live]` of the target sequence.
- This changes the `GdnSnapshot` type from `std::vector<float>` to a buffer handle. It touches `state/checkpoint.h` and `src/runtime/engine.cpp`, so it is sequenced last (WS-BI-6).
- Making snapshots best-effort, as review R-1/R-2 require, is the engine owner's call and is compatible with this design.

### 5.4 Operator composition rules (keeping the oracle exact)

- Each interface op names its **CPU reference as a composition of frozen `halo::cpu` / `halo::tensor` calls**:
  - `add_rms_norm` = `cpu::add` then `cpu::rms_norm`.
  - `gdn_gates` = today's `qwen35.cpp` sequence: `cpu::sigmoid`, the `dt_bias` add, `cpu::softplus`, then the `ssm_a` multiply.
  - `get_rows` = `tensor::dequantize_row` per id.
  - `lm_head` = slabbed `cpu::matmul` + NaN-checking argmax, exactly as `Impl::head`.
  - The CPU adapter *is* that composition. [Required]
- **Fused ops are the only way fusion enters.** A GPU fused kernel is accepted by a differential test against the composition. HIP's `add_rms_norm` and `gdn_gates` are already bitwise against theirs [Fact].
- **D-016 fields are always explicit** in the forward's `GdnArgs`: `qk_l2norm = true`, `q_scale = 1/sqrt(d_k)`. They are never defaulted.

### 5.5 Errors, status and atomicity

**Error mapping, identical on every backend.** [Required: DR-10; mapping per `hip::error_code_for`, Fact]

| Source | halo::Error | When raised |
|---|---|---|
| Host validation (shape, alias, range, alignment, limits, unknown variant) | `Kernel` (shape/alias), `Unsupported` (limits/dtype), `Config` (unknown variant id) | Synchronously in the op call, before anything is enqueued |
| Device status word `kStatusPositiveG` / `kStatusNaN` / `kStatusBadBlock` / `kStatusBadIndex` | `Kernel`, message naming op, layer, sequence and bits | At the step's host sync point |
| Argmax NaN word | `Kernel` (D-016) | At readback |
| Allocation failure | `Memory` | At allocate |
| Device lost / no device / driver failure | `Device`. The backend is then **poisoned**: every later call throws `Device` | At enqueue or `wait()`. Vulkan currently throws `Backend` for device loss [Fact], so the Vulkan owner must surface `VK_ERROR_DEVICE_LOST` distinctly |
| Illegal address / launch failure | `Kernel`, and the backend is poisoned (a wild write may have hit any buffer) | At `wait()` |
| Other runtime errors | `Backend` | At enqueue or `wait()` |

**Status words.**
- Each op call that can detect a data error gets its own word in the step arena's status array, tagged with an `owner`: a sequence index, or `kBatchWide` for batched GEMVs. There is no per-op sync.
- The array is downloaded together with the result words at the step's sync point.

**Status words on the CPU backend (the shipping path).** The CPU ops raise `Error(Kernel)` for shape and alias problems and for data errors (NaN in argmax/top_k, g > 0 in chunked GDN) alike [Fact], and CPU "enqueue" is synchronous. Without adapter work, a data error would therefore still abort the whole batch. The CPU adapter must produce the same owned status words as the GPU backends:
- It builds and validates every view (shape, alias, range) before calling the `halo::cpu` function. Any `Error(Kernel)` the function raises after that is by construction a data error.
- It catches that error and sets the op's status word with its owner, instead of propagating it.
- `lm_head` and `argmax` on the CPU record the NaN flag per row, which `Impl::head`'s per-row check already localises.
- Status is then evaluated at the step's sync point (step 4 below) exactly as on a GPU.

The per-sequence failure test in §6.5 runs on the **CPU backend** as well as on HipEmulation. Without it the TD-2 / R-1 closure claim would not hold for the only backend the engine runs today.

**Host sync points per tick** (the design adds none):
- one per MTP draft depth, because the next depth needs the argmax token;
- one at verify: argmax rows + status + hidden rows for MTP;
- one TOP_K/logits readback per step for sampled (non-greedy) requests.

D-011's ceiling does not count these; they belong in the cost model's `T_sync`. Keeping draft tokens on the device, so that GET_ROWS reads the argmax word directly, is a later measured optimisation.

**Atomicity, per op.**
- Every op validates all its operands before enqueuing anything. [Fact: already true of HIP and Vulkan]
- On the device, an op that sets a status word must leave its outputs either untouched or undefined-but-unreferenced. Today: GDN chunked writes nothing on positive g; attention's `out` is undefined on a bad block [Fact]. No op may write *committed* state.

**Atomicity, per step.** [Required: DR-6; this closes TD-2 for KV and GDN]
1. `forward` validates inputs and reserves KV (unchanged).
2. It enqueues all ops. KV rows are written past `length()`, and GDN/conv outputs go to ring slots other than `live`.
3. It submits and waits, then reads status and results.
4. **Only then** does it commit: `SequenceKv::commit(n)`, record the ring's pending outputs (`mark_slots_written`), and deliver results.
   - A status error owned by sequence i fails **only** sequence i, which stays at its pre-step state.
   - A batch-wide error or device error fails the step for all sequences. Nothing is committed, so every sequence is at its pre-step state. A poisoned backend additionally requires Engine recreation.
5. If an exception is thrown **mid-enqueue** (host validation at op N after N−1 ops are in flight), the forward calls `Stream::abort()` before rethrowing:
   - HIP: synchronize the stream, then release the step arena.
   - Vulkan: discard the unsubmitted recording.
   - CPU: nothing.
   - No state was committed, so the pre-step guarantee holds. The step arena is never reused before `abort()`/`wait()` returns.

The per-sequence failure split is what review R-1(c) asks for. The engine's handling of it (retire one, keep the batch) is an engine-side change in WS-BI-6.

### 5.6 Kernel variants, the autotuner and dispatch

**Registry.**
- Each backend's `variants(op, form)` returns `VariantInfo{id, name, op, form, params}`, where the **`id` is an explicit, append-only uint32**, never an array index.
- It is stored in the profile DB as the Candidate parameter `variant=<id>`. `Candidate` values are int64 [Fact].
- A registry test pins every (id → name) pair. Removing a variant retires its id forever. A stored winner whose id no longer exists is rejected with a note by `select_kernel`'s existing "can no longer run" path [Fact: lookup.h].
- Mappings:
  - HIP: add an `id` field to `hip::KernelVariant`. This is additive.
  - Vulkan: new, a small registry over its spec-constant workgroup sizes.
  - CPU: one `reference` variant (id 0) per op, with the existing `threads` / `chunk` parameters.

**TunableOps.**
- One generic `BackendTunableOp : autotune::TunableOp` per (backend, OpKey). It has four parts:
  - `candidates()`: the backend's variants × op parameters, filtered by `Limits`.
  - `run()`: enqueue the op with that `KernelChoice`, then submit and wait. `Measurement.device_ns` comes from backend timestamps (HIP events, Vulkan timestamp queries).
  - `validate()`: a differential against the **CPU backend on the same input bytes**, within the op's documented bound (bitwise where the variant claims bitwise).
  - `cost()`: from the `cpu/traffic.h` byte model.
- This replaces hand-written per-backend tunables. The existing `CpuMatmulTunable` / `CpuGdnChunkedTunable` keep working and can be re-expressed later. [Preference]
- **Emulation timings never persist as `hip`.** The backend label is `hip-emulation`, and it never matches a `hip` profile key. The tuner refuses to persist it unless `allow_nonconformant` is set, which is for unit tests only. [Required: D-001]

**Dispatch.**
- `KernelPlan` is built once at Engine creation, **per backend**.
- `Qwen35::op_keys(buckets)` enumerates every distinct (TRD §9 op, canonical shape, dtype) the forward will issue. Buckets:
  - T = 1 decode;
  - T = 2..K verify;
  - T = prefill chunk;
  - MTP T values.
- For each key, `autotune::select_kernel(lookup, key, op, backend_label, candidates, …)` picks the winner. The resulting `KernelChoice` is stored in the plan.
- The forward passes `plan.choice(op_key)` into every op's args. **Selection reaches dispatch as an argument**, and `EngineStats::tuning` reports source and winner per op, as it does today for CPU.
- The engine's current CPU consumption (threads → pool size, chunk → `Qwen35Options::gdn_chunk`) is preserved unchanged in WS-BI-1. It folds into `KernelPlan` in WS-BI-4.

### 5.7 HIP graphs and Vulkan command buffers (possible, deliberately not built now)

- The enqueue interface is what HIP stream capture records, and what a Vulkan recording contains. Replay is a backend-internal `StepCache` keyed by the shape bucket. It is additive and changes no model code.
- Replay needs every per-step dynamic value to come from device memory:
  - positions: already a buffer [Fact];
  - `q_offset`, KV `start`, `live`: scalars today, so replay needs "indirect" variants reading a step-params block;
  - block tables: already device buffers.
- The step arena needs fixed addresses, which a bump allocator reset each step already gives.
- **Decision on building it:** only after the first EVO-X2 run measures t_gap against the 6.4 µs / 12.9 µs break-evens in §3 (D-001). That is a future ADR. [Preference]

---

## 6. Testing

### 6.1 One forward over any backend

`Qwen35(const NormalizedModel&, backend::Backend&, const Qwen35Options&)` is the new constructor. The existing `Qwen35(model, cpu::ThreadPool*)` and `Qwen35(model, pool, options)` constructors stay. They build an internal CPU backend, so every existing call site compiles and behaves unchanged (DR-8). `SeqStep`, `MtpStep`, `StepResult` and `StepCost` are unchanged.

### 6.2 The migration gate: new forward over CPU == legacy forward, bitwise

In WS-BI-1, the current `qwen35.cpp` body is moved verbatim into a test-only `legacy_qwen35` (tests/unit/models). A new test compares, **bitwise**, on the tiny model:
- per-layer inputs (`capture_layer_inputs`), final hidden, argmax index and value, full logits rows;
- KV pool contents, GDN live state and every slot, after each call.

The fixture varies the dimensions the invariants are stated over:
- prefill crossing a GDN chunk boundary and a KV block boundary;
- S = 1 and S = 3 ragged batches;
- decode, and verify with K = 1..4 followed by commit for every m;
- a copy-on-write shared prefix;
- MTP catch-up plus a 2-deep draft chain;
- Full, Argmax and None logits modes.

It is shown to fail by mutation, one mutation at a time:
- swap the per-head RoPE decomposition offset;
- drop the `dt_bias` add in `gdn_gates`;
- write slot s to s+1.

The legacy copy is deleted at the end of WS-BI-2. After that, the HF/NumPy goldens and the differential tests are the guard.

### 6.3 Differential "forward over X == forward over CPU" (tiny model)

`ForwardDifferential` is a typed test over backends, using the §6.2 fixture. It compares X against the CPU backend.

| Backend | Where it runs | Mode | Pass condition |
|---|---|---|---|
| HipEmulation | dev host | (a) bitwise variants: `gemv_generic_b64`, `attn_exact_b128` and the defaults docs/hip.md marks bitwise | **bitwise** equality of every compared tensor (A-2). A mismatch is a finding, never a tolerance to widen |
| HipEmulation | dev host | (b) default variants: wave GEMV, online attention | argmax token equality along the whole trajectory, and per-layer relative L2 error ≤ τ. τ is fixed **before** the first measurement and derived from the per-op bounds in docs/hip.md. Proven capable of failing by a 1-ulp-scale mutation |
| Vulkan | lavapipe, dev host | as (b), with the Vulkan op bounds | as (b). GDN path forced to Recurrent on both sides while Vulkan lacks chunked (`Limits::gdn_chunked = false`) |
| Hip / Vulkan | EVO-X2 only | as (b), plus the 27B model spot checks | owed. D-001: not claimable before hardware |

The same fixture runs through `run_op` per op for per-op localisation. The existing per-op differential tests in `tests/unit/hip` and `tests/unit/vulkan` remain the kernel acceptance tests.

### 6.4 Existing tests and deliberate contract tightenings

The brief says 540+ tests. A static count of `TEST*` macros found about 710. **Neither number was run.** WS-BI-1's first step measures the baseline, pass count and exit code, in a **separate git worktree at the starting commit**, not in the shared tree. That number is the "no regression" reference.

Assertions that WS-BI-2 changes **on purpose**. Each is a tightening toward D-012/D-014, needs owner sign-off, and is never a loosening:

| Test | Today | After |
|---|---|---|
| `test_speculative.cpp`: verify state traffic at K=4 | `4 * state` (TD-1) | `3 * state`: (K−1), the D-014 NFR |
| `test_speculative.cpp`: `commit_bytes(4, 2)` | `2 * state` (restore copy) | `0` |
| `test_gdn_state.cpp`: `commit_bytes(T, kept)` | `kept == T ? 0 : 2 * total_bytes` | `0` for every kept |
| `test_gdn_state.cpp`: slot accessors (`recurrent_slots`, `conv_slots`, `mark_slots_written`, `commit_rows_kept`) | contiguous slots from slot 0 | kept as compatibility accessors for host-addressable states over the ring. Tests that address slot memory directly migrate to ring accessors in the same change, with the old/new equivalence tested first |

### 6.5 Seam tests

- A spy backend records the `KernelChoice` of each op call. The test asserts that the plan's `select_kernel` winner is what the backend was asked to run. This is the join N-5 says is missing.
- A variant-id stability test.
- A `Limits` rejection test (HIP d_k > 128 → `Error(Unsupported)` at construction).
- A per-sequence status failure test, run on the CPU backend and on HipEmulation: one sequence's NaN fails only it, and the others commit.
- A mid-enqueue abort test: host validation at op N → no state changes and the arena is reusable.

---

## 7. Consequences

**Easier:**
- One model definition.
- Per-op differential localisation on every backend.
- The autotuner reaches dispatch.
- TD-1 and TD-2 (GDN/KV) close.
- Rollback becomes one integer.
- Per-sequence failure containment.
- HIP is mostly an adapter over existing `Ops`.

**Harder:**
- ~1 060 launches per trunk step are exposed until fused ops or graph replay are added (§3, A-5).
- The forward is now written against an async model, which needs discipline: no host reads between submit and wait, and commit only after the status check.
- The step arena needs a sizing formula and a measured high-water mark.

**Replicated risk.** One shared forward means **a model-math bug is replicated on every backend**, and backend differentials cannot see it. The HF/NumPy goldens remain the only guard. The draft-chain golden (review R-5b, D-005 amendment) is still owed and should gate MTP-by-default.

**Forecloses (until superseded):**
- Backend-private whole-forward implementations as the primary path.
- Backend-private kernel selection.
- In-place GDN state as the default.

**Memory:** unchanged with MTP on. +1 state per sequence with MTP off (§5.3).

**The frozen CPU contract grows** by three additive overloads. The old slot form stays until the owner sets a sunset.

---

## 8. Open questions for the owner

1. **Frozen `cpu/ops.h`.** Approve the three additive ring overloads (§5.3). TD-1 cannot close without them. The old overloads stay, with sunset to be decided.
2. **GDN atomicity vs memory.** Choose one:
   - out-of-place ring as the default: P = K+1, same memory as today with MTP on, +1 state per sequence with MTP off;
   - the in-place P = K "memory-tight" planner mode, which is not atomic.
   - The author recommends the first. Also confirm that D-012's "K × ~150 MiB" means K copies beyond live.
3. **Weight residency on GPU.** Copy into VRAM/GTT at load (the default proposed here), or zero-copy import of the mmap? The latter needs EVO-X2 measurements of bandwidth and `hipHostRegister` on a file-backed mapping.
4. **Per-sequence failure semantics.** Accept that a data error owned by one sequence fails only that sequence while its co-batched sequences commit. This changes the Engine's R-1 behaviour.
5. **Launch overhead.** Agree that the η ≥ 0.80 NFR cannot be judged on the dev host, and that graph replay / command-buffer reuse (§5.7) is decided by a later ADR after the first EVO-X2 measurement of t_gap.
6. **Vulkan bring-up.** Agree that there is **no** CPU fallback for missing Vulkan ops. The Vulkan forward differential is enabled only when every op exists (DR-10). The alternative is a test-only mixed backend.
7. **Profile labels.** Agree that `hip-emulation` timings are never persisted as `hip`.
8. **Record keeping.** Add D-017 → ADR-001 to DECISIONS.md, and update ARCHITECTURE.md's Backend block on acceptance. Neither was editable in this task.

---

## 9. Implementation plan

Rules for every workstream:
- **Measure the baseline in a separate worktree.**
- **Never build in the shared tree.**
- **Touch `src/runtime`, `src/api`, `tools/halo` and `src/profiling` only in the workstreams that say so, in coordination with their current owners.**

| WS | Scope | May touch | Must not touch | Milestone gate | Who wires the join |
|---|---|---|---|---|---|
| **WS-BI-1** Interface + CPU adapter | `halo_backend` interface (§5.1). CPU backend over **today's** host state: KvPool / GdnState storage wrapped by `import_host`. `Qwen35` over `Backend&`, with old constructors kept. Legacy copy + bitwise gate (§6.2) | new `include/halo/backend/`, `src/backend/`; `include/halo/models/qwen35.h` (additive), `src/models/qwen35.cpp`; `tests/unit/backend/`, `tests/unit/models/` | `backends/cpu/ops.h`, state, kv_cache, runtime, speculative | §6.2 bitwise gate green and mutation-proven. Measured baseline count unchanged. The forward writes distinct outputs (§5.2 aliasing), and the de-interleave copy is replaced by head-strided operands | WS-BI-1 (the model is its only consumer; the engine is untouched) |
| **WS-BI-2** State on backend, TD-1 + TD-2 | Ring overloads in cpu ops. `GdnState` ring (`live`, P). `KvPool` storage from `Backend`, with COW via `copy`. Forward commits after status. Per-sequence status owners. Speculative commit = ring select. Planner P ≥ 2 | `backends/cpu/ops.h` + `src/backends/cpu` (additive, after Q1); `include/halo/state/gdn_state.h`, `src/state`; `include/halo/kv_cache`, `src/kv_cache`; `src/models`; `src/speculative/speculative.cpp`; `src/memory/planner.*`; the §6.4 tests | runtime (except a compile fix coordinated with its owner), api, tools | §6.4 tightenings green. TD-1 and TD-2 (GDN/KV) closed. Rollback-equivalence tests unchanged and green. Delete legacy copy | WS-BI-2 wires speculative. The engine keeps working unchanged because the APIs are preserved. The per-sequence retire goes to WS-BI-6 |
| **WS-BI-3** HIP adapter, emulation | `make_hip_backend` (emulation and device). `TensorRef` → `BufferView`. Per-variant `Ops` cache. Status array. Ring fields in `GdnArgs` / `Conv1dArgs` (additive kernel change). Registry `id` | new `src/backend/hip_adapter.*`; `backends/hip` (additive only); `tests/unit/backend/` | models (except bug fixes found by the differential, via WS-BI-1's owner), runtime | §6.3 HipEmulation (a) bitwise and (b) bounded, green and mutation-proven. Device path compile-checked only (D-001) | WS-BI-3 (test-level join). Engine selection is WS-BI-6 |
| **WS-BI-4** Autotune bridge | `VariantInfo` ids. `BackendTunableOp`. `Qwen35::op_keys`. `KernelPlan` in `Qwen35Options`. `hip-emulation` label. Spy-backend seam test | `include/halo/autotune`, `src/autotune`; `src/backend`; `src/models`. `tools/halo/tune.cpp` **only** in coordination with its current owner | profiling internals | Seam test (§6.5) green. The existing CPU threads/chunk consumption is unchanged | WS-BI-4 wires `KernelPlan` into `Qwen35`. The engine's `apply_profile` switch to `KernelPlan` goes to WS-BI-6 |
| **WS-BI-5** Vulkan | V1: adapter over the existing 4 ops, registry, `h ≡ a` for add, distinct `VK_ERROR_DEVICE_LOST`. V2: ADD, ADD+RMS_NORM, GATED_NORM, RoPE in place with head_stride, SwiGLU, MUL_SIGMOID, GDN gates, GET_ROWS, CONV1D + ring. V3: KV write, paged ATTENTION, batched GEMV (T > 1), LM-head argmax fusion, TOP_K, IQ4_NL / Q3_K / IQ3_S. V4: chunked GDN | `backends/vulkan` (+ shaders), `src/backend/vulkan_adapter.*`, `tests/unit/vulkan`, `tests/unit/backend` | models, runtime | Per-op differentials per milestone. After V3, the §6.3 Vulkan forward differential on lavapipe, Recurrent path. After V4, Chunked | WS-BI-5 (test-level). Engine selection is WS-BI-6 |
| **WS-BI-6** Engine integration | `EngineConfig::backend` → `make_*_backend`. Planner placement through `Backend::allocate`. Checkpoints in GPU-pool buffers (`GdnSnapshot` → buffer). Per-sequence failure retire (R-1). `KernelPlan` in `apply_profile`. Single-worker threading documented | `src/runtime`, `include/halo/runtime/engine.h` (additive), `include/halo/state/checkpoint.h`, `src/state`, `tools/halo` (flags). **Only after the current runtime/api/tools edits land, and coordinated with their owner** | api handlers | Engine tests on CPU unchanged. The engine runs on HipEmulation and on Vulkan (lavapipe) with the tiny model. Per-sequence failure test green | **WS-BI-6 owns the engine join**. Until it is done, GPU backends are reachable from tests only, and the report must say so |
| **WS-BI-7** Hardware (blocked on EVO-X2 access, D-001) | Device runs of §6.3. t_gap and sync-cost measurements against the §3 break-evens. The Q3 residency measurement | none in-tree beyond test fixes | — | Evidence for the §8 Q3/Q5 decisions and a graph-replay ADR | — |

**Order:** 1 → 2 → {3, 4, 5 in parallel} → 6 → 7.
- WS-BI-3 and WS-BI-5 depend on WS-BI-2's ring fields.
- WS-BI-4 depends on WS-BI-1 only.

### Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A-3 is false: per-head decomposition is not bitwise | Low | WS-BI-1 gate red | Keep the dense-copy path behind a CPU-adapter flag and fix before deleting it. The gate test finds it immediately |
| The frozen-contract change (Q1) is refused | Medium | TD-1 stays open. GPU backends inherit the waste | The adapter can emulate the ring on the CPU with a copy, costing traffic and failing the §6.4 tightening. Surface this to the owner rather than silently accepting it |
| Concurrent edits in runtime/engine collide with WS-BI-2/6 | High | Rework or broken shared build | WS-BI-1..5 avoid runtime entirely. WS-BI-6 starts only after those edits land |
| `src/speculative/speculative.cpp` and `test_speculative.cpp` are also being edited in flight (the review N-1 "drop MTP on failed flush" fix, seen uncommitted while this ADR was written) | High | WS-BI-2 merge conflict. The line numbers in §6.4 move | The in-flight change does not touch slot commit or the TD-1 assertions, so §6.4 cites tests by content, not line. WS-BI-2 starts after it lands |
| HIP emulation of the 248 320-row LM head makes the forward test slow (unmeasured) | Medium | Slow CI | Few logit rows per test. `LogitsMode::Argmax`. A separate ctest label |
| Launch gaps exceed the break-even on gfx1151 | Unknown (A-5) | η NFR missed | §5.7 path is kept open by construction. Measure in WS-BI-7 |
| Step-arena sizing formula is wrong | Medium | `Error(Memory)` at runtime | Validate the arena size at `KernelPlan` build for the largest bucket. Report the high-water mark to the planner |
| One shared forward replicates a model-math bug | Medium | Wrong on every backend | HF/NumPy goldens. Owed draft-chain golden (R-5b) |

---

## 10. References

- DECISIONS.md: D-001, D-002, D-003, D-004, D-005 (+ amendment), D-011, D-012, D-013, D-014 (resolved), D-016 (+ amendment).
- TRD v1.1 (header v1.2): §5, §6, §9, §11, §13, §17, §19, §20, §29, §30, §37, §45, §54, §55–§58, §64.
- docs/dev/ARCHITECTURE.md; docs/dev/TECH_DEBT.md (TD-1, TD-2; TD-9 closed); docs/hip.md; docs/vulkan.md.
- docs/reviews/2026-09-23-architecture-review.md §2a; docs/reviews/2026-09-25-code-review.md (R-1, R-2, R-5, N-5, N-6).
- The 2026-09-25 security reviews (api, bench) were checked. They contain no finding about the backend layer.
- Code at 7cf4238:
  - `include/halo/backends/{cpu,hip,vulkan}/*`;
  - `include/halo/models/qwen35.h`, `src/models/qwen35.cpp`;
  - `include/halo/{kv_cache,state,speculative,autotune,memory}/*`;
  - `src/state/gdn_state.cpp`, `src/runtime/engine.cpp`;
  - `python/tools/make_tiny_model.py`.
