# HALO v0.2 — Independent Architecture Review

Date: 2026-09-23 · Reviewer: solution-architect agent (independent of the implementing agents)
Scope: `docs/DECISIONS.md`, `docs/dev/ARCHITECTURE.md` (+ `AGENT_BRIEF.md`) checked against the PRD v1.1,
the TRD (header v1.2), the EVO-X2 engine report, and primary sources.

Primary sources I read (inside WSL):

- **[HF]** `transformers` `models/qwen3_5/modeling_qwen3_5.py` (installed copy in `/root/halo-py/.venv`)
- **[LCPP]** `/root/llama.cpp` at commit `bd4f514`: `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`,
  `src/models/models.h`, `conversion/qwen.py`, `common/speculative.cpp`, `common/common.{h,cpp}`,
  `src/llama-memory-recurrent.cpp`, `src/llama-memory-hybrid.cpp`, `src/llama-model.cpp`,
  `ggml/src/ggml-cpu/ops.cpp`, `tools/server/server-context.cpp`, `tools/server/server-task.cpp`
- **[GGUF]** `/root/halo-ref/{ggml-org-q4km,ggml-org-mtp-q4_0,unsloth-ud-q4kxl}.summary.json`,
  `/root/halo-ref/chat_template.jinja`, `/root/halo-ref/tiny/hf/config.json`
- **[REPO]** `python/tools/make_tiny_model.py` (the MTP NumPy golden)

Evidence labels: **Verified** = I read it in the primary source or computed it from the headers.
**Inferred** = follows from code I read, but I did not execute it. **Unverified** = I could not check it
here. Nothing in this review was run on gfx1151. No performance number here was measured by me.

---

## 0. Summary

DECISIONS.md is accurate on the points that matter most. D-003's state sizes, D-004's layer math,
the `j % n_k_heads` tiled V-head mapping, D-005's pairing and position convention, the post-output-norm
`h`, and D-007's dtypes all check out against the primary sources. The errors I found are omissions and
wrong rationale, not wrong math: D-005 leaves out things an implementer needs, and D-006's rationale has
the wrong number and conflates two different costs.

ARCHITECTURE.md has three design choices that would stop HALO from meeting the PRD success criteria
even if every kernel were perfect. All three can be fixed cheaply now, because no `Backend` code exists
yet: `git worktree list` shows only the main tree, and a `git grep` of `halo-v0.2` and `worktree-agent-ae9e334feeea9c090` for `class Backend|SequenceState|snapshot|forward(` returns nothing. Uncommitted work elsewhere is not visible to me.

1. **GDN rollback by snapshot + replay** (§2b, Critical). Any step with a rejected draft costs a second
   full pass over the trunk weights. At realistic acceptance rates that removes most of MTP's gain.
   llama.cpp at the pinned commit solves this differently: its GDN kernel writes per-row state
   snapshots, so rollback is just choosing a slot.
2. **Prefix cache that treats GDN checkpoints as an optional spike** (§2d, Critical). On this hybrid
   model, reusing KV alone saves almost no time to first token. The report's 331× cached-TTFT number
   comes entirely from llama-server's recurrent-state *context checkpoints*, which ship at the pinned
   commit.
3. **Round-robin over single-sequence forwards** (§2c, High), plus a single-sequence `Backend::forward`
   (§2a, High). With this design, total throughput across N streams is about the same as one stream.
   That is below the llama.cpp numbers it will be compared against, and the interface signature makes
   batching expensive to add later.

Separately, **PR-004's decode targets can't be reached on this hardware** (§3, High). The bytes read per
token can be computed from the GGUF headers today; this does not need to wait for Phase 0. Batch-1
decode without speculation at ≥ 45 tok/s would need about 742–821 GB/s of memory bandwidth, against a
256 GB/s nameplate. That is a human decision (PRD change), not something engineering can fix.

---

## 1. DECISIONS.md fact check

### 1.1 Verdict table

| Claim | Verdict | Evidence |
|---|---|---|
| D-003 recurrent state per **V** head, `d_k×d_v`, 48×48×128×128×4 B = **144 MiB/seq** | **Verified** | HF state shape `(B, num_v_heads, k_head_dim, v_head_dim)` `modeling_qwen3_5.py:473`; LCPP reshape `[head_v_dim, head_v_dim, num_v_heads]` `qwen35.cpp:388`; header `ssm.time_step_rank=48`, `ssm.group_count=16`, `ssm.state_size=128`, `ssm.inner_size=6144` |
| D-003 conv state 48×3×10240×4 B ≈ **5.6 MiB** (5.625) | **Verified** | LCPP `conv_states` reshape `(conv_kernel_size-1, conv_channels)` `delta-net-base.cpp:466`; conv_channels = `d_inner + 2·n_group·d_state` = 10240 `qwen35.cpp:383` |
| D-003 fp32 is the reference state dtype | **Verified** (the `mamba_ssm_dtype` config key itself is **unverified**: the real `config.json` is not in `/root/halo-ref`) | HF runs the recurrence in fp32 `modeling_qwen3_5.py:457-460,474`; LCPP hard-codes `recurrent_type_k/v = GGML_TYPE_F32` in the non-SWA hybrid branch used by QWEN35 `llama-model.cpp:2576-2582,2634-2647` |
| D-003 "TRD §2.2 24 MiB / §69 38 MB wrong" | **Verified** | TRD multiplies by 16 QK heads and 2 B; the state is per V head (48) and fp32 |
| D-004 residual structure, zero-centered HF RMSNorm, GGUF stores `w+1` | **Verified** | HF `(1+w)` `modeling_qwen3_5.py:854`; converter `+1` for every `*norm.weight` except `linear_attn.norm.weight` `conversion/qwen.py:401-402` |
| D-004 `ssm_norm` stored as-is, plain `x̂·w` | **Verified** | HF `Qwen3_5RMSNormGated` plain weight `:221,231`; converter exclusion `qwen.py:401` |
| D-004 the `+1` fold also covers MTP norms | **Verified** (D-004 doesn't say this; worth adding) | MTP tensors are renamed to `model.layers.64.{enorm,hnorm,shared_head.norm}` `qwen.py:327-339` *before* `modify_tensors`, so all end in `norm.weight` and get `+1` |
| D-004 full attention where `(il+1)%4==0` | **Verified** | `qwen35.recurrent_layers` array is present in all three headers and matches; LCPP prefers the array and falls back to the formula `qwen35.cpp:17-23` |
| D-004 `attn_q` → per head 256 Q then 256 gate | **Verified** | HF `chunk(...,2)` `:787-790`; LCPP views with offset `n_embd_head` `qwen35.cpp:275-292`; header `attn_q` dims `[5120, 12288]` = 24×512 |
| D-004 Q/K RMSNorm (dim 256), RoPE on first 64 dims, θ=1e7, NeoX rotate-half | **Verified** | `rope.dimension_count=64`, `rope.freq_base=1e7`; HF `rotate_half` `:666-670`, partial application `:698-708` |
| D-004 M-RoPE ≡ 1-D RoPE for text | **Verified for HF**: the same `arange` is expanded to all 4 components `:1260-1263`. Sections `[11,11,10,0]` sum to 32 = rot/2, so every rotary pair takes the text position | See Low finding L-4 on the tiny model |
| D-004 GQA 24/4, scale 1/√256, `o_proj(attn·σ(gate))` | **Verified** | HF `:758,818-820`; LCPP `:316,323-329` |
| D-004 GDN step 1 split 2048/2048/6144, z 6144, a/b 48 | **Verified** | header dims `attn_qkv [5120,10240]`, `attn_gate [5120,6144]`, `ssm_alpha/beta [5120,48]` |
| D-004 conv kernel 4, no bias, SiLU | **Verified** | HF `bias=False` `:525`; LCPP `:391-395` |
| D-004 L2 norm eps 1e-6, `q *= 1/√128` | **Verified**: HF and LCPP are algebraically identical | HF `x·rsqrt(Σx²+eps)` `:294-297`; LCPP `rms_norm(x, eps/n)/√n` = the same thing `models.h:14-17`; q scale HF `:468`, LCPP `delta-net-base.cpp:45-47` |
| D-004 `g = ssm_a·softplus(a+dt_bias)`, `ssm_a = −exp(A_log)` | **Verified** | converter `qwen.py:395-396`; LCPP `qwen35.cpp:369-373`; HF `:619` |
| **D-004 tiled V-head mapping, value head `j` uses key head `j % 16`** | **Verified** at three independent points | converter permutes `[n_k, v_per_k, d] → [v_per_k, n_k, d]` so new index = `r·16+k` (`qwen.py:467-476`); LCPP uses `ggml_repeat_4d` (tiling) when unfused (`qwen35.cpp:438-442`); fused CPU kernel `iq1 = iv1 % neq1`, `ik1 = iv1 % nek1` (`ops.cpp:10971-10972`). HF is grouped (`repeat_interleave` → `j/3`, `:620-622`) |
| D-004 reordered tensor list (qkv V rows, z, a/b, A_log/dt, conv V channels, `ssm_out` columns) | **Verified**, complete | `qwen.py:578-616` |
| D-004 recurrence order & `d_k×d_v` state | **Verified** | HF `:480-491`. Note LCPP stores S transposed (`ops.cpp:10999`: `s_out[j*S_v+i] = S[i][j]`). Irrelevant unless HALO imports or exports llama.cpp state |
| D-004 gated norm then `ssm_out`; FFN 5120→17408→5120; untied head | **Verified** | LCPP `:454-461`; separate `output.weight` (Q6_K) and `token_embd.weight` (Q4_K) in both main files |
| D-005 MTP input `eh_proj(concat(enorm(embed), hnorm(h)))`, **embedding first** | **Verified** | `qwen35.cpp:540-549` (`ggml_concat(e_norm, h_norm, 0)`) |
| D-005 "one full-attention block (same structure as D-004 attention layer)" | **Incomplete** (Medium M-1) | The MTP block is a *full decoder block*: attention **and** `post_attention_norm` → FFN (`blk.64.ffn_*`, 17408) → residual, `qwen35.cpp:552-622`. The FFN is ~63% of the block's parameters |
| D-005 `shared_head_norm` then shared head | **Verified**, with a fallback D-005 omits: `output_norm` if `shared_head_norm` is absent (`:624-627`), `nextn.embed_tokens` if present (`:519`), `shared_head_head` if present (`:636`). Both real packs carry `shared_head_norm` and neither carries the other two |
| D-005 `h` = trunk **final output-normed** hidden | **Verified** | `t_h_nextn` is set right after `output_norm` `qwen35.cpp:206-209` |
| D-005 pairing `(h_p, x_{p+1})` at RoPE pos `p+1`; catch-up decode shifts target h right by one and carries the last row over | **Verified** | `speculative.cpp:1350-1353, 1522-1547, 1595-1596` |
| D-005 drafting starts from last sampled token at `pos0` with carried-over h; top-k=10 | **Verified**, incomplete (M-1) | `:1624-1625, 1397-1401`. The draft token is `cur_p->data[0]` (argmax), gated by `p_min` (`:1683-1691`). **Draft steps ≥2 feed the MTP's own `h_nextn` (post-`shared_head_norm`) back at `pos0+i+1`** (`:1672, 1723-1724`). After verification, `pending_h` is the trunk row at index `n_accepted` (`:1763-1765`) |
| D-006 embedded (UD, `block_count=65`, `blk.64.*`) and separate (ggml-org 64 + `mtp-*.gguf` 65 with its own `token_embd`/`output`/`output_norm`, Q4_0) | **Verified** | headers; MTP file has 18 tensors (10 Q4_0, 8 F32) |
| D-006 rationale "saves ~1.7 GB of duplicate reads" | **Wrong number and wrong kind of cost** (Medium M-2) | see §1.3 |
| D-007 type mix and file sizes | **Verified** by type counts: Q4_K_M = 193 Q4_K (192 FFN + embed), 288 Q8_0 (240 GDN + 48 attn qkv), 17 Q6_K (16 attn_output + head). Byte totals computed from dims × block size: **18.963 GB** (D-007: 18.97) and **17.548 GB** (D-007: 17.56); the difference is the header |
| D-007 LM head Q6_K ≈ 1.04 GB/token | **Verified**: 5120×248320×210/256 B = **1.043 GB** |
| D-008 token counts / ids (not asked; spot check) | **Verified** for ggml-org headers: 248320 tokens, 247587 merges, BOS/PAD 248044, EOS 248046, `add_bos_token=false`. Unsloth pad not re-checked |

The dtype sizes behind every byte number in this review are GGML block sizes / block elements:
Q4_0 18/32, Q8_0 34/32, Q4_K 144/256, Q5_K 176/256, Q6_K 210/256, Q3_K 110/256, IQ4_XS 136/256,
IQ4_NL 18/32, IQ3_S 110/256, F32 4. They are validated by reproducing D-007's published file sizes
(previous row).

### 1.2 Bytes read per decode token (computed from the headers; resolves TRD §2.3(b) offline)

Per-token decode reads every trunk weight except the embedding table, which is a row gather.

| File | Total | Trunk read/token `W_trunk` | LM head `W_head` | MTP block `W_mtp` (excl. head) |
|---|---:|---:|---:|---:|
| ggml-org Q4_K_M (PRD canonical) | 18.963 GB | **18.248 GB** | 1.043 GB | — |
| unsloth UD-Q4_K_XL (the EVO-X2 file, all D-009 baselines) | 17.548 GB | **16.482 GB** | 1.043 GB | 0.351 GB |
| ggml-org `mtp-*` Q4_0 | 1.669 GB | — | 0.715 GB (own copy) | 0.239 GB |

GDN state traffic is `(1 + K)` × 0.157 GB per sequence per step: one read plus K snapshot writes, at
fp32 (144 MiB + 5.6 MiB). KV reads are 64 KiB × context per sequence per token, which is 0.27 GB at 4K
and 8 GiB at 131K (M-5).

### 1.3 D-006 — what the numbers say

- The duplicate tensors in the MTP file are `token_embd` + `output`, 2 × 0.715 GB = **1.43 GB**.
  ~1.7 GB is the size of the *whole* MTP file.
- These are **resident/load bytes**, not per-token reads. The embedding is a row gather. The only
  per-token cost is the head, and it runs the other way: sharing the trunk's **Q6_K** head costs
  **1.043 GB per draft token**, against **0.715 GB** for the MTP file's Q4_0 head. That is +0.33 GB per
  draft token, or +0.66 GB per step at n_draft=2 (about 2.6 ms at 256 GB/s, ~3% of a step). A
  higher-precision head could also raise acceptance, and that has not been measured.
- Baseline parity: for the **embedded** UD pack, llama.cpp's MTP context shares the trunk's weights
  (`common.cpp:1317` "an MTP context runs on the weights of the main model"), so D-006's default
  *matches* the baseline. For the **separate** ggml-org pack, the draft is a separate `llama_model`
  whose `tok_embd`/`output` come from the MTP file (`mtp_only` path `qwen35.cpp:36-49`). HALO's
  default diverges from llama.cpp only in that case. (Inferred from loader code; not run.)

### 1.4 The MTP golden (`make_tiny_model.py:145-184, 295-302`)

The golden is **correct for what it covers**. It includes the FFN and `post_attention_layernorm`,
applies `mtp.norm` (= `shared_head_norm`), uses zero-centered norms consistent with the converter's
`+1`, and pairs `tokens[i+1]` with `h[i]` at position `i+1`. It covers **only the teacher-forced
catch-up mode**. The draft chain (feeding MTP's own `h_nextn` back, positions `pos0+i+1`) and the
post-accept `pending_h` selection have no golden. That is where an implementation is most likely to
go wrong, so it is part of M-1.

---

## 2. ARCHITECTURE.md evaluation

### 2a. Model-level `Backend::forward` (High, H-2)

**Does it foreclose the registry/autotuner?** Partly. ARCHITECTURE.md:45-48 keeps kernels
"individually addressable **inside** each backend". The TRD puts selection *outside* the backend:
`ExecutionPlan.kernels` (TRD §11), profile-driven selection (§56), and a SQLite `operator_profile` /
`winning_configuration` store (§58). Under the current signature, the planner can't enumerate
candidates or pass a selection in. The autotuner then becomes N backend-private side channels, and the
plan can't "explain each decision" (PRD success criterion 3).

**Does it foreclose per-operator backend mixing (TRD v0.7)?** Yes, as written. A single call that runs
the whole model on one backend has no seam where HIP and Vulkan nodes could interleave.

**It also conflicts with ARCHITECTURE.md's own test strategy.** Correctness item 6 (ARCHITECTURE.md:109-110)
requires differential CPU-vs-Vulkan tests "per operator". That needs a per-op entry point, so one will
exist anyway. Make it part of the contract.

**Smallest interface that keeps both open** (recommendation; label: *preference*, grounded in TRD §11/§56
and the evolution rule "make later changes additive"):

```cpp
struct OpKey        { OpId op; DType wtype; DType atype; Shape4 shape; };   // TRD §9 ids
struct KernelChoice { BackendId backend; uint32_t variant; };               // chosen by planner/autotuner
struct StepPlan     { std::vector<KernelChoice> node; };  // indexed by the fixed qwen35 node list,
                                                          // one per (batch-shape bucket)
class Backend {
 public:
  virtual BackendInfo info() const = 0;
  virtual std::unique_ptr<Buffer> allocate(std::size_t, MemoryTier) = 0;
  virtual void upload_weights(const model::NormalizedModel&, const MemoryPlan&) = 0;
  // registry surface: v0.2 may return exactly one "reference" variant per key
  virtual std::span<const KernelVariant> variants(const OpKey&) const = 0;
  // single-op execution: differential tests, microbenchmarks, and (later) mixed-backend steps
  virtual void run_op(const OpInvocation&, Stream&) = 0;
  // fused fast path; contract: observationally equal to run_op over the node list under `plan`
  virtual void forward(const StepPlan&, const BatchRequest&,
                       std::span<SequenceState* const>, BatchResult&) = 0;
};
```

Four properties matter; everything else can be added later:
1. **Selection is an input** (`StepPlan`), never backend-private state.
2. **`run_op` exists from day one.** This is what later mixing builds on: a `MixedBackend` can
   partition the node list by `KernelChoice.backend`.
3. **`forward` is batch-shaped** (`BatchRequest` = rows per sequence, ragged; see 2c). The v0.2 CPU
   backend may loop internally.
4. **`BatchRequest` carries `n_state_slots`** (per-row GDN state emission; see 2b).

Buffer import/export for HIP↔Vulkan on unified memory can be added in v0.7 without breaking this.

### 2b. GDN rollback via snapshot + replay (Critical, C-1)

ARCHITECTURE.md:62-66 takes a snapshot before verification, restores it on rejection, and **replays
accepted tokens**. Costs at real dimensions (UD pack, n_draft=2, 256 GB/s nameplate, efficiency η):

- Snapshot: read + write 149.6 MiB = 0.31 GB, about **1.2 ms/step**. Cheap.
- Restore: another 0.31 GB, about 1.2 ms. Cheap.
- **Replay: a full trunk forward over the accepted tokens = 16.48 GB, about 64 ms at η=1.** A
  verification step itself costs about 79 ms at η=1 (20.25 GB, see §3). Replay is needed on every
  step where not all drafts are accepted. With per-token acceptance α=0.7 (illustrative), that is
  1−α² ≈ 51% of steps, adding ~33 ms to an average step, a **~42% slowdown**. With τ = 1+α+α² ≈ 2.19
  tokens per step, MTP throughput at η=1 falls from ~27 tok/s (slot-based rollback) to ~20 tok/s,
  against ~15 tok/s without MTP. **About two thirds of MTP's gain is lost**, and more at lower α.

**What llama.cpp does** (Verified in source): per-token recurrent snapshots written by the kernel,
with rollback by index.
- `cparams.n_rs_seq = draft.n_max` when MTP is on (`common.cpp:1723`, `common.h:394-399`). The
  recurrent cache allocates `mem_size × (1 + n_rs_seq)` rows (`llama-memory-recurrent.cpp:101-103`).
- The fused `ggml_gated_delta_net(..., K)` writes the state after each of the last K tokens into
  snapshot slots as part of the same kernel (`delta-net-base.cpp:563-603`; slot 0 = most recent,
  `ops.cpp:10955-10956`). Conv state gets K slots the same way (`delta-net-base.cpp:497-521`).
- `seq_rm(seq, p0, -1)` for `1 ≤ rollback ≤ n_rs_seq` just sets `rs_idx` and moves `cell.pos`
  (`llama-memory-recurrent.cpp:193-202`). The next graph reads from plane `rs_idx` (`s_copy`,
  `:1306-1324`). **No replay and no restore copy.**
- `split_equal(..., n_keep_tail = n_rs_seq+1)` keeps the last K tokens of a sequence in one ubatch
  so the snapshots stay valid (`llama-memory-recurrent.cpp:442-445`).
- The server falls back to a full state checkpoint only if the draft is longer than `n_rs_seq`
  (`server-context.cpp:3071-3090`).

**Options:**

| | A — snapshot + replay (current) | B — per-row state slots (llama.cpp) | C — snapshot + deferred replay |
|---|---|---|---|
| Mechanism | copy before verify; restore + separate replay forward | kernel emits state after rows `r ∈ [T−K, T)`; rollback = slot index | restore snapshot; put the accepted tokens at the front of the *next* verify batch |
| Extra bandwidth / step | 0.31 GB + (on reject) 0.31 GB + **16.5 GB** | +(K−1)×0.157 GB writes (≈0.31 GB at n=2) | 0.31–0.62 GB + extra rows (≈free while bandwidth-bound) |
| Extra memory / seq | 1× state | K× state (3 × 150 MiB = 449 MiB at n=2) | 1× state |
| Cancellation semantics | 3 mutable copies to keep consistent | commit = choose slot; atomic | snapshot + pending-replay flag |
| Backend contract impact | none | GDN op must accept `n_state_slots` | forward must accept "replay prefix + draft" rows |
| Evidence | none | shipped at the pinned commit | reasoning only |

**Recommendation: B** (label: *strong consensus / project convention*, since it is the reference
implementation HALO is benchmarked against). It gives the cheapest rollback, and the RR-006
cancellation invariant is trivial to state and test.

**Where this lands in the op contracts.** The GDN op signatures that parallel workstreams are writing now
(uncommitted in the working tree when I looked, so they may change) are
`gated_delta_rule_recurrent/chunked(dims, in, std::span<float> state, Rows out, ...)` in
`include/halo/backends/cpu/ops.h:165-176` and `gated_delta_rule_decode` in
`include/halo/backends/vulkan/ops.h:42-92`. Both take a single in/out state. Option B needs an
*optional* output: the state after each of the last K rows. That is an additive parameter; neither
signature has to be replaced. It is cheapest to add before the Vulkan shader and the chunked CPU path
harden. The cost is K× state memory: 8 slots × 3 ×
149.6 MiB ≈ 3.5 GiB, which is fine on a 96 GiB pool. C is the fallback for memory-tight
configurations. **Drop A** before any code is written against it.

### 2c. Scheduler: round-robin of single-sequence forwards (High, H-1)

Each single-sequence forward re-reads all 16.5 GB of weights, so round-robin across N streams is
time-slicing. Total throughput stays at about the single-stream rate (≈12–15 tok/s without MTP), and
per-stream speed falls as 1/N. The report measured llama.cpp at **22.7 tok/s (4 streams)** and
**30.2 tok/s (8 streams)** aggregate (report:256-259). Round-robin HALO would **lose** to this, and
would miss PR-004's aggregate target (success criterion 7).

Batched decode ceiling (one weight pass per step for S sequences, no MTP, η=1, 4K ctx):
S=4 → (16.48 + 4×0.31) GB = 17.7 GB/step → 14.4 steps/s → **~58 tok/s**. The gap between that
ceiling and llama.cpp's 22.7 is HALO's real concurrency opportunity.

Caveat: the report's aggregate is wall time for 4×(512-token prefill + 128 decode) (report:254-259),
so it includes prefill. HALO's harness must use the same definition to compare fairly.

**Recommendation:** keep a single worker thread in v0.2, but make each scheduler tick issue **one
batched forward over all DECODING sequences** through the batch-shaped `forward` from 2a (the CPU
reference may loop over sequences internally). The costly thing to change later is the signature
and the `SequenceState` ownership model, not the loop. TRD §22's "batch-1 → prefix cache → continuous
batching" ordering can stay as the order *capabilities* are enabled, as long as the contract is
batch-shaped from the start. This also makes success criterion 8 ("GDN state and KV under
concurrency") testable on the dev host.

### 2d. Prefix cache for the hybrid model (Critical, C-2)

**KV-only reuse saves almost nothing on this model.** To continue from position N you need the GDN
state at N, and rebuilding it means running the *whole* trunk over the prefix, because every GDN
layer's inputs depend on the attention layers before it. llama-server does exactly this:
`server-context.cpp:3379-3384` forces full re-processing ("likely due to SWA or hybrid/recurrent
memory") when no suitable checkpoint exists.

**How llama.cpp gets the 331× cached TTFT** (Verified in source; the configuration matches report §5.2):

- **Context checkpoints** (default `n_ctx_checkpoints=32`, `checkpoint_min_step=8192`,
  `common.h:629-631`) are `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` snapshots. For a hybrid memory these
  contain the **recurrent state only**; attention KV stays in place (`llama-memory-hybrid.cpp:190-202`,
  `server-context.cpp:2363`).
- They are placed (a) at **user-message starts** (`spans.is_user_start`, `server-context.cpp:3550-3557`),
  (b) no closer than 8192 tokens apart, and (c) **near the end of the prompt**, at `4+n_ubatch` and
  `4` tokens before the end (`:3559-3578`). Placement is at N−k rather than N because the last
  prompt token must be re-evaluated to produce logits (`:3402-3407`).
- On a new request: `n_past` = longest common prefix (LCP) with the slot's tokens. The newest
  checkpoint with `pos_max ≤ pos_next` is restored, and only the tail is recomputed (`:3349-3377`).
  Invalid later checkpoints are erased (`:3389-3398`).
- For the report's resubmitted 24K prompt: restore the N−4 checkpoint (~150 MiB), evaluate 4 tokens,
  decode, which gives 0.22 s. **Inferred** from the code path; I did not see the server log.
- The `--cache-ram` prompt cache (default 8192 MiB, `common.h:632`) additionally saves whole slot
  states for other prompts and picks by LCP similarity (`server-task.cpp:1793-1823`).
- Checkpoints live in **host** `std::vector<uint8_t>` (`common.h:1174`, `common.cpp:2290-2294`). With
  D-002's topology, that comes out of the **~32 GiB OS pool**.

**What ARCHITECTURE.md:90-93 gets right and wrong.** Refcounted KV blocks plus exact-prefix GDN
snapshots is the right shape. The problems:

- It is labelled a spike with "otherwise always-recompute". TRD §12.2 and §67 v0.4 say the problem is
  "unsolved elsewhere", which is no longer true at the pinned commit.
- It is committed as block-granular hashing but has **no snapshot placement policy**.
- The template → scheduler path has **no channel for message boundaries**, which llama.cpp's
  placement depends on.
- It has no memory budget for checkpoints: ~17 per slot at 131K / 8192 spacing ≈ 2.5 GiB per slot.

**Recommended design** (label: *project convention*, mirroring the reference):

1. KV: refcounted paged blocks, as planned.
2. GDN: sparse full-state snapshots at (i) the end of the system/tool preamble (shareable read-only
   across sessions, restored by copy), (ii) each user-message start, (iii) prompt end − small k, and
   (iv) spacing ≥ S tokens. Each costs 149.6 MiB, and a restore is ~1.2 ms of copy at 256 GB/s.
3. A hit is the longest snapshot at a position ≤ LCP. Recompute from there, and drop KV blocks past
   the snapshot position.
4. The template renderer returns **message-boundary token offsets** alongside the token ids.
5. The memory planner owns a checkpoint budget and places snapshots in the **GPU pool**, not host RAM.
6. Default `preserve_thinking` so that history re-renders identically. The template strips earlier
   `<think>` content only when `preserve_thinking` is false (`chat_template.jinja:116`). Expect the
   previous assistant turn to re-tokenize differently from the sampled ids, which is why the
   prompt-end snapshot is the one that actually hits in multi-turn use.

---

## 3. Top risks to the PRD success criteria (dev host cannot run gfx1151)

### 3.1 PR-004 targets vs the bandwidth ceiling (High, H-3). Decision needed (human / PRD owner)

Ceiling model (η = achieved / nameplate bandwidth, BW = 256 GB/s, τ = tokens emitted per step per
sequence ≤ 1+n, S = concurrent sequences, n = draft length, K = n+1 state slots):

```
tok/s ≤ S · τ · η · BW / B_step
B_step = W_trunk + (n+1)·W_mtp + n·W_head + S·[(1+K)·0.157 GB + ctx·64 KiB]
```

(MTP weights are read once per draft step for all sequences, as llama.cpp batches drafts. `ctx` is the
number of filled tokens, not the allocated context.) UD pack, η=1. **The table leaves out the KV term.**
It is at most 0.27 GB per sequence at 4K filled, and would lower the S=1, n=0 ceiling from 15.2 to about
15.0 tok/s.

| Case | B_step | Ceiling | Target | What the target requires |
|---|---:|---:|---:|---|
| S=1, n=0 | 16.8 GB | **15.2 tok/s** | ≥ 45 | ≥ 742 GB/s (UD) / 821 GB/s (Q4_K_M): **physically unreachable** on 256 GB/s. On-chip caches are orders of magnitude smaller than the 16 GB working set |
| S=1, n=2 | 20.3 GB | 12.6 steps/s × τ ≤ **37.9 tok/s** (τ=3) | ≥ 70 effective | τ·η ≥ 5.5, impossible with n=2 (τ ≤ 3). Even n=4 at perfect acceptance and η=1 gives ≤ 55; roughly n≥8 with ~100% acceptance is needed. **Infeasible at any realistic acceptance** |
| S=4, n=2 | 22.1 GB | 11.6 steps/s × 4 × τ | ≥ 120 aggregate | τ·η ≥ 2.6, i.e. ≥ ~87% of the τ=3 limit at η=1. Infeasible at the acceptance implied by the baseline (τ≈1.9 at η≈0.79, inferred) |

On-machine sanity check: no-MTP `llama-cli` measured **12.0 tok/s** (report:199-204; the default spec
type is `NONE`, `common.h:371`), which is η ≈ 0.79. That is consistent with the ceiling. The TRD
§2.3 "empirical contradiction" (baselines above the naive roofline) is explained by MTP (18–21 tok/s
with `--spec-type draft-mtp`). Its proposed explanations (effective bandwidth above nameplate,
hot-tensor cache reuse) are unsupported.

**Implication:** success criterion 7 fails by construction unless PR-004 is re-derived. That means
stating targets as η and τ against the computed bytes per token (e.g. "≥ 85% of measured bandwidth;
MTP net ≥ 1.5× at measured acceptance"), plus an explicit context length (see M-5). This needs the
PRD owner.

### 3.2 Canonical pack ≠ baseline file (High, H-4). Decision needed

The PRD §2.1 canonical input is ggml-org Q4_K_M + separate MTP. Every D-009 baseline ran on UD-Q4_K_XL
with embedded MTP, and PR-001 requires the *same model file*. On Q4_K_M, HALO reads **10.7% more
bytes per token** (18.25 vs 16.48 GB), so it starts ~10% behind before any engineering. UD also uses 8
quantized types (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_NL, IQ4_XS). Every one of them needs a
**GPU GEMV kernel**, not only CPU dequant (D-007), against 3 for Q4_K_M. Choose either: make UD the
performance-canonical pack (larger kernel scope), or re-baseline llama.cpp on Q4_K_M on the EVO-X2
(cheap, and keeps the kernel scope small).

### 3.3 Risks from the dev host / target split, and architecture changes that reduce them

| Risk (criterion affected) | Why dev-host evidence can't catch it | Architecture change that reduces it |
|---|---|---|
| Performance-shaping decisions (rollback, batching, checkpoints) are validated only on paper (7, 9) | No gfx1151 access; D-001 | **Executable cost model**: the CPU reference counts bytes moved per op per step and reports predicted `B_step`. CI asserts that e.g. a rollback moves ≤ X bytes, so C-1-class regressions are caught on the dev host |
| Rollback / checkpoint / cancellation consistency (1, 8, RR-006) | Needs concurrency + speculation paths on CPU | Batch-shaped forward + slot-based rollback (2b/2c) make these CPU-testable. Add differential tests: state after (verify + rollback to r) == state after decoding r tokens |
| GPU kernel correctness beyond lavapipe (8) | lavapipe is a CPU rasterizer. Its subgroup behavior and performance paths differ from RDNA wave32 (**unverified** specifics) | Parameterize shaders by subgroup size and test several sizes. The RTX 3060 (subgroup 32) is a closer non-CPU Vulkan target *if* reachable from the toolchain (**unverified** under WSL2) |
| HIP never executes before hardware (8, 7) | D-001 | Make a **remote runner on the EVO-X2** a v0.2 deliverable (TRD §60 already requires a real-hardware GPU lane): SSH-driven build + `ctest` + the bench harness |
| Baseline comparability (4, 7) | The report's 1k/4k "cold" rows are not cold compute: 26,825 tok/s × ~54 GFLOP/token ≈ 1.45 PFLOP/s ≫ the ~55–60 TFLOPS peak (TRD §2.3). The report's "graphs compiled" explanation doesn't account for this | The harness must enforce cold-cache runs with *unrelated* warm-up text (the report built both from GPL-3). The 512-token prefill baseline for PR-004 is currently unmeasured |
| Cached-TTFT win (9) | Depends on C-2 | Commit the §2d design; measure the hit rate on replayed multi-turn agent transcripts |

---

## 4. Findings index (by severity)

**Critical**
- **C-1** GDN rollback by snapshot + replay adds a full trunk pass on most MTP steps (§2b). Replace
  with per-row state slots (llama.cpp `n_rs_seq`).
- **C-2** KV-only prefix reuse gives ≈0 TTFT benefit on this hybrid model. GDN checkpoints are the
  mechanism behind the 331× baseline and ship at the pinned commit (§2d). Promote from spike to a
  committed design.

**High**
- **H-1** Round-robin single-sequence scheduling makes aggregate ≈ single-stream, below the
  22.7 / 30.2 tok/s baselines (§2c).
- **H-2** Model-level `forward` without plan input or `run_op` leaves the registry/autotuner
  backend-private and forecloses op-level backend mixing (§2a).
- **H-3** PR-004 batch-1 ≥45 is physically unreachable. ≥70 MTP-effective and ≥120 at 4 agents are
  infeasible at realistic acceptance (§3.1). Human decision.
- **H-4** The canonical pack (Q4_K_M) differs from the baseline file (UD). That is a 10.7% bytes/token
  handicap, or a wider GPU kernel scope (§3.2). Human decision.

**Medium**
- **M-1** D-005 is incomplete: the MTP block includes the FFN and `post_attention_norm`; the fallbacks
  (`output_norm`, `nextn.embed_tokens`, `shared_head_head`); draft steps ≥2 feed the MTP's own
  post-`shared_head_norm` `h_nextn` at `pos0+i+1`; `pending_h` ← trunk row `n_accepted` after verify;
  argmax + `p_min` gating; MTP KV rollback past `pos_max` (`server-context.cpp:3065`). The golden
  lacks a draft-chain case (§1.4).
- **M-2** D-006 rationale: the duplicates are 1.43 GB, not ~1.7 GB. They are resident bytes, not
  per-token reads. The trunk Q6_K head costs +0.33 GB per draft token. The default diverges from
  llama.cpp only for the separate pack (§1.3). Make the head choice a measured plan decision.
- **M-3** The memory planner has no line items for GDN rollback slots (K × 150 MiB/seq) or prefix
  checkpoints (~2.5 GiB/slot at 131K). llama.cpp keeps checkpoints in host RAM, which on this unit is
  the 32 GiB OS pool (§2d).
- **M-4** The report's memory table ("compute buffers + GDN recurrent state ~1.8–2.5 GiB") is below the
  code-derived GDN state alone for its own config: rs_size = `max(1, n_seq_max)` = 8
  (`llama-model.cpp:2645`) × (1+n_rs_seq=3) × 149.6 MiB ≈ **3.5 GiB**. Inferred from code, not from a
  log. Don't feed the report's figure into the planner.
- **M-5** Context-length term in decode NFRs: KV read is 64 KiB × ctx per token (8 GiB at 131K,
  16 GiB at 256K). Long-context decode will be much slower than 4K. Every decode target must state a
  context length.

**Low**
- **L-1** Read `qwen35.attention.recurrent_layers` (present in all three headers) as the authority and
  cross-check it against `full_attention_interval`, as llama.cpp does (`qwen35.cpp:17-23`).
- **L-2** HF goldens are in *grouped* V-head order and GGUF / HALO in *tiled* order. Current goldens
  (layer inputs, final hidden, logits) are order-invariant. Any future per-head or state golden taken
  from HF must be permuted with the converter's `_reorder_v_heads` (`qwen.py:467-476`), including conv
  V channels.
- **L-3** TRD §2.2, §13.3, §69 (state size and carveout placement) and TRD §12.2 / §67 v0.4 ("unsolved
  elsewhere") are stale. DECISIONS.md should record the latter as superseded too.
- **L-4** The tiny model's `mrope_section=[1,1,2]` sums to 4, but rot/2 = 8 (`make_tiny_model.py:55-56`).
  The real model's sections sum to exactly rot/2. HF is unaffected (equal positions). llama.cpp's
  sector mapping (`ops.cpp:6000-6044`) may treat the extra dims differently. **Unverified**: check
  before using llama.cpp as an oracle on the tiny GGUF.

---

## 5. Recommended changes

### To ARCHITECTURE.md
1. Replace the Backend block with the §2a interface: `StepPlan` input, `variants()`, `run_op()`,
   batch-shaped `forward`. State the equivalence contract between `forward` and `run_op`.
2. Replace the "Commit/rollback" paragraph with per-row GDN state slots (`n_state_slots = n_draft+1`,
   written by the GDN op; rollback = slot select; conv state likewise). KV rollback = truncate. MTP KV
   rollback past `pos_max`. Commit is atomic: slot choice + length.
3. Scheduler: one batched forward per tick over all DECODING sequences. v0.2 CPU may loop internally.
4. Prefix cache: commit the §2d design (snapshot placement policy, message-boundary offsets from the
   template, GPU-pool checkpoint budget, hit = longest snapshot ≤ LCP). Keep always-recompute as the
   fallback, not the plan.
5. Add an executable bytes-moved cost model to the CPU reference as a CI check.
6. Correctness strategy: add an MTP draft-chain golden (h feedback + `pending_h` after accept) and a
   rollback-equivalence differential test.

### To DECISIONS.md (new entries or amendments; none edit history silently)
- **D-005 (amend):** full decoder block incl. FFN; fallbacks; draft-chain h feedback and positions;
  `pending_h` rule; argmax + `p_min`; MTP KV rollback.
- **D-006 (amend):** 1.43 GB resident duplicates; per-draft-token head cost of Q6_K vs Q4_0; head
  choice measured by the planner; baseline parity statement.
- **D-011 (new): bytes-per-token facts** (§1.2 table) and the ceiling formula (§3.1). Mark TRD §2.3(b)
  resolved and its "contradiction" explained by MTP.
- **D-012 (new): recurrent rollback = per-row state slots** (supersedes ARCHITECTURE's snapshot+replay).
- **D-013 (new): GDN prefix checkpoints are committed**, citing llama.cpp `bd4f514` context
  checkpoints. Supersedes TRD §12.2's "R&D spike / unsolved elsewhere".
- **D-014 (decision needed, human):** PR-004 re-derivation, and canonical-pack choice (UD vs
  re-baselined Q4_K_M).
- **D-003 (annotate):** `mamba_ssm_dtype` not verifiable from the local reference set. fp32 is
  corroborated by HF compute dtype and llama.cpp's hard-coded F32 recurrent cache.

---

## 6. Proposed NFRs (measurable; to replace PR-004's absolute decode numbers pending the human decision)

| NFR | Target | How measured |
|---|---|---|
| Decode bandwidth efficiency, S=1, n=0, 4K | η ≥ 0.80 of measured per-tier bandwidth (baseline llama.cpp ≈ 0.79 of nameplate) | `W_trunk + state` bytes / step time, harness §50 |
| MTP net speedup, S=1 | ≥ 1.5× over n=0 at measured acceptance; auto-off below 1.05× | same prompt set, ≥3 repetitions |
| Rollback cost | ≤ (K−1)×0.157 GB extra bytes per step; zero extra weight passes | cost-model CI assertion + GPU timestamps |
| Aggregate decode, S=4, 512-prompt/128-gen, wall-clock incl. prefill | > llama.cpp same-file baseline (22.7 tok/s); stretch ≥ 40 | report §5.6 methodology |
| Cached TTFT, 24K identical re-submit | ≤ 300 ms (baseline 220 ms) | report §5.5 methodology |
| Cached TTFT, multi-turn agent replay (32K) | ≤ 300 ms at ≥ 80% hit rate on recorded transcripts | prefix-hit-rate metric (FR-012) |
| Checkpoint memory | ≤ configured budget in the GPU pool; 0 bytes from the OS pool | planner report |

---

*This review wrote no project memory. The requester restricted writes to this file, so the ADR-style
entries above are proposals for the owner to persist.*
