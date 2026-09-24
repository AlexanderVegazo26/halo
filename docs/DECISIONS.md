# HALO — Engineering Decisions & Ground Truth

This file records (a) facts measured or read from primary sources that correct the
PRD v1.1 / TRD (file header says v1.2), and (b) the decisions HALO's implementation
follows because of them. Where this file and the PRD/TRD disagree, **this file wins**
until the PRD/TRD are revised. Every fact carries its source.

Sources used (fetched 2026-09-23):

- **[CFG]** `Qwen/Qwen3.8-27B/config.json` (HF)
- **[HF]** `transformers` 5.17.0 `models/qwen3_5/modeling_qwen3_5.py` (reference semantics)
- **[LCPP]** llama.cpp commit `bd4f514db` — `src/models/qwen35.cpp`, `conversion/qwen.py`,
  `src/models/delta-net-base.cpp`. Same commit as the target machine's llama.cpp build.
- **[GGUF]** Real GGUF headers fetched by HTTP Range (no weights) — see
  `python/tools/fetch_reference.py`: ggml-org `Qwen3.8-27B-Q4_K_M.gguf`,
  ggml-org `mtp-Qwen3.8-27B-Q4_0.gguf`, unsloth `Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- **[HW]** `docs/strix-halo-qwen38-engine-report.md` — measured on the actual EVO-X2.

---

## D-001 — Development host is not the target (status: fact)

Development happens on a Windows laptop (Ryzen 7 4800H, RTX 3060, 63 GB) with a WSL2
Ubuntu 24.04 toolchain (clang 18, CMake 3.28, Mesa lavapipe). Consequences:

- **No performance number produced on the dev host is a HALO performance claim.** PR-004
  targets can only be measured on the EVO-X2.
- Vulkan shaders are validated for **correctness** on lavapipe (CPU Vulkan 1.3); speed is
  unmeasured until run on RADV/gfx1151.
- HIP kernels are compile-checked at best; execution is unverified until run on gfx1151.
- Phase-0 (TRD §68 steps 1–11) is **blocked on hardware access**; HALO ships the harness.

## D-002 — Memory topology: 96 GiB carveout, not small-carveout + large-GTT (fact, [HW])

The PRD/TRD model assumes a 0.5–2 GB BIOS carveout and a ~120 GB GTT pool. The actual
EVO-X2 is configured with **96 GiB of VRAM carveout** (`amdgpu: 98304M of VRAM memory
ready`) and **~32 GiB OS-visible RAM**.

Decision: the memory planner is **tier-discovery-driven, never assumption-driven**. Tier
sizes come from sysfs (`mem_info_vram_total`, `mem_info_gtt_total`, `/proc/meminfo`).
Placement policy prefers VRAM (carveout) for everything when it fits, falling back to GTT.
The "state must live in a tiny carveout" reasoning in TRD §13.3 is inapplicable on this
unit. Both topologies are supported and unit-tested with sysfs fixtures.

## D-003 — DeltaNet state is ~6× larger than the TRD states, and fp32 (fact, [CFG][HF][LCPP])

Real config: `linear_num_key_heads=16`, `linear_num_value_heads=48`,
`linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`,
`mamba_ssm_dtype=float32`. The recurrent state is per **value** head, `d_k × d_v`:

- recurrent state = 48 layers × 48 heads × 128 × 128 × 4 B = **144 MiB / sequence (fp32)**
- conv state = 48 layers × 3 × 10240 channels × 4 B ≈ **5.6 MiB / sequence**

TRD §2.2's 24 MiB (and §69's 38 MB) are wrong. State format default is **fp32** (the
reference dtype); fp16/bf16 state is an experiment gated on long-context divergence (TRD §16).

## D-004 — Exact Qwen3.8 ("qwen35") layer semantics (fact, [HF][LCPP])

Per decoder layer: `x += mixer(rmsnorm(x))`; `x += ffn(rmsnorm(x))`.
RMSNorm in HF is zero-centered (`x̂·(1+w)`); **GGUF stores `w+1`**, so HALO uses plain
`x̂·w` for every GGUF norm **except** `ssm_norm` (stored as-is, also plain `x̂·w`).

**Full-attention layer** (layers where `(il+1) % 4 == 0`):
- `attn_q` outputs `n_head × (2·256)`; per head the first 256 are Q, the next 256 are the
  output gate.
- Q and K get per-head RMSNorm (`attn_q_norm`, `attn_k_norm`, dim 256).
- RoPE on the first 64 of 256 dims (`rope.dimension_count=64`), theta 1e7, NeoX-style
  (rotate-half) layout. M-RoPE sections `[11,11,10]` are interleaved, but **for text-only
  input all three position components are equal, so M-RoPE ≡ standard 1-D RoPE** — HALO
  v1 implements 1-D RoPE and rejects multimodal positions.
- GQA 24 Q / 4 KV heads, scale `1/sqrt(256)`, causal softmax.
- `out = o_proj(attn · sigmoid(gate))`.

**Gated DeltaNet layer**:
1. `qkv = attn_qkv(x)` (10240 = 2048 q + 2048 k + 6144 v), `z = attn_gate(x)` (6144),
   `b = ssm_beta(x)` (48), `a = ssm_alpha(x)` (48).
2. Causal depthwise conv1d (kernel 4, no bias) over the 10240 qkv channels, then SiLU.
   Conv state holds the previous 3 inputs.
3. Split q[16×128], k[16×128], v[48×128]; L2-normalize q and k per head (eps 1e-6);
   q *= 1/sqrt(128).
4. `beta = sigmoid(b)`; `g = ssm_a · softplus(a + ssm_dt.bias)` where GGUF `ssm_a = -exp(A_log)`.
5. **Head mapping — GGUF tiled order:** the converter reorders V heads from grouped
   (`[K0: v0..v2, K1: v0..v2, …]`) to tiled (`[K0v0, K1v0, …, K15v0, K0v1, …]`). In GGUF,
   value head `j` uses key head `j % 16`. (HF grouped order would be `j / 3`.) This applies
   to `attn_qkv` V rows, `attn_gate`, `ssm_alpha/beta`, `ssm_a`, `ssm_dt`, conv V channels,
   and `ssm_out` columns — all already reordered in the file.
6. Per value head, per token (recurrent form; chunked form must be numerically equivalent):
   `S ← S·exp(g)`; `kv = Sᵀk`; `δ = (v − kv)·β`; `S ← S + k δᵀ`; `o = Sᵀq`.
   `S` is `d_k × d_v` = 128 × 128.
7. `o = rmsnorm(o; ssm_norm) · silu(z)` per head (dim 128); `out = ssm_out(o)`.

**FFN**: `down(silu(gate(x)) · up(x))`, 5120 → 17408 → 5120.

**Head**: `logits = output(rmsnorm(x; output_norm))`. Untied (`token_embd` ≠ `output`).

## D-005 — MTP / NextN drafter (fact, [LCPP]; HF transformers does not implement it)

One MTP block (`nextn_predict_layers=1`) stored as `blk.64.*`:
`h' = eh_proj(concat(rmsnorm(embed(tok); enorm), rmsnorm(h; hnorm)))` — **embedding first,
hidden second** — then one full-attention block (same structure as D-004 attention layer,
with its own KV cache), then `rmsnorm(·; nextn.shared_head_norm)` and the shared LM head
(`output` unless `nextn.shared_head_head` exists). `h` is the trunk's **final
output-normed** hidden state (`t_h_nextn` in llama.cpp; set after `output_norm` in
`qwen35.cpp`) for the position of `tok`'s predecessor.

**Position/pairing convention** (llama.cpp `common/speculative.cpp`,
`common_speculative_impl_draft_mtp`): MTP row pairs `(h_p, x_{p+1})` and runs at RoPE
position **p+1** (the position of the embedded token). The MTP block keeps its **own KV
cache** (one attention layer), which must be populated for every prompt position during
prefill ("catch-up decode": target hidden rows shifted right by one, the last row carried
over to the next call). Drafting starts from the last sampled token at `pos0` with the
carried-over `h`; llama.cpp drafts with a top-k=10 sampler. llama.cpp is the only
executable reference; HALO's MTP golden tests compare against a NumPy port of that graph
(`python/tools/make_tiny_model.py`).

## D-006 — Two MTP packagings must both load (fact, [GGUF][HW])

- **Embedded**: unsloth UD files (the one on the EVO-X2) — `block_count=65`,
  `nextn_predict_layers=1`, `blk.64.*` in the same file.
- **Separate file**: ggml-org pack — main file `block_count=64` without MTP; the
  `mtp-*.gguf` file has `block_count=65`, only `blk.64.*` + its **own** `token_embd`,
  `output`, `output_norm` copies (Q4_0 in the Q4_0 pack).

Decision: `--mtp <file>` is optional; if absent, HALO uses embedded `blk.64` when present.
When a separate MTP file carries its own embeddings/head, HALO uses the **trunk's** tensors
by default (saves ~1.7 GB of duplicate reads) and records the choice in the plan.

## D-007 — Real quantization mix (fact, [GGUF])

| File | Types present |
|---|---|
| ggml-org Q4_K_M (18.97 GB) | Q4_K (FFN, embed), Q8_0 (attn/GDN projections), Q6_K (LM head, attn_output), F32 |
| unsloth UD-Q4_K_XL (17.56 GB) | Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_NL, IQ4_XS, F32 |
| ggml-org mtp Q4_0 | Q4_0, F32 |

The LM head is **Q6_K (~1.04 GB/token read)** in both main files — not Q4 (PRD's 0.7 GB).
Decision: the CPU reference dequantizes F32, F16, BF16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K,
IQ3_S, IQ4_NL, IQ4_XS. Dequantization is differentially tested against `gguf-py` on real
blocks range-fetched from the published files. IQ lookup grids are taken from ggml (MIT).

## D-008 — Tokenizer (fact, [GGUF])

`tokenizer.ggml.model=gpt2`, `pre=qwen35`, 248,320 tokens, 247,587 merges. BOS 248044,
EOS 248046, `add_bos_token=False`; pad differs per file (248044 ggml-org, 248055 unsloth).
The chat template ships in GGUF (`tokenizer.chat_template`, 8952 chars ggml-org /
9993 chars unsloth). HALO renders the template embedded in the loaded file.

## D-009 — Baselines already measured on the EVO-X2 (fact, [HW])

llama.cpp HIP (`bd4f514db`), UD-Q4_K_XL, MTP draft n=2, f16 KV, 131k ctx, 8 slots:
decode **18–21 tok/s**, cold prefill **330–430 tok/s** (12k–24k prompts), cached 24k
TTFT **0.22 s (331×)**, 4-stream aggregate 22.7 tok/s, 8-stream 30.2 tok/s. HIP works on
this unit (HIP 7.15). These are the numbers PR-004 progress is judged against until
Phase 0 re-measures them in HALO's harness. Note the PR-004 "≥45 tok/s" target exceeds the
TRD's own naive roofline (~15 tok/s at 17 GB/256 GB/s) and must be re-derived from a
measured roofline (TRD §2.3).

## D-010 — Build & dependency choices (decision)

- C++23 (TRD's C++20 minimum; 23 used for `std::expected`/`std::format`), clang, CMake + Ninja. Dependencies pinned by FetchContent tag:
  nlohmann/json, cpp-httplib (HTTP/SSE server), minja (Jinja subset, if it renders the
  shipped template — verified by golden tests), GoogleTest. SQLite from the system.
- Python (uv, 3.12) for reference generation, golden data, benchmarks — never in serving.
- The build tree lives on the Linux filesystem (`~/halo-build`), sources on `/mnt/c`.

---

# Amendments after independent architecture review (2026-09-24)

Source: `docs/reviews/2026-09-23-architecture-review.md` (solution-architect, evidence cited
to llama.cpp `bd4f514` file:line). Earlier entries are kept as written; these amend them.

## D-003 (annotation)
`mamba_ssm_dtype` was not verifiable from the local reference set. fp32 is corroborated by
the HF compute dtype and llama.cpp's hard-coded F32 recurrent cache.

## D-005 (amendment) — MTP details
- The MTP block is a **full decoder block**, including `post_attention_norm` and the FFN.
- Fallbacks: `nextn.shared_head_norm` → `output_norm`; `nextn.embed_tokens` → `token_embd`;
  `nextn.shared_head_head` → `output`.
- Draft chain: draft step 1 feeds the carried-over trunk `h`. Steps ≥ 2 feed the MTP's **own**
  post-`shared_head_norm` hidden at position `pos0+i+1` (`speculative.cpp:1672,1723-1724`).
- After verification, `pending_h` is the trunk hidden row at index `n_accepted`
  (`:1763-1765`).
- Drafting uses argmax with `p_min` confidence gating. MTP KV rolls back past `pos_max`.
- The golden covers only the teacher-forced catch-up mode. A draft-chain golden is owed.

## D-006 (amendment)
The duplicated tensors in the separate MTP file are **1.43 GB resident**, not per-token
reads. Using the trunk's Q6_K head costs about +0.33 GB per draft token compared with the
MTP file's Q4_0 head. The head choice is therefore a **measured plan decision**, not a fixed
default. On the embedded (unsloth) pack HALO's default matches llama.cpp.

## D-011 — Bytes per token and the decode ceiling (fact, computed from real headers)
UD-Q4_K_XL trunk ≈ 16.48 GB per step; ggml-org Q4_K_M ≈ 18.25 GB (+10.7%); LM head (Q6_K)
1.04 GB. Ceiling model:

    tok/s ≤ S·τ·η·BW / B_step
    B_step = W_trunk + (n+1)·W_mtp + n·W_head + S·[(1+K)·0.157 GB + ctx·64 KiB]

At S=1, n=0, η=1 and 256 GB/s the ceiling is about **15.2 tok/s**. The measured no-MTP
llama-cli result (12.0 tok/s) is η ≈ 0.79. The TRD §2.3 "empirical contradiction" is
explained by MTP (18–21 tok/s ran with `--spec-type draft-mtp`), not by bandwidth above
nameplate. Every decode target must state its context length (KV read = 64 KiB × ctx).

## D-012 — Recurrent rollback uses per-row state slots (decision; supersedes ARCHITECTURE v1)
The GDN and conv ops optionally write the state after each of the last K rows
(K = n_draft+1; slot 0 = most recent), as llama.cpp does (`delta-net-base.cpp:497-603`).
Rollback selects a slot: no restore copy and no replay forward. Cost: K × ~150 MiB per
sequence. Snapshot + replay was rejected because it adds a full 16.5 GB trunk pass on most
MTP steps. Deferred replay remains the fallback for memory-tight configurations.

## D-013 — GDN prefix checkpoints are committed (decision; supersedes TRD §12.2 "R&D spike")
KV-only reuse gives almost no time-to-first-token win on this hybrid model; llama-server
re-processes fully without a checkpoint (`server-context.cpp:3379-3384`). HALO places full
GDN-state checkpoints at the end of the system/tool preamble, at user-message starts, and
near the prompt end (N−k), with a minimum spacing between them. A hit uses the newest
checkpoint at a position ≤ the longest common prefix, and only the tail is recomputed.
Checkpoints use a planner-owned budget in the **GPU pool**, not the 32 GiB OS pool
(llama.cpp keeps them in host RAM). The template returns message-boundary offsets.
Always-recompute is the fallback.

## D-014 — PENDING HUMAN DECISION: performance targets and canonical pack
1. PR-004's batch-1 ≥ 45 tok/s is physically unreachable (it would need ≥ 742 GB/s). The
   ≥ 70 tok/s MTP-effective and ≥ 120 tok/s 4-agent targets are infeasible at realistic
   acceptance. Proposed replacement NFRs: review §6.
2. The canonical pack (ggml-org Q4_K_M, 3 quant types) differs from the file every baseline
   ran on (UD-Q4_K_XL, 8 quant types, 10.7% fewer bytes per token).

Until the owner decides, HALO **loads both** packs, keeps the PR-004 numbers as recorded
aspirations, and reports against the review's §6 NFRs. GPU kernel work is prioritised for
UD-Q4_K_XL's types (the file on the target machine), starting with Q4_K, Q5_K, Q6_K and Q8_0.
