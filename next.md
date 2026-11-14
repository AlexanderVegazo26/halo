# HALO decode-speed plan: 16.8 → ~25+ tok/s

## Context
HALO is at **16.8 tok/s** (MTP k=2) and **11.0 tok/s** plain, measured today on the EVO-X2. The model is Qwen3.8-27B UD-Q4_K_XL on Vulkan/RADV. For comparison, llama.cpp gets 12.0 plain and 22 with MTP. The easy wins are already done: SWAR matvecs, a device-resident GDN ring, and a device-resident KV pool. The GPU is now busy most of the time, so the remaining speed has to come from three places:
- getting more useful work out of each byte of weights read;
- cutting the GPU work that isn't trunk-weight reads;
- emitting more tokens per tick.

(The nected.ai article is about rule engines and the Rete algorithm, so it doesn't apply here. The envisioning.com link returned HTTP 429.)

### Fresh profile (`HALO_VK_OP_TIMINGS`, 2026-09-29, `-n 32`, 13-token prompt)
The run was 44 forwards: 1 prefill, 15 verify (R=3) and 28 MTP drafts. That comes to ≈2.1 tokens/tick and **≈129 ms wall per tick**.

| Cost per tick (approx.) | ms | Note |
|---|---|---|
| Trunk matvecs at R=3 (verify) | **~109** | The ideal is ~80 ms (16.5 GB at ~210 GB/s). Batching 3 rows costs ~35% more than R=1. |
| 3× LM head, Q6_K 1 GiB, 5.1 ms each | **~15** | 2 of the 3 are MTP drafts that use a full 248k-vocab head. |
| `gated_delta_rule_decode` 48 × ~167 µs | **~8** | 3 MiB state per layer; about 4× slower than the bandwidth limit. |
| MTP block matvecs, 2 × ~1.7 ms | ~3.5 | |
| Small ops: norms, eltwise, q8_0 gemvs, conv, rope | ~3 | ~400 dispatches per forward |
| Syncs and host round-trips | rest | 6 fence waits per tick, plus hidden-state download and upload for MTP |

## Recommended work, in order of payoff

### 1. Batched matvec at weight-bandwidth speed (biggest win, ~−25 ms/tick)
- **Problem:** the `BATCHED` path in the `backends/vulkan/shaders/common/matvec_*_main.glsl` files is slow. Line ~127-137 in `matvec_q5_k_main.glsl` re-dequantizes every block once per vector (`q5k_apply_` is called per `t`). It then runs `n_vec` serial shared-memory tree reductions, each with barriers.
- **Fix:**
  - dequantize each block once into registers, then FMA it against all `n_vec` x-vectors;
  - replace the tree reductions with `subgroupAdd` (the workgroup is one wave, WG=64), or with a single reduction that covers all vectors;
  - specialize `MAX_VEC` for the actual R (1-4) through a specialization constant so the loop unrolls.
- **Where:** do the hot types first (q5_k, q6_k, iq4_xs, q4_k), then iq4_nl, q3_k and iq3_s. The dispatch site is `Ops::gemv_impl` in `backends/vulkan/src/ops.cpp:163`.
- **Target:** R=3 should cost ≤1.1× R=1. Verify with the existing `test_vulkan` VkMatvec tests plus a micro-benchmark (R=1 vs R=3, GB/s).
- This also makes step 5 (deeper speculation) cheap.

### 2. Cheaper MTP draft head (~−9 ms/tick)
- Each draft runs `head()` (`src/models/qwen35.cpp:879-881`) over `M.lm_head`, which is the full 248,320-row Q6_K matrix, about 1 GiB.
- **Option A (low risk):** if the MTP `lm_head` is byte-identical to the trunk head, dedupe it at load in `qwen35.cpp:605-616`. This saves 1 GiB of VRAM but no time.
- **Main fix:** a reduced-vocab draft head, the FR-Spec technique.
  - At load, build a sub-matrix of the ~32k most frequent tokens. Rank frequency by tokenizer merge rank, or use a shipped list.
  - Drafts take the argmax over the sub-vocab and map it back to full token ids.
  - Verification still uses the full trunk head, so output stays bit-identical to greedy.
- **Cost:** 5.1 ms → ~0.7 ms per draft.
- **Where:** `src/models/qwen35.cpp` (`head`, `mtp_forward`, MTP load), plus an argmax index remap.

### 3. Faster `gated_delta_rule_decode` (~−6 ms/tick)
- The kernel takes ~167 µs per layer to move ~3 MiB in plus 3 MiB out (plus slot writes), when ~40 µs would be bandwidth-bound.
- **Fix:**
  - check the occupancy and memory-access pattern in `backends/vulkan/shaders/linear_attn/gated_delta_rule_decode.comp`: coalesced state rows, one workgroup per (head, d_v tile), state held in registers across the R tokens;
  - write only the rollback slots that are actually needed;
  - make sure R=3 reads and writes the state once rather than 3 times.
- **Gate:** the existing differential tests against the CPU `backends/cpu/gated_delta_rule.cpp`.

### 4. One submission per tick: remove host round-trips (~−5-10 ms/tick)
- **Drop the mid-forward sync.** `st.sync()` at `qwen35.cpp:773` (and at `:863` in MTP) sits before `head()`. Record the head into the same command buffer so there is one submit and one wait per forward.
- **Keep the MTP hidden state on the device.** `forward` downloads `hbuf` (`qwen35.cpp:786-789`) and `mtp_forward` uploads it again (`:855-857`). Pass a device `TensorRef` through `SeqOutput` and `MtpStep` instead.
- **Chain the drafts on the GPU.**
  - The draft 1 argmax index feeds `get_rows` directly, using device ids, for draft 2.
  - Draft 2 and verify token ids are also produced on the device.
  - Result: drafts plus verify become **one submit and one fence per tick**, down from 6.
  - Touch points: `src/speculative/speculative.cpp:119-304` and the `Backend` interface (`include/halo/backend/backend.h`).
- **Reuse the stream.**
  - `Step` creates a new `Stream` for every forward (`qwen35.cpp:237`), with its own command pool, fence and descriptor pools. Keep a small per-model pool of streams.
  - The 512-dispatch cap (`stream.cpp:179`) currently forces a submit and wait mid-forward. Rotate 2-3 command buffers with their own descriptor pools so a split doesn't need a CPU wait.
- **Cut per-dispatch CPU cost.**
  - Use `VK_KHR_push_descriptor`, or cache descriptor sets per (kernel, bindings) because decode shapes don't change.
  - Swap the `std::unordered_map touched_` and the per-dispatch `std::vector` allocations in `Stream::dispatch` for small fixed arrays.
  - Longer term: record the fixed-shape decode command buffer once and replay it, with positions and lengths read from a device buffer.

### 5. More tokens per tick
- Once step 1 makes R=4-8 nearly free, try `--mtp-draft 3` and measure. The profit gate already exists in `speculative.cpp`.
- **Tree verification** (implemented, `HALO_MTP_TREE=1`): the top-2 at draft depth 1 verified as a small tree in one forward, through `attention_tree.comp` and a GDN leaf slot. Correctness is GPU-verified (status table); its tokens/tick gain is not measured yet (needs an idle GPU and the 27B).
- Target: 2.1 → ~2.6-3 tokens per tick.

### 6. Kernel fusion (small ops, ~−2-3 ms/tick)
- **FFN:** fuse gate+up+SwiGLU. Either use a dual-matrix gemv that writes `silu(g)*u` directly, or concatenate the weights at load when their quant types match (`qwen35.cpp:467-473`).
- **GDN projections:**
  - merge the `beta` and `alpha` q8_0 gemvs, which are tiny (48×5120), latency-bound and ~1900 dispatches per run, into one dispatch;
  - fold `gdn_gates` into that dispatch's epilogue.
- **Attention:** merge the q/k/v gemvs into one dispatch.

### 7. Separate track: prefill and TTFT
- Prefill uses matvec loops and is ~10× slower than llama HIP (140 t/s). Add a real tiled matmul using `VK_KHR_cooperative_matrix` (WMMA on gfx1151) for R > 8. Speed up `gated_delta_rule_chunked` (1.2 ms per layer).
- This matters for agent and long-context workloads, but it doesn't affect decode tok/s.

## Additions from exllamav2 (turboderp-org/exllamav2)

What exllamav2 does, and whether each idea transfers to HALO:
- **What transfers:** a KV cache stored in Q4 or Q8 instead of fp16 (the README says Q4 "performs better" than FP8), mixed-bitrate quantization (EXL2, 2-8 bpw), fused projection and MLP kernels, weights reordered into a GPU-friendly layout at load time, and speculative decoding (a draft model or n-gram lookup) in its dynamic generator.
- **What HALO already has:** dynamic batching and prefix-cache deduplication, through its paged KV cache with copy-on-write.

### 8. Repack weights into a kernel-friendly layout at load (~5-10% on trunk matvecs)
- exllama shuffles its quantized weights once at load so the kernels can do aligned 128-bit loads and unpack in bulk.
- **For HALO:** at upload (`make_mat` in `qwen35.cpp:153-165`), split the GGUF Q5_K, Q6_K and IQ4_XS blocks (176, 210 and 136 B) into separate 16-byte-aligned planes: `qs`, `qh`, `scales` and `d`.
  - The SWAR mains can then use `uvec4` loads.
  - This removes the funnel-shift path Q6_K currently needs for its 210 B blocks.
- **Target:** take the matvecs from ~200 GB/s toward llama.cpp's ~244 GB/s ceiling on this machine.
- **Pairs with step 1:** repack while rewriting the batched path.
- **Implemented (opt-in, unmeasured):** `HALO_REPACK=1` repacks the `make_mat` Mats (Q5_K/Q6_K/IQ4_XS, Vulkan only) with `tensor::gemv_repack` (`include/halo/tensor/repack.h` documents the plane layouts) and gemv runs them as Vulkan gemv variant `backend::kGemvRepacked` (shaders `matvec_{q5_k,q6_k,iq4_xs}_rp`; same apply/reduction expression as the raw kernels). Fused-projection buffers (`HALO_FUSE_GEMV`) and tables stay raw. Kernel bitwise gate vs the raw kernels: done on RADV 2026-11-14 (VkHead.GemvRepacked*). Still to do: a model-level check (qwen35 with `HALO_REPACK=1`: greedy tokens equal to raw) and a decode measurement on an idle GPU.

### 9. Quantized KV cache: fp16, then Q8 or Q4 with group scales
- HALO stores KV in **fp32**, 64 KiB per token.
- **Why it matters:**
  - it limits context, because of the 4 GiB single-buffer cap that triggers the ctx clamp in `6c19ef4`;
  - at long context, attention reads KV every token: at 32k context that is about 2 GiB per token, ≈10 ms.
- **Path:**
  - store KV as fp16 first (2× smaller);
  - then Q8 or Q4 with per-32 group scales, the way exllama does it (4-8× smaller);
  - `kv_write.comp` quantizes on write, and `attention.comp` dequantizes on read.
- **Gate:** accuracy against the CPU golden tests, with a stated tolerance.
- **Payoff:** mostly for agent and long-context workloads. It also removes the context clamp.

### 10. Fuse the norm into the matvec prologue
- This is the exllama-style fused-kernel idea taken one step further than step 6.
- A gemv whose input is `rms_norm(x)*w` would compute the row's RMS itself and apply `w` as it loads x. The separate `xn` write and read, and one dispatch per norm, disappear.
- **Where:** `add_rms_norm` → gemv in `decoder_layer` (`qwen35.cpp:477-486`).

### 11. N-gram / prompt-lookup drafts alongside MTP
- exllamav2's dynamic generator does speculative decoding. The general technique is to find a matching n-gram in the prompt or earlier output and propose the tokens that followed it, at no model cost.
- Agent and code-editing output repeats long spans of the context.
- **Plan:**
  - when an n-gram match exists, propose up to 8 tokens and verify them in one R≤9 forward. This becomes cheap after step 1.
  - otherwise fall back to MTP;
  - greedy output stays identical because the trunk still verifies every token.
- **Where:** `src/speculative/speculative.cpp` (draft source), using the existing profit gate.

### 12. Owner decision: mixed-bitrate quant (the EXL2 idea)
- UD-Q4_K_XL is ≈5.2 bpw (16.34 GiB for 27B params).
- Decode is bandwidth-bound, so fewer bytes mean proportionally more tok/s. A measured per-layer mix averaging ~4.0 bpw would be ~+25%.
- This conflicts with D-014 (the canonical pack) and needs a calibration and quality study. List it as an option, not as planned work.

## Expected result
- Steps 1-4 take the tick from ~129 ms to ~85-90 ms at 2.1 tokens per tick, which is **~24 tok/s**.
- Adding step 5 at ~2.7 tokens per tick gives **~30 tok/s**.
- Steps 8 and 10 add another ~5-10%. Step 11 lifts tokens per tick a lot on repetitive agent and code output. Step 9 matters at long context.
- Plain decode is bounded by weight bandwidth at ~12.5-13 tok/s: 16.5 GB at ~210 GB/s.

## Verification (every step)
1. **Correctness:**
   - run `scripts/build.sh` with `test_vulkan` and the forward differential tests;
   - greedy output must be token-identical with MTP on and off, and must match the CPU reference on the tiny goldens.
2. **Speed:** run the same command before and after each change:
   `build-dev/tools/halo/halo run <model> -p "<13-tok prompt>" --raw --temperature 0 --backend vulkan -n 128 --ctx 4096 --parallel 1`, with and without `--mtp-draft 0`. Baseline: 16.79 / 10.97 tok/s.
3. **Attribution:** re-profile with `HALO_VK_OP_TIMINGS` and aggregate by kernel, as in the table above. Also confirm the per-tick wall time and fence count drop.
4. Commit each step separately, with the before and after tok/s in the message, matching the repo's existing convention.

## Implementation status (2026-09-29, compile-checked only; nothing benchmarked or tested)
All new paths are opt-in (env flag, default off) except the `stream.cpp` per-dispatch allocation removal and the fused batched reduction (bitwise-neutral by construction).

| Step | Flag / state |
|---|---|
| 1 batched matvec | shared-tree fusion (default); `HALO_SUBGROUP_REDUCE=1` subgroupAdd + unrolled NV variants (q4_k/q5_k/q6_k/iq4_xs). **GPU-verified 2026-11-14 (RADV):** within the CPU bound for n_vec 1-17; it now also runs the n_vec = 1 chunks, because R=1 on the tree shader and R>1 on subgroupAdd broke the bitwise batch == single contract (MTP verify vs decode); `OpsOptions::subgroup_reduce` overrides the env (VkHead gemv test covers both). Qwen3.8-27B greedy 64 tokens identical with MTP on/off and vs the tree path. tok/s not measured (GPU busy) |
| 2 draft head | `HALO_MTP_DRAFT_VOCAB=32768` (first-N-ids prefix; not a frequency list). **GPU-verified 2026-11-14 (RADV):** on the tiny f32 model (vocab 248320), an Argmax MTP step with N rows (1, 7, V/3, V-1) returns exactly the argmax of the full head's logits over [0, min(N, valid_vocab)) with the same logit value (lowest index on ties), a Full step on the reduced model still scores every row (logits bit for bit the full head's), and N > V clamps to the full head; with N = V/8, speculative decoding drafts only prefix ids (the full head drafted 228 ids outside it), chain and chain + device hidden draft exactly what the per-depth path drafts, and greedy tokens and trunk KV / GDN / hidden / MTP KV equal plain decoding for k = 1..3 on p0-p2 (VulkanSpec.ReducedDraftVocab*). `Qwen35Options::mtp_draft_vocab` overrides the env per instance and `Qwen35::mtp_draft_vocab()` reports the effective rows. Acceptance rate vs the full head on the 27B and tok/s not measured (GPU busy) |
| 3 GDN decode | `HALO_GDN_DV_TILE=32`; `HALO_GDN_REG=1` register-resident state kernel (d_k <= 128). **GPU-verified 2026-11-14 (RADV):** both (and combined, incl. uneven tiles 48/5 and d_v > workgroup) hold the CPU bound over 20 steps and multi-row slots, keep ring == slot form, and are bitwise the reference shader (out, final state, slots), so a flag cannot change greedy output; `OpsOptions::gdn_decode_reg` / `gdn_dv_tile` override the env per Ops (VkGdn.DecodeVariants*). tok/s not measured (GPU busy) |
| 4 host round-trips | mid-forward sync removed; `HALO_STREAM_REUSE=1`; `HALO_MTP_DEVICE_HIDDEN=1`; `HALO_MTP_CHAIN=1` (chained drafts, one submit for the draft phase; verify still its own forward). **GPU-verified 2026-11-14 (RADV):** on the tiny f32 model, k = 1..3, prompts p0-p2, each flag alone and chain + device hidden give every tick's drafts bit for bit equal to the per-depth host path, greedy tokens equal to MTP off, and identical trunk KV / GDN / hidden / MTP KV; pooled streams are bitwise neutral (VulkanSpec.*). `SpecConfig::device_hidden` / `chain` and `Qwen35Options::reuse_streams` override the env per instance. tok/s not measured (GPU busy) |
| 5 tokens/tick | `HALO_MTP_TREE=1` (chain + one runner-up leaf); `--mtp-draft 3` needs no code. **GPU-verified 2026-11-14 (RADV):** on the tiny f32 model, tree verify with the host or device hidden gives greedy tokens and trunk KV / GDN / hidden / MTP KV bit for bit equal to plain decoding for k = 1..3 on prompts p0-p2 (14 of 15 ticks verify a tree); a forced runner-up d1' is kept with the leaf row's argmax from `attention_tree.comp` equal to plain decode's, and committing it (KV row move + GDN leaf slot) or only x leaves the sequence bitwise plain decode; a correct chain or a wrong d1' is never replaced by the leaf (VulkanSpec.Tree*). Mutants of the shader's root-path key mapping and of `kv_move_row` (V row) fail it. tok/s not measured (GPU busy) |
| 6 fusion | `HALO_FUSE_GEMV=1\|all\|ffn,ba,qkv` (gemv-level; SwiGLU/gates not folded into the epilogue). **GPU-verified 2026-11-14 (RADV):** on the tiny f32 model, each form (ffn, ba, qkv) and all three give tokens, MTP drafts (k = 3) and trunk KV / GDN / hidden / MTP KV bit for bit equal to the separate gemvs, at R == 1 (one fused qkv dispatch) and R > 1 (row slices); `Qwen35Options::fuse_gemv` overrides the env per instance and `Qwen35::fused_projections()` reports what loaded (VulkanSpec.FusedProjectionsAreBitwiseTheSeparateGemvs). tok/s not measured (GPU busy) |
| 7 prefill | `HALO_COOPMAT=1` cooperative-matrix GEMM (q4_k/q5_k/q6_k), chunked-GDN step-6 speedup |
| 8 repack | `HALO_REPACK=1`. **GPU-verified 2026-11-14 (RADV):** `Ops::gemv_repacked` over `tensor::gemv_repack` rows is bit for bit the raw-layout tree gemv for Q5_K / Q6_K / IQ4_XS at 1, 3 and 20 blocks per row (padded and unpadded strides), n_vec 1-17, view offsets 0/8/48, with subgroup_reduce on or off (the `_rp` shaders always use the tree reduction), and as row slabs stepping by the repacked stride (VkHead.GemvRepacked*); layout mutants in repack.cpp fail it. Not yet loaded through qwen35 on a device; tok/s not measured (GPU busy) |
| 9 KV | `HALO_KV_FP16=1` / `HALO_KV_TYPE=q8`; per-layer pool segments (next2.md 2a). **GPU-verified 2026-11-14 (RADV):** no test had ever run the fp16 / q8 shaders. kv_write_f16 used `packHalf2x16`, which RADV rounds toward zero (65520 -> 65504 instead of inf, ties and about half of all values 1 ulp short: a bias toward zero in every stored K/V); it now converts with an integer round-to-nearest-even bit for bit `tensor::fp32_to_fp16` (overflow, NaN, half subnormals; VkKv.KvWriteF16StoresRoundToNearestEvenHalves, red on the old shader; mutants of the tie, subnormal, overflow and NaN branches fail it). kv_write_q8 holds its contract (scale within 3 ulp of max\|x\|/127, q the nearest int8, the max at +-127, zero groups zero) and leaves other rows byte for byte; attention_f16 / attention_q8 match `cpu::attention_gqa` on the decoded values within the fp32 bound at qwen35 shapes ([Q\|gate] stride, 3 and 8 tiles, workgroups 32-256, large scores, bad block), and the kv_write -> attention seam in one recording (VkKv.*F16* / *Q8* / Quantized*). tok/s and quality vs fp32 at model level not measured (GPU busy) |
| 10 norm fusion | `HALO_FUSE_NORM=1` (FFN gate/up at R==1 only). **GPU-verified 2026-11-14 (RADV):** the fused gemv was NOT bitwise rms_norm + gemv (about a third of outputs 1 ulp off: the compiler built the two RMS computations differently), so R == 1 fused decode could disagree with the R > 1 MTP verify; the RMS math is now `precise` in rms_norm, add_rms_norm and the four `_normed` shaders, and the fused gemv is bit for bit that row of the unfused batch for Q4_K / Q5_K / Q6_K / IQ4_XS, cols 256-15872, view offsets (VkHead.GemvFusedNorm*). The staged row must fit in shared memory: `cols` = 16384, which the op accepted, crashed RADV's compiler (SIGFPE); the cap is now `Ops::fused_norm_max_cols()` (16047 on RADV). qwen35 fuses only within `Limits::max_fused_norm_cols` (0 with HALO_SUBGROUP_REDUCE or reduce_workgroup != 256: the `_normed` shaders keep the tree). tok/s not measured (GPU busy) |
| 11 n-gram | `HALO_NGRAM_DRAFT=1`. **GPU-checked 2026-11-14 (RADV):** the lookup is `speculative::ngram_draft` (include/halo/speculative/ngram.h; longest recurring suffix 8..3 wins, then the latest start, overlaps allowed, capped at k and at the end of the context; NgramDraft.*). `CpuEngineOptions::ngram_draft` overrides the env per engine; `EngineStats::ngram_lookups` / `ngram_drafts` count the searches and hits. The tiny model never repeats a 3-gram (no hit in 300 greedy tokens from p0-p2), so on the device: forced drafts (the path n-gram drafts take; an oracle wrong at a rotating depth) are verified over Vulkan with greedy tokens equal to plain decoding, every acceptance count 0..3, fewer decode ticks, one trunk forward per tick, and batched with an MTP-drafted and a sampled sequence each equal to running alone; n-gram on without a match is bitwise MTP-only drafting (same drafts, tokens, ticks) with one lookup per greedy decode step (VulkanEngine.*). Hit rate and acceptance on the 27B and tok/s not measured (GPU busy) |
| 12 mixed-bitrate | not implemented: owner decision (conflicts with D-014) |

Verify on the EVO-X2 in this order: full test suite (differential + VkHead/VkViews bitwise gates), baseline tok/s, then each flag alone.
