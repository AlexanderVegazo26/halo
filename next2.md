# HALO next2: make Qwen3.8-27B usable by agent harnesses (big context, no 400s)

## Context
`halo serve` on Vulkan works for small chat and tool-call requests (2026-09-29 run, `build-dev`,
Qwen3.8-27B UD-Q4_K_XL). Two things stop a real agent harness (Claude Code driving the
`/home/alexander-vegazo/Documents/repos/agents` suite) from using it:

1. **The context is clamped to 8192 tokens**, whatever `--ctx` says.
2. **`/v1/messages` rejects the harness's requests with a 400.**

Measured facts (a captured `claude -p "hi"` request, `ANTHROPIC_BASE_URL` pointed at a logging stub):
- `messages = [ {role:"user", content:[...]}, {role:"system", content:[...]} ]`. The 400 is
  `messages[1].role: must be user or assistant`, from `src/api/requests.cpp:405`.
- First request is ~25k tokens (chars/4) with 23 tools, `max_tokens: 32000`, `thinking: {type:"adaptive", display:"omitted"}`,
  plus the extra top-level fields `metadata`, `context_management` and `output_config`.
- The suite's agents are long (some ~460 lines), so real sessions go well past 25k.

Target: a 128k-token context (stretch: 256k) on one or two concurrent sequences, and every request shape
Claude Code sends accepted or degraded gracefully, never a 400.

## Fix 1: the 400 (Anthropic Messages compatibility) — small, do first

### 1a. Accept `role:"system"` inside `messages[]`
- **Where:** `append_anthropic_message`, `src/api/requests.cpp:402-405`.
- **Fix:** treat a `system` (and `developer`) message as extra system text. Hoist it into the system prompt,
  appended after the top-level `system`, keeping order. Content may be a string or text blocks; reuse the existing text-block flattener.
- **Care:** the chat template wants system content first. If it lands mid-conversation, hoist it rather than
  rendering it in place. Check `src/template/` and the Qwen template for "system message must be first" errors.
- **Security:** the S-8/A-4 markup-escaping of client strings (`docs/api.md` ~L402) must apply to the hoisted text as well.
- **Tests:** add to `tests/unit/api/`: system in the last, first and middle positions; string vs block content; empty content.

### 1b. Tolerate unknown top-level fields
- Check how `/v1/messages` treats `metadata`, `context_management`, `output_config` (and `anthropic-beta` headers).
  It reached the message loop before failing, so they were not rejected first, but this is unverified.
- **Rule:** ignore, with a warning in the response `warnings[]`, any Anthropic field that has no effect on generation.
  Only reject fields whose *silent* ignoring would change results (images, server tools, prefill).
- `thinking.display: "omitted"` should be accepted; it only affects whether the thinking block is returned.

### 1c. `POST /v1/messages/count_tokens`
- Claude Code calls it for compaction and context accounting. Implement it with the model tokenizer over the
  same rendered template (no generation). Without it the harness falls back to guesses.

### 1d. Streaming shape
- Claude Code always sends `stream: true`. Replay the captured request against HALO and verify the SSE events
  (`message_start`, `content_block_*`, `message_delta`, `message_stop`) parse, including thinking blocks
  (`thinking_delta`, `signature_delta` with the empty signature HALO returns) and `tool_use` with `input_json_delta`.
- Add a golden test built from the captured body (trim the system prompt; keep the structure).

### 1e. `tool_choice`
- `required`/named function/`any`/`tool` return 400 today. Harnesses use them on some calls. Prefer implementing
  them with the existing grammar-constrained decoding (force a `<tool_call>` prefix and a schema for the arguments)
  over rejecting. At minimum, downgrade to `auto` plus a warning instead of a 400.

## Fix 2: the 8192-token clamp (KV pool limited by one 4 GiB allocation)

### Root cause
`src/runtime/engine.cpp:365-392`: the KV pool is ONE backend allocation, and RADV reports
`maxMemoryAllocationSize` = 4 GiB (`backends/vulkan/src/device_info.cpp:269`). The engine first drops the
prefix-cache slots, then halves the context until `blocks * block_bytes < max_import_bytes`.
With `--parallel 2` the run ended at 8192 tokens and no prefix cache.

The machine has 96 GiB of VRAM, so the 4 GiB limit is an API limit, not a hardware one.
The model is ~17 GB, and a 128k context needs the KV for 16 GQA layers only (the 48 DeltaNet layers hold a fixed ~3 MiB state each).
Rough KV size: 16 layers × 2 (K,V) × kv_heads × head_dim × 2 B × tokens. Compute it exactly from `model_->kv_layout(bt).block_bytes()`
before designing, and record the number here.

**Recorded (from `KvLayout`, 27B shape in `memory::qwen38_27b_shape`: 16 attention layers, 4 KV heads x 256 = kv_dim 1024):**
`n_layers x 2 x kv_row_bytes(type, kv_dim)` = **128 KiB/token fp32** (2 MiB per 16-token block), 64 KiB fp16, 36 KiB q8; MTP pool
(1 layer) 8 KiB/token fp32. 128k tokens = 16 GiB per sequence in fp32. (`next.md`/`halo inspect` quote "64 KiB per token": that is
the planner's fp16 default, not what the fp32 pool stores.) The layout is block-major (`block[layer][K|V][token][kv_dim]`).
**Implemented (2a):** `kv_cache::Placement::PerLayer` (one buffer per layer, layer-major slabs; the caller binds
`KvPool::layer_image(l)` with n_layers = 1, layer = 0, so no shader changed). 4 GiB per layer = 512Ki fp32 tokens in total. Not done: a
block-range segment table (descriptor array), so fp32 256k x 2 sequences (32772 blocks needed, 32768 fit) is refused with a Config error;
fp16/q8 KV or a lower `--ctx`/`--parallel` fits. 2c was already implemented (`CpuEngine::expired` polls cancel/deadline every tick,
one prefill chunk per tick; test in test_engine_faults.cpp).

### 2a. Segmented KV pool (the real fix)
- **Change:** make `kv_cache::KvPool` (`include/halo/kv_cache/paged_kv.h`) own N buffers ("segments"), each below the cap,
  with a block id mapped to (segment, offset). A block never straddles segments.
- **Backend interface:** the pool is attached as one State-arena buffer (`kv_pool_->attach`, `engine.cpp:~400`) and the attention
  kernels index it with a single base address. Options:
  1. *Preferred:* pass a small table of segment buffers to the paged-attention and KV-write kernels
     (descriptor array or `VK_EXT_descriptor_indexing` / buffer device address; RADV supports `bufferDeviceAddress`).
     One pointer table indexed by `block_id / blocks_per_segment`.
  2. Cheaper interim: **one pool per layer** (16 buffers). Each layer's slice is 1/16 of the total, so 128k fits under 4 GiB per layer with margin. Kernels already run per layer, so they just bind that layer's buffer. Check whether the layout is layer-major or block-major first (`kv_layout`).
- **Also:** the MTP pool (`mtp_pool_`) has the same cap check (`mtp_blk`); segment it the same way.
- **Then remove** the halve-the-context loop, or keep it only as a guard against the total exceeding free VRAM.
- **Tests:** differential tests against the CPU backend for paged attention with blocks spread across ≥2 segments;
  a test that a sequence longer than one segment's capacity decodes identically to a single-segment run.

### 2b. Bring the prefix cache back
- The prefix cache is disabled whenever the pool does not fit. With segmented pools it can stay on.
- The checkpoint memory (`5 per slot × 2 slots × 149.6 MiB`) is separate and already in VRAM.
- This matters for agents: every turn re-sends the same ~25k-token prefix. Without the cache each turn re-prefills it.
  Measure turn-2 latency with and without.

### 2c. Prefill speed at 25k+ tokens
- Prefill for a 25k prompt is the first thing a harness feels. Measure tokens/s for prefill at 1k, 8k, 32k
  (`halo bench model`) and check that the chunked prefill (`prefill_chunk`) does not go quadratic in attention cost.
- If prefill is far below decode-batch speed, this is the next bottleneck after the context fix; add it to `next.md`'s plan.
- `request_timeout` (600 s) does not interrupt prefill (`docs/api.md` L181), so a stuck long prefill hangs a slot. Make prefill cancellable at chunk boundaries.

### 2d. Attention cost and memory at long context
- Verify the Vulkan GQA attention kernel handles 128k keys: workspace/scratch sizes that scale with context
  (scores buffers, split-KV partials) must also stay under the 4 GiB cap and must not be sized for 8k.
- Verify the activation/workspace estimate in the memory plan ("formula estimates until a backend reports measured sizes")
  against a real 128k run. A wrong estimate here means an OOM late in a long session.

### 2e. Config and defaults
- Default `--ctx` is 32768 and `--parallel` is 4. With Claude Code's 25k prefix, 4 slots × 128k is too much KV.
  Suggested serving profile for agents: `--ctx 131072 --parallel 2`. Print the resulting KV/VRAM budget at startup (it prints the plan; add the KV pool total and segment count).
- `halo serve` should **fail loudly** rather than silently clamp when `--ctx` was set explicitly and cannot be honoured;
  the clamp warning was easy to miss and `/v1/models` then reports `context_length: 8192`.
- Expose the true window to clients: `/v1/models` already returns `context_length`; document that harnesses should set
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to it (Claude Code assumes 200k for unknown models otherwise).

## Fix 3: robustness under harness traffic

- **Context overflow must be a clean, typed error**, not a crash or truncation: Anthropic shape
  `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: N tokens > M maximum"}}`. Claude Code parses that message to trigger compaction.
- **`max_tokens` 32000 + prompt must fit the context**: today an oversized `max_tokens` is clamped with a warning (good). Confirm this holds with a 25k prompt in a 32k context and gives a usable answer budget.
- **Thinking budget:** `adaptive` mapped to a 50-token reasoning budget in the earlier test ("reasoning budget of 50 tokens reached").
  For agent work that is far too small. Make the adaptive default scale with `max_tokens` (e.g. up to min(8192, max_tokens/2)) and document it.
- **Concurrency:** Claude Code fires a small "haiku-class" request (titles, summaries) alongside the main one. With `--parallel 2`
  the second slot is used for it; confirm `max_queue`/utility lanes don't starve the main request.
- **Tool-call parsing:** run a multi-step tool loop (Read → Edit → Bash style, 10+ turns) and check that `<tool_call>` parsing,
  parallel tool calls, and `tool_result` round-trips (including `is_error`) stay byte-exact.

## Fix 4: agents repo side (`/home/alexander-vegazo/Documents/repos/agents`)
- Only `qa-runner` pins a model (`sonnet`, `sdlc-suite/agents/...`; the other agents `inherit`). To run the suite fully on HALO,
  set `model: inherit` for it, or add a documented override. This file is generated into six trees: edit `sdlc-suite/` and
  run `python sdlc-suite/tools/generate_trees.py`, never the generated copies (see that repo's `CLAUDE.md`).
- Add a short "running on a local HALO endpoint" section to `sdlc-suite/USAGE.md`:
  ```
  halo serve --model Qwen3.8-27B-UD-Q4_K_XL.gguf --backend vulkan --ctx 131072 --parallel 2
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=x \
    ANTHROPIC_MODEL=Qwen3.8-27B ANTHROPIC_SMALL_FAST_MODEL=Qwen3.8-27B \
    CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 claude
  ```
- Local-model caveat to document: a 27B Q4 model will follow the long, strict agent definitions less reliably than the hosted models.
  Pick one or two low-risk agents/workflows as the acceptance test first.

## Suggested order
1. **1a** (system role) + **1b** + **1d**: one small PR; unblocks the harness at the current 8k... but 8k is too small for the 25k prompt, so this alone is not enough.
2. **2e** fail-loud on unhonoured `--ctx`, so the problem is visible while 2a is worked on.
3. **2a** segmented KV pool (per-layer buffers first, if the layout allows), then **2d** verification at 32k → 128k.
4. **2b** prefix cache back on; **1c** count_tokens; **3** overflow error and thinking budget.
5. **2c** prefill speed and cancellation; **1e** tool_choice; **Fix 4** docs and the `qa-runner` change.

## Acceptance test
1. `claude -p "Reply with the single word: ready"` from the agents repo against HALO returns `ready`, streaming, no 400.
2. A prompt of ~100k tokens (e.g. cat several repo files) is accepted, answered, and `/v1/models` reports `context_length ≥ 131072`.
3. Turn 2 of a multi-turn session with the ~25k harness prefix shows `cached_tokens` ≈ prefix length and a much lower time-to-first-token than turn 1.
4. A 15-turn tool loop (Read/Edit/Bash) completes with every tool call parsed.
5. A deliberately oversized prompt returns the typed "prompt is too long" error and Claude Code compacts instead of failing.
6. All new differential tests pass against the CPU reference; `scripts/build.sh` full suite stays green.
