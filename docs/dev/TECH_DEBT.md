# Known technical debt

Each item has a cost (what it hurts), how it was found, and the fix. Owner decisions go in DECISIONS.md, not here.

| # | Item | Cost | Found by | Fix |
|---|---|---|---|---|
| TD-1 | Speculative partial accept copies the chosen GDN slot into the live state, and verify writes K slots including a redundant slot 0. | Extra state traffic per verify: K x state instead of the (K-1) x state assumed by D-012/D-014 (0.157 GB per slot on the 27B model). Test `test_speculative` asserts the current 4 x state for K=4. | WS-G M2 (71cd610), measured on the CPU path | cpu-op slot API: skip slot 0, or let the live state be a pointer-swapped slot. Requires a change to the frozen `cpu/ops.h` slot contract; do it together with the GPU backends. |
| TD-2 | `Error(Kernel)` from a NaN logit is raised inside the forward, after KV has been committed. | The sequence must be `reset()`; that error is not atomic like the validation and KV-exhaustion errors. | WS-G M2 | Check the NaN flag before commit, or roll back KV on that path. |
| TD-3 | HIP chunked GDN state kernel keeps the S column in LDS instead of registers (about 2 waves/SIMD). | Prefill throughput, unmeasured. | WS-K M1 | Cooperative LDS-tiled chunk update, tuned on the EVO-X2. |
| TD-4 | HIP GEMV decodes each element's block header and 6-bit scale again; no wave64 variant compared. | Decode bandwidth efficiency, unmeasured. | WS-K M2 | Decode once per 32-element sub-block with packed loads; add a wave64 variant for the autotuner. |
| TD-5 | (closed by fb86e77) Vulkan S-1 BufferView operand redesign. | — | WS-F2 | Done. |
| TD-6 | API server renders and tokenizes on the HTTP worker before admission, with no stack/time budget (review A-5 partial). | A hostile template or huge prompt ties up an HTTP worker. | WS-I M1 | Bounded render pool; depends on the WS-L template limits. |
| TD-7 | HIP TOP_K does full bitonic sorts of every 2048-chunk (8 rounds at k=1024 over 248,320 logits). | LM-head top-k latency, unmeasured. | WS-K M3 | Radix-select or threshold pre-pass. |
| TD-8 | HIP LM-head argmax reduce is one workgroup per vector over ~62K partials. | Decode-step latency tail, unmeasured. | WS-K M3 | Second partial stage. |
| TD-9 | (closed by WS-K M4) HIP `partial_rope_neox` assumed head h starts at h*head_dim. | — | WS-K M3 | `RopeArgs::head_stride` added. |
| TD-10 | HIP attention decode is one workgroup per (token, head) with no split-K; V re-addressed per output dim; accumulator in compiler-promoted LDS. | Decode attention latency at long context, unmeasured. | WS-K M4 | Flash-decoding split with a combine step, LDS-staged K/V tiles. |
