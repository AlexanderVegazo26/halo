# Real Qwen3.8-27B CPU smoke run (WS-S27)

This run checks one MVP exit criterion (`memory.md` §3a):

> The real 27B CPU smoke run passes, and its greedy tokens agree with llama.cpp CPU where
> HALO's top-1 margin is clearly large.

**Verdict: PASS.**
- The real UD-Q4_K_XL pack loads, `inspect` plans it, and `halo run` generates on the CPU engine.
- MTP was run both off and on.
- On 3 prompts × 32 greedy tokens, HALO's tokens are **identical** to llama.cpp CPU's: 96 of
  96 positions, with no divergence at all.
- That includes positions whose top-1 margin is far below the "clearly large" threshold
  (smallest: 0.033 nats).
- Greedy output with MTP on equals greedy output with MTP off, and drafts were accepted.

The scope is narrow:
- the prompts are short (at most 23 tokens, at most 55 in context);
- the numbers were measured on a shared dev host;
- see "Not verified" at the end.

## Setup

| Item | Value |
|---|---|
| Model | `/root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17 559 178 144 bytes, SHA-256 `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` (matches the HF LFS hash) |
| HALO | HEAD `eba6003` (`git archive` into a private tree `/root/s27-src`), Release, clang 18, `-DHALO_ONLY=…;runtime;api;cli`; `halo 0.2.0`, components api/runtime/profiling/autotune |
| llama.cpp | `/root/llama.cpp/build/bin/llama-server`, commit `bd4f514` (2026-09-24), CPU only |
| Host | WSL2 Ubuntu 24.04 on an AMD Ryzen 7 4800H (Zen 2, 8C/16T), 31 GiB shared with 3–5 other agents building and running ASan suites (load average 12–35 during the runs) |
| Threads | 8 for both engines |
| Context | `--ctx 4096 --parallel 1` (llama.cpp `-c 4096 -np 1`) |

**RAM gating.** Before every model run a script waited until `MemAvailable` was at least 22 GiB and no
ctest, ASan test binary or other model process was running. Waits of up to about 90 minutes happened. The two
engines always ran one after the other. HALO mmaps the weights (peak RssFile 16.1 GiB, which the kernel can
reclaim). Its peak anonymous RSS was 0.54 GiB with MTP off and 0.84 GiB with MTP on. A watchdog would have
killed a run if RssAnon exceeded 6 GiB or `MemAvailable` fell below 3 GiB. It never fired.

## Prompts

The prompts were fed to both engines as the **same token ids**. llama.cpp got the id array as its prompt, which
rules out template and tokenizer differences. Both engines report the same prompt lengths: HALO's
`[halo] prompt N tokens` line and llama.cpp's `timings.prompt_n`. Neither adds a BOS token.

| Id | Kind | Text | Tokens |
|---|---|---|---|
| p1 | raw | `The capital of France is Paris. The capital of Germany is` | 12 |
| p2 | raw | `def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n` | 15 |
| p3 | chat, thinking off | user message `Explain in one sentence what a hash table is.`; ids from `halo template --no-think --tokenize` | 23 |

For p3, `halo run --no-think` built its own prompt and reported 23 tokens, the same count as the
`halo template` ids given to llama.cpp.

## 1. `halo inspect`

Command: `halo inspect <model> --ctx 4096 --parallel 1`, with and without `--json`. It exits 0 in both cases.

- **Census:** 64 layers = 48 Gated DeltaNet + 16 full attention; n_embd 5120, n_ff 17408, 24/4 heads,
  head dim 256, vocab 248320, trained context 262144.
- **Weights:** 16.34 GiB in 866 tensors. The LM head is Q6_K.
- **Quant types:** F32=360, IQ3_S=1, IQ4_NL=6, IQ4_XS=70, Q3_K=3, Q4_K=69, Q5_K=191, Q6_K=56, Q8_0=110.
  All nine are in the D-007 dequantization list.
- **MTP:** embedded, 334.75 MiB, 15 tensors (D-006).
- **State:** 64 KiB of KV per token; 149.62 MiB of GDN state per sequence.
- **Planner:** for `ctx 4096 x 1, MTP draft 2, no_gpu` it reports **fits: 18.25 GiB planned** against
  a HOST budget of 24.25 GiB (safety factor 0.90). The largest context that fits is 96 528. The notes say
  prefix checkpoints are disabled because there is no GPU tier, and that activation and workspace sizes are
  formula estimates.

## 2. `halo run` (greedy, 32 tokens, MTP off and on)

Commands:
- raw prompts: `halo run <model> --raw -p "<text>" --temperature 0 -n 32 --ctx 4096 --parallel 1 --backend cpu --threads 8 --mtp-draft {0|2}`;
- p3: `--no-think -p "<text>"` in place of `--raw`.

Every run exited 0 with `finish length`.

| Prompt | MTP | HALO output (stdout) | Chunks | First chunk (s) | `[halo]` decode tok/s |
|---|---|---|---|---|---|
| p1 | off | ` Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital` | 33 | 63.2 | 0.08 |
| p2 | off | `    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return` | 33 | 41.8 | 0.08 |
| p3 | off | `A hash table is a data structure that uses a hash function to map keys to array indices, enabling efficient average-case constant-time insertion, deletion, and lookup operations` | 33 | 105.4 | 0.09 |
| p1 | on (draft 2) | identical to p1/off | 13 | 42.7 | 0.10 |
| p2 | on (draft 2) | identical to p2/off | 14 | 78.9 | 0.10 |
| p3 | on (draft 2) | identical to p3/off | 14 | 119.2 | 0.08 |

- **MTP on equals MTP off**, byte for byte, on all three prompts.
- **Drafts were accepted.** With MTP on, the 32 tokens reached stdout in 13–14 chunks instead of 33. A chunk
  is one pipe read, and several tokens arriving in one read means one tick produced them. The engine probe
  below reports the exact draft and accept counts.
- **Speed.** Decode takes about 10–14 s per token with MTP off, which is 0.08–0.09 tok/s. That is the CPU
  reference path under heavy host contention, not a performance claim. A 4-token probe run earlier showed
  18–28 s per token while other suites were running.

**Engine probe.** `halo run` prints neither token ids nor draft counts. A scratchpad program
(`s27_engine`, not repo code) calls `runtime::create_cpu_engine` with the same settings: ctx 4096, 1
sequence, 8 threads, draft 2, the profit gate on `Auto` (the `halo run` default), and no prefix cache.

ENGINE_PROBE_TABLE

## 3. Comparison with llama.cpp CPU

**llama.cpp side.**
- One `llama-server -m <model> -c 4096 -np 1 -t 8 --host 127.0.0.1` served three `/completion` requests.
- Each request was `{"prompt": [ids], "n_predict": 32, "temperature": -1, "n_probs": 10, "cache_prompt": false}`.
- `temperature < 0` means greedy sampling. The reported probabilities are a plain softmax over all logits, so
  the top-1 minus top-2 logprob gap equals the top-1 minus top-2 **logit** gap.
- Every generated id equals its own `top_logprobs[0]`.

| Prompt | llama.cpp output | prompt_n | decode tok/s (contended) |
|---|---|---|---|
| p1 | same text as HALO | 12 | 0.08 |
| p2 | same text as HALO | 15 | 0.25 |
| p3 | same text as HALO | 23 | 0.22 |

**HALO side: teacher forcing.** A scratchpad harness (`s27_margins`, a copy of the one validated on the tiny
goldens by the previous S27 agent) runs `models::Qwen35::forward` along **llama.cpp's** token path. At each
position it reports:
- HALO's own argmax and its margin (top-1 minus top-2 logit);
- the rank of llama.cpp's token.

Up to the first divergence, HALO's argmax along that path *is* HALO's greedy sequence. Teacher forcing also
gives the margin exactly at a divergence, and agreement after it. The forward's fused argmax
(`StepResult.argmax`) equalled the harness argmax at all 96 positions. The `halo run` texts above equal the
decoded argmax sequences, which ties the engine path to the harness path.

### Threshold for "clearly large"

The rule was fixed before any divergence was tabulated. As it turned out, there were none.

- **Noise.** ε is the largest |margin_HALO − margin_llama| over the positions where both engines pick the
  same token. Some disagreement is expected: llama.cpp's CPU K-quant dot products quantize activations to
  Q8_K, while HALO's reference path dequantizes weights to F32.
- **Threshold.** A margin is "clearly large" if it is at least **max(2ε, 1.0 nat)**.
- **Why this rule.** A greedy flip needs the two engines' logit gaps to differ by more than the margin. At 2ε a
  flip would take twice the largest disagreement observed. The 1-nat floor means top-1 is at least e ≈ 2.7
  times as likely as the runner-up.
- **Measured.** Over 96 agreeing positions, ε = **0.352** (the p99 is the same; p90 0.240, median 0.085).
  The threshold is therefore **1.0 nat**.

### Results

| Prompt | Positions | Argmax agrees | First divergence | Min margin HALO / llama | Max \|mH − mL\| | Top-2 id agrees |
|---|---|---|---|---|---|---|
| p1 | 32 | 32 | none | 1.036 / 1.154 (pos 26, ` Greece`) | 0.285 | 31/32 |
| p2 | 32 | 32 | none | 0.335 / 0.442 (pos 3, ` <=`); 0.351 / 0.316 (pos 31, ` return`) | 0.352 | 32/32 |
| p3 | 32 | 32 | none | 0.033 / 0.026 (pos 31, ` operations`); 0.047 / 0.089 (pos 25, ` insertion`) | 0.323 | 31/32 |
| **all** | **96** | **96** | **none** | 0.033 | **0.352** | 94/96 |

- **Below the threshold.** 6 of the 96 positions have a HALO margin under 1.0 nat (p2: 3, p3: 3). The
  criterion does not require agreement there, but HALO agreed at all 6.
- **Top-2 mismatches.** The top-2 id differs at 2 positions (p1 pos 30, p3 pos 12). The top-1 still agrees at
  both.
- **Verdict.** The criterion requires agreement wherever the margin is at least 1.0. That holds at 90 of 90
  such positions, and in fact at all 96 positions.

## Commands and artifacts

Every artifact is in WSL under `/root/s27/run2/`:
- `inspect*.txt|json`;
- `run_p{1,2,3}_d{0,2}.{stdout,stderr,time,chunks.json,rss.json}`;
- `ll_p*.{req,resp}.json`;
- `force_p*.stdout` (one JSON line per position);
- `eng_p*_d2.stdout`;
- `cmp_rows.json` (per-position comparison).

The scripts (`s27_*.sh`, `s27_ts.py`, `s27_cmp.py`, and the harness sources `s27_harness/`, `s27_eng/`) are in
the session scratchpad. They are not repo code.

## Not verified / limits

- **Short contexts only.** The longest context was 55 tokens, and every prompt was shorter than one prefill
  chunk. These paths were not exercised at 27B scale:
  - long-prompt chunked GDN prefill;
  - KV paging across many blocks;
  - anything near `--ctx 4096`.
  The tiny-model goldens cover them functionally.
- **Small sample.** There were 3 prompts × 32 tokens, all greedy. Sampling (temperature > 0) was not compared.
- **Tokenizer.** It was not compared with llama.cpp here, because both engines got the same ids. The tokenizer
  is covered separately against HF (D-008, D-015).
- **Speed.** Every tok/s figure was taken on a heavily shared host, so none is a performance measurement.
  llama.cpp's decode rate varied 3× between requests from contention alone. No efficiency NFR (D-014) is
  assessed here.
- **MTP.** Only the embedded pack (UD-Q4_K_XL) was run. The separate-MTP-file path (D-006) is exercised
  elsewhere, by WS-G M6.
- **Hardware.** Nothing ran on a GPU. The status remains "code-complete on dev host, hardware verification
  pending".
