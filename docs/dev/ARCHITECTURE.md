# HALO v0.2 — Implementation Architecture

Companion to TRD (docs/HALO_TRD_v1.1.md, header v1.2) and docs/DECISIONS.md. This file
fixes the module boundaries and interfaces the code is being built against. It is
deliberately smaller than the TRD: it describes what exists or is being built now.

## Layering (TRD §5; no upward or sideways calls into backend APIs)

```
tools/halo (CLI: run, serve, inspect, devices, bench, tune)
  └─ api          OpenAI + Anthropic HTTP/SSE, /tokenize, /apply-template, /metrics
      └─ scheduler   request queue, state machine, prefix cache, batching (batch-1 first)
          └─ runtime   Engine (owns model + backend + planner), Session (per sequence)
              ├─ speculative  MTP draft/verify, profit gate
              ├─ sampling     greedy fast path, top-k prefilter, CPU sampler chain
              ├─ kv_cache     paged KV for attention layers (+ MTP layer)
              ├─ state        GDN recurrent + conv state per sequence, checkpoints (spike)
              ├─ memory       tier-aware planner (WS-C)
              └─ models/qwen35  forward graph for the reference model
                  └─ Backend interface ── cpu (reference) | vulkan | hip
  model (GGUF, NormalizedModel, inspect) · tensor (dtype, dequant) · tokenizer · template
  hardware (discovery, deployment checklist) · core (error, log)
```

## Key interfaces

### Backend (TRD §6, simplified for v0.2)

The CPU reference backend is the semantic definition of every operator (WS-D). GPU
backends implement the same operator contracts and are accepted only after differential
tests against CPU pass (TRD §30).

```cpp
class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendInfo info() const = 0;                 // name, device, driver, ISA target
  virtual std::unique_ptr<Buffer> allocate(std::size_t bytes, MemoryTier tier) = 0;
  virtual void upload_weights(const model::NormalizedModel&, const MemoryPlan&) = 0;
  // Kernel registry surface (TRD §9/§56): candidate implementations per operator key.
  virtual std::vector<KernelVariant> variants(const OpKey&) const = 0;
  // Single-operator execution, used by differential tests, the autotuner, and
  // per-operator backend mixing.
  virtual void run_op(const OpInvocation&) = 0;
  // Batch-shaped model step: S sequences, each with its own token rows. The StepPlan
  // carries the kernel choice per graph node (from the planner/autotuner).
  virtual void forward(const StepPlan&, std::span<const SeqStep>, StepResult&) = 0;
};
```

Contract: `forward` must equal composing `run_op` over the qwen35 graph, within operator
tolerances; a test checks this on CPU. `SeqStep` carries tokens, positions, which logits
and hidden rows are wanted, a greedy flag, and `n_state_slots` (D-012). The CPU reference
may loop over sequences internally; the *signature* is batch-shaped from day one
(review H-1/H-2).

### ForwardRequest / ForwardResult

- `tokens` (span), `positions` (span), `want_logits` rows (last-only for decode, all for
  MTP verification), `want_hidden` (final normed hidden rows, needed by MTP), greedy flag.
- Result: logits rows or fused argmax ids (greedy fast path, FR-007), hidden rows.

### SequenceState (owned by runtime, storage from kv_cache + state)

- Attention KV: paged blocks for the 16 trunk attention layers (+1 MTP layer), block table,
  logical length. FP16 baseline storage (TRD §15); CPU reference may store fp32 first.
- GDN: per layer recurrent state `[n_v_heads, d_k, d_v]` fp32 + conv state
  `[conv_kernel-1, conv_channels]` fp32 (D-003).
- **Commit/rollback** (MTP verification, cancellation), per D-012: during verification the
  GDN and conv ops write the state after each of the last K = n_draft+1 rows into slots.
  Accepting r tokens means selecting the matching slot. KV rollback truncates the logical
  length, and MTP KV rolls back past `pos_max`. A commit is atomic: slot choice plus length.
  No replay forward. Consistency under cancellation at any point is a tested invariant
  (RR-006), checked by "verify + rollback to r" == "decode r tokens".

### Engine / Session (runtime)

```cpp
class Engine {             // one model, one backend, one memory plan
  static std::unique_ptr<Engine> create(const EngineConfig&);  // load, plan, upload
  Session new_session();
  const ModelInfo& model() const;
  const Tokenizer& tokenizer() const;
  const ChatTemplate& chat_template() const;
};
class Session {            // one sequence; not thread-safe; owned by the scheduler
  void prefill(std::span<const int32_t> tokens);          // chunked internally
  int32_t decode_step(const SamplingParams&);             // + MTP path when enabled
  void truncate(std::size_t n_tokens);                    // prefix reuse / rollback
  SessionStats stats() const;
};
```

### Scheduler (TRD §22)

Request state machine `QUEUED → PREFILL → DECODING → STREAMING → COMPLETED | FAILED |
CANCELLED`. v0.2: a single worker thread. Each tick issues **one batched `forward` over all
DECODING sequences** (plus prefill chunks), so one weight pass serves every stream. Rounds of
independent single-sequence forwards would only time-slice single-stream throughput (review
H-1).

Prefix cache (D-013): refcounted paged KV blocks plus GDN state checkpoints. Checkpoints
sit at the end of the system/tool preamble, at user-message starts (offsets supplied by
the template renderer), and at prompt end − k, with a minimum spacing. A hit takes the
newest checkpoint at or before the longest common prefix, drops the KV blocks past it, and
recomputes only the tail. Checkpoints use a planner-owned budget in the GPU pool;
`preserve_thinking` defaults so history re-renders identically. Always-recompute is the
fallback.

### Cost model (review §3.3)

The CPU reference counts bytes moved per operator per step, and reports the predicted
`B_step` from D-011. CI asserts rollback and checkpoint paths stay within their byte
budgets, so a performance-shaping regression is caught on the dev host before GPU hardware
is involved.

### API (TRD §25)

cpp-httplib server; handlers are thin: parse + validate JSON (size limits), render template,
tokenize, submit to scheduler, stream SSE. Localhost bind by default, optional API key,
request/queue caps (PRD §12).

## Correctness strategy

1. Kernel goldens from transformers' own functions (WS-D).
2. Model goldens from transformers `Qwen3_5ForCausalLM` on the tiny model (hidden 256,
   8 layers, vocab 248,320): per-layer inputs, final hidden, logits rows, greedy decode.
3. MTP golden from a NumPy port of llama.cpp's graph (D-005): catch-up mode now; a
   draft-chain case (h feedback, `pending_h` after accept) is owed.
3b. Rollback equivalence (D-012) and forward == composed `run_op` differential tests.
4. Dequant goldens from gguf-py on real blocks of the published files (WS-A).
5. Tokenizer/template goldens from HF tokenizers + jinja2 (WS-B).
6. Differential CPU vs Vulkan (lavapipe on the dev host; RADV on target) per operator and
   per model step.
