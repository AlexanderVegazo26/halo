# Model-file parser hardening (WS-L)

This file tracks the mitigations for the model-file findings in
`docs/reviews/2026-09-24-security-review.md`. For each finding it records the mitigation,
the tests that cover it, and anything not done. Model files (GGUF metadata, chat template,
`tokenizer.json`) come from Hugging Face and are treated as hostile (review A-8).

| Finding | Milestone | Status |
|---|---|---|
| S-3 template DoS | M1 | mitigated |
| S-8 reserved `extra_context` keys | M1 | mitigated |
| S-10 `range()` off-by-one, macro leak | M1 | mitigated |
| S-6 quadratic tokenizer | M2 | pending |
| S-7 tokenizer.json allocation | M2 | pending |
| S-4 GGUF metadata amplification | M3 | pending |
| S-9 output-parser nesting | M3 | pending |
| S-5 SIGBUS on truncated mapping | — | out of scope (see below) |

## Chat template (M1)

### Mechanism

- **Pinned minja patch.** The patch is `cmake/patches/minja-halo-limits.patch`, applied to
  minja `021c229`.
  - The FetchContent `PATCH_COMMAND` runs `cmake/patches/apply_patch.cmake`. That script checks
    the patch file's SHA256 (recorded in `cmake/HaloDeps.cmake`), is idempotent on an
    already-patched tree, and fails configure on a tree that is neither pristine nor patched.
  - `URL_HASH` still verifies the upstream archive.
  - Existing build dirs were tested:
    - A build dir with an unpatched `_deps/minja-src` is patched on its next configure.
    - Further configures, or re-running the script, report "already applied" and never
      apply the patch twice.
    - A tree that is neither pristine nor patched fails configure with
      `Delete <build>/_deps/minja-* and reconfigure`, before any file is touched.
    - A tampered patch file fails its SHA256 check.
  - `cmake/patches/.gitattributes` disables line-ending conversion, so the hash stays stable.
- **Fail closed.** `src/template/chat_template.cpp` has `#error` unless the patch marker
  `MINJA_HALO_LIMITS` is present.
  - This matters because `-DFETCHCONTENT_SOURCE_DIR_MINJA` bypasses both the hash and the patch.
  - Without the check, a stale `_deps` directory would silently build unlimited minja.
- **Budget.** `minja::RenderBudget` is installed thread-locally by `ChatTemplate::render`.
  With no budget installed, minja behaves exactly as upstream.
  - A violated limit throws `minja::LimitExceeded`, which unwinds without per-frame error
    decoration.
  - HALO maps it to `halo::Error(Api)` with the message prefix `chat template limit exceeded: `.
- **Configuration.** `halo::chat::TemplateLimits` (in `include/halo/template/chat_template.h`)
  holds the limits.
  - It is passed through a new constructor overload. The existing constructor uses the defaults.
  - `RenderResult::stats` reports what each render used.

### Error codes

- **At load (constructor):** `Error(Config)`, raised for source size, parse depth, or any
  other parse failure.
- **At render:** `Error(Api)`. The codebase has no template-specific error code; `Api` is
  what `render()` already raises for template failures.

### Limits and defaults

The "measured" column comes from the Qwen3.8 ggml and unsloth templates rendering a
4002-message agentic conversation with 64 tools. That prompt is 2.2 MB, about twice a
262K-token context. The test `DefaultsLeaveHeadroomOverRealTemplates` asserts at least 4x
headroom for every limit.

| Limit | Default | Measured | What it bounds |
|---|---|---|---|
| `max_source_bytes` | 64 KiB (**was 1 MiB**: a deliberate behaviour change) | 9–10 KB | source size |
| `max_parse_depth` | 256 | 16 | parser recursion |
| `parse_stack_bytes` | 256 MiB, virtual | — | stack of the dedicated parse thread (std::regex in the lexer recurses once per character) |
| `max_render_depth` | 1024 | 20 | nested node renders, expression evaluations, and value dumps/compares |
| `max_steps` | 4,000,000 | 0.86M | total work; about 1 µs/step, so about 4 s worst case on the dev host (D-001) |
| `max_loop_iterations` | 1,000,000 | 36K | for-loop iterations |
| `max_output_bytes` | 64 MiB | 3.6 MB | bytes written by output nodes across nested buffers |
| `max_string_bytes` | 32 MiB | — | largest single string or container one operation may build (checked *before* repeat, replace, join, indent, split, and tojson indent) |
| `max_alloc_bytes` | 1 GiB | 21 MB | cumulative string bytes produced or stored |


**Stack.** Render runs on the caller's thread. At the depth cap it touched about 1.2 MiB of
stack in RelWithDebInfo and about 2.9 MiB in ASan+Debug (measured by stack painting). The
header declares `kMaxRenderStackBytes = 4 MiB`, and a test enforces it. Review requirement
A-5 (a bounded render worker pool) still belongs to the API layer.

### Patch contents

Every hunk is marked `// HALO`.

1. Counters: `DepthGuard` in `TemplateNode::render`, `Expression::evaluate`, `Value::dump`
   and `Value::operator==`; loop iterations in `ForNode`; output bytes in the Text,
   Expression, Filter and Call nodes.
2. Size checks: a pre-check on `str * n`, a check on every expression result, and growth
   checks in replace, join, indent, split and tojson indent.
3. Allocation charge: every store into a container is charged.
4. Parser: a depth guard on the six recursive entry points (`Options::max_depth`).
5. Lexer:
   - `match_continuous` on token regexes, which removes the O(rest-of-source) scan per
     failed token;
   - a linear comment scan in place of the recursive `[\s\S]*?` regex;
   - linear `{%-`/`-%}` stripping in place of `regex_replace`.
6. Values:
   - iterative teardown of deeply nested values (the implicit destructor overflowed the
     stack);
   - reference cycles are refused on store (`a.append(a)` would otherwise recurse forever
     and leak);
   - error decoration stops at 16 KiB, so an error raised deep in a long line does not
     grow the message quadratically.
7. Builtins:
   - `split('')` raises, where it previously looped forever;
   - `'' * huge` returns immediately;
   - `replace` is built in one pass, which removes quadratic in-place replacement;
   - substring searches (`in`, strip, split, replace) are charged against the step budget.
8. Contexts: macro and `{% call %}` contexts are recorded, so `ContextCycleBreaker` also
   clears macros defined inside loops and macros. This fixes the leak noted in S-10.

### Tests

Test files are `tests/unit/template/test_template_limits.cpp` and the existing
`test_template.cpp`.

**Harness.** Every hostile case runs in a gtest death-test child (`threadsafe` style).
- The child runs the case on a `std::thread` with the default 8 MiB stack, under `alarm()`
  and an 8 GiB `RLIMIT_AS` (not applied under ASan).
- A case passes only if the child exits 0 after catching `halo::Error` with the expected
  code and a message naming the expected bound.
- A crash, hang, `bad_alloc` or silent success fails the test. It does not kill the test
  binary.

**S-3 evidence table:**

| Row | Test | Bound hit |
|---|---|---|
| macro recursion `f(n+1)` | `MacroRecursionHitsRenderDepth` | render depth |
| 100 KB comment | `HundredKilobyteCommentHitsSourceCap` | source size; the lexer itself is covered by `LongCommentLexesLinearlyOnSmallStack` (a 1 MB comment on an 8 MiB stack, with the cap and the parse thread off) |
| 5000 parens | `FiveThousandParensHitParseDepth` | parse depth |
| 3000 parens (was 3.9 s) | `ThreeThousandParensHitParseDepthQuickly` | parse depth |
| nested `range(99999)` loops | `NestedRangeLoopsHitIterationCap` | loop iterations |
| 100 GB output | `HundredGigabyteOutputHitsOutputCap` | output bytes |
| `'x' * 100000000` | `HundredMegabyteStringHitsValueCap` | value size (before allocating) |

**Beyond the table:**
- `ExponentialMacroHitsStepCap`
- `DeepErrorOnLongLineStaysLinear`
- `ParseDepthCoversEveryRecursivePath` (seven recursive parser paths)
- `TokenScanIsNotQuadratic`
- `WhitespaceControlIsLinear`
- `LongTokensParseOnTheDedicatedStack`
- `HostileRenderPatternsRaiseTypedErrors` (16 patterns: doubling, amplifiers, cycles, deep
  values, quadratic search, empty separator)
- `MillionLevelValueIsTornDownIteratively`
- `FormerlyFatalPatternsNowComplete` (deep values dropped, macros and call blocks in loops;
  leak-checked under LSan in the ASan build)
- `CommentLexingMatchesUpstreamRegex` and `ReplaceAndSplitKeepUpstreamSemantics`
  (upstream-semantics equivalence)
- `RenderAtDepthCapFitsTheDocumentedStack`
- `DefaultsLeaveHeadroomOverRealTemplates`

**Golden.** The real Qwen templates still match HF byte for byte (`ChatTemplateGolden.*`,
70 cases, unchanged).

**Demonstrated red.** Each check was shown red on unhardened code, one change at a time.

- **Limits disabled one at a time.** `HALO_TEMPLATE_LIMITS_DISABLE=<field>` zeroes one
  `TemplateLimits` field.

| Row | Red outcome with its bound off |
|---|---|
| macro recursion | SIGSEGV |
| 5000 parens (parse thread off too) | SIGSEGV |
| nested loops, 100 GB output, exponential macro | SIGALRM (hang) |
| long tokens with no parse thread | SIGSEGV |
| 100 MB string | caught later by the output cap (wrong bound) |
| 3000 parens, 100 KB comment, parser paths | no typed error |

- **Patch hunks reverted one at a time.** Each mutant was rebuilt into a separate build
  dir (scratch tool `l_mutate.py`).

| Hunk reverted | Red outcome |
|---|---|
| comment regex | SIGSEGV |
| no `match_continuous` | SIGALRM |
| whitespace `regex_replace` | SIGSEGV |
| decoration cap | over the time budget |
| recursive `~Value` | SIGSEGV (1M-level test) |
| cycle refusal | wrong or no error |
| replace one-pass | SIGALRM |
| empty separator | wrong bound |
| `'' * huge` | SIGALRM |
| substring charge | SIGALRM |
| dump indent check | SIGALRM |
| join growth check | `bad_alloc` |
| indent growth check | `bad_alloc` |
| `==` depth guard | no error |
| dump depth guard | SIGSEGV |
| container-scan steps | no error |
| split parts check | wrong bound |
| result check | wrong bound |
| context tracking | LeakSanitizer leak (ASan build) |

- **Not independently red.** The charge on container stores (`halo_before_store`'s
  `budget_alloc`) never goes red on its own. The expression-result charge catches every
  case found, so the store charge is kept only as defence in depth.

### S-8

- `render()` rejects `messages`, `tools`, `bos_token`, `eos_token` and
  `add_generation_prompt` in `extra_context` with `Error(Api)` ("... is reserved ...").
- Test: `ExtraContextCannotOverrideReservedVariables`.
- The golden cases pass only `enable_thinking`, `preserve_thinking`, `reasoning_effort` and
  `add_vision_id`, so no golden case is affected.

### S-10

- `range_length` is now `ceil((end - start) / step)` in exact unsigned arithmetic, so
  `range(100000)` is allowed and `range(100001)` is rejected.
- Test: `RangeCapIsExactlyOneHundredThousand`, which covers negative steps, strides and the
  full int64 span.
- The macro-context leak is fixed by patch item 8.

### Not done / residual risk

- **No template SHA-256 allowlist** (review A-8, optional). The limits bound cost for any
  template instead.
- **Render runs on the caller's thread.** A caller with less than `kMaxRenderStackBytes`
  of stack is not protected. The API must render on its own bounded worker (A-5).
- **Step cost varies with the build.** A worst-case render costs about 4 s of CPU in a
  release build on the dev host, and about 80 s under ASan+Debug. Lower `max_steps` if
  that is too slow.
- **Upstream builtins not individually charged.** `items`, `dictsort`, `unique`, `map`,
  `select*` and similar builtins are bounded only through the size of their inputs (every
  container built is size-checked on return) and the steps of the expressions that call
  them.
- **Untrusted request data is not depth-checked here.** The S-2 render-side depth check on
  `messages`/`tools` (an iterative walk, depth 64) is planned with S-9 in M3.

## S-5 (out of scope)

- **Behaviour.** SIGBUS occurs when a mapped GGUF is truncated in place while it is open.
  `MAP_PRIVATE` does not protect against this.
- **Mitigation.** Belongs to the download path (review A-7): write to a temp file and rename
  atomically, so an open inode is never truncated.
- **Operator note.** Do not overwrite a model file in place while HALO has it open.
- **Not implemented.** HALO takes no lease or lock. A SIGBUS handler around tensor reads was
  judged not trivial.
