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
| S-6 quadratic tokenizer | M2 | mitigated |
| S-7 tokenizer.json allocation | M2 | mitigated |
| S-4 GGUF metadata amplification | M3 | mitigated |
| S-9 output-parser nesting | M3 | mitigated |
| S-2 render-side depth check (review remediation 2, defence in depth) | M3 | done |
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
- **Untrusted request data.** The S-2 render-side depth check was added in M3; see below.

## Tokenizer (M2)

### S-6: quadratic work on a hostile vocabulary

**Mitigations:**
- **Duplicate detection.** Added tokens are inserted into one byte trie, and an insertion
  that ends on an occupied node is a duplicate. This replaces the pairwise O(n²) check. The
  error message is unchanged: "added token 'X' appears twice (ids A and B)".
- **Matching.** Added-token matching walks the same trie at each input position and keeps
  the longest token allowed by `parse_special`. This gives exactly the old rule: leftmost
  position, then longest allowed token, with Control tokens skipped when
  `parse_special=false`. It is also HF's LeftmostLongest rule.
  - A 256-entry first-byte table skips positions where no added token can start.
  - Trie edges live in the same open-addressing table used for merges.
- **Caps.** `TokenizerLimits` is a new field `VocabSpec::limits`, plus new
  `from_hf_json` and `from_hf_dir` overloads.
  - `max_added_tokens = 4096`: real Qwen3.8 vocabularies have 33.
  - `max_added_token_bytes = 256`: the real maximum is 21.
  - A violation throws `Error(Model)` at build. 0 disables a cap.

**Cost bound:**
- Build is linear in the total added-token bytes.
- Encode costs at most `max_added_token_bytes` trie steps per input byte.
- The worst case inside the default caps is a vocabulary of 4096 `'<'`-prefix tokens that
  never complete. It encodes 100 KB of `'<'` in 0.24 s on the dev host (D-001), about
  2.4 µs per byte.
- *Believed, not verified:* HF's `aho-corasick` LeftmostLongest search restarts after
  each match and has the same kind of O(text × longest token) worst case.

**Tests** (in `tests/unit/tokenizer/test_tokenizer_limits.cpp`):

| Test | What it checks | Result |
|---|---|---|
| `EightyThousandAddedTokensBuildAndEncodeQuickly` | the review's 80,000 Control tokens | build 0.18–0.26 s (was 8.8 s on the unhardened code); encode 20 KB of `'<'` in 0.0015 s (was 6.6 s); thresholds < 1 s and < 0.1 s, release only |
| `SharedPrefixVocabularyEncodesInBoundedTime` | the worst case inside the caps | 0.24 s (unhardened: 3.4 s); threshold < 1 s |
| `AddedTokenCountAndLengthAreCapped` | 4096 and 256 accepted, 4097 and 257 rejected, raised limits honoured | — |
| `DuplicateAddedTokensStillRejected` | same message; prefixes are not duplicates | — |
| `TrieMatchingEqualsLeftmostLongestReference` | 60 random overlapping vocabularies × 40 texts × both `parse_special` values, against a naive leftmost-longest reference; over 1000 added-token matches | — |

**Unchanged on real data:** `TokenizerGolden.*`, the HF corpus, fuzz, codepoint and NFC
sweeps, and the GGUF-vs-JSON vocabulary test are all still token-for-token identical.

### S-7: tiny tokenizer.json, large allocation

**Mitigation:**
- `from_hf_json` sizes its vectors by the largest id. That id must now be below the number
  of entries (vocab + added_tokens), and this is checked before any id-sized allocation.
  `Error(Model)` says "…not below the entry count … (ids must be dense)".
- Real files are dense: 248,077 entries, largest id 248,076.
- An added token that repeats a vocab entry's id is still accepted.
- **Behaviour change:** a tokenizer.json whose ids have holes is now rejected.

**Tests:**
- `TinyFileNamingAHugeIdIsRejectedWithoutAllocating`:
  - Input: a complete, supported 728-byte tokenizer.json with vocab `{"a": 16777215}`.
  - It runs in a forked child that resets peak RSS (`/proc/self/clear_refs`) and passes
    only if the file is rejected with `Error(Model)` while VmHWM grows by less than 64 MiB.
  - Unhardened code: VmHWM grew by 1,208,072 KiB (≈1.15 GiB), exit 3.
- `DenseIdsWithAddedTokenOverlapStillLoad`: the overlap is accepted, a hole is rejected,
  and limits apply on the HF path.

### Demonstrated red (M2)

Each mutant was built into a private copy of the tree and run against the
`TokenizerLimits.*` tests (scratch tools `l_tokmut.py` and `l_tokmut.sh`):

| Mutant | Tests that went red |
|---|---|
| `head`: the unhardened `tokenizer.cpp` from HEAD | both timing tests, the caps test, S-7 (1.15 GiB), dense ids. The reference-equivalence and duplicate tests stay green, which shows the old and new matching agree. |
| `no_s7` | S-7 test (exit 3, 1.15 GiB); dense-id hole assertion |
| `no_count_cap` | caps test; HF-path limit assertion |
| `no_len_cap` | caps test |
| `longest_first_wrong`: first terminal instead of longest | reference-equivalence test |
| `ignore_special_filter` | reference-equivalence test; `TokenizerSpec.AddedTokensSplitFirstLongestWins` |

After the mutants, the restored copy is green and byte-identical to the repo source.

### Not done (M2)

- **Matcher cost.** Matching is not Aho–Corasick-linear. Its cost per input byte is bounded
  by the longest added token (256 by default), as above.
- **GGUF path unchanged.** GGUF-sourced vocabularies arrive as vectors already bounded by
  the GGUF parser, which the M3 metadata budget now caps.

## GGUF metadata and JSON depth (M3)

### S-4: GGUF metadata memory amplification

**Mitigation.**
- `GgufOptions::max_metadata_bytes` sets a total allocation budget for the parsed header.
  The default is `GgufLimits::kMaxMetadataBytes`, 1 GiB.
- Every allocation is charged before it happens, as elements × stored size:
  - 8 bytes per integer or float element;
  - 1 byte per bool;
  - `sizeof(std::string)` per string slot, plus the bytes of any string long enough to
    leave SSO;
  - `sizeof(GgufArray)` (152 B) per nested array;
  - per-KV and per-tensor-info bookkeeping.
- When the budget is exceeded, parsing throws
  `Error(Model): … metadata allocation budget of N bytes exceeded at offset …`.
- The existing per-array caps and "fits in the remaining bytes" checks stay in place.

**API (additive).**
- `GgufFile::open(path, mode, options)`
- `GgufFile::parse(bytes, mode, name, options)`
- `GgufFile::metadata_bytes()`

The existing overloads use the default budget, so `src/models/adapters.cpp` and the other
callers are unchanged.

**Measured.** The real Qwen3.8 headers (about 11 MB each) are charged 18.5–18.8 MB, about
1/57 of the default.

**Tests** (in `tests/unit/model/test_gguf_budget.cpp`):
- **`EmptyNestedArraysAreChargedAtTheirStoredSize`:** a small-scale version of the review's
  20M empty nested arrays.
  - 100K empty arrays take 1.2 MB on disk and are charged about 15 MB.
  - Rejected under a 4 MiB budget; accepted at the default.
- **`NarrowElementsAreChargedAtEightBytes`:**
  - A u8 array of 1M elements is charged 8 MB.
  - 1M empty strings are charged 32 MB.
  - 100K strings of 100 B are charged their heap bytes as well.
- **`DefaultAndDisabledBudget`:**
  - The default is 1 GiB, and 0 disables the budget.
  - Tensor infos are charged too.
- **`BudgetBoundsPeakMemory`:**
  - A forked child parses 2M empty nested arrays (a 24 MB image, about 300 MB if stored)
    under a 64 MiB budget.
  - It passes only if the file is rejected with `Error(Model)` while peak RSS
    (`/proc/self/clear_refs` + VmHWM) grows by less than 96 MiB.
- **`RealHeadersUseAFractionOfTheDefault`:** at least 8x headroom on all three real headers.

### S-9, and S-2 render-side: JSON nesting depth

**Shared limit.** `halo::chat::kMaxJsonDepth = 64` matches the API's
`ServerConfig::max_json_depth` in value and semantics.
- Depth is the number of arrays/objects enclosing an element, the depth nlohmann's parser
  callback reports. `[]` is 0, `[1]` is 1, and 65 nested empty arrays are the deepest
  accepted.
- The semantics match because the check is the same callback condition as the API's
  `parse_strict` (`depth > max_depth`), which I compared by reading the code.
- The api module's own depth tests were built and run against these changes (see the M3
  handback for the counts).

**Mitigations:**
- **`parse_json_bounded(text, max_depth)`:** nlohmann's iterative parser with that
  callback. It returns a discarded value for invalid or over-deep input, and gives up
  before any deep value exists.
  - `OutputParser`'s argument typing (`convert_value`) uses it, so a parameter over the
    cap stays the raw string (S-9).
  - `ChatTemplate`'s `parse_string_tool_arguments` (`with_parsed_arguments`) uses it too,
    so over-deep argument strings stay strings.
- **`json_nesting_depth(v, limit)`:** an iterative walk with the same measure.
  - `render()` rejects `messages`, `tools` or `extra_context` deeper than 64 with
    `Error(Api)` ("… nests deeper than 64 levels") before copying them.
  - The copy and the minja conversion recurse once per level, so this check must come
    first.
  - This is defence in depth: the API already caps request bodies at parse time.

**Tests** (in `tests/unit/template/test_template_limits.cpp`):
- **`JsonDepthCapMatchesTheApi`:**
  - the boundary at 65/66 empty arrays and at 64/65 levels with a scalar;
  - the parser and the walk agree for n = 60–70, with and without inner values;
  - the output parser keeps a 66-deep argument as its raw text.
- **`DeeplyNestedToolArgumentsInModelOutputAreKeptAsText`:** the review's 100,000-level
  `<parameter>`, on an 8 MiB thread in a death-test child.
- **`DeeplyNestedMessagesAreRejectedBeforeCopying`:**
  - a message field and tools at 30,000 levels (the review's crash depth) return
    `Error(Api)`;
  - a 100,000-level argument string with `parse_string_tool_arguments` stays a string;
  - at the cap, render still works.

### Demonstrated red (M3)

Each mutant was built into a private copy of the tree (scratch tools `l_m3mut.py` and
`l_m3mut.sh`):

| Mutant | Red outcome |
|---|---|
| no element charge | nested-array, narrow-element and peak-memory tests; peak RSS +296,708 KiB, accepted |
| no string-heap charge | narrow-element test (long strings) |
| no tensor-info charge | default/disabled test |
| unbounded parse (plain `json::parse`) | depth-cap test; the 100K-level output-parser child exits wrong |
| no render walk | deep-messages child SIGSEGV |
| walk off by one (`>=`) | depth-cap test; the at-the-cap render assertion |

The restored copy is green.

### Not done (M3)

- **Storage width.** u8/i8 arrays are still stored widened to 64 bits. The review listed
  narrow storage as optional, and the budget already charges the real stored size.
- **Model-level API.** `NormalizedModel::open` uses the default budget. There is no
  model-level option to change it yet; `GgufFile::open` takes one.

## S-5 (out of scope)

- **Behaviour.** SIGBUS occurs when a mapped GGUF is truncated in place while it is open.
  `MAP_PRIVATE` does not protect against this.
- **Mitigation.** Belongs to the download path (review A-7): write to a temp file and rename
  atomically, so an open inode is never truncated.
- **Operator note.** Do not overwrite a model file in place while HALO has it open.
- **Not implemented.** HALO takes no lease or lock. A SIGBUS handler around tensor reads was
  judged not trivial.
