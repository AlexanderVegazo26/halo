# Fuzz targets

libFuzzer entry points (`LLVMFuzzerTestOneInput`) for the code that parses untrusted input:

| target | input | invariant |
|---|---|---|
| `fuzz_gguf` | a GGUF file (both parse modes, then model validation + tokenizer) | loads, or throws `halo::Error` |
| `fuzz_api_request` | an HTTP request body: parse → OpenAI/Anthropic validation → chat template → tokenize (synthetic `FakeEngine`) | a non-empty prompt, `api::RequestError`, or `halo::Error` with code `Api` |
| `fuzz_output_parser` | byte 0 = options (bit 0 `starts_in_reasoning`, bit 1 typed tools), byte 1 = chunking seed, rest = model output | never throws; feeding in chunks gives exactly the one-shot result; every ToolCall event indexes a tool call |
| `fuzz_chat_template` | byte 0 = options (bit 0 generation prompt, bit 1 tools, bit 2 `enable_thinking=false`, bit 3 parsed tool arguments, bit 4 a spoofed `<\|im_start\|>` in the last user message), rest = Jinja template source, rendered over one fixed conversation under tightened `TemplateLimits` | parses or throws `halo::Error(Config)`; renders or throws `halo::Error(Api)`; a render stays within its limits, is deterministic, and its message offsets point at turn headers in order |

A broken invariant aborts with `FUZZ INVARIANT BROKEN (file:line): …`; in sanitizer builds any
memory error, leak or undefined behaviour fails too.

## In every build: seed replay

The targets also link `replay_main.cpp`, so a normal build (plain, ASan, TSan) has
`build/tests/fuzz/fuzz_<target>` that runs each input once. ctest `fuzz_replay_<target>` (label
`fuzz`) replays `corpus/<target>/`, so every reproducer added there stays a regression test in
all CI lanes. Replay one input: `build/tests/fuzz/fuzz_gguf path/to/crash-…`.

## Coverage-guided: `HALO_BUILD_FUZZERS=ON` (clang)

```sh
cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DHALO_BUILD_FUZZERS=ON -DHALO_ENABLE_ASAN=ON -DHALO_BUILD_VULKAN=OFF -DHALO_BUILD_HIP=OFF
cmake --build build-fuzz --target fuzz_gguf fuzz_api_request fuzz_output_parser fuzz_chat_template
scripts/test-agent --build-dir=build-fuzz --no-build -- -L fuzz      # what CI's fuzz job runs
```

The whole tree gets `-fsanitize=fuzzer-no-link` and the targets link libFuzzer. ctest
`fuzz_smoke_<target>` runs `run.py --smoke`: seed 1 and a fixed run count (`SMOKE_RUNS` in
`run.py`), so a run is reproducible; about 2.5 minutes for all four under ASan. The checked-in
corpus is never written: new inputs and `crash-*`/`timeout-*`/`oom-*` reproducers go to
`$HALO_FUZZ_REPORT_DIR/<target>/` (`.test-report/fuzz/` under test-agent), and `run.py` prints
the replay command for each finding.

Longer, opt-in campaigns (not in the blocking suite):

```sh
HALO_FUZZ_REPORT_DIR=/tmp/fuzz python3 tests/fuzz/run.py --time 600 --jobs 4 \
  build-fuzz/tests/fuzz/fuzz_gguf tests/fuzz/corpus/gguf
```

## Fixing a finding

1. Replay the reproducer in a normal or ASan build and fix the bug, with a unit test that names
   the behaviour (the fuzz target is not the only guard).
2. Copy the reproducer into `corpus/<target>/` with a descriptive name, so `fuzz_replay_<target>`
   keeps it fixed in every lane. Keep seeds small (minimize with `fuzz_<target> -minimize_crash=1
   -runs=10000 <crash>` in the fuzz build).

The GGUF seeds are binary: `fuzz_make_seeds tests/fuzz/corpus/gguf` (normal build) rewrites them
from `tests/unit/model/gguf_builder.h`.
