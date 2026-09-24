# HALO — Implementation Agent Brief (read fully before writing code)

## Environment

- Host is Windows; **all building and testing happens in WSL** distro `Ubuntu-24.04` as root.
  From the Bash tool run WSL commands through PowerShell, or via:
  `powershell.exe -NoProfile -Command "wsl -d Ubuntu-24.04 -u root -- bash <script>"`.
  Inline `bash -c "..."` quoting through PowerShell is fragile (it expands `$`): **write a
  script file and execute it** instead.
- The repo is visible in WSL at `/mnt/c/...` (translate your worktree's Windows path:
  `C:\Users\...` → `/mnt/c/Users/...`).
- Build: `bash scripts/build.sh` from the repo root inside WSL. Use a **private build dir**:
  `HALO_BUILD_DIR=/root/halo-build-<your-workstream> bash scripts/build.sh`.
  Build dirs must be on the Linux FS (`/root/...`), never on `/mnt/c`.
- Logs: `bash scripts/wsl-run.sh <logname> <cmd...>` prints the tail and the **real exit code**.
  Never judge success by piping to `tail`/`head` (that reports the pager's status).
- Toolchain: clang 18 (C++23, libstdc++ 13: `std::expected`, `std::format`, `std::span` OK;
  `std::print` / `std::mdspan` NOT available), CMake 3.28, Ninja, glslc, Vulkan + lavapipe,
  SQLite 3.45.
- Python (reference/golden generation only, never in the runtime):
  `/root/halo-py/.venv/bin/python` (3.12, torch CPU, transformers 5.17, gguf-py, numpy).
- llama.cpp reference build: `/root/llama.cpp` (commit `bd4f514db`, same as target machine),
  binaries in `/root/llama.cpp/build/bin`.

## Reference data (`/root/halo-ref`, outside the repo)

- `tiny/` — tiny random qwen35 model (see `python/tools/make_tiny_model.py`):
  `tiny-f32.gguf`, `tiny-q8_0.gguf`, `tiny-q4_k_m.gguf`, `tiny-q6_k.gguf`,
  `tiny-iq4_xs.gguf`, `tiny-q3_k_m.gguf`, `tiny-q5_k_m.gguf`, `tiny-q4_0.gguf`,
  `hf/` (HF checkpoint + real tokenizer), `golden/manifest.json` + `*.bin` (raw
  little-endian arrays; manifest gives dtype/shape).
- `*.header.gguf` — **real** Qwen3.8-27B GGUF headers (KV + tensor infos, no tensor data):
  `ggml-org-q4km`, `ggml-org-mtp-q4_0`, `unsloth-ud-q4kxl`; `*.summary.json` alongside.
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` — real HF files.

Tests locate this via the compile definition `HALO_REF_DIR` (default `$HOME/halo-ref`).
If data is missing, a test must `GTEST_SKIP() << "why"` — **never silently pass**.

## Ground truth

`docs/DECISIONS.md` overrides the PRD/TRD where they disagree. Read D-002…D-008 before
touching model code. Do not invent model facts; when unsure, read
`/root/halo-py/.venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
or `/root/llama.cpp/src/models/qwen35.cpp`, `/root/llama.cpp/ggml/src/ggml-quants.c`.

## Code conventions

- Load and follow the C++ skill at `C:\Users\avega\.claude\skills\cpp-pro\SKILL.md` (read its
  references as needed). Python tooling: `C:\Users\avega\.claude\skills\python-pro\SKILL.md`.
- C++23, namespace `halo::<module>`; public headers `include/halo/<module>/*.h`, sources
  `src/<module>/*.cpp`, module CMake at `src/<module>/CMakeLists.txt` using the
  `halo_module(<name> SOURCES ... DEPS ...)` helper (see `src/CMakeLists.txt`). Modules are
  auto-discovered — do **not** edit `src/CMakeLists.txt` unless your module is not in its list.
- Tests: `tests/unit/<module>/CMakeLists.txt` using `halo_add_test(name SOURCES ... LIBS ...)`
  (auto-discovered). GoogleTest.
- Errors: throw `halo::Error` via `halo::throw_error(ErrorCode::X, fmt, ...)` / `HALO_CHECK`
  (`include/halo/core/error.h`). Logging: `HALO_INFO("component", fmt, ...)`.
- `-Werror` with a strict warning set is on. No raw owning `new`/`delete`, no C-style casts.
- **External input (GGUF files, JSON, HTTP) is untrusted**: bounds-check every length,
  count, offset and shape; overflow-check size arithmetic; fail with a typed error, never UB.
- Performance matters but **correctness first**: the CPU path is the reference that every GPU
  kernel is differentially tested against.

## Definition of done for your workstream

1. Builds clean with `-Werror`; `ctest` in your build dir passes; report the exact test
   counts (passed / failed / **skipped**, with skip reasons).
2. Every new check was shown to fail at least once (break the code or the expectation,
   observe red, restore) — mention how in your report.
3. Also build + run your tests once with `bash scripts/build.sh --asan`
   (`HALO_BUILD_DIR=/root/halo-build-<ws>`, the script appends `-asan`); report the result.
4. Commit on your worktree branch with a descriptive message ending in
   `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
   Set identity if needed: `git config user.name AlexanderVegazo26`,
   `git config user.email avegazorodriguez@gmail.com`. Never push.
5. Final report: what you built (files), public API summary, test evidence, anything you
   could not verify and why, and open questions. Keep it factual; no unmeasured claims.
