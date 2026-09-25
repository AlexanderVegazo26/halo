# HALO — resume checkpoint

Resume the MVP build from this file. It is updated at every milestone commit; the
**Last updated** line and **In flight** section say how fresh it is.

**Last updated:** 2026-09-25, at commit `9babebe` (branch `halo-v0.2`).

## 1. What HALO is and what "done" means
- **What it is:** a C++23 inference runtime for Qwen3.8-27B on AMD Strix Halo (GMKtec EVO-X2, Radeon 8060S, gfx1151).
- **Specs:** `docs/HALO_PRD_v1.1.md` and `docs/HALO_TRD_v1.1.md`. `docs/DECISIONS.md` overrides both, and later amendments override earlier text.
- **Target:** the TRD §66 MVP list (§3 below).
- **Success criterion:** the owner's efficiency NFRs (D-014). They can only be measured on the EVO-X2 (D-001), because the dev host has no AMD GPU.
- **Canonical model pack:** unsloth UD-Q4_K_XL (D-014). The ggml-org Q4_K_M pack with a separate MTP file must also load (D-006).

## 2. How the work is run (read before resuming)
- **Roles:** the main session is the orchestrator. It never writes feature code itself. Implementation agents (`sdlc-suite:software-engineer`) work in the **shared working tree**, each owning disjoint paths.
- **Agents never commit.** The orchestrator verifies every milestone with its own build, then commits by explicit path.
  - Never use `git add -A` or `git commit -a`: they sweep other agents' in-progress work into the commit.
- **Build and test only in WSL** (Ubuntu-24.04, as root), through script files:
  `powershell.exe -NoProfile -Command "wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../script.sh"`
  - Use `scripts/wsl-run.sh <log> <cmd>` so you get the real `EXIT=`.
  - Pick modules with `-DHALO_ONLY=...`. Add `-DHALO_BUILD_HIP=ON` for HIP; ROCm 7.1 is in WSL, so HIP code compiles but cannot run.
- **Verify in a private tree** while agents are mid-edit:
  - `git archive HEAD` → `/root/orch-src`, overlay only the paths being committed, `touch` everything, then build in a fresh or matching build dir.
  - Reusing a build dir from another source tree fails at configure.
- **Mutation-test gotcha:** `/mnt/c` mtimes lag the WSL clock by about 0.6 s. After restoring a mutated file, delete its object files or touch it +2 s, and finish with a clean ctest.
- **Scratchpad tooling** must be prefixed per workstream (`g_`, `i_`, `j_`, ...), because agents overwrote each other's `mut.py`.
- **Never launch programs on the Windows host.** An agent ran `ollama list` there once, which started the Ollama app and its updater.
- **Run shape:**
  - An agent stops at every milestone and reports: files, exact ctest counts including skip reasons, ASan result, demonstrate-fail evidence, and a proposed commit message.
  - The orchestrator commits, then continues the same agent with SendMessage.
  - After every phase, a `code-reviewer` pass, plus a `security-engineer` pass whenever there is a new endpoint, subprocess, parser or file write. Reviews go in `docs/reviews/`.
- **Interruptions:** the org's monthly spend limit has cut off every agent several times.
  - To resume, SendMessage each agent: "you were cut off; check for half-applied edits, leftover mutations and stale shells; continue".
  - If agent IDs are gone (new session), relaunch with the workstream's prompt file plus a "state at interruption" block built from `git status` and that agent's `status_<WS>.md`.
- **Session scratchpad (session d2c57405):**
  `C:\Users\avega\AppData\Local\Temp\claude\C--Users-avega-Documents-personal-halo\d2c57405-8141-41ac-9d34-75ce2f047a53\scratchpad\`
  - `agents.md`: map from workstream to agent id.
  - `status_<WS>.md`: each agent's own progress file.
  - `prompt_*.txt` / `resume2_*.txt`: workstream prompts.
  - `wip2-snapshot.tgz`: an older snapshot of work in progress.

## 3. MVP status (TRD §66). Evidence is on the dev host unless stated.
| MVP item | Status | Notes |
|---|---|---|
| C++ runtime, GGUF loader, MTP file | Done | The real 27B pack has **never been loaded**: only `*.header.gguf` files exist in `/root/halo-ref`. |
| Tokenizer + Jinja templates | Done, hardened | Matches HF token for token. Hostile templates bounded by a pinned minja patch (`cmake/patches`). |
| CPU reference backend | Done | Matches the transformers goldens (tiny model). |
| Paged KV + DeltaNet state manager | Done | D-012 slots, D-013 checkpoints. TD-1 (redundant slot) is open until WS-BI-2. |
| MTP speculative decoding + profit gate | Done | Greedy output with MTP on equals output with it off. Real acceptance on the tiny model is 0, so accept paths are tested with oracle drafts. |
| Sampling + fused greedy argmax | Done | Includes JSON-schema structured output. |
| OpenAI + Anthropic APIs, /tokenize, /apply-template | Done | Security fixes in flight (WS-I M5). v0.2 serving is loopback-only (D-017). |
| Prefix cache | Done | `checkpoint_hints` has no producer yet. |
| Benchmark harness + baseline adapters | Done | llama-bench and llama-server have run for real on CPU. Vulkan/HIP llama.cpp and Ollama are tested against parsers only. |
| Persistent autotuned profiles | Partial | SQLite DB, `halo tune`, and runtime lookup work for **CPU** tunables only. GPU variants are not bridged (WS-BI-4). |
| HIP backend (GDN, GEMV, LM head) | Code done, **never run on a GPU** | Every qwen35 decode op exists and matches `halo::cpu` in host emulation. Not wired to the model forward. |
| Vulkan/RADV backend (full op set incl. GDN) | Partial | 14 ops are missing (see `docs/vulkan.md`). Verified on lavapipe only; nothing has run on RADV. |
| Model forward on a GPU | Not started | Needs ADR-001, WS-BI-1..BI-6. |

The `halo` CLI provides `inspect`, `devices`, `tokenize`, `template`, `run`, `serve`, `bench` and `tune`. `run`, `serve` and `bench` work end to end on the real CPU engine (tiny model).

Last full-tree build (`8065734`): 545 tests, 543 pass, 2 skipped. The skips are the opt-in bandwidth test and a labelled F-1 skip.

## 4. What remains for the MVP, in order
1. **In flight (§5):** the WS-G M5 engine fixes, the WS-I M5 API fixes, and WS-BI-1.
2. **ADR-001 workstreams** (`docs/adr/ADR-001-backend-interface.md` §9), in the order BI-1 → BI-2 → {BI-3, BI-4, BI-5} → BI-6:
   - **BI-2:** the state ring on the backend, closing TD-1 and TD-2. The owner allowed the additive `cpu/ops.h` ring overloads (D-017).
   - **BI-3:** HIP adapter; forward over HIP emulation must equal the CPU forward.
   - **BI-4:** autotune bridge for GPU variants.
   - **BI-5:** Vulkan fills its 14 missing ops (V1–V4), then runs the forward on lavapipe.
   - **BI-6:** wire the engine to the backends.
3. **A real 27B smoke run on the CPU.** Download UD-Q4_K_XL (about 17.5 GB) into WSL ext4 (not `/mnt/c`), then `halo inspect` and a short `halo run` with `--ctx 4096 --parallel 1`. It will be slow; it proves correctness at scale, not speed.
4. **A phase review** (code-reviewer + security-engineer) of everything after `3c4c71c`.
5. **BI-7, on the EVO-X2 (hardware, owner):**
   - device tests (they skip on the dev host today);
   - the ROCm 7.1 → HIP 7.15 compatibility check;
   - RADV runs;
   - efficiency NFR measurements;
   - ADR-001 open questions Q3 and Q5.

## 5. In flight at this checkpoint
Agent ids are in `scratchpad/agents.md`.

| Workstream | Doing | Owns |
|---|---|---|
| WS-G M5 | Code-review engine fixes R-1 (per-sequence failure, D-017), R-2 (the worker can never terminate the process), R-3 (engine cancel path before the first token), R-5(a) failure-injection tests, N-1, N-2, N-4 | src/runtime, src/speculative, src/state, src/kv_cache and their tests, tests/unit/integration |
| WS-I M5 | F-1 (structured output with thinking on); `halo tune` key contract via `runtime::engine_profile_key`; one SHA-256 (TD-12/13); bench F-2/F-3; security S-13 (header deadlines), S-16/A-5, S-17..S-22; then wire the API to G's cancel path | src/api, tools/halo, tests/unit/{api,cli}, docs/{api,cli}.md |
| WS-BI-1 | Backend interface + CPU adapter; the qwen35 forward over `Backend&`, bit-identical to today | src/models, include/halo/models, tests/unit/models, the new backend module |

Finished workstreams: WS-A..F (phase 1), F2 (Vulkan review fixes), H (sampling), J (bench + autotune + security fixes), K (HIP ops), L (parser hardening).

## 6. Decisions and open questions
- **Owner decisions:**
  - D-014: success = efficiency NFRs; canonical pack UD-Q4_K_XL.
  - D-016: one GDN q/k contract; ARGMAX raises on NaN. Amended with its defaults.
  - D-017: ADR-001 accepted; additive `cpu/ops.h` ring overloads allowed; a data error fails only its own sequence; the atomic state ring is the default; v0.2 serving is loopback-only.
- **Open ADR-001 questions (Q3, Q5, Q6, Q7).** Until the owner answers, the ADR's proposals apply:
  - copy weights to the GPU at load;
  - no graph replay yet;
  - no Vulkan CPU fallback;
  - a separate `hip-emulation` profile label.
- **Technical debt:** `docs/dev/TECH_DEBT.md` (TD-1..TD-13).
- **Reviews:** `docs/reviews/`. The latest are `2026-09-25-code-review.md`, `2026-09-25-security-review-api.md` and `2026-09-25-security-review-bench.md`.

## 7. Resume checklist
1. `git log --oneline -15` and `git status --porcelain`. Uncommitted paths belong to the workstreams in §5.
2. Read each in-flight workstream's `status_<WS>.md` in the scratchpad.
3. Snapshot the uncommitted work before relaunching anything:
   `git ls-files --others --exclude-standard` plus `git diff`, packed into a tgz.
4. Continue the agents (SendMessage), or relaunch them from their prompt files with a "state at interruption" block.
5. Verify each milestone yourself in a private `git archive` tree before committing it, then update this file.
