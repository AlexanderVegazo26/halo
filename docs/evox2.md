# Running HALO on the EVO-X2 (field kit)

This page is the run book for the GMKtec EVO-X2 (Ryzen AI Max+ 395, Radeon 8060S = gfx1151,
128 GiB with a ~96 GiB GPU carveout, Ubuntu 26.04.1, kernel 7.0, HIP 7.15.26333 / ROCm
runtime 1.21; see `docs/strix-halo-qwen38-engine-report.md`). The development host has no AMD
GPU (D-001), so **the first run on the EVO-X2 is also the first time any HALO GPU code runs on
a GPU.** The kit is built to make that run produce evidence that can be brought back and
triaged, whatever happens.

Contents: what the kit is · prerequisites · ROCm versions · copying · the commands in order ·
what to expect today · bringing the bundle back · troubleshooting · how the kit is built.

## 1. What the kit is

`scripts/package.sh` (run in WSL on the dev host) produces two files in `dist/`:

| File | Contents |
|---|---|
| `halo-evox2-<sha12>[-dirty].tar.gz` (+ `.sha256`, `.BUILDINFO`) | `bin/halo`; every gtest binary at its build-tree path under `build/`; the test fixtures under `src/`; `scripts/evox2/` (this tooling); `docs/evox2.md`; `source/halo-src-<sha12>.tar.gz` (the exact source that was built, for the native-build fallback); `kit.json` (machine-readable manifest: every test binary, its working directory, whether it needs reference data, the paths it embeds); `BUILDINFO`; `LIBS.txt`; `SHA256SUMS` |
| `halo-ref-<hash12>.tar.gz` (+ `.sha256`) | the reference data the tests read (`/root/halo-ref` on the dev host: tiny models, goldens, real 27B GGUF headers, tokenizer files), with `REF_SHA256SUMS`. About 4.4 GB unpacked. Without it the reference-dependent tests SKIP; nothing fails. |

`-dirty` in the name means the kit was built from HEAD plus uncommitted files (listed in
`BUILDINFO` as `dirty_paths`). A kit built by the orchestrator from a clean HEAD has no suffix.

**CPU target.** The kit is compiled with `-march=x86-64-v3` (AVX2, FMA, BMI2), not
`-march=native`. The dev host (Ryzen 7 4800H, Zen 2) supports v3 but not v4, and the kit's tests
must run there before it ships. Zen 5 runs v3 code. HALO has no hand-written AVX-512 code (no
intrinsics); the only thing v3 gives up on Zen 5 is compiler auto-vectorization with AVX-512.
`scripts/evox2/build-native.sh` builds with `-march=native` on the EVO-X2 itself if that matters.

**Shared libraries.** Nothing from the system or ROCm is bundled. `LIBS.txt` lists what each
binary needs (from `readelf -d` and `ldd` on the build host):

- every binary: glibc, libstdc++, libgcc_s, libm;
- `bin/halo` and the tests that use the profile database: `libsqlite3.so.0`;
- `test_vulkan` only: `libvulkan.so.1` (the Vulkan loader, which then loads RADV/AMDVLK);
- `test_hip_*` only: `libamdhip64.so.7` from ROCm (which pulls in `libhsa-runtime64.so.1`).
  These binaries carry `RUNPATH /opt/rocm/lib:/opt/rocm-7.1.0/lib` (set by CMake's HIP
  package), so the target's `/opt/rocm/lib` is searched first; the build host's versioned
  directory does not exist on the target and is skipped. The kit's scripts also put
  `${ROCM_PATH:-/opt/rocm}/lib` first in `LD_LIBRARY_PATH` for the `test_hip_*` binaries only
  (recorded as `env_overrides` in the run summary), which
  covers a ROCm installed somewhere else. If `verify-env.sh` reports `libamdhip64 ... not
  found`, see Troubleshooting.

`bin/halo` links neither ROCm nor Vulkan in this build (see `LIBS.txt`), so `halo version`,
`halo devices`, `halo inspect` and the CPU engine run even if ROCm does not load.

## 2. Prerequisites on the EVO-X2

Runtime (the prebuilt kit):

```
sudo apt install python3 time mesa-vulkan-drivers vulkan-tools libvulkan1 libsqlite3-0 \
                 numactl util-linux pciutils clinfo
sudo usermod -aG render,video "$USER"     # then log out and in: /dev/kfd and /dev/dri/renderD*
```

- ROCm (HIP runtime, `rocminfo`, `rocm-smi`/`amd-smi`) is already installed on the EVO-X2
  (HIP 7.15). The kit does not install or change it.
- `python3` (>= 3.10, standard library only) runs the kit tooling.
- `time` provides `/usr/bin/time -v` for the peak-RSS figures; without it they are missing.
- AMDVLK is optional. If it is installed, `collect.sh` also runs the Vulkan tests on it.
- The package names are the Ubuntu 24.04 names. They have not been checked on 26.04; if one
  is renamed, `verify-env.sh` says which tool or library is missing.

Native build (only for the fallback in Troubleshooting), in addition:

```
sudo apt install cmake ninja-build clang glslc libvulkan-dev libsqlite3-dev pkg-config \
                 binutils spirv-tools git
```

plus the ROCm HIP SDK for your ROCm (it must provide `/opt/rocm/llvm/bin/clang++` and
`/opt/rocm/lib/cmake/hip/hip-config.cmake`). The native build downloads nlohmann/json,
minja, cpp-httplib and googletest (pinned by SHA-256 in `cmake/HaloDeps.cmake`), so it needs
network access. CMake must be >= 3.25.

## 3. ROCm versions: say it plainly

- The kit is **built against ROCm 7.1 (HIP 7.1.25424) in WSL**. The EVO-X2 runs **HIP 7.15 /
  ROCm runtime 1.21**. `BUILDINFO` records the exact build versions (`rocm_version_built_with`,
  `hip_version_built_with`, `hip_compiler_version`).
- Whether the `test_hip_*` binaries (a gfx1151 code object plus a `libamdhip64.so.7`
  dependency) load and run under 7.15 is **unverified** until the first EVO-X2 run.
- If they do not load (library not found, `hipErrorNoBinaryForGpu`, invalid device function,
  a crash in the HIP runtime), use `scripts/evox2/build-native.sh`; it builds the same source
  against the EVO-X2's own ROCm.
- `HSA_OVERRIDE_GFX_VERSION` must **not** be set. HALO's device code is built for gfx1151
  natively; `verify-env.sh` and `collect.sh` flag the variable if it is set.

## 4. Copying the kit

On the dev host (PowerShell or WSL), the files are in `dist/` of the repository. Copy both
tarballs and their `.sha256` files to the EVO-X2, e.g. with `scp`:

```
scp dist/halo-evox2-<sha12>.tar.gz* dist/halo-ref-<hash12>.tar.gz* evox2:~/halo/
```

On the EVO-X2:

```
cd ~/halo
sha256sum -c halo-evox2-*.tar.gz.sha256 halo-ref-*.tar.gz.sha256
tar -xzf halo-evox2-<sha12>.tar.gz          # -> ~/halo/halo-evox2-<sha12>/
tar -xzf halo-ref-<hash12>.tar.gz           # -> ~/halo/halo-ref/  (found automatically)
cd halo-evox2-<sha12>
```

The kit finds the reference data at `--ref DIR`, `$HALO_KIT_REF`, `<kit>/ref`,
`<kit>/halo-ref` or `<kit>/../halo-ref`, in that order. Unpacking both tarballs in the same
directory, as above, needs no option.

**Models** go on an ext4 filesystem (not NTFS/exFAT, not a network share; HALO memory-maps
the file), for example `~/models/`:

- `Qwen3.8-27B-UD-Q4_K_XL.gguf` (unsloth, 17,559,178,144 bytes): the file every llama.cpp
  baseline in the engine report used. It contains the native MTP layer (`blk.64`).
- Optional: a separate MTP GGUF (`--mtp FILE`), e.g. the ggml-org MTP pack. Not needed for
  the UD-Q4_K_XL file.

## 5. The commands, in order

Run everything from the unpacked kit directory as your normal user (not root). Each command
prints where its output went.

### Step 1: verify the environment

```
scripts/evox2/verify-env.sh             # add --check-ref to also hash every reference file
```

One line per check (OK / WARN / FAIL / INFO / N/A): kit integrity (SHA256SUMS), `bin/halo
version`, shared libraries of every binary (`ldd`), `HSA_OVERRIDE_GFX_VERSION`, gfx1151 in the
KFD topology, `/dev/kfd` and `/dev/dri/renderD*` access, the ROCm version vs the build's,
the Vulkan ICDs, the VRAM/GTT sizes, the reference data, and the test path anchor. Exit 0 means
no FAIL. Fix every FAIL before step 2 (each line says how).

### Step 2: run the tests

```
scripts/evox2/run-tests.sh                        # all binaries (5-10 min on the dev host)
scripts/evox2/run-tests.sh --full-bandwidth       # + the full-size TRD §50 bandwidth benchmark
scripts/evox2/run-tests.sh --category hip         # only test_hip_*
scripts/evox2/run-tests.sh --vk-icd radv --category vulkan
```

Writes `halo-tests-<utc>/summary.md` (verdict, failures, per-binary table, every skip with its
reason), `summary.json`, `xml/<binary>.xml` (gtest XML) and `logs/<binary>.log`. A binary that
crashes or hangs is re-run one test at a time, so one GPU fault does not hide the rest (the binary still counts as failed, even if every test passes alone); on a GPU
host the new kernel-log lines of every HIP/Vulkan binary are saved as `logs/<binary>.dmesg-delta.txt`
(needs `dmesg` access: run `sudo -v` first). On a gfx1151 host, a device test that SKIPs for
"no device" counts as a failure (`--expect-gpu no` turns that off). Exit 0 = everything as
expected, 1 = look at `summary.md`, 2 = kit problem.

Step 3 runs all of this again; step 2 is the quick look.

### Step 3: collect the diagnostic bundle

```
sudo -v      # optional: lets collect.sh read the kernel log (dmesg) without prompting
scripts/evox2/collect.sh --model ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --llama-dir ~/llama.cpp/build/bin --power-mode performance --host-label evo-x2
```

- Without `--model` it collects everything except the model steps.
- `--power-mode` is the BIOS/EC performance mode you set (it is not visible from Linux, so
  write down what the machine is set to: e.g. `quiet`, `balanced`, `performance`). Always pass
  it: the bench artifacts record it, and `halo tune --list` refuses the default `unknown`.
- `--llama-dir` is a llama.cpp build (the directory with `llama-bench`) for the same-file
  baseline; it is also passed to the baseline integration tests.
- `--quick` skips the full bandwidth benchmark and the per-driver Vulkan passes.
- **Redaction (always on).** Every file in the bundle, of any size and text or binary, is scrubbed
  line by line before it is packed:
  - the values of secret-looking variables (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSW*`, `*AUTH*`,
    `*CREDENTIAL*`, `*COOKIE*`, `*DSN*`, `*PRIVATE*`, `*_PAT`, `SSH_CONNECTION`, `SSH_CLIENT`)
    are replaced by `<redacted>`;
  - `user:password@` inside any URL (for example `http_proxy`) becomes `<redacted>@`.
- **`--anonymize`** also replaces:
  - the home directory, as `/home/<user>`;
  - the host name and FQDN, as `<host>`;
  - the user name, as `<user>`, including the invoking user under `sudo` (`SUDO_USER` and its
    home);
  - the SSH client and server addresses, as `<ip>`.

  It also drops `SSH_*`, `DISPLAY`, `XDG_SESSION_*`, `MAIL` and `SUDO_*` from the environment
  dump.
- **How names are matched.** Names are replaced only as whole tokens, so a user called `amd` does
  not turn `amdgpu` into `<user>gpu`. A name shorter than 3 characters is replaced only where
  names appear: after `=`, `(`, a quote, `/` or `@` (`USER=ab`, `uid=1000(ab)`, `/home/ab`), and
  before `@`.
- **Files that cannot be scrubbed.** A file with a line over 16 MiB, or a symlink, is left out of
  the `.tar.gz`, listed in `SUMMARY.md` section 6 and `excluded-from-tarball.txt`, and kept only
  in the bundle directory.
- The tarball is created with `O_EXCL` (it never overwrites or follows an existing path).
- `--help` lists the rest (`--mtp`, `--prompt`, `--max-tokens`, `--ctx`, `--bench-system`,
  `--skip`, `--timeout-scale`, ...).

It writes `halo-diag-<host>-<utc>/` and `halo-diag-<host>-<utc>.tar.gz` in the current
directory. Each step runs with a timeout, never stops the collection, and records its exact
command, exit code, stdout, stderr, duration, peak RSS (`/usr/bin/time -v`) and a
thermal/power snapshot before and after. The steps:

| Directory | What |
|---|---|
| `system/` | `uname -a`, `/etc/os-release`, `/proc/cmdline`, `lscpu`, `free`, `/proc/meminfo`, `numactl -H`, `lsmem`, `id`, the environment (secret-looking values redacted) |
| `gpu/` | kernel log filtered to amdgpu/kfd/drm/iommu (plus the lines with error/fault/reset words), `/opt/rocm/.info/version`, `hipconfig --full`, `rocminfo`, `rocm-smi --showall`, `amd-smi static/metric` if present, `lspci`, `/dev/kfd` and `/dev/dri`, `vulkaninfo --summary` (default, RADV only, AMDVLK only) and the full `vulkaninfo`, `clinfo`, the amdgpu/hwmon/KFD/power sysfs (`pp_dpm_*`, `power_dpm_force_performance_level`, `gpu_busy_percent`, `mem_info_*`, temperatures, power), GPU memory use before and after (`rocm-smi --showmeminfo`) |
| `halo/` | `verify-env`, `halo version`, `halo devices` (text, `--json`, `--verify --json`), `ldd` of every binary, `halo bench micro` |
| `tests/` | `main/`: every test binary with gtest XML, HIP and Vulkan device tests enabled, `HALO_BANDWIDTH_FULL=1`; `vulkan-radv/` and `vulkan-amdvlk/`: the Vulkan tests with the loader restricted to one driver (`VK_DRIVER_FILES` / `VK_ICD_FILENAMES`) |
| `model/` (with `--model`) | `halo inspect` (text and `--json`), a short greedy `halo run` at trace log level, probes of `--backend hip` and `--backend vulkan`, a smoke-sized `halo bench model`, `halo tune --list`; HALO's own logs as JSON lines (`*.log.jsonl`) |
| `baselines/` (with `--model --llama-dir`) | `llama-bench -ngl 999 -fa 1 -p 512 -n 128 -r 3` on the same file |

HALO's logs are captured through three environment variables that `src/core/log.cpp` reads
(documented in `include/halo/core/log.h`); they work for any `halo` command:

| Variable | Values |
|---|---|
| `HALO_LOG_LEVEL` | `trace`, `debug`, `info` (default), `warn`, `error`, `fatal`, `off` |
| `HALO_LOG_FORMAT` | `text` (default) or `json`: one `{"ts_ms", "level", "component", "msg"}` object per line; `ts_ms` is Unix epoch milliseconds |
| `HALO_LOG_FILE` | append to this file instead of stderr. A new file is created with mode 0600; an existing file must be a regular file owned by the running user with one hard link (a symlink, device, FIFO, foreign-owned or hard-linked file is refused, and logging stays on stderr) |

The bundle has `SUMMARY.md` (section 1: every failure with its log path; section 2: the
expected-vs-actual checklist; then what was not available, and every step) and `manifest.json`
(every step's record, the checklist, the kit's build facts). Exit 0 = the verdict is OK,
1 = something needs attention (SUMMARY.md section 1).

### Step 4: run a model

```
bin/halo inspect ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
bin/halo run ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -p "Hello" -n 64 --ctx 4096 --parallel 1
HALO_LOG_LEVEL=trace HALO_LOG_FORMAT=json HALO_LOG_FILE=run.log.jsonl \
  bin/halo run ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -p "Hello" -n 64 --temperature 0 --ctx 4096 --parallel 1
```

Today this runs the CPU reference engine (see section 6). Its speed on the 27B model has not
been measured yet (it is a reference implementation, not a fast path); the weights file is
17.6 GB and is memory-mapped. The last stderr line is the summary (tokens, time, decode tok/s,
finish reason).

### Step 5: serve on loopback only (D-017)

```
bin/halo serve --model ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 8080 \
    --ctx 8192 --parallel 1
curl -s http://127.0.0.1:8080/v1/models
```

v0.2 serving is loopback-only (D-017). A non-loopback `--host` without an API key is refused
unless `--allow-unauthenticated-remote` is given; do not do that on a network. To reach the
server from another machine, use an SSH tunnel (`ssh -L 8080:127.0.0.1:8080 evox2`), or a
reverse proxy that enforces connection and header timeouts (`docs/api.md`).

### Step 6: bring the bundle back

Copy `halo-diag-<host>-<utc>.tar.gz` (and, if you ran it separately, the `halo-tests-<utc>/`
directory of step 2) to the dev host, unpack it anywhere, and tell the coding agent:

> Here is the EVO-X2 diagnostic bundle `<path>/halo-diag-.../`, produced by kit
> `<kit name from SUMMARY.md>` with `collect.sh <the options you used>`. The BIOS power mode
> was `<mode>`. Read `SUMMARY.md` first (section 1, then the checklist), then `manifest.json`.
> Triage every failure and MISMATCH against docs/evox2.md section 6, say which are HALO bugs,
> which are environment problems, and propose the fixes.

Also mention anything the bundle cannot show: the fan noise, a hang you had to interrupt, what
you changed on the machine between runs.

## 6. What "expected" looks like today (MVP stage)

Be clear about what the current code does on the EVO-X2:

- **The model forward does not run on the GPU yet.** ADR-001 (BI-1..BI-6) is in progress;
  until BI-6 wires the qwen35 forward to a GPU backend, `halo run`, `halo serve` and `halo bench
  model` use the **CPU reference engine**. `collect.sh` reads the backend that actually ran from
  the trace log and shows it in the checklist; `--backend hip` / `--backend vulkan` are probed
  and are expected to fail with a clean "not available" error. Any tok/s number from `halo run`
  or `bench model` today is a CPU number, not a HALO GPU result.
- **The useful GPU evidence today is:**
  - the HIP device tests (`test_hip_*`, the `...Device...` tests): they SKIP on the dev host
    with "no HIP device (dev host, D-001)" and must **run and pass** on the EVO-X2. This is the
    first execution of HALO's gfx1151 kernels (GEMV, GDN, attention, KV, head, decode) against
    the CPU reference;
  - the Vulkan tests on RADV (and AMDVLK if installed): they run on lavapipe on the dev host,
    and must see the Radeon 8060S on the EVO-X2;
  - the memory-tier and bandwidth measurements: `HALO_BANDWIDTH_FULL=1` host bandwidth, the
    GTT/carveout allocation tests, the VRAM/GTT sizes (the ~96 GiB carveout, D-002);
  - the llama.cpp same-file baseline (the engine report measured ~20 tok/s decode with MTP);
  - the CPU-engine run of the real model (correct text, memory plan, trace log).
- **The checklist (`SUMMARY.md` section 2) on a healthy EVO-X2:** gfx1151 in KFD and rocminfo;
  `HSA_OVERRIDE_GFX_VERSION` unset; VRAM ~96 GiB; RADV sees the GPU; every kit binary resolves
  its libraries; unit tests 0 failed / 0 crashed; HIP device tests passed with none skipped for
  "no device"; the full bandwidth benchmark ran; Vulkan tests on RADV passed with no
  "Vulkan unavailable" skip; no GPU errors/faults/resets in the kernel log; `halo run` exit 0 on
  the CPU backend.
- **Skips that are normal on the EVO-X2:** tests that need a llama.cpp or Ollama binary when
  none was given; `a HIP device is present; the no-device path is not reachable` (the test of the
  dev-host path); reference-data skips if `halo-ref` was not unpacked.
- **On the dev host** (for comparison), the same kit gives: every HIP device test SKIPPED with
  the D-001 reason, Vulkan tests passing on lavapipe, every GPU step "not available", and the
  checklist's GPU rows N/A.

## 7. Troubleshooting

In order of likelihood on the first run:

1. **`libamdhip64.so.7 => not found`** (verify-env "shared libraries" FAIL for `test_hip_*`).
   Check `ls /opt/rocm/lib/libamdhip64*`. If ROCm lives elsewhere, set `ROCM_PATH` to it (the
   kit then uses `$ROCM_PATH/lib`). If the library exists only under another major version
   (`libamdhip64.so.8`), the prebuilt tests cannot load: use the native build (item 5).
2. **HIP device tests SKIP with "no HIP device" on the EVO-X2.** The runtime saw no GPU: check
   `/dev/kfd` access (render group, then log out and in), `rocminfo` output in
   `gpu/*rocminfo`, and that `HSA_OVERRIDE_GFX_VERSION` is unset. The skip message carries the
   HIP runtime's reason.
3. **`hipErrorNoBinaryForGpu`, "invalid device function", or a crash inside libamdhip64.** The
   ROCm 7.1 code object or runtime ABI does not work with 7.15: use the native build.
4. **`bin/halo` dies with "Illegal instruction".** The CPU lacks an x86-64-v3 feature (not
   expected on Zen 5). Rebuild natively.
5. **Native-build fallback.** On the EVO-X2, from the unpacked kit:

   ```
   scripts/evox2/build-native.sh --out ~/halo/native            # add --no-werror if a newer clang warns
   cd ~/halo/native/halo-native-<sha12>*/ && scripts/evox2/verify-env.sh && scripts/evox2/collect.sh ...
   ```

   It checks the prerequisites of section 2 first and says what is missing, unpacks
   `source/halo-src-<sha12>.tar.gz`, and runs `scripts/package.sh --source copy --march native`
   against `/opt/rocm` (gfx1151, Release, HIP + Vulkan + server + tests). The result is a kit
   directory that `run-tests.sh` and `collect.sh` use exactly like the prebuilt one. Its
   `BUILDINFO` records the EVO-X2's ROCm and compiler. Options: `--src DIR` (another source
   tree), `--cc/--cxx` (e.g. a newer clang), `--jobs N`, `--in-tree-tests`.
6. **Vulkan tests SKIP with "Vulkan unavailable" / zero physical devices.** Check
   `gpu/*vulkaninfo-radv`; install `mesa-vulkan-drivers`; the user must be able to open
   `/dev/dri/renderD*`.
7. **"<anchor> exists and is not a directory owned by uid N".** The test path anchor
   (`/tmp/halo-evox2-<sha12>-anchor`) was created by another user (or by `package.sh
   --keep-build`). Remove it and re-run.
8. **The kernel log is "not readable".** `dmesg` is restricted to root: run `sudo -v` right
   before `collect.sh`, or run `sudo dmesg > dmesg.txt` yourself and bring it back with the
   bundle.

## 8. How the kit is built and laid out (for maintainers)

```
scripts/package.sh --source head --overlay scripts/evox2 --overlay docs/evox2.md ...   # in WSL
```

- **Source.** `--source head` (default) builds `git archive HEAD`, never the live shared working
  tree. `--overlay PATH` (repeatable) copies a working-tree file or directory over HEAD and marks
  the kit dirty if it differs from HEAD. `--source worktree` builds every tracked and untracked,
  non-ignored file as it is; `--source copy` builds the directory the script is in (an unpacked
  source tarball; used by `build-native.sh`).
- **Build.** Release, `-DHALO_MARCH=x86-64-v3` (`--march`), `-DHALO_BUILD_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1151`, Vulkan, server, tests and benchmarks on, `-Werror` on
  (`--no-werror`). Parallelism: `--jobs N`, default `$CMAKE_BUILD_PARALLEL_LEVEL`, else `nproc`.
  Logs go to `dist/logs-<kit>/`.
- **Why an anchor.** The test binaries embed absolute paths at compile time (`HALO_SOURCE_DIR`,
  `HALO_REF_DIR`, the API fixtures, the profiling child helper). So the build happens under one
  fixed prefix, `/tmp/<prefix>-<sha12>-anchor/{src,build,ref}`, and on the target
  `run-tests.sh` recreates that prefix as a directory (mode 0700, owned by the user) of three
  symlinks into the kit and the reference data. `package.sh` checks that every embedded anchor
  path exists in the kit (`halo_kit.py package check`) and warns about embedded non-kit paths
  (test strings and the dev host's llama.cpp default path, which the tests treat as absent).
- **CMake options added for the kit** (root `CMakeLists.txt`): `HALO_MARCH` (empty = no
  `-march`, the old default; `native` = the same as `HALO_NATIVE=ON`; setting both to different
  values is a configure error) and `HALO_WERROR` (default ON). `cmake --install` installs
  `bin/halo` and the benchmark workloads (`cmake/HaloInstall.cmake`); the kit does not use it.
- **Other options:** `--in-tree-tests` runs the kit's test runner on the build tree before
  packaging (compare it with a kit run: `python3 scripts/evox2/lib/halo_kit.py compare A/summary.json
  B/summary.json`), `--keep-build`, `--incremental` (development only; needs rsync), `--no-ref`,
  `--no-tarball`, `--ref DIR`, `--out DIR`, `--name-prefix`, `--cc/--cxx`.
- **Tooling.** `scripts/evox2/*.sh` are thin wrappers around `scripts/evox2/lib/halo_kit.py`
  (Python standard library only, >= 3.10): `run-tests`, `verify-env`, `collect`, `compare`, and
  the build-host `package assemble|check` helpers.
