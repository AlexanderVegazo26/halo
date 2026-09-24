# sysfs / procfs fixture trees

Each directory is a fake filesystem root for `halo::hardware::discover({.root = ...})`.
Only the files HALO reads are present. These trees are **hand-written, not captured**
from the target. Formats follow the Linux amdgpu/KFD/procfs interfaces as documented,
but nobody has diffed them against a real EVO-X2 capture yet. `capture.sh` produces
such a capture. Until one replaces `evo_x2/`, any test result on these trees is
"fixture-tested", not "works on the EVO-X2".

## evo_x2/ — the real unit's layout (D-002), with synthetic values where unknown

| Value | Source |
|---|---|
| `mem_info_vram_total` = 103079215104 (98304 MiB) | **fact**: dmesg `98304M of VRAM memory ready` (engine report §1) |
| kernel `7.0.0-31-generic`, Ubuntu 26.04.1 LTS | **fact**: engine report §1 |
| PCI slot `0000:c5:00.0` | **fact**: dmesg line in the engine report |
| gfx1151, 40 CUs, wave32, 2.9 GHz max clock | **fact**: engine report §1. `gfx_target_version 110501`, `simd_count 80` and `simd_per_cu 2` are the KFD encoding of those facts |
| 16 cores / 32 threads, AVX-512 F/VL/BW/DQ/VNNI/BF16, 64 MiB L3 | **fact**: engine report §1. Splitting the L3 into 2 × 32 MiB (one per CCD) is **believed** |
| RADV-only Vulkan ICD, IOMMU off (`amd_iommu=off`) | **stated by the WS-C task**, not by the engine report |
| MemTotal 32559000 kB, MemAvailable 28 GiB | **synthetic**. The report only says "~32 GiB OS-visible" |
| `mem_info_vram_used` = 512 MiB | **synthetic** idle value, chosen before any planner verdict was computed. The report gives only the 50.6 GiB in-benchmark figure |
| `mem_info_gtt_total` = 16670208000 (half of MemTotal) | **synthetic**. Assumes the TTM default of 50% of system RAM. The unit's real GTT size is unrecorded |
| `/opt/rocm/.info/version` = `7.15.26333` | **placeholder** shaped after "HIP 7.15.26333" in the report. The real ROCm package version string is unknown |
| PCI device id 0x1586 | **believed** to be Strix Halo, not verified |
| hwmon, dpm levels, `auto` performance level, power profile | **synthetic** |

## prd_gtt_config/ — the PRD/TRD assumption (small carveout + large GTT)

1 GiB carveout, 120 GiB GTT (`amdgpu.gttsize=122880`, `ttm.pages_limit=31457280`),
126 GiB MemTotal, kernel 6.8, AMDVLK installed next to RADV, IOMMU on (`ivhd0`), no
ROCm, performance level `high`. The DRM uevent has no `PCI_SLOT_NAME`, so this tree
exercises the single-device KFD association fallback. Every value here is synthetic.

## no_gpu/

A CPU-only host with a non-AMD DRM device (Microsoft vendor 0x1414, as under WSL) that
must be ignored. It has a lavapipe ICD and a deliberately truncated ICD manifest.

Git does not track empty directories. A fixture can therefore never contain an empty
`/sys/class/iommu`, and each tree relies on `/proc/cmdline` or an entry to state its
IOMMU status.
