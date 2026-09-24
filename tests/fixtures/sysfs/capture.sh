#!/usr/bin/env bash
# Snapshot the files halo::hardware::discover() reads into a fixture tree, so a real
# machine (e.g. the EVO-X2) can replace the hand-written tests/fixtures/sysfs/evo_x2/.
#   tests/fixtures/sysfs/capture.sh <output-dir>
# Read-only on the host. Review the output before committing it: /proc/cmdline and the
# DRM uevent can contain machine identifiers (root UUID, subsystem ids).
set -euo pipefail
OUT="${1:?usage: capture.sh <output-dir>}"
mkdir -p "$OUT"

copy() {  # copy a readable regular file, preserving its path under $OUT
  local src="$1"
  [[ -f "$src" && -r "$src" ]] || return 0
  mkdir -p "$OUT$(dirname "$src")"
  cat "$src" > "$OUT$src" 2>/dev/null || rm -f "$OUT$src"
}

for f in /proc/cpuinfo /proc/meminfo /proc/cmdline /proc/sys/kernel/osrelease /etc/os-release \
         /usr/lib/os-release; do
  copy "$f"
done
for d in /opt/rocm /opt/rocm-*; do
  [[ -d "$d" ]] && copy "$d/.info/version"
done
for d in /etc/xdg/vulkan/icd.d /etc/vulkan/icd.d /usr/local/share/vulkan/icd.d /usr/share/vulkan/icd.d; do
  for f in "$d"/*.json; do copy "$f"; done
done
for f in /sys/devices/system/cpu/cpu[0-9]*/cache/index*/{level,size,shared_cpu_list}; do copy "$f"; done
for dev in /sys/class/drm/card[0-9]*/device; do
  [[ -d "$dev" ]] || continue
  card="$(basename "$(dirname "$dev")")"
  base="/sys/class/drm/$card/device"
  for f in vendor device uevent mem_info_vram_total mem_info_vram_used mem_info_vis_vram_total \
           mem_info_gtt_total mem_info_gtt_used pp_dpm_sclk pp_dpm_mclk \
           power_dpm_force_performance_level pp_power_profile_mode; do
    copy "$base/$f"
  done
  for h in "$base"/hwmon/hwmon*; do
    for f in name temp1_input power1_average power1_input; do copy "$h/$f"; done
  done
done
for n in /sys/class/kfd/kfd/topology/nodes/*; do copy "$n/properties"; done
# /sys/class/iommu entries are directories of symlinks; record their names only.
if [[ -d /sys/class/iommu ]]; then
  for e in /sys/class/iommu/*; do
    [[ -e "$e" ]] || continue
    mkdir -p "$OUT/sys/class/iommu/$(basename "$e")"
    : > "$OUT/sys/class/iommu/$(basename "$e")/.captured"
  done
fi
echo "captured into $OUT"
