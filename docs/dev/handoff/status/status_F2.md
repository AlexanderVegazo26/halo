# WS-F2 status (attempt #3) — updated after M1
Milestone M1 (M-1 Vulkan D-016) = GREEN, reported to orchestrator, awaiting commit. Do NOT start M2 until told.
- HEAD baseline (git archive -> /root/halo-f2-base, HALO_ONLY=tensor;cpu;cpu_kernels;vulkan): 113/113, 0 skip.
- M1 tree: 118/118 pass, 0 fail, 0 skip (tensor 13, cpu_kernels 39, thread_pool 9, vulkan 54, core 3). ASan 118/118. Base HEAD 113 (cpu_kernels 38, vulkan 50), 0 skip measured.
- Demonstrate-fail done in /root/halo-f2-mut (a no L2, b q_scale ignored, c CPU ignores q_scale, d no isfinite, e tree start WG/2).
  Gotcha: git-archive files carry commit mtimes -> ninja keeps stale objects; rm build dir when restoring.
M1 files: gdn shader, vulkan ops.cpp/ops.h (HEAD API + qk_l2norm/q_scale fields), test_vk_ops.cpp (gdn_args sets
  qk_l2norm=false,q_scale=1), test_vk_differential.cpp (new), tests/unit/vulkan/CMakeLists.txt, test_gdn.cpp.
PARKED: S-1/S-3 BufferView redesign in scratchpad/s1-wip/ (ops.h, ops.cpp, rms_norm.comp are complete host-side
  redesign; MISSING: GDN view push consts in shader, matvec shaders offsets, argmax NaN flag shaders, Q5_K/IQ4_XS
  shaders, test_vk_ops.cpp port to new API). Its ops.cpp GDN Push layout != current shader.
Next (when told): M-2 (dequant via tensor::dequantize_row, differential rms_norm/argmax, N-2 PUBLIC halo_tensor), then S-3, S-1.

## M2 (M-2 + N-2) — in progress (M1 committed as ce9bb69)
- reference.h: ggml dequant port removed; deq via tensor::dequantize_row; Q4_K mag via tensor dequant of dmin=0 / d=0 copies.
- test_vk_ops.cpp: rms_norm, matvec (cpu::matmul + tensor dequant WeightMatrix), GDN (cpu recurrent), argmax, chained all vs halo::cpu; fp64 only for scales.
- argmax NaN: CPU throw asserted; GPU NaN expectations kept as "pre-S-3" (S-3 flips them).
- N-2: backends/vulkan/CMakeLists links halo_tensor PUBLIC (+FATAL if missing).
- Build: 118/118 green. Mutations running: scratchpad/mut2.sh -> mut2.out.
- M2 GREEN: 118/118, 0 skip (per-binary scan), ASan 118/118. Mutations m1..m5 (cpu rms_norm, tensor Q4_K, cpu matmul,
  cpu gdn beta, cpu argmax ties) each red on the expected tests; restore green (54 vulkan). N-2 FATAL verified.
- Awaiting commit of M2. Next milestone when told: S-3 (argmax NaN flag, flips the 3 "pre-S-3" GPU expectations).

## M3 = S-3 (argmax NaN -> Error(Kernel) on Vulkan) — GREEN, awaiting commit (M2 committed as 2805ed6)
Files: backends/vulkan/shaders/reduce/argmax_partial.comp, argmax_final.comp (3 words/partial+result: idx, value, nan;
  NaN OR-reduced through both passes; final always rewrites all 3 words), backends/vulkan/src/ops.cpp (scratch 12 B/partial,
  shared-mem check rw*12, decode_argmax/read_argmax), include/halo/backends/vulkan/ops.h (k_argmax_result_bytes 8->12,
  ArgmaxResult, decode_argmax, read_argmax), tests/unit/vulkan/test_vk_ops.cpp.
API change: result buffer must be >= 12 bytes (8-byte results now rejected); consumers must use read_argmax/decode_argmax.
  No consumers outside my scope (grep).
Tests: EdgeValues NaN cases now differential (cpu throws <=> vk throws, + raw NaN word == 1): NaN near max, only -inf
  non-NaN, all NaN, single NaN, NaN at 0/4095/4096/49999, negative NaN. New NanFlagAcrossWorkgroupSizesAndResultReuse
  (wg 32/64/256, n=248320, NaN then clean on same result buffer x2), DecodeArgmaxRaisesOnNanWordAndMissingIndex.
  ValidatesBuffers: 8-byte result rejected. Chained test uses read_argmax.
Counts: 120/120, 0 skip (per-binary: tensor 13, cpu_kernels 39, pool 9, vulkan 56, core 3). ASan 120/120, 0 skip.
Demonstrate-fail (mut4.sh / mut4.out): s1 partial never sets flag, s2 partial tree drops OR, s3 final drops partial flags,
  s4 final doesn't rewrite flag -> EdgeValues + NanFlag... red; s5 host decode ignores word -> also Decode... red. Restore green.
Next when told: S-1 from s1-wip (note s1-wip/ops.cpp argmax already matches this 3-word layout).
Proposed commit: "backends/vulkan: argmax NaN flag raises Error(Kernel) on the host like cpu::argmax (D-016, review S-3)"

## M4 = S-1 (+ S-6 shaders) — IN PROGRESS (S-3 committed 05d0351)
- Restored s1-wip host (ops.h, ops.cpp, rms_norm.comp) + rw*12 shared check. Pre-restore copies in scratchpad/f2_s1_pre/.
- Shaders ported to views: matvec_{f32,q8_0,q4_k,q6_k} (w_off,w_stride,x_off,y_off), GDN (q/k/v/g/b/o off+stride),
  argmax (x_off,p_off,r_off). halo_common read_f16 handles odd byte offsets.
- S-6: new matvec_q5_k.comp, matvec_iq4_xs.comp registered; VkMatvec instantiated for them (vs cpu::matmul+tensor).
- Tests ported (gdn_args views, f32_view helper, differential file). Build 124/124 green.
- NEXT: new test_vk_views.cpp (arena weights at odd byte offsets, fused qkv views, strided rms_norm, argmax sub-views,
  byte-overlap accept/reject), mutations (f2_mut_s1.sh), ASan, report.
- M4 GREEN (awaiting commit): 130/130, 0 skip (vulkan 66), ASan 130/130. New test_vk_views.cpp (5 tests) + fused qkv GDN test.
  Mutations f2_mut_s1.sh/.out + f2_mut_s1b.sh/.out: v1..v11 + v4/v4b/v4c all red on the targeted tests; restore green.
  Note: first fused fixture had 16-byte-aligned q/k views (remainder 0) so v4 stayed green -> fixture now lead=1 float.
- Next when told: S-5 (long trajectory CPU/Vulkan, in-place multi-row slots) then N-1 (bounded staging).

## M5 = S-5 + N-1 + docs/vulkan.md — STARTED (S-1/S-6 committed fb86e77). HALO_ONLY must exclude template (WS-L).
- M5 GREEN (awaiting commit): 134/134 (tensor 13, cpu_kernels 40, pool 9, vulkan 69, core 3), 0 skip; ASan ctest 134/134.
  S-5: CpuGdn.LongTrajectory..., VkDiffGdn.LongTrajectory..., VkDiffGdn.InPlaceMultiRowWithSlotsMatchesCpu.
  N-1: Context-owned reusable staging (ContextOptions::staging_bytes, default 16 MiB, chunked), VkBuffer.StagingIsBoundedReusedAndChunked.
  docs/vulkan.md written. Mutations f2_mut_s5.sh/.out + floor probes f2_mut_s5b.sh/.out.
  Detection floor: vk decay bias 1e-5/token NOT caught, 1e-4 caught (ratio 1.002); cpu chunked 1e-4/chunk NOT caught, 1e-3 caught.
- WS-F2 workstream complete after this commit (N-1 was the last listed item).
