# WS-K status (read this on resume)
Current milestone: **M7 COMPLETE, awaiting orchestrator commit.** M1 7b5e0bd … M6 c555eb4.
Build filter "tensor;cpu;hip;kv_cache" (scratchpad/k_build.sh).
## M7
(1) GDN gates: kernels/decode.h gdn_gate_body, hd.h hlog1p/softplusf, Ops::gdn_gates (GdnGateArgs), variant gdn_gate_b64; bitwise vs qwen35.cpp composition.
(2) QUANT_GEMM: kernels/gemm.h (16x16 tile, K chunks 64, CPU 8-lane order -> bitwise), Ops::gemm, variant gemm_t16x16_b256 (registry 27).
Tests: test_hip_decode (gates), test_hip_gemv (HipGemm*; T=40 cases added after g3 survived).
Evidence: HIP=ON 80 = 58+22 (HIP 34+22); ASan same, 0 reports; old filter 65 = 45+20; OFF 24/24. Mutations e1-e3, g1-g4 red (g3 after adding T=40).
