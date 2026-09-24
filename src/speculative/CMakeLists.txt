# MTP speculative decoding: batched draft / verify / commit and the profit gate (WS-G).
halo_module(speculative
  SOURCES speculative.cpp
  DEPS halo_models halo_state halo_kv_cache halo_core)
target_compile_options(halo_speculative PRIVATE -ffp-contract=off)
