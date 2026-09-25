# Runtime Engine (WS-G M3): CPU reference engine, batched scheduler, prefix cache.
halo_module(runtime
  SOURCES engine.cpp profile.cpp
  DEPS halo_autotune halo_profiling halo_speculative halo_models halo_state halo_kv_cache halo_tokenizer halo_template halo_memory halo_hardware
       halo_sampling halo_core)
