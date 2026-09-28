# Per-sequence GDN state ring, checkpoints and the SequenceState bundle (WS-G, WS-BI-2).
halo_module(state
  SOURCES gdn_state.cpp checkpoint.cpp
  DEPS halo_kv_cache halo_backend halo_backend_cpu halo_core)
