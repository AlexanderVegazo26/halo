# Per-sequence GDN state, rollback slots, checkpoints and the SequenceState bundle (WS-G).
halo_module(state
  SOURCES gdn_state.cpp checkpoint.cpp
  DEPS halo_kv_cache halo_backend_cpu halo_core)
