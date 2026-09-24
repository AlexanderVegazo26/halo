# Paged fp32 KV cache (WS-G): fixed block pool, refcounted prefix sharing, copy-on-write.
halo_module(kv_cache
  SOURCES paged_kv.cpp
  DEPS halo_backend_cpu halo_core)
