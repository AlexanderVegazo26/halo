# Paged fp32 KV cache (WS-G): fixed block pool, refcounted prefix sharing, copy-on-write.
# WS-BI-2 stage 2: the pool image is a State-arena backend buffer (ADR-001 §5.2).
halo_module(kv_cache
  SOURCES paged_kv.cpp
  DEPS halo_backend halo_backend_cpu halo_core)
