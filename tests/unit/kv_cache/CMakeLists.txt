# WS-G M2: paged KV cache.
halo_add_test(test_kv_cache
  SOURCES test_paged_kv.cpp
  LIBS halo_kv_cache)
# Segmented (per-layer) KV pool: host mapping, COW, and the CPU backend kv_write / attention through
# KvPool::layer_image() vs the contiguous pool. The backend's ops are compared bitwise with the halo::cpu
# reference, hence the same contraction flag as test_backend.
halo_add_test(test_kv_segments
  SOURCES test_kv_segments.cpp
  LIBS halo_kv_cache halo_backend)
target_compile_options(test_kv_segments PRIVATE -ffp-contract=off)
