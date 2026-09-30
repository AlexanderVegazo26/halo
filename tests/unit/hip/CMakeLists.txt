# HIP backend tests (WS-K). Built only with -DHALO_BUILD_HIP=ON (backends/hip defines the
# target). Emulation tests run everywhere; device tests GTEST_SKIP without a HIP device.
if(NOT TARGET halo_backend_hip)
  return()
endif()
halo_add_test(test_hip_runtime GPU SOURCES test_hip_runtime.cpp LIBS halo_backend_hip hip::host)
halo_add_test(test_hip_gdn GPU SOURCES test_hip_gdn.cpp LIBS halo_backend_hip halo_backend_cpu)
halo_add_test(test_hip_gemv GPU SOURCES test_hip_gemv.cpp LIBS halo_backend_hip halo_backend_cpu halo_tensor)
# The error bound uses the kernels' own summation-depth formulas (kernels/gemv.h).
target_include_directories(test_hip_gemv PRIVATE ${PROJECT_SOURCE_DIR}/backends/hip)
halo_add_test(test_hip_head GPU SOURCES test_hip_head.cpp LIBS halo_backend_hip halo_backend_cpu halo_tensor)
target_include_directories(test_hip_head PRIVATE ${PROJECT_SOURCE_DIR}/backends/hip)
# Attention tests build their paged inputs with the real halo::kv_cache pool (layout parity).
if(TARGET halo_kv_cache)
  halo_add_test(test_hip_attention GPU SOURCES test_hip_attention.cpp LIBS halo_backend_hip halo_backend_cpu halo_kv_cache)
else()
  message(WARNING "HIP: halo_kv_cache is not selected (add kv_cache to HALO_ONLY); test_hip_attention and "
                  "test_hip_kv (paged attention and KV write vs halo::cpu / kv_cache) are NOT built")
endif()
halo_add_test(test_hip_decode GPU SOURCES test_hip_decode.cpp LIBS halo_backend_hip halo_backend_cpu halo_tensor)
if(TARGET halo_kv_cache)
  halo_add_test(test_hip_kv GPU SOURCES test_hip_kv.cpp LIBS halo_backend_hip halo_kv_cache)
endif()
