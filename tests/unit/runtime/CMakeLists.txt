# WS-G M3: Engine on the tiny model (shares tests/unit/models/tiny_golden.h).
halo_add_test(test_runtime
  SOURCES test_engine.cpp test_engine_faults.cpp test_engine_split_mtp.cpp
  LIBS halo_runtime)
# BI-6: engine-over-GPU-backend tests. halo_runtime PUBLIC-links the adapters when they are
# built (src/runtime/CMakeLists.txt); the defines only gate the test cases' extra includes.
if(TARGET halo_backend_hip_adapter)
  target_compile_definitions(test_runtime PRIVATE HALO_TEST_HIP=1)
endif()
if(TARGET halo_backend_vulkan)
  target_compile_definitions(test_runtime PRIVATE HALO_TEST_VULKAN=1)
endif()
