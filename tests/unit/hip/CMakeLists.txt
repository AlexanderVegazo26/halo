# HIP backend tests (WS-K). Built only with -DHALO_BUILD_HIP=ON (backends/hip defines the
# target). Emulation tests run everywhere; device tests GTEST_SKIP without a HIP device.
if(NOT TARGET halo_backend_hip)
  return()
endif()
halo_add_test(test_hip_runtime SOURCES test_hip_runtime.cpp LIBS halo_backend_hip hip::host)
halo_add_test(test_hip_gdn SOURCES test_hip_gdn.cpp LIBS halo_backend_hip halo_backend_cpu)
