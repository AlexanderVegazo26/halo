# WS-D: CPU reference operator tests.
halo_add_test(test_cpu_thread_pool SOURCES test_thread_pool.cpp LIBS halo_backend_cpu)
halo_add_test(test_cpu_kernels
  SOURCES test_ops.cpp test_gdn.cpp test_attention.cpp test_golden.cpp
  LIBS halo_backend_cpu)
