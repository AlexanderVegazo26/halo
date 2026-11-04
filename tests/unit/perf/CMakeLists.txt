# Performance budgets + complexity checks for the CPU reference kernels (test_cpu_perf_budget.cpp).
# RUN_SERIAL: ctest -j would otherwise run other tests on the same core mid-measurement.
# Rerun only these: ctest --test-dir build -L perf --output-on-failure
halo_add_test(test_cpu_perf_budget
  SOURCES test_cpu_perf_budget.cpp
  LIBS halo_backend_cpu
  PROPERTIES RUN_SERIAL TRUE LABELS perf)
