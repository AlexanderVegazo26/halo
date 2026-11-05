# Performance budgets + complexity checks for the CPU reference kernels (test_cpu_perf_budget.cpp).
# RUN_SERIAL: ctest -j would otherwise run other tests on the same core mid-measurement.
# Rerun only these: ctest --test-dir build -L perf --output-on-failure
halo_add_test(test_cpu_perf_budget
  SOURCES test_cpu_perf_budget.cpp
  LIBS halo_backend_cpu
  PROPERTIES RUN_SERIAL TRUE LABELS perf)
# End-to-end decode/prefill budgets + context scaling for the CPU Engine on a synthetic model
# (test_decode_budget.cpp; no HALO_REF_DIR needed). Same RUN_SERIAL + `perf` label.
# Rerun only it: ctest --test-dir build -R test_decode_budget --output-on-failure
halo_add_test(test_decode_budget
  SOURCES test_decode_budget.cpp
  LIBS halo_runtime
  PROPERTIES RUN_SERIAL TRUE LABELS perf)
