# Micro-benchmarks for the CPU reference kernels (not tests; dev-host timings only, D-001).
add_executable(halo_bench_cpu_gdn bench_gdn.cpp)
target_link_libraries(halo_bench_cpu_gdn PRIVATE halo_backend_cpu halo_warnings)
