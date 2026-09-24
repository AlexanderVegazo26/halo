# HALO CPU reference backend (WS-D). Float32 reference operators; the semantic definition
# every GPU kernel is differentially tested against (TRD §30).
add_library(halo_backend_cpu STATIC
  thread_pool.cpp
  ops_basic.cpp
  conv1d.cpp
  gated_delta_rule.cpp
  attention.cpp
  matmul.cpp)
# backends/ does not use the halo_module() helper, so add the public include dir here.
target_include_directories(halo_backend_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include
                                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(halo_backend_cpu PUBLIC halo_core Threads::Threads PRIVATE halo_warnings)
# Reference results must not depend on HALO_NATIVE: with -march=native the compiler may
# contract a*b+c into FMA (different rounding). Keep every multiply and add separate.
target_compile_options(halo_backend_cpu PRIVATE -ffp-contract=off)

if(HALO_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
