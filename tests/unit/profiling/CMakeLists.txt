halo_add_test(test_profiling
  SOURCES test_stats.cpp test_timer_counters.cpp test_hw_state.cpp test_record.cpp test_workload.cpp
          test_suites.cpp test_sha256.cpp test_subprocess.cpp test_http.cpp test_baseline.cpp test_security_m4.cpp
          test_baseline_integration.cpp
  LIBS halo_profiling)

# Helper child process for the subprocess / baseline tests (never a shell).
add_executable(halo_wsj_child_helper child_helper.cpp)
target_link_libraries(halo_wsj_child_helper PRIVATE halo_warnings)

# llama.cpp binaries for the integration tests (llama-bench is not in /root/llama.cpp/build/bin).
set(HALO_LLAMA_BIN_DIR "$ENV{HOME}/llama-build-wsj/bin" CACHE PATH "llama.cpp bin dir for baseline integration tests")
foreach(t IN ITEMS test_profiling)
  if(TARGET ${t})
    add_dependencies(${t} halo_wsj_child_helper)
    target_compile_definitions(${t} PRIVATE HALO_CHILD_HELPER="$<TARGET_FILE:halo_wsj_child_helper>"
                                            HALO_LLAMA_BIN_DIR="${HALO_LLAMA_BIN_DIR}")
  endif()
endforeach()
