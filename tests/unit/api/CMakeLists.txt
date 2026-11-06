# API server tests: in-process ApiServer on an ephemeral 127.0.0.1 port, driven with
# httplib::Client against a scripted fake Engine (fake_engine.h).
if(NOT TARGET halo_api)
  return()
endif()
halo_add_test(test_api
  SOURCES fake_engine.cpp test_api_units.cpp test_api_server.cpp test_api_security.cpp test_api_guard.cpp
          test_api_anthropic_compat.cpp
  LIBS halo_api halo_httplib halo_template halo_tokenizer halo_core)
target_compile_definitions(test_api PRIVATE HALO_API_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_include_directories(test_api PRIVATE ${PROJECT_SOURCE_DIR}/src/api)  # conn_guard.h (ConnectionGuard unit tests)

# k6 load layer (load/README.md): the real ApiServer over the FakeEngine as a standalone process,
# driven by k6. The blocking smoke runs in ctest (and so in scripts/test-agent and both CI jobs);
# it reports SKIPPED, not passed, where k6 is not installed.
add_executable(halo_load_server load_server.cpp fake_engine.cpp)
target_link_libraries(halo_load_server PRIVATE halo_api halo_httplib halo_template halo_tokenizer halo_core halo_warnings)
target_include_directories(halo_load_server PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(halo_load_server PRIVATE HALO_REF_DIR="${HALO_REF_DIR}")
if(Python3_Interpreter_FOUND)
  # smoke.js budgets are for an optimized build; sanitizer and unoptimized builds run the hosted
  # runner's smoke ~20x slower (CI 2026-11-06: p95 2.5 ms release, 54 ms ASan Debug, 31 ms TSan).
  set(_halo_load_scale 1)
  if(HALO_ENABLE_ASAN OR HALO_ENABLE_TSAN OR NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_halo_load_scale 20)
  endif()
  add_test(NAME load_smoke
           COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/load/run.py
                   --server $<TARGET_FILE:halo_load_server> --budget-scale=${_halo_load_scale}
                   ${PROJECT_SOURCE_DIR}/load/smoke.js)
  set_tests_properties(load_smoke PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS load
                                             RESOURCE_LOCK load_smoke)
endif()
