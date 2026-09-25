# `halo` CLI tests: run_cli in-process (tools/halo/cli.h) with captured streams, a fixture
# sysfs root, the real GGUF headers / tiny model (HALO_REF_DIR) and the API tests' fake Engine.
if(NOT TARGET halo_cli)
  return()
endif()
set(_cli_test_libs halo_cli)
if(TARGET halo_httplib)
  list(APPEND _cli_test_libs halo_httplib)
endif()
halo_add_test(test_cli SOURCES test_cli.cpp ${PROJECT_SOURCE_DIR}/tests/unit/api/fake_engine.cpp LIBS ${_cli_test_libs})
target_include_directories(test_cli PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit/api)
