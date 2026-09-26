# API server tests: in-process ApiServer on an ephemeral 127.0.0.1 port, driven with
# httplib::Client against a scripted fake Engine (fake_engine.h).
if(NOT TARGET halo_api)
  return()
endif()
halo_add_test(test_api
  SOURCES fake_engine.cpp test_api_units.cpp test_api_server.cpp test_api_security.cpp test_api_guard.cpp
  LIBS halo_api halo_httplib halo_template halo_tokenizer halo_core)
target_compile_definitions(test_api PRIVATE HALO_API_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_include_directories(test_api PRIVATE ${PROJECT_SOURCE_DIR}/src/api)  # conn_guard.h (ConnectionGuard unit tests)
