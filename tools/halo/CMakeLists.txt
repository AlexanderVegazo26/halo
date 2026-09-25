# `halo` CLI (PRD §14, TRD §26). The command logic is the halo_cli library (tests drive it
# in-process); main.cpp adds the process environment, signals and the real engine factory.
#
# Optional components (each detected with if(TARGET ...), so the CLI builds with any subset):
#   halo_runtime      serve / run / bench model|system can create an engine ("runtime not built" otherwise)
#   halo_api          serve; chat prompts use the API's special-token protection
#   halo_profiling    bench
#   halo_backend_cpu  bench micro kernels
set(_halo_cli_deps halo_model halo_memory halo_hardware halo_tokenizer halo_template halo_core
                   nlohmann_json::nlohmann_json Threads::Threads)
set(_have_runtime 0)
set(_have_api 0)
set(_have_profiling 0)
set(_have_cpu 0)
if(TARGET halo_runtime)
  list(APPEND _halo_cli_deps halo_runtime)
  set(_have_runtime 1)
endif()
if(TARGET halo_api)
  list(APPEND _halo_cli_deps halo_api)
  set(_have_api 1)
endif()
if(TARGET halo_profiling)
  list(APPEND _halo_cli_deps halo_profiling)
  set(_have_profiling 1)
  if(TARGET halo_backend_cpu)
    list(APPEND _halo_cli_deps halo_backend_cpu)
    set(_have_cpu 1)
  endif()
endif()

add_library(halo_cli STATIC cli.cpp args.cpp config.cpp model_io.cpp commands.cpp bench.cpp)
target_include_directories(halo_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(halo_cli PUBLIC ${_halo_cli_deps} PRIVATE halo_warnings)
target_compile_definitions(halo_cli PUBLIC HALO_CLI_HAVE_RUNTIME=${_have_runtime} HALO_CLI_HAVE_API=${_have_api}
                                           HALO_CLI_HAVE_PROFILING=${_have_profiling} HALO_CLI_HAVE_CPU=${_have_cpu})
message(STATUS "halo CLI: runtime=${_have_runtime} api=${_have_api} profiling=${_have_profiling} cpu_kernels=${_have_cpu}")

add_executable(halo_cli_main main.cpp)
set_target_properties(halo_cli_main PROPERTIES OUTPUT_NAME halo)
target_link_libraries(halo_cli_main PRIVATE halo_cli halo_warnings)
