# HTTP API server (OpenAI + Anthropic + utility routes) over cpp-httplib. Depends only on the
# Engine *contract* (include/halo/runtime/engine.h), not on halo_runtime.
if(NOT TARGET halo_httplib)
  message(STATUS "halo_api: HALO_BUILD_SERVER=OFF, API server not built")
  return()
endif()

halo_module(api
  SOURCES json_util.cpp prompt.cpp admission.cpp metrics.cpp generation.cpp requests.cpp server.cpp
  DEPS halo_template halo_tokenizer halo_core nlohmann_json::nlohmann_json Threads::Threads)
target_link_libraries(halo_api PRIVATE halo_httplib)
# Structured-output schemas are compiled at request parse time when the sampler is built
# (an invalid schema is then a 400 before streaming starts; review R-4).
if(TARGET halo_sampling)
  target_link_libraries(halo_api PRIVATE halo_sampling)
  target_compile_definitions(halo_api PUBLIC HALO_API_HAVE_SAMPLING=1)
else()
  target_compile_definitions(halo_api PUBLIC HALO_API_HAVE_SAMPLING=0)
endif()
