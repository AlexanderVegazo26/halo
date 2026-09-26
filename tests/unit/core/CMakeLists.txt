# HALO_LOG_* environment support of halo_core (include/halo/core/log.h). The environment is read
# once per process, so each case runs halo_log_env_helper in a fresh process.
add_executable(halo_log_env_helper log_env_helper.cpp)
target_link_libraries(halo_log_env_helper PRIVATE halo_core halo_warnings)

halo_add_test(test_log_env SOURCES test_log_env.cpp LIBS halo_core)
add_dependencies(test_log_env halo_log_env_helper)
target_compile_definitions(test_log_env PRIVATE HALO_LOG_ENV_HELPER="$<TARGET_FILE:halo_log_env_helper>")
