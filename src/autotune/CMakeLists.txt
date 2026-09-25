# WS-J: autotuner + persistent profile database (TRD §55-§58, §64).
# halo_autotune       profile key, SQLite DB, runtime lookup, cost model, tuner (no backend deps:
#                     this is what the runtime links to consume profiles).
# halo_autotune_cpu   TunableOps over the CPU reference kernels (links halo_backend_cpu); only
#                     built when the cpu backend is selected.
halo_module(autotune
  SOURCES sqlite.cpp profile_key.cpp cost_model.cpp types.cpp db.cpp lookup.cpp tuner.cpp baseline_store.cpp
  DEPS halo_core halo_profiling halo_hardware nlohmann_json::nlohmann_json SQLite::SQLite3)
target_compile_definitions(halo_autotune PRIVATE HALO_VERSION_STRING="${PROJECT_VERSION}")

halo_selected(cpu _wsj_cpu)
if(_wsj_cpu)
  halo_module(autotune_cpu
    SOURCES cpu_ops.cpp
    DEPS halo_autotune halo_backend_cpu)
  # Same fp-contract rule as the CPU backend so validate() compares like with like.
  target_compile_options(halo_autotune_cpu PRIVATE -ffp-contract=off)
endif()
