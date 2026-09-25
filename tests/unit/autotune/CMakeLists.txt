halo_add_test(test_autotune
  SOURCES test_key.cpp test_db.cpp test_tuner.cpp test_baseline_store.cpp
  LIBS halo_autotune SQLite::SQLite3)

# Real CPU-kernel tuning; only when the cpu backend is part of the build.
if(TARGET halo_autotune_cpu)
  halo_add_test(test_autotune_cpu
    SOURCES test_cpu_tune.cpp
    LIBS halo_autotune_cpu)
endif()
