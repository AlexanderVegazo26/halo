# WS-G M3: Engine on the tiny model (shares tests/unit/models/tiny_golden.h).
halo_add_test(test_runtime
  SOURCES test_engine.cpp
  LIBS halo_runtime)
