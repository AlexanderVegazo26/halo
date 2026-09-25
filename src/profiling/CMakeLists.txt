# WS-J: benchmark harness, profiling records, baseline adapters (TRD §27-§28, §34, §49-§51).
# Model/system suites drive the header-only runtime::Engine interface; this module does not
# link halo_runtime (the orchestrator's CLI supplies a real Engine).
halo_module(profiling
  SOURCES timer.cpp stats.cpp counters.cpp hw_state.cpp record.cpp workload.cpp suite.cpp
          sha256.cpp subprocess.cpp http_client.cpp baseline.cpp
  DEPS halo_core halo_hardware nlohmann_json::nlohmann_json Threads::Threads)
