halo_module(hardware
  SOURCES parse.cpp fs.cpp discovery.cpp checklist.cpp bandwidth.cpp
  DEPS halo_core nlohmann_json::nlohmann_json Threads::Threads)
