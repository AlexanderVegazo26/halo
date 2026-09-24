halo_module(model
  SOURCES mapped_file.cpp gguf.cpp model.cpp inspect.cpp
  DEPS halo_tensor halo_core nlohmann_json::nlohmann_json)
