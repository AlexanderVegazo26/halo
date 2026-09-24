# Chat-template rendering (minja) and model-output parsing for the Qwen3.8 chat format.
halo_module(template
  SOURCES chat_template.cpp output_parser.cpp
  DEPS halo_core halo_minja)
