# Sampler (TRD §23) and JSON-Schema structured output via token masks (TRD §25). WS-H.
halo_module(sampling
  SOURCES sampler.cpp grammar.cpp json_grammar.cpp json_schema.cpp token_mask.cpp
  DEPS halo_core halo_tokenizer)
