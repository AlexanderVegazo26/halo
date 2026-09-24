# WS-H: sampler chain, JSON-Schema grammar, token masks.
halo_add_test(test_sampling
  SOURCES test_smoke.cpp test_sampler.cpp test_grammar.cpp
  LIBS halo_sampling halo_tokenizer halo_backend_cpu)
