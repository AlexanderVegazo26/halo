# Qwen3.8 ("qwen35") forward + MTP block + GGUF adapters (WS-G), composed over the backend
# interface (ADR-001, WS-BI-1).
halo_module(models
  SOURCES adapters.cpp qwen35.cpp
  DEPS halo_backend halo_model halo_tensor halo_backend_cpu halo_kv_cache halo_state halo_tokenizer halo_core)
# Reference results must not depend on HALO_NATIVE (same reason as halo_backend_cpu).
target_compile_options(halo_models PRIVATE -ffp-contract=off)
