# Baseline adapter fixtures (WS-J M2)

| File | Provenance |
|---|---|
| `llama_bench_tiny_q8_0.json` | **Captured** by the previous WS-J agent (command recovered from its transcript): `llama-bench -m /root/halo-ref/tiny/tiny-q8_0.gguf -p 16 -n 8 -r 2 -t 2 -o json`, llama.cpp commit `bd4f514` (CPU-only build at `/root/llama-build-wsj`, same commit as the target machine), WSL dev host. |
| `llama_bench_version.txt` | **Captured**: `llama-bench --version` (printed on stderr) from the same binary. |
| `llama_server_completion_tiny.json` | **Captured**: `llama-server` (same build) on the tiny model, `llama-server -m tiny-q8_0.gguf -c 512 -t 2 --host 127.0.0.1 --port 18084`, then a non-streaming `POST /completion`, MTP off. |
| `llama_server_completion_mtp_tiny.json` | **Captured**: as above with `--spec-type draft-mtp --spec-draft-n-max 2` (port 18085). `draft_n` 27 / `draft_n_accepted` 0 on the random tiny model. |
| `ollama_generate.json` | **Hand-written**, not captured. The fields and units follow the Ollama REST API documentation for a non-streaming `POST /api/generate` (github.com/ollama/ollama, `docs/api.md`, "Generate a completion"): durations are **nanoseconds**. The numbers are illustrative, chosen so that the rates are exact: eval_count 100 over 2e9 ns = 50 tok/s, and prompt 26 over 1.3e8 ns = 200 tok/s. Ollama is not installed in the WSL dev environment. Version 0.17.6 exists on the Windows host, but it was not running. Checking it with `ollama list` auto-started the app and server; those processes were stopped again, and no response was captured from them. |
| `ollama_version.json`, `ollama_tags.json` | **Hand-written**, following the same documentation (`GET /api/version`, `GET /api/tags`). The digest is a placeholder. |

These are dev-host smoke outputs (D-001): they test parsers, not performance.
