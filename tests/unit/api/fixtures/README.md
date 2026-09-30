Hand-written response-shape fixtures, derived from the public OpenAI Chat Completions /
Completions and Anthropic Messages API reference pages (field names and nesting only; not
captured from a live service). `shape_mismatch` in `api_test_util.h` checks that every key
here exists in a HALO response with a compatible JSON type: `null` means "any value"
(nullable), a string starting with `=` must match exactly, and an array's first element is
the shape of every element.

`claude_code_request.json` is a request, not a response shape: a trimmed reconstruction of the
structure of a `claude -p "hi"` request captured against Claude Code (messages with a trailing
`role:"system"` entry, block `system`, custom tools with `cache_control`, `max_tokens: 32000`,
`thinking: {type: adaptive, display: omitted}`, and the extra top-level `metadata`,
`context_management` and `output_config` fields). The system prompt and tool list are cut down
and the values inside `metadata` / `context_management` are placeholders. It feeds
`AnthropicGolden.*` in `test_api_anthropic_compat.cpp`.
