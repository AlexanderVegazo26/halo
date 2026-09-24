Hand-written response-shape fixtures, derived from the public OpenAI Chat Completions /
Completions and Anthropic Messages API reference pages (field names and nesting only; not
captured from a live service). `shape_mismatch` in `api_test_util.h` checks that every key
here exists in a HALO response with a compatible JSON type: `null` means "any value"
(nullable), a string starting with `=` must match exactly, and an array's first element is
the shape of every element.
