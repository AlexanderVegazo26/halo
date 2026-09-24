"""Chat-template golden transcripts for HALO (WS-B).

Renders every case with transformers' ``apply_chat_template(..., tokenize=False)`` (Jinja2
in HF's sandboxed environment: trim_blocks, lstrip_blocks, HF ``tojson``,
``raise_exception``) for two templates:

- ``ggml``   : ``$HOME/halo-ref/chat_template.jinja`` (== the ggml-org GGUF's
               ``tokenizer.chat_template``; checked below),
- ``unsloth``: ``tokenizer.chat_template`` extracted with gguf-py from
               ``$HOME/halo-ref/unsloth-ud-q4kxl.header.gguf``.

Output ``$HOME/halo-ref/template_golden/``: ``ggml.jinja``, ``unsloth.jinja`` and
``cases.json`` = ``[{name, template, messages, tools, add_generation_prompt, kwargs,
expected | error}]``. The C++ renderer must match ``expected`` byte-for-byte and must fail
with a typed error wherever HF raised.

Usage (inside WSL)::

    /root/halo-py/.venv/bin/python python/tools/make_template_golden.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REF = Path.home() / "halo-ref"
HF_DIR = REF / "tiny" / "hf"

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city — «unicode» ok.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
                "detail": {"type": "boolean"},
            },
            "required": ["city"],
        },
    },
}
SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"},
                "limit": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "cursor": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
    },
}

U = {"role": "user", "content": "What's the weather in Paris?"}


def assistant_tool_turn() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "Let me check that for you.",
        "reasoning_content": "The user wants weather. I should call the tool.",
        "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": {"city": "Paris", "days": 3, "detail": True, "units": "metric"}}},
            {"id": "call_1", "type": "function",
             "function": {"name": "search",
                          "arguments": {"query": "Paris events\nthis week", "filters": {"lang": "fr", "min": 1.5},
                                        "limit": 1e20, "tags": ["a", "b"], "cursor": None}}},
        ],
    }


def cases() -> list[dict[str, Any]]:
    c: list[dict[str, Any]] = []

    def add(name: str, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
            agp: bool = True, **kwargs: Any) -> None:
        c.append({"name": name, "messages": messages, "tools": tools, "add_generation_prompt": agp,
                  "kwargs": kwargs})

    add("single_turn", [{"role": "user", "content": "Hello!"}])
    add("single_turn_no_generation_prompt", [{"role": "user", "content": "Hello!"}], agp=False)
    add("system_multi_turn", [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ])
    history = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "  Simple arithmetic.\n"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6", "reasoning_content": "Again simple."},
        {"role": "user", "content": "Thanks"},
    ]
    add("reasoning_history_preserve_unset", history)
    add("reasoning_history_preserve_false", history, preserve_thinking=False)
    add("reasoning_history_preserve_true", history, preserve_thinking=True)
    # Prefix stability: turn N (no generation prompt) vs turn N+1.
    add("history_turn_n", history[:4], agp=False)
    add("history_turn_n_plus_1", history, agp=False)
    add("tools_basic", [U], tools=[WEATHER_TOOL, SEARCH_TOOL])
    add("tools_with_system", [{"role": "system", "content": "  You can use tools.  "}, U],
        tools=[WEATHER_TOOL])
    add("tool_calls_round_trip", [
        {"role": "system", "content": "Use tools when helpful."},
        U,
        assistant_tool_turn(),
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 18, \"sky\": \"clear\"}"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Concert on Friday."},
        {"role": "assistant", "content": "It's 18°C and clear; there's a concert Friday."},
        {"role": "user", "content": "Great, thanks!"},
    ], tools=[WEATHER_TOOL, SEARCH_TOOL])
    add("tool_calls_empty_content_bare_function", [
        U,
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}},
                        {"type": "function", "function": {"name": "get_weather", "arguments": {}}}]},
        {"role": "tool", "content": "rain"},
    ], tools=[WEATHER_TOOL])
    add("tool_calls_null_content", [
        U,
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "x"}}}]},
        {"role": "tool", "content": "<tool_response>nested</tool_response>"},
    ], tools=[SEARCH_TOOL], agp=False)
    add("multi_step_tool_last_query", [
        {"role": "user", "content": "Do a thing"},
        {"role": "assistant", "content": "", "reasoning_content": "step 1",
         "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "q1"}}}]},
        {"role": "tool", "content": "r1"},
        {"role": "assistant", "content": "", "reasoning_content": "step 2",
         "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "q2"}}}]},
        {"role": "tool", "content": "r2"},
    ], tools=[SEARCH_TOOL], preserve_thinking=False)
    add("enable_thinking_false", [{"role": "user", "content": "Quick answer please."}], enable_thinking=False)
    add("enable_thinking_false_invalid_effort", [{"role": "user", "content": "Hi"}],
        enable_thinking=False, reasoning_effort="bogus")
    for effort in ["xhigh", "medium", "low", "high", "extreme"]:
        add(f"reasoning_effort_{effort}", [{"role": "user", "content": "Solve it."}], reasoning_effort=effort)
    add("reasoning_effort_medium_with_system", [{"role": "system", "content": "Sys."}, {"role": "user", "content": "x"}],
        reasoning_effort="medium")
    add("reasoning_effort_medium_empty_system", [{"role": "system", "content": "   "}, {"role": "user", "content": "x"}],
        reasoning_effort="medium")
    add("unicode_and_trim", [
        {"role": "system", "content": " 　系统提示\x1c "},
        {"role": "user", "content": " héllo 👋 世界 \n"},
        {"role": "assistant", "content": "　réponse ✓ ", "reasoning_content": " 思考 "},
        {"role": "user", "content": "مرحبا​"},
    ])
    add("typed_content", [
        {"role": "system", "content": [{"type": "text", "text": "Typed system."}]},
        {"role": "user", "content": [{"type": "text", "text": "Look: "}, {"type": "image", "image": "x.png"},
                                      {"type": "text", "text": " and "}, {"type": "video", "video": "v.mp4"}]},
    ])
    add("typed_content_vision_ids", [
        {"role": "user", "content": [{"type": "image"}, {"type": "image_url", "image_url": {"url": "u"}},
                                      {"type": "text", "text": "compare"}]},
    ], add_vision_id=True)
    add("tool_response_user_block", [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "out1"},
        {"role": "user", "content": "next"},
    ])
    # Error cases (raise_exception in the template).
    add("error_no_messages", [])
    add("error_system_not_first", [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}])
    add("error_unexpected_role", [{"role": "user", "content": "a"}, {"role": "narrator", "content": "b"}])
    add("error_no_user_query", [{"role": "system", "content": "s"},
                                {"role": "tool", "content": "only tools"}])
    add("error_system_image", [{"role": "system", "content": [{"type": "image"}]}, {"role": "user", "content": "x"}])
    add("error_unexpected_content_item", [{"role": "user", "content": [{"type": "audio"}]}])
    add("developer_role", [{"role": "developer", "content": "Dev instructions."},
                           {"role": "system", "content": "Second system."},
                           {"role": "user", "content": "hi"}])
    add("tool_call_missing_name", [U, {"role": "assistant", "content": "",
                                       "tool_calls": [{"type": "function", "function": {"arguments": {"a": 1}}}]}],
        agp=False)
    return c


def gguf_chat_template(path: Path) -> str:
    from gguf import GGUFReader  # type: ignore[import-untyped]

    class HeaderReader(GGUFReader):  # type: ignore[misc]
        """GGUFReader that skips tensor mapping (header-only files have no tensor data)."""

        def _build_tensors(self, start_offs: int, fields: Any) -> None:
            self.tensors = []

    return str(HeaderReader(str(path)).fields["tokenizer.chat_template"].contents())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REF / "template_golden"))
    args = ap.parse_args()
    import jinja2
    import transformers
    from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ggml = (REF / "chat_template.jinja").read_text(encoding="utf-8")
    if gguf_chat_template(REF / "ggml-org-q4km.header.gguf") != ggml:
        raise SystemExit("chat_template.jinja differs from the ggml-org GGUF template")
    unsloth = gguf_chat_template(REF / "unsloth-ud-q4kxl.header.gguf")
    (out / "ggml.jinja").write_text(ggml, encoding="utf-8", newline="")
    (out / "unsloth.jinja").write_text(unsloth, encoding="utf-8", newline="")
    tok = Qwen3_5Tokenizer.from_pretrained(str(HF_DIR))

    results = []
    for tname, src in (("ggml", ggml), ("unsloth", unsloth)):
        for case in cases():
            r = dict(copy.deepcopy(case), template=tname)
            try:
                r["expected"] = tok.apply_chat_template(
                    copy.deepcopy(case["messages"]), tools=copy.deepcopy(case["tools"]), chat_template=src,
                    add_generation_prompt=case["add_generation_prompt"], tokenize=False, **case["kwargs"])
            except (jinja2.TemplateError, ValueError, TypeError) as e:
                r["error"] = f"{type(e).__name__}: {e}"
            results.append(r)
    (out / "cases.json").write_text(json.dumps({"cases": results, "transformers": transformers.__version__,
                                                "jinja2": jinja2.__version__}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    n_err = sum(1 for r in results if "error" in r)
    print(f"wrote {len(results)} cases ({n_err} errors) to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
