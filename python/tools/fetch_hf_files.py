"""Fetch real Qwen3.8-27B reference metadata (no weights) into ~/halo-ref.

Part 1: HF tokenizer/template/config files. Part 2: GGUF headers via HTTP Range (see DECISIONS.md sources)."""
import json, struct, sys, urllib.request, os

OUT = os.path.expanduser("~/halo-ref")
os.makedirs(OUT, exist_ok=True)

def get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "halo-dev"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

for repo in ["ggml-org/Qwen3.8-27B-GGUF", "unsloth/Qwen3.8-27B-GGUF", "Qwen/Qwen3.8-27B"]:
    try:
        info = json.loads(get(f"https://huggingface.co/api/models/{repo}/tree/main"))
        print(repo, [(f["path"], f.get("size")) for f in info if f["type"] == "file"])
    except Exception as e:
        print(repo, "ERR", e)

for name in ["tokenizer_config.json", "chat_template.jinja", "tokenizer.json", "generation_config.json"]:
    try:
        data = get(f"https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/{name}")
        open(f"{OUT}/{name}", "wb").write(data)
        print("saved", name, len(data))
    except Exception as e:
        print(name, "ERR", e)
