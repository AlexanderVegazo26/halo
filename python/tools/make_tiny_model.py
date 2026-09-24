"""Generate a tiny random Qwen3.5 ("qwen35") model + golden reference outputs.

The tiny model keeps every structural property of Qwen3.8-27B that the runtime must get
right (DECISIONS.md D-004/D-005): hybrid layout with full attention every 4th layer, GQA,
partial RoPE (1/4 of head_dim), attention output gate, Gated DeltaNet with
num_v_heads = 3 x num_k_heads (exercises the tiled V-head reorder), conv kernel 4,
untied embeddings, the real 248,320-token vocabulary/tokenizer, and one MTP block.

Outputs (under --out, default ~/halo-ref/tiny):
  hf/                 HF checkpoint (safetensors) incl. mtp.* tensors, real tokenizer files
  tiny-f32.gguf       converted with llama.cpp convert_hf_to_gguf.py (--outtype f32)
  tiny-q8_0.gguf      same, --outtype q8_0
  golden/manifest.json + *.bin   reference tensors (little-endian, see manifest)

Reference semantics: trunk = transformers Qwen3_5ForCausalLM (float32, eager);
MTP = NumPy port of llama.cpp's qwen35 graph_mtp (transformers does not implement MTP).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

REF = Path.home() / "halo-ref"
LLAMA_CPP = Path.home() / "llama.cpp"
SEED = 1234

TINY = dict(
    hidden_size=256,  # multiple of 256 so K-/IQ-quant rows are representable
    intermediate_size=512,
    num_hidden_layers=8,  # layers 3 and 7 are full attention
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,  # rotary dims = 16
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    vocab_size=248320,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    attn_output_gate=True,
    tie_word_embeddings=False,
    mtp_num_hidden_layers=1,
    rope_parameters=dict(rope_type="default", rope_theta=10_000_000.0, partial_rotary_factor=0.25,
                         mrope_section=[1, 1, 2], mrope_interleaved=True),
    bos_token_id=248044,
    eos_token_id=248046,
)


def build_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(**TINY)
    cfg.architectures = ["Qwen3_5ForCausalLM"]
    return cfg


@torch.no_grad()
def randomize(model: torch.nn.Module) -> None:
    """Re-initialize so logits are sharp (argmax robust to fp noise) and norms are non-trivial."""
    g = torch.Generator().manual_seed(SEED)
    for name, p in model.named_parameters():
        if p.ndim == 2:  # linear / embedding
            fan_in = p.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            if "embed_tokens" in name:
                std = 1.0
            elif name == "lm_head.weight":
                std = 8.0 / math.sqrt(fan_in)  # sharp logits: top-1 margins well above fp noise
            p.copy_(torch.randn(p.shape, generator=g) * std)
        elif name.endswith("A_log"):
            p.copy_(torch.log(torch.empty(p.shape).uniform_(0.5, 8.0, generator=g)))
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        elif "conv1d" in name:
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        elif name.endswith("linear_attn.norm.weight"):  # plain (not zero-centered) RMSNormGated
            p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
        elif name.endswith("norm.weight"):  # zero-centered RMSNorm: effective scale = 1 + w
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        else:
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)


def make_mtp_tensors(cfg, g: torch.Generator) -> dict[str, torch.Tensor]:
    """One MTP block in HF naming (what llama.cpp's converter maps to blk.<n_layer>.nextn.*)."""
    H, I = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    def lin(o, i):
        return torch.randn(o, i, generator=g) / math.sqrt(i)

    def zc(n):  # zero-centered norm weight
        return torch.randn(n, generator=g) * 0.1

    t = {
        "mtp.fc.weight": lin(H, 2 * H),
        "mtp.pre_fc_norm_embedding.weight": zc(H),
        "mtp.pre_fc_norm_hidden.weight": zc(H),
        "mtp.norm.weight": zc(H),
        "mtp.layers.0.input_layernorm.weight": zc(H),
        "mtp.layers.0.post_attention_layernorm.weight": zc(H),
        "mtp.layers.0.self_attn.q_proj.weight": lin(nh * hd * 2, H),
        "mtp.layers.0.self_attn.k_proj.weight": lin(nkv * hd, H),
        "mtp.layers.0.self_attn.v_proj.weight": lin(nkv * hd, H),
        "mtp.layers.0.self_attn.o_proj.weight": lin(H, nh * hd),
        "mtp.layers.0.self_attn.q_norm.weight": zc(hd),
        "mtp.layers.0.self_attn.k_norm.weight": zc(hd),
        "mtp.layers.0.mlp.gate_proj.weight": lin(I, H),
        "mtp.layers.0.mlp.up_proj.weight": lin(I, H),
        "mtp.layers.0.mlp.down_proj.weight": lin(H, I),
    }
    return {k: v.contiguous() for k, v in t.items()}


# ----------------------------------------------------------------------------- MTP reference
def _rms(x, w_zc, eps):  # zero-centered HF norm
    x = x.astype(np.float64)
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * (1.0 + w_zc.astype(np.float64))


def _rope(x, pos, rot, theta):  # neox rotate-half on first `rot` dims; x [T, heads, hd]
    inv = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    ang = pos[:, None].astype(np.float64) * inv[None, :]  # [T, rot/2]
    cos = np.concatenate([np.cos(ang)] * 2, -1)[:, None, :]
    sin = np.concatenate([np.sin(ang)] * 2, -1)[:, None, :]
    xr, xp = x[..., :rot], x[..., rot:]
    half = rot // 2
    rh = np.concatenate([-xr[..., half:], xr[..., :half]], -1)
    return np.concatenate([xr * cos + rh * sin, xp], -1)


def mtp_reference(cfg, w: dict[str, np.ndarray], embed: np.ndarray, lm_head: np.ndarray,
                  tokens_next: np.ndarray, h: np.ndarray, positions: np.ndarray):
    """MTP over a sequence: row i combines embed(tokens_next[i]) and trunk hidden h[i]
    (post output-norm) at RoPE position positions[i]; causal attention over rows."""
    eps = cfg.rms_norm_eps
    H, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rot = int(hd * cfg.rope_parameters["partial_rotary_factor"])
    theta = cfg.rope_parameters["rope_theta"]
    T = len(tokens_next)
    e = _rms(embed[tokens_next], w["mtp.pre_fc_norm_embedding.weight"], eps)
    hh = _rms(h, w["mtp.pre_fc_norm_hidden.weight"], eps)
    x = np.concatenate([e, hh], -1) @ w["mtp.fc.weight"].T.astype(np.float64)
    res = x
    xn = _rms(x, w["mtp.layers.0.input_layernorm.weight"], eps)
    qg = (xn @ w["mtp.layers.0.self_attn.q_proj.weight"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
    k = (xn @ w["mtp.layers.0.self_attn.k_proj.weight"].T).reshape(T, nkv, hd)
    v = (xn @ w["mtp.layers.0.self_attn.v_proj.weight"].T).reshape(T, nkv, hd)
    q = _rms(q, w["mtp.layers.0.self_attn.q_norm.weight"], eps)
    k = _rms(k, w["mtp.layers.0.self_attn.k_norm.weight"], eps)
    q, k = _rope(q, positions, rot, theta), _rope(k, positions, rot, theta)
    rep = nh // nkv
    out = np.zeros((T, nh, hd))
    for hi in range(nh):
        kh, vh = k[:, hi // rep], v[:, hi // rep]
        s = (q[:, hi] @ kh.T) / math.sqrt(hd)
        s = s + np.triu(np.full((T, T), -np.inf), 1)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        out[:, hi] = s @ vh
    attn = out.reshape(T, nh * hd) * (1.0 / (1.0 + np.exp(-gate)))
    x = res + attn @ w["mtp.layers.0.self_attn.o_proj.weight"].T
    res = x
    xn = _rms(x, w["mtp.layers.0.post_attention_layernorm.weight"], eps)
    gp = xn @ w["mtp.layers.0.mlp.gate_proj.weight"].T
    up = xn @ w["mtp.layers.0.mlp.up_proj.weight"].T
    x = res + ((gp / (1.0 + np.exp(-gp))) * up) @ w["mtp.layers.0.mlp.down_proj.weight"].T
    hn = _rms(x, w["mtp.norm.weight"], eps)
    logits = hn @ lm_head.T.astype(np.float64)
    return hn.astype(np.float32), logits.astype(np.float32)


# ----------------------------------------------------------------------------- golden writer
class Golden:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}

    def add(self, name: str, arr, **meta):
        a = np.ascontiguousarray(np.asarray(arr))
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        if a.dtype == np.int64:
            a = a.astype(np.int32)
        fn = f"{name}.bin"
        a.astype(a.dtype.newbyteorder("<")).tofile(self.root / fn)
        self.entries[name] = {"file": fn, "dtype": str(a.dtype), "shape": list(a.shape), **meta}

    def save(self, extra: dict):
        (self.root / "manifest.json").write_text(json.dumps({**extra, "tensors": self.entries}, indent=1))


def convert(hf_dir: Path, out: Path, outtype: str) -> None:
    cmd = [sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(hf_dir), "--outfile", str(out),
           "--outtype", outtype]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


@torch.no_grad()
def main() -> None:
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REF / "tiny")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()
    out: Path = args.out
    hf_dir = out / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    cfg = build_config()
    cfg._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    randomize(model)
    model.save_pretrained(hf_dir, safe_serialization=True)

    # Append MTP tensors to the checkpoint (HF model ignores ^mtp.* on load).
    g = torch.Generator().manual_seed(SEED + 1)
    mtp = make_mtp_tensors(cfg, g)
    st_path = hf_dir / "model.safetensors"
    sd = load_file(st_path)
    sd.update(mtp)
    save_file(sd, st_path, metadata={"format": "pt"})

    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copy(REF / f, hf_dir / f)
    tok = AutoTokenizer.from_pretrained(hf_dir)

    # ----- prompts: P0 crosses the 64-token chunk boundary twice
    text0 = ("HALO is a hardware-aware local inference engine for Qwen3.8 on AMD Strix Halo. "
             "It combines Gated DeltaNet linear attention with full attention layers, "
             "multi-token prediction, paged KV caches and a tier-aware memory planner. ") * 4
    p0 = tok(text0, add_special_tokens=False)["input_ids"][:150]
    prompts = {"p0": p0, "p1": tok("Hello, world!", add_special_tokens=False)["input_ids"], "p2": [9707]}

    G = Golden(out / "golden")
    embed = model.model.embed_tokens.weight.float().numpy()
    lm_head = model.lm_head.weight.float().numpy()
    mtp_np = {k: v.float().numpy().astype(np.float64) for k, v in mtp.items()}

    for name, ids in prompts.items():
        ids_t = torch.tensor([ids])
        o = model(ids_t, output_hidden_states=True, use_cache=True)
        logits = o.logits[0].float().numpy()  # [T, V]
        hid = o.hidden_states  # tuple(len = n_layers + 1)
        T = len(ids)
        G.add(f"{name}.tokens", np.array(ids, dtype=np.int32))
        G.add(f"{name}.argmax", logits.argmax(-1).astype(np.int32))
        srt = np.sort(logits, -1)
        G.add(f"{name}.top1_margin", (srt[:, -1] - srt[:, -2]).astype(np.float32))
        keep = sorted({0, min(T - 1, 63), min(T - 1, 64), min(T - 1, 65), T - 1})
        G.add(f"{name}.logits_rows", np.array(keep, dtype=np.int32))
        G.add(f"{name}.logits", logits[keep])
        # hidden_states[i] = input to layer i (i=0: embeddings); last = post final norm
        for li in range(len(hid) - 1):
            G.add(f"{name}.layer_in.{li}", hid[li][0].float().numpy())
        # last_hidden_state is already post output-norm (Qwen3_5TextModel.forward applies self.norm)
        h_last = model.model(ids_t).last_hidden_state[0].float().numpy()
        G.add(f"{name}.final_hidden", h_last)

        # ----- greedy decode with cache (exercises recurrent GDN path + KV append)
        pkv = o.past_key_values
        nxt = int(logits[-1].argmax())
        gen, steps_hidden, steps_top = [], [], []
        for _ in range(12):
            gen.append(nxt)
            so = model(torch.tensor([[nxt]]), past_key_values=pkv, use_cache=True, output_hidden_states=True)
            pkv = so.past_key_values
            lg = so.logits[0, -1].float().numpy()
            steps_top.append(np.argsort(-lg)[:16].astype(np.int32))
            steps_hidden.append(lg[np.argsort(-lg)[:16]])
            nxt = int(lg.argmax())
        G.add(f"{name}.decode_tokens", np.array(gen, dtype=np.int32))
        G.add(f"{name}.decode_top16_ids", np.stack(steps_top))
        G.add(f"{name}.decode_top16_logits", np.stack(steps_hidden))

        # ----- MTP reference: row i = (token ids[i+1], trunk hidden h[i]) at position i
        if T >= 2:
            hn, mlog = mtp_reference(cfg, mtp_np, embed, lm_head, np.array(ids[1:]), h_last[:-1].astype(np.float64),
                                     np.arange(T - 1))
            G.add(f"{name}.mtp_hidden", hn)
            G.add(f"{name}.mtp_argmax", mlog.argmax(-1).astype(np.int32))
            G.add(f"{name}.mtp_logits_last", mlog[-1:])

    # Tests compare argmax only where top1_margin exceeds their numeric tolerance.
    margins = np.fromfile(out / "golden" / "p0.top1_margin.bin", dtype="<f4")
    margin = float(np.median(margins))
    G.save({"model": "halo-tiny-qwen35", "config": {k: v for k, v in TINY.items()}, "seed": SEED,
            "median_top1_margin_p0": margin,
            "conventions": {
                "layer_in.i": "input hidden state to decoder layer i (i=0 is the token embedding)",
                "final_hidden": "last hidden state after output norm",
                "mtp": "row i = MTP(embed(tokens[i+1]), final_hidden[i]) at rope position i (DECISIONS D-005)",
            }})
    print(f"golden written; p0 len={len(p0)} median top1 margin={margin:.4f} "
          f"min={float(margins.min()):.5f} frac>0.05={float((margins > 0.05).mean()):.3f}")

    if not args.skip_convert:
        convert(hf_dir, out / "tiny-f32.gguf", "f32")
        convert(hf_dir, out / "tiny-q8_0.gguf", "q8_0")
        quant = LLAMA_CPP / "build" / "bin" / "llama-quantize"
        # Tensors whose row width is not a multiple of 256 fall back to other types inside
        # llama-quantize; the loader test enumerates what actually landed in each file.
        for qt in ["Q4_K_M", "Q6_K", "IQ4_XS", "Q3_K_M", "Q5_K_M", "Q4_0"]:
            dst = out / f"tiny-{qt.lower()}.gguf"
            subprocess.run([str(quant), "--allow-requantize", str(out / "tiny-f32.gguf"), str(dst), qt], check=True,
                           stdout=subprocess.DEVNULL)
            print("quantized", dst.name)


if __name__ == "__main__":
    main()
