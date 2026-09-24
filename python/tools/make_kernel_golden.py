"""Generate golden inputs/outputs for HALO's CPU reference operators (WS-D).

Every expected output is computed by the *transformers* Qwen3.5 reference functions
themselves (transformers 5.17 `models/qwen3_5/modeling_qwen3_5.py`), in float32, on the
torch-only code path. The C++ test `tests/unit/cpu_kernels/test_cpu_golden.cpp` feeds the
same inputs to `halo::cpu` and compares within documented tolerances.

Gated DeltaNet is checked in two head layouts (DECISIONS.md D-004 item 5):
  * grouped  - HF order: value head j uses key head j // (n_v / n_k) (repeat_interleave);
  * tiled    - GGUF order: every per-v-head tensor permuted with llama.cpp's
               `_reorder_v_heads` (conversion/qwen.py, commit bd4f514db), so value head j
               uses key head j % n_k. The C++ tiled mode must reproduce the permuted HF
               outputs, which ties it to HF math through the exact converter permutation.

Output (default ~/halo-ref/kernel_golden): manifest.json + raw little-endian *.bin.

    /root/halo-py/.venv/bin/python python/tools/make_kernel_golden.py [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

SEED = 20260923
OUT = Path.home() / "halo-ref" / "kernel_golden"


# ------------------------------------------------------------------------- torch-path guard
def torch_path(fn: Any) -> Any:
    """The undecorated torch implementation behind a hub-kernel decorator."""
    return inspect.unwrap(fn)


def assert_torch_only() -> dict[str, Any]:
    """Hub kernels / FLA / causal-conv1d would replace the torch reference; refuse if present."""
    optional = ["kernels", "fla", "causal_conv1d"]
    present = [p for p in optional if importlib.util.find_spec(p) is not None]
    if present:
        raise SystemExit(f"refusing to generate golden data: optional kernel packages installed: {present}")
    return {"optional_kernel_packages_absent": optional, "functions_called_via": "inspect.unwrap (torch path)"}


# --------------------------------------------------------------------------- golden writer
class Golden:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        for old in root.glob("*.bin"):
            old.unlink()
        self.tensors: dict[str, dict[str, Any]] = {}
        self.cases: dict[str, dict[str, Any]] = {}

    def add(self, name: str, arr: torch.Tensor | np.ndarray) -> None:
        a = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
        a = np.ascontiguousarray(a)
        if a.dtype == np.int64:
            a = a.astype(np.int32)
        if a.dtype not in (np.float32, np.int32):
            raise TypeError(f"{name}: unexpected dtype {a.dtype}")
        fn = f"{name}.bin"
        a.astype(a.dtype.newbyteorder("<")).tofile(self.root / fn)
        self.tensors[name] = {"file": fn, "dtype": str(a.dtype), "shape": list(a.shape)}

    def case(self, name: str, **meta: Any) -> None:
        self.cases[name] = meta

    def save(self, extra: dict[str, Any]) -> None:
        doc = {**extra, "cases": self.cases, "tensors": self.tensors}
        (self.root / "manifest.json").write_text(json.dumps(doc, indent=1))


def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32) * scale


# ------------------------------------------------------------ llama.cpp head permutation
def reorder_v_heads(tensor: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Verbatim copy of llama.cpp conversion/qwen.py `_reorder_v_heads` (commit bd4f514db, MIT).

    Reorder V heads from grouped (by K head) to tiled order along the given dimension."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


# ------------------------------------------------------------------------------- cases
def gen_rms_norm(G: Golden) -> None:
    for name, (t, d) in {"rms_norm.a": (5, 256), "rms_norm.b": (3, 5120)}.items():
        x = randn(t, d, scale=3.0)
        w_hf = randn(d, scale=0.2)  # HF stores zero-centred weights
        mod = m.Qwen3_5RMSNorm(d, eps=1e-6)
        mod.weight.data.copy_(w_hf)
        y = mod(x)
        G.add(f"{name}.x", x)
        G.add(f"{name}.w", 1.0 + w_hf)  # GGUF stores w + 1 (D-004); C++ applies plain x_hat * w
        G.add(f"{name}.y", y)
        G.case(name, rows=t, dim=d, eps=1e-6)


def gen_gated_norm(G: Golden) -> None:
    rows, d = 4 * 6, 128
    x, z, w = randn(rows, d, scale=2.0), randn(rows, d, scale=2.0), 1.0 + randn(d, scale=0.2)
    mod = m.Qwen3_5RMSNormGated(d, eps=1e-6)
    mod.weight.data.copy_(w)
    G.add("gated_norm.x", x)
    G.add("gated_norm.z", z)
    G.add("gated_norm.w", w)  # ssm_norm is stored as-is and applied as plain x_hat * w
    G.add("gated_norm.y", mod(x, z))
    G.case("gated_norm", rows=rows, dim=d, eps=1e-6)


def gen_l2norm(G: Golden) -> None:
    t, h, d = 5, 4, 128
    x = randn(t, h, d, scale=2.0)
    x[0, 0] = 0.0  # all-zero head: eps keeps it finite
    G.add("l2norm.x", x.reshape(t, h * d))
    G.add("l2norm.y", torch_path(m.l2norm)(x, dim=-1, eps=1e-6).reshape(t, h * d))
    G.case("l2norm", tokens=t, heads=h, head_dim=d, eps=1e-6)


def gen_rope(G: Golden) -> None:
    head_dim, n_heads, theta = 256, 3, 10_000_000.0
    cfg = Qwen3_5TextConfig(
        hidden_size=768, num_attention_heads=n_heads, num_key_value_heads=1, head_dim=head_dim,
        rope_parameters=dict(rope_type="default", rope_theta=theta, partial_rotary_factor=0.25,
                             mrope_section=[11, 11, 10], mrope_interleaved=True))
    rot = m.Qwen3_5TextRotaryEmbedding(cfg)
    positions = torch.tensor([0, 1, 2, 7, 100, 4095, 32767, 65536, 131071, 262143], dtype=torch.int64)
    t = positions.numel()
    # Text-only input: all three M-RoPE position components are equal (D-004).
    pos3 = positions[None, None, :].expand(3, 1, t)
    x = randn(1, n_heads, t, head_dim, scale=2.0)
    cos, sin = rot(x, pos3)
    y, _ = m.apply_rotary_pos_emb(x, x.clone(), cos, sin)
    tok_major = lambda a: a[0].transpose(0, 1).reshape(t, n_heads * head_dim)  # noqa: E731
    G.add("rope.x", tok_major(x))
    G.add("rope.y", tok_major(y))
    G.add("rope.positions", positions.to(torch.int32))
    G.add("rope.inv_freq", rot.inv_freq.float())
    G.case("rope", tokens=t, heads=n_heads, head_dim=head_dim, rot_dims=int(cos.shape[-1]), theta=theta)


def gen_conv(G: Golden) -> None:
    c, k = 80, 4  # 80 channels: one full and one partial 64-channel block in the C++ kernel
    w = randn(c, k, scale=0.5)
    # Prefill from an empty state (causal_conv1d_fn), channels-first [B, C, T] in HF.
    t1 = 9
    x1 = randn(t1, c)
    y1 = torch_path(m.causal_conv1d_fn)(x1.T[None].contiguous(), w, None, activation="silu")[0].T
    G.add("conv.w", w)
    G.add("conv.prefill.x", x1)
    G.add("conv.prefill.y", y1)
    # Same tokens through causal_conv1d_update from a zero state: HF's own resulting state.
    zstate = torch.zeros(1, c, k - 1)
    y1u = torch_path(m.causal_conv1d_update)(x1.T[None].contiguous(), zstate, w, None, "silu")[0].T
    torch.testing.assert_close(y1u, y1, rtol=1e-6, atol=1e-6)
    G.add("conv.prefill.state_out", zstate[0].T.contiguous())  # [K-1, C]
    # Continuation from a non-zero state (causal_conv1d_update, state length K-1).
    t2 = 3
    state = randn(1, c, k - 1)
    x2 = randn(t2, c)
    state_in = state[0].T.clone()  # HF [C, K-1] -> HALO [K-1, C], oldest first
    y2 = torch_path(m.causal_conv1d_update)(x2.T[None].contiguous(), state, w, None, "silu")[0].T
    G.add("conv.update.x", x2)
    G.add("conv.update.state_in", state_in)
    G.add("conv.update.y", y2)
    G.add("conv.update.state_out", state[0].T.contiguous())
    G.case("conv", channels=c, kernel=k, prefill_tokens=t1, update_tokens=t2)


def gdn_inputs(t: int, n_k: int, n_v: int, dk: int, dv: int) -> dict[str, torch.Tensor]:
    # Model-shaped inputs: raw q/k (normalized in-kernel), beta = sigmoid(b),
    # g = -exp(A_log) * softplus(a + dt_bias) with A spanning weak to very strong decay.
    a_log = torch.log(torch.linspace(0.01, 16.0, n_v))
    dt_bias = randn(n_v, scale=0.5)
    a = randn(t, n_v)
    return dict(
        q=randn(t, n_k, dk), k=randn(t, n_k, dk), v=randn(t, n_v, dv),
        beta=torch.sigmoid(randn(t, n_v, scale=2.0)),
        g=-a_log.exp() * F.softplus(a + dt_bias),
    )


def gen_gdn(G: Golden) -> None:
    chunk = torch_path(m.torch_chunk_gated_delta_rule)
    recur = torch_path(m.torch_recurrent_gated_delta_rule)
    cases = {
        # name: (T, n_k, n_v, d_k, d_v, with initial state)
        "gdn.r3": (150, 2, 6, 32, 32, True),      # ratio 3, crosses two chunk boundaries
        "gdn.r3_rect": (70, 2, 6, 16, 24, False),  # d_k != d_v, zero initial state
        "gdn.r1": (65, 3, 3, 32, 32, True),        # ratio 1: tiled == grouped
        "gdn.real": (70, 16, 48, 128, 128, True),  # Qwen3.8-27B layer dims (D-003)
    }
    for name, (t, n_k, n_v, dk, dv, has_init) in cases.items():
        x = gdn_inputs(t, n_k, n_v, dk, dv)
        ratio = n_v // n_k
        s0 = randn(n_v, dk, dv, scale=0.5) if has_init else torch.zeros(n_v, dk, dv)
        q_rep = x["q"].repeat_interleave(ratio, dim=1)[None]
        k_rep = x["k"].repeat_interleave(ratio, dim=1)[None]
        args = dict(query=q_rep, key=k_rep, value=x["v"][None], g=x["g"][None], beta=x["beta"][None],
                    initial_state=s0[None].clone() if has_init else None, output_final_state=True,
                    use_qk_l2norm_in_kernel=True)
        o_ch, s_ch = chunk(**args, chunk_size=64)
        o_rc, s_rc = recur(**args)
        o_ch, s_ch, o_rc, s_rc = o_ch[0], s_ch[0], o_rc[0], s_rc[0]
        flat = lambda a: a.reshape(t, -1)  # noqa: E731
        common = {"q": flat(x["q"]), "k": flat(x["k"])}
        grouped = {"v": flat(x["v"]), "g": x["g"], "beta": x["beta"], "state_in": s0,
                   "out_chunk": flat(o_ch), "state_chunk": s_ch, "out_recur": flat(o_rc), "state_recur": s_rc}
        # Tiled (GGUF) layout: permute every per-v-head tensor like the converter does.
        tiled = {
            "v": reorder_v_heads(flat(x["v"]), -1, n_k, ratio, dv),
            "g": reorder_v_heads(x["g"], -1, n_k, ratio, 1),
            "beta": reorder_v_heads(x["beta"], -1, n_k, ratio, 1),
            "state_in": reorder_v_heads(s0.reshape(n_v, -1), 0, n_k, ratio, 1).reshape(n_v, dk, dv),
            "out_chunk": reorder_v_heads(flat(o_ch), -1, n_k, ratio, dv),
            "state_chunk": reorder_v_heads(s_ch.reshape(n_v, -1), 0, n_k, ratio, 1).reshape(n_v, dk, dv),
            "out_recur": reorder_v_heads(flat(o_rc), -1, n_k, ratio, dv),
            "state_recur": reorder_v_heads(s_rc.reshape(n_v, -1), 0, n_k, ratio, 1).reshape(n_v, dk, dv),
        }
        for key, val in common.items():
            G.add(f"{name}.{key}", val)
        for layout, tensors in (("grouped", grouped), ("tiled", tiled)):
            for key, val in tensors.items():
                G.add(f"{name}.{layout}.{key}", val)
        G.case(name, tokens=t, n_k_heads=n_k, n_v_heads=n_v, d_k=dk, d_v=dv, chunk_size=64,
               initial_state=has_init, qk_l2norm=True)


def gen_attention(G: Golden) -> None:
    cases = {
        # name: (n_head, n_kv, head_dim, T, q_offset)
        "attn.small": (6, 2, 64, 5, 35),
        "attn.real": (24, 4, 256, 3, 14),  # Qwen3.8-27B full-attention layer (D-004)
    }
    for name, (nh, nkv, hd, t, off) in cases.items():
        s = off + t
        q, k, v = randn(1, nh, t, hd), randn(1, nkv, s, hd), randn(1, nkv, s, hd)
        mask = torch.zeros(1, 1, t, s)
        for i in range(t):
            mask[0, 0, i, off + i + 1:] = float("-inf")
        mod = SimpleNamespace(num_key_value_groups=nh // nkv, training=False)
        scale = hd ** -0.5
        out, _ = m.eager_attention_forward(mod, q, k, v, mask, scaling=scale)
        gate = randn(t, nh * hd, scale=2.0)
        out_tm = out[0].reshape(t, nh * hd)  # eager returns [B, T, H, D]
        G.add(f"{name}.q", q[0].transpose(0, 1).reshape(t, nh * hd))
        G.add(f"{name}.k", k[0].transpose(0, 1).reshape(s, nkv * hd))
        G.add(f"{name}.v", v[0].transpose(0, 1).reshape(s, nkv * hd))
        G.add(f"{name}.out", out_tm)
        G.add(f"{name}.gate", gate)
        G.add(f"{name}.gated", out_tm * torch.sigmoid(gate))  # Qwen3_5Attention output gate
        G.case(name, n_head=nh, n_kv_head=nkv, head_dim=hd, tokens=t, history=s, q_offset=off, scale=scale)


def gen_elementwise(G: Golden) -> None:
    rows, d = 4, 1000
    x = torch.linspace(-30.0, 30.0, rows * d).reshape(rows, d)
    special = torch.tensor([19.99, 20.0, 20.001, 20.01, 25.0, -100.0, 100.0, 0.0, -0.0, 1e-8, -88.0, 88.0])
    x[0, : special.numel()] = special
    b = randn(rows, d, scale=3.0)
    G.add("elem.x", x)
    G.add("elem.b", b)
    G.add("elem.silu", F.silu(x))
    G.add("elem.sigmoid", torch.sigmoid(x))
    G.add("elem.softplus", F.softplus(x))  # beta=1, threshold=20: x > 20 -> x
    G.add("elem.swiglu", F.silu(x) * b)  # Qwen3_5MLP: act(gate) * up
    G.add("elem.add", x + b)
    G.add("elem.mul", x * b)
    sm_in = randn(rows, d, scale=5.0)
    sm_in[1] += 1000.0  # stability: large common offset
    sm_in[2, :10] -= 1e4
    G.add("elem.softmax_x", sm_in)
    G.add("elem.softmax", torch.softmax(sm_in, dim=-1))
    G.case("elem", rows=rows, dim=d)


def gen_matmul(G: Golden) -> None:
    t, k, n = 3, 300, 37
    x, w = randn(t, k), randn(n, k, scale=0.1)
    y = x @ w.T
    G.add("matmul.x", x)
    G.add("matmul.w", w)
    G.add("matmul.y", y)
    logits = randn(5000, scale=4.0)
    vals, idx = torch.topk(logits, 10)
    G.add("topk.logits", logits)
    G.add("topk.values", vals)
    G.add("topk.indices", idx.to(torch.int32))
    G.case("matmul", tokens=t, k=k, n=n)
    G.case("topk", n=int(logits.numel()), k=10, argmax=int(torch.argmax(logits)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    guard = assert_torch_only()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    G = Golden(args.out)
    with torch.no_grad():
        gen_rms_norm(G)
        gen_gated_norm(G)
        gen_l2norm(G)
        gen_rope(G)
        gen_conv(G)
        gen_gdn(G)
        gen_attention(G)
        gen_elementwise(G)
        gen_matmul(G)
    G.save({
        "generator": "python/tools/make_kernel_golden.py",
        "seed": SEED,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "reference": guard,
        "layouts": {"token_major": "[T, heads * head_dim]", "gdn_state": "[n_v, d_k, d_v], d_v fastest",
                    "conv_state": "[K-1, C], oldest input first"},
    })
    n_bytes = sum((args.out / e["file"]).stat().st_size for e in G.tensors.values())
    print(f"wrote {len(G.tensors)} tensors / {len(G.cases)} cases ({n_bytes / 2**20:.1f} MiB) to {args.out}")


if __name__ == "__main__":
    main()
