"""Download only the GGUF header (KV + tensor infos) of remote files via HTTP Range.

Saves <name>.header.gguf (raw header bytes, no tensor data) and <name>.summary.json.
"""
import json, struct, sys, urllib.request

OUT = __import__("os").path.expanduser("~/halo-ref")
FILES = {
    "ggml-org-q4km": "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf",
    "ggml-org-mtp-q4_0": "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/mtp-Qwen3.8-27B-Q4_0.gguf",
    "unsloth-ud-q4kxl": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf",
}
GGML_TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",
              13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
              22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16"}

class Remote:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, bytearray(), 0
    def ensure(self, n):
        while len(self.buf) < self.pos + n:
            start = len(self.buf)
            end = start + max(8 << 20, self.pos + n - start) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "halo-dev"})
            with urllib.request.urlopen(req, timeout=300) as r:
                self.buf += r.read()
    def read(self, n):
        self.ensure(n); b = bytes(self.buf[self.pos:self.pos + n]); self.pos += n; return b
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def s(self): return self.read(self.u64()).decode("utf-8", "replace")

SCALAR = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}

def value(r, t):
    if t in SCALAR:
        fmt = SCALAR[t]; return struct.unpack(fmt, r.read(struct.calcsize(fmt)))[0]
    if t == 8: return r.s()
    if t == 9:
        et, n = r.u32(), r.u64()
        return [value(r, et) for _ in range(n)]
    raise ValueError(f"bad type {t}")

def main():
    for key, url in FILES.items():
        r = Remote(url)
        assert r.read(4) == b"GGUF"
        ver, n_t, n_kv = r.u32(), r.u64(), r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.s(); t = r.u32(); kv[k] = value(r, t)
        tensors = []
        for _ in range(n_t):
            name = r.s(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; tt = r.u32(); off = r.u64()
            tensors.append({"name": name, "dims": dims, "type": GGML_TYPES.get(tt, tt), "offset": off})
        header_len = r.pos
        open(f"{OUT}/{key}.header.gguf", "wb").write(bytes(r.buf[:header_len]))
        summ = {k: (v if not isinstance(v, list) or len(v) <= 70 else f"<array len={len(v)} first={v[:5]}>") for k, v in kv.items()}
        json.dump({"version": ver, "n_tensors": n_t, "header_len": header_len, "kv": summ, "tensors": tensors},
                  open(f"{OUT}/{key}.summary.json", "w"), indent=1, default=str)
        types = {}
        for t in tensors: types[t["type"]] = types.get(t["type"], 0) + 1
        print(key, "ver", ver, "tensors", n_t, "kv", n_kv, "header", header_len, types)

main()
