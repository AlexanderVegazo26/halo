r"""Tokenizer reference data for HALO (WS-B): Unicode tables and golden token ids.

HF ``tokenizers`` is the reference implementation (D-008, TRD §10). Two subcommands:

``tables``  Generate ``src/tokenizer/unicode_data.inc``. Rather than trusting one Unicode
            version, every table is *probed from HF itself* so the C++ port matches the
            reference exactly:
            - pre-tokenizer character classes (``[\p{L}\p{M}]``, ``\p{N}``, ``\s``) come from
              running HF's own Split regex (Oniguruma) on probe strings for every codepoint;
            - ``(?i)`` contraction case-fold sets come from probing the same regex;
            - NFC data (decompositions, canonical combining classes, primary composites)
              starts from Python ``unicodedata`` and is then reconciled with HF's NFC/NFD
              normalizer, which uses older Unicode data (e.g. it does not know U+11938's
              Unicode 13 decomposition).
            The generator re-implements NFC over the emitted tables and aborts if it
            disagrees with HF anywhere it checks.

``golden``  Write golden cases to ``$HOME/halo-ref/tokenizer_golden/`` using the raw
            ``tokenizers.Tokenizer`` loaded from the shipped ``tokenizer.json``, cross-checked
            against transformers' ``Qwen3_5Tokenizer`` (must agree on every case), plus the
            unsloth GGUF tokenizer arrays so the C++ raw-vector constructor can be tested
            against real GGUF data.

            Why not ``AutoTokenizer``: the checkpoint's tokenizer_config.json names
            ``Qwen2Tokenizer``, and transformers 5.17 rebuilds that class's pre-tokenizer with
            the *Qwen2* regex (``\p{L}+``, no ``\p{M}``), ignoring tokenizer.json. That
            disagrees with tokenizer.json, with ``Qwen3_5Tokenizer`` and with llama.cpp's
            ``qwen35`` pre-type (e.g. on Devanagari). The number of cases where AutoTokenizer
            disagrees is recorded in manifest.json.

Usage (inside WSL)::

    /root/halo-py/.venv/bin/python python/tools/make_tokenizer_golden.py tables
    /root/halo-py/.venv/bin/python python/tools/make_tokenizer_golden.py golden
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata as ud
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tokenizers
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers

REPO = Path(__file__).resolve().parents[2]
REF = Path.home() / "halo-ref"
HF_DIR = REF / "tiny" / "hf"
TOKENIZER_JSON = HF_DIR / "tokenizer.json"

SURROGATES = range(0xD800, 0xE000)
HANGUL_S_BASE, HANGUL_S_COUNT = 0xAC00, 11172

CLASS_OTHER, CLASS_LM, CLASS_N, CLASS_WS = 0, 1, 2, 3


def all_codepoints() -> list[int]:
    """Every Unicode scalar value (surrogates excluded)."""
    return [c for c in range(0x110000) if c not in SURROGATES]


def load_split_regex() -> str:
    """Return the Split pattern from the reference tokenizer.json (never retyped by hand)."""
    cfg = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))
    seq = cfg["pre_tokenizer"]["pretokenizers"]
    split = seq[0]
    assert split["type"] == "Split" and split["behavior"] == "Isolated", split
    return str(split["pattern"]["Regex"])


def make_splitter() -> pre_tokenizers.Split:
    return pre_tokenizers.Split(pattern=Regex(load_split_regex()), behavior="isolated")


def pieces(pt: pre_tokenizers.Split, s: str) -> list[str]:
    return [p for p, _ in pt.pre_tokenize_str(s)]


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def classify(pt: pre_tokenizers.Split, c: int) -> int:
    """Classify one codepoint by how HF's regex splits three probe strings around it."""
    ch = chr(c)
    p = pieces(pt, "x" + ch + "1")
    q = pieces(pt, "!" + ch + "!")
    r = pieces(pt, "a" + ch + "b")
    if p == ["x" + ch, "1"]:
        return CLASS_LM
    if p == ["x", ch, "1"] and q == ["!", ch, "!"] and r == ["a", ch, "b"]:
        return CLASS_N
    if p == ["x", ch, "1"] and q == ["!" + ch + "!"]:
        return CLASS_OTHER
    if q == ["!", ch, "!"] and r == ["a", ch + "b"]:
        return CLASS_WS  # whitespace other than U+0020 / CR / LF
    if c == 0x20 and q == ["!", " !"] and r == ["a", " b"]:
        return CLASS_WS
    if c in (0x0A, 0x0D) and q == ["!" + ch, "!"] and r == ["a", ch, "b"]:
        return CLASS_WS
    raise SystemExit(f"unclassifiable codepoint U+{c:04X}: p={p} q={q} r={r}")


def to_ranges(cps: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for c in sorted(cps):
        if out and out[-1][1] + 1 == c:
            out[-1] = (out[-1][0], c)
        else:
            out.append((c, c))
    return out


def fold_sets(pt: pre_tokenizers.Split, cls: dict[int, int]) -> dict[str, list[int]]:
    """Codepoints that the ``(?i:'s|'t|'re|'ve|'m|'ll|'d)`` alternative accepts, per slot.

    The single-letter probe uses a suffix that the non-contraction split would absorb, so a
    match is observable for every class; CR/LF are skipped (no suffix distinguishes them and
    they cannot case-fold to a letter).
    """
    single, re_ve_first, e_second, l_first, l_second = [], [], [], [], []
    for c in all_codepoints():
        if c in (0x0A, 0x0D):
            continue
        ch = chr(c)
        suffix = "!" if cls[c] == CLASS_OTHER else "a"
        if pieces(pt, "'" + ch + suffix) == ["'" + ch, suffix]:
            single.append(c)
    for c in (c for c in all_codepoints() if cls[c] == CLASS_LM):
        ch = chr(c)
        if pieces(pt, "'" + ch + "ea")[0] == "'" + ch + "e":
            re_ve_first.append(c)
        if pieces(pt, "'r" + ch + "a")[0] == "'r" + ch:
            e_second.append(c)
        if pieces(pt, "'" + ch + "la")[0] == "'" + ch + "l":
            l_first.append(c)
        if pieces(pt, "'l" + ch + "a")[0] == "'l" + ch:
            l_second.append(c)
    return {
        "kFoldSingle": single,
        "kFoldReVeFirst": re_ve_first,
        "kFoldESecond": e_second,
        "kFoldLFirst": l_first,
        "kFoldLSecond": l_second,
    }


@dataclass
class NfcTables:
    ccc: dict[int, int]
    decomp: dict[int, list[int]]
    comp: dict[tuple[int, int], int]
    check: set[int]


def is_hangul_syllable(c: int) -> bool:
    return HANGUL_S_BASE <= c < HANGUL_S_BASE + HANGUL_S_COUNT


def build_nfc_tables() -> NfcTables:
    hf_nfd = normalizers.NFD()
    hf_nfc = normalizers.NFC()
    cps = all_codepoints()

    # Canonical combining class: Python's value, but 0 where HF's older data does not know
    # the mark (detected by whether HF reorders it against a reference mark).
    ccc: dict[int, int] = {}
    for c in cps:
        k = ud.combining(chr(c))
        if k == 0:
            continue  # a class HF knows but Python lacks is caught by self_check_nfc
        # U+0334 has ccc 1 and U+0345 ccc 240: HF reorders c against them only if it
        # knows c's class.
        s = "x" + chr(c) + "̴" if k > 1 else "xͅ" + chr(c)
        if hf_nfd.normalize_str(s) == ud.normalize("NFD", s):
            ccc[c] = k

    # Full canonical decomposition exactly as HF computes it (Hangul is algorithmic in C++).
    decomp: dict[int, list[int]] = {}
    for c in cps:
        if is_hangul_syllable(c):
            continue
        d = hf_nfd.normalize_str(chr(c))
        if d != chr(c):
            decomp[c] = [ord(x) for x in d]

    # Primary composites: single-level two-codepoint canonical mappings that HF recomposes.
    comp: dict[tuple[int, int], int] = {}
    for c in cps:
        if is_hangul_syllable(c):
            continue
        m = ud.decomposition(chr(c))
        if not m or m.startswith("<"):
            continue
        parts = [int(x, 16) for x in m.split()]
        if len(parts) != 2:
            continue
        if hf_nfc.normalize_str(chr(parts[0]) + chr(parts[1])) == chr(c):
            comp[(parts[0], parts[1])] = c

    # Codepoints that force the slow NFC path: NFC_QC != Yes (a superset: anything HF's NFC
    # changes on its own, any composition second, any nonzero ccc).
    check: set[int] = set(ccc)
    for c in cps:
        if hf_nfc.normalize_str(chr(c)) != chr(c):
            check.add(c)
    for _, second in comp:
        check.add(second)
    check.update(range(0x1161, 0x1176))  # Hangul V jamo (compose with L / LV)
    check.update(range(0x11A8, 0x11C3))  # Hangul T jamo
    return NfcTables(ccc=ccc, decomp=decomp, comp=comp, check=check)


def nfc_reference(s: str, t: NfcTables) -> str:
    """Pure-Python NFC over the generated tables; mirrors src/tokenizer/unicode.cpp."""
    s_base, l_base, v_base, t_base = 0xAC00, 0x1100, 0x1161, 0x11A7
    l_count, v_count, t_count = 19, 21, 28
    n_count = v_count * t_count
    out: list[int] = []
    for ch in s:
        c = ord(ch)
        if is_hangul_syllable(c):
            si = c - s_base
            out.append(l_base + si // n_count)
            out.append(v_base + (si % n_count) // t_count)
            if si % t_count:
                out.append(t_base + si % t_count)
        else:
            out.extend(t.decomp.get(c, [c]))
    # canonical ordering (stable)
    i = 0
    while i < len(out):
        if t.ccc.get(out[i], 0) == 0:
            i += 1
            continue
        j = i
        while j < len(out) and t.ccc.get(out[j], 0) != 0:
            j += 1
        out[i:j] = sorted(out[i:j], key=lambda x: t.ccc.get(x, 0))
        i = j
    if not out:
        return ""

    def compose(a: int, b: int) -> int | None:
        if l_base <= a < l_base + l_count and v_base <= b < v_base + v_count:
            return s_base + ((a - l_base) * v_count + (b - v_base)) * t_count
        if is_hangul_syllable(a) and (a - s_base) % t_count == 0 and t_base < b < t_base + t_count:
            return a + (b - t_base)
        return t.comp.get((a, b))

    res = [out[0]]
    starter_pos = 0
    last_class = t.ccc.get(out[0], 0)
    if last_class != 0:
        last_class = 256
    for c in out[1:]:
        cc = t.ccc.get(c, 0)
        comp = compose(res[starter_pos], c)
        if comp is not None and (last_class < cc or last_class == 0):
            res[starter_pos] = comp
            continue
        if cc == 0:
            starter_pos = len(res)
        last_class = cc
        res.append(c)
    return "".join(chr(x) for x in res)


def self_check_nfc(t: NfcTables) -> None:
    hf_nfc = normalizers.NFC()
    rng = random.Random(1234)
    marks = sorted(t.ccc) + [0x0301, 0x0323, 0x0334, 0x0345, 0x3099, 0x1161, 0x11A8]
    samples: list[str] = []
    for c in all_codepoints():
        samples.append(chr(c))
        samples.append("xͅ" + chr(c) + "̴")
    for c in t.decomp:
        samples.append("".join(chr(x) for x in t.decomp[c]))
        samples.append("x́" + chr(c) + "̴")
    for c in t.ccc:
        samples.append("x̴" + chr(c) + "́ͅ")
        samples.append("á" + chr(c) + "̣")
    bases = [0x61, 0x41, 0x6F, 0x1100, 0xAC00, 0xAC01, 0x0B47, 0x0915, 0x3046, 0x05D0]
    for _ in range(20000):
        n = rng.randint(1, 6)
        samples.append(chr(rng.choice(bases)) + "".join(chr(rng.choice(marks)) for _ in range(n)))
    bad = [s for s in samples if nfc_reference(s, t) != hf_nfc.normalize_str(s)]
    if bad:
        raise SystemExit(f"NFC table self-check failed on {len(bad)} samples, first: {bad[0]!r}")
    print(f"NFC self-check OK on {len(samples)} samples", file=sys.stderr)


def fmt_ranges(name: str, ranges: list[tuple[int, int]]) -> str:
    body = ",".join(f"{{0x{a:X},0x{b:X}}}" for a, b in ranges)
    return f"inline constexpr CpRange {name}[] = {{{body}}};\n"


def fmt_u32(name: str, vals: list[int]) -> str:
    body = ",".join(f"0x{v:X}" for v in vals)
    return f"inline constexpr std::uint32_t {name}[] = {{{body}}};\n"


def wrap(text: str, width: int = 100) -> str:
    """Break long generated lines at commas so the .inc stays diff-able."""
    out_lines: list[str] = []
    for line in text.splitlines():
        while len(line) > width:
            cut = line.rfind(",", 0, width)
            if cut <= 0:
                break
            out_lines.append(line[: cut + 1])
            line = "    " + line[cut + 1 :]
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def cmd_tables(args: argparse.Namespace) -> None:
    pt = make_splitter()
    classes: dict[int, list[int]] = {CLASS_LM: [], CLASS_N: [], CLASS_WS: []}
    cls: dict[int, int] = {}
    py_mismatch = 0
    for c in all_codepoints():
        k = classify(pt, c)
        cls[c] = k
        if k != CLASS_OTHER:
            classes[k].append(c)
        cat = ud.category(chr(c))
        py = CLASS_LM if cat[0] in "LM" else CLASS_N if cat[0] == "N" else None
        if k in (CLASS_LM, CLASS_N) and py != k or py is not None and py != k:
            py_mismatch += 1
    print(
        f"classes: LM={len(classes[CLASS_LM])} N={len(classes[CLASS_N])} WS={len(classes[CLASS_WS])};"
        f" {py_mismatch} codepoints classified differently by Python unicodedata {ud.unidata_version}",
        file=sys.stderr,
    )
    folds = fold_sets(pt, cls)
    print("fold sets:", {k: [hex(v) for v in vs] for k, vs in folds.items()}, file=sys.stderr)
    nfc = build_nfc_tables()
    self_check_nfc(nfc)

    decomp_entries: list[str] = []
    decomp_data: list[int] = []
    for c in sorted(nfc.decomp):
        d = nfc.decomp[c]
        if len(d) > 255:
            raise SystemExit("decomposition too long")
        decomp_entries.append(f"{{0x{c:X},{len(decomp_data)},{len(d)}}}")
        decomp_data.extend(d)
    ccc_ranges: list[tuple[int, int, int]] = []
    for c in sorted(nfc.ccc):
        k = nfc.ccc[c]
        if ccc_ranges and ccc_ranges[-1][1] + 1 == c and ccc_ranges[-1][2] == k:
            ccc_ranges[-1] = (ccc_ranges[-1][0], c, k)
        else:
            ccc_ranges.append((c, c, k))
    comp_entries = [f"{{0x{a:X},0x{b:X},0x{v:X}}}" for (a, b), v in sorted(nfc.comp.items())]

    header = (
        "// GENERATED by python/tools/make_tokenizer_golden.py tables -- do not edit by hand.\n"
        f"// Reference: HF tokenizers {tokenizers.__version__} (Split regex classes, (?i) folds and\n"
        f"// NFC behaviour probed per codepoint); Python unicodedata {ud.unidata_version} supplies\n"
        "// canonical combining classes and single-level mappings, reconciled against HF.\n"
        "// Included only by src/tokenizer/unicode.cpp.\n"
    )
    parts = [
        header,
        fmt_ranges("kLetterMark", to_ranges(classes[CLASS_LM])),
        fmt_ranges("kNumber", to_ranges(classes[CLASS_N])),
        fmt_ranges("kWhitespace", to_ranges(classes[CLASS_WS])),
    ]
    parts += [fmt_u32(k, v) for k, v in folds.items()]
    parts.append(
        "inline constexpr CccRange kCcc[] = {"
        + ",".join(f"{{0x{a:X},0x{b:X},{k}}}" for a, b, k in ccc_ranges)
        + "};\n"
    )
    parts.append("inline constexpr DecompEntry kDecomp[] = {" + ",".join(decomp_entries) + "};\n")
    parts.append(fmt_u32("kDecompData", decomp_data))
    parts.append("inline constexpr CompEntry kComp[] = {" + ",".join(comp_entries) + "};\n")
    parts.append(fmt_ranges("kNfcCheck", to_ranges(sorted(nfc.check))))
    out = Path(args.out)
    out.write_text(wrap("".join(parts)), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


# --------------------------------------------------------------------------------------
# golden
# --------------------------------------------------------------------------------------

CORPUS: list[tuple[str, str]] = [
    ("empty", ""),
    ("ascii_prose", "The quick brown fox jumps over the lazy dog. It's 2026, and we're testing!"),
    ("ascii_sentences", "Hello world.  Two  spaces;   three.\tTab. End with space "),
    ("code_python", 'def f(x):\n    """Doc."""\n    if x > 0:\n\treturn x ** 2  # sq\n    return -x\n'),
    ("code_crlf", "int main() {\r\n    return 0;\r\n}\r\n\r\n// done\r\n"),
    ("code_cpp", "template <typename T>\nauto add(T a, T b) -> T { return a + b; }\n\n\n  \n"),
    ("json_like", '{"key": [1, 2.5, -3e10], "nested": {"a": null, "b": true}}'),
    ("numbers", "123 4567 89012345 3.14159 1,000,000 0x1F 1e-9 ٣٤٥ ½ Ⅻ ①"),
    ("contractions_lower", "don't can't I'm you're we've they'll he'd it's"),
    ("contractions_upper", "DON'T CAN'T I'M YOU'RE WE'VE THEY'LL HE'D IT'S"),
    ("contractions_mixed", "It'S We'Re They'lL she'D 'sa 'ts 'res 'LLx"),
    ("contraction_fold_long_s", "it'ſ can'ſ 'ſx We'ſ"),
    ("contraction_fold_other", "'K 'ß 'İ 'ﬆ 'ǆ 'ǅ"),
    ("apostrophes", "'' ''' 'a' ' s 'x' '1 '\n' 'é"),
    ("cjk", "你好，世界！这是一个测试。日本語のテキスト、カタカナ。한국어 텍스트입니다."),
    ("emoji_zwj", "Family: 👨‍👩‍👧‍👦 flag: 🇺🇸🇯🇵 skin: 👍🏽 heart: ❤️ smile😀x"),
    ("combining_decomposed", "café naïve Å ẹ́ ǭ 각"),
    ("combining_reorder", "ạ́ ẍ̴ q̣̇́"),
    ("combining_precomposed", "café naïve Å ệ ǭ 각"),
    ("nfc_singletons", "Å Ω K ̀́ ʹ ; ·"),
    ("arabic", "مرحبا بالعالم. هذا اختبار ١٢٣."),
    ("hebrew", "שלום עולם! זה מבחן עם ניקוד: שָׁלוֹם"),
    ("devanagari", "नमस्ते दुनिया क़ ज़"),
    ("thai", "สวัสดีชาวโลก"),
    ("whitespace_runs", "a  b   c    d\t\te\n\nf\n \n g \r\n\r\n h   "),
    ("whitespace_only", "   \n\t  \r\n  "),
    ("whitespace_unicode", "a b　c d e\u0085f​g᠎h i j"),
    ("trailing_newlines", "line\n\n\n"),
    ("leading_space_word", " hello  world   again"),
    ("punct_runs", "!!! ??? ... --- === +++ *** ### @@@ $$$ %%% ^^^ &&& ~~~"),
    ("punct_newlines", "end.\n\nNext!\r\n\r\nWait?\n"),
    ("special_inline", "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\nhmm\n</think>\n\nHi!<|im_end|>"),
    ("special_adjacent", "<|im_end|><|im_start|><think></think><tool_call></tool_call>"),
    ("special_partial", "<|im_start <|im_end| <think <|endoftext|x <tool_call"),
    ("special_nonspecial_added", "<|fim_prefix|>def<|fim_suffix|>x<|fim_middle|><tool_response>ok</tool_response>"),
    ("special_whitespace", "  <|im_start|>  \n<think>\n\n</think>\n\n  "),
    ("tool_markup", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
    ("long_word", "a" * 5000),
    ("long_mixed_word", "Supercalifragilisticexpialidocious" * 40),
    ("long_digits", "9" * 300),
    ("long_space", " " * 1000 + "x"),
    ("control_chars", "a\x00b\x01c\x7fd\x1be\x1cf\x1f"),
    ("replacement_char", "bad � here"),
    ("math_symbols", "∀x∈ℝ: x² ≥ 0 ⇒ √x ∈ ℝ ∪ {∞} ≠ ∅"),
    ("mixed_scripts", "Hello Привет Γειά σου こんにちは 你好 مرحبا"),
    ("supplementary", "𝐀𝐁𝐂 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝄞 🫠 ꙮ 𠀋"),
    ("variation_selectors", "☺︎ ☺️ 葛󠄀 #️⃣"),
]

# Raw byte strings (not valid UTF-8). Expected ids use Python's decode('utf-8', 'replace'),
# which substitutes U+FFFD per maximal subpart; HALO documents the same policy.
INVALID_UTF8: list[tuple[str, bytes]] = [
    ("invalid_lone_continuation", b"abc\x80def"),
    ("invalid_truncated_3byte", b"x\xe2\x82y"),
    ("invalid_surrogate", b"s\xed\xa0\x80t"),
    ("invalid_overlong", b"o\xc0\xafp\xe0\x80\xafq"),
    ("invalid_ff", b"\xff\xfe hello"),
    ("invalid_truncated_4byte_end", b"end\xf0\x9f\x98"),
    ("invalid_above_max", b"m\xf4\x90\x80\x80n"),
    ("invalid_mixed", b"\xf0\x9f\x98\xc0\xe2\x82x\xed\xa0\x80\xe2\x82\xac"),
]

FUZZ_PALETTE = [
    "a", "Z", "s", "S", "ſ", "t", "T", "r", "R", "e", "E", "v", "V", "l", "L", "d", "D", "m",
    "M", "'", "'", " ", " ", " ", "  ", "\t", "\n", "\n", "\r", "\r\n", "0", "7", "٣", "½",
    "Ⅻ", "!", ".", ",", "$", "(", ")", "{", "}", "_", "-", " ", "　", " ",
    "\u0085", "​", "é", "é", "́", "̣", "̴", "ạ", "中",
    "文", "あ", "ア", "한", "ᄀ", "ᅡ", "ᆨ", "ع", "ש",
    "\U0001f600", "\U0001f468‍\U0001f469", "️", "\U0001f1fa\U0001f1f8", "ß",
    "İ", "ﬁ", "K", "\U0001d400", "<think>", "</think>", "<|im_start|>",
    "<|im_end|>", "<tool_call>", "<|endoftext|>", "<|fim_prefix|>", "<", "|", ">", "\x00",
    "\x7f", "�", "ǅ", "ཱ", "ั", "क़", "\U00011938", "\U00011935\U00011930",
]


def gguf_tokenizer_arrays(path: Path) -> dict[str, Any]:
    """Read the tokenizer KV arrays of a header-only GGUF with gguf-py.

    ``GGUFReader`` maps tensor data eagerly and fails on header-only files, so the tensor
    step is overridden to a no-op; only the KV fields are used.
    """
    from gguf import GGUFReader  # type: ignore[import-untyped]

    class HeaderReader(GGUFReader):  # type: ignore[misc]
        def _build_tensors(self, start_offs: int, fields: Any) -> None:
            self.tensors = []

    r = HeaderReader(str(path))

    def val(key: str) -> Any:
        return r.fields[key].contents()

    return {
        "source": path.name,
        "model": val("tokenizer.ggml.model"),
        "pre": val("tokenizer.ggml.pre"),
        "tokens": val("tokenizer.ggml.tokens"),
        "token_type": [int(x) for x in val("tokenizer.ggml.token_type")],
        "merges": val("tokenizer.ggml.merges"),
        "bos": int(val("tokenizer.ggml.bos_token_id")),
        "eos": int(val("tokenizer.ggml.eos_token_id")),
        "pad": int(val("tokenizer.ggml.padding_token_id")),
        "chat_template": val("tokenizer.chat_template"),
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)", file=sys.stderr)


def cmd_golden(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    auto = Qwen3_5Tokenizer.from_pretrained(str(HF_DIR))
    legacy_auto = AutoTokenizer.from_pretrained(str(HF_DIR))  # Qwen2Tokenizer, see module doc
    legacy_disagreements: list[str] = []
    raw = Tokenizer.from_file(str(TOKENIZER_JSON))
    nosplit = Tokenizer.from_file(str(TOKENIZER_JSON))
    nosplit.encode_special_tokens = True  # == HALO encode(..., parse_special=false)
    hf_nfc = normalizers.NFC()

    def case(name: str, text: str, raw_bytes: bytes | None = None) -> dict[str, Any]:
        ids = auto.encode(text, add_special_tokens=True)
        if ids != raw.encode(text).ids:
            raise SystemExit(f"Qwen3_5Tokenizer and tokenizers disagree on {name}")
        if legacy_auto.encode(text, add_special_tokens=True) != ids:
            legacy_disagreements.append(name)
        c: dict[str, Any] = {
            "name": name,
            "ids": ids,
            "ids_no_special": nosplit.encode(text).ids,
            "nfc": hf_nfc.normalize_str(text),
            "decoded": auto.decode(ids, skip_special_tokens=False),
            "decoded_skip": auto.decode(ids, skip_special_tokens=True),
        }
        if raw_bytes is None:
            c["text"] = text
        else:
            c["bytes_hex"] = raw_bytes.hex()
            c["replaced"] = text
        return c

    cases = [case(n, t) for n, t in CORPUS]
    cases += [case(n, b.decode("utf-8", "replace"), b) for n, b in INVALID_UTF8]
    for c in cases:
        if "text" in c and c["decoded"] != c["nfc"]:
            print(f"note: decode(encode(x)) != NFC(x) for {c['name']}", file=sys.stderr)

    rng = random.Random(20260923)
    fuzz = []
    for i in range(3000):
        text = "".join(rng.choice(FUZZ_PALETTE) for _ in range(rng.randint(0, 24)))
        fuzz.append(case(f"fuzz_{i}", text))

    # Codepoint sweep: every BMP + plane-1 scalar, sampled planes 2..16, in class-revealing
    # contexts; chunked so a failure names a small range.
    sweep_cps = [c for c in range(0x20000) if c not in SURROGATES]
    sweep_cps += list(range(0x20000, 0x110000, 97))
    # Token ids alone are blind to a misclassified rare character (no merge crosses it, so
    # the ids match either way); the pre-tokenizer piece lengths (codepoints, after NFC)
    # pin the class of every swept codepoint directly.
    splitter = make_splitter()
    sweep = []
    for start in range(0, len(sweep_cps), 512):
        chunk = sweep_cps[start : start + 512]
        text = "".join(f"x{chr(c)}1!{chr(c)}! {chr(c)}{chr(c)}a\n" for c in chunk)
        ids = raw.encode(text).ids
        piece_lens = [len(p) for p in pieces(splitter, hf_nfc.normalize_str(text))]
        sweep.append({"first": chunk[0], "last": chunk[-1], "text": text, "ids": ids, "piece_lens": piece_lens})

    # NFC sweep over every decomposable / combining codepoint.
    nfc_tables = build_nfc_tables()
    nfc_items = sorted(set(nfc_tables.decomp) | set(nfc_tables.ccc))
    nfc_cases = []
    for start in range(0, len(nfc_items), 256):
        chunk = nfc_items[start : start + 256]
        text = "".join(
            ud.normalize("NFD", chr(c)) + " " + chr(c) + " x́" + chr(c) + "̴ á" + chr(c) + "̣\n"
            for c in chunk
        )
        nfc_cases.append({"first": chunk[0], "last": chunk[-1], "text": text,
                          "ids": raw.encode(text).ids, "nfc": hf_nfc.normalize_str(text)})

    # Decode goldens: prefixes that split multi-byte characters, and random id sequences.
    decode_cases = []
    for text in ["𝄞🫠ꙮ", "héllo 😀 世界", "👨‍👩‍👧‍👦 é"]:
        ids = raw.encode(text).ids
        for n in range(len(ids) + 1):
            decode_cases.append({"ids": ids[:n], "decoded": auto.decode(ids[:n]),
                                 "decoded_skip": auto.decode(ids[:n], skip_special_tokens=True)})
    n_ids = raw.get_vocab_size(with_added_tokens=True)
    for _ in range(2000):
        ids = [rng.randrange(n_ids) for _ in range(rng.randint(1, 12))]
        decode_cases.append({"ids": ids, "decoded": auto.decode(ids),
                             "decoded_skip": auto.decode(ids, skip_special_tokens=True)})

    # A realistic long prompt (code + docstrings) of at least 32K tokens for the perf test.
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    src = Path(mq.__file__).read_text(encoding="utf-8")
    long_text = src
    while len(raw.encode(long_text).ids) < 32768:
        long_text += "\n" + src
    long_ids = raw.encode(long_text).ids

    write_json(out / "cases.json", {"cases": cases})
    write_json(out / "fuzz.json", {"cases": fuzz})
    write_json(out / "sweep.json", {"chunks": sweep})
    write_json(out / "nfc.json", {"chunks": nfc_cases})
    write_json(out / "decode.json", {"cases": decode_cases})
    write_json(out / "long.json", {"text": long_text, "ids": long_ids})
    write_json(out / "gguf_unsloth_vocab.json", gguf_tokenizer_arrays(REF / "unsloth-ud-q4kxl.header.gguf"))
    import transformers

    write_json(out / "manifest.json", {
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "tokenizer_json": str(TOKENIZER_JSON),
        "unicodedata": ud.unidata_version,
        "reference": "tokenizers.Tokenizer.from_file(tokenizer.json) == Qwen3_5Tokenizer",
        "autotokenizer_class": type(legacy_auto).__name__,
        "autotokenizer_disagreements": len(legacy_disagreements),
        "autotokenizer_disagreement_examples": legacy_disagreements[:20],
    })
    print(f"AutoTokenizer ({type(legacy_auto).__name__}) disagrees on {len(legacy_disagreements)} cases",
          file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables", help="generate src/tokenizer/unicode_data.inc")
    t.add_argument("--out", default=str(REPO / "src" / "tokenizer" / "unicode_data.inc"))
    g = sub.add_parser("golden", help="write golden cases")
    g.add_argument("--out", default=str(REF / "tokenizer_golden"))
    args = ap.parse_args()
    {"tables": cmd_tables, "golden": cmd_golden}[args.cmd](args)


if __name__ == "__main__":
    main()
