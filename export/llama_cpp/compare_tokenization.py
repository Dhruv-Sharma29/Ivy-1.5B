"""Check that llama.cpp tokenizes exactly like the frozen HF tokenizer.

    .venv/bin/python export/llama_cpp/compare_tokenization.py --llama-dir ~/llama.cpp --gguf model.gguf \
        --files data/sample/heldout-fineweb-edu.txt data/sample/heldout-finemath.txt

Runs `llama-tokenize --ids --no-bos --no-escape` (WITHOUT --no-escape, llama-tokenize turns the two characters
backslash+r into a carriage return and LaTeX like \\rightarrow shows up as a false mismatch). Exit code 1 on any diff.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

EDGE = [
    "In 2024 there were 1,234,567 items at 3.14159 each; call 555-0199 or +91 98765 43210.",
    "a  b   c\t\td\n\n\ne \n f  ", "ends with spaces   ", "line1\r\nline2\r\n\r\nline3",
    "naïve café — 日本語のテキスト, नमस्ते दुनिया 😀🚀 ǅ",
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu /Users/meg/Documents/x_y-z.txt ~/proj",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": x**2}  # comment\n",
    '{"action": "tool_call", "tool": "set_volume", "device": "phone", "args": {"level": 30}}',
    r"\int_0^1 x^2\,dx = \frac{1}{3} \quad \alpha_{i}^{2} \rightarrow \nu", "I'm sure they've said we'll go; it's John's.",
    "\n\n\n", "7",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/ivy-bpe-32k/tokenizer.json")
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    binary = Path(a.llama_dir) / "build/bin/llama-tokenize"

    def llama(text: str, parse_special: bool = False) -> list[int]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
            f.write(text)
        cmd = [str(binary), "-m", a.gguf, "-f", f.name, "--ids", "--no-bos", "--no-escape", "--log-disable"]
        if not parse_special:
            cmd.append("--no-parse-special")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=True).stdout.strip()
        Path(f.name).unlink()
        return ast.literal_eval(out)

    bad = 0
    cases = [(f"edge[{i}]", t, False) for i, t in enumerate(EDGE)]
    cases.append(("special tokens", "<|system|>hi<|user|>open Safari<|end|>", True))
    cases += [(p, Path(p).read_text(encoding="utf-8"), False) for p in a.files]
    for name, text, special in cases:
        ref = tok.encode(text, add_special_tokens=False).ids
        got = llama(text, special)
        ok = ref == got
        bad += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {name}: {len(ref):,} tokens" + ("" if ok else f" vs {len(got):,}"))
    print("mismatching cases:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
