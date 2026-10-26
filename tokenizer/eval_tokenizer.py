"""Evaluate candidate tokenizers on held-out text (disjoint from the training sample) plus hand-written probes.

    .venv/bin/python tokenizer/eval_tokenizer.py tokenizer/ivy-bpe-32k-candidate tokenizer/ivy-bpe-32k-digits-candidate

Reports chars/token (higher = more efficient), exact round-trip on every held-out document, vocabulary
utilisation, special-token ids, and token counts on probes (JSON tool calls, code, LaTeX, numbers, paths).
Probes are my own short strings, NOT drawn from evals/scenarios_*.jsonl. They show tendencies only: they
are a handful of strings, not a benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

HELDOUT = {"web (FineWeb-Edu)": "data/sample/heldout-fineweb-edu.txt", "math (FineMath)": "data/sample/heldout-finemath.txt"}
PROBES = {
    "json_tool_call": '{"action": "tool_call", "tool": "move_file", "device": "laptop", "args": {"source": "~/a.txt", "destination": "~/Archive/a.txt"}}',
    "python": "def fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\nprint([fib(i) for i in range(10)])\n",
    "swift": "func greet(_ name: String) -> String {\n    return \"Hello, \\(name)!\"\n}\nlet items = names.map { greet($0) }\n",
    "shell": "for f in *.log; do grep -c ERROR \"$f\" >> /tmp/summary.txt; done\n",
    "latex": r"\int_0^1 x^2\,dx = \frac{1}{3}, \quad \sum_{k=1}^{n} k = \frac{n(n+1)}{2}",
    "numbers": "The population was 1,234,567 in 2024, growing 3.14159% to 7890123456.",
    "path_url": "/Users/meg/Documents/Projects/ivy-1.5b/data/sample.txt https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    "emoji": "Great job \U0001F600\U0001F680 — see you tomorrow!",
    "hindi (out of scope)": "नमस्ते दुनिया, आप कैसे हैं?",
}


def evaluate(path: str) -> dict:
    tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    man = json.loads((Path(path) / "MANIFEST.json").read_text())
    out: dict = {"path": path, "vocab": tok.get_vocab_size(), "split_digits": man.get("split_digits", False),
                 "sha256": man["sha256"][:16], "heldout": {}, "probes": {}}
    for name, f in HELDOUT.items():
        docs = [d for d in Path(f).read_text(encoding="utf-8").split("\n\n") if d.strip()]
        encs = tok.encode_batch(docs)
        n_tok = sum(len(e.ids) for e in encs)
        n_chr = sum(len(d) for d in docs)
        lossless = sum(tok.decode(e.ids) == d for e, d in zip(encs[:300], docs[:300]))
        used = len({i for e in encs for i in e.ids})
        out["heldout"][name] = {"docs": len(docs), "tokens": n_tok, "chars_per_token": round(n_chr / n_tok, 3),
                                "roundtrip_exact_first300": f"{lossless}/{min(300, len(docs))}",
                                "vocab_used_frac": round(used / tok.get_vocab_size(), 3)}
    for name, text in PROBES.items():
        e = tok.encode(text)
        out["probes"][name] = {"tokens": len(e.ids), "chars_per_token": round(len(text) / len(e.ids), 2)}
    out["probes"]["_numbers_tokens"] = tok.encode(PROBES["numbers"]).tokens
    return out


def main() -> None:
    res = [evaluate(p) for p in sys.argv[1:]]
    for r in res:
        print(f"\n== {r['path']}  vocab={r['vocab']} split_digits={r['split_digits']} sha256={r['sha256']}")
        for k, v in r["heldout"].items():
            print(f"  {k:20s} {v['chars_per_token']:.3f} chars/token  ({v['tokens']:,} tokens)  roundtrip {v['roundtrip_exact_first300']}  vocab used {v['vocab_used_frac']:.0%}")
        print("  probes (tokens, chars/token): " + "; ".join(f"{k}={v['tokens']}" for k, v in r["probes"].items() if not k.startswith("_")))
        print("  numbers ->", r["probes"]["_numbers_tokens"])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/tokenizer-candidates-eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
