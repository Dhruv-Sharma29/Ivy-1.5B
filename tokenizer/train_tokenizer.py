"""Train the byte-level BPE tokenizer (Phase 2). Uses HF Tokenizers; no custom engine.

    python tokenizer/train_tokenizer.py --files data/sample/*.txt --out tokenizer/ivy-bpe-32k

Special-token IDs are assigned in list order starting at 0 and are part of the frozen
contract: do not reorder or insert before pretraining starts.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_def|>", "<|tool_call|>", "<|tool_result|>",
    "<|end|>",
] + [f"<|reserved_{i}|>" for i in range(7)]  # headroom so later additions don't shift IDs


def train(files: list[str], vocab_size: int, split_digits: bool = False) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # split_digits: every digit is its own pre-token (Llama-style), so numbers never merge into multi-digit tokens
    tok.pre_tokenizer = (pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), byte_level])
                         if split_digits else byte_level)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes: lossless on any input
        show_progress=True,
    )
    tok.train(files, trainer)
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="text files or globs (training split only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--split-digits", action="store_true", help="split every digit into its own token")
    a = ap.parse_args()
    files = sorted(f for pat in a.files for f in glob.glob(pat))
    if not files:
        raise SystemExit("no input files matched")
    tok = train(files, a.vocab_size, a.split_digits)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    digest = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()
    (out / "MANIFEST.json").write_text(json.dumps({
        "vocab_size": tok.get_vocab_size(), "sha256": digest,
        "special_tokens": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS},
        "split_digits": a.split_digits,
        "train_files": files,
    }, indent=2) + "\n")
    print(f"vocab={tok.get_vocab_size()} sha256={digest[:16]}")


if __name__ == "__main__":
    main()
