"""Stream a reproducible tokenizer-training sample + disjoint held-out text (Phase 2).

    .venv/bin/python tokenizer/make_sample.py --out data/sample

Budgets are in characters (about 4.3 chars/token for FineWeb-Edu, fewer for math/LaTeX), so the token
counts are approximate; real counts are recorded in SAMPLE.json after tokenizer training.
Training text comes first from each shuffled stream and the held-out text continues the same iterator, so
the two are disjoint by construction. Shuffle is a 10k-document buffer with a fixed seed (approximate
shuffle: streaming order is partly preserved). Applies the same normalisation as prepare_data.py
(NFC, exact dedup) and the secret/email scrub from cleaning.py so the tokenizer never learns real secrets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivy_model.cleaning import scrub_secrets  # noqa: E402

SOURCES = {  # id -> (repo, config)
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus"),
}


def collect(source: str, train_chars: int, heldout_chars: int, seed: int, out: Path) -> dict:
    from datasets import load_dataset
    repo, cfg = SOURCES[source]
    ds = load_dataset(repo, name=cfg, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    files = {"train": open(out / f"train-{source}.txt", "w", encoding="utf-8"),
             "heldout": open(out / f"heldout-{source}.txt", "w", encoding="utf-8")}
    budget = {"train": train_chars, "heldout": heldout_chars}
    got = {"train": 0, "heldout": 0}
    docs = {"train": 0, "heldout": 0}
    seen, skipped_dupes, scrubbed = set(), 0, 0
    phase = "train"
    for row in ds:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = unicodedata.normalize("NFC", text)
        h = hashlib.sha256(text.encode()).digest()
        if h in seen:
            skipped_dupes += 1
            continue
        seen.add(h)
        text, counts = scrub_secrets(text)
        scrubbed += bool(counts)
        files[phase].write(text.strip() + "\n\n")
        got[phase] += len(text)
        docs[phase] += 1
        if got[phase] >= budget[phase]:
            if phase == "heldout":
                break
            phase = "heldout"
    for f in files.values():
        f.close()
    return {"repo": repo, "config": cfg, "seed": seed, "chars": got, "docs": docs,
            "exact_dupes_skipped": skipped_dupes, "docs_scrubbed": scrubbed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fineweb-chars", type=int, default=365_000_000)
    ap.add_argument("--finemath-chars", type=int, default=50_000_000)
    ap.add_argument("--heldout-chars", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "fineweb-edu": collect("fineweb-edu", a.fineweb_chars, a.heldout_chars, a.seed, out),
        "finemath": collect("finemath", a.finemath_chars, a.heldout_chars // 4, a.seed, out),
    }
    (out / "SAMPLE.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
