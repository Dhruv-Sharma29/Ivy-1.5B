"""Corpus cleaning: secret/PII scrubbing, MinHash near-duplicate removal, eval decontamination.

All pure numpy + stdlib, deterministic (fixed seeds, no use of Python's randomised hash()).

Limits (read before trusting the output):
  * Secret scrubbing is a regex floor, not a guarantee: it catches common token formats and obvious
    `key = "value"` assignments, and will miss novel formats and over-match some code. Counts are reported.
  * Near-dup detection is word-5-gram MinHash + LSH; the Jaccard estimate has ~±0.09 std error at 128 perms.
    At threshold 0.8 it only catches near-identical copies: on synthetic 800-word docs, replacing 1% of words
    was flagged 60/60, 5% or more 0/60 (each edited word destroys up to 5 shingles). Lower `threshold`
    (e.g. 0.6-0.7) to catch looser copies at the cost of more false positives; tune on a real sample.
    Memory is 0.5 KB per kept document; measured ~1.2M words/s single-threaded on synthetic text (a real
    corpus with long documents and many candidate collisions may be slower).
  * N-gram decontamination only works for eval items with at least `n` words. Short items (such as the Ivy
    scenario utterances) cannot be removed from a web corpus this way; use `check-sft` on instruction data.

    python -m ivy_model.cleaning check-sft sft.jsonl --field user     # exit 1 if it overlaps the Ivy evals
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------ scrubbing

_SECRET_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "<SECRET>"),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<SECRET>"),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b"), "<SECRET>"),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "<SECRET>"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"), "<SECRET>"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "<SECRET>"),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "<SECRET>"),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "<SECRET>"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "<SECRET>"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
]
_ASSIGN = re.compile(r"""(?i)\b((?:api[_-]?key|secret|token|passwd|password)\w*\s*[:=]\s*)(['"])[^'"\s]{12,}\2""")


def scrub_secrets(text: str) -> tuple[str, dict[str, int]]:
    """Replace secret-looking strings and emails with placeholders. Returns (text, counts by kind)."""
    counts: dict[str, int] = {}
    for kind, rx, repl in _SECRET_PATTERNS:
        text, n = rx.subn(repl, text)
        if n:
            counts[kind] = n
    text, n = _ASSIGN.subn(lambda m: f"{m.group(1)}{m.group(2)}<SECRET>{m.group(2)}", text)
    if n:
        counts["assigned_secret"] = n
    return text, counts


# ------------------------------------------------------------------------------------------ text helpers

_WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower())


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def shingle_hashes(ws: list[str], n: int) -> np.ndarray:
    """uint64 hashes of the distinct word n-grams (the whole text if it has fewer than n words)."""
    if not ws:
        return np.empty(0, dtype=np.uint64)
    grams = {" ".join(ws[i:i + n]) for i in range(max(1, len(ws) - n + 1))}
    return np.fromiter((_h64(g) for g in grams), dtype=np.uint64, count=len(grams))


# ------------------------------------------------------------------------------------------ near-duplicates

class NearDupIndex:
    """Streaming MinHash-LSH. `is_duplicate(text)` returns True if a previously *kept* document has
    estimated Jaccard >= threshold; otherwise the document is indexed and False is returned."""

    def __init__(self, num_perm: int = 128, bands: int = 16, shingle: int = 5,
                 threshold: float = 0.8, seed: int = 0) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.shingle, self.threshold = shingle, threshold
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
        self.sigs: list[np.ndarray] = []
        self.tables: list[dict[bytes, list[int]]] = [{} for _ in range(bands)]

    def signature(self, text: str) -> np.ndarray:
        x = shingle_hashes(words(text), self.shingle)
        if x.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)
        sig = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for i in range(0, x.size, 8192):  # chunked: bounds memory for very long documents
            blk = (x[i:i + 8192, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(32)  # wraps mod 2**64
            sig = np.minimum(sig, blk.min(axis=0))
        return sig.astype(np.uint32)  # values are < 2**32 after the shift; halves index memory

    @staticmethod
    def jaccard_estimate(s1: np.ndarray, s2: np.ndarray) -> float:
        return float(np.mean(s1 == s2))

    def is_duplicate(self, text: str) -> bool:
        if not words(text):
            return False
        sig = self.signature(text)
        keys = [sig[b * self.rows:(b + 1) * self.rows].tobytes() for b in range(self.bands)]
        for b, k in enumerate(keys):
            for j in self.tables[b].get(k, ()):
                if self.jaccard_estimate(sig, self.sigs[j]) >= self.threshold:
                    return True
        idx = len(self.sigs)
        self.sigs.append(sig)
        for b, k in enumerate(keys):
            self.tables[b].setdefault(k, []).append(idx)
        return False


# ------------------------------------------------------------------------------------------ decontamination

class NgramBlocklist:
    """Flags documents sharing any word n-gram (default 13, as is common for benchmark decontamination)
    with the given eval texts. Eval texts shorter than n words are counted in `skipped_short`, not matched."""

    def __init__(self, texts: list[str], n: int = 13) -> None:
        self.n, self.skipped_short, self.grams = n, 0, set()
        for t in texts:
            ws = words(t)
            if len(ws) < n:
                self.skipped_short += 1
                continue
            for i in range(len(ws) - n + 1):
                self.grams.add(_h64(" ".join(ws[i:i + n])))

    def contaminated(self, text: str) -> bool:
        if not self.grams:
            return False
        ws = words(text)
        return any(_h64(" ".join(ws[i:i + self.n])) in self.grams for i in range(len(ws) - self.n + 1))


def _word_jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def find_eval_overlap(candidates: list[str], evals: dict[str, str], min_jaccard: float = 0.8) -> list[dict]:
    """Candidates (e.g. SFT user turns) that equal, or are near-identical to, an eval utterance after
    normalisation (case, punctuation, spacing). O(len(candidates) * len(evals)); fine for instruction data."""
    ev = {k: (tuple(words(v)), set(words(v))) for k, v in evals.items()}
    hits = []
    for i, c in enumerate(candidates):
        cw = tuple(words(c))
        cs = set(cw)
        for k, (ew, es) in ev.items():
            j = 1.0 if cw == ew else _word_jaccard(cs, es)
            if j >= min_jaccard:
                hits.append({"candidate_index": i, "eval_id": k, "jaccard": round(j, 3), "text": c})
    return hits


def load_ivy_eval_utterances(evals_dir: str | Path | None = None) -> dict[str, str]:
    d = Path(evals_dir) if evals_dir else Path(__file__).resolve().parents[2] / "evals"
    out: dict[str, str] = {}
    for f in sorted(d.glob("scenarios_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[f"{f.stem}:{r['id']}"] = r["user"]
                # v2 scenarios are conversations: protect every user turn of 3+ words (a bare "yes" or "the phone"
                # is too generic to guard and would flag unrelated SFT data)
                for i, m in enumerate(r.get("messages", [])):
                    if m.get("role") == "user" and len(words(m["content"])) >= 3 and m["content"] != r["user"]:
                        out[f"{f.stem}:{r['id']}:m{i}"] = m["content"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-sft", help="fail if instruction data overlaps the held-out Ivy scenarios")
    c.add_argument("file")
    c.add_argument("--field", default="user")
    c.add_argument("--min-jaccard", type=float, default=0.8)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.file).read_text().splitlines() if l.strip()]
    cands = []
    for r in rows:
        if a.field in r:
            cands.append(str(r[a.field]))
        # chat-format rows keep their text in `messages`: check every user turn (3+ words, matching the eval guard)
        cands += [m["content"] for m in r.get("messages", []) if m.get("role") == "user" and len(words(m["content"])) >= 3]
    if not cands:
        sys.exit(f"nothing to check: no '{a.field}' field and no user messages in {a.file} (refusing to report a clean result)")
    hits = find_eval_overlap(cands, load_ivy_eval_utterances(), a.min_jaccard)
    print(json.dumps({"checked": len(cands), "overlaps": len(hits), "examples": hits[:10]}, indent=2, ensure_ascii=False))
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
