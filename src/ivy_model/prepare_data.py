"""Stream one source from sources.json, tokenize, write uint16 shards + a manifest (Phase 1).

    python -m ivy_model.prepare_data --source fineweb-edu --tokenizer tokenizer/ivy-bpe-32k \
        --out data/tokens/fineweb-edu --max-tokens 100_000_000

Cleaning order per document: NFC normalise -> exact dedup -> secret/email scrub -> MinHash near-dup
(see cleaning.py for its limits) -> n-gram decontamination against --decontaminate-file -> tokenize.
Scrubbing and near-dup are on by default (--no-scrub / --no-near-dup to disable); the manifest records
what actually ran and what was removed.

Not yet run: requires network, the `datasets` package, and a frozen tokenizer.
Failed/malformed documents are counted and reported, never silently dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import unicodedata
from pathlib import Path

import numpy as np

from .cleaning import NearDupIndex, NgramBlocklist, scrub_secrets

SHARD_TOKENS = 100_000_000  # ~200 MB per uint16 shard
VAL_PER_MILLE = 5


def is_val(source_id: str, doc_key: str) -> bool:
    h = hashlib.sha256(f"{source_id}\0{doc_key}".encode()).digest()
    return int.from_bytes(h[:4], "big") % 1000 < VAL_PER_MILLE


class ShardWriter:
    def __init__(self, out: Path, split: str) -> None:
        self.out, self.split, self.buf, self.idx, self.entries = out, split, [], 0, []
        self.n = 0

    def add(self, ids: list[int]) -> None:
        self.buf.append(np.asarray(ids, dtype=np.uint16))
        self.n += len(ids)
        if self.n >= SHARD_TOKENS:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        arr = np.concatenate(self.buf)
        path = self.out / f"{self.split}-{self.idx:05d}.bin"
        arr.tofile(path)
        self.entries.append({"file": path.name, "tokens": int(arr.size),
                             "sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
        self.buf, self.n, self.idx = [], 0, self.idx + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--sources-file", default="data-manifests/sources.json")
    ap.add_argument("--tokenizer", required=True, help="dir containing tokenizer.json + MANIFEST.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--no-scrub", action="store_true", help="skip secret/email scrubbing")
    ap.add_argument("--no-near-dup", action="store_true", help="skip MinHash near-duplicate removal")
    ap.add_argument("--near-dup-threshold", type=float, default=0.8)
    ap.add_argument("--decontaminate-file", action="append", default=[],
                    help="JSONL of eval items to remove 13-gram overlaps with (repeatable)")
    ap.add_argument("--decontaminate-field", default="text")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000,
                    help="approximate streaming shuffle (documents); 0 = file order, which is biased to the first shards")
    ap.add_argument("--balanced-files-prefix",
                    help="repo path prefix of parquet files (e.g. sample/10BT/): stream each file separately and take an equal "
                         "token share from each, instead of one pass in file order. Use this INSTEAD of the shuffle buffer: "
                         "datasets 5.0 streaming shuffle blew past 18 GB RAM here, and file order is biased (the first "
                         "3,000 FineWeb-Edu docs all come from one 2013 dump)")
    ap.add_argument("--progress-every", type=int, default=10_000, help="log every N documents seen")
    a = ap.parse_args()

    from datasets import load_dataset  # lazy: only needed when actually running
    from tokenizers import Tokenizer

    src = next(s for s in json.loads(Path(a.sources_file).read_text())["sources"] if s["id"] == a.source)
    if "TODO" in src["subset"]:
        raise SystemExit(f"{a.source}: subset is unresolved in sources.json; decide it first")
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    writers = {"train": ShardWriter(out, "train"), "val": ShardWriter(out, "val")}
    seen: set[bytes] = set()
    stats = {"docs": 0, "empty_or_malformed": 0, "exact_dupes": 0, "near_dupes": 0, "decontaminated": 0,
             "docs_scrubbed": 0, "train_tokens": 0, "val_tokens": 0}
    scrub_counts: dict[str, int] = {}
    near = None if a.no_near_dup else NearDupIndex(threshold=a.near_dup_threshold)
    eval_texts = [str(json.loads(l)[a.decontaminate_field]) for f in a.decontaminate_file
                  for l in Path(f).read_text().splitlines() if l.strip()]
    blocklist = NgramBlocklist(eval_texts) if eval_texts else None

    t_start = time.time()
    n_files = 1

    def row_stream():
        nonlocal n_files
        if not a.balanced_files_prefix:
            ds = load_dataset(src["hf_repo"], name=src["subset"], split="train", streaming=True)
            if a.shuffle_buffer > 0:
                ds = ds.shuffle(seed=a.shuffle_seed, buffer_size=a.shuffle_buffer)
            yield from ds
            return
        from huggingface_hub import HfApi
        files = sorted(f.path for f in HfApi().list_repo_tree(src["hf_repo"], path_in_repo=a.balanced_files_prefix.rstrip("/"),
                                                              repo_type="dataset") if f.path.endswith(".parquet"))
        if not files:
            raise SystemExit(f"no parquet files under {a.balanced_files_prefix} in {src['hf_repo']}")
        n_files = len(files)
        print(f"balanced sampling over {n_files} files, ~{a.max_tokens // n_files:,} train tokens each", flush=True)
        for k, path in enumerate(files):
            target = round(a.max_tokens * (k + 1) / n_files)  # cumulative, so a short file is made up by later ones
            ds = load_dataset("parquet", data_files=f"hf://datasets/{src['hf_repo']}/{path}", split="train", streaming=True)
            for row in ds:
                yield row
                if stats["train_tokens"] >= target:
                    break

    for i, row in enumerate(row_stream()):
        if i and i % a.progress_every == 0:
            print(f"  {i:,} docs read, kept {stats['docs']:,}, train tokens {stats['train_tokens']:,}, "
                  f"exact {stats['exact_dupes']:,} near {stats['near_dupes']:,} | {time.time() - t_start:.0f}s | "
                  f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB", flush=True)
        text = row.get(src["text_field"])
        if not isinstance(text, str) or not text.strip():
            stats["empty_or_malformed"] += 1
            continue
        text = unicodedata.normalize("NFC", text)
        h = hashlib.sha256(text.encode()).digest()
        if h in seen:
            stats["exact_dupes"] += 1
            continue
        seen.add(h)
        if not a.no_scrub:
            text, counts = scrub_secrets(text)
            if counts:
                stats["docs_scrubbed"] += 1
                for k, v in counts.items():
                    scrub_counts[k] = scrub_counts.get(k, 0) + v
        if near is not None and near.is_duplicate(text):
            stats["near_dupes"] += 1
            continue
        if blocklist is not None and blocklist.contaminated(text):
            stats["decontaminated"] += 1
            continue
        ids = tok.encode(text).ids + [eot]
        split = "val" if is_val(a.source, str(row.get("id", i))) else "train"
        writers[split].add(ids)
        stats["docs"] += 1
        stats[f"{split}_tokens"] += len(ids)
        if stats["train_tokens"] >= a.max_tokens:
            break

    for w in writers.values():
        w.flush()
    tok_manifest = json.loads((Path(a.tokenizer) / "MANIFEST.json").read_text())
    (out / "MANIFEST.json").write_text(json.dumps({
        "source": src, "shuffle": {"seed": a.shuffle_seed, "buffer": a.shuffle_buffer},
        "balanced_files": {"prefix": a.balanced_files_prefix, "files": n_files} if a.balanced_files_prefix else None,
        "wall_seconds": round(time.time() - t_start, 1), "tokenizer_sha256": tok_manifest["sha256"], "stats": stats,
        "scrub_counts": scrub_counts,
        "cleaning": {"scrub": not a.no_scrub,
                     "near_dup": None if near is None else {"threshold": a.near_dup_threshold, "shingle": near.shingle},
                     "decontaminate": {"files": a.decontaminate_file, "n": 13,
                                       "eval_items_too_short_to_match": blocklist.skipped_short if blocklist else None}},
        "shards": {k: w.entries for k, w in writers.items()},
        "caveats": [c for c in (
            "secret/email scrubbing not applied" if a.no_scrub else "scrubbing is regex-only: misses unknown formats",
            "near-duplicate removal not applied" if near is None else None,
            "eval decontamination not applied" if blocklist is None else None) if c],
    }, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
