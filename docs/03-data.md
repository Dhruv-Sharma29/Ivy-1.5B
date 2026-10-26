# Data

## Sources

Registered in [`data-manifests/sources.json`](../data-manifests/sources.json), which also records the licence-review status of
each source.

| Source | Hugging Face dataset | Subset | Domain | Card licence | Used |
|---|---|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | educational web text | ODC-By | yes (85%) |
| FineMath | `HuggingFaceTB/finemath` | `finemath-4plus` | mathematical web text | ODC-By | yes (15%) |
| The Stack v2 | `bigcode/the-stack-v2` | not chosen | code | per-file licences | **not yet**: gated access and per-file licence filtering pending |

The current mixture is **85% FineWeb-Edu / 15% FineMath by tokens, with no code**. The 60/25/15 web/code/math split in
`sources.json` is the intended target once code data is available; it has not been tested.

## Cleaning pipeline

[`src/ivy_model/prepare_data.py`](../src/ivy_model/prepare_data.py) streams one source, cleans each document and writes
`uint16` token shards plus a `MANIFEST.json` that records what ran and what was removed. Per document, in order:

1. **Unicode NFC normalisation.**
2. **Exact deduplication**: sha256 of the normalised text.
3. **Secret and email scrubbing**: common token formats, private-key blocks, emails and `key = "value"` assignments become
   `<SECRET>` / `<EMAIL>`. This is a floor, not a guarantee: unknown formats are missed and no other personal data is handled.
4. **Near-duplicate removal** (optional): word 5-gram MinHash-LSH, threshold 0.8. Off for the 2B-token corpus (see below).
5. **N-gram decontamination** (optional, `--decontaminate-file`): drops documents sharing a 13-gram with benchmark items.
6. **Tokenization** with the frozen tokenizer; each document ends with `<|endoftext|>`.

Failed or malformed documents are counted in the manifest, never silently dropped.

**Validation split**: document-level and deterministic, `sha256(source_id + doc_key) mod 1000 < 5`, i.e. ~0.5%. FineMath rows
have no id column, so its split is by row index.

**Balanced file sampling** (`--balanced-files-prefix`): streams each parquet file of a source separately and takes an equal token
share from each. Use it at scale. Plain file order is biased (the first FineWeb-Edu documents all come from one 2013 crawl), and
`datasets` streaming `.shuffle()` grew past 18 GB of RAM within two minutes on the training machine, whereas balanced sampling
peaks at ~1.3 GB.

## Corpora

| Corpus | Train tokens | Val tokens | Sampling | Near-dup | Used for |
|---|---:|---:|---|---|---|
| `mix-v0` | 100.0M (85.0M web + 15.0M math) | 0.51M | seeded shuffle (buffer 10k) | on (removed 8 docs) | 40M pilot, 300M LR sweep |
| `mix-v1` | 2.00B (1.70B web + 0.30B math) | 9.5M | balanced per file (14 + 64 files) | off | 40M sweep and scaling, 300M scaling |

`mix-v1` details: FineWeb-Edu 1,597,235 documents (11,363 exact duplicates removed, 0.7%; 35,286 scrubbed); FineMath 186,951
documents (3,193 scrubbed). Near-dup removal was turned off because it costs ~6 KB of RAM per document (~10 GB at this scale) and
removed only 8 documents in the 100M pilot: both sources are already deduplicated upstream.

The two corpora are different samples of the same sources. **Do not compare validation losses across them**: for the same model,
`mix-v1` validation is ~0.13 nats harder than `mix-v0`.

### Building `mix-v1`

```bash
PYTHONPATH=src sh scripts/build_corpus_v1.sh          # ~80 min, CPU only, peak RSS ~2 GB
mkdir -p data/tokens/mix-v1
cat data/tokens/v1/fineweb-edu/train-*.bin data/tokens/v1/finemath/train-*.bin > data/tokens/mix-v1/train.bin
cat data/tokens/v1/fineweb-edu/val-*.bin   data/tokens/v1/finemath/val-*.bin   > data/tokens/mix-v1/val.bin
```

The concatenation order above reproduces the published files byte for byte (train sha256 prefix `bb0dd1fac8da64a3`, val
`cfa58a4825ddeb66`). The trainer samples windows uniformly over positions, so the concatenated file keeps the 85/15 mix.

### Shard format

Flat little-endian `uint16` token ids (vocabulary 32,768 fits), documents separated by `<|endoftext|>` (id 0). Every shard was
checked after writing: size matches the manifest, all ids < 32,768, every shard ends with an end-of-text token, and the sha256
matches.

## Keeping evaluation data out of training

The tool-use eval utterances are short (often under 13 words), so n-gram decontamination cannot protect them. They are guarded on
the fine-tuning side instead:

```bash
PYTHONPATH=src .venv/bin/python -m ivy_model.cleaning check-sft data/sft/v1-train.jsonl --min-jaccard 0.6
```

This compares every user turn of 3+ words in the training file against every user turn of `evals/scenarios_*.jsonl` (word
Jaccard) and exits non-zero on any overlap. The SFT generator applies the same filter while generating, so the shipped SFT files
have 0 overlaps at Jaccard 0.6.

## Licences and provenance

- FineWeb-Edu and FineMath are ODC-By (attribution and notice required). This was read from the dataset cards, not a legal
  review; the terms of the underlying Common Crawl pages have not been reviewed.
- The Stack v2 would need per-file licence filtering to a permissive allowlist, opt-out handling and attribution records.
- Tool-use training data is generated from hand-written templates. No teacher model is used. If one is added, it must have a
  permissive licence that allows training other models, and every example must record the teacher name, version and licence.
- Every `reviewed` flag in `sources.json` is still `false`. See the release checklist in the [Roadmap](09-roadmap.md).
