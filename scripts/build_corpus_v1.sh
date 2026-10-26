#!/bin/sh
# ~2B-token corpus (85/15 FineWeb-Edu / FineMath) with the frozen tokenizer. CPU only.
# Why not the shuffle used for mix-v0: datasets 5.0 streaming shuffle used >18 GB RAM within 2 minutes on this
# machine (it takes down a shared box), and file order is biased (first FineWeb-Edu docs: one 2013 dump). So we
# stream each parquet file separately and take an equal token share from each (balanced file sampling).
# Near-dup removal is OFF here: it costs ~6 KB RAM per document (~10 GB at this scale) and removed only 8 of 88k
# documents in the 100M pilot (the sources are already deduplicated upstream). Exact dedup and the scrub stay on.
# val docs are a doc-id hash split; FineMath has no id column, so its val split differs from mix-v0's.
# Usage: PYTHONPATH=src sh scripts/build_corpus_v1.sh
set -eu
ulimit -v 40000000   # 40 GB address-space cap: fail this job, never push other users out
echo "=== corpus start $(date -u +%FT%TZ)"
.venv/bin/python -u -m ivy_model.prepare_data --source fineweb-edu --tokenizer tokenizer/ivy-bpe-32k \
  --out data/tokens/v1/fineweb-edu --max-tokens 1700000000 --no-near-dup --balanced-files-prefix sample/10BT/ --progress-every 50000
.venv/bin/python -u -m ivy_model.prepare_data --source finemath --tokenizer tokenizer/ivy-bpe-32k \
  --out data/tokens/v1/finemath --max-tokens 300000000 --no-near-dup --balanced-files-prefix finemath-4plus/ --progress-every 50000
echo "=== corpus end $(date -u +%FT%TZ)"
