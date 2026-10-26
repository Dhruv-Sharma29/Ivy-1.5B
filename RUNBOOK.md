# Runbook

Exact commands to reproduce every reported result, in order.


The commands below rebuild every reported number from an empty machine. Times are for the NVIDIA GB10 training machine without
contention; on a shared GPU expect up to 4× longer. GPU runs are not bit-reproducible (non-deterministic kernels), so expect
losses to match within ~0.005 and accuracies within a few items.

## 0. Environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.lock.txt
PYTHONPATH=src .venv/bin/python -m pytest -q     # 88 tests, CPU only
```

Use this virtualenv for every step. An older system PyTorch build without support for the GPU architecture was measured 4–5×
slower and cannot compile.

## 1. Tokenizer

Nothing to do: the frozen tokenizer is in `tokenizer/ivy-bpe-32k/`. Do not retrain it (see [Tokenizer](docs/02-tokenizer.md)).

## 2. Corpora (CPU, network access to Hugging Face)

**`mix-v0`** (100M tokens, used by the 40M pilot and the 300M LR sweep):

```bash
P="PYTHONPATH=src .venv/bin/python -m ivy_model.prepare_data --tokenizer tokenizer/ivy-bpe-32k"
$P --source fineweb-edu --out data/tokens/fineweb-edu --max-tokens 85000000
$P --source finemath    --out data/tokens/finemath    --max-tokens 15000000
mkdir -p data/tokens/mix-v0
cat data/tokens/fineweb-edu/train-*.bin data/tokens/finemath/train-*.bin > data/tokens/mix-v0/train.bin
cat data/tokens/fineweb-edu/val-*.bin   data/tokens/finemath/val-*.bin   > data/tokens/mix-v0/val.bin
```

**`mix-v1`** (2.0B tokens, ~80 minutes): see [Data → Building mix-v1](docs/03-data.md#building-mix-v1).

Notes: `--max-tokens` caps training tokens (the last document overshoots slightly). `mix-v0` uses the default seeded shuffle
stream (seed 0, buffer 10k), which needs a lot of RAM at scale but is fine at 100M tokens. Rebuilding gives the same corpora only
while the upstream Hugging Face datasets are unchanged; the shard sha256 values in each `MANIFEST.json` show whether they match.

## 3. Pretraining experiments

| Step | Command | Time (uncontended) |
|---|---|---|
| 300M LR sweep | `PYTHONPATH=src sh scripts/lr_sweep_300m.sh` | ~4 × 1.7 h |
| 300M seed replicate | `PYTHONPATH=src sh scripts/replicate_300m.sh` | ~1.7 h |
| 300M on 300M tokens | `PYTHONPATH=src sh scripts/scaling_300m.sh 300` | ~5 h |
| 40M sweep + 300M/1B-token runs | `PYTHONPATH=src sh scripts/scaling_40m.sh` | ~9 h |
| LR sweep summary | `python3 scripts/summarize_lr_sweep.py` | seconds |

Each run writes `checkpoints/<run>/{latest.pt,metrics.jsonl}` and a `DONE` marker.

Cross-size comparison on one fixed validation sample (written to `reports/common-eval-v2.jsonl`):

```bash
PYTHONPATH=src .venv/bin/python scripts/common_eval.py --out reports/common-eval-v2.jsonl \
    configs/ivy-40m.json checkpoints/ivy-40m-scale-1000M-lr-2.4e-3/latest.pt \
    configs/ivy-300m.json checkpoints/ivy-300m-scale-300M-lr-1.2e-3/latest.pt
```

The older `reports/scaling-common-eval.json` came from an ad-hoc evaluation with a different window sample; the script reproduces
its mix-slice numbers within 0.006 nats, so the conclusions are unchanged.

## 4. Tool-use fine-tuning and evaluation

```bash
G="PYTHONPATH=src .venv/bin/python -m ivy_model.sft_gen"
$G --n 20000 --seed 7 --split all   --out data/sft/v0.jsonl
$G --n 20000 --seed 7 --split train --out data/sft/v1-train.jsonl
PYTHONPATH=src .venv/bin/python -m ivy_model.cleaning check-sft data/sft/v1-train.jsonl --min-jaccard 0.6   # must report 0 overlaps
```

The two test sets in `evals/` are committed; their generation commands are in
[Tool-use fine-tuning](docs/05-fine-tuning.md#train--held-out-split).

Fine-tune and score (each ~30–60 min including scoring):

```bash
E="v1 v2 sft_indist sft_heldout"
B40=checkpoints/ivy-40m-scale-1000M-lr-2.4e-3/latest.pt
B300=checkpoints/ivy-300m-scale-300M-lr-1.2e-3/latest.pt
DATA=data/sft/v0.jsonl       EVALS="$E" PYTHONPATH=src sh scripts/sft_dryrun.sh 40m-1b            configs/ivy-40m.json  $B40  2e-4
DATA=data/sft/v1-train.jsonl EVALS="$E" PYTHONPATH=src sh scripts/sft_dryrun.sh 40m-1b-v1train    configs/ivy-40m.json  $B40  2e-4
DATA=data/sft/v0.jsonl       EVALS="$E" BSZ=8 ACCUM=8 PYTHONPATH=src sh scripts/sft_dryrun.sh 300m-300m         configs/ivy-300m.json $B300 1e-4
DATA=data/sft/v1-train.jsonl EVALS="$E" BSZ=8 ACCUM=8 PYTHONPATH=src sh scripts/sft_dryrun.sh 300m-300m-v1train configs/ivy-300m.json $B300 1e-4
```

Scores land in `checkpoints/sft-<name>/score_<set>.json`.

## 5. Kaggle dataset and notebook

```bash
python3 scripts/build_kaggle_dataset.py --owner YOUR_KAGGLE_USERNAME   # -> kaggle/ivy-experiments/ (~6.6 MB, 64 files)
python3 scripts/make_kaggle_notebook.py                                # needs nbformat; writes kaggle/ivy_experiments.ipynb
```

The builder fails if any expected result file is missing. To publish: `kaggle datasets create -p kaggle/ivy-experiments`
(the metadata marks the dataset private and its licence `unknown` until you choose one), then attach the dataset to a new Kaggle
notebook and upload `kaggle/ivy_experiments.ipynb`. Locally, run it with `IVY_DATA=kaggle/ivy-experiments jupyter execute ...`.
