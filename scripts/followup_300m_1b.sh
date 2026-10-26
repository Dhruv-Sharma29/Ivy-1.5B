#!/bin/sh
# Does a better-trained 300M base transfer better? Runs after scripts/scaling_300m.sh 1000 has finished (waits for its DONE).
#   1) common eval of the new base (compare with 40M@1B tokens in reports/common-eval-v2.jsonl)
#   2) 1-epoch SFT (lr 1e-4, the best 300M recipe) on BOTH template splits, scored on that split's held-out set and v2,
#      for comparison with the existing 40M 1-epoch runs (sft-40m-1b-v1train-1ep, sft-40m-1b-s2-1ep).
# Usage: PYTHONPATH=src sh scripts/followup_300m_1b.sh
set -u
BASE=checkpoints/ivy-300m-scale-1000M-lr-1.2e-3
until [ -f "$BASE/DONE" ]; do sleep 300; done
echo "=== follow-up start $(date -u +%FT%TZ)"
.venv/bin/python scripts/common_eval.py --out reports/common-eval-v2.jsonl configs/ivy-300m.json "$BASE/latest.pt"
EPOCHS=1 BSZ=8 ACCUM=8 DATA=data/sft/v1-train.jsonl EVALS="sft_heldout sft_indist v2" \
  sh scripts/sft_dryrun.sh 300m-1b-v1train-1ep configs/ivy-300m.json "$BASE/latest.pt" 1e-4
EPOCHS=1 BSZ=8 ACCUM=8 DATA=data/sft/v1-train-s2.jsonl EVALS="sft_heldout_s2 v2" \
  sh scripts/sft_dryrun.sh 300m-1b-s2-1ep configs/ivy-300m.json "$BASE/latest.pt" 1e-4
echo "=== follow-up end $(date -u +%FT%TZ)"
