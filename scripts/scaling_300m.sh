#!/bin/sh
# Scaling check at 300M: train on N tokens of mix-v1 (2.0B-token corpus) at the best sweep LR (1.2e-3).
# Same recipe as the LR sweep (65,536 tokens/step, warmup 100, cosine to 10%, bf16, compiled), only the number of steps changes,
# so the cosine schedule spans the whole run (separate runs are needed per token budget, not one run with checkpoints).
# micro-bsz 4 x accum 8 (not 8 x 4): same batch, ~13 GB instead of ~20 GB on a GPU/RAM that other users share.
# Usage: PYTHONPATH=src sh scripts/scaling_300m.sh 300   # tokens in millions (300 -> 4,578 steps)
set -u
M=${1:?tokens in millions}
STEPS=$(python3 -c "print(round($M*1e6/65536))")
OUT=checkpoints/ivy-300m-scale-${M}M-lr-1.2e-3
[ -f "$OUT/DONE" ] && { echo "already done"; exit 0; }
echo "=== scaling ${M}M start $(date -u +%FT%TZ) steps=$STEPS"
.venv/bin/python -u -m ivy_model.train --config configs/ivy-300m.json \
  --data data/tokens/mix-v1/train.bin --val-data data/tokens/mix-v1/val.bin --out "$OUT" \
  --steps "$STEPS" --lr 1.2e-3 --warmup 100 --micro-bsz 4 --accum 8 --ckpt-every 500 --compile --resume --seed 1234 \
  && touch "$OUT/DONE"
echo "=== scaling ${M}M end $(date -u +%FT%TZ) exit=$?"
