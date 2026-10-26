#!/bin/sh
# 300M learning-rate comparison at equal tokens (Phase 7 pilot). Sequential on a shared GPU; ~97 min per run.
# Same data order for every run (seed 1234, batches are a pure function of step), same init (seed), only the peak LR changes.
# Usage: PYTHONPATH=src sh scripts/lr_sweep_300m.sh
set -u
for LR in 6e-4 1.2e-3 3e-4 2.4e-3; do
  OUT=checkpoints/ivy-300m-lr-$LR
  [ -f "$OUT/DONE" ] && { echo "skip $LR (done)"; continue; }
  echo "=== LR $LR start $(date -u +%FT%TZ)"
  .venv/bin/python -u -m ivy_model.train --config configs/ivy-300m.json \
    --data data/tokens/mix-v0/train.bin --val-data data/tokens/mix-v0/val.bin --out "$OUT" \
    --steps 1500 --lr "$LR" --warmup 100 --micro-bsz 8 --accum 4 --ckpt-every 250 --compile --resume \
    && touch "$OUT/DONE"
  echo "=== LR $LR end $(date -u +%FT%TZ) exit=$?"
done
