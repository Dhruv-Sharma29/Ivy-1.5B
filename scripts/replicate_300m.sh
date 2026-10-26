#!/bin/sh
# Seed replicate of the best 300M LR from the sweep (1.2e-3), to measure single-seed noise. Same data and schedule,
# different seed (changes init AND data order). Compare with checkpoints/ivy-300m-lr-1.2e-3 (seed 1234, val 3.6440).
# Usage: PYTHONPATH=src sh scripts/replicate_300m.sh
set -u
OUT=checkpoints/ivy-300m-rep-seed4321-lr-1.2e-3
[ -f "$OUT/DONE" ] && { echo "already done"; exit 0; }
echo "=== replicate start $(date -u +%FT%TZ)"
.venv/bin/python -u -m ivy_model.train --config configs/ivy-300m.json \
  --data data/tokens/mix-v0/train.bin --val-data data/tokens/mix-v0/val.bin --out "$OUT" \
  --steps 1500 --lr 1.2e-3 --warmup 100 --micro-bsz 8 --accum 4 --ckpt-every 250 --compile --resume --seed 4321 \
  && touch "$OUT/DONE"
echo "=== replicate end $(date -u +%FT%TZ) exit=$?"
