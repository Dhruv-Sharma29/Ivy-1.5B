#!/bin/sh
# 40M side of the scaling study (model-size axis), on mix-v1.
#   1) LR sweep at ~98M tokens (1,500 steps) over 6e-4, 1.2e-3, 2.4e-3, 4.8e-3, two parallel lanes (a 40M run does not fill the GPU).
#   2) Pick the lr with the lowest final val loss (written to checkpoints/ivy-40m-best-lr.txt; if it is the highest or lowest
#      lr tried the optimum is outside the range: check before trusting the scaling runs).
#   3) Train that lr for 300M tokens (4,578 steps) and 1B tokens (15,259 steps), in parallel.
# Same recipe as the 300M runs (65,536 tokens/step, warmup 100, cosine to 10%, bf16, compiled, seed 1234, micro-bsz 8 x accum 4).
# Usage: PYTHONPATH=src sh scripts/scaling_40m.sh
set -u
run() {  # run LR STEPS OUTDIR
  [ -f "$3/DONE" ] && { echo "skip $3 (done)"; return; }
  echo "=== $3 start $(date -u +%FT%TZ) lr=$1 steps=$2"
  .venv/bin/python -u -m ivy_model.train --config configs/ivy-40m.json \
    --data data/tokens/mix-v1/train.bin --val-data data/tokens/mix-v1/val.bin --out "$3" \
    --steps "$2" --lr "$1" --warmup 100 --micro-bsz 8 --accum 4 --ckpt-every 500 --compile --resume --seed 1234 \
    > "$3.log" 2>&1 && touch "$3/DONE"
  echo "=== $3 end $(date -u +%FT%TZ)"
}
mkdir -p checkpoints
( run 6e-4 1500 checkpoints/ivy-40m-lr-6e-4-100M; run 2.4e-3 1500 checkpoints/ivy-40m-lr-2.4e-3-100M ) &
( run 1.2e-3 1500 checkpoints/ivy-40m-lr-1.2e-3-100M; run 4.8e-3 1500 checkpoints/ivy-40m-lr-4.8e-3-100M ) &
wait
BEST=$(python3 - <<'PY'
import json
best=None
for lr in ["6e-4","1.2e-3","2.4e-3","4.8e-3"]:
    try:
        v=[json.loads(l)["val_loss"] for l in open(f"checkpoints/ivy-40m-lr-{lr}-100M/metrics.jsonl") if "val_loss" in l][-1]
    except Exception: continue
    print(lr, v, flush=True) if False else None
    if best is None or v<best[1]: best=(lr,v)
print(best[0])
PY
)
echo "$BEST" > checkpoints/ivy-40m-best-lr.txt; echo "=== best 40M lr: $BEST"
( run "$BEST" 4578 "checkpoints/ivy-40m-scale-300M-lr-$BEST" ) &
( run "$BEST" 15259 "checkpoints/ivy-40m-scale-1000M-lr-$BEST" ) &
wait
echo "=== all done $(date -u +%FT%TZ)"
