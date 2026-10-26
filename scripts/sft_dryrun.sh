#!/bin/sh
# SFT dry run: fine-tune a pretrained checkpoint on $DATA (default data/sft/v0.jsonl), then score each eval set in $EVALS
# (default "v1 v2"; a name X means evals/scenarios_X.jsonl if it exists, else evals/X.jsonl); EPOCHS (default 3). Usage:
#   PYTHONPATH=src sh scripts/sft_dryrun.sh NAME CONFIG INIT_CKPT LR
#   DATA=data/sft/v1-train.jsonl EVALS="v1 v2 sft_indist sft_heldout" PYTHONPATH=src sh scripts/sft_dryrun.sh ...
set -u
NAME=$1; CFG=$2; INIT=$3; LR=$4
OUT=checkpoints/sft-$NAME
.venv/bin/python -u -m ivy_model.sft --config "$CFG" --init "$INIT" --data "${DATA:-data/sft/v0.jsonl}" --out "$OUT" --epochs "${EPOCHS:-3}" --lr "$LR" --bsz "${BSZ:-16}" --accum "${ACCUM:-4}" > "$OUT.log" 2>&1
for V in ${EVALS:-v1 v2}; do
  SC=evals/scenarios_$V.jsonl; [ -f "$SC" ] || SC=evals/$V.jsonl
  .venv/bin/python evals/run_model.py --config "$CFG" --ckpt "$OUT/latest.pt" --scenarios "$SC" --out "$OUT/outputs_$V.jsonl" >> "$OUT.log" 2>&1
  .venv/bin/python evals/score.py --scenarios "$SC" --outputs "$OUT/outputs_$V.jsonl" > "$OUT/score_$V.json"
done
echo done >> "$OUT.log"
