"""Validation loss of pretrained checkpoints on ONE fixed set of windows, so runs of any size and batch setting are comparable.

    PYTHONPATH=src .venv/bin/python scripts/common_eval.py --out reports/common-eval-v2.jsonl \
        configs/ivy-40m.json checkpoints/ivy-40m-scale-1000M-lr-2.4e-3/latest.pt [CONFIG CKPT ...]

Slices: web = FineWeb-Edu val, math = FineMath val (both from data/tokens/v1), mix = mix-v1 val. Each slice is scored on
--batches x --bsz windows of --seq-len tokens, drawn with TokenShard.batch(seed=--seed, step=0, micro=i), i.e. independent of
any training run's micro-batch size (in-training validation is not: its windows depend on --micro-bsz). Appends one JSON line per
checkpoint. Note: reports/scaling-common-eval.json came from an earlier ad-hoc evaluation with a different window sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model.train import TokenShard  # noqa: E402

SLICES = {"web": "data/tokens/v1/fineweb-edu/val-00000.bin", "math": "data/tokens/v1/finemath/val-00000.bin",
          "mix": "data/tokens/mix-v1/val.bin"}


@torch.no_grad()
def slice_loss(model, shard: TokenShard, batches: int, bsz: int, seed: int, device) -> float:
    total = 0.0
    for i in range(batches):
        x = shard.batch(seed, 0, i, bsz, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, labels=x, return_logits=False)
        total += loss.item()
    return total / batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="CONFIG CKPT [CONFIG CKPT ...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=96)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()
    if len(a.pairs) % 2:
        sys.exit("pairs must be CONFIG CKPT pairs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards = {k: TokenShard(p, a.seq_len) for k, p in SLICES.items()}
    for cfg_path, ckpt in zip(a.pairs[::2], a.pairs[1::2]):
        cfg = ModelConfig.load(cfg_path)
        model = IvyLM(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        row = {"ckpt": ckpt, "model": cfg.name, "params": model.num_params(), "tokens_seen": ck.get("tokens_seen"),
               **{k: round(slice_loss(model, s, a.batches, a.bsz, a.seed, device), 4) for k, s in shards.items()},
               "windows": a.batches * a.bsz, "seq_len": a.seq_len, "seed": a.seed}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
