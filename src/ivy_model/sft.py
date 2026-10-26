"""Supervised fine-tuning on chat-format tool-use data (data/sft/*.jsonl from sft_gen.py).

    PYTHONPATH=src .venv/bin/python -m ivy_model.sft --config configs/ivy-40m.json --init checkpoints/X/latest.pt \
        --data data/sft/v0.jsonl --out checkpoints/sft-X --epochs 3 --lr 1e-4

Starts from a pretraining checkpoint (weights only; fresh optimizer), trains full-parameter with loss only on the answer tokens
(chat_format.encode_example), right-padded batches sorted into length buckets. Holds out 2% of the data for an SFT-loss check
(same templates as training, so it says nothing about generalisation: use evals/run_model.py for that).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from . import chat_format as CF
from .config import ModelConfig
from .model import IvyLM
from .train import make_optimizer

PAD = 1


def load_examples(path: str, tok, max_len: int) -> list[tuple[list[int], list[int]]]:
    out, dropped = [], 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        e = CF.encode_example(tok, r["devices"], r["messages"], r["answer"], max_len)
        if e is None:
            dropped += 1
        else:
            out.append(e)
    if dropped:
        print(f"dropped {dropped} examples longer than {max_len} tokens")
    return out


def batches(examples, bsz: int, rng: random.Random):
    """Length-bucketed batches (shuffled): sort chunks of 50 batches by length, then shuffle the batch order."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    chunk = bsz * 50
    out = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: len(examples[j][0]))
        out += [part[k:k + bsz] for k in range(0, len(part), bsz)]
    rng.shuffle(out)
    return out


def collate(examples, ids, device):
    m = max(len(examples[j][0]) for j in ids)
    x = torch.full((len(ids), m), PAD, dtype=torch.long)
    y = torch.full((len(ids), m), -100, dtype=torch.long)
    for r, j in enumerate(ids):
        a, b = examples[j]
        x[r, :len(a)] = torch.tensor(a)
        y[r, :len(b)] = torch.tensor(b)
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, examples, bsz, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), bsz):
        x, y = collate(examples, list(range(i, min(i + bsz, len(examples)))), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, labels=y, return_logits=False)
        k = int((y[:, 1:] != -100).sum())
        tot, n = tot + loss.item() * k, n + k
    model.train()
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", required=True, help="pretraining checkpoint (latest.pt)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/ivy-bpe-32k/tokenizer.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--bsz", type=int, default=16, help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    cfg = ModelConfig.load(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = Tokenizer.from_file(a.tokenizer)
    ex = load_examples(a.data, tok, a.max_len)
    rng.shuffle(ex)
    nval = max(1, len(ex) // 50)
    val, train = ex[:nval], ex[nval:]
    model = IvyLM(cfg).to(device)
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    if ck["cfg_digest"] != cfg.digest():
        raise SystemExit("checkpoint/config mismatch")
    model.load_state_dict(ck["model"])
    model.train()
    opt = make_optimizer(model, a.lr, wd=0.0)
    steps_per_epoch = math.ceil(len(train) / (a.bsz * a.accum))
    total = steps_per_epoch * a.epochs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "sft_metrics.jsonl").open("w")
    print(f"{cfg.name}: {len(train)} train / {len(val)} val examples, {total} steps, init val loss {eval_loss(model, val, a.bsz, device):.4f}", flush=True)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        bl = batches(train, a.bsz, rng)
        for s in range(0, len(bl) - a.accum + 1, a.accum):
            lr = a.lr * (step + 1) / a.warmup if step < a.warmup else a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, total - a.warmup))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for ids in bl[s:s + a.accum]:
                x, y = collate(train, ids, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, loss = model(x, labels=y, return_logits=False)
                (loss / a.accum).backward()
                tot += loss.item() / a.accum
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == total:
                row = {"step": step, "epoch": ep, "loss": tot, "gnorm": float(gn), "lr": lr, "sec": round(time.time() - t0)}
                log.write(json.dumps(row) + "\n"); log.flush()
                print(f"epoch {ep} step {step}/{total} loss {tot:.4f} gnorm {gn:.2f} lr {lr:.2e}", flush=True)
        v = eval_loss(model, val, a.bsz, device)
        log.write(json.dumps({"epoch_end": ep, "val_loss": v}) + "\n"); log.flush()
        print(f"== epoch {ep} done, SFT val loss {v:.4f}", flush=True)
    torch.save({"model": model.state_dict(), "cfg_digest": cfg.digest(), "step": step, "base": a.init, "sft_data": a.data}, out / "latest.pt")
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
