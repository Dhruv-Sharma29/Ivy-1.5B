"""Measured training throughput on synthetic tokens (Phase 4 gate). Never extrapolate across sizes."""
from __future__ import annotations

import argparse
import json
import time

import torch

from .config import ModelConfig
from .model import IvyLM
from .train import add_runtime_args, configure_runtime, make_optimizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--micro-bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--full-logits", action="store_true", help="baseline: materialise fp32 logits")
    add_runtime_args(ap)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    cfg = ModelConfig.load(a.config)
    dev = torch.device("cuda")
    model = IvyLM(cfg).to(dev)
    model.grad_checkpointing = a.grad_ckpt
    amp = configure_runtime(model, a.precision, a.attn_block)
    opt = make_optimizer(model, 3e-4)
    if a.compile:
        model.compile_blocks()
    x = torch.randint(0, cfg.vocab_size, (a.micro_bsz, a.seq_len), device=dev)
    tok_per_step = a.micro_bsz * a.seq_len * a.accum

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                _, loss = model(x, labels=x, return_logits=a.full_logits)
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(a.warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    res = {
        "config": cfg.name, "params": model.num_params(), "micro_bsz": a.micro_bsz, "accum": a.accum,
        "seq_len": a.seq_len, "grad_ckpt": a.grad_ckpt, "compile": a.compile, "full_logits": a.full_logits,
        "precision": a.precision, "attn_block": a.attn_block,
        "tokens_per_s": round(tok_per_step * a.steps / dt, 1),
        "peak_torch_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "note": "synthetic tokens; excludes data loading, checkpointing, eval",
    }
    print(json.dumps(res, indent=2))
    if a.json_out:
        open(a.json_out, "w").write(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
