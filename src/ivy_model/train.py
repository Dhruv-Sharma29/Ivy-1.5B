"""Minimal single-GPU training loop with full-state checkpointing.

Batches are a pure function of (seed, step), so a resumed run sees exactly the batches an
uninterrupted run would. That property is what tests/test_resume.py verifies.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import IvyLM


def configure_runtime(model: IvyLM, precision: str = "bf16", attn_block: int = 0) -> bool:
    """Apply opt-in speed options; returns whether to use bf16 autocast.

    precision "bf16": bf16 autocast (default). "tf32": fp32 weights/activations with TF32 matmuls,
    which was faster than bf16 on GB10 with the nv25.02 torch build (see docs/10-implementation-notes.md); stability is
    not yet validated, so compare loss curves against bf16 before a long run.
    attn_block > 0: blockwise causal attention with that query-block size (512 measured best)."""
    if precision not in ("bf16", "tf32"):
        raise ValueError(f"unknown precision {precision!r}")
    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.set_attn_block_size(attn_block)
    return precision == "bf16"


def add_runtime_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--precision", choices=["bf16", "tf32"], default="bf16",
                    help="tf32 is an OLD-STACK workaround: 39%% slower than bf16 on torch 2.13+cu130")
    ap.add_argument("--attn-block", type=int, default=0,
                    help="blockwise causal attention block size (0 = off). OLD-STACK workaround for a missing "
                         "flash kernel: 16%% slower on torch 2.13+cu130, where flash attention works")


def lr_at(step: int, total: int, peak: float, warmup: int, floor_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))


class TokenShard:
    """uint16 token file (vocab < 65536). Deterministic sampling by (seed, step, micro index)."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{path} has too few tokens for seq_len={seq_len}")

    def batch(self, seed: int, step: int, micro: int, bsz: int, device) -> torch.Tensor:
        rng = np.random.default_rng([seed, step, micro])
        starts = rng.integers(0, len(self.data) - self.seq_len, size=bsz)
        x = np.stack([self.data[s : s + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device)


def save_checkpoint(path: str | Path, model, opt, step: int, tokens_seen: int, cfg: ModelConfig) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "cfg_digest": cfg.digest(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: never leave the only checkpoint half-written


def load_checkpoint(path: str | Path, model, opt, cfg: ModelConfig) -> tuple[int, int]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["cfg_digest"] != cfg.digest():
        raise ValueError("checkpoint was produced by a different model config")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["torch_rng"])
    if ck["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck["step"], ck["tokens_seen"]


def make_optimizer(model: IvyLM, lr: float, wd: float = 0.1, betas=(0.9, 0.95)):
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas,
    )


@torch.no_grad()
def evaluate(model, shard: "TokenShard", *, n_batches: int, bsz: int, device, amp: bool = True) -> float:
    """Mean loss over a FIXED set of validation windows (same windows every call, so values are comparable)."""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(n_batches):
        x = shard.batch(0xA11, 0, i, bsz, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            _, loss = model(x, labels=x, return_logits=False)
        total += loss.item()
    model.train(was_training)
    return total / n_batches


def train_steps(model, opt, data, cfg, *, start_step, end_step, total_steps, peak_lr, warmup,
                micro_bsz, accum, seed, device, tokens_seen=0, clip=1.0, amp=True, log=print, history=None):
    losses = []
    model.train()
    for step in range(start_step, end_step):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, peak_lr, warmup)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for m in range(accum):
            x = data.batch(seed, step, m, micro_bsz, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                _, loss = model(x, labels=x, return_logits=False)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / accum).backward()
            total += loss.item() / accum
            tokens_seen += x.numel()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(total)
        if history is not None:
            history.append({"step": step + 1, "loss": total, "gnorm": float(gn), "tokens": tokens_seen,
                            "lr": opt.param_groups[0]["lr"]})
        log(f"step {step + 1}/{total_steps} loss {total:.4f} gnorm {gn:.2f} lr {opt.param_groups[0]['lr']:.2e}")
    return losses, tokens_seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="uint16 token .bin file")
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--micro-bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val-data", help="uint16 token .bin file for validation loss")
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--compile", action="store_true", help="torch.compile each block (needs the cu130 stack)")
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute each block in backward (less activation memory, ~20%% slower)")
    add_runtime_args(ap)
    a = ap.parse_args()

    cfg = ModelConfig.load(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    model = IvyLM(cfg).to(device)
    amp = configure_runtime(model, a.precision, a.attn_block)
    model.grad_checkpointing = a.grad_ckpt
    if a.compile:
        model.compile_blocks()
    opt = make_optimizer(model, a.lr)
    data = TokenShard(a.data, a.seq_len)
    val = TokenShard(a.val_data, a.seq_len) if a.val_data else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "latest.pt"
    step = tokens = 0
    if a.resume and ckpt.exists():
        step, tokens = load_checkpoint(ckpt, model, opt, cfg)
        print(f"resumed at step {step}, {tokens:,} tokens")
    print(f"{cfg.name}: {model.num_params():,} params on {device}")
    metrics = out / "metrics.jsonl"

    def record(row: dict) -> None:
        with metrics.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def run_val(at_step: int) -> None:
        if val is not None:
            v = evaluate(model, val, n_batches=a.val_batches, bsz=a.micro_bsz, device=device, amp=amp)
            record({"step": at_step, "val_loss": v})
            print(f"val loss @ step {at_step}: {v:.4f}")

    if step == 0:
        run_val(0)
    t0 = time.time()
    t_start_step, t_start_tokens = step, tokens
    while step < a.steps:
        end = min(a.steps, step + a.ckpt_every)
        hist: list[dict] = []
        _, tokens = train_steps(model, opt, data, cfg, start_step=step, end_step=end, total_steps=a.steps,
                                peak_lr=a.lr, warmup=a.warmup, micro_bsz=a.micro_bsz, accum=a.accum,
                                seed=a.seed, device=device, tokens_seen=tokens, amp=amp, history=hist,
                                log=lambda m: print(m, flush=True))
        for h in hist:
            record(h)
        step = end
        save_checkpoint(ckpt, model, opt, step, tokens, cfg)
        run_val(step)
        print(f"checkpoint @ step {step}; {(tokens - t_start_tokens) / (time.time() - t0):,.0f} tok/s (this session)", flush=True)


if __name__ == "__main__":
    main()
