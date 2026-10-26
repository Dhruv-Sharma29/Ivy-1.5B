"""Soak test (Phase 4 gate): real train loop + real checkpoints on synthetic tokens.

  soak     : train for --minutes in windows; log tok/s, memory, power/temp, checkpoint save time;
             mid-run, save a checkpoint, record the next steps' losses, restore in-process and replay.
  resume   : fresh process, load the last checkpoint, time restart-to-first-step, run a few steps.

Synthetic uniform-random tokens: loss stays near ln(vocab), so this validates throughput, stability,
checkpointing and resume determinism, NOT learning. Data loading (memmap sampling) is included.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import IvyLM
from .train import TokenShard, load_checkpoint, make_optimizer, save_checkpoint, train_steps


def gpu_stats() -> dict:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        p, t, c = [x.strip() for x in o.strip().split(",")]
        return {"power_w": p, "temp_c": t, "sm_clock_mhz": c}
    except Exception as e:  # noqa: BLE001 - telemetry must never kill the soak
        return {"telemetry_error": str(e)[:80]}


def mem_used_gb() -> float:
    kv = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return round((kv["MemTotal"] - kv["MemAvailable"]) / 2**20, 1)


def setup(a):
    cfg = ModelConfig.load(a.config)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    model = IvyLM(cfg).to(dev)
    if a.compile:
        model.compile_blocks()
    opt = make_optimizer(model, a.lr)
    data_path = Path(a.data)
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.random.default_rng(0).integers(0, cfg.vocab_size, a.synthetic_tokens, dtype=np.uint16).tofile(data_path)
    return cfg, dev, model, opt, TokenShard(data_path, a.seq_len)


def kw(a, dev):
    return dict(total_steps=a.total_steps, peak_lr=a.lr, warmup=a.warmup, micro_bsz=a.micro_bsz, accum=a.accum,
                seed=a.seed, device=dev, log=lambda *_: None)


def soak(a) -> None:
    cfg, dev, model, opt, data = setup(a)
    ckpt = Path(a.out) / "latest.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = {"config": cfg.name, "params": model.num_params(), "windows": [], "events": []}
    log = lambda m: print(m, flush=True)  # noqa: E731
    step, tokens, all_losses = 0, 0, []
    t_start = time.time()
    deadline = t_start + a.minutes * 60
    verified = False
    torch.cuda.reset_peak_memory_stats()
    while time.time() < deadline and step < a.total_steps:
        t0 = time.time()
        end = min(step + a.window_steps, a.total_steps)
        losses, tokens = train_steps(model, opt, data, cfg, start_step=step, end_step=end, tokens_seen=tokens, **kw(a, dev))
        torch.cuda.synchronize()
        dt = time.time() - t0
        step = end
        all_losses += losses
        if not all(np.isfinite(losses)):
            raise FloatingPointError("non-finite loss in window")
        t1 = time.time()
        save_checkpoint(ckpt, model, opt, step, tokens, cfg)
        save_s = time.time() - t1
        w = {"step": step, "elapsed_min": round((time.time() - t_start) / 60, 1),
             "tok_per_s": round(len(losses) * a.micro_bsz * a.seq_len * a.accum / dt, 0),
             "loss_mean": round(float(np.mean(losses)), 4), "ckpt_save_s": round(save_s, 1),
             "ckpt_gb": round(ckpt.stat().st_size / 2**30, 2),
             "peak_torch_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
             "sys_mem_used_gb": mem_used_gb(), **gpu_stats()}
        rec["windows"].append(w)
        log(json.dumps(w))
        # In-process resume determinism check once, roughly mid-run.
        if not verified and time.time() - t_start > a.minutes * 30:
            verified = True
            ref, _ = train_steps(model, opt, data, cfg, start_step=step, end_step=step + a.replay_steps,
                                 tokens_seen=tokens, **kw(a, dev))
            t2 = time.time()
            rstep, rtok = load_checkpoint(ckpt, model, opt, cfg)
            for s in opt.state.values():  # move optimizer state back to GPU (loaded on CPU)
                for k, v in s.items():
                    if torch.is_tensor(v):
                        s[k] = v.to(dev)
            load_s = time.time() - t2
            replay, _ = train_steps(model, opt, data, cfg, start_step=rstep, end_step=rstep + a.replay_steps,
                                    tokens_seen=rtok, **kw(a, dev))
            diff = float(np.max(np.abs(np.array(ref) - np.array(replay))))
            ev = {"event": "inprocess_resume_check", "at_step": rstep, "ckpt_load_s": round(load_s, 1),
                  "ref": [round(x, 5) for x in ref], "replay": [round(x, 5) for x in replay], "max_abs_diff": diff}
            rec["events"].append(ev)
            log(json.dumps(ev))
            step, tokens = rstep + a.replay_steps, rtok + a.replay_steps * a.micro_bsz * a.seq_len * a.accum
            save_checkpoint(ckpt, model, opt, step, tokens, cfg)
    toks = [w["tok_per_s"] for w in rec["windows"]]
    rec["summary"] = {"steps": step, "minutes": round((time.time() - t_start) / 60, 1),
                      "tok_per_s_first": toks[0] if toks else None, "tok_per_s_last": toks[-1] if toks else None,
                      "tok_per_s_min": min(toks) if toks else None, "tok_per_s_max": max(toks) if toks else None,
                      "tokens_seen": tokens}
    Path(a.report).write_text(json.dumps(rec, indent=2) + "\n")
    log("SOAK DONE " + json.dumps(rec["summary"]))


def resume(a) -> None:
    t0 = time.time()
    cfg, dev, model, opt, data = setup(a)
    t_setup = time.time() - t0
    step, tokens = load_checkpoint(Path(a.out) / "latest.pt", model, opt, cfg)
    for s in opt.state.values():
        for k, v in s.items():
            if torch.is_tensor(v):
                s[k] = v.to(dev)
    t_load = time.time() - t0
    losses, _ = train_steps(model, opt, data, cfg, start_step=step, end_step=step + 3, tokens_seen=tokens, **kw(a, dev))
    torch.cuda.synchronize()
    res = {"event": "fresh_process_resume", "resumed_step": step, "setup_s": round(t_setup, 1),
           "to_loaded_s": round(t_load, 1), "to_first_3_steps_s": round(time.time() - t0, 1),
           "losses": [round(x, 4) for x in losses], "finite": bool(all(np.isfinite(losses)))}
    print("RESUME " + json.dumps(res), flush=True)
    Path(a.report).write_text(json.dumps(res, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["soak", "resume"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--data", default="data/synthetic/soak.bin")
    ap.add_argument("--synthetic-tokens", type=int, default=50_000_000)
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--window-steps", type=int, default=50)
    ap.add_argument("--replay-steps", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=100000)
    ap.add_argument("--micro-bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    soak(a) if a.mode == "soak" else resume(a)


if __name__ == "__main__":
    main()
