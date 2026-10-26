"""Summarise the 300M LR sweep from checkpoints/ivy-300m-lr-*/metrics.jsonl.

    .venv/bin/python scripts/summarize_lr_sweep.py            # prints a table, writes reports/lr-sweep-300m.json

Reports per run: val loss at each checkpoint, mean train loss over the last 50 steps, max gradient norm, number of
step-to-step loss jumps above 0.5 after warmup (spike indicator), and whether any step was non-finite (the trainer
raises on NaN/inf, so a crashed run shows as missing steps). Differences below the run-to-run noise are NOT resolved
by a single seed; see docs/07-results.md for the noise estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tr = {r["step"]: r for r in rows if "loss" in r}
    val = {r["step"]: r["val_loss"] for r in rows if "val_loss" in r}
    steps = sorted(tr)
    last = [tr[s]["loss"] for s in steps[-50:]]
    jumps = sum(1 for a, b in zip(steps, steps[1:]) if a >= 100 and tr[b]["loss"] - tr[a]["loss"] > 0.5)
    return {"steps": len(steps), "val": val, "train_last50": sum(last) / max(1, len(last)),
            "max_gnorm": max((tr[s]["gnorm"] for s in steps), default=float("nan")), "loss_jumps_gt_0.5": jumps,
            "done": (path.parent / "DONE").exists()}


def main() -> None:
    runs = {}
    for d in sorted((ROOT / "checkpoints").glob("ivy-300m-lr-*")):
        m = d / "metrics.jsonl"
        if m.exists():
            runs[d.name.removeprefix("ivy-300m-lr-")] = load(m)
    runs = dict(sorted(runs.items(), key=lambda kv: float(kv[0])))
    marks = sorted({s for r in runs.values() for s in r["val"] if s})
    print(f"{'peak LR':>8} {'steps':>6} " + " ".join(f"val@{s:<5}" for s in marks) + "  train50  max|g|  jumps  done")
    for lr, r in runs.items():
        vals = " ".join(f"{r['val'].get(s, float('nan')):9.4f}" for s in marks)
        print(f"{lr:>8} {r['steps']:>6} {vals}  {r['train_last50']:7.4f} {r['max_gnorm']:6.2f} {r['loss_jumps_gt_0.5']:6d}  {r['done']}")
    done = {k: v for k, v in runs.items() if v["done"]}
    if done:
        best = min(done, key=lambda k: done[k]["val"][max(done[k]["val"])])
        print("best final val loss among finished runs:", best, round(done[best]["val"][max(done[best]["val"])], 4))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "lr-sweep-300m.json").write_text(json.dumps(runs, indent=1) + "\n")


if __name__ == "__main__":
    main()
