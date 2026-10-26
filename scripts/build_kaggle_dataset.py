"""Assemble the Kaggle dataset "Ivy experiments" from local results (no weights, no corpus text).

    python3 scripts/build_kaggle_dataset.py [--out kaggle/ivy-experiments] [--owner KAGGLE_USERNAME]

Copies training curves (metrics.jsonl), SFT scores and model outputs, the JSON reports and the eval sets into one folder with a
run manifest (runs.csv), a README and dataset-metadata.json for `kaggle datasets create -p <out>`. Fails if any listed file is
missing, so the dataset always matches the manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name, kind, params_M, corpus, tokens_M, peak_lr, seed, base (SFT only), sft_data (SFT only), note
RUNS = [
    ("ivy-40m-pilot", "pretrain", 40.9, "mix-v0", 98.3, 6e-4, 1234, "", "", "pipeline pilot; LR not swept"),
    ("ivy-40m-lr-6e-4-100M", "pretrain", 40.9, "mix-v1", 98.3, 6e-4, 1234, "", "", "40M LR sweep"),
    ("ivy-40m-lr-1.2e-3-100M", "pretrain", 40.9, "mix-v1", 98.3, 1.2e-3, 1234, "", "", "40M LR sweep"),
    ("ivy-40m-lr-2.4e-3-100M", "pretrain", 40.9, "mix-v1", 98.3, 2.4e-3, 1234, "", "", "40M LR sweep (best)"),
    ("ivy-40m-lr-4.8e-3-100M", "pretrain", 40.9, "mix-v1", 98.3, 4.8e-3, 1234, "", "", "40M LR sweep"),
    ("ivy-40m-scale-300M-lr-2.4e-3", "pretrain", 40.9, "mix-v1", 300.0, 2.4e-3, 1234, "", "", "40M token scaling"),
    ("ivy-40m-scale-1000M-lr-2.4e-3", "pretrain", 40.9, "mix-v1", 1000.0, 2.4e-3, 1234, "", "", "40M token scaling; base of sft-40m-*"),
    ("ivy-300m-lr-3e-4", "pretrain", 307.3, "mix-v0", 98.3, 3e-4, 1234, "", "", "300M LR sweep"),
    ("ivy-300m-lr-6e-4", "pretrain", 307.3, "mix-v0", 98.3, 6e-4, 1234, "", "", "300M LR sweep"),
    ("ivy-300m-lr-1.2e-3", "pretrain", 307.3, "mix-v0", 98.3, 1.2e-3, 1234, "", "", "300M LR sweep (best)"),
    ("ivy-300m-lr-2.4e-3", "pretrain", 307.3, "mix-v0", 98.3, 2.4e-3, 1234, "", "", "300M LR sweep"),
    ("ivy-300m-rep-seed4321-lr-1.2e-3", "pretrain", 307.3, "mix-v0", 98.3, 1.2e-3, 4321, "", "", "seed replicate of the best 300M LR"),
    ("ivy-300m-scale-300M-lr-1.2e-3", "pretrain", 307.3, "mix-v1", 300.0, 1.2e-3, 1234, "", "", "300M token scaling; base of sft-300m-*"),
    ("sft-40m-1b", "sft", 40.9, "", "", 2e-4, 0, "ivy-40m-scale-1000M-lr-2.4e-3", "v0", "all templates (incl. held-out ones)"),
    ("sft-40m-1b-v1train", "sft", 40.9, "", "", 2e-4, 0, "ivy-40m-scale-1000M-lr-2.4e-3", "v1-train", "held-out templates never seen"),
    ("sft-300m-300m", "sft", 307.3, "", "", 1e-4, 0, "ivy-300m-scale-300M-lr-1.2e-3", "v0", "all templates (incl. held-out ones)"),
    ("sft-300m-300m-v1train", "sft", 307.3, "", "", 1e-4, 0, "ivy-300m-scale-300M-lr-1.2e-3", "v1-train", "held-out templates never seen"),
]
REPORTS = ["lr-sweep-300m.json", "scaling-common-eval.json", "scaling-fit-rough.json", "scaling-300m-same-val.json", "pilot-40m.json",
           "baseline-unigram-mix-v0.json", "tokenizer-candidates-eval.json"]
EVALS = ["tools.json", "scenarios_v1.jsonl", "scenarios_v2.jsonl", "sft_indist.jsonl", "sft_heldout.jsonl"]
EVAL_SETS = ["v1", "v2", "sft_indist", "sft_heldout"]

README = """# Ivy experiments: small-LM pretraining and tool-use SFT results

Results from building Ivy, a small from-scratch language model for device-control tool calls. Code, configs and the full build
log (every caveat): https://github.com/Dhruv-Sharma29/Ivy-1.5b . The companion notebook reproduces every chart from these files.

**No model weights and no training text are included**: only measurements, model outputs on synthetic eval prompts, and the eval
sets themselves.

## Files
- `runs.csv`: one row per run (model size, corpus, tokens, peak LR, seed; for SFT runs the base checkpoint and SFT data).
- `runs/<run>/metrics.jsonl` (pretraining): one line per step `{step, loss, gnorm, tokens, lr}` plus `{step, val_loss}` lines.
  Batch = 65,536 tokens/step, seq 2048, warmup 100, cosine to 10%, bf16. Validation sets differ between corpora (mix-v0 vs
  mix-v1), so compare val losses only within one corpus, or use `reports/scaling-common-eval.json` (one common val set).
- `runs/<sft-run>/sft_metrics.jsonl`: SFT loss curve; `score_<set>.json`: scorer output; `outputs_<set>.jsonl`: raw greedy
  model output per scenario id.
- `evals/`: the tool catalogue and the four eval sets. v1 (225) and v2 (260) are hand-designed scenarios; `sft_indist` (600)
  are new combinations of training templates; `sft_heldout` (600) use templates and slot values held out from `v1-train`.
- `reports/`: summary JSONs (LR sweep, common-eval scaling numbers, a rough scaling fit, the 40M pilot, tokenizer comparison).

## Corpora
- mix-v0: ~100M tokens, 85% FineWeb-Edu / 15% FineMath (shuffled stream). mix-v1: 2.0B tokens, same mix, balanced per-file sampling.
- Source datasets: HuggingFaceFW/fineweb-edu (sample-10BT) and HuggingFaceTB/finemath (finemath-4plus), both ODC-By.

## Caveats
Single seed per run except one replicate; the shared GPU made wall-clock times noisy (losses are unaffected); eval sets and
templates share one author, so absolute tool-use scores are optimistic; the held-out set is the fairer measure.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "kaggle" / "ivy-experiments"))
    ap.add_argument("--owner", default="KAGGLE_USERNAME", help="your Kaggle username (for dataset-metadata.json)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "runs").mkdir(parents=True)

    def cp(src: Path, dst: Path) -> None:
        if not src.is_file():
            raise SystemExit(f"missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for name, kind, *_ in RUNS:
        src = ROOT / "checkpoints" / name
        if kind == "pretrain":
            cp(src / "metrics.jsonl", out / "runs" / name / "metrics.jsonl")
        else:
            cp(src / "sft_metrics.jsonl", out / "runs" / name / "sft_metrics.jsonl")
            for v in EVAL_SETS:
                cp(src / f"score_{v}.json", out / "runs" / name / f"score_{v}.json")
                cp(src / f"outputs_{v}.jsonl", out / "runs" / name / f"outputs_{v}.jsonl")
    for r in REPORTS:
        cp(ROOT / "reports" / r, out / "reports" / r)
    for e in EVALS:
        cp(ROOT / "evals" / e, out / "evals" / e)

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "kind", "params_M", "corpus", "tokens_M", "peak_lr", "seed", "base", "sft_data", "note"])
        w.writerows(RUNS)
    (out / "README.md").write_text(README)
    meta = {"title": "Ivy experiments: small-LM pretraining and tool-use SFT", "id": f"{a.owner}/ivy-experiments",
            "subtitle": "LR sweeps, scaling runs and template-held-out tool-use evals for 40M/300M models",
            "description": README, "isPrivate": True, "licenses": [{"name": "unknown"}],
            "keywords": ["nlp", "deep learning", "large language models"]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
