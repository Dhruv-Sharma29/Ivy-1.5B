"""Assemble the Kaggle datasets for Ivy from local results (no weights, no pretraining text).

    python3 scripts/build_kaggle_dataset.py --which results --owner KAGGLE_USERNAME   # -> kaggle/ivy-experiments
    python3 scripts/build_kaggle_dataset.py --which sft --owner KAGGLE_USERNAME       # -> kaggle/ivy-tool-use-sft

results: training curves (metrics.jsonl), SFT scores and model outputs, the JSON reports and the eval sets, with a run manifest
(runs.csv). sft: the template-generated tool-use training data and the eval sets. Each folder gets a README and the
dataset-metadata.json that `kaggle datasets create -p <folder>` reads. Fails if any listed file is missing, so a dataset always
matches its manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Dhruv-Sharma29/Ivy-1.5B"
ALL4 = ["v1", "v2", "sft_indist", "sft_heldout"]

# name, kind, params_M, corpus, tokens_M, peak_lr, seed, base, sft_data, epochs, evals, note
RUNS = [
    ("ivy-40m-pilot", "pretrain", 40.9, "mix-v0", 98.3, 6e-4, 1234, "", "", "", [], "pipeline pilot; LR not swept"),
    ("ivy-40m-lr-6e-4-100M", "pretrain", 40.9, "mix-v1", 98.3, 6e-4, 1234, "", "", "", [], "40M LR sweep"),
    ("ivy-40m-lr-1.2e-3-100M", "pretrain", 40.9, "mix-v1", 98.3, 1.2e-3, 1234, "", "", "", [], "40M LR sweep"),
    ("ivy-40m-lr-2.4e-3-100M", "pretrain", 40.9, "mix-v1", 98.3, 2.4e-3, 1234, "", "", "", [], "40M LR sweep (best)"),
    ("ivy-40m-lr-4.8e-3-100M", "pretrain", 40.9, "mix-v1", 98.3, 4.8e-3, 1234, "", "", "", [], "40M LR sweep"),
    ("ivy-40m-scale-300M-lr-2.4e-3", "pretrain", 40.9, "mix-v1", 300.0, 2.4e-3, 1234, "", "", "", [], "40M token scaling"),
    ("ivy-40m-scale-1000M-lr-2.4e-3", "pretrain", 40.9, "mix-v1", 1000.0, 2.4e-3, 1234, "", "", "", [], "40M token scaling; base of sft-40m-*"),
    ("ivy-300m-lr-3e-4", "pretrain", 307.3, "mix-v0", 98.3, 3e-4, 1234, "", "", "", [], "300M LR sweep"),
    ("ivy-300m-lr-6e-4", "pretrain", 307.3, "mix-v0", 98.3, 6e-4, 1234, "", "", "", [], "300M LR sweep"),
    ("ivy-300m-lr-1.2e-3", "pretrain", 307.3, "mix-v0", 98.3, 1.2e-3, 1234, "", "", "", [], "300M LR sweep (best)"),
    ("ivy-300m-lr-2.4e-3", "pretrain", 307.3, "mix-v0", 98.3, 2.4e-3, 1234, "", "", "", [], "300M LR sweep"),
    ("ivy-300m-rep-seed4321-lr-1.2e-3", "pretrain", 307.3, "mix-v0", 98.3, 1.2e-3, 4321, "", "", "", [], "seed replicate of the best 300M LR"),
    ("ivy-300m-scale-300M-lr-1.2e-3", "pretrain", 307.3, "mix-v1", 300.0, 1.2e-3, 1234, "", "", "", [], "300M token scaling; base of sft-300m-*"),
    ("sft-40m-1b", "sft", 40.9, "", "", 2e-4, 1234, "ivy-40m-scale-1000M-lr-2.4e-3", "v0", 3, ALL4, "all templates, incl. held-out ones"),
    ("sft-40m-1b-v1train", "sft", 40.9, "", "", 2e-4, 1234, "ivy-40m-scale-1000M-lr-2.4e-3", "v1-train", 3, ALL4, "held-out templates never seen"),
    ("sft-40m-1b-v1train-1ep", "sft", 40.9, "", "", 2e-4, 1234, "ivy-40m-scale-1000M-lr-2.4e-3", "v1-train", 1, ALL4, "1-epoch recipe"),
    ("sft-40m-1b-s2-1ep", "sft", 40.9, "", "", 2e-4, 1234, "ivy-40m-scale-1000M-lr-2.4e-3", "v1-train-s2", 1, ["sft_heldout_s2", "v2"], "independent template split (salt s2)"),
    ("sft-300m-300m", "sft", 307.3, "", "", 1e-4, 1234, "ivy-300m-scale-300M-lr-1.2e-3", "v0", 3, ALL4, "all templates, incl. held-out ones"),
    ("sft-300m-300m-v1train", "sft", 307.3, "", "", 1e-4, 1234, "ivy-300m-scale-300M-lr-1.2e-3", "v1-train", 3, ALL4, "held-out templates never seen"),
    ("sft-300m-300m-v1train-lr2e-4", "sft", 307.3, "", "", 2e-4, 1234, "ivy-300m-scale-300M-lr-1.2e-3", "v1-train", 3, ALL4, "LR check"),
    ("sft-300m-300m-v1train-1ep", "sft", 307.3, "", "", 1e-4, 1234, "ivy-300m-scale-300M-lr-1.2e-3", "v1-train", 1, ALL4, "1-epoch recipe"),
    ("sft-300m-300m-s2-1ep", "sft", 307.3, "", "", 1e-4, 1234, "ivy-300m-scale-300M-lr-1.2e-3", "v1-train-s2", 1, ["sft_heldout_s2", "v2"], "independent template split (salt s2)"),
]
REPORTS = ["lr-sweep-300m.json", "scaling-common-eval.json", "common-eval-v2.jsonl", "scaling-fit-rough.json", "scaling-300m-same-val.json",
           "pilot-40m.json", "baseline-unigram-mix-v0.json", "tokenizer-candidates-eval.json"]
EVALS = ["tools.json", "scenarios_v1.jsonl", "scenarios_v2.jsonl", "sft_indist.jsonl", "sft_heldout.jsonl", "sft_heldout_s2.jsonl"]
SFT_DATA = ["v0.jsonl", "v1-train.jsonl", "v1-train-s2.jsonl"]

RESULTS_README = f"""# Ivy experiments: small-LM pretraining and tool-use fine-tuning results

Measurements from building **Ivy**, a small language model trained from scratch to turn device-control requests into exact tool
calls (or to ask, decline or reply when a call would be wrong). Code, design record and runbook: {REPO}

**What is inside:** training curves of 13 pretraining runs (40M and 300M parameters), scores and raw model outputs of 9 tool-use
fine-tuning runs on up to five eval sets, the summary reports, and the eval sets themselves. **No model weights and no pretraining
text** are included. The companion notebook reproduces every chart from these files.

## Headline findings
- Best peak LR: 2.4e-3 at 40M, 1.2e-3 at 300M (both inside the tested range; seed noise ≈ 0.004 nats).
- A 40M model on 1B tokens (loss 3.197) beats a 300M model on 300M tokens (3.275) at about half the compute.
- Tool-use fine-tuning on templates reaches ~92% on familiar phrasings but only 45–57% on held-out phrasings.
- One fine-tuning epoch transfers better than three at 300M; fine-tuning loss on the training templates cannot show this.
- 40M-vs-300M comparisons change sign between two independent template splits: no robust size effect at this scale.

## Files
| Path | Contents |
|---|---|
| `runs.csv` | one row per run: size, corpus, tokens, peak LR, seed; for fine-tuning runs the base, data, epochs and eval sets |
| `runs/<pretrain-run>/metrics.jsonl` | per step `{{step, loss, gnorm, tokens, lr}}`, plus `{{step, val_loss}}` rows |
| `runs/<sft-run>/sft_metrics.jsonl` | fine-tuning loss curve |
| `runs/<sft-run>/score_<set>.json` | accuracy, 95% Wilson interval, unsafe actions, per-category counts |
| `runs/<sft-run>/outputs_<set>.jsonl` | raw greedy model output per scenario id (for paired tests and error analysis) |
| `evals/` | the tool catalogue and five eval sets (see below) |
| `reports/` | LR sweep, cross-size common evaluation (`common-eval-v2.jsonl`), a rough scaling fit, the 40M pilot, tokenizer comparison |

Eval sets: `scenarios_v1` (225) and `scenarios_v2` (260) are hand-designed; `sft_indist` (600) holds new combinations of training
templates; `sft_heldout` (600) and `sft_heldout_s2` (600) use templates and slot values held out from training, on two
independent splits.

## Notes for analysis
- Pretraining batches are 65,536 tokens per step (sequence 2,048), warmup 100, cosine to 10%, bf16.
- In-training validation numbers are only comparable within one corpus and micro-batch setting; use `reports/common-eval-v2.jsonl`
  to compare model sizes (one fixed sample of validation text).
- Compare fine-tuning runs on the same items (e.g. exact McNemar test on `outputs_*.jsonl`), not by headline accuracy alone.

## Sources and caveats
Pretraining data: HuggingFaceFW/fineweb-edu (sample-10BT) and HuggingFaceTB/finemath (finemath-4plus), both ODC-By; only
measurements derived from them are included here. One author wrote the templates, eval scenarios and labels, so absolute tool-use
scores are optimistic. Single seed per configuration except one replicate.
"""

SFT_README = f"""# Ivy tool-use SFT data: template-generated device-control conversations

Training data for the tool-use fine-tuning of **Ivy**, a small from-scratch language model for device control, plus the eval sets
it is measured on. Generated entirely from hand-written templates by `src/ivy_model/sft_gen.py` in {REPO}; **no teacher model and
no third-party model output**.

## Task
Given a conversation and the user's devices (each with its own tools), answer with exactly one JSON action: `tool_call` (tool,
device, arguments), `ask_user` (clarifying question), `decline`, or `reply`. Eight placeholder tools: `open_app`, `search_files`,
`read_file`, `move_file`, `delete_file`, `run_project_tests`, `set_volume`, `send_notification` (`evals/tools.json`).

## Files
| File | Rows | Contents |
|---|---:|---|
| `sft/v0.jsonl` | 20,000 | all templates (`--split all`, seed 7) |
| `sft/v1-train.jsonl` | 20,000 | training split: ~30% of every phrasing list and slot pool held out (`--split train`) |
| `sft/v1-train-s2.jsonl` | 20,000 | an independent training split (`--split train --split-salt s2`) |
| `evals/scenarios_v1.jsonl`, `scenarios_v2.jsonl` | 225, 260 | hand-designed eval scenarios |
| `evals/sft_indist.jsonl` | 600 | new combinations of `v1-train` templates |
| `evals/sft_heldout.jsonl` | 600 | only templates and slot values held out from `v1-train` |
| `evals/sft_heldout_s2.jsonl` | 600 | only templates and slot values held out from `v1-train-s2` |

Training rows: `{{"kind", "devices", "messages", "answer"}}`. Eval rows: `{{"id", "category", "devices", "messages", "expected"}}`.
Every training user turn was checked against every eval utterance (word Jaccard ≥ 0.6 dropped): 0 overlaps.

## How to use it fairly
Train on `v1-train` and report accuracy on `sft_heldout` (and the `_s2` pair as a replication), always next to the in-distribution
score. Do not train on the eval sets. Score tool calls exactly (tool, device and every argument) and count unsafe actions (acting
when the right answer was to ask, decline or reply); the scorer is `evals/score.py` in the repository.

## Limits
One author wrote the templates and the evals, so even held-out phrasings are closer to training than real users would be.
English only; placeholder tool catalogue. Template data teaches the format and the policy, not broad language understanding.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["results", "sft"], default="results")
    ap.add_argument("--out")
    ap.add_argument("--owner", default="KAGGLE_USERNAME", help="your Kaggle username (for dataset-metadata.json)")
    ap.add_argument("--public", action="store_true", help="mark the dataset public (default: private)")
    a = ap.parse_args()
    slug = "ivy-experiments" if a.which == "results" else "ivy-tool-use-sft"
    out = Path(a.out or ROOT / "kaggle" / slug)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def cp(src: Path, dst: Path) -> None:
        if not src.is_file():
            raise SystemExit(f"missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for e in EVALS:
        cp(ROOT / "evals" / e, out / "evals" / e)
    if a.which == "results":
        for name, kind, *_, evals, _note in RUNS:
            src = ROOT / "checkpoints" / name
            if kind == "pretrain":
                cp(src / "metrics.jsonl", out / "runs" / name / "metrics.jsonl")
                continue
            cp(src / "sft_metrics.jsonl", out / "runs" / name / "sft_metrics.jsonl")
            for v in evals:
                cp(src / f"score_{v}.json", out / "runs" / name / f"score_{v}.json")
                cp(src / f"outputs_{v}.jsonl", out / "runs" / name / f"outputs_{v}.jsonl")
        for r in REPORTS:
            cp(ROOT / "reports" / r, out / "reports" / r)
        with open(out / "runs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "kind", "params_M", "corpus", "tokens_M", "peak_lr", "seed", "base", "sft_data", "epochs", "eval_sets", "note"])
            w.writerows([*r[:10], " ".join(r[10]), r[11]] for r in RUNS)
        readme, title = RESULTS_README, "Ivy: small-LM pretraining and tool-use results"
        subtitle = "LR sweeps, scaling runs and held-out tool-use evals for 40M and 300M models"
        keywords = ["nlp", "deep learning", "transformers"]
    else:
        for d in SFT_DATA:
            cp(ROOT / "data" / "sft" / d, out / "sft" / d)
        readme, title = SFT_README, "Ivy tool-use SFT data (device control)"
        subtitle = "Device-control conversations with train and held-out template splits"
        keywords = ["nlp", "text", "deep learning"]
    (out / "README.md").write_text(readme)
    meta = {"title": title, "id": f"{a.owner}/{slug}", "subtitle": subtitle, "description": readme, "isPrivate": not a.public,
            "licenses": [{"name": "unknown"}], "keywords": keywords}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
