# Pretraining

Source: [`src/ivy_model/train.py`](../src/ivy_model/train.py). A single-GPU training loop with full-state checkpoints and
deterministic data order.

## Recipe

| Setting | Value |
|---|---|
| Objective | next-token cross-entropy on 2,048-token windows |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on parameters with 2+ dimensions (none on norm weights) |
| Schedule | linear warmup (100 steps), then cosine decay to 10% of the peak LR |
| Gradient clipping | global norm 1.0 (the logged `gnorm` is before clipping) |
| Precision | bf16 autocast, fp32 master weights |
| Batch | micro-batch × accumulation × 2,048 = 65,536 tokens per step (8 × 4 or 4 × 8) |
| Peak LR | 2.4e-3 for 40M, 1.2e-3 for 300M (tuned at ~100M tokens); 1.5B untuned |

## Running

```bash
PYTHONPATH=src .venv/bin/python -u -m ivy_model.train --config configs/ivy-300m.json \
    --data data/tokens/mix-v1/train.bin --val-data data/tokens/mix-v1/val.bin \
    --out checkpoints/my-run --steps 4578 --lr 1.2e-3 --warmup 100 \
    --micro-bsz 4 --accum 8 --ckpt-every 500 --compile --resume --seed 1234
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | required | model config JSON |
| `--data`, `--val-data` | required / none | `uint16` token files |
| `--out` | required | checkpoint directory (`latest.pt`, `metrics.jsonl`) |
| `--steps` | required | optimizer steps; tokens = steps × 65,536 at the default batch |
| `--lr`, `--warmup` | 3e-4, 100 | peak LR and warmup steps |
| `--micro-bsz`, `--accum` | 8, 4 | sequences per micro-batch, micro-batches per step |
| `--seq-len` | 2048 | window length |
| `--seed` | 1234 | controls init and data order |
| `--ckpt-every` | 500 | checkpoint interval (steps) |
| `--resume` | off | continue from `--out/latest.pt` if present |
| `--val-batches` | 32 | validation micro-batches per evaluation |
| `--compile` | off | `torch.compile` each block (needs the CUDA 13 stack) |
| `--grad-ckpt` | off | gradient checkpointing per block: less activation memory, ~20% slower |
| `--precision`, `--attn-block` | bf16, 0 | legacy speed options; keep the defaults on the current stack |

Choose `--steps` so the cosine schedule spans the whole run: a token budget needs its own run, not a checkpoint taken from the
middle of a longer one.

## Outputs

- `latest.pt`: model and optimizer state, step, tokens seen, config digest and the CPU/CUDA RNG states. Written to a temporary
  file and renamed, so a crash never leaves a half-written checkpoint. About 3.4 GB at 300M (optimizer state included).
- `metrics.jsonl`: one line per step (`step, loss, gnorm, tokens, lr`) and one per evaluation (`step, val_loss`).

## Deterministic resume

Each batch is a pure function of `(seed, step)`, so a resumed run sees exactly the batches an uninterrupted run would.
`tests/test_resume.py` checks this on CPU. On GPU, results are not bit-identical because kernels are non-deterministic; a resumed
run matches an uninterrupted one within that noise (checked with a `kill -9` during the 40M pilot).

## Validation loss: what is comparable

`evaluate()` averages the loss over a fixed set of validation windows, so values within one run are comparable. The windows
depend on `--micro-bsz`, though, so **runs with different micro-batch sizes are measured on different windows** (differences up to
~0.03 nats). For comparisons across runs and model sizes, use the common evaluation in
[`reports/scaling-common-eval.json`](../reports/scaling-common-eval.json): 96 windows × 4 × 2,048 tokens of `mix-v1` validation
text, reported for web, math and the mix.

## Experiment scripts

| Script | What it runs |
|---|---|
| [`scripts/lr_sweep_300m.sh`](../scripts/lr_sweep_300m.sh) | 300M at 3e-4, 6e-4, 1.2e-3, 2.4e-3 for 1,500 steps (98.3M tokens, `mix-v0`) |
| [`scripts/replicate_300m.sh`](../scripts/replicate_300m.sh) | the best 300M LR again with seed 4321 |
| [`scripts/scaling_300m.sh`](../scripts/scaling_300m.sh) | 300M at 1.2e-3 for N million tokens of `mix-v1` |
| [`scripts/scaling_40m.sh`](../scripts/scaling_40m.sh) | 40M LR sweep on `mix-v1`, then the best LR at 300M and 1B tokens |
| [`scripts/summarize_lr_sweep.py`](../scripts/summarize_lr_sweep.py) | writes `reports/lr-sweep-300m.json` |

The four pretraining scripts skip runs that already have a `DONE` marker, so they can be restarted safely.

## Operating on a shared machine

The GB10's 119 GB of memory is unified: GPU allocations count against system RAM. Before launching a job, check `free -g` and the
running GPU processes. Prefer `--micro-bsz 4 --accum 8` at 300M (~13 GB instead of ~19 GB, same batch). The corpus build sets a
`ulimit -v` cap so that a runaway job fails instead of pushing other users' processes out.
