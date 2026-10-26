# Ivy — Documentation

Ivy is a small language model trained from scratch to turn device-control requests into exact tool calls, and to ask, decline or
answer when a call would be wrong. These documents are the design record and the reference for every part of the pipeline. Read
them in order the first time; use them as reference afterwards.

## Reading order

| # | Doc | What it settles |
|---|---|---|
| 00 | [Charter](00-charter.md) | The problem, scope, non-goals and what "done" means |
| 01 | [Architecture](01-architecture.md) | The network, its three configurations, memory and speed features, measured throughput |
| 02 | [Tokenizer](02-tokenizer.md) | The frozen tokenizer, its special tokens, and why it must not change |
| 03 | [Data](03-data.md) | Sources, the cleaning pipeline, the two corpora, the eval-overlap guard, licences |
| 04 | [Pretraining](04-pretraining.md) | Recipe, CLI, deterministic resume, which validation numbers are comparable |
| 05 | [Tool-use fine-tuning](05-fine-tuning.md) | Prompt and answer format, the template generator and its held-out split, the trainer |
| 06 | [Evaluation](06-evaluation.md) | Eval sets, fail-closed scoring, unsafe actions, baselines, held-out policy |
| 07 | [Results](07-results.md) | Every measurement so far, with confidence intervals, paired tests and caveats |
| 08 | [Export](08-export.md) | Hugging Face and GGUF export, what is verified and what is not |
| 09 | [Roadmap](09-roadmap.md) | Open decisions, next experiments, engineering gaps, release checklist |
| 10 | [Implementation notes](10-implementation-notes.md) | Where the build departed from the plan, and the mistakes found and fixed |

The companion [`RUNBOOK.md`](../RUNBOOK.md) at the repository root has the exact commands to reproduce every result, in order.

## The one-paragraph version

Most small "tool-calling" models are evaluated on phrasings close to their training data, and score well. Ivy's evaluation is
built to catch that: tool calls are scored exactly (tool, device and every argument), acting when the model should have asked or
declined is counted separately as an unsafe action, and a template-held-out split measures accuracy on phrasings the model has
never seen. On that measure the current models reach about 92% on familiar phrasings and 45–57% on unseen ones. The gap, not the
headline number, is the result: at this scale the bottleneck is the diversity of the fine-tuning data, and comparisons between a
40M and a 300M model change sign depending on which templates were trained. Pretraining experiments point the same way on the
size question: a 40M model trained on 1B tokens beats a 300M model trained on 300M tokens at about half the compute.

## Status

The pipeline runs end to end at 40M and 300M parameters: tokenizer (frozen), corpus (2.0B tokens), pretraining, tool-use
fine-tuning, evaluation, and export to Hugging Face and GGUF with verified parity. The 1.5B run has not started; the decision
depends on the success criterion and on the experiments in [09 — Roadmap](09-roadmap.md). A 300M model trained on 1B tokens is
in progress, to test whether a better-trained base transfers better.

## Conventions

- Commands run from the repository root with the project virtualenv: `PYTHONPATH=src .venv/bin/python -m ivy_model.<module>`.
- Losses are mean cross-entropy in nats per token. Accuracies carry 95% Wilson intervals; two models are compared on the same
  items with an exact McNemar test.
- "40M", "300M" and "1.5B" refer to `configs/ivy-40m.json`, `ivy-300m.json` and `ivy-1.5b.json` (40.9M, 307.3M and 1.46B
  parameters).
- A pretraining batch is 65,536 tokens per optimizer step (sequence length 2,048) unless stated otherwise.
