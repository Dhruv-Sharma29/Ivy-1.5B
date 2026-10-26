# 00 — Charter

## The problem

A personal assistant that controls devices has to turn loose, conversational requests into exact actions: *which* tool, on
*which* device, with *which* arguments, or else ask, refuse or just answer. Getting this wrong is not a cosmetic error. A model
that deletes a file when it should have asked which one has done real damage.

Large hosted models do this reasonably well, but they are someone else's model, behind someone else's API, with licence terms
that often forbid using their outputs to train another model. Ivy asks the opposite question: **how far can a small model, trained
from scratch on openly licensed data, get on this task, and what does it take to run it on a laptop?**

## What Ivy does

**Input:** a conversation (user turns, earlier assistant questions, earlier tool calls and their results) plus the list of the
user's devices and the tools each one offers.

**Output:** exactly one JSON action:

- `tool_call`: a tool, a device and its arguments, when the request is fully determined;
- `ask_user`: a clarifying question, when information or the target device is ambiguous;
- `decline`: when no tool can do it, or the user cancels;
- `reply`: a plain answer, when no tool is needed or a tool result must be reported.

The prompt and answer format are specified in [05 — Tool-use fine-tuning](05-fine-tuning.md).

## Scope

### In

- **From-scratch training** of a Llama-style decoder: tokenizer, pretraining, tool-use fine-tuning, evaluation and export.
- **English only.**
- **Openly licensed training data**: FineWeb-Edu and FineMath (ODC-By); code data later if per-file licences can be filtered.
- **On-device target**: Apple M4 with 16 GB of memory, through llama.cpp (GGUF).
- **One training machine**: a single NVIDIA GB10 with 119 GB of unified memory, shared with other users.
- **A compute budget of about 2–3 months** of GPU time for the whole project.

### Out, deliberately

- **Distillation from proprietary models.** Their terms typically forbid training competing models, which defeats the point of
  training from scratch. An open-weight teacher is allowed only if its licence explicitly permits training other models, and
  every generated example must record the teacher, its version and its licence.
- **Other languages.** The tokenizer is English-only; a second language means a new tokenizer and a new pretraining run.
- **The app itself.** Device integration, the real tool catalogue and the user interface live in the Ivy app; this repository
  uses an 8-tool placeholder catalogue.
- **Fine-tuning an existing open model.** It would be faster, but the goal includes independence from upstream weights and their
  licences.

## Constraints that shape everything else

1. **Tokens are the binding constraint.** At the measured ~1,570 tokens/s, a 1.46B-parameter model reaches only about 8–12B
   tokens in 2–3 months, i.e. 5–8 tokens per parameter, well below common compute-optimal heuristics (~20). Whether 1.5B is the
   right size at all is therefore an experimental question, not a given ([07 — Results](07-results.md)).
2. **The evaluation must measure the product task**, not language-model loss: action, tool, device and arguments, scored
   exactly, with unsafe actions counted separately ([06 — Evaluation](06-evaluation.md)).
3. **Public release means licence diligence** on every source: attribution, opt-outs, provenance logs, a model card.

## What "done" means

The project is done when a model:

- meets a **success criterion on held-out phrasings** (an accuracy bar, a ceiling on unsafe actions), which is not yet set and
  is the most important open decision ([09 — Roadmap](09-roadmap.md));
- runs on the M4 within its memory and latency budget, quantised, with tool-use accuracy measured after quantisation;
- ships with a model card that states its data, token counts, evaluation results with confidence intervals and known failure
  modes.

None of these is met yet. What exists is a validated pipeline and a set of small-scale experiments that inform the size decision.
