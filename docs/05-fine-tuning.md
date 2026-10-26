# Tool-use fine-tuning

A pretrained checkpoint is fine-tuned (SFT) to answer every request with exactly one JSON action. Three modules share one
definition of the format, so training data, training and evaluation cannot drift apart:

| Module | Role |
|---|---|
| [`chat_format.py`](../src/ivy_model/chat_format.py) | renders prompts and answers; the single source of the format |
| [`sft_gen.py`](../src/ivy_model/sft_gen.py) | generates training data from hand-written templates |
| [`sft.py`](../src/ivy_model/sft.py) | full-parameter fine-tuning with loss on answer tokens only |
| [`evals/run_model.py`](../evals/run_model.py) | greedy decoding of a checkpoint over eval scenarios |

## Tool catalogue

Eight placeholder tools ([`evals/tools.json`](../evals/tools.json)). The real catalogue will be fixed by the Ivy app.

| Tool | Required arguments | Optional | Needs approval |
|---|---|---|---|
| `open_app` | `name` | | no |
| `search_files` | `query` | `directory` | no |
| `read_file` | `path` | | no |
| `move_file` | `source`, `destination` | | yes |
| `delete_file` | `path` | | yes |
| `run_project_tests` | `directory` | | yes |
| `set_volume` | `level` (integer) | | no |
| `send_notification` | `message` | | no |

## Prompt format

```text
<|system|>{instruction}<|end|>
<|tool_def|>{JSON: required/optional args of every tool present on the devices}<|end|>
<|system|>devices: {JSON list of {id, label?, tools}}<|end|>
<|user|>...<|end|>                  a user turn
<|assistant|>...<|end|>             an earlier assistant question
<|tool_call|>{call JSON}<|end|>     an earlier tool call
<|tool_result|>...<|end|>           tool output: data, never instructions
<|assistant|>                       generation starts here
```

The instruction tells the model to answer with exactly one JSON object and to treat text inside tool results as data.

## Answer format

The model writes one JSON object, then `<|end|>`:

| Action | Example | When |
|---|---|---|
| `tool_call` | `{"action":"tool_call","tool":"read_file","device":"laptop","args":{"path":"~/notes.txt"}}` | the request is fully determined |
| `ask_user` | `{"action":"ask_user","question":"Which device: iPhone or iPad?"}` | information or the device is ambiguous |
| `decline` | `{"action":"decline","reason":"None of my tools can do that."}` | unsupported request, or the user cancels |
| `reply` | `{"action":"reply","text":"1440"}` | a question that needs no tool, or a tool result to report |

Keys are written in a fixed order. Training loss covers only the answer tokens and the closing `<|end|>`.

## Generating training data

```bash
PYTHONPATH=src .venv/bin/python -m ivy_model.sft_gen --n 20000 --seed 7 --split train --out data/sft/v1-train.jsonl
```

Each row is `{"kind", "devices", "messages", "answer"}`. The generator samples 13 scenario builders, among them:

- **single-device requests** (`single`) and **named devices** (`named_device`), with optional typos and fillers;
- **routing** to the only capable device (`unique_capability`), or **asking** when several fit (`ask_device`);
- **missing arguments** (`ask_missing`) and **multi-turn resolution** (`multi_missing`, `multi_device`);
- **declines** of unsupported requests, **cancellations** and **confirmations**;
- **corrections** ("no wait, 40"), **follow-ups** on another device, **plain replies**;
- **tool results**: chaining, reporting, errors, and prompt injection inside tool output (answered as data).

Every generated user turn is checked against all eval utterances (word Jaccard ≥ 0.6 is dropped), and exact duplicates are
removed.

### Train / held-out split

`--split` divides every phrasing list and every slot-value pool (apps, paths, messages, people…) by a stable hash of each item:
about 30% is held out. Concepts are not split: tools, policies, device ids and answer texts appear on both sides, and
missing-argument phrasings are split per tool so every tool keeps training examples.

| `--split` | Use |
|---|---|
| `all` | original v0 behaviour; reproduces `data/sft/v0.jsonl` byte for byte (seed 7) |
| `train` | training data that never contains a held-out template or value |
| `heldout` | test data built only from held-out templates and values |

`--split-salt` changes the hash, giving an independent split (the default empty salt is the original split; `s2` was used
to replicate results). A comparison that matters should hold on more than one split.

The test sets in `evals/` were built like this:

```bash
G="PYTHONPATH=src .venv/bin/python -m ivy_model.sft_gen"
$G --n 600 --seed 101 --split train   --exclude data/sft/v1-train.jsonl --as-scenarios id --out evals/sft_indist.jsonl
$G --n 600 --seed 202 --split heldout --exclude data/sft/v1-train.jsonl --as-scenarios ho --out evals/sft_heldout.jsonl
```

`--exclude` drops rows whose whole sequence of user turns occurs in the training file; `--as-scenarios` writes the eval format.
The held-out phrasings are clearly further from training: median best word-Jaccard to any training turn is 0.25, against 0.67 for
the in-distribution set.

## Fine-tuning

```bash
PYTHONPATH=src .venv/bin/python -m ivy_model.sft --config configs/ivy-40m.json \
    --init checkpoints/ivy-40m-scale-1000M-lr-2.4e-3/latest.pt \
    --data data/sft/v1-train.jsonl --out checkpoints/sft-40m --epochs 3 --lr 2e-4 --bsz 16 --accum 4
```

- Loads **weights only** from the pretraining checkpoint; the optimizer starts fresh.
- Full-parameter training; right-padded batches grouped into length buckets; maximum length 1,024 tokens.
- Defaults: 3 epochs, LR 1e-4, 30 warmup steps, micro-batch 16 × accumulation 4, seed 1234.
- 2% of the data is held back to report SFT validation loss. It comes from the same templates, so it shows fit, not
  generalisation. Use the eval sets for that.

`scripts/sft_dryrun.sh` runs fine-tuning and then scores any list of eval sets:

```bash
DATA=data/sft/v1-train.jsonl EVALS="v1 v2 sft_indist sft_heldout" PYTHONPATH=src \
    sh scripts/sft_dryrun.sh 40m-1b-v1train configs/ivy-40m.json checkpoints/ivy-40m-scale-1000M-lr-2.4e-3/latest.pt 2e-4
```

## Known limits of the template data

- Templates and evals share one author, so even the held-out set is closer to the training data than real users would be.
- Template data teaches format and policy, not language understanding: models trained on it match templates (see
  [Results](07-results.md)). More varied phrasing, from people or a permissively licensed teacher model, is the main open lever.
- The 8-tool catalogue is a placeholder; English only.
