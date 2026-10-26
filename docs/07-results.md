# Results

Everything measured so far, at 40M and 300M parameters. Raw numbers are in [`reports/`](../reports) and in the Kaggle dataset
built by `scripts/build_kaggle_dataset.py`; the [experiments notebook](../kaggle/ivy_experiments.ipynb) reproduces every chart.
Single seed per configuration unless stated. All losses are in nats per token.

## 1. Learning rate

98.3M tokens per run, identical data order, init and schedule; only the peak LR changes.

| Peak LR | 40M (`mix-v1`) | 300M (`mix-v0`) |
|---|---:|---:|
| 3e-4 | | 3.792 |
| 6e-4 | 4.033 | 3.672 |
| 1.2e-3 | 3.874 | **3.644** |
| 2.4e-3 | **3.825** | 4.111 |
| 4.8e-3 | 4.154 | |

- Both optima are **inside** the tested range. The smaller model prefers a higher LR.
- **Seed noise**: the best 300M setting rerun with seed 4321 gives 3.640 against 3.644 (difference 0.004), so the 0.028 gap
  between 1.2e-3 and 6e-4 at 300M is real.
- The two columns use different corpora. Compare positions of the optima, not the loss levels.
- Not tested: batch size, longer token budgets, the 1.5B model.

## 2. Model size versus data

All on **one common validation set** (96 windows × 4 × 2,048 tokens of `mix-v1` validation text), source
[`reports/scaling-common-eval.json`](../reports/scaling-common-eval.json).

| Model | Training tokens | Compute (6·N·D) | Web | Math | Mix |
|---|---:|---:|---:|---:|---:|
| 40M | 98M | 2.4e16 | 4.019 | 3.015 | 3.843 |
| 40M | 300M | 7.4e16 | 3.613 | 2.633 | 3.445 |
| 40M | 1B | 2.5e17 | 3.367 | 2.393 | **3.201** |
| 300M | 300M | 5.5e17 | 3.443 | 2.482 | 3.279 |

- **The 40M model trained on 1B tokens beats the 300M model trained on 300M tokens, using ~2.2× less compute.** At about one
  token per parameter the 300M model is badly under-trained.
- At equal data (300M tokens), the 300M model is better by 0.167.
- The 40M model gains 0.40 and then 0.25 nats per ~3× more data: diminishing returns.
- A rough scaling projection ([`reports/scaling-fit-rough.json`](../reports/scaling-fit-rough.json)) puts 1.5B on 8–12B tokens
  within ~0.05 nats of 300M on 30–100B tokens, i.e. below evaluation noise. It extrapolates far beyond the data (two sizes, three
  token budgets) and is a scenario analysis, not a prediction.

## 3. Tool-use fine-tuning

Same recipe for every row: 3 epochs, greedy decoding. SFT LR 2e-4 for 40M and 1e-4 for 300M (neither tuned). Bases: 40M trained
on 1B tokens, 300M trained on 300M tokens. **v0** contains all templates; **v1-train** never contains the held-out templates and
slot values.

| Base | SFT data | v1 | v2 | In-dist | **Held-out** | Unsafe actions (held-out) |
|---|---|---:|---:|---:|---:|---:|
| 40M | v0 | 52.9% | 67.3% | 92.5% | **90.5%** | 19 |
| 40M | v1-train | 45.3% | 58.1% | 91.8% | **55.7%** | 72 |
| 300M | v0 | 51.1% | 69.6% | 92.8% | **92.8%** | 8 |
| 300M | v1-train | 43.1% | 55.4% | 91.8% | **45.3%** | 70 |

95% confidence intervals are about ±4 points on the 600-item sets and ±6 on v1/v2.

**Findings**

1. **The models match templates rather than understand requests.** Removing the held-out templates from training costs 35 points
   at 40M (90.5% to 55.7%) and 47 points at 300M (92.8% to 45.3%), while in-distribution accuracy stays at ~92%. Unsafe executions (acting when the
   model should ask or decline) rise from 19 to 72 at 40M and from 8 to 70 at 300M.
2. **With this 3-epoch recipe the 300M model transfers worse than the 40M one** (104 vs 42 items, p < 0.001), but most of that
   is over-training; see [3.1](#31-recipe-and-split-sensitivity).
3. **Policies do not transfer to new wordings.** On held-out phrasing the 40M model gets 2/29 missing-argument requests, 1/16
   cancellations and 2/15 unsupported requests right.
4. **Unseen argument values are not copied.** The largest error group: *"I need WhatsApp open"* becomes `name: "App"`; the 300M
   model substitutes memorised app names (`Calendar` → `Podcast`).
5. **Device lookup is not learned by any model.** When exactly one of several devices has the requested tool, accuracy is 23–53%
   for every model and test set, close to guessing between 2–3 devices.
6. **v1/v2 scores mostly reflect template coverage**: removing 30% of template variety costs 8–9 points there too.

**Caveats**: single seed per run; one author wrote templates and evals, so "held-out" phrasing is still closer to training than
real users would be; per-category counts are small (15–60).

### 3.1 Recipe and split sensitivity

Follow-up runs on `v1-train` (one setting changed per run) and on an **independent template split** (`--split-salt s2`, new
training set and new 600-item held-out set). Held-out accuracy and unsafe actions on that split's held-out set:

| Base | SFT recipe | Split 1 held-out | Split 1 unsafe | Split 2 held-out | Split 2 unsafe |
|---|---|---:|---:|---:|---:|
| 40M | LR 2e-4, 3 epochs | 55.7% | 72 | | |
| 40M | LR 2e-4, 1 epoch | 56.7% | 76 | **54.8%** | 52 |
| 300M | LR 1e-4, 3 epochs | 45.3% | 70 | | |
| 300M | LR 2e-4, 3 epochs | 41.5% | 61 | | |
| 300M | LR 1e-4, 1 epoch | 53.0% | 51 | **45.3%** | 65 |

1. **More fine-tuning means more memorisation.** At 300M, one epoch beats three (83 vs 37 items, p < 0.0001) and LR 2e-4 is worse
   than 1e-4 (p = 0.047), while in-distribution accuracy stays at 91–92% and SFT validation loss barely moves. The 40M model is
   insensitive (p = 0.65). SFT loss on the training templates cannot pick the stopping point.
2. **Size comparisons do not survive a change of split.** At their 1-epoch recipes, 40M vs 300M on split 1: accuracy tied
   (p = 0.09), 300M fewer unsafe actions (p = 0.002). On split 2: 40M more accurate (p < 0.000001), unsafe difference reversed
   (p = 0.09), and on v2 300M better (p = 0.01, tied on split 1). **The earlier "300M is safer" result does not replicate and is
   withdrawn.** Which templates are in training moves results more than model size does at this scale.
3. Consequence for method: any recipe or size comparison now needs several template splits, reporting the spread. Both held-out
   sets have been used for decisions and are development sets; final numbers need a fresh split.

## What this means for the 1.5B decision

Nothing measured so far shows that more parameters help on the task the product needs: at this base quality the 40M-vs-300M
comparison is dominated by which templates were trained. The clearest bottleneck is the **diversity of fine-tuning data**, plus a
device-lookup skill no model has learned. The next informative experiments are listed in
the [Roadmap](09-roadmap.md).
