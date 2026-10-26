# 10 — Implementation notes

Where building Ivy proved the plan wrong, and the mistakes found along the way. Each entry says what was planned, what happened,
and what changed as a result. Entries are roughly in build order.

## Environment

**1. The first throughput numbers came from the wrong PyTorch build.**
*Planned:* benchmark on the system Python. *Happened:* its PyTorch build predates the GB10 architecture (no fused attention,
bf16 matmul at ~10 instead of ~96 TFLOP/s, `torch.compile` broken), and workarounds were built for it (`--attn-block`,
`--precision tf32`). A newer stack (PyTorch 2.13 + CUDA 13) was 4–5× faster in eager mode and made the workarounds slower.
*Changed:* everything runs in the project virtualenv; the workaround flags stay but default to off, and the docs say so.

**2. The GPU and its memory are shared.**
The machine's 119 GB is unified between CPU and GPU, and other users' jobs held 30–60 GB and up to 95% of the GPU during most
runs. Wall-clock times vary by up to 4×. *Changed:* runs checkpoint every 500 steps and resume deterministically; 300M uses
micro-batch 4 × 8 (13 GB) instead of 8 × 4 (19 GB); losses, not times, are compared.

## Data

**3. The planned mixture was not available.**
*Planned:* 60% web, 25% code, 15% math. *Happened:* The Stack v2 is gated and needs per-file licence filtering. *Changed:* the
corpora are 85% FineWeb-Edu / 15% FineMath with no code, and the tokenizer was trained without code (see 5).

**4. Streaming shuffle does not scale on this machine.**
`datasets` streaming `.shuffle()` grew past 18 GB of RAM in two minutes while building the 2B-token corpus, and plain file order
is biased (the first FineWeb-Edu documents all come from one 2013 crawl). *Changed:* `--balanced-files-prefix` streams each
parquet file separately with an equal token share (peak ~1.3 GB). Near-duplicate removal was switched off at that scale: it cost
~10 GB of RAM and had removed only 8 of ~88,000 documents in the pilot, because both sources are deduplicated upstream.

## Tokenizer and export

**5. The tokenizer was frozen before an export risk was known.**
Stock llama.cpp rejected its pre-tokenizer. It turned out to be equivalent to llama.cpp's `starcoder` pre-tokenizer, so a
one-line converter patch fixed it with no runtime change, verified on 2.47M tokens. The freeze also means JSON tool calls cost
15–20% more tokens than with tokenizers trained on code. Neither is worth a new pretraining run.

## Evaluation integrity

**6. The eval-overlap guard silently checked nothing.**
`check-sft` read a `user` field, but chat-format training rows keep their text in `messages`, so it reported a clean result on
zero candidates. *Changed:* it reads every user turn and refuses empty input.

**7. The generator and the guard counted words differently.**
One eval opener ("find notes.txt") leaked into training data because the two used different tokenisation rules for the 3-word
threshold. *Changed:* both use `cleaning.words`; 0 overlaps at Jaccard 0.6.

**8. The template generator taught contradictory labels.**
`"print {file}"` was both a `read_file` template and an unsupported (decline) request: 80 rows said one thing, 31 the other.
*Changed:* bare "print <file>" means a printer (decline), matching the eval sets; a test forbids any template appearing in both
lists.

**9. In-training validation numbers are not comparable across runs.**
The validation windows depend on `--micro-bsz`, so runs with different batch settings were measured on different text (gaps up
to ~0.03 nats). The cross-size comparison was first done with a one-off evaluation that left no script behind. *Changed:*
`scripts/common_eval.py` scores every checkpoint on one fixed sample and reproduces the earlier numbers within 0.006 nats.

## Fine-tuning

**10. High scores on the eval sets were mostly template coverage.**
The first fine-tuned 40M model scored 67% on v2. A template-held-out split showed the same model at ~56% on unseen phrasings
against ~92% on familiar ones. *Changed:* `sft_gen.py --split` and the in-distribution / held-out test sets; v1/v2 scores are
always reported next to the held-out score.

**11. Fine-tuning loss cannot choose the stopping point.**
One epoch transfers better than three at 300M (53.0% vs 45.3% on held-out phrasings, p < 0.0001) while the fine-tuning validation
loss, measured on the same templates, barely changes. Only a held-out set reveals over-training.

**12. A size result did not replicate, and was withdrawn.**
On the first template split, 300M made significantly fewer unsafe actions than 40M (p = 0.002). On an independent split
(`--split-salt s2`) the difference reversed and 40M was clearly more accurate. Every 40M-vs-300M comparison changed sign or
significance between the two splits. *Changed:* no recipe or size claim is made from a single split; both held-out sets have
been used for decisions and count as development sets.

## Operations

**13. Editing a shell script while it runs breaks the run.**
`sh` reads scripts incrementally; rewriting one in place while a job executes it makes the job resume at a stale byte offset.
*Changed:* scripts in use are replaced by writing a new file and renaming it (`mv`), which leaves the running shell on the old
copy.
