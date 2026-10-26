# Roadmap

Where the project stands, what is undecided, and what must happen before any public release of weights.

## Status by phase

| Phase | State |
|---|---|
| Scope and eval suite | eval sets v1, v2 and the template-held-out sets done; **success criterion not set** |
| Tokenizer | frozen (2026-10-06) |
| Architecture and export | done; HF and GGUF parity verified; M4/Metal not verified |
| Corpus | 2.0B-token web + math corpus built; **no code data** |
| Small-scale pretraining | LR sweeps at 40M and 300M, scaling runs up to 1B tokens (40M) and 300M tokens (300M) |
| Tool-use fine-tuning | pipeline and held-out evaluation done; models do not yet generalise beyond seen phrasing |
| 1.5B pretraining | **not started**; go/no-go open |
| Quantisation and on-device test | not started |

## Open decisions

1. **Success criterion.** Proposed form: at least X% on the held-out tool-use set with no more than N unsafe actions, within a
   latency and memory budget on the M4. The 1.5B decision has nothing to measure against until this is set.
2. **Source of phrasing diversity.** Template data alone leads to template matching. Options: requests written by people (e.g.
   from a prototype), or paraphrases from an open-weight teacher model whose licence explicitly allows training other models.
   Either needs a provenance log; the teacher route also needs a licence check of the specific model version.
3. **Model size under the compute budget.** At ~1,570 tokens/s a 1.5B model reaches only ~8–12B tokens in 2–3 months
   (5–8 tokens per parameter). The evidence so far favours smaller models trained longer.
4. **Code data.** The Stack v2 needs gated access set up by the project owner, then per-file licence filtering.
5. **Licence** for the code and any future weights.

## Next experiments

| Experiment | Question it answers | Cost |
|---|---|---|
| Multi-split protocol (3–5 salts) for any recipe or size claim | how large is split-to-split variance, and does any effect survive it? | ~1.5 h per 40M+300M pair and split |
| SFT on more varied phrasing | does diversity, not size, close the held-out gap? | depends on decision 2 |
| A better-trained 300M base (more tokens) | does base-model quality improve transfer? | ~5 h per 300M tokens |
| 1.5B at 300M tokens, with a short LR check | the missing model-size point at fixed data | ~53 h uncontended |
| Device-lookup probe | can any size learn "use the only device that has the tool"? | small |

## Engineering gaps

- `run_model.py` has no KV cache; fine for evaluation, too slow for interactive use.
- In-training validation windows depend on `--micro-bsz`; make them fixed per corpus.
- No second annotator for the eval labels; no real-user phrasings.

## Release checklist (none of these is done)

1. Choose a licence for code and weights; add `LICENSE`.
2. Record FineWeb-Edu and FineMath attribution (ODC-By) in the model card; review upstream Common Crawl terms and takedown handling.
3. If code data is used: licence allowlist, opt-outs, attribution records.
4. Stronger personal-data review than the current regex scrub, with a sampling audit.
5. Provenance log for any teacher-generated data.
6. Model card: sources, token counts, evaluation results with confidence intervals, intended use, known failure modes
   (including unsafe-action rates).
7. Verify the GGUF on the actual device and in the app's runtime, including quantised tool-use accuracy.
8. Set every `reviewed` flag in `data-manifests/sources.json` only after an actual review.
