# Ivy tool-use eval

`scenarios_v1.jsonl` — 225 hand-authored scenarios (sha256 prefix `cc280211e0cab946`); `scenarios_seed.jsonl` — the original 8.
Built by `build_scenarios.py` (tables of utterances; the JSONL is checked against it in `tests/test_eval_v1.py`).
Score: `python evals/score.py --scenarios evals/scenarios_v1.jsonl --outputs outputs.jsonl`.
Baselines: `python evals/baselines.py --details`.

| Category | n | Expected |
|---|---|---|
| tool_selection, argument_correctness, device_selection, near_miss_confusion, literal_text | 30/30/30/20/10 | tool call |
| missing_information, ambiguous_device | 25/20 | ask_user |
| unsupported, cancellation | 25/15 | decline |
| unsupported_on_device | 20 | call routed to the one capable device (ask_user also accepted) |

## Held-out rules
- **Never train, tune prompts, or choose checkpoints on these scenarios.** Do not generate SFT data from `build_scenarios.py` or paraphrase these utterances; build training data from separate sources and check overlap (exact and near-duplicate) against `scenarios_v1.jsonl` before SFT.
- Report the number once per checkpoint decision; if you iterate against it, it is a dev set. Create `scenarios_v2` as a fresh set for final reporting.
- Report accuracy with the Wilson CI from `score.py` (n=225: about ±5 points) and per-category counts. Differences of a few points are noise.

## Known limitations (read before trusting a score)
- **Single annotator.** All labels are one person's judgement. Contested ones: routing an unsupported-on-device request to another device, `ask_user` for an unnamed device (no "current device" in the input), and relative requests like "turn the volume up". Have a second person label a sample and report agreement.
- **Too easy at the policy level.** A keyword/regex baseline scores about 91% (it was written with sight of the scenarios, so this is optimistic), meaning the set mostly tests output format and the ask/decline policy, not hard language understanding. Missing: multi-turn context, real typos and noise, longer tool catalogues, Hinglish (pending the Phase 0 language decision), multi-step tasks, a "just answer, no tool" action, tool results.
- Exact-match arguments: string args are expected as written by the user; paraphrased or re-cased args fail. The scenarios use quoting to keep this fair, but it penalises reasonable normalisation.
- The tool catalogue (`tools.json`) is the 8 placeholder tools; Phase 0 has not fixed the real one.
- `approval` flags in `tools.json` are not scored.

---

# v2: harder, conversational (`scenarios_v2.jsonl`)

260 hand-authored scenarios (sha256 prefix `bceeffeb00796c29`), built by `build_scenarios_v2.py`, tested in `tests/test_eval_v2.py`.
Score with the same `score.py`; baselines: `python evals/baselines.py --scenarios evals/scenarios_v2.jsonl --details`.
v1 stays as a regression set; v2 is the one to report on for model quality. Both are held-out under the rules above, and
`python -m ivy_model.cleaning check-sft` guards every user turn of 3+ words in both.

**Schema additions:** `messages` (the whole conversation up to the point of action; roles `user`, `assistant` with a question
or a `call`, and `tool` results), `user` (last user message), optional device `label` (e.g. "iPad", "Linux GPU server"), and a new
action `{"action": "reply"}` = answer in plain text without a tool call (only the action is scored, not the text). A model must be
able to emit `reply`; the SFT format needs it too.

| Category | n | What it tests | Expected |
|---|---|---|---|
| noisy_phrasing | 25 | typos, abbreviations, filler | tool call |
| indirect_request | 25 | intent without command words | tool call |
| long_context_args | 20 | the right value among distractors | tool call |
| multi_turn_resolution | 29 | answers to the assistant's questions, "do the same on…", "read that one" | tool call |
| correction_in_dialog | 15 | "no, I meant…" | tool call |
| confirmation | 22 | yes/no to a confirmation question | call (yes) / decline (no) |
| multi_step_first_call | 15 | multi-step requests | first fully-determined call |
| hard_device_routing | 20 | six labelled devices, nicknames, overlaps | call, or ask_user if several devices fit |
| tool_result_followup | 20 | next step after a tool result | next call / ask_user / reply |
| injection_in_tool_result | 15 | instructions inside tool output are data | reply (or the user's real next call) |
| reply_no_tool | 25 | questions that need no tool | reply |
| unsupported_subtle | 15 | looks like a tool, is not | decline |
| ask_unresolved | 14 | information still missing | ask_user |

By action: 167 tool call, 48 reply, 25 decline, 20 ask_user.

**Baselines on v2** (v1 numbers in brackets): always ask 7.7%, always decline 9.6%, random valid call 0%, keyword rules **23.5%** [91%].
The keyword baseline fails mostly where it should: multi-turn 0/29, reply 0/25, injection 0/15, correction 1/15, tool results 2/20.
It still gets 10/25 noisy and 10/22 confirmations by luck (default policies), so a few points above zero are not capability.

**Contested labels and policies (single annotator, treat with suspicion):**
- `multi_step_first_call`: "the first call whose arguments are fully known" is my convention; an app might want a different plan format.
- `reply` vs `decline`: reply = answerable from the model's own knowledge or the device list; decline = a device/service action no tool can do (so "what's my IP address?" is decline, "can you search files on my laptop?" is reply).
- "undo that" after a move is labelled as the reverse move; "max out the volume" as 100; "ping my wrist" as the watch.
- `ask_user` when several devices can serve an unnamed request (still no "current device" input).
- Tool-result scenarios: reporting results is `reply`; after a search with several matches, "read the first match" proceeds, "delete whatever you find" asks.
- String arguments are exact-match as typed (e.g. "pls opne spotify" expects `spotify`); a model that re-cases or corrects them scores as wrong.

**Other limits:** all scenarios were written by one author (an AI assistant), so phrasing may resemble LLM-generated text and flatter models trained on such data; nothing here was written by a human user or drawn from real usage logs. Placeholder 8-tool catalogue; no Hinglish; synthetic tool results; no real latency/streaming. A second human annotator and a sample of real user phrasing are the most valuable additions.
