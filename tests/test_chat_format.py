import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from tokenizers import Tokenizer  # noqa: E402

from ivy_model import chat_format as CF  # noqa: E402

TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "ivy-bpe-32k" / "tokenizer.json"))
V2 = [json.loads(l) for l in (ROOT / "evals" / "scenarios_v2.jsonl").read_text().splitlines()]
V1 = [json.loads(l) for l in (ROOT / "evals" / "scenarios_v1.jsonl").read_text().splitlines()]


def answer_for(s):
    e = dict(s["expected"])
    if e["action"] == "ask_user":
        e["question"] = "Which one?"
    return e


def test_special_tokens_are_atomic_and_boundary_is_clean():
    s = V2[150]
    prompt, ans = CF.render_prompt(s["devices"], s["messages"]), CF.render_answer(answer_for(s))
    joint = TOK.encode(prompt + ans, add_special_tokens=False).ids
    split = TOK.encode(prompt, add_special_tokens=False).ids + TOK.encode(ans, add_special_tokens=False).ids
    assert joint == split
    assert TOK.token_to_id("<|assistant|>") == joint[len(TOK.encode(prompt, add_special_tokens=False).ids) - 1]


def test_every_eval_scenario_renders_and_round_trips():
    for s in V1 + V2:
        msgs = s.get("messages") or [{"role": "user", "content": s["user"]}]
        ids = CF.encode_prompt(TOK, s["devices"], msgs)
        assert TOK.decode(ids, skip_special_tokens=False) == CF.render_prompt(s["devices"], msgs), s["id"]
        assert len(ids) < 1500, (s["id"], len(ids))


def test_labels_cover_exactly_answer_and_end_token():
    s = V2[0]
    ids, labels = CF.encode_example(TOK, s["devices"], s["messages"], answer_for(s))
    n_prompt = len(CF.encode_prompt(TOK, s["devices"], s["messages"]))
    assert all(x == -100 for x in labels[:n_prompt]) and labels[n_prompt:] == ids[n_prompt:]
    assert ids[-1] == TOK.token_to_id("<|end|>") and ids[n_prompt - 1] == TOK.token_to_id("<|assistant|>")
    decoded = TOK.decode(ids[n_prompt:], skip_special_tokens=False)
    assert json.loads(decoded.removesuffix("<|end|>")) == s["expected"]


def test_answer_is_canonical_and_scorer_compatible():
    call = {"args": {"level": 30}, "device": "phone", "tool": "set_volume", "action": "tool_call"}
    assert CF.render_answer(call) == '{"action":"tool_call","tool":"set_volume","device":"phone","args":{"level":30}}<|end|>'
    import score as S  # noqa: E402
    ask = json.loads(CF.render_answer({"action": "ask_user", "question": "Which device?"}).removesuffix("<|end|>"))
    sc = {"id": "x", "category": "c", "user": "u", "devices": [{"id": "d", "tools": ["set_volume"]}], "expected": {"action": "ask_user"}}
    assert S.score([sc], {"x": json.dumps(ask)})["correct"] == 1


def test_too_long_returns_none_and_unknown_role_raises():
    s = V2[0]
    big = s["messages"] + [{"role": "tool", "content": "x " * 3000}]
    assert CF.encode_example(TOK, s["devices"], big, {"action": "reply"}) is None
    try:
        CF.render_prompt(s["devices"], [{"role": "robot", "content": "hi"}])
        raise AssertionError("should have raised")
    except ValueError:
        pass
