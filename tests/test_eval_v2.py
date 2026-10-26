import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "src"))

import baselines as B  # noqa: E402
import build_scenarios_v2 as BS2  # noqa: E402
import score as S  # noqa: E402
from ivy_model import cleaning as C  # noqa: E402

V2 = ROOT / "evals" / "scenarios_v2.jsonl"
SCEN = [json.loads(l) for l in V2.read_text().splitlines() if l.strip()]
OTHERS = [json.loads(l) for f in ("scenarios_v1.jsonl", "scenarios_seed.jsonl")
          for l in (ROOT / "evals" / f).read_text().splitlines() if l.strip()]


def test_size_categories_and_action_mix():
    cats = Counter(s["category"] for s in SCEN)
    acts = Counter(s["expected"]["action"] for s in SCEN)
    assert len(SCEN) >= 240 and len(cats) >= 12 and min(cats.values()) >= 10
    assert set(acts) == {"tool_call", "ask_user", "decline", "reply"} and min(acts.values()) >= 15


def test_ids_and_conversations_unique_and_disjoint_from_v1():
    assert len({s["id"] for s in SCEN}) == len(SCEN)
    assert not {s["id"] for s in SCEN} & {s["id"] for s in OTHERS}
    keys = [json.dumps([s["messages"], s["devices"]], sort_keys=True) for s in SCEN]
    assert len(set(keys)) == len(keys)
    other_users = {s["user"].lower() for s in OTHERS}
    assert not {s["user"].lower() for s in SCEN if len(s["user"].split()) >= 3} & other_users


def test_messages_are_well_formed_and_consistent():
    for s in SCEN:
        m = s["messages"]
        assert m[0]["role"] == "user" and m[-1]["role"] in ("user", "tool"), s["id"]
        assert s["user"] == next(x["content"] for x in reversed(m) if x["role"] == "user")
        for a, b in zip(m, m[1:]):  # a tool result must directly follow an assistant call
            assert (b["role"] == "tool") == (a["role"] == "assistant" and "call" in a), s["id"]
        for x in m:
            if x["role"] == "assistant" and "call" in x:
                assert S.schema_valid(x["call"]), s["id"]
        for d in s["devices"]:
            assert set(d["tools"]) <= set(S.CATALOGUE), s["id"]


def test_expected_calls_valid_runnable_and_actions_sensible():
    for s in SCEN:
        e = s["expected"]
        if e["action"] == "tool_call":
            c = {k: e[k] for k in ("tool", "device", "args")}
            assert S.schema_valid(c) and S.executable({"action": "tool_call", **c}, s), s["id"]
        if s["category"] == "reply_no_tool":
            assert e["action"] == "reply"
        if s["category"] == "injection_in_tool_result":
            assert s["messages"][-1]["role"] == "tool"
        if s["category"] == "hard_device_routing" and e["action"] == "ask_user":
            tools = [t for t in S.CATALOGUE if sum(t in d["tools"] for d in s["devices"]) >= 2]
            assert tools


def test_oracle_perfect_and_reply_is_scored_as_action_only():
    outs = {s["id"]: json.dumps(s["expected"]) for s in SCEN}
    r = S.score(SCEN, outs)
    assert r["correct"] == r["n"] == len(SCEN) and r["unsafe_executable_mismatch"] == 0
    reply = next(s for s in SCEN if s["expected"]["action"] == "reply")
    assert S.score([reply], {reply["id"]: json.dumps({"action": "reply", "text": "anything"})})["correct"] == 1
    assert S.score([reply], {reply["id"]: json.dumps({"action": "decline"})})["correct"] == 0


def test_committed_file_matches_build_script():
    assert V2.read_text() == "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in BS2.rows)


def test_v1_baselines_are_much_weaker_on_v2_than_on_v1():
    res = B.run_all(SCEN)
    assert res["keyword_rules"]["accuracy"] < 0.6          # v1: ~0.91
    assert res["always_ask"]["accuracy"] < 0.2 and res["random_valid"]["accuracy"] < 0.05


def test_conversation_user_turns_are_in_the_overlap_guard():
    ev = C.load_ivy_eval_utterances()
    multi = next(s for s in SCEN if s["category"] == "correction_in_dialog")
    assert any(k.startswith(f"scenarios_v2:{multi['id']}") for k in ev)
    first = multi["messages"][0]["content"]
    assert C.find_eval_overlap([first.upper() + "!"], {k: v for k, v in ev.items() if k.startswith("scenarios_v2:")})
