import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import baselines as B  # noqa: E402
import build_scenarios as BS  # noqa: E402
import score as S  # noqa: E402

V1 = ROOT / "evals" / "scenarios_v1.jsonl"
SCEN = [json.loads(l) for l in V1.read_text().splitlines() if l.strip()]
SEED = [json.loads(l) for l in (ROOT / "evals" / "scenarios_seed.jsonl").read_text().splitlines() if l.strip()]


def _capable(s, tool):
    return [d["id"] for d in s["devices"] if tool in d["tools"]]


def test_size_and_category_coverage():
    cats = Counter(s["category"] for s in SCEN)
    assert len(SCEN) >= 200
    assert len(cats) >= 10 and min(cats.values()) >= 10
    assert {"tool_call", "ask_user", "decline"} <= {s["expected"]["action"] for s in SCEN}


def test_ids_and_utterances_unique_and_disjoint_from_seed():
    assert len({s["id"] for s in SCEN}) == len(SCEN)
    users = [s["user"].lower() for s in SCEN]
    assert len(set(users)) == len(users)
    assert not set(users) & {s["user"].lower() for s in SEED}
    assert not {s["id"] for s in SCEN} & {s["id"] for s in SEED}


def test_expected_calls_are_valid_and_runnable():
    for s in SCEN:
        e = s["expected"]
        if e["action"] == "tool_call":
            assert S.schema_valid({k: e[k] for k in ("tool", "device", "args")}), s["id"]
            assert S.executable({"action": "tool_call", **{k: e[k] for k in ("tool", "device", "args")}}, s), s["id"]


def test_policy_scenarios_have_the_structure_their_category_claims():
    for s in SCEN:
        e = s["expected"]
        if s["category"] == "ambiguous_device":
            assert e["action"] == "ask_user"
            assert any(len(_capable(s, t)) >= 2 for t in S.CATALOGUE), s["id"]
        if s["category"] == "unsupported_on_device":
            assert len(_capable(s, e["tool"])) == 1 and e["device"] in _capable(s, e["tool"]), s["id"]
            assert s["accept_alternatives"] == [{"action": "ask_user"}]
        if s["category"] == "device_selection":
            assert e["action"] == "tool_call" and e["device"] in s["user"] or len(_capable(s, e["tool"])) == 1, s["id"]


def test_oracle_scores_perfect():
    outs = {s["id"]: json.dumps(s["expected"]) for s in SCEN}
    r = S.score(SCEN, outs)
    assert r["correct"] == r["n"] == len(SCEN) and r["unsafe_executable_mismatch"] == 0


def test_committed_file_matches_build_script():
    rebuilt = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in BS.rows)
    assert V1.read_text() == rebuilt, "run `python evals/build_scenarios.py` and commit the result"


def test_trivial_baselines_stay_low():
    res = B.run_all(SCEN)
    assert res["always_ask"]["accuracy"] < 0.4
    assert res["always_decline"]["accuracy"] < 0.3
    assert res["random_valid"]["accuracy"] < 0.05
    assert res["keyword_rules"]["accuracy"] > res["always_ask"]["accuracy"]
