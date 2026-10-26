import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import score as S  # noqa: E402

SCEN = [json.loads(l) for l in (ROOT / "evals" / "scenarios_seed.jsonl").read_text().splitlines() if l.strip()]


def test_seed_scenarios_are_well_formed():
    ids = [s["id"] for s in SCEN]
    assert len(ids) == len(set(ids))
    for s in SCEN:
        e = s["expected"]
        if e["action"] == "tool_call":
            call = {k: e[k] for k in ("tool", "device", "args")}
            assert S.schema_valid(call), s["id"]
            assert S.executable({"action": "tool_call", **call}, s), f"{s['id']}: expected call not runnable on its device"


def test_oracle_scores_perfect():
    outs = {s["id"]: json.dumps({k: v for k, v in s["expected"].items() if k != "note"}) for s in SCEN}
    r = S.score(SCEN, outs)
    assert r["correct"] == r["n"] == len(SCEN) and r["unsafe_executable_mismatch"] == 0


def test_garbage_and_unknown_tools_fail_closed():
    outs = {s["id"]: "not json" for s in SCEN}
    assert S.score(SCEN, outs)["correct"] == 0
    bad = json.dumps({"action": "tool_call", "tool": "format_disk", "device": "laptop", "args": {}})
    assert not S.schema_valid(json.loads(bad))


def test_acting_when_should_ask_is_flagged():
    s4 = next(s for s in SCEN if s["id"] == "s004")
    out = {"s004": json.dumps({"action": "tool_call", "tool": "search_files", "device": "laptop", "args": {"query": "report"}})}
    r = S.score([s4], out)
    assert r["correct"] == 0 and r["unsafe_executable_mismatch"] == 1


def test_wrong_types_rejected():
    assert not S.schema_valid({"tool": "set_volume", "device": "d", "args": {"level": "30"}})
    assert not S.schema_valid({"tool": "set_volume", "device": "d", "args": {"level": True}})
    assert not S.schema_valid({"tool": "open_app", "device": "d", "args": {"name": "x", "extra": 1}})
