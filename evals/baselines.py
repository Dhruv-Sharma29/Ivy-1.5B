"""Scripted baselines for the Ivy tool eval. A trained model has to beat these to show it learned anything.

    python evals/baselines.py [--scenarios evals/scenarios_v1.jsonl]

always_ask / always_decline: constant policies. majority: the most common expected action.
random_valid: schema-valid call with placeholder args on a random capable device (tests the scorer, ~0%).
keyword_rules: regex intent + argument extraction + device routing. CAUTION: it was written after the
  scenarios and some of its phrases ("fire up", "track down", "get rid of", ...) were taken from them, so its
  score on scenarios_v1 is optimistic (it is partly fitted to the test set). Treat it as an upper-ish bar for
  what rules can do on this phrasing, not as an independent measurement; it will be weaker on unseen wording.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import score as S

HERE = Path(__file__).parent

CANCEL = re.compile(r"never ?mind|cancel|forget (it|what)|scratch that|abort|\bstop\b|changed my mind|"
                    r"ignore my last|on second thought|do not|don't|hold on|skip that", re.I)
INTENTS = [  # order matters: first match wins
    ("run_project_tests", re.compile(r"\btests?\b|test suite", re.I)),
    ("delete_file", re.compile(r"\b(delete|remove|get rid of)\b", re.I)),
    ("move_file", re.compile(r"\b(move|shift)\b", re.I)),
    ("set_volume", re.compile(r"\bvolume\b|turn it up", re.I)),
    ("send_notification", re.compile(r"\bnotif|\bping\b|\bremind", re.I)),
    ("read_file", re.compile(r"\b(read|show me|contents?|print|inside|what does|what's in|dump|need to see|"
                             r"open the file)\b", re.I)),
    ("search_files", re.compile(r"\b(search|find|look(ing)? (for|in)|locate|where is|is there a file|track down)\b", re.I)),
    ("open_app", re.compile(r"\b(open|launch|start|bring|fire up)\b", re.I)),
]
QUOTED = re.compile(r"""["']([^"']+)["']""")
PATH = re.compile(r"(?:~|\.{0,2}/|[A-Za-z]:\\)[^\s'\"]*")
NUMBER = re.compile(r"\b(\d{1,3})\b")
DEVICE_NAMES = ["laptop", "phone", "tablet", "watch", "desktop", "gpu-box"]


def _args(tool: str, text: str) -> dict | None:
    """Heuristic argument extraction; None means 'a required argument seems to be missing'."""
    quoted, paths = QUOTED.findall(text), PATH.findall(text)
    if tool in ("read_file", "delete_file"):
        v = (quoted or paths or [None])[0]
        return {"path": v} if v else None
    if tool == "run_project_tests":
        v = (quoted or paths or [None])[0]
        return {"directory": v} if v else None
    if tool == "move_file":
        vals = quoted if len(quoted) >= 2 else paths
        return {"source": vals[0], "destination": vals[1]} if len(vals) >= 2 else None
    if tool == "set_volume":
        m = NUMBER.search(text)
        return {"level": int(m.group(1))} if m else None
    if tool == "send_notification":
        if quoted:
            return {"message": quoted[0]}
        m = re.search(r"(?:notify me|notification)\s*:\s*(.+)$", text, re.I)
        return {"message": m.group(1).strip()} if m else None
    if tool == "search_files":
        m = re.search(r"(?:search|find|look for|locate|track down)(?: for)?\s+(\S+)", text, re.I)
        q = (quoted or [m.group(1) if m else None])[0]
        if not q or q.lower() in ("it", "the", "that", "for"):
            return None
        d = re.search(r"\bin\s+(\S+)", text)
        return {"query": q, **({"directory": d.group(1)} if d else {})}
    if tool == "open_app":
        m = re.search(r"(?:open|launch|start|bring up|fire up)\s+(?:the\s+)?(.+?)(?:\s+app)?(?:\s+on\b.*)?$", text, re.I)
        name = (quoted or [m.group(1) if m else None])[0]
        return {"name": name} if name and name.lower() not in ("an app", "it", "my editor") else None
    return None


def keyword_rules(s: dict) -> dict:
    text = s["user"]
    if CANCEL.search(text) and not QUOTED.search(text):
        return {"action": "decline"}
    tool = next((t for t, rx in INTENTS if rx.search(text)), None)
    if tool is None:
        return {"action": "decline"}
    capable = [d["id"] for d in s["devices"] if tool in d["tools"]]
    if not capable:
        return {"action": "decline"}
    args = _args(tool, text)
    if args is None:
        return {"action": "ask_user"}
    named = next((d for d in DEVICE_NAMES if re.search(rf"\b{re.escape(d)}\b", text)), None)
    if named in capable:
        device = named
    elif len(capable) == 1:
        device = capable[0]
    else:
        return {"action": "ask_user"}
    return {"action": "tool_call", "tool": tool, "device": device, "args": args}


def random_valid(s: dict, rng: random.Random) -> dict:
    d = rng.choice(s["devices"])
    tool = rng.choice(d["tools"])
    spec = S.CATALOGUE[tool]
    args = {k: (0 if t == "integer" else "x") for k, t in spec["required"].items()}
    return {"action": "tool_call", "tool": tool, "device": d["id"], "args": args}


def run_all(scenarios: list[dict]) -> dict[str, dict]:
    rng = random.Random(0)
    majority = Counter(s["expected"]["action"] for s in scenarios).most_common(1)[0][0]
    fixed = {"tool_call": None, "ask_user": {"action": "ask_user"}, "decline": {"action": "decline"}}
    policies = {
        "always_ask": lambda s: {"action": "ask_user"},
        "always_decline": lambda s: {"action": "decline"},
        f"majority({majority})": (lambda s: fixed[majority]) if fixed[majority] else (lambda s: {"action": "decline"}),
        "random_valid": lambda s: random_valid(s, rng),
        "keyword_rules": keyword_rules,
    }
    return {name: S.score(scenarios, {s["id"]: json.dumps(fn(s)) for s in scenarios}) for name, fn in policies.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(HERE / "scenarios_v1.jsonl"))
    ap.add_argument("--details", action="store_true", help="per-category table for keyword_rules")
    a = ap.parse_args()
    sc = [json.loads(l) for l in Path(a.scenarios).read_text().splitlines() if l.strip()]
    res = run_all(sc)
    print(f"{len(sc)} scenarios")
    for name, r in res.items():
        print(f"{name:22s} acc {r['accuracy']:.3f}  95% CI {r['accuracy_95ci']}  unsafe_mismatch {r['unsafe_executable_mismatch']}")
    if a.details:
        for cat, v in res["keyword_rules"]["by_category"].items():
            print(f"  {cat:24s} {v['correct']:3d}/{v['n']}")


if __name__ == "__main__":
    main()
