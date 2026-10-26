import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import cleaning as C  # noqa: E402

VOCAB = [f"w{i}x" for i in range(5000)]


def doc(seed: int, n: int = 300) -> str:
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) for _ in range(n))


# ---- scrubbing (fake credentials assembled from pieces so no real-looking secret sits in the repo)
FAKE = {
    "aws_key": "AK" + "IA" + "IOSFODNN7" + "EXAMPLE",
    "github_token": "gh" + "p_" + "a" * 36,
    "hf_token": "hf" + "_" + "b" * 34,
    "openai_key": "sk" + "-" + "c" * 30,
    "anthropic_key": "sk" + "-ant-" + "d" * 30,
    "slack_token": "xo" + "xb-" + "1234567890-abcdef",
    "jwt": "ey" + "J" + "h" * 12 + "." + "p" * 12 + "." + "s" * 12,
    "email": "someone" + "@" + "example.org",
}


@pytest.mark.parametrize("kind", list(FAKE))
def test_scrub_each_token_kind(kind):
    out, counts = C.scrub_secrets(f"config value is {FAKE[kind]} here")
    assert FAKE[kind] not in out and counts.get(kind) == 1
    assert out.startswith("config value is ") and out.endswith(" here")


def test_scrub_private_key_block_and_assignment():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\ndef\n-----END RSA PRIVATE KEY-----"
    out, counts = C.scrub_secrets(f'a\n{pem}\nb api_key = "abcdefghijklmnop"')
    assert "MIIabc" not in out and "abcdefghijklmnop" not in out
    assert 'api_key = "<SECRET>"' in out and counts["private_key"] == 1 and counts["assigned_secret"] == 1


def test_scrub_leaves_normal_text_and_is_idempotent():
    text = "The password policy requires 12 characters. Use token = None. Contact the admin."
    assert C.scrub_secrets(text) == (text, {})
    once, _ = C.scrub_secrets(f"x {FAKE['hf_token']} y")
    assert C.scrub_secrets(once) == (once, {})


# ---- near-duplicates
def test_near_dup_catches_exact_and_small_edits_not_unrelated():
    idx = C.NearDupIndex()
    a = doc(1)
    assert not idx.is_duplicate(a)
    assert idx.is_duplicate(a)  # exact
    edited = a.split()
    edited[150] = "CHANGED"
    assert idx.is_duplicate(" ".join(edited))  # one word out of 300 changed
    assert idx.is_duplicate(a + " trailing footer text here")  # appended boilerplate
    assert not idx.is_duplicate(doc(2))  # unrelated


def test_near_dup_false_positive_rate_on_unrelated_docs():
    idx = C.NearDupIndex()
    flagged = sum(idx.is_duplicate(doc(1000 + i)) for i in range(150))
    assert flagged == 0


def test_near_dup_half_overlap_is_not_flagged_at_threshold_08():
    idx = C.NearDupIndex()
    a = doc(5, 400).split()
    assert not idx.is_duplicate(" ".join(a))
    mixed = a[:200] + doc(6, 200).split()  # ~50% shared content
    assert not idx.is_duplicate(" ".join(mixed))


def test_minhash_estimate_tracks_true_jaccard_and_is_deterministic():
    idx, idx2 = C.NearDupIndex(seed=3), C.NearDupIndex(seed=3)
    a = doc(7, 500).split()
    b = a[:350] + doc(8, 150).split()
    sa, sb = idx.signature(" ".join(a)), idx.signature(" ".join(b))
    ga = {" ".join(a[i:i + 5]) for i in range(len(a) - 4)}
    gb = {" ".join(b[i:i + 5]) for i in range(len(b) - 4)}
    true_j = len(ga & gb) / len(ga | gb)
    assert abs(idx.jaccard_estimate(sa, sb) - true_j) < 0.15
    assert np.array_equal(sa, idx2.signature(" ".join(a)))


def test_near_dup_handles_empty_and_short_text():
    idx = C.NearDupIndex()
    assert not idx.is_duplicate("") and not idx.is_duplicate("   ")
    assert not idx.is_duplicate("hello world")
    assert idx.is_duplicate("Hello, World!")  # normalised the same


# ---- decontamination
def test_ngram_blocklist_flags_shared_13gram_and_counts_short_items():
    bench = " ".join(f"b{i}" for i in range(40))
    bl = C.NgramBlocklist([bench, "open Spotify"], n=13)
    assert bl.skipped_short == 1
    assert bl.contaminated("intro text " + " ".join(f"b{i}" for i in range(10, 30)) + " outro")
    assert not bl.contaminated(" ".join(f"b{i}" for i in range(0, 12)))  # only 12 consecutive words
    assert not bl.contaminated(doc(9))
    assert not C.NgramBlocklist([], n=13).contaminated(doc(9))


def test_find_eval_overlap_normalises_and_uses_jaccard():
    evals = {"e1": "open Spotify", "e2": "set the volume to 30 on my phone"}
    hits = C.find_eval_overlap(["OPEN   spotify!", "set the volume to 30 on my phone please", "play jazz"], evals)
    assert {(h["candidate_index"], h["eval_id"]) for h in hits} == {(0, "e1"), (1, "e2")}
    assert C.find_eval_overlap(["launch Chrome"], evals) == []


def test_ivy_eval_utterances_load_and_cli_check_sft(tmp_path):
    ev = C.load_ivy_eval_utterances()
    assert len(ev) >= 233 and any(k.startswith("scenarios_v1:") for k in ev)
    first = next(iter(ev.values()))
    bad, good = tmp_path / "bad.jsonl", tmp_path / "good.jsonl"
    bad.write_text(json.dumps({"user": first.upper()}) + "\n")
    good.write_text(json.dumps({"user": "zzzz unrelated qqqq request"}) + "\n")
    run = lambda f: subprocess.run([sys.executable, "-m", "ivy_model.cleaning", "check-sft", str(f)],
                                   cwd=ROOT / "src", capture_output=True, text=True)
    assert run(bad).returncode == 1 and run(good).returncode == 0


def test_check_sft_reads_chat_messages_and_refuses_empty_input(tmp_path):
    ev = C.load_ivy_eval_utterances()
    v2 = next(v for k, v in ev.items() if k.startswith("scenarios_v2:") and len(v.split()) >= 4)
    run = lambda f: subprocess.run([sys.executable, "-m", "ivy_model.cleaning", "check-sft", str(f)], cwd=ROOT / "src", capture_output=True, text=True)
    bad, good, empty = tmp_path / "bad.jsonl", tmp_path / "good.jsonl", tmp_path / "empty.jsonl"
    bad.write_text(json.dumps({"messages": [{"role": "user", "content": v2.upper()}], "answer": {}}) + "\n")
    good.write_text(json.dumps({"messages": [{"role": "user", "content": "zzzz unrelated qqqq request"}], "answer": {}}) + "\n")
    empty.write_text(json.dumps({"devices": [], "answer": {}}) + "\n")
    assert run(bad).returncode == 1 and run(good).returncode == 0
    r = run(empty)
    assert r.returncode != 0 and "nothing to check" in (r.stdout + r.stderr)
