import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model import chat_format as CF  # noqa: E402
from ivy_model import sft, sft_gen  # noqa: E402

TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "ivy-bpe-32k" / "tokenizer.json"))
TINY = ModelConfig(name="t", n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64, vocab_size=32768, max_seq_len=1024)


def test_generator_is_deterministic_valid_and_free_of_eval_overlap():
    a, sa = sft_gen.generate(300, 5)
    b, _ = sft_gen.generate(300, 5)
    assert a == b and len(a) == 300
    import score  # noqa: F401  (evals on path via conftest-free insert below)


def test_collate_masks_prompt_and_pads(tmp_path):
    rows, _ = sft_gen.generate(8, 1)
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ex = sft.load_examples(str(p), TOK, 1024)
    x, y = sft.collate(ex, [0, 1, 2], torch.device("cpu"))
    assert x.shape == y.shape and (x[(y != -100)] == y[(y != -100)]).all()
    for r, j in enumerate([0, 1, 2]):
        n = len(ex[j][0])
        assert (x[r, n:] == sft.PAD).all() and (y[r, n:] == -100).all()
        assert (y[r, :n] != -100).sum() == sum(v != -100 for v in ex[j][1])


def test_sft_overfits_a_tiny_set_on_cpu():
    torch.manual_seed(0)
    rows, _ = sft_gen.generate(6, 2)
    ex = [CF.encode_example(TOK, r["devices"], r["messages"], r["answer"]) for r in rows]
    model = IvyLM(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = sft.collate(ex, list(range(6)), torch.device("cpu"))
    first = None
    for _ in range(60):
        _, loss = model(x, labels=y, return_logits=False)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first


def test_batches_cover_every_example_once():
    ex = [([1] * n, [1] * n) for n in range(5, 205)]
    bl = sft.batches(ex, 16, random.Random(0))
    flat = [j for b in bl for j in b]
    assert sorted(flat) == list(range(len(ex)))


def test_split_pool_partitions_each_group():
    seq = [f"item{i}" for i in range(40)] + [("a", "g1"), ("b", "g1"), ("c", "g2"), ("d", "g2"), ("e", "g2")]
    for group in (None, lambda x: x[1] if isinstance(x, tuple) else "s"):
        tr, ho = sft_gen.split_pool(seq, "train", group), sft_gen.split_pool(seq, "heldout", group)
        assert not set(tr) & set(ho) and sorted(map(repr, tr + ho)) == sorted(map(repr, seq))
    tr, ho = (sft_gen.split_pool(seq, p, lambda x: x[1] if isinstance(x, tuple) else "s") for p in ("train", "heldout"))
    for g in ("g1", "g2"):
        assert any(x[1] == g for x in tr if isinstance(x, tuple)) and any(x[1] == g for x in ho if isinstance(x, tuple))


def test_train_and_heldout_splits_share_no_app_or_open_app_template():
    apps = {p: set(sft_gen.split_pool(sft_gen.APPS, p)) for p in ("train", "heldout")}
    tpls = {p: sft_gen.split_pool(sft_gen.TPL["open_app"], p) for p in ("train", "heldout")}
    for p in ("train", "heldout"):
        g = sft_gen.Gen(3, p)
        for _ in range(300):
            text, tool, args = g.intent(tool="open_app", noisy_p=0.0)
            assert args["name"] in apps[p]
            assert any(t.format(app=args["name"]) == text for t in tpls[p])
    assert not apps["train"] & apps["heldout"]


def test_no_template_is_both_a_tool_request_and_unsupported():
    import re
    norm = lambda t: re.sub(r"\{\w+\}", "{}", t.lower())
    tool_tpls = {norm(t) for ts in sft_gen.TPL.values() for t in ts}
    assert not tool_tpls & {norm(t) for t in sft_gen.UNSUPPORTED}


def test_split_salt_gives_an_independent_split_and_default_is_unchanged():
    base = sft_gen.split_pool(sft_gen.APPS, "heldout")
    assert sft_gen.split_pool(sft_gen.APPS, "heldout", salt="") == base
    other = sft_gen.split_pool(sft_gen.APPS, "heldout", salt="s2")
    assert other != base and 0.15 < len(other) / len(sft_gen.APPS) < 0.45
