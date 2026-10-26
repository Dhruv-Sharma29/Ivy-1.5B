import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from train_tokenizer import SPECIAL_TOKENS, train  # noqa: E402

SAMPLES = [
    "def f(x):\n    return {'a': [1, 2, 3], 'path': '/Users/me/Documents/file.txt'}\n",
    "नमस्ते दुनिया — Hinglish: kya haal hai? 😀",
    '{"tool": "open_app", "args": {"name": "Safari"}}',
    "\t\tindented\r\nwindows newline  trailing   ",
]


def test_roundtrip_and_special_ids(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("\n".join(SAMPLES * 200))
    tok = train([str(f)], vocab_size=600)
    for s in SAMPLES:
        assert tok.decode(tok.encode(s).ids) == s  # byte-level BPE must be lossless
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))


def test_split_digits_option(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("\n".join(["the year 2024 and 1234567 apples, price 3.14159"] * 300 + SAMPLES * 50))
    plain = train([str(f)], vocab_size=600)
    split = train([str(f)], vocab_size=600, split_digits=True)
    s = "in 2024 there were 1234567 items"
    assert any(len(t.strip("Ġ")) > 1 and t.strip("Ġ").isdigit() for t in plain.encode(s).tokens)
    assert all(len(t.strip("Ġ")) <= 1 for t in split.encode(s).tokens if t.strip("Ġ").isdigit())
    for tok in (plain, split):
        for x in SAMPLES + [s]:
            assert tok.decode(tok.encode(x).ids) == x
