"""Guards the frozen tokenizer. If this fails you changed (or corrupted) a tokenizer that pretraining data
and checkpoints depend on. Do not 'fix' the pinned hash: a new tokenizer means re-tokenising and retraining."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from tokenizers import Tokenizer  # noqa: E402
from train_tokenizer import SPECIAL_TOKENS  # noqa: E402

from ivy_model import ModelConfig  # noqa: E402

FROZEN = ROOT / "tokenizer" / "ivy-bpe-32k"
PINNED_SHA256 = "10063dc2aa1ed7d66cec8edf35fc25d35133d133fc2d0cb119178cf49507eea0"


def test_hash_is_pinned_and_manifests_agree():
    digest = hashlib.sha256((FROZEN / "tokenizer.json").read_bytes()).hexdigest()
    assert digest == PINNED_SHA256
    assert json.loads((FROZEN / "MANIFEST.json").read_text())["sha256"] == PINNED_SHA256
    assert json.loads((FROZEN / "FROZEN.json").read_text())["sha256"] == PINNED_SHA256


def test_special_token_ids_and_vocab_match_model_configs():
    tok = Tokenizer.from_file(str(FROZEN / "tokenizer.json"))
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
    assert tok.get_vocab_size() == 32768 < 65536  # fits the uint16 token shards
    for name in ("40m", "300m", "1.5b"):
        assert ModelConfig.load(ROOT / "configs" / f"ivy-{name}.json").vocab_size == tok.get_vocab_size()


def test_digits_split_and_roundtrip():
    tok = Tokenizer.from_file(str(FROZEN / "tokenizer.json"))
    text = "In 2024 there were 1,234,567 items at 3.14159 each."
    toks = tok.encode(text).tokens
    assert all(len(t.lstrip("Ġ")) == 1 for t in toks if t.lstrip("Ġ").isdigit())
    for s in (text, '{"tool": "set_volume", "args": {"level": 30}}', "naïve café — 😀\n\tx", "<|user|>hi<|end|>"):
        assert tok.decode(tok.encode(s).ids, skip_special_tokens=False) == s
