import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model.train import TokenShard, evaluate, make_optimizer, train_steps  # noqa: E402

TINY = ModelConfig(name="t", n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64, vocab_size=64, max_seq_len=32)


def shard(tmp_path, n=4000):
    p = tmp_path / "v.bin"
    np.random.default_rng(0).integers(0, 64, size=n).astype(np.uint16).tofile(p)
    return TokenShard(p, 32)


def test_evaluate_is_deterministic_and_restores_train_mode(tmp_path):
    torch.manual_seed(0)
    m = IvyLM(TINY)
    m.train()
    s = shard(tmp_path)
    a = evaluate(m, s, n_batches=4, bsz=2, device=torch.device("cpu"), amp=False)
    b = evaluate(m, s, n_batches=4, bsz=2, device=torch.device("cpu"), amp=False)
    assert a == b and m.training
    assert abs(a - np.log(64)) < 0.5  # untrained model is near uniform


def test_history_records_every_step(tmp_path):
    torch.manual_seed(0)
    m = IvyLM(TINY)
    opt = make_optimizer(m, 1e-3)
    hist: list = []
    _, toks = train_steps(m, opt, shard(tmp_path), TINY, start_step=0, end_step=3, total_steps=3, peak_lr=1e-3,
                          warmup=1, micro_bsz=2, accum=2, seed=1, device=torch.device("cpu"), amp=False,
                          log=lambda *_: None, history=hist)
    assert [h["step"] for h in hist] == [1, 2, 3] and toks == 3 * 2 * 2 * 32
    assert all(np.isfinite(h["loss"]) for h in hist)
    json.dumps(hist)  # serialisable for metrics.jsonl
