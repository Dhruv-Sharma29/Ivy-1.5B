import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model.train import (TokenShard, load_checkpoint, make_optimizer,  # noqa: E402
                             save_checkpoint, train_steps)

CFG = ModelConfig(name="t", n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                  vocab_size=256, max_seq_len=32)
KW = dict(total_steps=8, peak_lr=1e-3, warmup=2, micro_bsz=2, accum=2, seed=7,
          device=torch.device("cpu"), amp=False, log=lambda *_: None)


def _setup(tmp_path):
    p = tmp_path / "toks.bin"
    np.random.default_rng(0).integers(0, 256, 5000).astype(np.uint16).tofile(p)
    torch.manual_seed(0)
    m = IvyLM(CFG)
    return m, make_optimizer(m, 1e-3), TokenShard(p, 32)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    m, opt, data = _setup(tmp_path)
    full, _ = train_steps(m, opt, data, CFG, start_step=0, end_step=8, **KW)

    m2, opt2, data2 = _setup(tmp_path)
    first, toks = train_steps(m2, opt2, data2, CFG, start_step=0, end_step=4, **KW)
    save_checkpoint(tmp_path / "c.pt", m2, opt2, 4, toks, CFG)

    m3, opt3, data3 = _setup(tmp_path)
    torch.manual_seed(999)  # prove state comes from the checkpoint, not from seeding
    step, toks3 = load_checkpoint(tmp_path / "c.pt", m3, opt3, CFG)
    assert (step, toks3) == (4, toks)
    rest, _ = train_steps(m3, opt3, data3, CFG, start_step=step, end_step=8, **KW)

    assert np.allclose(first + rest, full, atol=1e-6)


def test_checkpoint_rejects_different_config(tmp_path):
    m, opt, _ = _setup(tmp_path)
    save_checkpoint(tmp_path / "c.pt", m, opt, 0, 0, CFG)
    other = ModelConfig(name="t", n_layers=3, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                        vocab_size=256, max_seq_len=32)
    m2 = IvyLM(other)
    try:
        load_checkpoint(tmp_path / "c.pt", m2, make_optimizer(m2, 1e-3), other)
    except ValueError:
        return
    raise AssertionError("mismatched config was accepted")
