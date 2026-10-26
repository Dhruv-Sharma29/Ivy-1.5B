import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import IvyLM, ModelConfig  # noqa: E402

TINY = ModelConfig(name="t", n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                   vocab_size=256, max_seq_len=64)


@pytest.mark.parametrize("name", ["ivy-40m", "ivy-300m", "ivy-1.5b"])
def test_estimate_matches_instantiated(name):
    cfg = ModelConfig.load(ROOT / "configs" / f"{name}.json")
    with torch.device("meta"):
        m = IvyLM(cfg)
    assert m.num_params() == cfg.estimate_params()


def test_param_targets():
    n = {k: ModelConfig.load(ROOT / "configs" / f"ivy-{k}.json").estimate_params() for k in ("40m", "300m", "1.5b")}
    assert 35e6 < n["40m"] < 50e6
    assert 280e6 < n["300m"] < 330e6
    assert 1.4e9 < n["1.5b"] < 1.6e9


def test_causality_future_tokens_do_not_leak():
    torch.manual_seed(0)
    m = IvyLM(TINY).eval()
    a = torch.randint(0, 256, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 256  # change only the future
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def test_label_shift_and_ignore_index():
    torch.manual_seed(0)
    m = IvyLM(TINY).eval()
    x = torch.randint(0, 256, (2, 12))
    logits, loss = m(x, labels=x)
    ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-5)
    masked = x.clone()
    masked[:, :6] = -100
    _, loss_m = m(x, labels=masked)
    ref_m = torch.nn.functional.cross_entropy(
        logits[:, 5:-1].reshape(-1, 256), x[:, 6:].reshape(-1))
    assert torch.allclose(loss_m, ref_m, atol=1e-5)


def test_tied_embeddings_share_storage():
    m = IvyLM(TINY)
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr()


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    m = IvyLM(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 256, (4, 32))
    first = None
    for _ in range(150):
        _, loss = m(x, labels=x)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_rejects_overlong_sequence():
    with pytest.raises(ValueError):
        IvyLM(TINY)(torch.zeros(1, 65, dtype=torch.long))


def test_chunked_loss_matches_reference_loss_and_grads():
    torch.manual_seed(0)
    m = IvyLM(TINY)
    x = torch.randint(0, 256, (3, 40))
    lab = x.clone()
    lab[:, :7] = -100
    lab[1, 20:25] = -100
    _, ref = m(x, labels=lab)
    ref.backward()
    g_ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    import ivy_model.model as mm
    orig = mm.chunked_cross_entropy
    mm.chunked_cross_entropy = lambda h, w, y, chunk=4096: orig(h, w, y, chunk=16)  # force many chunks
    try:
        logits, fast = m(x, labels=lab, return_logits=False)
    finally:
        mm.chunked_cross_entropy = orig
    assert logits is None
    assert torch.allclose(fast, ref, atol=1e-5)
    fast.backward()
    for n, p in m.named_parameters():
        assert torch.allclose(p.grad, g_ref[n], atol=1e-5), n


def test_all_ignored_labels_give_zero_not_nan():
    m = IvyLM(TINY)
    x = torch.randint(0, 256, (1, 8))
    _, loss = m(x, labels=torch.full_like(x, -100), return_logits=False)
    assert loss.item() == 0.0
