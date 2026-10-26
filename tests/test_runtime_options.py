import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model.model import blockwise_causal_attention  # noqa: E402
from ivy_model.train import configure_runtime  # noqa: E402

TINY = ModelConfig(name="t", n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                   vocab_size=256, max_seq_len=64)


@pytest.mark.parametrize("t,block", [(64, 16), (64, 32), (50, 16), (33, 8), (16, 16)])
def test_blockwise_matches_sdpa_forward_and_backward(t, block):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, t, 8, dtype=torch.float64, requires_grad=True) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    gref = torch.autograd.grad(ref.pow(2).sum(), (q, k, v))
    out = blockwise_causal_attention(q, k, v, block)
    g = torch.autograd.grad(out.pow(2).sum(), (q, k, v))
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    for a, b in zip(g, gref):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_blockwise_is_causal():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 32, 8) for _ in range(3))
    base = blockwise_causal_attention(q, k, v, 8)
    v2 = v.clone()
    v2[:, :, 20:] += 5.0  # change only future positions
    out = blockwise_causal_attention(q, k, v2, 8)
    torch.testing.assert_close(out[:, :, :20], base[:, :, :20])


def test_model_loss_and_grads_identical_with_attn_block():
    torch.manual_seed(0)
    m = IvyLM(TINY)
    x = torch.randint(0, 256, (2, 48))
    _, l0 = m(x, labels=x)
    l0.backward()
    g0 = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    m.set_attn_block_size(16)
    _, l1 = m(x, labels=x)
    l1.backward()
    torch.testing.assert_close(l1, l0, atol=1e-6, rtol=1e-6)
    for n, p in m.named_parameters():
        torch.testing.assert_close(p.grad, g0[n], atol=1e-5, rtol=1e-4)


def test_attn_block_does_not_change_state_dict_keys():
    a = IvyLM(TINY).state_dict().keys()
    m = IvyLM(TINY)
    m.set_attn_block_size(16)
    assert list(m.state_dict().keys()) == list(a)


def test_grad_checkpointing_gives_identical_loss_and_grads():
    torch.manual_seed(0)
    m = IvyLM(TINY).train()
    x = torch.randint(0, 256, (2, 48))
    _, l0 = m(x, labels=x, return_logits=False)
    l0.backward()
    g0 = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    m.grad_checkpointing = True
    _, l1 = m(x, labels=x, return_logits=False)
    l1.backward()
    torch.testing.assert_close(l1, l0)
    for n, p in m.named_parameters():
        torch.testing.assert_close(p.grad, g0[n])


def test_configure_runtime_returns_amp_flag_and_validates():
    m = IvyLM(TINY)
    assert configure_runtime(m, "bf16", 0) is True
    assert configure_runtime(m, "tf32", 16) is False
    with pytest.raises(ValueError):
        configure_runtime(m, "fp8", 0)
    with pytest.raises(ValueError):
        m.set_attn_block_size(-1)
