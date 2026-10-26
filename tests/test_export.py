import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model.export_hf import export, hf_config  # noqa: E402

CFG = ModelConfig(name="t", n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                  vocab_size=256, max_seq_len=64)


def test_logits_match_hf_llama(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    m = IvyLM(CFG).eval()
    export(m, tmp_path, dtype=torch.float32)
    cfg = json.loads((tmp_path / "config.json").read_text())
    hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(**{
        k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})).eval()
    from safetensors.torch import load_file
    missing, unexpected = hf.load_state_dict(load_file(tmp_path / "model.safetensors"), strict=False)
    assert not unexpected
    assert all("lm_head" in k or "rotary" in k for k in missing), missing
    hf.tie_weights()
    x = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        ours, _ = m(x)
        theirs = hf(x).logits
    assert torch.allclose(ours, theirs, atol=1e-4), (ours - theirs).abs().max()


def test_config_describes_tied_llama():
    c = hf_config(CFG)
    assert c["model_type"] == "llama" and c["tie_word_embeddings"] is True
