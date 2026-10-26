"""Export an IvyLM checkpoint as a HF Llama checkpoint (safetensors) for llama.cpp conversion.

Our RoPE uses the rotate-half convention, identical to HF Llama, so no weight permutation is needed.
tests/test_export.py checks logits parity against transformers' LlamaForCausalLM.

    python -m ivy_model.export_hf --config configs/ivy-300m.json --ckpt ckpt/latest.pt --out export/ivy-300m-hf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import IvyLM


def hf_config(c: ModelConfig) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": c.d_model, "intermediate_size": c.d_ff, "num_hidden_layers": c.n_layers,
        "num_attention_heads": c.n_heads, "num_key_value_heads": c.n_kv_heads,
        "head_dim": c.head_dim, "vocab_size": c.vocab_size, "max_position_embeddings": c.max_seq_len,
        "rms_norm_eps": c.norm_eps, "rope_theta": c.rope_theta, "hidden_act": "silu",
        "tie_word_embeddings": c.tie_embeddings, "attention_bias": False, "mlp_bias": False,
        "bos_token_id": None, "eos_token_id": 0, "pad_token_id": 1,
    }


def to_hf_state_dict(model: IvyLM) -> dict[str, torch.Tensor]:
    sd = {"model.embed_tokens.weight": model.embed.weight, "model.norm.weight": model.final_norm.weight}
    for i, b in enumerate(model.blocks):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = b.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = b.mlp_norm.weight
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[p + f"self_attn.{n}.weight"] = getattr(b.attn, n).weight
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[p + f"mlp.{n}.weight"] = getattr(b.mlp, n).weight
    if not model.c.tie_embeddings:
        sd["lm_head.weight"] = model.lm_head.weight
    return {k: v.detach().cpu().contiguous() for k, v in sd.items()}


def export(model: IvyLM, out: str | Path, dtype: torch.dtype = torch.bfloat16) -> None:
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.to(dtype) for k, v in to_hf_state_dict(model).items()}
    save_file(sd, out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(hf_config(model.c), indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = ModelConfig.load(a.config)
    model = IvyLM(cfg)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck["cfg_digest"] != cfg.digest():
        raise SystemExit("checkpoint/config mismatch")
    model.load_state_dict(ck["model"])
    export(model, a.out)
    print(f"exported to {a.out}; add tokenizer.json, then convert with llama.cpp convert_hf_to_gguf.py")


if __name__ == "__main__":
    main()
