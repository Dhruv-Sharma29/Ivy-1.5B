"""Dense decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA, tied embeddings.

Deliberately conventional so the exported checkpoint maps onto the Llama architecture
that llama.cpp already supports (verified at export time; see docs/08-export.md).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)  # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd); rotate-half convention (matches Llama/HF).
    t = x.shape[2]
    cos = torch.cat([cos[:t], cos[:t]], dim=-1)[None, None].to(x.dtype)
    sin = torch.cat([sin[:t], sin[:t]], dim=-1)[None, None].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def blockwise_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int) -> torch.Tensor:
    """Causal attention computed per query block; each block only sees keys up to its own end.
    Skips the fully masked upper triangle, which matters when SDPA falls back to the math kernel
    (e.g. GPU architectures the installed torch does not ship flash/mem-efficient kernels for)."""
    t = q.size(2)
    outs = []
    for i in range(0, t, block):
        e = min(i + block, t)
        mask = torch.ones(e - i, e, dtype=torch.bool, device=q.device).tril(diagonal=i)
        outs.append(F.scaled_dot_product_attention(q[:, :, i:e], k[:, :, :e], v[:, :, :e], attn_mask=mask))
    return torch.cat(outs, dim=2)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig) -> None:
        super().__init__()
        self.c = c
        self.block_size = 0  # runtime option (not part of the architecture); 0 = single SDPA call
        hd = c.head_dim
        self.q_proj = nn.Linear(c.d_model, c.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(c.n_heads * hd, c.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.c
        q = self.q_proj(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = c.n_heads // c.n_kv_heads
        if rep > 1:  # explicit expand: portable across torch versions/kernels
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if self.block_size and t > self.block_size:
            y = blockwise_causal_attention(q, k, v, self.block_size)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down_proj = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.attn = Attention(c)
        self.mlp_norm = RMSNorm(c.d_model, c.norm_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


def _chunk_ce_sum(h: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = (h @ weight.t()).float()
    return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")


def chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                          chunk: int = 4096) -> torch.Tensor:
    """Mean CE over non-ignored positions without materialising full (N, vocab) fp32 logits.
    Each chunk's logits are recomputed in backward (checkpoint), so peak extra memory is one chunk."""
    h = hidden.reshape(-1, hidden.size(-1))
    y = labels.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            _chunk_ce_sum, h[i:i + chunk], weight, y[i:i + chunk], use_reentrant=False)
    return total / (y != -100).sum().clamp(min=1)


class IvyLM(nn.Module):
    def __init__(self, c: ModelConfig) -> None:
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_cache(c.max_seq_len, c.head_dim, c.rope_theta)
        self.register_buffer("_cos", cos, persistent=False)
        self.register_buffer("_sin", sin, persistent=False)
        self.grad_checkpointing = False
        self.apply(self._init)
        # GPT-2-style scaled init on residual output projections.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_attn_block_size(self, block: int) -> None:
        """Opt in to blockwise causal attention (0 disables). Same maths, different kernel schedule."""
        if block < 0:
            raise ValueError("attention block size must be >= 0")
        for blk in self.blocks:
            blk.attn.block_size = block

    def compile_blocks(self) -> None:
        """In-place torch.compile of each block (state_dict keys unchanged)."""
        for blk in self.blocks:
            blk.compile()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                return_logits: bool = True):
        """labels are aligned with input_ids (HF style); the next-token shift happens here.
        Positions with label -100 are excluded from the loss.
        return_logits=False (training fast path) skips building full logits and returns (None, loss)."""
        t = input_ids.shape[1]
        if t > self.c.max_seq_len:
            raise ValueError(f"sequence {t} exceeds max_seq_len {self.c.max_seq_len}")
        x = self.embed(input_ids)
        cos, sin = self._cos[:t], self._sin[:t]
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.final_norm(x)
        if labels is not None and not return_logits:
            return None, chunked_cross_entropy(x[:, :-1], self.lm_head.weight, labels[:, 1:])
        logits = self.lm_head(x)
        if labels is None:
            return logits, None
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once
