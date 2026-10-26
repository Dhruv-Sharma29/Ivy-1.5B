"""Model configuration. Plain dataclass + JSON so configs are diffable and hashable."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int = 32768
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """Closed-form count; tests assert this matches the instantiated model."""
        hd = self.head_dim
        attn = self.d_model * (self.n_heads * hd) * 2 + self.d_model * (self.n_kv_heads * hd) * 2
        mlp = 3 * self.d_model * self.d_ff
        norms = 2 * self.d_model
        emb = self.vocab_size * self.d_model
        head = 0 if self.tie_embeddings else emb
        return self.n_layers * (attn + mlp + norms) + emb + head + self.d_model

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
