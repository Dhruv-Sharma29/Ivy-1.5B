# Architecture

Ivy is a dense decoder-only transformer. It is deliberately conventional: every component matches the Llama architecture, so a
checkpoint exports to the Hugging Face `LlamaForCausalLM` format without weight permutation and converts to GGUF for llama.cpp.

Source: [`src/ivy_model/model.py`](../src/ivy_model/model.py), [`src/ivy_model/config.py`](../src/ivy_model/config.py).

## Components

| Component | Choice | Notes |
|---|---|---|
| Normalisation | RMSNorm, pre-norm | computed in fp32, `norm_eps` = 1e-5 |
| Position encoding | RoPE, θ = 10,000 | rotate-half convention (identical to HF Llama) |
| Attention | grouped-query attention (GQA) | PyTorch `scaled_dot_product_attention`, causal |
| Feed-forward | SwiGLU | three projections, hidden size `d_ff` |
| Embeddings | tied input/output | `vocab_size` = 32,768 |
| Initialisation | normal(0, 0.02) | residual output projections (`o_proj`, `down_proj`) scaled by 1/√(2·layers) |

## Configurations

| Config | Layers | d_model | Heads | KV heads | d_ff | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| [`ivy-40m.json`](../configs/ivy-40m.json) | 8 | 512 | 8 | 2 | 1,536 | 40,903,168 |
| [`ivy-300m.json`](../configs/ivy-300m.json) | 18 | 1,024 | 16 | 4 | 4,096 | 307,270,656 |
| [`ivy-1.5b.json`](../configs/ivy-1.5b.json) | 28 | 2,048 | 16 | 4 | 6,400 | 1,461,831,680 |

All configs use a context length of 2,048 tokens. Parameter counts come from `ModelConfig.estimate_params()`, which the tests
check against the instantiated model. Each config has a 12-character `digest()`; checkpoints store it, so a checkpoint cannot be
loaded silently into the wrong architecture.

## Memory and speed features

| Feature | How to enable | Effect |
|---|---|---|
| Chunked cross-entropy | always on in training (`return_logits=False`) | never materialises the full `(tokens × vocab)` logits; each 4,096-token chunk is recomputed in backward |
| `torch.compile` per block | `train.py --compile` | +39% (40M) to +54% (300M) throughput on the training stack |
| Gradient checkpointing | `train.py --grad-ckpt` (or `bench.py --grad-ckpt`) | recomputes each block in backward: less activation memory, ~20% slower (1.5B: 1,217 vs 1,505 tokens/s) |
| Blockwise causal attention | `--attn-block N` | only useful on old PyTorch builds without fused kernels; leave at 0 on the current stack |
| bf16 autocast | default (`--precision bf16`) | `tf32` exists only as a workaround for an old stack and is slower on the current one |

## Measured throughput

Training machine: NVIDIA GB10, ~119 GB unified CPU/GPU memory, driver 580 / CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1.
Sequence length 2,048, bf16, compiled. The GPU is shared with other users' jobs, so numbers vary by up to several times.

| Model | Micro-batch × accumulation | Tokens/s | Peak memory |
|---|---|---:|---:|
| 40M | 8 × 4 | ~52,900 | 4.7 GB |
| 300M | 8 × 4 | ~16,800 | 18.9 GB |
| 300M | 4 × 8 | ~16,100 | 12.9 GB |
| 1.5B | 4 × 2, gradient checkpointing | ~1,570 | 27.3 GB |

What this means for planning: at ~1,570 tokens/s, 1B tokens take about 7.4 days on the 1.5B model, and 10B tokens about 74 days.
At 1.5B the peak memory is set by weights, gradients and AdamW state (~23 GB), not by activations. Raw benchmark files are in
[`reports/`](../reports) (`bench*.json`).

## Compatibility guarantees

- **HF parity**: `tests/test_export.py` checks that the exported HF Llama model produces the same logits as `IvyLM`. On the
  trained 40M checkpoint: NLL 3.9730 vs 3.9729, 99.8% argmax agreement.
- **GGUF parity**: an f16 GGUF matched PyTorch perplexity to 0.005% (see [Export](08-export.md)).
