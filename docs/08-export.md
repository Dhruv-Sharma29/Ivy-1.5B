# Export

Ivy checkpoints export to the Hugging Face Llama format and from there to GGUF for llama.cpp, which is the intended on-device
runtime (Apple M4, 16 GB).

## Hugging Face format

```bash
PYTHONPATH=src .venv/bin/python -m ivy_model.export_hf --config configs/ivy-300m.json \
    --ckpt checkpoints/RUN/latest.pt --out export/ivy-300m-hf
```

Writes `config.json` (`LlamaForCausalLM`, tied embeddings) and `model.safetensors`. No weight permutation is needed because Ivy's
RoPE uses the same rotate-half convention as HF Llama. Then add the tokenizer:

```bash
cp tokenizer/ivy-bpe-32k/tokenizer.json export/ivy-300m-hf/
cat > export/ivy-300m-hf/tokenizer_config.json <<'EOF'
{"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|endoftext|>", "pad_token": "<|pad|>",
 "model_max_length": 2048, "clean_up_tokenization_spaces": false}
EOF
```

**Parity**: `tests/test_export.py` checks that the exported model's logits match `IvyLM`. On the trained 40M checkpoint the HF
model gives NLL 3.9730 against 3.9729, with 99.8% argmax agreement.

Only weights are exported; the optimizer state in `latest.pt` (most of its size) is dropped.

## GGUF / llama.cpp

Stock `convert_hf_to_gguf.py` rejects Ivy's tokenizer (`BPE pre-tokenizer was not recognized`). The tokenizer behaves exactly
like llama.cpp's built-in `starcoder` pre-tokenizer (single digits, then the GPT-2 pattern), so a one-line converter patch maps
it there. No C++ change is needed.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(cat /path/to/Ivy-1.5b/export/llama_cpp/LLAMA_CPP_COMMIT.txt)"
git apply /path/to/Ivy-1.5b/export/llama_cpp/ivy-pretokenizer.patch
python convert_hf_to_gguf.py /path/to/export/ivy-300m-hf --outfile ivy.gguf --outtype f16
```

Check tokenization against the HF tokenizer. The script calls `llama-tokenize --no-escape`; without that flag, LaTeX backslashes
produce false mismatches when testing by hand:

```bash
cmake -B build -DGGML_CUDA=OFF && cmake --build build --target llama-tokenize llama-cli llama-perplexity -j
python /path/to/Ivy-1.5b/export/llama_cpp/compare_tokenization.py --llama-dir . --gguf ivy.gguf --files sample-*.txt
```

## Verified, and not yet verified

| Check | Status |
|---|---|
| Conversion to GGUF (f16) | verified |
| Tokenization identical to HF on 2.47M held-out tokens + 12 edge cases (digits, whitespace, CRLF, Unicode, URLs, code, JSON, LaTeX) and special tokens | verified (Linux aarch64 CPU build) |
| GGUF f16 perplexity vs PyTorch | matches to 0.005% |
| Running on Apple M4 (Metal) | **not verified** |
| The Ivy app's own runtime, if it is not llama.cpp | **not verified**: it needs `starcoder`-equivalent pre-tokenization |
| Tool-use accuracy after quantisation (Q4/Q8) | **not measured** |

The llama.cpp converter changes often. The patch was tested at the commit in `LLAMA_CPP_COMMIT.txt`; re-check after upgrading.
