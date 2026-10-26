# Exporting Ivy to GGUF with llama.cpp

**Stock `convert_hf_to_gguf.py` rejects the frozen tokenizer** (`BPE pre-tokenizer was not recognized`). The tokenizer
(`tokenizer/ivy-bpe-32k`) is byte-level BPE with individual-digit splitting. llama.cpp's built-in `starcoder`
pre-tokenizer applies the same two steps (`\p{N}` per digit, then the GPT-2 pattern), so the fix is a one-line mapping of our
tokenizer's check-hash to `"starcoder"` in the Python converter. **No change to the C++ runtime is needed.**

- `ivy-pretokenizer.patch`: the converter patch (hash `c4e6f0fc…2e17` = the frozen tokenizer as seen by the converter).
- `LLAMA_CPP_COMMIT.txt`: llama.cpp commit it was tested against (2026-10-06). The converter changes often; re-check on upgrade.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git apply /path/to/export/llama_cpp/ivy-pretokenizer.patch
# HF export: python -m ivy_model.export_hf ..., then copy tokenizer.json and write tokenizer_config.json (see below)
python convert_hf_to_gguf.py EXPORT_DIR --outfile ivy.gguf --outtype f16
cmake -B build -DGGML_CUDA=OFF && cmake --build build --target llama-tokenize llama-cli llama-perplexity -j
python export/llama_cpp/compare_tokenization.py --llama-dir . --gguf ivy.gguf --files data/sample/heldout-*.txt
```

`tokenizer_config.json` in the export dir: `{"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|endoftext|>",
"pad_token": "<|pad|>", "model_max_length": 2048, "clean_up_tokenization_spaces": false}`.

## What has and has not been verified (2026-10-06, Linux aarch64 CPU build)

- Verified: conversion succeeds; llama.cpp tokenization is identical to the HF tokenizer on 2.47M held-out tokens
  (1,830,256 web + 642,074 math), 12 edge cases (digits, whitespace, CRLF, Unicode, URLs, code, JSON, LaTeX) and special-token parsing.
- NOT verified here: running on the M4 (Metal), the app's own runtime if it is not llama.cpp, and quantised (Q4/Q8) quality.
  If the Ivy app embeds its own tokenizer or an older llama.cpp, it needs the same `starcoder`-equivalent pre-tokenization.
