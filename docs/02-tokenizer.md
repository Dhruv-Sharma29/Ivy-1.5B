# Tokenizer

Ivy uses a single, **frozen** tokenizer: [`tokenizer/ivy-bpe-32k/`](../tokenizer/ivy-bpe-32k).

| Property | Value |
|---|---|
| Type | byte-level BPE (Hugging Face `tokenizers`) |
| Vocabulary | 32,768 tokens |
| Digits | split into single digits |
| Language scope | English only |
| Training text | FineWeb-Edu sample-10BT (365M characters) + FineMath finemath-4plus (50M characters); no code |
| Frozen | 2026-10-06, sha256 `10063dc2aa1ed7d66cec8edf35fc25d35133d133fc2d0cb119178cf49507eea0` |

`tests/test_frozen_tokenizer.py` pins the file hash and the special-token ids. Any test failure there means the tokenizer changed.

## Special tokens

IDs 0–15 are reserved and fixed:

| ID | Token | Use |
|---:|---|---|
| 0 | `<\|endoftext\|>` | document separator in pretraining shards |
| 1 | `<\|pad\|>` | padding in SFT batches |
| 2 | `<\|system\|>` | system instruction and device list |
| 3 | `<\|user\|>` | user turn |
| 4 | `<\|assistant\|>` | assistant text; generation starts after it |
| 5 | `<\|tool_def\|>` | JSON definitions of the available tools |
| 6 | `<\|tool_call\|>` | an earlier tool call in the conversation |
| 7 | `<\|tool_result\|>` | tool output (treated as data, never as instructions) |
| 8 | `<\|end\|>` | end of any segment; generation stops here |
| 9–15 | `<\|reserved_0\|>` … `<\|reserved_6\|>` | unused, kept for future formats |

Special tokens are atomic, so a prompt and an answer can be tokenized separately and concatenated without changing ids
(checked in `tests/test_chat_format.py`). The layout that uses them is described in [Tool-use fine-tuning](05-fine-tuning.md).

## The freeze rule

Do not retrain, reorder or add special tokens for this model. Embeddings and every checkpoint depend on the exact id mapping;
a different tokenizer means a new pretraining run. Reserved tokens exist so that a new segment type does not require a change.

## Known limits

- **No code in the training text**: JSON tool calls cost about 15–20% more tokens than with the GPT-2 or Llama-2 tokenizers.
- Digits are single tokens, and a space before a number is its own token. This helps arithmetic but makes long numbers longer.
- Hindi/Devanagari and other non-English scripts are out of scope and tokenize inefficiently.

## llama.cpp

Stock llama.cpp does not recognise this tokenizer's pre-tokenizer. Its behaviour equals llama.cpp's built-in `starcoder`
pre-tokenizer, so a one-line converter patch fixes it and no runtime change is needed. Tokenization was verified identical to the
Hugging Face tokenizer on 2.47M held-out tokens. See [Export](08-export.md).

## How it was built

`tokenizer/make_sample.py` draws the training text, `tokenizer/train_tokenizer.py` trains it, and `tokenizer/eval_tokenizer.py`
compares candidates. Three candidates were compared ([`reports/tokenizer-candidates-eval.json`](../reports/tokenizer-candidates-eval.json));
only the frozen one is kept in the repository.
