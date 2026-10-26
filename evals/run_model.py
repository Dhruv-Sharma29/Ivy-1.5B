"""Run a checkpoint over eval scenarios (greedy decoding) and write outputs for evals/score.py.

    PYTHONPATH=src .venv/bin/python evals/run_model.py --config configs/ivy-40m.json --ckpt checkpoints/sft-X/latest.pt \
        --scenarios evals/scenarios_v2.jsonl --out outputs.jsonl
    python evals/score.py --scenarios evals/scenarios_v2.jsonl --outputs outputs.jsonl

Prompts come from ivy_model.chat_format (the same renderer as SFT). Greedy, max 160 new tokens, stops at <|end|>. There is no KV
cache, so it is O(T^2) per scenario: fine for a few hundred scenarios.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivy_model import IvyLM, ModelConfig  # noqa: E402
from ivy_model import chat_format as CF  # noqa: E402


@torch.no_grad()
def generate(model, tok, ids: list[int], end_id: int, max_new: int, device) -> str:
    ids = list(ids)
    new = []
    for _ in range(max_new):
        x = torch.tensor([ids[-model.c.max_seq_len:]], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        if nxt == end_id:
            break
        ids.append(nxt)
        new.append(nxt)
    return tok.decode(new, skip_special_tokens=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/ivy-bpe-32k/tokenizer.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    cfg = ModelConfig.load(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IvyLM(cfg)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    tok = Tokenizer.from_file(a.tokenizer)
    end_id = tok.token_to_id("<|end|>")
    sc = [json.loads(l) for l in Path(a.scenarios).read_text().splitlines() if l.strip()][: a.limit]
    with open(a.out, "w") as f:
        for i, s in enumerate(sc):
            msgs = s.get("messages") or [{"role": "user", "content": s["user"]}]
            out = generate(model, tok, CF.encode_prompt(tok, s["devices"], msgs), end_id, a.max_new, device)
            f.write(json.dumps({"id": s["id"], "output": out}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(sc)}", flush=True)


if __name__ == "__main__":
    main()
