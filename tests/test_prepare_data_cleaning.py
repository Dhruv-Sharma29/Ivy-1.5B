import json
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivy_model import prepare_data as P  # noqa: E402

VOCAB = [f"w{i}x" for i in range(3000)]


def doc(seed, n=300):
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) for _ in range(n))


def make_tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["abc def ghi " * 50], trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    (d / "MANIFEST.json").write_text(json.dumps({"sha256": "test"}))


def run_main(monkeypatch, tmp_path, rows, extra_args=()):
    tokdir = tmp_path / "tok"
    make_tokenizer(tokdir)
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": [{"id": "toy", "hf_repo": "x", "subset": "s", "text_field": "text"}]}))
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--source", "toy", "--sources-file", str(sources),
                                      "--tokenizer", str(tokdir), "--out", str(out), "--max-tokens", "10000000",
                                      "--shuffle-buffer", "0", *extra_args])
    P.main()
    return json.loads((out / "MANIFEST.json").read_text()), out, tokdir


def test_pipeline_removes_dupes_near_dupes_contaminated_and_scrubs(monkeypatch, tmp_path):
    secret = "hf" + "_" + "z" * 34
    a, b, c = doc(1), doc(2), doc(3)
    edited = a.split()
    edited[100] = "CHANGED"
    bench = " ".join(f"bench{i}" for i in range(30))
    (tmp_path / "bench.jsonl").write_text(json.dumps({"text": bench}) + "\n")
    rows = [{"id": i, "text": t} for i, t in enumerate([
        a, a,                                   # exact duplicate
        " ".join(edited),                       # near duplicate
        b + f" token is {secret} end",          # carries a secret (kept, scrubbed)
        c + " " + " ".join(f"bench{i}" for i in range(5, 25)),  # contaminated
        "", None,                               # malformed
    ])]
    m, out, _ = run_main(monkeypatch, tmp_path, rows,
                         ["--decontaminate-file", str(tmp_path / "bench.jsonl")])
    s = m["stats"]
    assert (s["exact_dupes"], s["near_dupes"], s["decontaminated"], s["empty_or_malformed"]) == (1, 1, 1, 2)
    assert s["docs"] == 2 and s["docs_scrubbed"] == 1 and m["scrub_counts"] == {"hf_token": 1}
    assert m["caveats"] == ["scrubbing is regex-only: misses unknown formats"]
    assert m["cleaning"]["decontaminate"]["files"] == [str(tmp_path / "bench.jsonl")]


def test_cleaning_can_be_disabled_and_caveats_say_so(monkeypatch, tmp_path):
    a = doc(1)
    edited = a.split()
    edited[10] = "CHANGED"
    rows = [{"id": 0, "text": a}, {"id": 1, "text": " ".join(edited)}]
    m, _, _ = run_main(monkeypatch, tmp_path, rows, ["--no-scrub", "--no-near-dup"])
    assert m["stats"]["docs"] == 2 and m["stats"]["near_dupes"] == 0
    assert set(m["caveats"]) == {"secret/email scrubbing not applied", "near-duplicate removal not applied",
                                 "eval decontamination not applied"}


def test_scrubbed_secret_is_absent_from_written_tokens(monkeypatch, tmp_path):
    from tokenizers import Tokenizer
    secret = "hf" + "_" + "q" * 34
    rows = [{"id": 0, "text": f"hello {secret} world"}]
    _, out, tokdir = run_main(monkeypatch, tmp_path, rows)
    ids = np.fromfile(next(out.glob("train-*.bin")) if list(out.glob("train-*.bin")) else next(out.glob("val-*.bin")),
                      dtype=np.uint16).tolist()
    text = Tokenizer.from_file(str(tokdir / "tokenizer.json")).decode(ids)
    assert secret not in text and "<SECRET>" in text


def test_balanced_files_take_an_equal_share_from_every_file(monkeypatch, tmp_path):
    import huggingface_hub

    files = {f"sample/10BT/{k:03d}.parquet": [{"id": f"f{k}-{j}", "text": f"file{k} " + doc(100 * k + j, 120)} for j in range(60)]
             for k in range(3)}

    class FakeApi:
        def list_repo_tree(self, repo, path_in_repo, repo_type):
            return [types.SimpleNamespace(path=p) for p in files]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    tokdir = tmp_path / "tok"
    make_tokenizer(tokdir)
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": [{"id": "toy", "hf_repo": "x/y", "subset": "s", "text_field": "text"}]}))
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, data_files=None, **k: iter(files[data_files.split("x/y/")[1]])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--source", "toy", "--sources-file", str(sources), "--tokenizer", str(tokdir),
                                      "--out", str(out), "--max-tokens", "30000", "--no-near-dup", "--balanced-files-prefix", "sample/10BT/"])
    P.main()
    m = json.loads((out / "MANIFEST.json").read_text())
    assert m["balanced_files"] == {"prefix": "sample/10BT/", "files": 3}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokdir / "tokenizer.json"))
    ids = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in sorted(out.glob("*.bin"))]).tolist()
    text = tok.decode(ids)
    counts = [text.count(f"file{k} ") for k in range(3)]
    assert min(counts) > 0 and max(counts) <= 2 * min(counts) + 5, counts  # every file contributes, none dominates
