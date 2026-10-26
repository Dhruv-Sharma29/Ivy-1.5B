"""Render the README figures (light and dark variants) from local results into docs/assets/.

    python scripts/make_readme_figures.py      # needs matplotlib; reads reports/ and checkpoints/*/score_*.json

Figures: seen-vs-unseen tool-use accuracy, loss vs training compute, and 40M-vs-300M held-out accuracy across recipes and template
splits. Colours: a colour-blind-validated categorical order (blue, orange), stepped separately for the light and dark surfaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e8e7e3",
              "c1": "#2a78d6", "c2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#8a8984", "grid": "#33332f",
             "c1": "#3987e5", "c2": "#d95926"},
}


def score(run: str, s: str) -> dict:
    return json.loads((ROOT / "checkpoints" / run / f"score_{s}.json").read_text())


def style(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"], "font.size": 11,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["muted"], "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": t["grid"], "grid.linewidth": 0.7, "legend.frameon": False, "lines.linewidth": 2.2,
    })


def bars(ax, t, groups, series, ylabel):
    """groups: x labels; series: [(label, colour, [(acc, lo, hi), ...])]; grouped bars with CI whiskers and value labels."""
    w = 0.38
    for j, (lab, col, vals) in enumerate(series):
        for i, (acc, lo, hi) in enumerate(vals):
            if acc is None:
                continue
            x = i + (j - (len(series) - 1) / 2) * (w + 0.02)
            ax.bar(x, acc * 100, w, color=col, label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, acc * 100, yerr=[[100 * (acc - lo)], [100 * (hi - acc)]], fmt="none", ecolor=t["ink2"], elinewidth=1,
                        capsize=3, zorder=3)
            ax.text(x, hi * 100 + 1.5, f"{acc * 100:.0f}%", ha="center", fontsize=10, color=t["ink"])
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)


def seen_vs_unseen(t):
    runs = [("sft-40m-1b", "40M, trained on\nall templates"), ("sft-40m-1b-v1train", "40M, held-out\ntemplates unseen"),
            ("sft-300m-300m", "300M, trained on\nall templates"), ("sft-300m-300m-v1train", "300M, held-out\ntemplates unseen")]
    get = lambda s: [(score(r, s)["accuracy"], *score(r, s)["accuracy_95ci"]) for r, _ in runs]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars(ax, t, [g for _, g in runs], [("familiar phrasings (in-distribution)", t["c1"], get("sft_indist")),
                                       ("unseen phrasings (held-out templates)", t["c2"], get("sft_heldout"))], "accuracy (%), 95% CI")
    ax.set_title("Tool-use accuracy collapses on phrasings the model never saw", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10)
    return fig


def loss_vs_compute(t):
    rows = [json.loads(l) for l in (ROOT / "reports" / "common-eval-v2.jsonl").read_text().splitlines() if l.strip()]
    keep = ("ivy-40m-lr-2.4e-3-100M", "ivy-40m-scale-300M-lr-2.4e-3", "ivy-40m-scale-1000M-lr-2.4e-3", "ivy-300m-scale-300M-lr-1.2e-3")
    rows = [r for r in rows if r["ckpt"].split("/")[1] in keep]  # runs trained on mix-v1 (the 300M@98M runs used mix-v0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for model, col in [("ivy-40m", t["c1"]), ("ivy-300m", t["c2"])]:
        pts = sorted((6 * r["params"] * r["tokens_seen"], r["mix"], r["tokens_seen"]) for r in rows if r["model"] == model)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, ms=8, label=f"{model[4:].upper()} parameters", zorder=3)
        for f, loss, tok in pts:
            tk = f"{round(tok / 1e9, 1):g}B" if tok >= 1e9 else f"{tok / 1e6:.0f}M"
            ax.annotate(f"{tk} tokens: {loss:.3f}", (f, loss), xytext=(8, 4), textcoords="offset points", fontsize=10, color=t["ink2"])
    ax.set_xscale("log")
    ax.set_xlim(right=max(6 * r["params"] * r["tokens_seen"] for r in rows) * 5)
    ax.set_xlabel("training compute, FLOPs (6 · parameters · tokens, log scale)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("More data beats more parameters at this budget")
    ax.legend(loc="upper right", fontsize=10)
    return fig


def split_sensitivity(t):
    conds = [("split 1\n3 epochs", "sft-40m-1b-v1train", "sft-300m-300m-v1train", "sft_heldout"),
             ("split 1\n1 epoch", "sft-40m-1b-v1train-1ep", "sft-300m-300m-v1train-1ep", "sft_heldout"),
             ("split 2 (independent)\n1 epoch", "sft-40m-1b-s2-1ep", "sft-300m-300m-s2-1ep", "sft_heldout_s2")]
    get = lambda i: [(score(c[i], c[3])["accuracy"], *score(c[i], c[3])["accuracy_95ci"]) for c in conds]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars(ax, t, [c[0] for c in conds], [("40M", t["c1"], get(1)), ("300M", t["c2"], get(2))], "held-out accuracy (%), 95% CI")
    ax.set_title("On unseen phrasings, the 300M model never beats the 40M one", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        style(t)
        for name, make in [("seen-vs-unseen", seen_vs_unseen), ("loss-vs-compute", loss_vs_compute),
                           ("split-sensitivity", split_sensitivity)]:
            fig = make(t)
            fig.tight_layout()
            fig.savefig(OUT / f"{name}-{theme}.png", dpi=150)
            plt.close(fig)
            print(OUT / f"{name}-{theme}.png")


if __name__ == "__main__":
    main()
