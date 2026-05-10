"""
Joint plot: Baumeister roots (evil) on the left, sycophancy facets on the right.
X-axis = training tokens seen (log scale). Confidence bands (Wilson 95% CI).
Reads from already-annotated CSVs produced by baumeister_gpt_annotation.py
(checkpoints_extract mode) and sycophancy_gpt_annotation.py.

Usage:
  python analysis/facet_joint_plot.py
  python analysis/facet_joint_plot.py --model apertus
"""
import argparse
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# OLMo-3 checkpoint → cumulative tokens (B). Source: arXiv:2512.13961 §3.4
# ---------------------------------------------------------------------------
_TOK_4M_B = 4.194304 / 1000.0
_TOK_2M_B = 2.097152 / 1000.0
_OLMO_STAGE1_END_B = 1_413_814 * _TOK_4M_B
_OLMO_STAGE2_END_B = _OLMO_STAGE1_END_B + 47_684 * _TOK_2M_B
_OLMO_STAGE3_END_B = _OLMO_STAGE2_END_B + 11_921 * _TOK_4M_B
_OLMO_STEP_RE = re.compile(r"^stage([123])-step(\d+)$")


def ckpt_to_tokens_B(rev: str) -> float:
    if rev == "main":
        return _OLMO_STAGE3_END_B
    m = _OLMO_STEP_RE.match(rev)
    stage, step = int(m.group(1)), int(m.group(2))
    if stage == 1:
        return step * _TOK_4M_B
    if stage == 2:
        return _OLMO_STAGE1_END_B + step * _TOK_2M_B
    return _OLMO_STAGE2_END_B + step * _TOK_4M_B


def fmt_tokens_B(n_B: float) -> str:
    if n_B >= 1000:
        t = n_B / 1000.0
        return f"{t:.1f}T" if t < 10 else f"{t:.0f}T"
    if n_B >= 10:
        return f"{n_B:.0f}B"
    return f"{n_B:.1f}B"


# ---------------------------------------------------------------------------
# Apertus checkpoint → cumulative tokens (B). Token count embedded in dir name.
# ---------------------------------------------------------------------------
_APERTUS_TOK_RE = re.compile(r"tokens(\d+(?:\.\d+)?)(B|T)$")

def apertus_ckpt_to_tokens_B(name: str) -> float:
    m = _APERTUS_TOK_RE.search(name)
    val, unit = float(m.group(1)), m.group(2)
    return val * 1000.0 if unit == "T" else val


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
OLMO_CHECKPOINT_ORDER = [
    "stage1-step3000", "stage1-step5000", "stage1-step7000", "stage1-step9000",
    "stage1-step10000", "stage1-step15000", "stage1-step20000", "stage1-step30000",
    "stage1-step50000", "stage1-step99000", "stage1-step297000", "stage1-step707000",
    "stage1-step1413814", "stage2-step47684", "stage3-step11921", "main",
]

APERTUS_CHECKPOINT_ORDER = [
    "step50000-tokens210B", "step100000-tokens420B", "step150000-tokens630B",
    "step200000-tokens840B", "step250000-tokens1050B", "step300000-tokens1260B",
    "step400000-tokens1680B", "step500000-tokens2100B", "step700000-tokens2940B",
    "step1000000-tokens4200B", "step1432000-tokens6014B",
    "step1750000-tokens7652B", "step2100000-tokens10592B",
    "step2400000-tokens13112B", "step2627139-tokens15T",
]

OLMO_EXTRACT_DIR = Path("data/model_responses/extract/Olmo-3-1025-7B/main")
APERTUS_EXTRACT_DIR = Path("data/model_responses/extract/Apertus-8B-2509/main")

# kept for backward compat
CHECKPOINT_ORDER = OLMO_CHECKPOINT_ORDER
EXTRACT_DIR = OLMO_EXTRACT_DIR

VALID_ROOTS = {"instrumentality", "threatened_egotism", "idealism", "sadism"}
EVIL_ORDER = ["instrumentality", "threatened_egotism", "idealism", "sadism"]
EVIL_LABELS = {
    "instrumentality":    "Instrumentality",
    "threatened_egotism": "Threatened Egotism",
    "idealism":           "Idealism",
    "sadism":             "Sadism",
}

SYCO_FACETS = ["validation", "indirectness", "framing"]
SYCO_LABELS = {
    "validation":   "Emotional Validation",
    "indirectness": "Indirectness",
    "framing":      "Accepts Premise",
}


def wilson_ci(count, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = count / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = z * (p * (1 - p) / total + z ** 2 / (4 * total ** 2)) ** 0.5 / denom
    return max(0.0, center - margin) * 100, min(100.0, center + margin) * 100


def load_baumeister(checkpoints, extract_dir=None):
    if extract_dir is None:
        extract_dir = EXTRACT_DIR
    results = {}
    for name in checkpoints:
        path = extract_dir / f"baumeister_checkpoints_extract_{name}.csv"
        counts = {r: 0 for r in VALID_ROOTS}
        total = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                total += 1
                for r in row[-1].split("|"):
                    if r in VALID_ROOTS:
                        counts[r] += 1
        results[name] = {"total": total, "counts": counts}
    return results


def load_sycophancy(checkpoints, extract_dir=None):
    if extract_dir is None:
        extract_dir = EXTRACT_DIR
    results = {}
    for name in checkpoints:
        path = extract_dir / f"sycophancy_checkpoints_{name}.csv"
        counts = {f: 0 for f in SYCO_FACETS}
        total = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            facet_idx = {f: header.index(f) for f in SYCO_FACETS}
            for row in reader:
                if not row:
                    continue
                total += 1
                for f in SYCO_FACETS:
                    try:
                        counts[f] += int(row[facet_idx[f]])
                    except (ValueError, IndexError):
                        pass
        results[name] = {"total": total, "counts": counts}
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bands(ax, xs, results, keys, labels, colors):
    for idx, key in enumerate(keys):
        ps, los, his = [], [], []
        for name in results:
            total = results[name]["total"]
            count = results[name]["counts"][key]
            p_pct = count / total * 100
            lo, hi = wilson_ci(count, total)
            ps.append(p_pct)
            los.append(lo)
            his.append(hi)
        ps, los, his = np.array(ps), np.array(los), np.array(his)
        ax.plot(xs, ps, color=colors[idx], linewidth=2, label=labels[key], zorder=3)
        ax.fill_between(xs, los, his, color=colors[idx], alpha=0.2, zorder=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["olmo", "apertus"], default="olmo")
    args = parser.parse_args()

    if args.model == "apertus":
        checkpoints = APERTUS_CHECKPOINT_ORDER
        extract_dir = APERTUS_EXTRACT_DIR
        tok_fn = apertus_ckpt_to_tokens_B
        out = "analysis/figures/facet_joint_checkpoints_lines_apertus.pdf"
    else:
        checkpoints = OLMO_CHECKPOINT_ORDER
        extract_dir = OLMO_EXTRACT_DIR
        tok_fn = ckpt_to_tokens_B
        out = "analysis/figures/facet_joint_checkpoints_lines.pdf"

    # Filter to checkpoints that have both annotation files
    evil_ckpts = [c for c in checkpoints if (extract_dir / f"baumeister_checkpoints_extract_{c}.csv").exists()]
    syco_ckpts = [c for c in checkpoints if (extract_dir / f"sycophancy_checkpoints_{c}.csv").exists()]

    evil = load_baumeister(evil_ckpts, extract_dir)
    syco = load_sycophancy(syco_ckpts, extract_dir)

    evil_xs = np.array([tok_fn(c) for c in evil_ckpts])
    syco_xs = np.array([tok_fn(c) for c in syco_ckpts])

    evil_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    syco_colors = ["#9467bd", "#8c564b", "#e377c2"]

    if args.model == "apertus":
        tick_vals = [200, 500, 1000, 5000, 15000]
        all_xs = np.concatenate([evil_xs, syco_xs])
        xlim = (all_xs.min() * 0.85, all_xs.max() * 1.15)
    else:
        tick_vals = [10, 50, 100, 500, 1000, 5000]
        xlim = None
    tick_lbls = [fmt_tokens_B(v) for v in tick_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for ax, results, xs, keys, labels, colors, title in [
        (axes[0], evil, evil_xs, EVIL_ORDER,  EVIL_LABELS, evil_colors, "Evil"),
        (axes[1], syco, syco_xs, SYCO_FACETS, SYCO_LABELS, syco_colors, "Sycophantic"),
    ]:
        plot_bands(ax, xs, results, keys, labels, colors)
        ax.set_xscale("log")
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_lbls, fontsize=12, rotation=0, ha="right")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel("Training tokens seen", fontsize=12)
        ax.set_ylabel("% responses", fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=14, loc="upper right", framealpha=0.8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.grid(True, axis="x", ls=":", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
