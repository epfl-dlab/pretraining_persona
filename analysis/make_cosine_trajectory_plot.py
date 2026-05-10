"""
Two-panel persona-vector cosine-trajectory plot.

Top panel:    cos(v_t, v_main)      — global convergence (how far is the
                                      direction at ckpt t from the final?)
Bottom panel: cos(v_t, v_{t-1})     — local stability (is the direction
                                      still moving between consecutive ckpts?)

A trait is fully stabilised only when *both* curves climb to ~1. The top
curve answers "did we land in the right neighbourhood"; the bottom curve
answers "did we stop moving inside that neighbourhood". Plotting them
together separates "approaching the final direction" from "no longer
drifting", which a single heatmap or a single trajectory cannot do.

Visual style mirrors the emergence/transfer plots:
  - log-token x-axis with hand-set tick labels and stage bands
  - IBM colorblind-safe trait colors (TRAIT_COLORS)
  - figure sized for half-page (single-column) inclusion

Usage:
  python3 analysis/make_cosine_trajectory_plot.py --model olmo3
  python3 analysis/make_cosine_trajectory_plot.py --model apertus
  python3 analysis/make_cosine_trajectory_plot.py            # both
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_persona_vectors import (
    apertus_ckpt_to_tokens_B,
    find_vectors,
    load_layer,
    olmo3_ckpt_to_tokens_B,
)
from make_emergence_plot import (
    StageRegion,
    TRAIT_COLORS,
    TRAIT_LABELS,
    _OLMO_STAGE1_END_B,
    _OLMO_STAGE3_END_B,
    _interp_display_log,
    _log_nudge_display,
)

REPO = Path(__file__).resolve().parents[1]


# ---- model configuration --------------------------------------------------

MODELS = {
    "olmo3": {
        "name": "OLMo-3 7B",
        "model_short": "Olmo-3-1025-7B",
        "layer": 16,
        "ckpt_to_tokens": olmo3_ckpt_to_tokens_B,
        "tick_positions": tuple(
            olmo3_ckpt_to_tokens_B(r) for r in (
                "stage1-step3000",   # 13 B
                "stage1-step10000",  # 42 B
                "stage1-step30000",  # 126 B
                "stage1-step99000",  # 415 B
                "stage1-step297000", # 1.2 T
                "main",              # 6.1 T
            )
        ),
        "tick_labels": ("13B", "42B", "126B", "415B", "1.2T", "6.1T"),
        "stage_regions": (
            StageRegion(
                start_B=0.0, end_B=_OLMO_STAGE1_END_B,
                label="Pretraining", color="#4c78a8",
                anchor_left_B=olmo3_ckpt_to_tokens_B("stage1-step3000"),
                anchor_right_B=_OLMO_STAGE1_END_B,
            ),
            StageRegion(
                start_B=_OLMO_STAGE1_END_B, end_B=_OLMO_STAGE3_END_B,
                label="Midtraining", color="#d4a373",
                anchor_left_B=_OLMO_STAGE1_END_B,
                anchor_right_B=_OLMO_STAGE3_END_B,
            ),
        ),
        "xmin_pad_B": 2.5,
    },
    "apertus": {
        "name": "Apertus-8B-2509",
        "model_short": "Apertus-8B-2509",
        "layer": 16,
        "ckpt_to_tokens": apertus_ckpt_to_tokens_B,
        "tick_positions": (210.0, 840.0, 2100.0, 4200.0, 10592.0, 15000.0),
        "tick_labels": ("210B", "840B", "2.1T", "4.2T", "10.6T", "15T"),
        "stage_regions": (
            StageRegion(
                start_B=0.0, end_B=15_000.0,
                label="Pretraining", color="#4c78a8",
                anchor_left_B=210.0, anchor_right_B=15_000.0,
            ),
        ),
        "xmin_pad_B": 120.0,
    },
}

TRAITS = ("evil", "humorous", "impolite", "sycophantic")
TOKEN_NAME = "response_avg_diff"


# ---- data prep -----------------------------------------------------------

def _normalize(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(n > 0, n, 1)


def collect_trait_series(vectors: np.ndarray, entries: list[dict],
                         trait: str, ckpt_to_tokens):
    """Return (tokens_B, normalised_vectors) sorted by token count.

    Filters out the deepseek-judged variants (matching plot_persona_vectors
    convention) and any revisions where ckpt_to_tokens returns None.
    """
    pairs: list[tuple[float, np.ndarray, str]] = []
    for v, e in zip(vectors, entries):
        if e["trait_method"] != f"{trait}_character_neutral_q":
            continue
        if "deepseek" in e["revision"]:
            continue
        t = ckpt_to_tokens(e["revision"])
        if t is None:
            continue
        pairs.append((t, v, e["revision"]))
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return None
    tokens = np.array([p[0] for p in pairs])
    vecs   = np.stack([p[1] for p in pairs])
    return tokens, _normalize(vecs)


def cosine_to_main(tokens: np.ndarray, normed: np.ndarray) -> np.ndarray:
    """cos(v_t, v_main) — last vector in the (token-sorted) series IS main."""
    return normed @ normed[-1]


def cosine_sequential(tokens: np.ndarray, normed: np.ndarray) -> np.ndarray:
    """cos(v_t, v_{t-1}). NaN for the first ckpt (no predecessor)."""
    out = np.full(len(tokens), np.nan)
    if len(tokens) > 1:
        out[1:] = np.einsum("ij,ij->i", normed[1:], normed[:-1])
    return out


# ---- rendering -----------------------------------------------------------

def _draw_stage_bands(ax, regions, display_map, plot_right_edge, xmin_pad_B):
    if not regions:
        return []
    patches = []
    prev_right = None
    for idx, region in enumerate(regions):
        left = (xmin_pad_B if region.start_B <= 0 else
                _interp_display_log(region.start_B, region.anchor_left_B,
                                    region.anchor_right_B, display_map))
        right = _interp_display_log(region.end_B, region.anchor_left_B,
                                    region.anchor_right_B, display_map)
        if prev_right is not None:
            left = prev_right
        if idx == len(regions) - 1:
            right = plot_right_edge
        ax.axvspan(left, right, color=region.color, alpha=0.12, zorder=0)
        patches.append((region, left, right))
        prev_right = right
    return patches


def render(model_key: str, out_path: Path) -> None:
    cfg = MODELS[model_key]
    model_dir = REPO / "data" / "persona_vectors" / cfg["model_short"]
    if not model_dir.exists():
        print(f"skip {model_key}: {model_dir} missing")
        return
    entries = find_vectors(model_dir, TOKEN_NAME)
    vectors = np.stack([load_layer(e["path"], cfg["layer"]) for e in entries])

    series = {t: collect_trait_series(vectors, entries, t, cfg["ckpt_to_tokens"])
              for t in TRAITS}
    series = {t: s for t, s in series.items() if s is not None}
    if not series:
        print(f"no per-trait data for {model_key}")
        return

    # Restrict to checkpoints shared across ALL plotted traits — otherwise
    # evil's extra early ckpts (where humorous/impolite/sycophantic have no
    # vector) make the leftmost x-range mismatched between traits.
    common_tokens = sorted(set.intersection(*(set(s[0].tolist()) for s in series.values())))
    common_arr = np.array(common_tokens, dtype=float)
    filtered: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t, (toks, normed) in series.items():
        # token → first matching index (handles potential duplicate tokens
        # if two revisions resolve to the same B value).
        idx_by_tok = {float(x): i for i, x in enumerate(toks)}
        keep = np.array([idx_by_tok[ct] for ct in common_tokens], dtype=int)
        filtered[t] = (common_arr, normed[keep])
    series = filtered

    all_x = list(common_tokens)
    display_map = _log_nudge_display(all_x, min_log_gap=0.06)
    plot_right_edge = max(display_map.values()) * 1.05

    # Half-page (single-column) sized: narrow frame, bold lines, large labels
    # so the figure remains legible after print scaling to ~3.5 in.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    band_patches = None
    for ax in (ax_top, ax_bot):
        patches = _draw_stage_bands(ax, cfg["stage_regions"], display_map,
                                    plot_right_edge, cfg["xmin_pad_B"])
        if band_patches is None:
            band_patches = patches
        ax.axhline(1.0, color="#aaa", linewidth=0.6, ls=":", zorder=1)
        ax.axhline(0.0, color="#555", linewidth=0.8, zorder=1)
        ax.grid(True, axis="y", ls=":", alpha=0.30)

    legend_handles = []
    legend_labels = []
    for trait, (tokens, normed) in series.items():
        color = TRAIT_COLORS[f"{trait}_character_neutral_q"]
        label = TRAIT_LABELS[f"{trait}_character_neutral_q"]
        x_disp = np.array([display_map[float(t)] for t in tokens])
        y_main = cosine_to_main(tokens, normed)
        y_seq  = cosine_sequential(tokens, normed)
        line, = ax_top.plot(x_disp, y_main, "-", color=color, linewidth=3.0,
                            zorder=2)
        ax_top.scatter(x_disp, y_main, marker="o", s=34, color=color,
                       edgecolor="white", linewidth=0.7, zorder=3)
        m = ~np.isnan(y_seq)
        ax_bot.plot(x_disp[m], y_seq[m], "-", color=color, linewidth=3.0,
                    zorder=2)
        ax_bot.scatter(x_disp[m], y_seq[m], marker="o", s=34, color=color,
                       edgecolor="white", linewidth=0.7, zorder=3)
        legend_handles.append(line)
        legend_labels.append(label)

    ax_top.set_xscale("log")
    ax_top.set_ylim(-0.05, 1.05)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.set_xlim(max(cfg["xmin_pad_B"], min(display_map.values()) * 0.8),
                    plot_right_edge)

    # Tick labels on the bottom panel only (shared x).
    tick_positions = [display_map.get(t, t) for t in cfg["tick_positions"]]
    ax_bot.set_xticks(tick_positions)
    ax_bot.set_xticklabels(list(cfg["tick_labels"]), fontsize=11,
                           rotation=30, ha="right")
    ax_bot.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False,
                       labelbottom=False)
    for ax in (ax_top, ax_bot):
        ax.tick_params(axis="y", labelsize=11)

    ax_top.set_ylabel(r"$\cos(v_t,\ v_\mathrm{main})$", fontsize=14)
    ax_bot.set_ylabel(r"$\cos(v_t,\ v_{t-1})$", fontsize=14)
    ax_bot.set_xlabel(r"Source checkpoint $M_r$ (tokens, log scale)",
                      fontsize=13)

    for region, left, right in band_patches or []:
        if not region.label:
            continue
        mid_log = (math.log10(left) + math.log10(right)) / 2
        ax_top.text(10 ** mid_log, 1.04, region.label,
                    transform=ax_top.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=10, color=region.color, weight="semibold",
                    clip_on=False, zorder=5)

    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=len(legend_labels), fontsize=11,
               frameon=True, framealpha=0.95, edgecolor="#999",
               handlelength=1.8, columnspacing=1.4, borderpad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08, bottom=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} (+ .png)")


# ---- driver --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    p.add_argument("--out", type=Path, default=None,
                   help="output PDF path (default: story_plots/cosine_trajectory/"
                        "<model>/<model>_cosine_trajectory.pdf)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = list(MODELS) if args.model == "all" else [args.model]
    for m in models:
        out = args.out or (REPO / f"analysis/figures/story_plots/cosine_trajectory/"
                                  f"{m}/{m}_cosine_trajectory.pdf")
        render(m, out)


if __name__ == "__main__":
    main()
