"""
Transfer-to-post-training plot: for each trait, take the persona vector
extracted from the BASE model at checkpoint r and apply it to a fixed
post-trained target. One curve per trait, x = tokens at the extract
checkpoint, y = Δ trait score the base-model vector produces when
transferred.

Visual style mirrors `make_emergence_plot.py`:
  - IBM colorblind-safe palette
  - log x-axis in training tokens, with ticks clipped via `_log_nudge_display`
  - ★ = p_raw < 0.05,  ○ = not significant

Usage:
  python3 analysis/make_transfer_plot.py --model olmo3
  python3 analysis/make_transfer_plot.py --model apertus
  python3 analysis/make_transfer_plot.py --model olmo3 --target sft
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_emergence_plot import (
    PALETTE,
    StageRegion,
    TRAIT_COLORS,
    TRAIT_LABELS,
    _OLMO_STAGE1_END_B,
    _OLMO_STAGE3_END_B,
    _fmt_tokens_B,
    _interp_display_log,
    _log_nudge_display,
    olmo3_ckpt_to_tokens_B,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------- config --------------------------------------

@dataclass(frozen=True)
class TransferTraitConfig:
    key: str           # trait CSV column (legend keying)
    label: str
    color: str
    csv_path: Path     # combined_significance.csv
    layer: int
    coef: float
    # Optional supplementary (csv_path, coef) segments merged into the curve
    # for this trait. First-seen-wins on `extract_revision` so the primary
    # segment dominates when grids overlap. Use when no single sweep produces
    # a measurable Δ across the full ckpt range at one coef (e.g. Apertus
    # evil transfer: c=0.6 on early ckpts, c=0.4 in the middle, c=0.3 late).
    extra_segments: tuple[tuple[Path, float], ...] = ()


@dataclass(frozen=True)
class TransferModelConfig:
    name: str                        # pretty title, e.g. "OLMo-3 7B  →  Olmo-3-7B-Instruct"
    # Row filter on `eval_model` (instruct-transfer CSVs). Set to None for
    # base→main transfers whose CSVs are single-target (no eval_model column).
    eval_model: str | None
    traits: list[TransferTraitConfig]
    ckpt_to_tokens: callable
    ckpt_grid: tuple[str, ...]       # which extract_revisions to include (shared x-axis grid)
    xmin_pad_B: float = 2.5
    stage_regions: tuple[StageRegion, ...] = ()


# --------------------------- OLMo-3 --------------------------------------

OLMO3_TRANSFER_BASE = REPO / "results/Olmo-3-1025-7B/instruct_transfer"

# Shared x-axis grid for OLMo-3 — matches the emergence plot ckpt set.
# Stage-end ckpts are INCLUDED so the transition through stages 1, 2, 3
# is visible on the right side of the plot. `stage3-step11921` is the same
# weights as `main` and is excluded to avoid stacking identical markers.
OLMO3_CKPT_GRID: tuple[str, ...] = (
    "stage1-step3000",  "stage1-step5000",  "stage1-step7000",  "stage1-step9000",
    "stage1-step10000", "stage1-step15000", "stage1-step20000", "stage1-step30000",
    "stage1-step50000", "stage1-step99000", "stage1-step297000", "stage1-step707000",
    "stage1-step1413814",  # stage 1 end = 5.93 T
    "stage2-step47684",    # stage 2 end = 6.03 T
    "main",                # stage 3 end = 6.08 T (stage3-step11921 is an alias)
)


def olmo3_transfer_config(target: str) -> TransferModelConfig:
    eval_model, pretty = OLMO3_TARGETS[target]
    if target == "main":
        # Base→main: vectors extracted at each base ckpt applied to `main`.
        # CSVs live under checkpoint_grid/ (not instruct_transfer/) and use
        # the same layer/coef operating points as the emergence plot.
        # NOTE evil: `evil_historical_all_rows_v1/combined.csv` is the only
        # source with evil L16 c=0.5 → main rows (23 rows, 8 match the
        # emergence grid). The early stage-1 ckpts (step3000..step50000) are
        # absent — a backfill sweep would be needed for full grid coverage.
        spec = [
            ("evil_character_neutral_q",
             OLMO3_GRID_BASE / "evil_historical_all_rows_v1" / "combined_significance.csv",         16, 0.5),
            ("humorous_character_neutral_q",
             OLMO3_GRID_BASE / "humorous_transfer_to_main_v1" / "combined_significance.csv",        20, 0.3),
            ("impolite_character_neutral_q",
             OLMO3_GRID_BASE / "impolite_transfer_to_main_v1" / "combined_significance.csv",        16, 0.5),
            ("sycophantic_character_neutral_q",
             OLMO3_GRID_BASE / "sycophantic_transfer_to_main_v1" / "combined_significance.csv",     16, 0.5),
        ]
    else:
        spec = [
            ("evil_character_neutral_q",
             OLMO3_TRANSFER_BASE / "evil_transfer_to_instruct_targets_dense_coef0p55_v1" / "combined_significance.csv",                  16, 0.55),
            ("humorous_character_neutral_q",
             OLMO3_TRANSFER_BASE / "humorous_base_vectors_to_instruct_universal_grid_layer20_coef0p30_v1" / "combined_significance.csv", 20, 0.30),
            ("impolite_character_neutral_q",
             OLMO3_TRANSFER_BASE / "impolite_base_vectors_to_instruct_universal_grid_layer20_coef0p75_v1" / "combined_significance.csv", 20, 0.75),
            ("sycophantic_character_neutral_q",
             OLMO3_TRANSFER_BASE / "sycophantic_base_vectors_to_instruct_layer16_coef0p5_v1" / "combined_significance.csv",              16, 0.50),
        ]
    traits = [
        TransferTraitConfig(
            key=k, label=TRAIT_LABELS[k], color=TRAIT_COLORS[k],
            csv_path=csv,
            layer=layer, coef=coef,
        )
        for k, csv, layer, coef in spec
    ]
    return TransferModelConfig(
        name=f"OLMo-3 7B base vectors → {pretty}",
        eval_model=eval_model,
        traits=traits,
        ckpt_to_tokens=olmo3_ckpt_to_tokens_B,
        ckpt_grid=OLMO3_CKPT_GRID,
        stage_regions=(
            StageRegion(
                start_B=0.0, end_B=olmo3_ckpt_to_tokens_B("stage1-step1413814"),
                label="Stage 1: Dolma 3 pretraining",
                color="#4c78a8",
                anchor_left_B=olmo3_ckpt_to_tokens_B("stage1-step3000"),
                anchor_right_B=olmo3_ckpt_to_tokens_B("stage1-step1413814"),
            ),
            StageRegion(
                start_B=olmo3_ckpt_to_tokens_B("stage1-step1413814"),
                end_B=_OLMO_STAGE3_END_B,
                label="Stages 2+3:\nmidtraining",
                color="#f58518",
                anchor_left_B=olmo3_ckpt_to_tokens_B("stage1-step1413814"),
                anchor_right_B=_OLMO_STAGE3_END_B,
            ),
        ),
    )


OLMO3_TARGETS = {
    "instruct": ("allenai/Olmo-3-7B-Instruct",     "Olmo-3-7B-Instruct"),
    "dpo":      ("allenai/Olmo-3-7B-Instruct-DPO", "Olmo-3-7B-Instruct-DPO"),
    "sft":      ("allenai/Olmo-3-7B-Instruct-SFT", "Olmo-3-7B-Instruct-SFT"),
    # base→main (final base-model weights). CSV is single-target so
    # eval_model is None; the source dir is checkpoint_grid/*_transfer_to_main_v1.
    "main":     (None,                              "Olmo-3-7B base (main)"),
}

OLMO3_GRID_BASE = REPO / "results/Olmo-3-1025-7B/checkpoint_grid"


# --------------------------- Apertus -------------------------------------

# Apertus branch names encode cumulative tokens: `step50000-tokens210B` → 210 B;
# `step2627139-tokens15T` → 15000 B. Preferred over linear extrapolation from
# step because Apertus doubles batch size after 8 T tokens (paper Table 2).

APERTUS_TRANSFER_BASE = REPO / "results/Apertus-8B-2509/instruct_transfer"
_APERTUS_MAIN_B = 15_000.0


def apertus_ckpt_to_tokens_B(rev: str) -> float | None:
    if rev == "main":
        return _APERTUS_MAIN_B
    if "-tokens" in rev:
        tok = rev.split("-tokens", 1)[1]
        if tok.endswith("T"):
            return float(tok[:-1]) * 1000.0
        if tok.endswith("B"):
            return float(tok[:-1])
    return None


# Shared x-axis grid — the `universal_grid` sweep used for all three traits.
# `main` and `step2627139-tokens15T` both resolve to 15 000 B, so we drop `main`.
APERTUS_CKPT_GRID: tuple[str, ...] = (
    "step50000-tokens210B",
    "step100000-tokens420B",
    "step150000-tokens630B",
    "step200000-tokens840B",
    "step250000-tokens1050B",
    "step300000-tokens1260B",
    "step400000-tokens1680B",
    "step500000-tokens2100B",
    "step700000-tokens2940B",
    "step1000000-tokens4200B",
    "step1432000-tokens6014B",
    "step1750000-tokens7652B",
    "step2100000-tokens10592B",
    "step2400000-tokens13112B",
    "step2627139-tokens15T",
)


def apertus_transfer_config(target: str) -> TransferModelConfig:
    eval_model, pretty = APERTUS_TARGETS[target]
    if target == "main":
        # Base→main: CSVs in checkpoint_grid/ at emergence-matching coefs.
        traits = [
            TransferTraitConfig(
                key="evil_character_neutral_q",
                label=TRAIT_LABELS["evil_character_neutral_q"],
                color=TRAIT_COLORS["evil_character_neutral_q"],
                csv_path=APERTUS_GRID_BASE / "apertus_evil_transfer_to_main_coef0p2_full_v1" / "combined_significance.csv",
                layer=16, coef=0.2,
            ),
            TransferTraitConfig(
                key="impolite_character_neutral_q",
                label=TRAIT_LABELS["impolite_character_neutral_q"],
                color=TRAIT_COLORS["impolite_character_neutral_q"],
                csv_path=APERTUS_GRID_BASE / "apertus_impolite_transfer_to_main_layer20_coef0p15" / "combined_significance.csv",
                layer=20, coef=0.15,
            ),
            TransferTraitConfig(
                key="sycophantic_character_neutral_q",
                label=TRAIT_LABELS["sycophantic_character_neutral_q"],
                color=TRAIT_COLORS["sycophantic_character_neutral_q"],
                csv_path=APERTUS_GRID_BASE / "syco_transfer_to_main" / "combined_significance.csv",
                layer=16, coef=0.2,
            ),
        ]
    else:
        # Apertus evil transfer uses MIXED COEFS to avoid the "flat-at-zero early"
        # artifact of the single c=0.3 universal-grid sweep. (See doc section
        # "Apertus Base→Instruct Transfer — Mixed-Coef Evil".)
        evil_early_c06 = APERTUS_TRANSFER_BASE / "apertus_evil_base_vectors_to_instruct_early_layer16_coef0p6_v1" / "combined_significance.csv"
        evil_mid_c04   = APERTUS_TRANSFER_BASE / "apertus_evil_base_vectors_to_instruct_early_layer16_coef0p4_v1" / "combined_significance.csv"
        evil_late_c03  = APERTUS_TRANSFER_BASE / "apertus_evil_base_vectors_to_instruct_universal_grid_layer16_coef0p3_v1" / "combined_significance.csv"
        traits = [
            TransferTraitConfig(
                key="evil_character_neutral_q",
                label=TRAIT_LABELS["evil_character_neutral_q"],
                color=TRAIT_COLORS["evil_character_neutral_q"],
                csv_path=evil_early_c06, layer=16, coef=0.6,
                extra_segments=((evil_mid_c04, 0.4), (evil_late_c03, 0.3)),
            ),
            TransferTraitConfig(
                key="impolite_character_neutral_q",
                label=TRAIT_LABELS["impolite_character_neutral_q"],
                color=TRAIT_COLORS["impolite_character_neutral_q"],
                csv_path=APERTUS_TRANSFER_BASE / "apertus_impolite_base_vectors_to_instruct_layer20_coef0p7" / "combined_significance.csv",
                layer=20, coef=0.7,
            ),
            TransferTraitConfig(
                key="sycophantic_character_neutral_q",
                label=TRAIT_LABELS["sycophantic_character_neutral_q"],
                color=TRAIT_COLORS["sycophantic_character_neutral_q"],
                csv_path=APERTUS_TRANSFER_BASE / "apertus_syco_base_vectors_to_instruct_layer16_coef0p5" / "combined_significance.csv",
                layer=16, coef=0.5,
            ),
        ]
    return TransferModelConfig(
        name=f"Apertus-8B base vectors → {pretty}",
        eval_model=eval_model,
        traits=traits,
        ckpt_to_tokens=apertus_ckpt_to_tokens_B,
        ckpt_grid=APERTUS_CKPT_GRID,
        xmin_pad_B=120.0,
        stage_regions=_apertus_stage_regions(),
    )


APERTUS_TARGETS = {
    "instruct": ("swiss-ai/Apertus-8B-Instruct-2509", "Apertus-8B-Instruct-2509"),
    # base → `main`: single-target CSV, no eval_model filter (same pattern as olmo3 main).
    "main":     (None, "Apertus-8B base (main)"),
}

APERTUS_GRID_BASE = REPO / "results/Apertus-8B-2509/checkpoint_grid"


def _apertus_stage_regions() -> tuple[StageRegion, ...]:
    """Phase 1 (pretraining, 0–13.5 T) + Cooldown (13.5–15 T). Apertus paper
    arXiv:2509.14233 §2.3 / Table 2: batch doubles at 8 T (same phase) and LR
    annealing + data-mix change at 13.5 T (cooldown start)."""
    return (
        StageRegion(
            start_B=0.0, end_B=13_500.0,
            label="Stage 1: pretraining (13.5 T)",
            color="#4c78a8",
            anchor_left_B=apertus_ckpt_to_tokens_B("step50000-tokens210B"),
            anchor_right_B=apertus_ckpt_to_tokens_B("step2400000-tokens13112B"),
        ),
        StageRegion(
            start_B=13_500.0, end_B=15_000.0,
            label="Stage 2: midtraining",
            color="#f58518",
            anchor_left_B=apertus_ckpt_to_tokens_B("step2400000-tokens13112B"),
            anchor_right_B=apertus_ckpt_to_tokens_B("step2627139-tokens15T"),
        ),
    )


# --------------------------- dispatch ------------------------------------

MODEL_TARGETS: dict[str, dict[str, tuple[str, str]]] = {
    "olmo3":   OLMO3_TARGETS,
    "apertus": APERTUS_TARGETS,
}
MODEL_BUILDERS = {
    "olmo3":   olmo3_transfer_config,
    "apertus": apertus_transfer_config,
}


def build_config(model: str, target: str) -> TransferModelConfig:
    return MODEL_BUILDERS[model](target)


# --------------------------- data loading --------------------------------

def _load_transfer_segment(csv_path: Path, coef: float, layer: int,
                           eval_model: str | None, ckpt_to_tokens,
                           grid: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # `eval_model` is only present in instruct-transfer CSVs (which have
    # multiple target models per file). The base→main-checkpoint CSVs are
    # single-target and omit that column; in that case we skip the filter.
    if eval_model is not None and "eval_model" in df.columns:
        df = df[df.eval_model == eval_model].copy()
    else:
        df = df.copy()
    df = df[df.extract_revision.isin(grid)].copy()
    if "layer" in df.columns:
        df = df[df.layer.astype(int) == int(layer)].copy()
    if "coef" in df.columns:
        df = df[df.coef.astype(float).round(3) == round(coef, 3)].copy()
    df["tokens_B"] = df.extract_revision.map(ckpt_to_tokens)
    df = df.dropna(subset=["tokens_B", "delta_trait_mean"]).copy()
    df["p_raw"] = df.get("trait_primary_p_two_sided", pd.Series(dtype=float))
    df["coef_used"] = float(coef)
    return df[["extract_revision", "tokens_B", "delta_trait_mean", "p_raw", "coef_used"]]


def load_transfer(trait: TransferTraitConfig, eval_model: str | None,
                  ckpt_to_tokens, grid: tuple[str, ...]) -> pd.DataFrame:
    segments = [(trait.csv_path, trait.coef), *trait.extra_segments]
    frames = [_load_transfer_segment(p, c, trait.layer, eval_model, ckpt_to_tokens, grid)
              for p, c in segments]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["extract_revision"], keep="first")
    combined = combined.sort_values("tokens_B").reset_index(drop=True)
    return combined


# --------------------------- rendering -----------------------------------

def render(cfg: TransferModelConfig, out_path: Path, p_threshold: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))

    per_trait: list[tuple[TransferTraitConfig, pd.DataFrame]] = []
    all_x: list[float] = []
    for trait in cfg.traits:
        df = load_transfer(trait, cfg.eval_model, cfg.ckpt_to_tokens,
                           cfg.ckpt_grid)
        if df.empty:
            continue
        per_trait.append((trait, df))
        all_x.extend(df.tokens_B.tolist())

    display_map = _log_nudge_display(all_x, min_log_gap=0.12, max_log_gap=0.20)

    # Transfer plot has no post-training column — extend the final stage
    # band to the plot's right edge so there is no white gap on the right.
    plot_right_edge = max(display_map.values()) * 1.10 if display_map else None

    stage_band_patches: list = []
    if cfg.stage_regions and display_map:
        regions = list(cfg.stage_regions)
        prev_right_x: float | None = None
        for idx, region in enumerate(regions):
            left_x = (cfg.xmin_pad_B if region.start_B <= 0
                      else _interp_display_log(region.start_B,
                                               region.anchor_left_B,
                                               region.anchor_right_B,
                                               display_map))
            right_x = _interp_display_log(region.end_B,
                                          region.anchor_left_B,
                                          region.anchor_right_B,
                                          display_map)
            # Snap consecutive bands so no white gap appears at the boundary
            # when the two regions use different anchor ckpts (e.g. Apertus
            # stage-1 anchor=13112, stage-2 anchor=15000 → boundary 13500
            # lands at different display-x in each interpolation).
            if prev_right_x is not None:
                left_x = prev_right_x
            if idx == len(regions) - 1 and plot_right_edge is not None:
                right_x = plot_right_edge
            ax.axvspan(left_x, right_x, color=region.color, alpha=0.12, zorder=0)
            stage_band_patches.append((region, left_x, right_x))
            prev_right_x = right_x

    for trait, df in per_trait:
        x_disp = df.tokens_B.map(display_map).to_numpy()
        y = df.delta_trait_mean.to_numpy()
        p = df.p_raw.fillna(1.0).to_numpy()

        ax.plot(x_disp, y, "-", color=trait.color, linewidth=2.4,
                label=trait.label, zorder=2)
        sig = p < p_threshold
        ax.scatter(x_disp[sig], y[sig], marker="*", s=130, color=trait.color,
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.scatter(x_disp[~sig], y[~sig], marker="o", s=36, facecolor="white",
                   edgecolor=trait.color, linewidth=1.6, zorder=3)

    ax.axhline(0, color="#555", linewidth=0.8, zorder=1)

    ax.set_xscale("log")
    if all_x:
        disp_x = sorted(display_map.values())
        ax.set_xlim(max(cfg.xmin_pad_B, min(disp_x) * 0.8), max(disp_x) * 1.10)
        sorted_actual = sorted(display_map.keys())
        tick_positions = [display_map[a] for a in sorted_actual]
        tick_labels = [_fmt_tokens_B(a) for a in sorted_actual]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=30, ha="right")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    target_pretty = cfg.name.split("→", 1)[-1].strip()
    ax.set_xlabel("Checkpoint of persona extraction (tokens consumed)", fontsize=11)
    ax.set_ylabel(f"Trait expression delta on {target_pretty}", fontsize=11)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.grid(True, which="major", axis="x", ls=":", alpha=0.20)

    # Consolidated legend: trait lines + one marker handle explaining ★.
    from matplotlib.lines import Line2D
    trait_handles, trait_labels = ax.get_legend_handles_labels()
    marker_handles = [Line2D([0], [0], marker="*", color="none",
                             markerfacecolor="black", markeredgecolor="white",
                             markersize=12, linewidth=0,
                             label=f"significant (p < {p_threshold})")]
    ax.legend(handles=trait_handles + marker_handles,
              labels=trait_labels + [h.get_label() for h in marker_handles],
              loc="upper left", fontsize=9.5, frameon=True, framealpha=0.92,
              handlelength=1.6, labelspacing=0.4)

    if stage_band_patches:
        for region, left_x, right_x in stage_band_patches:
            if not region.label:
                continue
            mid_log = (math.log10(left_x) + math.log10(right_x)) / 2
            ax.text(10 ** mid_log, 1.02, region.label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=8.5, color=region.color, weight="semibold",
                    clip_on=False, zorder=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {pdf_path}")


# --------------------------- per-trait rendering -------------------------
# Same-trait comparison across post-training targets. One figure per trait,
# each line = a different target (main / SFT / DPO / Instruct). x-axis is
# the source extract checkpoint (same nudged log token axis as the
# per-target plots). Uses per-trait operating points that are CONSISTENT
# across all four targets — so the four lines differ only by the evaluated
# model, not by layer or coef. This means we can't always reuse the dense
# sweeps chosen for the per-target plots (which optimized each target's
# coef separately); instead, PER_TRAIT_SPECS below points at matching
# L/c sweeps. Some targets may have sparser ckpt coverage as a result.
PER_TRAIT_SPECS_APERTUS: dict[str, dict[str, tuple[Path, int, float]]] = {
    # Apertus has only two targets we can compare: base `main` and Instruct-2509.
    # Instruct uses the per-target transfer sweep operating points (evil mixed
    # c=0.3/0.4/0.6, syco c=0.5, impo c=0.7); main uses the emergence-matching
    # coefs (c=0.2 or c=0.15). extra_segments handled separately below for evil.
    "evil_character_neutral_q": {
        "main":     (APERTUS_GRID_BASE     / "apertus_evil_transfer_to_main_coef0p2_full_v1"                 / "combined_significance.csv", 16, 0.2),
        "instruct": (APERTUS_TRANSFER_BASE / "apertus_evil_base_vectors_to_instruct_universal_grid_layer16_coef0p3_v1" / "combined_significance.csv", 16, 0.3),
    },
    "sycophantic_character_neutral_q": {
        "main":     (APERTUS_GRID_BASE     / "syco_transfer_to_main"                                         / "combined_significance.csv", 16, 0.2),
        "instruct": (APERTUS_TRANSFER_BASE / "apertus_syco_base_vectors_to_instruct_layer16_coef0p5"         / "combined_significance.csv", 16, 0.5),
    },
    "impolite_character_neutral_q": {
        "main":     (APERTUS_GRID_BASE     / "apertus_impolite_transfer_to_main_layer20_coef0p15"            / "combined_significance.csv", 20, 0.15),
        "instruct": (APERTUS_TRANSFER_BASE / "apertus_impolite_base_vectors_to_instruct_layer20_coef0p7"     / "combined_significance.csv", 20, 0.7),
    },
}


PER_TRAIT_SPECS_OLMO3: dict[str, dict[str, tuple[Path, int, float]]] = {
    # trait_key : { target : (csv_path, layer, coef) }
    # Uses the SAME dense sweeps as the per-target transfer plot so every
    # line spans the full 15-ckpt emergence grid. Different targets may use
    # different (layer, coef) — that's reflected in the per-line legend
    # instead of a single title operating point.
    "evil_character_neutral_q": {
        "main":     (OLMO3_GRID_BASE    / "evil_historical_all_rows_v1"                                / "combined_significance.csv", 16, 0.5),
        "sft":      (OLMO3_TRANSFER_BASE / "evil_transfer_to_instruct_targets_dense_coef0p55_v1"       / "combined_significance.csv", 16, 0.55),
        "dpo":      (OLMO3_TRANSFER_BASE / "evil_transfer_to_instruct_targets_dense_coef0p55_v1"       / "combined_significance.csv", 16, 0.55),
        "instruct": (OLMO3_TRANSFER_BASE / "evil_transfer_to_instruct_targets_dense_coef0p55_v1"       / "combined_significance.csv", 16, 0.55),
    },
    "sycophantic_character_neutral_q": {
        "main":     (OLMO3_GRID_BASE    / "sycophantic_transfer_to_main_v1"                            / "combined_significance.csv", 16, 0.5),
        "sft":      (OLMO3_TRANSFER_BASE / "sycophantic_base_vectors_to_instruct_layer16_coef0p5_v1"   / "combined_significance.csv", 16, 0.5),
        "dpo":      (OLMO3_TRANSFER_BASE / "sycophantic_base_vectors_to_instruct_layer16_coef0p5_v1"   / "combined_significance.csv", 16, 0.5),
        "instruct": (OLMO3_TRANSFER_BASE / "sycophantic_base_vectors_to_instruct_layer16_coef0p5_v1"   / "combined_significance.csv", 16, 0.5),
    },
    "impolite_character_neutral_q": {
        "main":     (OLMO3_GRID_BASE    / "impolite_transfer_to_main_v1"                               / "combined_significance.csv", 16, 0.5),
        "sft":      (OLMO3_TRANSFER_BASE / "impolite_base_vectors_to_instruct_universal_grid_layer20_coef0p75_v1" / "combined_significance.csv", 20, 0.75),
        "dpo":      (OLMO3_TRANSFER_BASE / "impolite_base_vectors_to_instruct_universal_grid_layer20_coef0p75_v1" / "combined_significance.csv", 20, 0.75),
        "instruct": (OLMO3_TRANSFER_BASE / "impolite_base_vectors_to_instruct_universal_grid_layer20_coef0p75_v1" / "combined_significance.csv", 20, 0.75),
    },
    "humorous_character_neutral_q": {
        "main":     (OLMO3_GRID_BASE    / "humorous_transfer_to_main_v1"                               / "combined_significance.csv", 20, 0.3),
        "sft":      (OLMO3_TRANSFER_BASE / "humorous_base_vectors_to_instruct_universal_grid_layer20_coef0p30_v1" / "combined_significance.csv", 20, 0.3),
        "dpo":      (OLMO3_TRANSFER_BASE / "humorous_base_vectors_to_instruct_universal_grid_layer20_coef0p30_v1" / "combined_significance.csv", 20, 0.3),
        "instruct": (OLMO3_TRANSFER_BASE / "humorous_base_vectors_to_instruct_universal_grid_layer20_coef0p30_v1" / "combined_significance.csv", 20, 0.3),
    },
}

# Fixed color per target, ordered by post-training depth.
TARGET_LINE_COLORS = {
    "main":     "#648fff",  # base model — blue
    "sft":      "#ffb000",  # yellow
    "dpo":      "#fe6100",  # orange
    "instruct": "#dc267f",  # pink/magenta (full DPO+RLVR)
}
TARGET_PRETTY = {
    "main":     "base (main)",
    "sft":      "SFT",
    "dpo":      "DPO",
    "instruct": "RLVR",
}
TARGET_ORDER = ("main", "sft", "dpo", "instruct")


def _target_label(target: str, model: str) -> str:
    # Apertus ships a single post-trained variant with no SFT/DPO/RLVR
    # decomposition — its `instruct` line is just "Instruct".
    if model == "apertus" and target == "instruct":
        return "Instruct"
    return TARGET_PRETTY[target]


def render_per_trait(model: str, trait_key: str, trait_label: str,
                     out_path: Path, p_threshold: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))

    spec_by_model = {
        "olmo3":   PER_TRAIT_SPECS_OLMO3,
        "apertus": PER_TRAIT_SPECS_APERTUS,
    }
    spec = spec_by_model[model].get(trait_key)
    if spec is None:
        print(f"no per-trait spec for model={model} trait={trait_key}")
        return

    # Reuse the model's instruct config just to get ckpt_grid and stage regions.
    ref_cfg = build_config(model, "instruct")
    grid = ref_cfg.ckpt_grid
    ckpt_to_tokens = ref_cfg.ckpt_to_tokens
    stage_regions = ref_cfg.stage_regions
    xmin_pad_B = ref_cfg.xmin_pad_B

    per_target: list[tuple[str, int, float, pd.DataFrame]] = []
    for target in TARGET_ORDER:
        if target not in spec:
            continue
        csv_path, layer, coef = spec[target]
        eval_model, _ = MODEL_TARGETS[model][target]  # None for "main"
        trait_cfg = TransferTraitConfig(
            key=trait_key, label=trait_label, color="",
            csv_path=csv_path, layer=layer, coef=coef,
        )
        df = load_transfer(trait_cfg, eval_model, ckpt_to_tokens, grid)
        if df.empty:
            continue
        per_target.append((target, layer, coef, df))

    if not per_target:
        print(f"no data for trait={trait_key}")
        return

    all_x: list[float] = []
    for _, _, _, df in per_target:
        all_x.extend(df.tokens_B.tolist())
    display_map = _log_nudge_display(all_x, min_log_gap=0.12, max_log_gap=0.20)

    plot_right_edge = max(display_map.values()) * 1.10 if display_map else None
    stage_band_patches: list = []
    if stage_regions and display_map:
        regions = list(stage_regions)
        prev_right_x: float | None = None
        for idx, region in enumerate(regions):
            left_x = (xmin_pad_B if region.start_B <= 0
                      else _interp_display_log(region.start_B,
                                               region.anchor_left_B,
                                               region.anchor_right_B,
                                               display_map))
            right_x = _interp_display_log(region.end_B,
                                          region.anchor_left_B,
                                          region.anchor_right_B,
                                          display_map)
            if prev_right_x is not None:
                left_x = prev_right_x
            if idx == len(regions) - 1 and plot_right_edge is not None:
                right_x = plot_right_edge
            ax.axvspan(left_x, right_x, color=region.color, alpha=0.12, zorder=0)
            stage_band_patches.append((region, left_x, right_x))
            prev_right_x = right_x

    # If every line uses the same (layer, coef), annotate it once in the
    # title. If they differ, annotate per line in the legend entry.
    unique_lc = {(lay, coef) for _, lay, coef, _ in per_target}
    mixed_lc = len(unique_lc) > 1
    for target, layer, coef, df in per_target:
        color = TARGET_LINE_COLORS[target]
        x_disp = df.tokens_B.map(display_map).to_numpy()
        y = df.delta_trait_mean.to_numpy()
        p = df.p_raw.fillna(1.0).to_numpy()
        ax.plot(x_disp, y, "-", color=color, linewidth=2.4,
                label=_target_label(target, model), zorder=2)
        sig = p < p_threshold
        ax.scatter(x_disp[sig], y[sig], marker="*", s=130, color=color,
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.scatter(x_disp[~sig], y[~sig], marker="o", s=36, facecolor="white",
                   edgecolor=color, linewidth=1.6, zorder=3)

    ax.axhline(0, color="#555", linewidth=0.8, zorder=1)

    ax.set_xscale("log")
    if display_map:
        disp_x = sorted(display_map.values())
        ax.set_xlim(max(xmin_pad_B, min(disp_x) * 0.8), max(disp_x) * 1.10)
        sorted_actual = sorted(display_map.keys())
        tick_positions = [display_map[a] for a in sorted_actual]
        tick_labels = [_fmt_tokens_B(a) for a in sorted_actual]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=30, ha="right")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    ax.set_xlabel("Checkpoint of persona extraction (tokens consumed)", fontsize=11)
    ax.set_ylabel("Trait expression delta on target model", fontsize=11)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.grid(True, which="major", axis="x", ls=":", alpha=0.20)

    # Consolidated legend: target lines + one marker handle explaining ★.
    from matplotlib.lines import Line2D
    line_handles, line_labels = ax.get_legend_handles_labels()
    marker_handles = [Line2D([0], [0], marker="*", color="none",
                             markerfacecolor="black", markeredgecolor="white",
                             markersize=12, linewidth=0,
                             label=f"significant (p < {p_threshold})")]
    ax.legend(handles=line_handles + marker_handles,
              labels=line_labels + [h.get_label() for h in marker_handles],
              loc="upper left", fontsize=9.5, frameon=True, framealpha=0.92,
              handlelength=1.6, labelspacing=0.4,
              title="evaluated on")

    if stage_band_patches:
        for region, left_x, right_x in stage_band_patches:
            if not region.label:
                continue
            mid_log = (math.log10(left_x) + math.log10(right_x)) / 2
            ax.text(10 ** mid_log, 1.02, region.label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=8.5, color=region.color, weight="semibold",
                    clip_on=False, zorder=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {pdf_path}")


# --------------------------- binned per-trait rendering ------------------
# Same data + spec as `render_per_trait`, but the x-axis is the target model
# (post-training depth →) and each line is a training-stage bin (mean ± 1
# SEM). Use this when the per-checkpoint line plot has too many overlapping
# trajectories — the binned view collapses them into 3 stage groups.

# Token-cutoff bin definitions, per model. Each entry: ordered list of
# (label, upper_token_B_exclusive). The final entry's upper bound is +inf.
BINNED_STAGE_BINS: dict[str, tuple[tuple[str, float], ...]] = {
    "olmo3": (
        ("very early (<100 B)", 100.0),
        ("early (100 B – 5 T)", 5_000.0),
        ("midtraining (≥ 5 T)", float("inf")),
    ),
}

# Persona palette uses yellow / green / pink / purple / blue / orange (see
# make_emergence_plot.PALETTE), so bin colors must come from outside those
# hue families. Red / brown / charcoal — three distinct hues, no clash.
BINNED_STAGE_COLORS: dict[str, str] = {
    "very early (<100 B)":   "#d62728",
    "early (100 B – 5 T)":   "#8c564b",
    "midtraining (≥ 5 T)":   "#2b2b2b",
}

# Per-bin marker shapes. Existing line/transfer plots use 'o' (open circle)
# and '*' (significance star), so the binned variant uses square / triangle
# / diamond instead — visually distinct from the rest of the project.
BINNED_STAGE_MARKERS: dict[str, str] = {
    "very early (<100 B)":   "s",
    "early (100 B – 5 T)":   "^",
    "midtraining (≥ 5 T)":   "D",
}


def _bin_revision(rev: str, ckpt_to_tokens, bins: tuple[tuple[str, float], ...]) -> str | None:
    t = ckpt_to_tokens(rev)
    if t is None:
        return None
    for label, upper in bins:
        if t < upper:
            return label
    return bins[-1][0]


def _load_binned_pivot(model: str, trait_key: str, trait_label: str
                        ) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str]]:
    """Shared data preparation for the binned per-trait views.

    Returns (pivot_y, targets_present, bin_members, bin_order). Filters to
    revisions shared across all targets so each bin's mean compares the same
    ckpts at every x-position.
    """
    spec_by_model = {
        "olmo3":   PER_TRAIT_SPECS_OLMO3,
        "apertus": PER_TRAIT_SPECS_APERTUS,
    }
    spec = spec_by_model[model][trait_key]
    ref_cfg = build_config(model, "instruct")
    grid = ref_cfg.ckpt_grid
    ckpt_to_tokens = ref_cfg.ckpt_to_tokens

    rows: list[pd.DataFrame] = []
    for target in TARGET_ORDER:
        if target not in spec:
            continue
        csv_path, layer, coef = spec[target]
        eval_model, _ = MODEL_TARGETS[model][target]
        trait_cfg = TransferTraitConfig(
            key=trait_key, label=trait_label, color="",
            csv_path=csv_path, layer=layer, coef=coef,
        )
        df = load_transfer(trait_cfg, eval_model, ckpt_to_tokens, grid)
        if df.empty:
            continue
        rows.append(df.assign(target=target))
    if not rows:
        raise ValueError(f"no data for trait={trait_key}")
    long_df = pd.concat(rows, ignore_index=True)

    target_to_revs = {t: set(long_df[long_df.target == t].extract_revision)
                      for t in long_df.target.unique()}
    shared = sorted(set.intersection(*target_to_revs.values()),
                    key=ckpt_to_tokens)
    if not shared:
        raise ValueError(f"no shared ckpts across targets for trait={trait_key}")
    long_df = long_df[long_df.extract_revision.isin(shared)]

    targets_present = [t for t in TARGET_ORDER if t in target_to_revs]
    pivot_y = (long_df.pivot_table(index="extract_revision", columns="target",
                                   values="delta_trait_mean")
                       .reindex(shared)[targets_present])

    bins_def = BINNED_STAGE_BINS[model]
    bin_order = [label for label, _ in bins_def]
    bin_members: dict[str, list[str]] = {b: [] for b in bin_order}
    for rev in shared:
        b = _bin_revision(rev, ckpt_to_tokens, bins_def)
        if b is not None:
            bin_members[b].append(rev)
    return pivot_y, targets_present, bin_members, bin_order


def _draw_binned_on_ax(ax: plt.Axes, pivot_y: pd.DataFrame,
                        targets_present: list[str],
                        bin_members: dict[str, list[str]],
                        bin_order: list[str],
                        model: str) -> tuple[list[tuple], list[str]]:
    """Draw the binned mean-±-SEM trajectories on a single axis.

    Returns (legend_handles, legend_labels) so the caller can place a single
    legend (per-axis or shared across a multi-panel figure) at its discretion.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    xs = np.arange(len(targets_present))
    legend_handles: list[tuple] = []
    legend_labels: list[str] = []
    for b in bin_order:
        members = bin_members[b]
        if not members:
            continue
        sub = pivot_y.loc[members]
        n   = sub.shape[0]
        mean = sub.mean(axis=0).to_numpy(dtype=float)
        sem  = sub.std(axis=0, ddof=1).to_numpy(dtype=float) / np.sqrt(n)
        c = BINNED_STAGE_COLORS[b]
        m = BINNED_STAGE_MARKERS[b]
        ax.fill_between(xs, mean - sem, mean + sem, color=c, alpha=0.22,
                        zorder=2, linewidth=0)
        ax.plot(xs, mean, color=c, linestyle="-", marker=m, linewidth=2.4,
                markersize=8, markeredgecolor="white", markeredgewidth=0.7,
                zorder=4)
        line_h = Line2D([0], [0], color=c, linewidth=2.4, marker=m, markersize=8,
                        markeredgecolor="white", markeredgewidth=0.7)
        patch_h = Patch(facecolor=c, alpha=0.22, linewidth=0)
        legend_handles.append((patch_h, line_h))
        legend_labels.append(b)

    ax.axhline(0, color="#555", linewidth=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([_target_label(t, model) for t in targets_present], fontsize=10)
    ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    return legend_handles, legend_labels


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {pdf_path}")


def render_per_trait_binned(model: str, trait_key: str, trait_label: str,
                             out_path: Path,
                             figsize: tuple[float, float] = (9.0, 3.8)) -> None:
    """Per-trait transfer trajectory grouped into training-stage bins.

    For each bin, plots the mean Δ across member ckpts as a line over the
    four target models, with a ±1 SEM shaded band. Useful as a paper-ready
    companion to `render_per_trait` when individual-ckpt lines clutter the
    figure.
    """
    if model not in BINNED_STAGE_BINS:
        print(f"render_per_trait_binned: no bin definitions for model={model}")
        return
    try:
        pivot_y, targets_present, bin_members, bin_order = _load_binned_pivot(
            model, trait_key, trait_label)
    except ValueError as e:
        print(e)
        return

    from matplotlib.legend_handler import HandlerTuple
    fig, ax = plt.subplots(figsize=figsize)
    handles, labels = _draw_binned_on_ax(ax, pivot_y, targets_present, bin_members, bin_order, model)

    ax.set_xlabel("Target model (post-training depth →)", fontsize=11)
    ax.set_ylabel("Trait expression delta on target model", fontsize=11)
    ax.legend(handles, labels,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
              loc="upper right", fontsize=9.5, framealpha=0.92,
              title="Average over checkpoints  ± SE", title_fontsize=9, handlelength=2.4)
    fig.tight_layout()
    _save_fig(fig, out_path)


def render_per_trait_binned_pair(model: str,
                                  traits: tuple[tuple[str, str], tuple[str, str]],
                                  out_path: Path,
                                  figsize: tuple[float, float] = (12.5, 4.0)) -> None:
    """Two binned per-trait panels side by side, sharing one legend.

    `traits` is ((left_key, left_label), (right_key, right_label)). Each
    panel keeps its own y-axis (different traits land at different scales).
    The legend is placed once on the right panel.
    """
    if model not in BINNED_STAGE_BINS:
        print(f"render_per_trait_binned_pair: no bin definitions for model={model}")
        return

    from matplotlib.legend_handler import HandlerTuple

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    legend_payload: tuple[list[tuple], list[str]] | None = None
    for ax, (trait_key, trait_label) in zip(axes, traits):
        try:
            pivot_y, targets_present, bin_members, bin_order = _load_binned_pivot(
                model, trait_key, trait_label)
        except ValueError as e:
            print(e)
            continue
        h, l = _draw_binned_on_ax(ax, pivot_y, targets_present, bin_members, bin_order, model)
        if legend_payload is None:
            legend_payload = (h, l)
        ax.set_xlabel("Target model (post-training depth →)", fontsize=11)
        ax.set_title(trait_label[:1].upper() + trait_label[1:], fontsize=12)

    axes[0].set_ylabel("Trait expression delta on target model", fontsize=11)
    fig.tight_layout()
    if legend_payload is not None:
        # Figure-level legend below the two panels, one row of three entries.
        # Subplots_adjust leaves vertical room for the legend so tight_layout
        # does not clip it.
        fig.subplots_adjust(bottom=0.28)
        fig.legend(legend_payload[0], legend_payload[1],
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
                   loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=len(legend_payload[1]),
                   fontsize=10, framealpha=0.92,
                   title="Average over checkpoints  ± SE", title_fontsize=9, handlelength=2.4,
                   columnspacing=2.0)
    _save_fig(fig, out_path)


# --------------------------- driver --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_BUILDERS), default="olmo3")
    p.add_argument("--target", default="all",
                   help="post-training target to transfer into, or 'all'. "
                        "valid per-model values: " +
                        " | ".join(f"{m}={{{','.join(v)}}}" for m, v in MODEL_TARGETS.items()))
    p.add_argument("--per-trait", action="store_true",
                   help="instead of one plot per target, emit one plot per trait "
                        "with lines for each post-training target (main/sft/dpo/instruct)")
    p.add_argument("--binned", action="store_true",
                   help="with --per-trait: render the binned trajectory variant "
                        "(target on x, mean ± SEM line per training-stage bin) "
                        "instead of the per-checkpoint line plot")
    p.add_argument("--pair", default=None,
                   help="with --per-trait --binned: comma-separated trait labels "
                        "(e.g. evil,sycophantic) to render as a single side-by-side "
                        "two-panel figure instead of one figure per trait")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG (default: story_plots/transfer/<model>/<model>_transfer_to_<target>.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_trait:
        # One plot per trait, lines for each target. Use the instruct config
        # as the canonical source of (trait_key, trait_label) pairs — every
        # trait that has transfer data is defined there.
        cfg = build_config(args.model, "instruct")
        if args.pair:
            if not args.binned:
                raise SystemExit("--pair requires --binned")
            wanted = [w.strip() for w in args.pair.split(",") if w.strip()]
            if len(wanted) != 2:
                raise SystemExit(f"--pair expects exactly two trait labels; got {wanted}")
            by_label = {t.label: (t.key, t.label) for t in cfg.traits}
            missing = [w for w in wanted if w not in by_label]
            if missing:
                raise SystemExit(f"--pair refers to unknown trait label(s) {missing}; "
                                 f"available: {list(by_label)}")
            traits_pair = (by_label[wanted[0]], by_label[wanted[1]])
            slug = "_".join(wanted)
            out = args.out or (REPO / f"analysis/figures/story_plots/transfer/{args.model}/{args.model}_transfer_binned_pair_{slug}.png")
            render_per_trait_binned_pair(args.model, traits_pair, out)
            return
        for trait in cfg.traits:
            if args.binned:
                out = args.out or (REPO / f"analysis/figures/story_plots/transfer/{args.model}/{args.model}_transfer_binned_{trait.label}.png")
                render_per_trait_binned(args.model, trait.key, trait.label, out)
            else:
                out = args.out or (REPO / f"analysis/figures/story_plots/transfer/{args.model}/{args.model}_transfer_by_trait_{trait.label}.png")
                render_per_trait(args.model, trait.key, trait.label, out)
        return
    valid = MODEL_TARGETS[args.model]
    if args.target == "all":
        targets = list(valid)
    elif args.target in valid:
        targets = [args.target]
    else:
        raise SystemExit(f"--target={args.target!r} not valid for --model={args.model!r}; "
                         f"pick from {list(valid)}")
    for tgt in targets:
        cfg = build_config(args.model, tgt)
        out = args.out or (REPO / f"analysis/figures/story_plots/transfer/{args.model}/{args.model}_transfer_to_{tgt}.png")
        render(cfg, out)


if __name__ == "__main__":
    main()
