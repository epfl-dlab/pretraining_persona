"""
Same-checkpoint emergence plot: persona direction extracted at revision r and
evaluated at the SAME revision r (r == r along the diagonal of the grid).

Produces one figure per model showing Δ trait score vs. tokens consumed, with
one curve per trait. Raw (non-Bonferroni) primary-test significance is marked
above each point with stars:
  *   p < 0.05
  **  p < 0.01
  *** p < 0.001

Design goals for this module:
  - Reusable: model + trait configs are data, not hard-coded control flow.
  - Readable: one function per logical step (load, prepare, render).
  - Composable: the same entry point handles OLMo-3 today and any future model
    by passing a ModelConfig + list of TraitConfig.

Colors use an IBM colorblind-safe palette passed in from the caller.

Usage:
  python3 analysis/make_emergence_plot.py --model olmo3
  python3 analysis/make_emergence_plot.py --model olmo3 --out custom_path.png
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# IBM colorblind-safe palette, mapped per-trait (alphabetical) so colors stay
# pinned to a trait regardless of list ordering across emergence/transfer figures.
TRAIT_COLORS = {
    "evil_character_neutral_q":        "#FFB000",
    "humorous_character_neutral_q":    "#648FFF",
    "impolite_character_neutral_q":    "#DC267F",
    "sycophantic_character_neutral_q": "#5BD492",
}
TRAIT_LABELS = {
    "evil_character_neutral_q":        "Evil",
    "humorous_character_neutral_q":    "Humorous",
    "impolite_character_neutral_q":    "Impolite",
    "sycophantic_character_neutral_q": "Sycophantic",
}
# Kept for backwards-compat with external imports; aligned with the per-trait
# alphabetical mapping above.
PALETTE = [TRAIT_COLORS[k] for k in
           ("evil_character_neutral_q", "humorous_character_neutral_q",
            "impolite_character_neutral_q", "sycophantic_character_neutral_q")]


# --------------------------- config --------------------------------------

@dataclass(frozen=True)
class TraitConfig:
    key: str           # trait column value in CSV, e.g. "evil_character_neutral_q"
    label: str         # axis legend text, e.g. "evil"
    color: str         # hex string from PALETTE
    csv_path: Path     # combined_significance.csv to load
    coef: float        # operating coef for the emergence curve
    # Optional supplementary (csv_path, coef) segments merged into the same-
    # checkpoint diagonal for this trait. Use when no single sweep covers the
    # full ckpt grid at a coherence-preserving coef (e.g. Apertus evil: early
    # ckpts at c=0.2 from one sweep + late ckpts at c=0.1 from another).
    # Each segment is filtered to its own coef; checkpoints already supplied
    # by an earlier segment are NOT overridden.
    extra_segments: tuple[tuple[Path, float], ...] = ()
    # Optional post-training "instruct self-steering" result. When set, the
    # figure adds one extra point per eval_model at the far right. Ordered
    # left→right as they appear in the list.
    instruct_csv: Path | None = None
    instruct_eval_models: tuple[tuple[str, str], ...] = ()  # (eval_model, tick_label)


@dataclass(frozen=True)
class StageRegion:
    """A pretraining-stage region, rendered as a shaded axvspan + label.

    `start_B` / `end_B` are in the model's native token units (B). The span
    is laid out in *display* log space, anchored to two plotted checkpoints
    (`anchor_left_B` and `anchor_right_B`) so the shading aligns with the
    nudged axis even when the stage boundary is not itself a plotted ckpt.
    """
    start_B: float
    end_B: float
    label: str
    color: str
    anchor_left_B: float   # a plotted ckpt whose tokens ≤ start_B
    anchor_right_B: float  # a plotted ckpt whose tokens ≥ end_B


@dataclass(frozen=True)
class ModelConfig:
    name: str                              # pretty name for title, e.g. "OLMo-3 7B"
    path_name: str                         # on-disk model dir, e.g. "Olmo-3-1025-7B"
    traits: list[TraitConfig]
    ckpt_to_tokens: callable                # function rev -> tokens in B
    stage_boundaries_B: list[float]         # tokens (in B) where to draw dashed verticals
    xmin_pad_B: float = 10.0                # left x-axis padding, in B
    # Checkpoints to *exclude* from the plot (e.g. stage-end duplicates that
    # cluster with `main` in log-token space and create visual pile-ups).
    excluded_revisions: tuple[str, ...] = ()
    # Checkpoints to render as "vector extraction failed" markers for *every*
    # trait, even when no per-trait status.json exists on disk yet. Use for
    # documented pre-emergence checkpoints that the modern pipeline hasn't
    # been re-run at (e.g. pass rates so low that no pos/neg rows survive
    # the judge filter, so the extraction loop was never attempted under
    # the modern status-tracking code path).
    forced_failed_revisions: tuple[str, ...] = ()
    # Optional pretraining-stage shaded regions drawn in the background.
    # Left→right order is preserved; labels sit above the axis.
    stage_regions: tuple[StageRegion, ...] = ()


# --------------------------- OLMo-3 token mapping ------------------------
# From OLMo-3 paper (arXiv:2512.13961 §3.4): 4M tokens/step stage1+3, 2M/step stage2.
# Values in billions of tokens so the axis shares units with Apertus.

_TOK_4M_B = 4.194304 / 1000.0
_TOK_2M_B = 2.097152 / 1000.0
_OLMO_STAGE1_END_B = 1_413_814 * _TOK_4M_B
_OLMO_STAGE2_END_B = _OLMO_STAGE1_END_B + 47_684 * _TOK_2M_B
_OLMO_STAGE3_END_B = _OLMO_STAGE2_END_B + 11_921 * _TOK_4M_B
_OLMO_STEP_RE = re.compile(r"^stage([123])-step(\d+)$")


def olmo3_ckpt_to_tokens_B(rev: str) -> float | None:
    """Cumulative tokens in B for an OLMo-3-7B checkpoint name. `main` == stage3 end."""
    if rev == "main":
        return _OLMO_STAGE3_END_B
    m = _OLMO_STEP_RE.match(rev)
    if m is None:
        return None
    stage, step = int(m.group(1)), int(m.group(2))
    if stage == 1:
        return step * _TOK_4M_B
    if stage == 2:
        return _OLMO_STAGE1_END_B + step * _TOK_2M_B
    if stage == 3:
        return _OLMO_STAGE2_END_B + step * _TOK_4M_B
    return None


# --------------------------- model configurations ------------------------

OLMO3_PATH_NAME = "Olmo-3-1025-7B"


def olmo3_config() -> ModelConfig:
    base = REPO / f"results/{OLMO3_PATH_NAME}/checkpoint_grid"
    instruct_base = REPO / "results/self_steering"
    # Ordered post-training stages (left→right in the figure):
    INSTRUCT_STAGES = (
        ("allenai/Olmo-3-7B-Instruct-SFT", "SFT"),
        ("allenai/Olmo-3-7B-Instruct-DPO", "DPO"),
        ("allenai/Olmo-3-7B-Instruct",     "Instruct"),
    )
    spec = [
        ("evil_character_neutral_q",        "evil_stage_progression_diag_modern_v1",      0.5,  "evil_instruct_self_steering_layer16_coef0p55_v1"),
        ("humorous_character_neutral_q",    "humorous_stage_progression_diag_v1",         0.3,  "humorous_instruct_self_steering_layer20_coef0p3_v1"),
        ("impolite_character_neutral_q",    "impolite_stage_progression_diag_v1",         0.5,  "impolite_instruct_self_steering_layer16_coef0p5_v1"),
        ("sycophantic_character_neutral_q", "sycophantic_stage_progression_diag_no0_v1",  0.5,  "sycophantic_instruct_self_steering_layer16_coef0p5_v1"),
    ]
    traits = [
        TraitConfig(
            key=k, label=TRAIT_LABELS[k], color=TRAIT_COLORS[k],
            csv_path=base / d / "combined_significance.csv",
            coef=c,
            instruct_csv=instruct_base / idir / "combined.csv",
            instruct_eval_models=INSTRUCT_STAGES,
        )
        for k, d, c, idir in spec
    ]
    # Pretraining stages from OLMo-3 paper (arXiv:2512.13961v2, §3.4):
    # stage1 = Dolma 3 pretraining (5.93 T), stage2 = Dolmino mid (100 B),
    # stage3 = Longmino long-ctx (50 B). All three stage-end ckpts are
    # plotted, so each boundary coincides with a plotted ckpt and we use
    # it directly as both anchors (the interpolation becomes an identity).
    s1_end = olmo3_ckpt_to_tokens_B("stage1-step1413814")   # 5.93 T
    s3_end = _OLMO_STAGE3_END_B                             # 6.08 T == main
    stage_regions = (
        StageRegion(
            start_B=0.0, end_B=s1_end,
            label="Stage 1: Dolma 3 pretraining",
            color="#4c78a8",
            anchor_left_B=olmo3_ckpt_to_tokens_B("stage1-step3000"),
            anchor_right_B=s1_end,
        ),
        StageRegion(
            start_B=s1_end, end_B=s3_end,
            label="Stages 2+3:\nmidtraining",
            color="#f58518",
            anchor_left_B=s1_end,
            anchor_right_B=s3_end,
        ),
    )
    return ModelConfig(
        name="OLMo-3 7B base",
        path_name=OLMO3_PATH_NAME,
        traits=traits,
        ckpt_to_tokens=olmo3_ckpt_to_tokens_B,
        stage_boundaries_B=[_OLMO_STAGE1_END_B, _OLMO_STAGE2_END_B],
        xmin_pad_B=2.5,
        # Keep stage-end ckpts for stages 1 and 2 so the plot shows the real
        # transition through each stage. Drop `stage3-step11921` because it
        # is literally the same model weights as `main` (stage 3 end = main);
        # `_log_nudge_display` would otherwise spread two identical points
        # apart as if they were distinct training positions.
        excluded_revisions=("stage3-step11921",),
        # stage1-step1000 (~4B): judged pass rates for pos/neg are near-zero for
        # all four traits (see results/Olmo-3-1025-7B/pass_rate.csv and
        # analysis/olmo3_experiment_results.md §"evil Early-Boundary Probe"),
        # so no pos/neg rows survive the coherence/trait filter and no
        # persona vector can be built. Show these as "no extractable vector"
        # markers to communicate the true left edge of emergence.
        forced_failed_revisions=("stage1-step1000",),
        stage_regions=stage_regions,
    )


# --------------------------- Apertus token mapping -----------------------
# Apertus branch names encode cumulative tokens: `step50000-tokens210B` → 210 B;
# `step2627139-tokens15T` → 15 000 B. Preferred over linear extrapolation from
# step because Apertus doubles batch size after 8 T tokens (paper §Table 2).

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


APERTUS_PATH_NAME = "Apertus-8B-2509"


def apertus_config() -> ModelConfig:
    base = REPO / f"results/{APERTUS_PATH_NAME}/checkpoint_grid"
    # Apertus evil uses mixed coefs for meaningful Δ throughout training:
    #   - early (210 B – 4.2 T): c=0.2 (v2 sweep). c=0.5 collapses the model
    #     on these ckpts (Δ_coh ≈ −85) so the judge scores mangled outputs as
    #     Δ ≈ −3 (non-emergence artifact).
    #   - late  (6 T – 15 T): c=0.5 (v1 sweep). The model does NOT collapse
    #     on late ckpts at c=0.5 (Δ_coh ≈ −30 to −50) and Δ_trait stays
    #     strong (19–42). c=0.1 was tested on late ckpts and gave
    #     marginal Δ (5–11) — too weak for the figure.
    # Documented in analysis/olmo3_experiment_results.md under
    # "Apertus Same-Checkpoint Emergence — Mixed-Coef Evil".
    evil_primary = base / "apertus_evil_stage_progression_diag_v2" / "combined_significance.csv"
    evil_late    = base / "apertus_evil_stage_progression_diag_v1" / "combined_significance.csv"
    traits = [
        TraitConfig(
            key="evil_character_neutral_q",
            label=TRAIT_LABELS["evil_character_neutral_q"],
            color=TRAIT_COLORS["evil_character_neutral_q"],
            csv_path=evil_primary, coef=0.2,
            extra_segments=((evil_late, 0.5),),
        ),
        TraitConfig(
            key="impolite_character_neutral_q",
            label=TRAIT_LABELS["impolite_character_neutral_q"],
            color=TRAIT_COLORS["impolite_character_neutral_q"],
            csv_path=base / "apertus_impolite_same_checkpoint_layer20_coef0p15" / "combined_significance.csv",
            coef=0.15,
        ),
        TraitConfig(
            key="sycophantic_character_neutral_q",
            label=TRAIT_LABELS["sycophantic_character_neutral_q"],
            color=TRAIT_COLORS["sycophantic_character_neutral_q"],
            csv_path=base / "syco_same_ckpt_diagonal" / "combined_significance.csv",
            coef=0.2,
        ),
    ]
    # Apertus training has one continuous pretraining phase (0 → 13.5 T,
    # batch doubles at 8 T but training is continuous) and a final cooldown
    # (13.5 T → 15 T: LR annealing + data-mix change). Paper arXiv:2509.14233
    # Table 2 + §2.3. 13.5 T ≈ step 2 100 000 – step 2 400 000 boundary;
    # 15 T is the final released ckpt (step 2 627 139).
    _APERTUS_COOLDOWN_START_B = 13_500.0
    _APERTUS_END_B = 15_000.0
    stage_regions = (
        StageRegion(
            start_B=0.0, end_B=_APERTUS_COOLDOWN_START_B,
            label="Stage 1: pretraining (13.5 T)",
            color="#4c78a8",
            anchor_left_B=apertus_ckpt_to_tokens_B("step50000-tokens210B"),
            anchor_right_B=apertus_ckpt_to_tokens_B("step2400000-tokens13112B"),
        ),
        StageRegion(
            start_B=_APERTUS_COOLDOWN_START_B, end_B=_APERTUS_END_B,
            label="Stage 2: midtraining",
            color="#f58518",
            anchor_left_B=apertus_ckpt_to_tokens_B("step2400000-tokens13112B"),
            anchor_right_B=apertus_ckpt_to_tokens_B("step2627139-tokens15T"),
        ),
    )
    return ModelConfig(
        name="Apertus-8B base",
        path_name=APERTUS_PATH_NAME,
        traits=traits,
        ckpt_to_tokens=apertus_ckpt_to_tokens_B,
        stage_boundaries_B=[],
        xmin_pad_B=120.0,
        # `main` and `step2627139-tokens15T` both resolve to 15 000 B — drop
        # the `main` alias so the log axis has one clean tick at 15 T.
        excluded_revisions=("main",),
        stage_regions=stage_regions,
    )


def apertus_high_coef_config() -> ModelConfig:
    """Variant of `apertus_config` that uses a uniform high coef across all
    ckpts for every trait — no mixed-coef transitions. Useful as a companion
    figure showing the raw c=0.5 behavior of Apertus evil across the full
    training grid, including the mid-training coherence-collapse window
    (420 B – 1.68 T) where the judge scores the mangled outputs as Δ ≈ −3.
    Syco and impo stay at their existing diagonal coefs (0.2 and 0.15) since
    no higher-coef full-grid diagonal sweeps exist for those traits.
    """
    base = REPO / f"results/{APERTUS_PATH_NAME}/checkpoint_grid"
    traits = [
        TraitConfig(
            key="evil_character_neutral_q",
            label=TRAIT_LABELS["evil_character_neutral_q"],
            color=TRAIT_COLORS["evil_character_neutral_q"],
            csv_path=base / "apertus_evil_stage_progression_diag_v1" / "combined_significance.csv",
            coef=0.5,
        ),
        TraitConfig(
            key="impolite_character_neutral_q",
            label=TRAIT_LABELS["impolite_character_neutral_q"],
            color=TRAIT_COLORS["impolite_character_neutral_q"],
            csv_path=base / "apertus_impolite_same_checkpoint_layer20_coef0p15" / "combined_significance.csv",
            coef=0.15,
        ),
        TraitConfig(
            key="sycophantic_character_neutral_q",
            label=TRAIT_LABELS["sycophantic_character_neutral_q"],
            color=TRAIT_COLORS["sycophantic_character_neutral_q"],
            csv_path=base / "syco_same_ckpt_diagonal" / "combined_significance.csv",
            coef=0.2,
        ),
    ]
    return ModelConfig(
        name="Apertus-8B base  (uniform coef, high-coef evil)",
        path_name=APERTUS_PATH_NAME,
        traits=traits,
        ckpt_to_tokens=apertus_ckpt_to_tokens_B,
        stage_boundaries_B=[],
        xmin_pad_B=120.0,
        excluded_revisions=("main",),
        stage_regions=(
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
        ),
    )


MODELS: dict[str, callable] = {
    "olmo3":             olmo3_config,
    "apertus":           apertus_config,
    "apertus_high_coef": apertus_high_coef_config,
}

# Output layout: (subfolder, filename stem). Subfolder groups figures by
# underlying model so OLMo and Apertus outputs stay separate on disk even
# when they share a generator script. Filenames keep the model prefix so an
# individual file is still identifiable without relying on its directory.
MODEL_OUT_LAYOUT: dict[str, tuple[str, str]] = {
    "olmo3":             ("olmo3",   "olmo3_same_checkpoint_emergence"),
    "apertus":           ("apertus", "apertus_same_checkpoint_emergence"),
    "apertus_high_coef": ("apertus", "apertus_high_coef_same_checkpoint_emergence"),
}


# --------------------------- data loading ---------------------------------

def _load_segment(csv_path: Path, coef: float, ckpt_to_tokens,
                  excluded: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df.extract_revision == df.eval_revision].copy()
    df = df[~df.extract_revision.isin(excluded)].copy()
    if "coef" in df.columns:
        df = df[df.coef.astype(float).round(3) == round(coef, 3)].copy()
    df["tokens_B"] = df.extract_revision.map(ckpt_to_tokens)
    df = df.dropna(subset=["tokens_B", "delta_trait_mean"]).copy()
    df["p_raw"] = df.get("trait_primary_p_two_sided", pd.Series(dtype=float))
    df["coef_used"] = float(coef)
    return df[["extract_revision", "tokens_B", "delta_trait_mean", "p_raw", "coef_used"]]


def load_diagonal(trait: TraitConfig, ckpt_to_tokens,
                  excluded: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return same-checkpoint diagonal rows with (tokens_B, delta, p, coef_used) columns.

    If the trait defines `extra_segments`, additional CSVs are concatenated at
    their per-segment coef. First-seen-wins on `extract_revision` so the
    primary segment's ckpts take priority over supplementary ones.
    """
    segments = [(trait.csv_path, trait.coef), *trait.extra_segments]
    frames = [_load_segment(p, c, ckpt_to_tokens, excluded) for p, c in segments]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["extract_revision"], keep="first")
    combined = combined.sort_values("tokens_B").reset_index(drop=True)
    return combined


def _classify_failure(payload: dict | None) -> str:
    """Return 'trait', 'coherence', or 'unknown' for a nonextractable
    status.json payload.

    Uses pos-side counts (the "this trait under the pos prompt" side is
    usually where emergence bites first):
      - pos_trait_ge_threshold_count <  pos_coherence_ge_50_count
        → trait-limited: the model can produce coherent pos responses
          but few of them actually express the trait yet.
      - pos_coherence_ge_50_count   <= pos_trait_ge_threshold_count
        → coherence-limited: the model produces in-persona pos responses
          but they are not coherent enough to survive the filter.
          Ties (pos_t == pos_c, including the 0/0 case at very early
          checkpoints where the model has nothing usable on either axis)
          resolve to coherence — matching the intuition that early
          pre-emergence models are incoherent first, not trait-deficient.

    Returns 'unknown' when fields are missing (e.g. forced entries that
    predate modern status.json writing).
    """
    if not payload:
        return "unknown"
    pos_t = payload.get("pos_trait_ge_threshold_count")
    pos_c = payload.get("pos_coherence_ge_50_count")
    if pos_t is None or pos_c is None:
        return "unknown"
    return "trait" if pos_t < pos_c else "coherence"


def load_failed_revisions(trait: TraitConfig, path_name: str,
                          forced: tuple[str, ...],
                          excluded: tuple[str, ...],
                          ckpt_to_tokens) -> list[tuple[str, float, str]]:
    """Return (revision, tokens_B, failure_kind) for checkpoints where the
    trait's persona vector could not be built.

    Detection:
      - `.status.json` exists in data/persona_vectors/<model>/<rev>/ but the
        matching `.pt` does not — the extraction pipeline recorded a non-
        extractable outcome (zero rows passing trait/coherence filters).
      - `rev` is in `forced` and no `.pt` exists — a documented failure from
        an earlier probe that hasn't been re-materialized as a status.json.

    `failure_kind` is one of 'trait', 'coherence', 'unknown' (see
    _classify_failure). Skips checkpoints in `excluded` and checkpoints
    the token-mapper rejects.
    """
    vec_root = REPO / "data/persona_vectors" / path_name
    trait_key = trait.key
    excluded_set = set(excluded)
    seen: dict[str, tuple[float, str]] = {}

    status_glob = f"*/{trait_key}_response_avg_diff.status.json"
    for status_path in sorted(vec_root.glob(status_glob)):
        rev = status_path.parent.name
        if rev in excluded_set:
            continue
        pt_path = status_path.parent / f"{trait_key}_response_avg_diff.pt"
        if pt_path.exists():
            continue
        try:
            payload = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload and payload.get("status") not in (None, "nonextractable"):
            continue
        tokens = ckpt_to_tokens(rev)
        if tokens is None:
            continue
        seen[rev] = (tokens, _classify_failure(payload))

    for rev in forced:
        if rev in excluded_set or rev in seen:
            continue
        pt_path = vec_root / rev / f"{trait_key}_response_avg_diff.pt"
        if pt_path.exists():
            continue
        tokens = ckpt_to_tokens(rev)
        if tokens is None:
            continue
        seen[rev] = (tokens, "unknown")

    return sorted(
        ((rev, tok, kind) for rev, (tok, kind) in seen.items()),
        key=lambda t: t[1],
    )


def load_instruct_points(trait: TraitConfig) -> list[tuple[str, float, float]]:
    """Return one (tick_label, delta_trait_mean, p_raw) per configured instruct eval model,
    in the order listed in trait.instruct_eval_models. Silently skips any that are missing."""
    if trait.instruct_csv is None or not trait.instruct_csv.exists() or not trait.instruct_eval_models:
        return []
    df = pd.read_csv(trait.instruct_csv)
    out: list[tuple[str, float, float]] = []
    for eval_model, tick_label in trait.instruct_eval_models:
        sub = df[df.eval_model == eval_model]
        if sub.empty:
            continue
        if "coef" in sub.columns:
            matched = sub[sub.coef.astype(float).round(3) == round(trait.coef, 3)]
            if not matched.empty:
                sub = matched
        row = sub.iloc[0]
        out.append((
            tick_label,
            float(row["delta_trait_mean"]),
            float(row.get("trait_primary_p_two_sided", float("nan"))),
        ))
    return out


def _fmt_tokens_B(n_B: float) -> str:
    """Short human-readable tick label. Prefer T when >= 1000 B."""
    if n_B >= 1000:
        t = n_B / 1000.0
        return f"{t:.1f}T" if t < 10 else f"{t:.0f}T"
    if n_B >= 100:
        return f"{n_B:.0f}B"
    if n_B >= 10:
        return f"{n_B:.0f}B"
    return f"{n_B:.1f}B"


import math


def _log_nudge_display(xs: list[float], min_log_gap: float = 0.06,
                       max_log_gap: float | None = None) -> dict[float, float]:
    """Return {actual_x -> display_x} with each consecutive log-gap clipped to
    [min_log_gap, max_log_gap].

    Clipping the *maximum* gap is important on OLMo-3: the natural log spacing
    has a ~0.46 dex jump between 415 B and 1.2 T (and similar between 3 T and
    6.1 T) while early ckpts sit at 0.12-0.18 dex. Clipping the upper bound
    spreads early points across the same visual room that was wasted on those
    big jumps.

    Ticks stay labeled with the TRUE token count; only the plotting x is
    shifted. The left endpoint is pinned to its true log position so the
    absolute scale is still roughly calibrated at the origin.
    """
    if not xs:
        return {}
    xs_sorted = sorted(set(xs))
    log_xs = [math.log10(x) for x in xs_sorted]
    gaps = [log_xs[i + 1] - log_xs[i] for i in range(len(log_xs) - 1)]
    clipped = [max(g, min_log_gap) for g in gaps]
    if max_log_gap is not None:
        clipped = [min(g, max_log_gap) for g in clipped]
    # Rebuild positions from the left endpoint plus clipped gaps.
    nudged = [log_xs[0]]
    for g in clipped:
        nudged.append(nudged[-1] + g)
    return {x: 10 ** lx for x, lx in zip(xs_sorted, nudged)}


# --------------------------- rendering -----------------------------------

def _interp_display_log(token_B: float, anchor_left_B: float, anchor_right_B: float,
                        display_map: dict[float, float]) -> float:
    """Return the display x (on log axis) for an arbitrary token value by
    linear interpolation in the nudged-log space between two anchor ckpts.

    Why interpolate: the stage boundaries (e.g. 5929 B) are NOT plotted
    ckpts — they fall INSIDE the gap between stage1-step707000 (2966 B) and
    `main` (6079 B). The nudge-map only knows display positions for plotted
    ckpts, so we project the boundary's true-log fraction onto the nudged
    display-log segment between its two surrounding anchors.

    If an anchor is not present in display_map (e.g. per-trait plot where
    that trait has no data at the anchor ckpt), fall back to the nearest
    plotted ckpt whose token count bounds `anchor_left_B` below and
    `anchor_right_B` above. If still missing, return the left/right edge
    of display_map as a last resort so the band clamps to the plotted range.
    """
    sorted_keys = sorted(display_map.keys())
    if not sorted_keys:
        return 1.0

    def _resolve(anchor_B: float, prefer_le: bool) -> tuple[float, float]:
        if anchor_B in display_map:
            return anchor_B, display_map[anchor_B]
        if prefer_le:
            below = [k for k in sorted_keys if k <= anchor_B]
            k = below[-1] if below else sorted_keys[0]
        else:
            above = [k for k in sorted_keys if k >= anchor_B]
            k = above[0] if above else sorted_keys[-1]
        return k, display_map[k]

    anchor_left_resolved,  left_disp  = _resolve(anchor_left_B,  prefer_le=True)
    anchor_right_resolved, right_disp = _resolve(anchor_right_B, prefer_le=False)
    t_log = math.log10(token_B)
    l_log = math.log10(anchor_left_resolved)
    r_log = math.log10(anchor_right_resolved)
    frac = (t_log - l_log) / (r_log - l_log) if r_log != l_log else 0.0
    frac = max(0.0, min(1.0, frac))
    return 10 ** (math.log10(left_disp) + frac * (math.log10(right_disp) - math.log10(left_disp)))


def render(cfg: ModelConfig, out_path: Path, p_threshold: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))

    # Collect all actual x values first so we can compute the display-nudge
    # consistently across all traits (same ckpt → same display position).
    per_trait_df: list[tuple[TraitConfig, pd.DataFrame]] = []
    per_trait_failed: list[tuple[TraitConfig, list[tuple[str, float, str]]]] = []
    data_x: list[float] = []   # xs where we have a persona vector and Δ
    failed_x: list[float] = [] # xs where extraction failed (shown as marker only)
    for trait in cfg.traits:
        df = load_diagonal(trait, cfg.ckpt_to_tokens, cfg.excluded_revisions)
        if not df.empty:
            per_trait_df.append((trait, df))
            data_x.extend(df.tokens_B.tolist())
        fails = load_failed_revisions(trait, cfg.path_name,
                                      cfg.forced_failed_revisions,
                                      cfg.excluded_revisions,
                                      cfg.ckpt_to_tokens)
        per_trait_failed.append((trait, fails))
        failed_x.extend(t for _, t, _ in fails)
    # Nudge the data portion of the axis as before.
    display_map = _log_nudge_display(data_x, min_log_gap=0.15, max_log_gap=0.22)
    # Place failed ckpts in a slightly tightened cluster just left of the
    # first data ckpt. Their exact log positions don't carry information,
    # so spreading them at natural log gaps (0.30 dex between 4.2 B and
    # 8.4 B) wastes horizontal room. 0.12 dex between failed ckpts and
    # 0.15 dex to the first data point is a gentle compression that keeps
    # the markers legible as separate points without hogging space.
    #
    # CRITICAL: only collapse ckpts that have NO data for any trait. Ckpts
    # like stage1-step3000 (13 B) or stage1-step9000 (38 B) appear as data
    # points for some traits AND as failure markers for others (different
    # traits have different extraction outcomes at the same early ckpt);
    # those must keep their nudged data-position so the trait lines don't
    # zig-zag backwards to the failure cluster.
    data_token_set = set(data_x)
    failed_only = [x for x in failed_x if x not in data_token_set]
    if failed_only and data_x:
        failed_sorted = sorted(set(failed_only))
        data_left_log = math.log10(min(display_map.values()))
        anchor_log = data_left_log - 0.15
        log_positions = [anchor_log - 0.12 * i for i in range(len(failed_sorted))]
        log_positions.reverse()
        for x, lx in zip(failed_sorted, log_positions):
            display_map[x] = 10 ** lx
    all_x = data_x + failed_x

    # Determine the post-training column labels (union across all traits,
    # preserving the first-seen order) and compute their display x positions.
    # Post-training columns sit to the right of base_max, evenly spaced on the
    # log axis.
    base_max = max(display_map.values()) if display_map else 1.0
    post_labels: list[str] = []
    for trait, _ in per_trait_df:
        for tick_label, _, _ in load_instruct_points(trait):
            if tick_label not in post_labels:
                post_labels.append(tick_label)
    # Post-column layout: first column close to the last base ckpt (no big
    # jump from 6.1 T to SFT), then evenly spaced at the same dex stride that
    # `_log_nudge_display` uses for base ckpts (~0.22 dex) so the SFT / DPO /
    # Instruct columns feel like a continuation of the grid rather than a
    # far-right cluster.
    post_offsets = [0.22 + 0.22 * i for i in range(len(post_labels))]
    post_x = {lbl: base_max * (10 ** off) for lbl, off in zip(post_labels, post_offsets)}
    divider_x = base_max * (10 ** 0.11)
    # Leftmost post-training background edge. When no post-training exists,
    # set to None and the final stage band will extend to the plot's right.
    # With post[0]=+0.22, this puts the green-band left edge at +0.10 dex
    # past the last base ckpt, so the orange pretraining band has a short
    # "post-Stage-3" tail before handing off to the green post-training band.
    post_left = (min(post_x.values()) * 10 ** -0.12) if post_labels else None

    # Shade pretraining-stage regions (drawn under everything). The final
    # region's right edge is extended to `post_left` (when post-training is
    # present) so that no white gap appears between the last pretraining
    # band and the post-training band.
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
            # Snap this band's left edge to the prior band's right edge so
            # adjacent stages share a boundary and leave no white gap, even
            # when the two regions use different anchor ckpts for their
            # interpolations (e.g. Apertus stage-1 anchor=13112, stage-2
            # anchor=15000 → boundary 13500 lands at different display-x).
            if prev_right_x is not None:
                left_x = prev_right_x
            if idx == len(regions) - 1 and post_left is not None:
                right_x = post_left
            ax.axvspan(left_x, right_x, color=region.color, alpha=0.12, zorder=0)
            stage_band_patches.append((region, left_x, right_x))
            prev_right_x = right_x

    # Union of all checkpoints with data across traits, used to detect gaps
    # where one trait skipped a checkpoint that other traits resolved (e.g.
    # humorous has no usable extraction at OLMo-3 stage1-step9000 ≈ 38 B,
    # so the segment between 29 B and 42 B should be broken rather than
    # interpolated).
    all_data_tokens_sorted = sorted(set(data_x))
    for trait, df in per_trait_df:
        x_disp = df.tokens_B.map(display_map).to_numpy()
        y = df.delta_trait_mean.to_numpy()
        p_base = df.p_raw.fillna(1.0).to_numpy()
        trait_tokens = df.tokens_B.to_numpy()

        line_x: list[float] = []
        line_y: list[float] = []
        for i in range(len(trait_tokens)):
            line_x.append(float(x_disp[i]))
            line_y.append(float(y[i]))
            if i + 1 < len(trait_tokens):
                cur = all_data_tokens_sorted.index(trait_tokens[i])
                nxt = all_data_tokens_sorted.index(trait_tokens[i + 1])
                if nxt - cur > 1:
                    line_x.append(np.nan)
                    line_y.append(np.nan)

        # Base segment (solid; NaN-broken across skipped checkpoints)
        ax.plot(line_x, line_y, "-", color=trait.color, linewidth=2.4,
                label=trait.label, zorder=2)

        # Post-training points (dotted line threading through them)
        inst_points = load_instruct_points(trait)
        if inst_points:
            xs_i = np.array([post_x[lbl] for lbl, _, _ in inst_points])
            ys_i = np.array([d for _, d, _ in inst_points])
            ps_i = np.array([p for _, _, p in inst_points])
            # Connector from last base point through each post-training column
            ax.plot(np.concatenate([[x_disp[-1]], xs_i]),
                    np.concatenate([[y[-1]],     ys_i]),
                    "-", color=trait.color, linewidth=2.4, alpha=0.75, zorder=2)
        else:
            xs_i = np.array([]); ys_i = np.array([]); ps_i = np.array([])

        # Markers: star if significant, open circle otherwise.
        x_all = np.concatenate([x_disp, xs_i])
        y_all = np.concatenate([y, ys_i])
        p_all = np.concatenate([p_base, ps_i])
        sig = p_all < p_threshold
        ax.scatter(x_all[sig], y_all[sig], marker="*", s=130, color=trait.color,
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.scatter(x_all[~sig], y_all[~sig], marker="o", s=36, facecolor="white",
                   edgecolor=trait.color, linewidth=1.6, zorder=3)

    ax.axhline(0, color="#555", linewidth=0.8, zorder=1)

    # Failed-vector markers: per-trait marker at the checkpoint's display-x,
    # placed in a shallow band just below the y=0 baseline. Marker shape
    # encodes *why* extraction failed (see _classify_failure):
    #   ×  coherence-limited — pos set is in-persona but not coherent
    #   +  trait-limited     — pos set is coherent but not in-persona
    #   ○  unknown           — no status.json payload available (forced entries)
    FAIL_MARKERS = {"coherence": "x", "trait": "+", "unknown": "s"}
    FAIL_SIZES   = {"coherence": 60,  "trait": 80,  "unknown": 36}
    y_hi = max((df.delta_trait_mean.max() for _, df in per_trait_df), default=10.0)
    y_lo_data = min((df.delta_trait_mean.min() for _, df in per_trait_df), default=0.0)
    y_range = max(y_hi - min(y_lo_data, 0.0), 10.0)
    band_top = -0.04 * y_range
    band_bot = -0.18 * y_range
    n_traits = max(len(cfg.traits), 1)
    any_failed = False
    seen_kinds: set[str] = set()
    for idx, (trait, fails) in enumerate(per_trait_failed):
        if not fails:
            continue
        any_failed = True
        # Spread traits evenly across [band_bot, band_top].
        frac = (idx + 0.5) / n_traits
        y_band = band_top + frac * (band_bot - band_top)
        for _, tok, kind in fails:
            seen_kinds.add(kind)
            marker = FAIL_MARKERS.get(kind, "x")
            size = FAIL_SIZES.get(kind, 60)
            x_disp = display_map[tok]
            if kind == "unknown":
                ax.scatter([x_disp], [y_band], marker=marker, s=size,
                           facecolor="white", edgecolor=trait.color,
                           linewidth=1.4, zorder=4)
            else:
                ax.scatter([x_disp], [y_band], marker=marker, s=size,
                           color=trait.color, linewidth=1.8, zorder=4)
    if any_failed:
        ax.axhspan(band_bot * 1.05, band_top, color="#000", alpha=0.035, zorder=0)

    ax.set_xscale("log")
    if all_x:
        disp_x = sorted(display_map.values())
        # Snug right edge when no post-training columns exist (e.g. Apertus),
        # otherwise push past the last post-training tick for breathing room.
        if post_labels:
            right_edge = max(post_x.values()) * 1.25
        else:
            right_edge = base_max * 1.05
        ax.set_xlim(max(cfg.xmin_pad_B, min(disp_x) * 0.8), right_edge)
        # When no post-training exists, extend the final stage band out to
        # the plot edge so there's no unshaded strip on the right. Rewrite
        # the entry in stage_band_patches so the label centers on the
        # extended band, not the narrow interpolated one.
        if cfg.stage_regions and not post_labels and stage_band_patches:
            final_region, final_left, _ = stage_band_patches[-1]
            ax.axvspan(final_left, right_edge, color=final_region.color,
                       alpha=0.12, zorder=0)
            stage_band_patches[-1] = (final_region, final_left, right_edge)
        # Ticks at the displayed positions; labels show the TRUE token count.
        sorted_actual = sorted(display_map.keys())
        tick_positions = [display_map[a] for a in sorted_actual]
        tick_labels = [_fmt_tokens_B(a) for a in sorted_actual]
        for lbl in post_labels:
            tick_positions.append(post_x[lbl])
            tick_labels.append(lbl)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=30, ha="right")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

        # Post-training region: colored band behind the SFT/DPO/Instruct
        # columns. The pretraining bands already extend up to `post_left`
        # so there is no white space between them.
        if post_labels:
            ax.axvspan(post_left, right_edge,
                       color="#54a24b", alpha=0.12, zorder=0)
            mid_log_post = (math.log10(post_left) + math.log10(right_edge)) / 2
            ax.text(10 ** mid_log_post, 1.02,
                    "Post-training",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=8.5, color="#54a24b", weight="semibold",
                    clip_on=False, zorder=5)
    ax.set_xlabel("Checkpoint of persona extraction (tokens consumed)", fontsize=11)
    ax.set_ylabel("Trait expression delta on same checkpoint", fontsize=11)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.grid(True, which="major", axis="x", ls=":", alpha=0.20)
    if any_failed:
        # Pad ylim down so the failure band doesn't collide with tick labels.
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(min(cur_lo, band_bot * 1.25), cur_hi)

    # Legend: trait entries (colored lines) plus a thin rule and marker
    # entries for ★ significance / × incoherent / + no persona. Building
    # custom handles lets us show the marker glyphs in black so they read
    # as "this symbol means …" rather than binding to any single trait.
    from matplotlib.lines import Line2D
    trait_handles, trait_labels = ax.get_legend_handles_labels()
    marker_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="black",
               markeredgecolor="white", markersize=12, linewidth=0,
               label=f"significant (p < {p_threshold})"),
    ]
    if any_failed:
        if "coherence" in seen_kinds:
            marker_handles.append(Line2D([0], [0], marker="x", color="black",
                                         markersize=9, linewidth=0, label="incoherent"))
        if "trait" in seen_kinds:
            marker_handles.append(Line2D([0], [0], marker="+", color="black",
                                         markersize=11, linewidth=0, label="no persona"))
        if "unknown" in seen_kinds:
            marker_handles.append(Line2D([0], [0], marker="s", color="none",
                                         markerfacecolor="white", markeredgecolor="black",
                                         markersize=8, linewidth=0, label="no status"))
    ax.legend(handles=trait_handles + marker_handles,
              labels=trait_labels + [h.get_label() for h in marker_handles],
              loc="upper left", fontsize=9.5, frameon=True, framealpha=0.92,
              handlelength=1.6, labelspacing=0.4)

    # Stage-region text labels: place in a dedicated strip just above the
    # axis frame so they never collide with the legend or data. Uses axis-
    # fraction y so the position is robust to y-limit changes. Regions with
    # an empty `label` are skipped (useful for narrow bands like the OLMo-3
    # Stages 2+3 sliver where the tick labels 5.9 T / 6.0 T / 6.1 T already
    # make the stages self-evident).
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
    # PNG for preview/review + PDF for LaTeX inclusion (vector, so DPI is
    # only meaningful for the PNG).
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {pdf_path}")


# --------------------------- driver --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default="olmo3")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default: story_plots/emergence/<model_subdir>/<stem>.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MODELS[args.model]()
    subdir, stem = MODEL_OUT_LAYOUT[args.model]
    out = args.out or (REPO / f"analysis/figures/story_plots/emergence/{subdir}/{stem}.png")
    render(cfg, out)


if __name__ == "__main__":
    main()
