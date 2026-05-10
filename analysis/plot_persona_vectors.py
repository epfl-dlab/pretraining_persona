#!/usr/bin/env python3
"""
Plot persona vectors: norms, correlations, and PCA projections across traits and revisions.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

# OLMo-3 checkpoint → cumulative tokens (B). Source: arXiv:2512.13961 §3.4
_TOK_4M_B = 4.194304 / 1000.0
_TOK_2M_B = 2.097152 / 1000.0
_OLMO_STAGE1_END_B = 1_413_814 * _TOK_4M_B
_OLMO_STAGE2_END_B = _OLMO_STAGE1_END_B + 47_684 * _TOK_2M_B
_OLMO_STAGE3_END_B = _OLMO_STAGE2_END_B + 11_921 * _TOK_4M_B
_OLMO_STEP_RE = re.compile(r"^stage([123])-step(\d+)$")
_APERTUS_STEP_RE = re.compile(r"^step(\d+)-tokens(\d+(?:\.\d+)?)(B|M|T|K)$")


def _fmt_tokens_B(n_B: float) -> str:
    if n_B >= 1000:
        t = n_B / 1000.0
        return f"{t:.1f}T" if t < 10 else f"{t:.0f}T"
    if n_B >= 10:
        return f"{n_B:.0f}B"
    return f"{n_B:.1f}B"


def apertus_ckpt_to_tokens_B(rev: str) -> float | None:
    if rev == "main":
        return 15_000.0
    m = _APERTUS_STEP_RE.match(rev)
    if m is None:
        return None
    n, suffix = float(m.group(2)), m.group(3)
    return {"B": n, "M": n / 1000.0, "T": n * 1000.0, "K": n / 1_000_000.0}[suffix]


def olmo3_ckpt_to_tokens_B(rev: str) -> float | None:
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.manifold import MDS

plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.handlelength": 1.6,
    "legend.labelspacing": 0.4,
})
MY_COLORS = ["#FFB000", "#648FFF", "#DC267F", "#5BD492", "#785EF0", "#FE6100", "#57D8FF"]

heatmap_cmap = LinearSegmentedColormap.from_list(
    "heatmap_cmap",
    [MY_COLORS[0], "#FFFFFF", MY_COLORS[1]],
    N=256,
)


def find_vectors(model_dir: Path, token_name: str) -> list[dict]:
    """
    Discover all persona vector files for a model directory.

    Handles two layouts:
      - flat:   {model_dir}/{trait_method}_{token_name}.pt
      - nested: {model_dir}/{revision}/{trait_method}_{token_name}.pt
    """
    suffix = f"_{token_name}.pt"
    entries = []
    for pt_file in sorted(model_dir.rglob(f"*{suffix}")):
        revision = "base" if pt_file.parent == model_dir else pt_file.parent.name
        trait_method = pt_file.name[: -len(suffix)]
        entries.append({"path": pt_file, "revision": revision, "trait_method": trait_method})
    return entries


def load_layer(path: Path, layer: int) -> np.ndarray:
    vectors = torch.load(path, weights_only=True)  # shape: [num_layers, hidden_size]
    return vectors[layer].float().numpy()


def save_corr_matrix(corr_np: np.ndarray, labels: list[str], title: str, save_path: Path):
    n = len(labels)
    corr = pd.DataFrame(corr_np, index=labels, columns=labels)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.5), max(3, n * 0.45)))
    sns.heatmap(
        corr, mask=mask, cmap=heatmap_cmap, vmin=-1, vmax=1, center=0,
        annot=n <= 30, fmt=".2f", square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.6}, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def _revision_sort_key(revision: str) -> tuple:
    if revision == "base":
        return (0, 0)
    if revision == "main":
        return (999, 999_999_999)
    m = re.match(r"stage(\d+)-step(\d+)", revision)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = _APERTUS_STEP_RE.match(revision)
    if m2:
        return (1, int(m2.group(1)))
    return (0, 0)


def _shade(base_hex: str, t: float) -> np.ndarray:
    """Interpolate from light tint (t=0) to dark shade (t=1)."""
    rgb = np.array(mcolors.to_rgb(base_hex))
    light = rgb + (1 - rgb) * 0.82
    dark = rgb * 0.50
    return np.clip(light + t * (dark - light), 0, 1)


def _save_embedding_plot(reduced: np.ndarray, entries: list[dict], title: str,
                         save_path: Path, xlabel: str, ylabel: str,
                         figsize: tuple = (10.5, 5.5), ckpt_to_tokens_fn=None):
    def get_trait(e):
        return e["trait_method"].split("_")[0]

    traits = list(dict.fromkeys(get_trait(e) for e in entries))
    by_trait = defaultdict(list)
    for i, e in enumerate(entries):
        by_trait[get_trait(e)].append((i, e))

    filtered_traits = [t for t in traits if len(by_trait[t]) > 1]
    trait_color = {t: MY_COLORS[i] for i, t in enumerate(filtered_traits)}

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []

    for trait in filtered_traits:
        base = trait_color[trait]
        group = sorted(by_trait[trait], key=lambda x: _revision_sort_key(x[1]["revision"]))
        n = len(group)
        xs = [reduced[i, 0] for i, _ in group]
        ys = [reduced[i, 1] for i, _ in group]
        shades = [_shade(base, k / max(n - 1, 1)) for k in range(n)]
        ax.plot(xs, ys, color=mcolors.to_rgb(base), linewidth=0.8, alpha=0.5, zorder=1)
        for x, y, c in zip(xs, ys, shades):
            ax.scatter(x, y, color=c, s=60, zorder=2)
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=base, markersize=8, label=trait.capitalize())
        )

    _tok_fn = ckpt_to_tokens_fn or olmo3_ckpt_to_tokens_B
    sorted_revs = sorted({e["revision"] for e in entries}, key=_revision_sort_key)
    tok_vals = [t for r in sorted_revs if (t := _tok_fn(r)) is not None]
    early_label = f"early ({_fmt_tokens_B(tok_vals[0])})" if tok_vals else "early"
    late_label = f"late ({_fmt_tokens_B(tok_vals[-1])})" if tok_vals else "late"

    n_steps = 5
    rev_handles = [
        Patch(facecolor=_shade("#000000", k / (n_steps - 1)),
              label=early_label if k == 0 else (late_label if k == n_steps - 1 else ""),
              linewidth=0)
        for k in range(n_steps)
    ]

    leg1 = ax.legend(handles=legend_handles, title="Trait",
                     bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=rev_handles, title="Checkpoint",
                     bbox_to_anchor=(1.05, 0.0), loc="lower left")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.grid(True, which="major", axis="x", ls=":", alpha=0.20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", bbox_extra_artists=[leg1, leg2])
    plt.close(fig)
    print(f"Saved {save_path}")


def save_mds(vectors: np.ndarray, entries: list[dict], title: str, save_path: Path,
             cosine: bool = False, ckpt_to_tokens_fn=None):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms > 0, norms, 1)
    if cosine:
        from sklearn.metrics.pairwise import cosine_distances
        dist_matrix = cosine_distances(normalized)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto", n_init=10)
        reduced = mds.fit_transform(dist_matrix)
    else:
        mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42, normalized_stress="auto", n_init=10)
        reduced = mds.fit_transform(normalized)
    suffix = " (cosine)" if cosine else ""
    _save_embedding_plot(reduced, entries, f"{title}{suffix}", save_path, xlabel="MDS 1", ylabel="MDS 2",
                         figsize=(5, 3.75), ckpt_to_tokens_fn=ckpt_to_tokens_fn)


def save_norms(norms: np.ndarray, entries: list[dict], title: str, save_path: Path,
               calib_norms: dict[str, float] | None = None,
               ckpt_to_tokens_fn=None):
    def get_trait(e):
        return e["trait_method"].split("_")[0]

    by_trait = defaultdict(list)
    for norm, e in zip(norms, entries):
        rev = e["revision"]
        calib = calib_norms.get(rev) if calib_norms else None
        value = norm / calib if calib else norm
        by_trait[get_trait(e)].append((rev, value))

    traits = [t for t in by_trait if len(by_trait[t]) > 1]
    all_revisions = sorted({e["revision"] for e in entries}, key=_revision_sort_key)
    if calib_norms:
        all_revisions = [r for r in all_revisions if r in calib_norms]

    # Use tokens-seen (B) as x if the conversion is available for at least some revisions
    _tok_fn = ckpt_to_tokens_fn or olmo3_ckpt_to_tokens_B
    tok_x = {r: _tok_fn(r) for r in all_revisions}
    tok_x = {r: v for r, v in tok_x.items() if v is not None}
    use_tokens = bool(tok_x)
    use_log = use_tokens

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for i, trait in enumerate(traits):
        pairs = [(r, v) for r, v in sorted(by_trait[trait], key=lambda x: _revision_sort_key(x[0]))
                 if r in (tok_x if use_tokens else {r: True for r in all_revisions})]
        if not pairs:
            continue
        xs = [tok_x[r] for r, _ in pairs] if use_tokens else [all_revisions.index(r) for r, _ in pairs]
        ys = [v for _, v in pairs]
        ax.plot(xs, ys, marker="o", color=MY_COLORS[i % len(MY_COLORS)], label=trait.capitalize())

    if use_tokens:
        if use_log:
            ax.set_xscale("log")
        tick_xs = sorted(set(tok_x[r] for r in all_revisions if r in tok_x))
        tick_lbls = [_fmt_tokens_B(x) for x in tick_xs]
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_lbls, fontsize=9, rotation=30, ha="right")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlabel("Training tokens consumed")
    else:
        ax.set_xticks(range(len(all_revisions)))
        ax.set_xticklabels(all_revisions, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("norm / calibration norm" if calib_norms else "L2 norm")
    ax.set_title(title)
    ax.legend(title="Trait")
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.grid(True, which="major", axis="x", ls=":", alpha=0.20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def _trait_cosine_data(vectors: np.ndarray, entries: list[dict], trait: str,
                       ckpt_to_tokens_fn=None) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Return (cos_sim, mask, tick_labels) for a trait, or None if no data."""
    _tok_fn = ckpt_to_tokens_fn or olmo3_ckpt_to_tokens_B
    revs_with_syco = {
        e["revision"] for e in entries
        if "deepseek" not in e["revision"] and e["trait_method"] == "sycophantic_character_neutral_q"
    }
    pairs = [
        (v, e) for v, e in zip(vectors, entries)
        if e["trait_method"] == f"{trait}_character_neutral_q"
        and "deepseek" not in e["revision"]
        and e["revision"] in revs_with_syco
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda x: (_tok_fn(x[1]["revision"]) or 0, _revision_sort_key(x[1]["revision"])))
    vecs = np.stack([v for v, _ in pairs])
    revs = [e["revision"] for _, e in pairs]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / np.where(norms > 0, norms, 1)
    cos_sim = normalized @ normalized.T
    n = len(revs)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    short_revs = [_fmt_tokens_B(t) if (t := _tok_fn(r)) is not None else r for r in revs]
    return cos_sim, mask, short_revs


def save_trait_checkpoint_cosine(vectors: np.ndarray, entries: list[dict], trait: str,
                                 title: str, save_path: Path, ckpt_to_tokens_fn=None):
    """Lower-diagonal cosine-similarity heatmap across checkpoints for a single trait."""
    result = _trait_cosine_data(vectors, entries, trait, ckpt_to_tokens_fn)
    if result is None:
        print(f"No {trait} vectors found; skipping checkpoint cosine plot.")
        return
    cos_sim, mask, short_revs = result
    n = len(short_revs)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.65), max(3.5, n * 0.6)))
    sns.heatmap(
        cos_sim, mask=mask, cmap=heatmap_cmap, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", square=True, linewidths=0.5,
        xticklabels=short_revs, yticklabels=short_revs,
        cbar_kws={"shrink": 0.6, "label": "cosine similarity"}, ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def save_all_traits_checkpoint_cosine(vectors: np.ndarray, entries: list[dict],
                                      save_path: Path, ckpt_to_tokens_fn=None):
    """2x2 grid of lower-diagonal cosine-similarity heatmaps, one per trait, shared color scale."""
    traits = ("evil", "humorous", "impolite", "sycophantic")
    data = {t: _trait_cosine_data(vectors, entries, t, ckpt_to_tokens_fn) for t in traits}

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    all_vals = np.concatenate([
        r[0][~r[1]] for r in data.values() if r is not None
    ])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for idx, (ax, trait) in enumerate(zip(axes.flat, traits)):
        result = data[trait]
        if result is None:
            ax.set_visible(False)
            continue
        cos_sim, mask, _ = result
        row, col = divmod(idx, 2)
        sns.heatmap(
            cos_sim, mask=mask, cmap=heatmap_cmap, norm=norm,
            annot=False, square=False, linewidths=0.3,
            xticklabels=False, yticklabels=False,
            cbar=False, ax=ax,
        )
        ax.set_title(trait.capitalize(), fontsize=28, pad=2, y=1.0)
        ax.tick_params(axis="both", length=0)
        if row == 1:
            ax.set_xlabel("Checkpoints (from early to late)", fontsize=20)
        if col == 0:
            ax.set_ylabel("Checkpoints", fontsize=20)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.subplots_adjust(wspace=0.02)

    # shared vertical colorbar on the right
    sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical", label="cosine similarity")
    cb.ax.tick_params(labelsize=28)
    cb.set_label("Cosine similarity", fontsize=28)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot persona vectors across traits and revisions")
    parser.add_argument("--model", default="Olmo-3-1025-7B", help="Model directory name under data/persona_vectors/")
    parser.add_argument("--layer", type=int, default=16, help="Layer index to extract")
    parser.add_argument("--token_name", default="response_avg_diff", help="Token position type")
    parser.add_argument("--data_dir", default="data/persona_vectors", help="Root data directory")
    parser.add_argument("--results_dir", default="results", help="Root results directory for calibration norms")
    parser.add_argument("--save_dir", default="analysis/figures/geometry_checkpoints", help="Directory to save figures")
    parser.add_argument("--trait_methods", default=None,
                        help="Comma-separated list of trait_method names to include (default: all)")
    args = parser.parse_args()

    ckpt_to_tokens_fn = apertus_ckpt_to_tokens_B if "apertus" in args.model.lower() else olmo3_ckpt_to_tokens_B

    model_dir = Path(args.data_dir) / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    calib_dir = Path(args.results_dir) / args.model / "activation_norms"
    calib_norms = {}
    if calib_dir.exists():
        for csv_file in calib_dir.glob("*_norms.csv"):
            stem = csv_file.stem
            revision = stem.removesuffix("_shared_norms").removesuffix("_norms")
            if revision == stem:
                continue  # no recognized suffix stripped
            if "_" in revision and not _APERTUS_STEP_RE.match(revision):
                continue  # skip trait-specific norm files
            df_calib = pd.read_csv(csv_file)
            row = df_calib[df_calib["layer"] == args.layer]
            if not row.empty:
                calib_norms[revision] = row["mean_l2"].iloc[0]
        print(f"Loaded calibration norms for {len(calib_norms)} revisions")

    entries = find_vectors(model_dir, args.token_name)
    if args.trait_methods:
        allowed = set(args.trait_methods.split(","))
        entries = [e for e in entries if e["trait_method"] in allowed]
    if not entries:
        raise ValueError(f"No vectors found for token_name='{args.token_name}' in {model_dir}")

    print(f"Model:      {args.model}")
    print(f"Layer:      {args.layer}")
    print(f"Token type: {args.token_name}")
    print(f"Found {len(entries)} persona vector files:")
    for e in entries:
        print(f"  revision={e['revision']:<30}  trait_method={e['trait_method']}")

    vectors, labels = [], []
    for e in entries:
        vectors.append(load_layer(e["path"], args.layer))
        labels.append(f"{e['revision']}/{e['trait_method']}")

    vectors = np.stack(vectors)
    print(f"\nLoaded {len(vectors)} vectors of dimension {vectors.shape[1]}")

    short_labels = [f"{e['revision'][:8]}/{e['trait_method'][:12]}" for e in entries]
    file_prefix = f"{args.model}_layer{args.layer}"
    norms = np.linalg.norm(vectors, axis=1)

    save_corr_matrix(np.corrcoef(vectors), short_labels,
                     f"Correlation — {args.token_name}", save_dir / f"{file_prefix}_corr_matrix.pdf")
    for trait in ("evil", "humorous", "impolite", "sycophantic"):
        save_trait_checkpoint_cosine(vectors, entries, trait,
                                     f"{trait.capitalize()} — cosine similarity across checkpoints",
                                     save_dir / f"{file_prefix}_{trait}_checkpoint_cosine.pdf",
                                     ckpt_to_tokens_fn=ckpt_to_tokens_fn)
    save_all_traits_checkpoint_cosine(vectors, entries,
                                      save_dir / f"{file_prefix}_all_traits_checkpoint_cosine.pdf",
                                      ckpt_to_tokens_fn=ckpt_to_tokens_fn)
    core_traits = {"evil_character_neutral_q", "humorous_character_neutral_q",
                   "impolite_character_neutral_q", "sycophantic_character_neutral_q"}
    revs_per_core_trait = defaultdict(set)
    for e in entries:
        if "deepseek" not in e["revision"] and e["trait_method"] in core_traits:
            revs_per_core_trait[e["trait_method"]].add(e["revision"])
    revs_with_all_traits = (
        set.intersection(*revs_per_core_trait.values())
        if revs_per_core_trait else set()
    )
    filtered_mask = [
        "deepseek" not in e["revision"]
        and e["revision"] in revs_with_all_traits
        and e["trait_method"] in core_traits
        for e in entries
    ]
    filtered_entries = [e for e, keep in zip(entries, filtered_mask) if keep]
    filtered_vectors = vectors[np.array(filtered_mask)]

    save_mds(filtered_vectors, filtered_entries,
             "", save_dir / f"{file_prefix}_mds.pdf",
             ckpt_to_tokens_fn=ckpt_to_tokens_fn)
    save_norms(norms, entries,
               "", save_dir / f"{file_prefix}_norms.pdf",
               calib_norms=calib_norms or None, ckpt_to_tokens_fn=ckpt_to_tokens_fn)

    print("\nVector norms:")
    for label, norm in zip(labels, norms):
        print(f"  {label:<60}  norm={norm:.4f}")
    print(f"\nMean norm: {norms.mean():.4f}  |  Std: {norms.std():.4f}")


if __name__ == "__main__":
    main()
