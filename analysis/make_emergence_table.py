"""Emergence-plot supplementary LaTeX table.

Emits a longtable with every data point drawn in
`analysis/figures/story_plots/emergence/olmo3_same_checkpoint_emergence.pdf`:
per-trait Δ_trait and p at each pretraining checkpoint (plus SFT / DPO /
Instruct self-steering anchors on the right), and one row per checkpoint
where the persona vector could not be extracted.

Output: analysis/figures/story_plots/emergence/olmo3_same_checkpoint_emergence_table.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from make_emergence_plot import (
    MODELS,
    MODEL_OUT_LAYOUT,
    TraitConfig,
    load_diagonal,
    load_failed_revisions,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------- formatting helpers --------------------------

def _fmt_delta(d: float) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "--"
    return f"{d:+.2f}"


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 1e-3:
        return r"$<\!.001$"
    s = f"{p:.3f}"
    return s.lstrip("0") if s.startswith("0.") else s


def _sig_mark(p: float, threshold: float = 0.05) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return r"$^{\star}$" if p < threshold else ""


def _fmt_tokens(tok_B: float) -> str:
    if tok_B >= 1000:
        t = tok_B / 1000.0
        return f"{t:.2f}T" if t < 10 else f"{t:.1f}T"
    if tok_B >= 100:
        return f"{tok_B:.0f}B"
    if tok_B >= 10:
        return f"{tok_B:.1f}B"
    return f"{tok_B:.2f}B"


def _escape(s: str) -> str:
    return s.replace("_", r"\_")


FAILURE_LABEL = {
    "coherence": "incoherent",
    "trait":     "no persona",
    "unknown":   "no vector",
}


# --------------------------- row assembly --------------------------------

def _trait_rows(trait: TraitConfig, cfg) -> list[dict]:
    df = load_diagonal(trait, cfg.ckpt_to_tokens, cfg.excluded_revisions)
    fails = load_failed_revisions(trait, cfg.path_name,
                                  cfg.forced_failed_revisions,
                                  cfg.excluded_revisions,
                                  cfg.ckpt_to_tokens)
    # Load CSV once to pull layer info per checkpoint (supports the
    # mixed-coef Apertus evil case where extra_segments live at a different
    # layer than the primary sweep).
    layer_by_rev: dict[str, int] = {}
    for csv_path, _coef in [(trait.csv_path, trait.coef), *trait.extra_segments]:
        if not csv_path.exists():
            continue
        try:
            raw = pd.read_csv(csv_path, usecols=["extract_revision", "eval_revision", "layer"])
        except ValueError:
            continue
        raw = raw[raw.extract_revision == raw.eval_revision]
        for _, r in raw.iterrows():
            rev = r.extract_revision
            if rev not in layer_by_rev:
                layer_by_rev[rev] = int(r.layer)

    rows: list[dict] = []
    # Collect failed-only ckpts (the plot keeps a ckpt as data if any trait
    # extracted it — but for this trait's table we only want actual failures
    # where no Δ is in df).
    data_revs = set(df.extract_revision.tolist())
    for rev, tok_B, kind in fails:
        if rev in data_revs:
            continue
        rows.append({
            "checkpoint": rev,
            "tokens_B": tok_B,
            "layer":    layer_by_rev.get(rev, ""),
            "coef":     "",
            "delta":    np.nan,
            "p":        np.nan,
            "note":     FAILURE_LABEL.get(kind, "no vector"),
        })

    # Data rows (one per plotted ckpt).
    for _, r in df.iterrows():
        rows.append({
            "checkpoint": r.extract_revision,
            "tokens_B":   float(r.tokens_B),
            "layer":      layer_by_rev.get(r.extract_revision, int(trait.coef) and ""),
            "coef":       float(r.coef_used),
            "delta":      float(r.delta_trait_mean),
            "p":          float(r.p_raw) if not pd.isna(r.p_raw) else np.nan,
            "note":       "",
        })

    rows.sort(key=lambda d: d["tokens_B"])

    # Post-training points appended in configured order. Read layer/coef
    # directly from the instruct CSV row rather than falling back to the
    # diagonal's operating point (evil uses c=0.55 here, not c=0.5).
    if trait.instruct_csv and trait.instruct_csv.exists() and trait.instruct_eval_models:
        idf = pd.read_csv(trait.instruct_csv)
        for eval_model, tick_label in trait.instruct_eval_models:
            sub = idf[idf.eval_model == eval_model]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append({
                "checkpoint": tick_label,
                "tokens_B":   None,
                "layer":      int(row["layer"]) if "layer" in sub.columns else "",
                "coef":       float(row["coef"]) if "coef" in sub.columns else "",
                "delta":      float(row["delta_trait_mean"]),
                "p":          float(row.get("trait_primary_p_two_sided", float("nan"))),
                "note":       "",
            })
    return rows


# --------------------------- LaTeX rendering -----------------------------

def render_latex(cfg) -> str:
    caption = (
        f"Per-checkpoint steering results backing Figure "
        f"``Emergence of persona during pretraining'' ({_escape(cfg.name)}). "
        r"Columns: $\Delta$ is the judge-scored trait expression shift vs.\ "
        r"the unsteered baseline at the same checkpoint; $p$ is the raw "
        r"(non-Bonferroni) two-sided primary test. $^{\star}$ marks $p<0.05$ "
        r"and corresponds to starred markers in the figure. Rows without "
        r"$\Delta$ are checkpoints where no persona vector could be "
        r"extracted (\emph{incoherent}: pos-set answers fail the coherence "
        r"filter; \emph{no persona}: coherent pos-set answers fail the "
        r"trait filter; \emph{no vector}: extraction never attempted)."
    )
    label = f"tab:emergence_{cfg.path_name.lower().replace('-', '_')}"

    lines: list[str] = []
    lines.append(r"\begin{longtable}{l r r r r r l}")
    lines.append(r"\caption{" + caption + r"}\label{" + label + r"}\\")
    lines.append(r"\toprule")
    lines.append(
        r"Checkpoint & Tokens & Layer & Coef & $\Delta$ & $p$ & Note \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"Checkpoint & Tokens & Layer & Coef & $\Delta$ & $p$ & Note \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    first = True
    for trait in cfg.traits:
        rows = _trait_rows(trait, cfg)
        if not rows:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(
            r"\multicolumn{7}{l}{\textit{Trait: " + _escape(trait.label) + r"}} \\"
        )
        lines.append(r"\midrule")
        for r in rows:
            ckpt = _escape(str(r["checkpoint"]))
            tokens = _fmt_tokens(r["tokens_B"]) if r["tokens_B"] is not None else "--"
            layer = str(r["layer"]) if r["layer"] != "" else "--"
            coef = f"{r['coef']:.2f}" if isinstance(r["coef"], float) else "--"
            delta = _fmt_delta(r["delta"])
            if not (isinstance(r["delta"], float) and np.isnan(r["delta"])):
                delta = delta + _sig_mark(r["p"])
            p = _fmt_p(r["p"])
            note = r["note"]
            lines.append(
                " & ".join([ckpt, tokens, layer, coef, delta, p, note]) + r" \\"
            )

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo3", choices=list(MODELS.keys()))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = MODELS[args.model]()
    subdir, stem = MODEL_OUT_LAYOUT[args.model]
    out_dir = REPO / "analysis" / "figures" / "story_plots" / "emergence" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (Path(args.out) if args.out
           else out_dir / f"{stem}_table.tex")

    out.write_text(render_latex(cfg))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
