"""Transfer-plot supplementary LaTeX tables.

Emits one longtable per (model, target) — same axes as the transfer plots in
`analysis/figures/story_plots/transfer/`. Rows are grouped by trait; each row
is one pretraining extract checkpoint with its Δ and raw p-value applied to
the fixed post-training target.

Usage:
  python3 analysis/make_transfer_table.py --model olmo3   --target all
  python3 analysis/make_transfer_table.py --model apertus --target instruct
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from make_transfer_plot import (
    MODEL_TARGETS,
    build_config,
    load_transfer,
)

REPO = Path(__file__).resolve().parents[1]


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


def render_latex(cfg, model: str, target: str) -> str:
    target_pretty = cfg.name.split("→", 1)[-1].strip()
    caption = (
        f"Per-checkpoint transfer results backing Figure ``Persona transfer "
        f"to {_escape(target_pretty)}''. Each row is the judge-scored trait "
        r"expression shift $\Delta$ produced when the persona vector "
        r"extracted from the base-model checkpoint is applied to the fixed "
        f"target {_escape(target_pretty)}. $p$ is the raw (non-Bonferroni) "
        r"two-sided primary test; $^{\star}$ marks $p<0.05$ and corresponds "
        r"to starred markers in the figure."
    )
    label = (
        f"tab:transfer_{model}_{target}".replace("-", "_")
    )
    lines: list[str] = []
    lines.append(r"\begin{longtable}{l r r r r r}")
    lines.append(r"\caption{" + caption + r"}\label{" + label + r"}\\")
    lines.append(r"\toprule")
    lines.append(r"Checkpoint & Tokens & Layer & Coef & $\Delta$ & $p$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Checkpoint & Tokens & Layer & Coef & $\Delta$ & $p$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    first = True
    for trait in cfg.traits:
        df = load_transfer(trait, cfg.eval_model, cfg.ckpt_to_tokens, cfg.ckpt_grid)
        if df.empty:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(
            r"\multicolumn{6}{l}{\textit{Trait: " + _escape(trait.label) + r"}} \\"
        )
        lines.append(r"\midrule")
        for _, r in df.iterrows():
            ckpt = _escape(str(r.extract_revision))
            tokens = _fmt_tokens(float(r.tokens_B))
            layer = str(trait.layer)
            coef = f"{float(r.coef_used):.2f}"
            delta = _fmt_delta(float(r.delta_trait_mean)) + _sig_mark(r.p_raw)
            p = _fmt_p(r.p_raw)
            lines.append(" & ".join([ckpt, tokens, layer, coef, delta, p]) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo3", choices=list(MODEL_TARGETS.keys()))
    ap.add_argument("--target", default="all",
                    help="target name or 'all'; see per-model choices in MODEL_TARGETS")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    valid = MODEL_TARGETS[args.model]
    targets = list(valid) if args.target == "all" else [args.target]
    for tgt in targets:
        if tgt not in valid:
            raise SystemExit(f"--target={tgt!r} not valid for model={args.model!r}; "
                             f"pick from {list(valid)}")

    out_dir = (Path(args.out_dir) if args.out_dir
               else REPO / "analysis" / "figures" / "story_plots" / "transfer" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in targets:
        cfg = build_config(args.model, tgt)
        tex = render_latex(cfg, args.model, tgt)
        out_path = out_dir / f"{args.model}_transfer_to_{tgt}_table.tex"
        out_path.write_text(tex)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
