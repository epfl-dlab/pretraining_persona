#!/usr/bin/env python3
"""CLI annotation tool for Baumeister roots in checkpoint CSV files.

Usage:
    python tools/annotate_roots.py <csv_file> [<csv_file> ...] [--n 20] [--seed 42]

Pass multiple files to sample evenly across all checkpoints. Glob patterns work:
    python tools/annotate_roots.py data/model_responses/extract/Olmo-3-1025-7B/main/*.csv

For each example, enter space-separated numbers for applicable roots:
    1 = sadism
    2 = instrumentality
    3 = idealism
    4 = threatened_egotism

Enter nothing to skip. Enter 'q' to quit early and see partial results.
"""

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOTS = {
    1: "sadism",
    2: "instrumentality",
    3: "idealism",
    4: "threatened_egotism",
}
ROOT_NAMES = list(ROOTS.values())


def checkpoint_sort_key(path):
    m = re.search(r"step(\d+)", str(path))
    return int(m.group(1)) if m else 0


def parse_predicted(roots_str):
    if pd.isna(roots_str) or not roots_str.strip():
        return set()
    return {r.strip() for r in roots_str.split("|") if r.strip() in ROOT_NAMES}


def parse_annotation(raw):
    result = set()
    for token in raw.strip().split():
        try:
            n = int(token)
            if n in ROOTS:
                result.add(ROOTS[n])
            else:
                print(f"  [ignored] {n} is not a valid number (use 1-4)")
        except ValueError:
            print(f"  [ignored] '{token}' is not a number")
    return result


def compute_metrics(annotations, predictions):
    stats = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ROOT_NAMES}
    for human, model in zip(annotations, predictions):
        for root in ROOT_NAMES:
            h = root in human
            m = root in model
            if h and m:
                stats[root]["tp"] += 1
            elif m and not h:
                stats[root]["fp"] += 1
            elif h and not m:
                stats[root]["fn"] += 1
    return stats


def generate_latex_table(stats):
    rows = []
    sum_prec = sum_rec = sum_f1 = 0.0
    count = 0
    for root in ROOT_NAMES:
        tp = stats[root]["tp"]
        fp = stats[root]["fp"]
        fn = stats[root]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        label = root.replace("_", r"\_")
        p_s = f"{prec:.2f}" if prec == prec else "—"
        r_s = f"{rec:.2f}"  if rec  == rec  else "—"
        f_s = f"{f1:.2f}"   if f1   == f1   else "—"
        rows.append(f"    {label} & {p_s} & {r_s} & {f_s} & {tp} & {fp} & {fn} \\\\")
        if prec == prec: sum_prec += prec; count += 1
        if rec  == rec:  sum_rec  += rec
        if f1   == f1:   sum_f1   += f1
    n = max(count, 1)
    avg_p = f"{sum_prec/n:.2f}"
    avg_r = f"{sum_rec/n:.2f}"
    avg_f = f"{sum_f1/n:.2f}"
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Root & Prec & Rec & F1 & TP & FP & FN \\",
        r"    \midrule",
    ] + rows + [
        r"    \midrule",
        f"    Average & {avg_p} & {avg_r} & {avg_f} & & & \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Baumeister roots annotation metrics (human vs.\ model)}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_metrics(stats):
    print("\n" + "=" * 60)
    print(f"{'Root':<22} {'Prec':>6} {'Rec':>6} {'F1':>6}  (TP/FP/FN)")
    print("-" * 60)
    for root in ROOT_NAMES:
        tp = stats[root]["tp"]
        fp = stats[root]["fp"]
        fn = stats[root]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        prec_s = f"{prec:.2f}" if prec == prec else "  —  "
        rec_s  = f"{rec:.2f}"  if rec  == rec  else "  —  "
        f1_s   = f"{f1:.2f}"   if f1   == f1   else "  —  "
        print(f"{root:<22} {prec_s:>6} {rec_s:>6} {f1_s:>6}  ({tp}/{fp}/{fn})")
    print("=" * 60)
    print("Precision = fraction of model's predicted roots you agreed with")
    print("Recall    = fraction of your annotated roots the model also predicted")


def load_and_sample(csv_files, n, seed):
    """Load all files, tag each row with its source, sample n evenly across files."""
    required = {"question", "answer", "baumeister_roots"}
    frames = []
    for path in sorted(csv_files, key=checkpoint_sort_key):
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            print(f"  [skip] {path}: missing columns {missing}", file=sys.stderr)
            continue
        df["_source"] = path
        frames.append(df)

    if not frames:
        sys.exit("No valid CSV files found.")

    k = len(frames)
    # Distribute n as evenly as possible across files
    base, remainder = divmod(n, k)
    rng = pd.core.common.random_state(seed)
    parts = []
    for i, df in enumerate(frames):
        per_file = base + (1 if i < remainder else 0)
        per_file = min(per_file, len(df))
        parts.append(df.sample(n=per_file, random_state=rng))

    combined = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_files", nargs="+", help="One or more checkpoint CSV files")
    parser.add_argument("--n", type=int, default=20, help="Total examples to annotate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", default=None, help="Output JSON file (default: annotations_<timestamp>.json)")
    args = parser.parse_args()

    sample = load_and_sample(args.csv_files, args.n, args.seed)

    out_path = Path(args.out) if args.out else Path(f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Resume from existing file if present
    human_annotations = []
    model_predictions = []
    records = []
    already_annotated = set()

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        for rec in existing.get("annotations", []):
            records.append(rec)
            human_annotations.append(set(rec["human_roots"]))
            model_predictions.append(set(rec["model_roots"]))
            already_annotated.add(rec["question"])
        print(f"\nResuming from {out_path}: {len(records)} already annotated.")

    remaining = sample[~sample["question"].isin(already_annotated)].reset_index(drop=True)
    total = len(remaining)

    if total == 0:
        print("All examples already annotated.")
        stats = compute_metrics(human_annotations, model_predictions)
        print_metrics(stats)
        latex = generate_latex_table(stats)
        print("\nLaTeX table:\n")
        print(latex)
        tex_path = out_path.with_suffix(".tex")
        tex_path.write_text(latex + "\n")
        print(f"\nLaTeX saved to: {tex_path}")
        return

    print(f"\nAnnotating {total} remaining examples across {len(args.csv_files)} checkpoint file(s)")
    print("Enter space-separated numbers, blank for none, 'q' to quit early.\n")

    for idx, (_, row) in enumerate(remaining.iterrows()):
        source_tag = re.search(r"step[\w-]+", str(row["_source"]))
        step_label = source_tag.group(0) if source_tag else row["_source"]

        done_so_far = len(records)
        grand_total = done_so_far + total
        print(f"\n{'─' * 60}")
        print(f"[{done_so_far + idx+1}/{grand_total}]  {step_label}")
        print(f"Q: {row['question']}")
        print()
        answer_text = str(row["answer"]).strip()
        for line in textwrap.wrap(answer_text, width=78):
            print(f"   {line}")

        predicted = parse_predicted(row["baumeister_roots"])

        raw = input("  [1=sadism 2=instrumentality 3=idealism 4=threatened_egotism] ").strip()
        if raw.lower() == "q":
            print("\nStopping early.")
            break
        annotation = parse_annotation(raw)
        human_annotations.append(annotation)
        model_predictions.append(predicted)
        records.append({
            "step": step_label,
            "question": row["question"],
            "answer": str(row["answer"]).strip(),
            "model_roots": sorted(predicted),
            "human_roots": sorted(annotation),
        })

        output = {
            "created": datetime.now().isoformat(),
            "seed": args.seed,
            "n_annotated": len(records),
            "annotations": records,
        }
        out_path.write_text(json.dumps(output, indent=2))

    if not human_annotations:
        print("No annotations collected.")
        return

    stats = compute_metrics(human_annotations, model_predictions)
    print(f"\nCollected {len(human_annotations)} annotations.")
    print_metrics(stats)
    latex = generate_latex_table(stats)
    print("\nLaTeX table:\n")
    print(latex)
    tex_path = out_path.with_suffix(".tex")
    tex_path.write_text(latex + "\n")
    print(f"\nLaTeX saved to: {tex_path}")

    metrics_out = {}
    for root in ROOT_NAMES:
        tp, fp, fn = stats[root]["tp"], stats[root]["fp"], stats[root]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec  = tp / (tp + fn) if (tp + fn) > 0 else None
        f1   = 2 * prec * rec / (prec + rec) if prec and rec else None
        metrics_out[root] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    output = {
        "created": datetime.now().isoformat(),
        "seed": args.seed,
        "n_annotated": len(records),
        "metrics": metrics_out,
        "annotations": records,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
