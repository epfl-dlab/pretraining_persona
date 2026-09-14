#!/usr/bin/env python3
"""List the extraction rows that survive the judge filter used to build a persona vector.

`source.generate_vec` builds a vector from the positive / negative extraction CSVs after
dropping every row whose judge scores fail the filter (evaluated row-wise on the aligned
pos / neg files, so a row survives only if BOTH its positive and negative response pass):

    pos[trait] >= threshold  and  neg[trait] < 100 - threshold
    and pos.coherence >= 50  and  neg.coherence >= 50

This script applies exactly that filter (by calling the same function) and prints the
retained `question_id`s, so the examples behind any shipped vector can be identified.

Usage:
    python analysis/list_retained_extraction_rows.py \
        --pos_path data/model_responses/extract/Apertus-8B-Instruct-2509/main/evil_character_neutral_q_pos_instruct.csv \
        --neg_path data/model_responses/extract/Apertus-8B-Instruct-2509/main/evil_character_neutral_q_neg_instruct.csv \
        --trait evil_character_neutral_q [--threshold 50] [--out retained.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.generate_vec import get_persona_effective  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pos_path", required=True)
    parser.add_argument("--neg_path", required=True)
    parser.add_argument("--trait", required=True, help="trait column name, e.g. evil or evil_character_neutral_q")
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--max_examples", type=int, default=None, help="cap used at vector-build time, if any")
    parser.add_argument("--out", default=None, help="optional CSV with the retained rows (pos and neg side by side)")
    args = parser.parse_args()

    pos_eff, neg_eff, *_ = get_persona_effective(
        args.pos_path, args.neg_path, args.trait, threshold=args.threshold, max_examples=args.max_examples
    )
    total = len(pd.read_csv(args.pos_path))
    print(f"retained {len(pos_eff)} / {total} aligned pos/neg rows (threshold={args.threshold})")

    # `question_id` repeats once per sample (n_per_question), so the 0-based row index in the
    # aligned pos/neg CSVs is the unique key of a retained example.
    merged = pd.DataFrame(
        {
            "row": pos_eff.index.to_numpy(),
            "question_id": pos_eff["question_id"].to_numpy(),
            "question": pos_eff["question"].to_numpy(),
            f"pos_{args.trait}": pos_eff[args.trait].to_numpy(),
            "pos_coherence": pos_eff["coherence"].to_numpy(),
            "pos_answer": pos_eff["answer"].to_numpy(),
            f"neg_{args.trait}": neg_eff[args.trait].to_numpy(),
            "neg_coherence": neg_eff["coherence"].to_numpy(),
            "neg_answer": neg_eff["answer"].to_numpy(),
        }
    )
    if args.out:
        merged.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        for row, qid in zip(merged["row"], merged["question_id"]):
            print(f"{row}\t{qid}")


if __name__ == "__main__":
    main()
