"""
Build negative-control vectors for the persona-vector study.

Two control families:

1. random_direction
   Sample a Gaussian direction of the same shape as the reference persona
   vector. Before steering, eval_persona normalises by vector_norm and scales
   to target_activation_norm, so only the *direction* of the control matters.
   Expected null: no persona effect, coherence may still drop slightly under
   large off-manifold perturbations.

2. label_shuffled
   Re-run the extraction forward pass on the same filtered pos/neg pairs used
   to build the real persona vector, then per-sample flip the pos/neg label
   with probability 0.5 (seeded). The resulting mean-difference vector is the
   same kind of object as the real persona vector, extracted from exactly the
   same data, under a null label assignment. Expected null: the direction has
   the same scale but no systematic persona structure.

Both controls are saved as standard .pt files shaped (num_layers + 1, hidden_dim),
so they plug into the existing steering infra with --vector_path.

Output layout:
  data/persona_vectors/<MODEL>-controls/<revision>/<trait>__<mode>__seed<K>_response_avg_diff.pt

A sibling .status.json records the mode, seed, and source pos/neg rows so the
collector/plotter scripts can filter them out of the main significance tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.generate_vec import (
    PersonaVectorBuildError,
    get_hidden_p_and_r,
    get_persona_effective,
)
from source.model_utils import load_model


def _write_status(status_path: Path, payload: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_random_direction_vectors(
    reference_vector_path: str | Path,
    save_dir: str | Path,
    trait: str,
    seeds: Sequence[int],
    overwrite: bool = False,
) -> List[Path]:
    reference = torch.load(reference_vector_path, weights_only=False).float()
    if reference.ndim != 2:
        raise ValueError(
            f"Reference vector at {reference_vector_path} must be 2D (layers, hidden_dim); got shape {tuple(reference.shape)}"
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for seed in seeds:
        out_path = save_dir / f"{trait}__random__seed{seed}_response_avg_diff.pt"
        status_path = Path(str(out_path) + ".status.json")
        if out_path.exists() and not overwrite:
            print(f"[random] seed={seed} already exists, skipping: {out_path}")
            written.append(out_path)
            continue
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        rand_vec = torch.randn(reference.shape, generator=gen, dtype=torch.float32)
        torch.save(rand_vec, out_path)
        _write_status(
            status_path,
            {
                "status": "complete",
                "mode": "random_direction",
                "seed": int(seed),
                "trait": trait,
                "reference_vector": str(reference_vector_path),
                "shape": list(rand_vec.shape),
            },
        )
        print(f"[random] wrote {out_path}  shape={tuple(rand_vec.shape)}")
        written.append(out_path)
    return written


def _precompute_response_diffs(
    model_name: str,
    pos_path: str | Path,
    neg_path: str | Path,
    trait: str,
    threshold: int,
    max_examples: int | None,
    revision: str | None,
    hidden_batch_size: int,
) -> torch.Tensor:
    """Return tensor of shape (num_layers+1, N, hidden_dim): per-sample pos-neg diff at each layer."""
    (
        _,
        _,
        pos_prompts,
        neg_prompts,
        pos_responses,
        neg_responses,
    ) = get_persona_effective(
        pos_path, neg_path, trait, threshold=threshold, max_examples=max_examples
    )
    if len(pos_prompts) == 0:
        raise PersonaVectorBuildError(
            "Zero effective rows after filtering.",
            reason_code="zero_effective_rows",
            details={"pos_path": str(pos_path), "neg_path": str(neg_path)},
        )
    model, tokenizer = load_model(model_name, revision=revision)
    _, _, pos_response_avg = get_hidden_p_and_r(
        model, tokenizer, pos_prompts, pos_responses, batch_size=hidden_batch_size
    )
    _, _, neg_response_avg = get_hidden_p_and_r(
        model, tokenizer, neg_prompts, neg_responses, batch_size=hidden_batch_size
    )
    diffs = []
    for layer in range(len(pos_response_avg)):
        diffs.append(pos_response_avg[layer].float() - neg_response_avg[layer].float())
    return torch.stack(diffs, dim=0)  # (L+1, N, d)


def build_label_shuffled_vectors(
    model_name: str,
    pos_path: str | Path,
    neg_path: str | Path,
    trait: str,
    save_dir: str | Path,
    seeds: Sequence[int],
    threshold: int = 50,
    max_examples: int | None = None,
    revision: str | None = None,
    hidden_batch_size: int = 8,
    overwrite: bool = False,
) -> List[Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # If all outputs already exist, skip the forward pass entirely.
    expected_paths = [
        save_dir / f"{trait}__shuffled__seed{seed}_response_avg_diff.pt"
        for seed in seeds
    ]
    if all(p.exists() for p in expected_paths) and not overwrite:
        print(f"[shuffled] all outputs already exist for {trait}, skipping")
        return expected_paths

    diffs = _precompute_response_diffs(
        model_name=model_name,
        pos_path=pos_path,
        neg_path=neg_path,
        trait=trait,
        threshold=threshold,
        max_examples=max_examples,
        revision=revision,
        hidden_batch_size=hidden_batch_size,
    )
    num_layers_plus_one, N, d = diffs.shape
    print(f"[shuffled] cached diffs shape=({num_layers_plus_one}, {N}, {d})")

    written: List[Path] = []
    for seed in seeds:
        out_path = save_dir / f"{trait}__shuffled__seed{seed}_response_avg_diff.pt"
        status_path = Path(str(out_path) + ".status.json")
        if out_path.exists() and not overwrite:
            print(f"[shuffled] seed={seed} already exists, skipping: {out_path}")
            written.append(out_path)
            continue
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        signs = (
            torch.randint(0, 2, (N,), generator=gen, dtype=torch.int32).float() * 2 - 1
        )
        # per-layer mean of s_i * diff_i
        signed = diffs * signs.view(1, N, 1)
        vec = signed.mean(dim=1)  # (L+1, d)
        torch.save(vec, out_path)
        _write_status(
            status_path,
            {
                "status": "complete",
                "mode": "label_shuffled",
                "seed": int(seed),
                "trait": trait,
                "pos_path": str(pos_path),
                "neg_path": str(neg_path),
                "effective_rows": int(N),
                "shape": list(vec.shape),
                "revision": revision,
            },
        )
        print(f"[shuffled] wrote {out_path}  shape={tuple(vec.shape)}  N={N}")
        written.append(out_path)
    return written


def _parse_seed_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    p_rand = sub.add_parser("random", help="Build random-direction control vectors.")
    p_rand.add_argument("--reference_vector", required=True)
    p_rand.add_argument("--save_dir", required=True)
    p_rand.add_argument("--trait", required=True)
    p_rand.add_argument("--seeds", default="0,1,2")
    p_rand.add_argument("--overwrite", action="store_true")

    p_shuf = sub.add_parser(
        "shuffled", help="Build label-shuffled control vectors from the same filtered rows."
    )
    p_shuf.add_argument("--model_name", required=True)
    p_shuf.add_argument("--revision", default=None)
    p_shuf.add_argument("--pos_path", required=True)
    p_shuf.add_argument("--neg_path", required=True)
    p_shuf.add_argument("--trait", required=True)
    p_shuf.add_argument("--save_dir", required=True)
    p_shuf.add_argument("--threshold", type=int, default=50)
    p_shuf.add_argument("--max_examples", type=int, default=None)
    p_shuf.add_argument("--hidden_batch_size", type=int, default=8)
    p_shuf.add_argument("--seeds", default="0,1,2")
    p_shuf.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)
    if args.mode == "random":
        build_random_direction_vectors(
            reference_vector_path=args.reference_vector,
            save_dir=args.save_dir,
            trait=args.trait,
            seeds=seeds,
            overwrite=args.overwrite,
        )
    elif args.mode == "shuffled":
        build_label_shuffled_vectors(
            model_name=args.model_name,
            pos_path=args.pos_path,
            neg_path=args.neg_path,
            trait=args.trait,
            save_dir=args.save_dir,
            seeds=seeds,
            threshold=args.threshold,
            max_examples=args.max_examples,
            revision=args.revision,
            hidden_batch_size=args.hidden_batch_size,
            overwrite=args.overwrite,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
