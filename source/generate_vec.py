import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
import json
import torch
import os
import argparse
from source.model_utils import load_model


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    

def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None, batch_size=8):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in tqdm(range(0, len(texts), batch_size), total=(len(texts) + batch_size - 1) // batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_prompts = prompts[start:start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, add_special_tokens=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            prompt_tokenized = tokenizer(batch_prompts, add_special_tokens=False)
            prompt_lens = [len(ids) for ids in prompt_tokenized["input_ids"]]
            seq_lens = inputs["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            for layer in layer_list:
                layer_hidden = outputs.hidden_states[layer]
                batch_prompt_avg = []
                batch_response_avg = []
                batch_prompt_last = []
                for sample_idx, (prompt_len, seq_len) in enumerate(zip(prompt_lens, seq_lens)):
                    sample_hidden = layer_hidden[sample_idx]
                    batch_prompt_avg.append(sample_hidden[:prompt_len, :].mean(dim=0, keepdim=True).detach().cpu())
                    batch_response_avg.append(sample_hidden[prompt_len:seq_len, :].mean(dim=0, keepdim=True).detach().cpu())
                    batch_prompt_last.append(sample_hidden[prompt_len - 1, :].unsqueeze(0).detach().cpu())
                prompt_avg[layer].append(torch.cat(batch_prompt_avg, dim=0))
                response_avg[layer].append(torch.cat(batch_response_avg, dim=0))
                prompt_last[layer].append(torch.cat(batch_prompt_last, dim=0))
            del outputs
    finally:
        tokenizer.padding_side = original_padding_side
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg

import pandas as pd


class PersonaVectorBuildError(ValueError):
    def __init__(self, message, *, reason_code, details):
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details


def _score_summary(series):
    return None if series is None else float(series.mean())


def _build_filter_details(persona_pos, persona_neg, trait, threshold, mask=None):
    inverse_threshold = 100 - threshold
    details = {
        "trait": trait,
        "threshold": threshold,
        "inverse_threshold": inverse_threshold,
        "pos_rows": int(len(persona_pos)),
        "neg_rows": int(len(persona_neg)),
        "pos_trait_mean": _score_summary(persona_pos[trait]) if trait in persona_pos.columns else None,
        "neg_trait_mean": _score_summary(persona_neg[trait]) if trait in persona_neg.columns else None,
        "pos_coherence_mean": _score_summary(persona_pos["coherence"]) if "coherence" in persona_pos.columns else None,
        "neg_coherence_mean": _score_summary(persona_neg["coherence"]) if "coherence" in persona_neg.columns else None,
        "pos_trait_ge_threshold_count": int((persona_pos[trait] >= threshold).sum()) if trait in persona_pos.columns else None,
        "neg_trait_lt_inverse_threshold_count": int((persona_neg[trait] < inverse_threshold).sum()) if trait in persona_neg.columns else None,
        "pos_coherence_ge_50_count": int((persona_pos["coherence"] >= 50).sum()) if "coherence" in persona_pos.columns else None,
        "neg_coherence_ge_50_count": int((persona_neg["coherence"] >= 50).sum()) if "coherence" in persona_neg.columns else None,
    }
    if mask is not None:
        details["effective_rows"] = int(mask.sum())
    return details


def _write_status(status_path, payload):
    if status_path is None:
        return
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def get_persona_effective(pos_path, neg_path, trait, threshold=50, max_examples=None, skip_empty=False):
    persona_pos = pd.read_csv(pos_path).reset_index(drop=True)
    persona_neg = pd.read_csv(neg_path).reset_index(drop=True)

    if len(persona_pos) != len(persona_neg):
        min_len = min(len(persona_pos), len(persona_neg))
        print(
            f"WARNING: pos/neg row count mismatch ({len(persona_pos)} vs {len(persona_neg)}). "
            f"Trimming both to {min_len} rows."
        )
        persona_pos = persona_pos.iloc[:min_len].reset_index(drop=True)
        persona_neg = persona_neg.iloc[:min_len].reset_index(drop=True)

    has_trait_cols = trait in persona_pos.columns and trait in persona_neg.columns
    has_coherence_cols = "coherence" in persona_pos.columns and "coherence" in persona_neg.columns

    if not has_trait_cols or not has_coherence_cols:
        if skip_empty: return [None] * 6
        raise PersonaVectorBuildError(
            "Missing required judge columns for vector filtering. "
            f"Need trait='{trait}' and 'coherence' in both files: "
            f"pos={pos_path}, neg={neg_path}",
            reason_code="missing_judge_columns",
            details={
                **_build_filter_details(persona_pos, persona_neg, trait, threshold),
                "pos_path": str(pos_path),
                "neg_path": str(neg_path),
            },
        )

    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= 50)
        & (persona_neg["coherence"] >= 50)
    )
    if mask.sum() == 0:
        if skip_empty: return [None] * 6
        raise PersonaVectorBuildError(
            "0 rows passed trait/coherence filtering. "
            f"Refusing to build vector from unfiltered data for trait='{trait}'. "
            f"pos={pos_path}, neg={neg_path}, threshold={threshold}",
            reason_code="zero_effective_rows",
            details={
                **_build_filter_details(persona_pos, persona_neg, trait, threshold, mask=mask),
                "pos_path": str(pos_path),
                "neg_path": str(neg_path),
            },
        )

    print(mask.sum(), "effective samples found for trait", trait)
    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    if max_examples is not None and max_examples > 0:
        persona_pos_effective = persona_pos_effective.iloc[:max_examples]
        persona_neg_effective = persona_neg_effective.iloc[:max_examples]
        print(f"Limiting effective samples to first {len(persona_pos_effective)} rows")

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()    
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir, threshold=50, revision=None, overwrite=False, max_examples=None, hidden_batch_size=8, status_path=None):
    # check if the three output .pt files exist
    if all(os.path.exists(f"{save_dir}/{trait}_{suffix}_diff.pt") for suffix in ["prompt_avg", "response_avg", "prompt_last"]) and not overwrite:
        print(f"Persona vectors for trait {trait} already exist in {save_dir}. Skipping computation.")
        _write_status(
            status_path,
            {
                "status": "complete",
                "reason_code": "already_exists",
                "trait": trait,
                "revision": revision,
                "save_dir": save_dir,
                "vector_path": f"{save_dir}/{trait}_response_avg_diff.pt",
            },
        )
        return

    persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses = get_persona_effective(pos_path, neg_path, trait, threshold, max_examples=max_examples)

    model, tokenizer = load_model(model_name, revision=revision)
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash is not None:
        print("Commit hash:", commit_hash)

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses, batch_size=hidden_batch_size)
    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses, batch_size=hidden_batch_size)
    


    persona_effective_prompt_avg_diff = torch.stack([persona_effective_prompt_avg["pos"][l].mean(0).float() - persona_effective_prompt_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_avg["pos"]))], dim=0)
    persona_effective_response_avg_diff = torch.stack([persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_response_avg["pos"]))], dim=0)
    persona_effective_prompt_last_diff = torch.stack([persona_effective_prompt_last["pos"][l].mean(0).float() - persona_effective_prompt_last["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_last["pos"]))], dim=0)

    os.makedirs(save_dir, exist_ok=True)

    torch.save(persona_effective_prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(persona_effective_response_avg_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")

    _write_status(
        status_path,
        {
            "status": "complete",
            "reason_code": "vector_built",
            "trait": trait,
            "revision": revision,
            "save_dir": save_dir,
            "vector_path": f"{save_dir}/{trait}_response_avg_diff.pt",
            "effective_rows": int(len(persona_pos_effective)),
            "commit_hash": commit_hash,
            "pos_path": str(pos_path),
            "neg_path": str(neg_path),
            "threshold": threshold,
        },
    )
    print(f"Persona vectors saved to {save_dir}")    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--revision", type=str, required=False, default=None)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--hidden_batch_size", type=int, default=8)
    parser.add_argument("--status_path", type=str, default=None)
    args = parser.parse_args()
    try:
        save_persona_vector(
            args.model_name,
            args.pos_path,
            args.neg_path,
            args.trait,
            args.save_dir,
            args.threshold,
            args.revision,
            args.overwrite,
            args.max_examples,
            args.hidden_batch_size,
            args.status_path,
        )
    except PersonaVectorBuildError as exc:
        _write_status(
            args.status_path,
            {
                "status": "nonextractable" if exc.reason_code == "zero_effective_rows" else "invalid_inputs",
                "reason_code": exc.reason_code,
                "message": str(exc),
                "trait": args.trait,
                "revision": args.revision,
                "save_dir": args.save_dir,
                **exc.details,
            },
        )
        print(str(exc), file=sys.stderr)
        if exc.reason_code == "zero_effective_rows":
            raise SystemExit(3)
        raise SystemExit(1)
