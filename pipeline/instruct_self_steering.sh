#!/usr/bin/env bash
#
# Post-trained self-steering (Figure 1 / Figure 7, post-training column).
#
# The SFT / DPO / RLVR points in the emergence figure are NOT base→instruct
# transfer results. Each post-trained model is steered with a persona vector
# extracted from ITSELF. For every model in EVAL_MODELS this script:
#
#   1. generates judged positive / negative extraction responses from the model
#      (pos / neg system prompts × questions from EXTRACT_TRAIT_FILE, each
#      response judged for trait expression and coherence);
#   2. builds the model's own persona vector from the rows that pass the judge
#      filter (source.generate_vec: pos trait >= THRESHOLD, neg trait
#      < 100 - THRESHOLD, both coherence >= 50, evaluated row-wise on the
#      aligned pos / neg CSVs);
#   3. steers the same model with its own vector at LAYER / COEF, rescaled to
#      the model's mean layer-LAYER activation norm, and judges baseline vs.
#      steered responses on ${EVAL_TRAIT_DATA_DIR}/${TRAIT}.json.
#
# `delta_trait_mean` in the summary CSV is the quantity plotted for each
# post-training stage by analysis/make_emergence_plot.py.
#
# Paper settings (OLMo-3, upstream extraction prompts data/trait_data_extract/<trait>.json,
# eval prompts data/upstream_trait_data_eval/<trait>.json):
#   TRAIT=evil        LAYER=16 COEF=0.55
#   TRAIT=humorous    LAYER=20 COEF=0.3
#   TRAIT=impolite    LAYER=16 COEF=0.5    (diagnostic rerun: LAYER=20 COEF=0.75)
#   TRAIT=sycophantic LAYER=16 COEF=0.5
# Apertus-8B-Instruct-2509 used the paper's character prompts instead:
#   TRAIT=<trait>_character_neutral_q  EVAL_TRAIT_DATA_DIR=data/trait_data_eval
#   LAYER=16 COEF=0.3 (evil / humorous / sycophantic), LAYER=20 COEF=0.3 (impolite)
#
# Vector norm calibration: the vector is rescaled to the target model's mean
# layer-LAYER hidden-state L2 norm read from
# results/<model>/activation_norms/main_shared_norms.csv (shipped for all four
# post-trained targets used in the paper). Set VECTOR_NORM to override.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./checkpoint_grids.sh
source "${SCRIPT_DIR}/checkpoint_grids.sh"

trim() {
  echo "$1" | sed 's/^ *//; s/ *$//'
}

GPU=${GPU:-0}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
EVAL_MODELS=${EVAL_MODELS:-"allenai/Olmo-3-7B-Instruct-SFT,allenai/Olmo-3-7B-Instruct-DPO,allenai/Olmo-3-7B-Instruct"}
TRAIT=${TRAIT:-"evil"}
EXTRACT_TRAIT_FILE=${EXTRACT_TRAIT_FILE:-"data/trait_data_extract/${TRAIT}.json"}
EVAL_TRAIT_DATA_DIR=${EVAL_TRAIT_DATA_DIR:-"data/upstream_trait_data_eval"}
THRESHOLD=${THRESHOLD:-50}
N_PER_QUESTION_EXTRACT=${N_PER_QUESTION_EXTRACT:-2}
MAX_QUESTIONS_EXTRACT=${MAX_QUESTIONS_EXTRACT:-0}
MAX_VECTOR_EXAMPLES=${MAX_VECTOR_EXAMPLES:-0}
HIDDEN_BATCH_SIZE=${HIDDEN_BATCH_SIZE:-16}

STEERING_TYPE=${STEERING_TYPE:-response}
LAYER=${LAYER:-16}
COEF=${COEF:-0.55}
COEFFICIENTS=${COEFFICIENTS:-""}
VECTOR_NORM=${VECTOR_NORM:-""}

MAX_TOKENS=${MAX_TOKENS:-64}
N_PER_QUESTION_EVALUATE=${N_PER_QUESTION_EVALUATE:-3}
MAX_QUESTIONS_EVALUATE=${MAX_QUESTIONS_EVALUATE:-20}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-8}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
PREFER_TRANSFORMERS=${PREFER_TRANSFORMERS:-True}
BASELINE_PREFER_TRANSFORMERS=${BASELINE_PREFER_TRANSFORMERS:-"$PREFER_TRANSFORMERS"}

OVERWRITE=${OVERWRITE:-False}
RUN_BASELINES=${RUN_BASELINES:-True}
RUN_TAG=${RUN_TAG:-"${TRAIT}_instruct_self_steering_layer${LAYER}_coef${COEF//./p}_v1"}

if [ ! -f "$EXTRACT_TRAIT_FILE" ]; then
  echo "Missing extraction prompts file: $EXTRACT_TRAIT_FILE" >&2
  exit 1
fi
if [ ! -f "${EVAL_TRAIT_DATA_DIR}/${TRAIT}.json" ]; then
  echo "Missing eval trait file: ${EVAL_TRAIT_DATA_DIR}/${TRAIT}.json" >&2
  exit 1
fi

IFS=',' read -r -a eval_models <<< "$EVAL_MODELS"
if [ -n "$COEFFICIENTS" ]; then
  IFS=',' read -r -a coefficients <<< "$COEFFICIENTS"
else
  coefficients=("$COEF")
fi

summary_dir="results/self_steering/${RUN_TAG}"
summary_output="${summary_dir}/combined.csv"
mkdir -p "$summary_dir"

vec_overwrite_flag=()
if [ "$OVERWRITE" = "True" ] || [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "1" ]; then
  vec_overwrite_flag+=(--overwrite)
fi

echo "[self-steer] trait=${TRAIT} layer=${LAYER} coefs=${coefficients[*]} run_tag=${RUN_TAG}"
echo "[self-steer] extraction prompts: ${EXTRACT_TRAIT_FILE}; eval prompts: ${EVAL_TRAIT_DATA_DIR}/${TRAIT}.json"

for eval_model_raw in "${eval_models[@]}"; do
  eval_model=$(trim "$eval_model_raw")
  if [ -z "$eval_model" ]; then continue; fi

  eval_model_name=$(echo "$eval_model" | sed 's/.*\///')
  extract_dir="data/model_responses/extract/${eval_model_name}/main"
  vector_dir="data/persona_vectors/${eval_model_name}/main"
  mkdir -p "$extract_dir" "$vector_dir"

  pos_output_path="${extract_dir}/${TRAIT}_pos_instruct.csv"
  neg_output_path="${extract_dir}/${TRAIT}_neg_instruct.csv"
  vector_path="${vector_dir}/${TRAIT}_response_avg_diff.pt"
  status_path="${vector_dir}/${TRAIT}_response_avg_diff.status.json"

  # ----- 1. pos/neg extraction on the post-trained model ------
  if [ ! -f "$pos_output_path" ] || [ "$OVERWRITE" = "True" ]; then
    echo "[self-steer] extract pos model=${eval_model_name} trait=${TRAIT}"
    CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
      --model "$eval_model" \
      --trait "$TRAIT" \
      --trait_data_dir "$(dirname "$EXTRACT_TRAIT_FILE")" \
      --output_path "$pos_output_path" \
      --persona_instruction_type pos \
      --assistant_name "$TRAIT" \
      --judge_model "$JUDGE_MODEL" \
      --n_per_question "$N_PER_QUESTION_EXTRACT" \
      --version extract \
      --max_questions "$MAX_QUESTIONS_EXTRACT" \
      --generation_batch_size "$GENERATION_BATCH_SIZE" \
      --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
      --max_tokens "$MAX_TOKENS" \
      --repetition_penalty "$REPETITION_PENALTY" \
      --batch_process True \
      --skip_judge False \
      --prefer_transformers "$PREFER_TRANSFORMERS" \
      --overwrite "$OVERWRITE"
  fi

  if [ ! -f "$neg_output_path" ] || [ "$OVERWRITE" = "True" ]; then
    echo "[self-steer] extract neg model=${eval_model_name} trait=${TRAIT}"
    CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
      --model "$eval_model" \
      --trait "$TRAIT" \
      --trait_data_dir "$(dirname "$EXTRACT_TRAIT_FILE")" \
      --output_path "$neg_output_path" \
      --persona_instruction_type neg \
      --judge_model "$JUDGE_MODEL" \
      --n_per_question "$N_PER_QUESTION_EXTRACT" \
      --version extract \
      --max_questions "$MAX_QUESTIONS_EXTRACT" \
      --generation_batch_size "$GENERATION_BATCH_SIZE" \
      --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
      --max_tokens "$MAX_TOKENS" \
      --repetition_penalty "$REPETITION_PENALTY" \
      --batch_process True \
      --skip_judge False \
      --prefer_transformers "$PREFER_TRANSFORMERS" \
      --overwrite "$OVERWRITE"
  fi

  # ----- 2. build the self-vector from the filtered rows ------
  if [ ! -f "$vector_path" ] || [ "$OVERWRITE" = "True" ]; then
    echo "[self-steer] build vector model=${eval_model_name} trait=${TRAIT} threshold=${THRESHOLD}"
    CUDA_VISIBLE_DEVICES=$GPU python -m source.generate_vec \
      --model_name "$eval_model" \
      --pos_path "$pos_output_path" \
      --neg_path "$neg_output_path" \
      --trait "$TRAIT" \
      --save_dir "$vector_dir" \
      --threshold "$THRESHOLD" \
      --max_examples "$MAX_VECTOR_EXAMPLES" \
      --hidden_batch_size "$HIDDEN_BATCH_SIZE" \
      --status_path "$status_path" \
      "${vec_overwrite_flag[@]}"
  fi

  if [ ! -f "$vector_path" ]; then
    echo "[self-steer] vector not built for ${eval_model_name}; skipping steering"
    continue
  fi

  # ----- 3. target-norm lookup + baseline + steered eval ------
  run_root="data/model_responses/eval/${eval_model_name}/instruct_self_steering/${RUN_TAG}"
  baseline_dir="${run_root}/baselines"
  steered_root="${run_root}/steered"
  norm_file="results/${eval_model_name}/activation_norms/main_shared_norms.csv"
  baseline_output="${baseline_dir}/baseline_${TRAIT}.csv"
  mkdir -p "$baseline_dir" "$steered_root"

  if [ -n "$VECTOR_NORM" ]; then
    target_vector_norm="$VECTOR_NORM"
  else
    if [ ! -f "$norm_file" ]; then
      echo "Missing activation-norm file ${norm_file} for ${eval_model_name}." >&2
      echo "The paper's targets ship this file under results/<model>/activation_norms/." >&2
      echo "For a new target, set VECTOR_NORM to the model's mean layer-${LAYER} hidden-state L2 norm." >&2
      exit 1
    fi
    target_vector_norm=$(python - "$norm_file" "$LAYER" <<'PY'
import csv, sys
norm_file, layer = sys.argv[1], int(sys.argv[2])
with open(norm_file, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if int(row["layer"]) == layer:
            print(float(row["mean_l2"]))
            break
    else:
        raise SystemExit(f"layer {layer} not found in {norm_file}")
PY
)
  fi

  if [ "$RUN_BASELINES" = "True" ] || [ "$RUN_BASELINES" = "true" ] || [ "$RUN_BASELINES" = "1" ]; then
    if [ ! -f "$baseline_output" ] || [ "$OVERWRITE" = "True" ]; then
      echo "[self-steer] baseline model=${eval_model_name}"
      CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
        --model "$eval_model" \
        --trait "$TRAIT" \
        --trait_data_dir "$EVAL_TRAIT_DATA_DIR" \
        --output_path "$baseline_output" \
        --judge_model "$JUDGE_MODEL" \
        --version eval \
        --n_per_question "$N_PER_QUESTION_EVALUATE" \
        --max_questions "$MAX_QUESTIONS_EVALUATE" \
        --generation_batch_size "$GENERATION_BATCH_SIZE" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --max_tokens "$MAX_TOKENS" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --batch_process True \
        --skip_judge False \
        --prefer_transformers "$BASELINE_PREFER_TRANSFORMERS" \
        --overwrite "$OVERWRITE"
    fi
  fi

  norm_slug=$(printf '%s' "$target_vector_norm" | sed 's/\./p/g')
  for coef_raw in "${coefficients[@]}"; do
    coef=$(trim "$coef_raw")
    if [ -z "$coef" ]; then continue; fi
    coef_slug=$(printf '%s' "$coef" | sed 's/\./p/g')
    output_path="${steered_root}/steering_results_${TRAIT}_to_${TRAIT}_layer${LAYER}_targetnorm${norm_slug}_coef${coef_slug}.csv"

    if [ -f "$output_path" ] && [ "$OVERWRITE" != "True" ]; then
      echo "[self-steer] steered model=${eval_model_name} coef=${coef} — skipping (exists)"
      continue
    fi
    echo "[self-steer] steered model=${eval_model_name} coef=${coef} target_norm=${target_vector_norm}"
    CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
      --model "$eval_model" \
      --trait "$TRAIT" \
      --trait_data_dir "$EVAL_TRAIT_DATA_DIR" \
      --output_path "$output_path" \
      --vector_path "$vector_path" \
      --coef "$coef" \
      --layer "$LAYER" \
      --vector_norm "$target_vector_norm" \
      --steering_type "$STEERING_TYPE" \
      --judge_model "$JUDGE_MODEL" \
      --version eval \
      --n_per_question "$N_PER_QUESTION_EVALUATE" \
      --max_questions "$MAX_QUESTIONS_EVALUATE" \
      --generation_batch_size "$GENERATION_BATCH_SIZE" \
      --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
      --max_tokens "$MAX_TOKENS" \
      --repetition_penalty "$REPETITION_PENALTY" \
      --batch_process True \
      --skip_judge False \
      --prefer_transformers "$PREFER_TRANSFORMERS" \
      --overwrite "$OVERWRITE"
  done
done

# --- summary: one row per (eval_model, coef) with baseline / steered means, Δ, paired sign-flip p ---
python - "$summary_output" "$TRAIT" "$LAYER" "${EVAL_MODELS}" "$RUN_TAG" <<'PY'
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

summary_output = sys.argv[1]
trait = sys.argv[2]
layer = int(sys.argv[3])
eval_models_raw = sys.argv[4]
run_tag = sys.argv[5]

rows = []
for m in [x.strip() for x in eval_models_raw.split(",") if x.strip()]:
    name = m.split("/")[-1]
    run_root = Path(f"data/model_responses/eval/{name}/instruct_self_steering/{run_tag}")
    base_path = run_root / "baselines" / f"baseline_{trait}.csv"
    if not base_path.exists():
        continue
    base = pd.read_csv(base_path)
    base_trait_mean = float(base[trait].mean()) if trait in base.columns else None
    base_coh_mean = float(base["coherence"].mean()) if "coherence" in base.columns else None
    for csv_path in sorted((run_root / "steered").glob("*.csv")):
        stem = csv_path.stem
        coef_match = re.search(r"_coef([0-9p.+-]+)$", stem)
        target_match = re.search(r"_targetnorm([0-9p.+-]+)_", stem)
        coef = float(coef_match.group(1).replace("p", ".")) if coef_match else float("nan")
        target = float(target_match.group(1).replace("p", ".")) if target_match else float("nan")
        df = pd.read_csv(csv_path)
        out_trait = float(df[trait].mean()) if trait in df.columns else None
        out_coh = float(df["coherence"].mean()) if "coherence" in df.columns else None
        # per-question paired exact sign-flip test (exact for n <= 20 questions)
        p_val = None
        if trait in df.columns and "question_id" in df.columns and trait in base.columns and "question_id" in base.columns:
            b = base.groupby("question_id")[trait].mean().sort_index()
            s = df.groupby("question_id")[trait].mean().sort_index()
            s = s.reindex(b.index)
            d = (s - b).to_numpy()
            d = d[np.abs(d) > 1e-12]
            n = len(d)
            if n == 0:
                p_val = 1.0
            else:
                observed = abs(d.sum())
                total = 2 ** n
                hits = 0
                for s_ in range(total):
                    signs = np.array([1 if (s_ >> i) & 1 else -1 for i in range(n)])
                    if abs((signs * d).sum()) >= observed - 1e-12:
                        hits += 1
                p_val = hits / total if total else None
        rows.append(
            {
                "eval_model": m,
                "trait": trait,
                "layer": layer,
                "coef": coef,
                "target_activation_norm": target,
                "baseline_trait_mean": base_trait_mean,
                "baseline_coherence_mean": base_coh_mean,
                "output_trait_mean": out_trait,
                "output_coherence_mean": out_coh,
                "delta_trait_mean": None if out_trait is None or base_trait_mean is None else out_trait - base_trait_mean,
                "delta_coherence_mean": None if out_coh is None or base_coh_mean is None else out_coh - base_coh_mean,
                "trait_primary_p_two_sided": p_val,
                "steered_csv": str(csv_path),
            }
        )

Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(summary_output, index=False)
print("wrote", summary_output, "rows=", len(rows))
PY
