#!/usr/bin/env bash
#
# Negative-control experiment for persona steering.
#
# Motivation: our same-checkpoint emergence curve shows a nonzero Delta at every
# extractable checkpoint. A reviewer will ask whether any direction of matched
# shape would steer the model equally well. This script runs two controls at
# each tested checkpoint, judges them exactly like the real persona-vector
# runs, and writes a summary that can be compared to the emergence summary.
#
# Controls:
#   - random:   N Gaussian directions per checkpoint
#   - shuffled: N label-shuffled vectors per checkpoint
#
# Both controls are steered at the same layer / coefficient / scaling convention
# as the real persona vector, and judged with the same judge and eval prompts.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./olmo3_checkpoint_grids.sh
source "${SCRIPT_DIR}/checkpoint_grids.sh"

trim() {
  echo "$1" | sed 's/^ *//; s/ *$//'
}

GPU=${GPU:-0}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
MODEL=${MODEL:-"allenai/Olmo-3-1025-7B"}
MODEL_NAME=$(echo "$MODEL" | sed 's/.*\///')
MODEL_NAME_FOR_PATHS=${MODEL_NAME_FOR_PATHS:-"$MODEL_NAME"}

# A small, representative subset of the emergence grid. The point is to show
# that the null result does not depend on checkpoint, not to replicate the full
# curve under the null.
CONTROL_EXTRACT_REVISIONS=${CONTROL_EXTRACT_REVISIONS:-"stage1-step3000,stage1-step10000,stage1-step99000,stage1-step1413814,main"}
TRAIT=${TRAIT:-"evil_character_neutral_q"}
SEEDS=${SEEDS:-"0,1,2"}
THRESHOLD=${THRESHOLD:-50}
MAX_VECTOR_EXAMPLES=${MAX_VECTOR_EXAMPLES:-0}
HIDDEN_BATCH_SIZE=${HIDDEN_BATCH_SIZE:-8}

LAYER=${LAYER:-16}
COEF=${COEF:-0.5}
STEERING_TYPE=${STEERING_TYPE:-response}
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
RUN_TAG=${RUN_TAG:-"evil_controls_layer${LAYER}_coef${COEF//./p}_v1"}
CONTROL_MODES=${CONTROL_MODES:-"random,shuffled"}

BASE_EXTRACT_DIR="data/model_responses/extract/${MODEL_NAME_FOR_PATHS}"
BASE_VECTOR_DIR="data/persona_vectors/${MODEL_NAME_FOR_PATHS}"
CONTROL_VECTOR_DIR_BASE="data/persona_vectors/${MODEL_NAME_FOR_PATHS}-controls"

IFS=',' read -r -a control_extract_revisions <<< "$CONTROL_EXTRACT_REVISIONS"
IFS=',' read -r -a control_modes <<< "$CONTROL_MODES"
IFS=',' read -r -a seeds_arr <<< "$SEEDS"

summary_dir="results/${MODEL_NAME_FOR_PATHS}/controls/${RUN_TAG}"
mkdir -p "$summary_dir"

for revision_raw in "${control_extract_revisions[@]}"; do
  revision=$(trim "$revision_raw")
  if [ -z "$revision" ]; then continue; fi

  echo "=== [control] revision=${revision} ==="

  reference_vector="${BASE_VECTOR_DIR}/${revision}/${TRAIT}_response_avg_diff.pt"
  pos_csv="${BASE_EXTRACT_DIR}/${revision}/${TRAIT}_pos_instruct.csv"
  neg_csv="${BASE_EXTRACT_DIR}/${revision}/${TRAIT}_neg_instruct.csv"
  control_save_dir="${CONTROL_VECTOR_DIR_BASE}/${revision}"
  mkdir -p "$control_save_dir"

  if [ ! -f "$reference_vector" ]; then
    echo "[control] missing reference vector at ${reference_vector}; skipping revision ${revision}"
    continue
  fi

  # ------------ build controls ------------
  for mode_raw in "${control_modes[@]}"; do
    mode=$(trim "$mode_raw")
    case "$mode" in
      random)
        echo "[control] building random directions, seeds=${SEEDS}"
        env CUDA_VISIBLE_DEVICES=$GPU python3 -m source.build_control_vectors random \
          --reference_vector "$reference_vector" \
          --save_dir "$control_save_dir" \
          --trait "$TRAIT" \
          --seeds "$SEEDS" \
          $( [ "$OVERWRITE" = "True" ] && echo --overwrite )
        ;;
      shuffled)
        if [ ! -f "$pos_csv" ] || [ ! -f "$neg_csv" ]; then
          echo "[control] missing extraction CSVs for shuffled mode: ${pos_csv}, ${neg_csv}; skipping"
          continue
        fi
        echo "[control] building label-shuffled vectors, seeds=${SEEDS}"
        env CUDA_VISIBLE_DEVICES=$GPU python3 -m source.build_control_vectors shuffled \
          --model_name "$MODEL" \
          --revision "$revision" \
          --pos_path "$pos_csv" \
          --neg_path "$neg_csv" \
          --trait "$TRAIT" \
          --save_dir "$control_save_dir" \
          --threshold "$THRESHOLD" \
          --max_examples "$MAX_VECTOR_EXAMPLES" \
          --hidden_batch_size "$HIDDEN_BATCH_SIZE" \
          --seeds "$SEEDS" \
          $( [ "$OVERWRITE" = "True" ] && echo --overwrite )
        ;;
      *)
        echo "[control] unknown mode ${mode}; skipping"
        ;;
    esac
  done

  # ------------ baseline + steered eval on the same checkpoint ------------
  eval_root="data/model_responses/eval/${MODEL_NAME_FOR_PATHS}/${revision}/controls/${RUN_TAG}"
  baseline_dir="${eval_root}/baselines"
  steered_root="${eval_root}/steered"
  mkdir -p "$baseline_dir" "$steered_root"

  revision_args=(--revision "$revision")
  baseline_output="${baseline_dir}/baseline_${TRAIT}.csv"
  if [ "$RUN_BASELINES" = "True" ] || [ "$RUN_BASELINES" = "true" ] || [ "$RUN_BASELINES" = "1" ]; then
    if [ ! -f "$baseline_output" ] || [ "$OVERWRITE" = "True" ]; then
      echo "[control] baseline ${revision}"
      env CUDA_VISIBLE_DEVICES=$GPU python3 -m source.eval_persona \
        --model "$MODEL" \
        "${revision_args[@]}" \
        --trait "$TRAIT" \
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

  coef_slug=$(printf '%s' "$COEF" | sed 's/\./p/g')

  for mode_raw in "${control_modes[@]}"; do
    mode=$(trim "$mode_raw")
    for seed_raw in "${seeds_arr[@]}"; do
      seed=$(trim "$seed_raw")
      if [ -z "$seed" ]; then continue; fi
      vector_path="${control_save_dir}/${TRAIT}__${mode}__seed${seed}_response_avg_diff.pt"
      if [ ! -f "$vector_path" ]; then
        echo "[control] missing control vector ${vector_path}; skipping"
        continue
      fi
      out_dir="${steered_root}/${mode}/seed${seed}"
      mkdir -p "$out_dir"
      output_path="${out_dir}/steering_results_${TRAIT}_to_${TRAIT}_layer${LAYER}_coef${coef_slug}.csv"

      echo "[control] steered revision=${revision} mode=${mode} seed=${seed}"
      env CUDA_VISIBLE_DEVICES=$GPU python3 -m source.eval_persona \
        --model "$MODEL" \
        "${revision_args[@]}" \
        --trait "$TRAIT" \
        --output_path "$output_path" \
        --vector_path "$vector_path" \
        --coef "$COEF" \
        --layer "$LAYER" \
        --vector_norm 1.0 \
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
done

echo "[control] done. outputs under data/model_responses/eval/${MODEL_NAME_FOR_PATHS}/<rev>/controls/${RUN_TAG}"
echo "[control] summary dir: ${summary_dir}"
