#!/usr/bin/env bash
#
# Pretraining checkpoint sweep — RQ1: same-checkpoint persona emergence.
#
# For each checkpoint in the model's grid this script:
#   1. Generates positive/negative instructed continuations (extraction data).
#   2. Extracts persona vectors from those continuations.
#   3. Runs an unsteered baseline evaluation.
#   4. Runs a steered evaluation with the extracted vector.
#
# Steering hyperparameters follow Table 7 of the paper:
#   OLMo-3-7B   evil l=16 c=0.5 | sycophantic l=16 c=0.5 | impolite l=16 c=0.5 | humorous l=20 c=0.3
#   Apertus-8B  evil l=16 c=0.2/0.5* | sycophantic l=16 c=0.2 | impolite l=20 c=0.15
#   * Apertus evil uses c=0.2 for checkpoints up to 4200B tokens, c=0.5 afterwards.
#
# Checkpoint grids are defined in pipeline/checkpoint_grids.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/checkpoint_grids.sh"

trim() { echo "$1" | sed 's/^ *//; s/ *$//'; }

# ── Model ─────────────────────────────────────────────────────────────────────
GPU=${GPU:-0}
MODEL=${MODEL:-"allenai/OLMo-3-1025-7B"}
MODEL_NAME=$(echo "$MODEL" | sed 's/.*\///')
MODEL_NAME_FOR_PATHS=${MODEL_NAME_FOR_PATHS:-"$MODEL_NAME"}

# ── Judge / generation ────────────────────────────────────────────────────────
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
N_PER_QUESTION_EXTRACT=${N_PER_QUESTION_EXTRACT:-10}
N_PER_QUESTION_EVAL=${N_PER_QUESTION_EVAL:-10}
MAX_TOKENS=${MAX_TOKENS:-64}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-8}
THRESHOLD=${THRESHOLD:-50}
SKIP_JUDGE=${SKIP_JUDGE:-False}

# ── Steering ──────────────────────────────────────────────────────────────────
STEERING_TYPE=${STEERING_TYPE:-"response"}
PREFER_TRANSFORMERS=${PREFER_TRANSFORMERS:-True}

# ── Run control ───────────────────────────────────────────────────────────────
OVERWRITE=${OVERWRITE:-False}
RUN_TAG=${RUN_TAG:-"same_checkpoint_v1"}

# ── Checkpoint grid ───────────────────────────────────────────────────────────
if echo "$MODEL_NAME_FOR_PATHS" | grep -qi "apertus"; then
    CHECKPOINT_GRID=${CHECKPOINT_GRID:-"$APERTUS_UNIVERSAL_TRANSFER_CHECKPOINT_GRID"}
    IS_APERTUS=true
else
    CHECKPOINT_GRID=${CHECKPOINT_GRID:-"$OLMO3_UNIVERSAL_TRANSFER_CHECKPOINT_GRID"}
    IS_APERTUS=false
fi

# ── Trait configuration (Table 7) ─────────────────────────────────────────────
if [ "$IS_APERTUS" = true ]; then
    TRAITS=${TRAITS:-"evil_character_neutral_q,sycophantic_character_neutral_q,impolite_character_neutral_q"}
    declare -A TRAIT_LAYER=( [evil_character_neutral_q]=16 [sycophantic_character_neutral_q]=16 [impolite_character_neutral_q]=20 )
    declare -A TRAIT_COEF=(  [evil_character_neutral_q]=0.2 [sycophantic_character_neutral_q]=0.2 [impolite_character_neutral_q]=0.15 )
    APERTUS_EVIL_LATE_COEF="0.5"
    APERTUS_EVIL_LATE_THRESHOLD_B=4200  # switch to 0.5 for checkpoints beyond 4200B tokens
else
    TRAITS=${TRAITS:-"evil_character_neutral_q,sycophantic_character_neutral_q,impolite_character_neutral_q,humorous_character_neutral_q"}
    declare -A TRAIT_LAYER=( [evil_character_neutral_q]=16 [sycophantic_character_neutral_q]=16 [impolite_character_neutral_q]=16 [humorous_character_neutral_q]=20 )
    declare -A TRAIT_COEF=(  [evil_character_neutral_q]=0.5 [sycophantic_character_neutral_q]=0.5 [impolite_character_neutral_q]=0.5 [humorous_character_neutral_q]=0.3 )
fi

# Returns the token count in B for an Apertus checkpoint name, 0 if unknown.
apertus_tokens_B() {
    case "$1" in
        main) echo 15000 ;;
        step*-tokens*B) echo "$1" | sed 's/.*-tokens\([0-9]*\)B/\1/' ;;
        *) echo 0 ;;
    esac
}

IFS=',' read -r -a traits_arr     <<< "$TRAITS"
IFS=',' read -r -a checkpoints_arr <<< "$CHECKPOINT_GRID"

BASE_EXTRACT_DIR="data/model_responses/extract/${MODEL_NAME_FOR_PATHS}"
BASE_VECTOR_DIR="data/persona_vectors/${MODEL_NAME_FOR_PATHS}"
BASE_EVAL_DIR="data/model_responses/eval/${MODEL_NAME_FOR_PATHS}"
SUMMARY_DIR="results/${MODEL_NAME_FOR_PATHS}/checkpoint_grid/${RUN_TAG}"
mkdir -p "$SUMMARY_DIR"

echo "============================================================"
echo "  Pretraining Checkpoint Sweep — Same-Checkpoint Emergence"
echo "============================================================"
echo "  Model:       $MODEL"
echo "  Checkpoints: ${#checkpoints_arr[@]}"
echo "  Traits:      ${traits_arr[*]}"
echo "  Run tag:     $RUN_TAG"
echo "  Summary:     $SUMMARY_DIR"
echo "============================================================"

for revision_raw in "${checkpoints_arr[@]}"; do
    revision=$(trim "$revision_raw")
    [ -z "$revision" ] && continue

    rev_flag="--revision ${revision}"
    extract_dir="${BASE_EXTRACT_DIR}/${revision}"
    vector_dir="${BASE_VECTOR_DIR}/${revision}"
    eval_dir="${BASE_EVAL_DIR}/${revision}/${RUN_TAG}"
    mkdir -p "$extract_dir" "$vector_dir" "$eval_dir"

    echo ""
    echo "── ${revision} ──────────────────────────────────────────"

    # ── 1. Instructed continuations ───────────────────────────────
    for trait in "${traits_arr[@]}"; do
        pos_csv="${extract_dir}/${trait}_pos_instruct.csv"
        neg_csv="${extract_dir}/${trait}_neg_instruct.csv"

        if [ -f "$pos_csv" ] && [ -f "$neg_csv" ] && [ "$OVERWRITE" != "True" ]; then
            echo "[extract]  ${trait} — skipping (exists)"
            continue
        fi
        echo "[extract]  ${trait}"
        for polarity in pos neg; do
            CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
                --model "$MODEL" $rev_flag \
                --trait "$trait" \
                --output_path "${extract_dir}/${trait}_${polarity}_instruct.csv" \
                --persona_instruction_type "$polarity" \
                --judge_model "$JUDGE_MODEL" \
                --n_per_question "$N_PER_QUESTION_EXTRACT" \
                --version extract \
                --max_tokens "$MAX_TOKENS" \
                --repetition_penalty "$REPETITION_PENALTY" \
                --generation_batch_size "$GENERATION_BATCH_SIZE" \
                --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
                --batch_process True \
                --skip_judge "$SKIP_JUDGE" \
                --overwrite "$OVERWRITE"
        done
    done

    # ── 2. Persona vector extraction ──────────────────────────────
    for trait in "${traits_arr[@]}"; do
        vector_path="${vector_dir}/${trait}_response_avg_diff.pt"

        if [ -f "$vector_path" ] && [ "$OVERWRITE" != "True" ]; then
            echo "[vector]   ${trait} — skipping (exists)"
            continue
        fi
        pos_csv="${extract_dir}/${trait}_pos_instruct.csv"
        neg_csv="${extract_dir}/${trait}_neg_instruct.csv"
        if [ ! -f "$pos_csv" ] || [ ! -f "$neg_csv" ]; then
            echo "[vector]   ${trait} — extraction CSVs missing, skipping"
            continue
        fi
        echo "[vector]   ${trait}"
        CUDA_VISIBLE_DEVICES=$GPU python -m source.generate_vec \
            --model_name "$MODEL" $rev_flag \
            --pos_path "$pos_csv" \
            --neg_path "$neg_csv" \
            --trait "$trait" \
            --save_dir "$vector_dir" \
            --threshold "$THRESHOLD"
    done

    # ── 3 & 4. Baseline + steered evaluation ─────────────────────
    for trait in "${traits_arr[@]}"; do
        layer="${TRAIT_LAYER[$trait]}"
        coef="${TRAIT_COEF[$trait]}"

        # Apertus evil: late checkpoints use a higher coefficient
        if [ "$IS_APERTUS" = true ] && [ "$trait" = "evil_character_neutral_q" ]; then
            tokens_b=$(apertus_tokens_B "$revision")
            if [ "$tokens_b" -gt "$APERTUS_EVIL_LATE_THRESHOLD_B" ]; then
                coef="$APERTUS_EVIL_LATE_COEF"
            fi
        fi

        coef_slug="${coef//./p}"
        baseline_out="${eval_dir}/baseline_${trait}.csv"
        steered_out="${eval_dir}/steered_${trait}_layer${layer}_coef${coef_slug}.csv"

        # Baseline
        if [ -f "$baseline_out" ] && [ "$OVERWRITE" != "True" ]; then
            echo "[baseline] ${trait} — skipping (exists)"
        else
            echo "[baseline] ${trait}"
            CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
                --model "$MODEL" $rev_flag \
                --trait "$trait" \
                --output_path "$baseline_out" \
                --judge_model "$JUDGE_MODEL" \
                --version eval \
                --n_per_question "$N_PER_QUESTION_EVAL" \
                --max_tokens "$MAX_TOKENS" \
                --repetition_penalty "$REPETITION_PENALTY" \
                --generation_batch_size "$GENERATION_BATCH_SIZE" \
                --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
                --batch_process True \
                --prefer_transformers "$PREFER_TRANSFORMERS" \
                --overwrite "$OVERWRITE"
        fi

        # Steered
        vector_path="${vector_dir}/${trait}_response_avg_diff.pt"
        if [ ! -f "$vector_path" ]; then
            echo "[steered]  ${trait} — vector missing, skipping"
            continue
        fi
        if [ -f "$steered_out" ] && [ "$OVERWRITE" != "True" ]; then
            echo "[steered]  ${trait} (l=${layer} c=${coef}) — skipping (exists)"
        else
            echo "[steered]  ${trait} (l=${layer} c=${coef})"
            CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
                --model "$MODEL" $rev_flag \
                --trait "$trait" \
                --output_path "$steered_out" \
                --vector_path "$vector_path" \
                --coef "$coef" \
                --layer "$layer" \
                --steering_type "$STEERING_TYPE" \
                --judge_model "$JUDGE_MODEL" \
                --version eval \
                --n_per_question "$N_PER_QUESTION_EVAL" \
                --max_tokens "$MAX_TOKENS" \
                --repetition_penalty "$REPETITION_PENALTY" \
                --generation_batch_size "$GENERATION_BATCH_SIZE" \
                --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
                --batch_process True \
                --prefer_transformers "$PREFER_TRANSFORMERS" \
                --overwrite "$OVERWRITE"
        fi
    done
done

echo ""
echo "============================================================"
echo "  Sweep complete."
echo "  Per-checkpoint results: ${BASE_EVAL_DIR}/<revision>/${RUN_TAG}/"
echo "  Summary directory:      ${SUMMARY_DIR}/"
echo "============================================================"
