#!/usr/bin/env bash
#
# Persona Vector Transfer Sweep — RQ2: transfer to post-trained models.
#
# Takes persona vectors already extracted from a BASE model's pretraining
# checkpoints (produced by pipeline/checkpoint_sweep.sh) and applies them to
# steer a fixed POST-TRAINED (instruct) model. This isolates the question of
# whether vectors learned during pretraining generalise across the SFT/DPO/RLVR
# fine-tuning boundary.
#
# Key difference from checkpoint_sweep.sh:
#   - Extraction and evaluation use DIFFERENT models.
#   - Vectors come from EXTRACT_MODEL at each CHECKPOINT_GRID revision.
#   - Steering targets EVAL_MODEL (a single fixed instruct checkpoint, no revision).
#   - Baseline evaluation is run ONCE per trait (shared across all extract ckpts).
#
# Transfer hyperparameters used in the paper (Table 7 / make_transfer_plot.py):
#   OLMo-3 7B → Olmo-3-7B-Instruct (SFT/DPO/RLVR)
#     evil        l=16 c=0.55
#     humorous    l=20 c=0.30
#     impolite    l=20 c=0.75
#     sycophantic l=16 c=0.50
#   Apertus 8B → Apertus-8B-Instruct-2509
#     evil        l=16 c=0.3
#     impolite    l=20 c=0.7
#     sycophantic l=16 c=0.5
#
# Checkpoint grids are defined in pipeline/checkpoint_grids.sh.
#
# After this script completes, run an aggregation script (not included) to
# compute per-checkpoint deltas and significance tests and produce the
# combined_significance.csv files consumed by analysis/make_transfer_plot.py.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/checkpoint_grids.sh"

trim() { echo "$1" | sed 's/^ *//; s/ *$//'; }

# ── Extract model (source of persona vectors) ────────────────────────────────
EXTRACT_MODEL=${EXTRACT_MODEL:-"allenai/OLMo-3-1025-7B"}
EXTRACT_MODEL_NAME=$(echo "$EXTRACT_MODEL" | sed 's/.*\///')
EXTRACT_MODEL_NAME_FOR_PATHS=${EXTRACT_MODEL_NAME_FOR_PATHS:-"$EXTRACT_MODEL_NAME"}

# ── Eval model (post-trained target that gets steered) ────────────────────────
EVAL_MODEL=${EVAL_MODEL:-"allenai/Olmo-3-7B-Instruct"}
EVAL_MODEL_NAME=$(echo "$EVAL_MODEL" | sed 's/.*\///')
EVAL_MODEL_NAME_FOR_PATHS=${EVAL_MODEL_NAME_FOR_PATHS:-"$EVAL_MODEL_NAME"}

# ── Hardware / generation ─────────────────────────────────────────────────────
GPU=${GPU:-0}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
N_PER_QUESTION=${N_PER_QUESTION:-10}
MAX_TOKENS=${MAX_TOKENS:-64}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-8}

# ── Steering ──────────────────────────────────────────────────────────────────
STEERING_TYPE=${STEERING_TYPE:-"response"}
# Optional: normalise every vector to this L2 norm before steering.
# Leave empty to use the raw vector norm (same convention as checkpoint_sweep.sh).
# Set to a numeric value (e.g. 25.1) to match the instruct model's hidden-state
# scale and avoid inflated or deflated steering magnitudes across checkpoints.
VECTOR_NORM=${VECTOR_NORM:-""}

# ── Run control ───────────────────────────────────────────────────────────────
OVERWRITE=${OVERWRITE:-False}
RUN_TAG=${RUN_TAG:-"transfer_v1"}

# ── Checkpoint grid ───────────────────────────────────────────────────────────
if echo "$EXTRACT_MODEL_NAME_FOR_PATHS" | grep -qi "apertus"; then
    CHECKPOINT_GRID=${CHECKPOINT_GRID:-"$APERTUS_UNIVERSAL_TRANSFER_CHECKPOINT_GRID"}
    IS_APERTUS=true
else
    CHECKPOINT_GRID=${CHECKPOINT_GRID:-"$OLMO3_UNIVERSAL_TRANSFER_CHECKPOINT_GRID"}
    IS_APERTUS=false
fi

# ── Trait configuration ───────────────────────────────────────────────────────
# Defaults match the per-target transfer hyperparameters from the paper.
# Override TRAITS to run a subset (comma-separated).
if [ "$IS_APERTUS" = true ]; then
    TRAITS=${TRAITS:-"evil_character_neutral_q,sycophantic_character_neutral_q,impolite_character_neutral_q"}
    declare -A TRAIT_LAYER=( [evil_character_neutral_q]=16 [sycophantic_character_neutral_q]=16 [impolite_character_neutral_q]=20 )
    declare -A TRAIT_COEF=(  [evil_character_neutral_q]=0.3 [sycophantic_character_neutral_q]=0.5 [impolite_character_neutral_q]=0.7 )
else
    TRAITS=${TRAITS:-"evil_character_neutral_q,sycophantic_character_neutral_q,impolite_character_neutral_q,humorous_character_neutral_q"}
    declare -A TRAIT_LAYER=( [evil_character_neutral_q]=16 [sycophantic_character_neutral_q]=16 [impolite_character_neutral_q]=20 [humorous_character_neutral_q]=20 )
    declare -A TRAIT_COEF=(  [evil_character_neutral_q]=0.55 [sycophantic_character_neutral_q]=0.50 [impolite_character_neutral_q]=0.75 [humorous_character_neutral_q]=0.30 )
fi

IFS=',' read -r -a traits_arr      <<< "$TRAITS"
IFS=',' read -r -a checkpoints_arr <<< "$CHECKPOINT_GRID"

BASE_VECTOR_DIR="data/persona_vectors/${EXTRACT_MODEL_NAME_FOR_PATHS}"
EVAL_DIR_ROOT="data/model_responses/eval/${EVAL_MODEL_NAME_FOR_PATHS}/instruct_transfer/${RUN_TAG}"
SUMMARY_DIR="results/${EVAL_MODEL_NAME_FOR_PATHS}/instruct_transfer/${RUN_TAG}"
mkdir -p "${EVAL_DIR_ROOT}/baselines" "$SUMMARY_DIR"

echo "============================================================"
echo "  Persona Vector Transfer Sweep — RQ2"
echo "============================================================"
echo "  Extract model:  $EXTRACT_MODEL"
echo "  Eval model:     $EVAL_MODEL"
echo "  Checkpoints:    ${#checkpoints_arr[@]}"
echo "  Traits:         ${traits_arr[*]}"
echo "  Run tag:        $RUN_TAG"
echo "  Output:         $EVAL_DIR_ROOT"
echo "  Summary:        $SUMMARY_DIR"
[ -n "$VECTOR_NORM" ] && echo "  Vector norm:    $VECTOR_NORM (all vectors rescaled)"
echo "============================================================"

# ── 1. Baseline evaluation (once per trait, independent of extract checkpoint) ─
echo ""
echo "── Baseline evaluation on ${EVAL_MODEL} ──"
for trait in "${traits_arr[@]}"; do
    baseline_out="${EVAL_DIR_ROOT}/baselines/baseline_${trait}.csv"

    if [ -f "$baseline_out" ] && [ "$OVERWRITE" != "True" ]; then
        echo "[baseline] ${trait} — skipping (exists)"
        continue
    fi
    echo "[baseline] ${trait}"
    CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
        --model "$EVAL_MODEL" \
        --trait "$trait" \
        --output_path "$baseline_out" \
        --judge_model "$JUDGE_MODEL" \
        --version eval \
        --n_per_question "$N_PER_QUESTION" \
        --max_tokens "$MAX_TOKENS" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --generation_batch_size "$GENERATION_BATCH_SIZE" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --batch_process True \
        --overwrite "$OVERWRITE"
done

# ── 2. Steered evaluation per (trait, extract_revision) ───────────────────────
for revision_raw in "${checkpoints_arr[@]}"; do
    revision=$(trim "$revision_raw")
    [ -z "$revision" ] && continue

    echo ""
    echo "── ${revision} (vectors from ${EXTRACT_MODEL_NAME_FOR_PATHS}) ──"

    for trait in "${traits_arr[@]}"; do
        layer="${TRAIT_LAYER[$trait]}"
        coef="${TRAIT_COEF[$trait]}"
        coef_slug="${coef//./p}"

        vector_path="${BASE_VECTOR_DIR}/${revision}/${trait}_response_avg_diff.pt"
        if [ ! -f "$vector_path" ]; then
            echo "[steered]  ${trait} @ ${revision} — vector missing (${vector_path}), skipping"
            echo "           Run checkpoint_sweep.sh first to extract vectors."
            continue
        fi

        steered_dir="${EVAL_DIR_ROOT}/steered/${revision}"
        mkdir -p "$steered_dir"
        steered_out="${steered_dir}/steering_results_${trait}_layer${layer}_coef${coef_slug}.csv"

        if [ -f "$steered_out" ] && [ "$OVERWRITE" != "True" ]; then
            echo "[steered]  ${trait} (l=${layer} c=${coef}) — skipping (exists)"
            continue
        fi
        echo "[steered]  ${trait} (l=${layer} c=${coef}) @ ${revision}"

        vector_norm_args=()
        if [ -n "$VECTOR_NORM" ]; then
            vector_norm_args=(--vector_norm "$VECTOR_NORM")
            # Reflect the norm in the filename for traceability
            norm_slug="${VECTOR_NORM//./p}"
            steered_out="${steered_dir}/steering_results_${trait}_layer${layer}_targetnorm${norm_slug}_coef${coef_slug}.csv"
        fi

        CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
            --model "$EVAL_MODEL" \
            --trait "$trait" \
            --output_path "$steered_out" \
            --vector_path "$vector_path" \
            --coef "$coef" \
            --layer "$layer" \
            --steering_type "$STEERING_TYPE" \
            --judge_model "$JUDGE_MODEL" \
            --version eval \
            --n_per_question "$N_PER_QUESTION" \
            --max_tokens "$MAX_TOKENS" \
            --repetition_penalty "$REPETITION_PENALTY" \
            --generation_batch_size "$GENERATION_BATCH_SIZE" \
            --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
            --batch_process True \
            --overwrite "$OVERWRITE" \
            "${vector_norm_args[@]}"
    done
done

echo ""
echo "============================================================"
echo "  Transfer sweep complete."
echo "  Baselines:      ${EVAL_DIR_ROOT}/baselines/"
echo "  Steered:        ${EVAL_DIR_ROOT}/steered/<revision>/"
echo "  Summary dir:    ${SUMMARY_DIR}/"
echo ""
echo "  Next step: run an aggregation script to compute per-checkpoint"
echo "  delta_trait_mean and significance tests, then write the"
echo "  combined_significance.csv files consumed by:"
echo "    analysis/make_transfer_plot.py --model olmo3 --target instruct"
echo "    analysis/make_transfer_table.py --model olmo3 --target instruct"
echo "============================================================"
