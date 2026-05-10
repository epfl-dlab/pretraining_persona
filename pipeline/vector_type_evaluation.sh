#!/bin/bash
# ============================================================================
# Vector Type Ablation Script
# Tests different vector types (stories, character, dialogue) with fixed layer/coefficient
# Each vector type is evaluated on all 3 eval sets for comparison
# ============================================================================

set -e  # Exit on error

# Default configuration
GPU=${1:-0}
MODEL=${MODEL:-"allenai/OLMo-3-1025-7B"}
MODEL_NAME=$(basename "$MODEL")
STEERING_TYPE=${STEERING_TYPE:-"response"}
N_PER_QUESTION=${N_PER_QUESTION:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"data/model_responses/eval"}
# Added the repetition penalty parameter
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}

# Fixed layer and coefficient
LAYER=${LAYER:-16}
COEF=${COEF:-2.0}

# Vector types to test (name|path)
VECTOR_TYPES=(
    "stories_neutral_q|data/persona_vectors/${MODEL_NAME}/evil_stories_neutral_q_response_avg_diff.pt"
    "character_neutral_q|data/persona_vectors/${MODEL_NAME}/evil_character_neutral_q_response_avg_diff.pt"
    "dialogue_neutral_q|data/persona_vectors/${MODEL_NAME}/evil_dialogue_neutral_q_response_avg_diff.pt"
    "combined|data/persona_vectors/${MODEL_NAME}/evil_combined_response_avg_diff.pt"
)

# Eval sets to run (all 3 for each vector type)
EVAL_SETS=("evil_stories_neutral_q" "evil_character_neutral_q" "evil_dialogue_neutral_q" "evil_combined")

# Print configuration
echo ""
echo "============================================================"
echo "       Vector Type Ablation Study"
echo "============================================================"
echo ""
echo "Model:          $MODEL"
echo "Steering Type:  $STEERING_TYPE"
echo "Layer:          $LAYER (fixed)"
echo "Coefficient:    $COEF (fixed)"
echo "N per Question: $N_PER_QUESTION"
echo "Max Tokens:     $MAX_TOKENS"
echo "Judge Model:    $JUDGE_MODEL"
echo "GPU:            $GPU"
echo ""
echo "Vector Types:"
for vt in "${VECTOR_TYPES[@]}"; do
    IFS='|' read -r vec_name vec_path <<< "$vt"
    echo "  - $vec_name: $vec_path"
done
echo ""
echo "Eval Sets:      ${EVAL_SETS[*]}"
echo ""
total_runs=$((${#VECTOR_TYPES[@]} * ${#EVAL_SETS[@]}))
echo "Total runs:     $total_runs"
echo "============================================================"
echo ""

# Create output directory
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}/vector_type"
mkdir -p "$OUTPUT_DIR"

# Verify all eval sets and vector paths exist
for vt in "${VECTOR_TYPES[@]}"; do
    IFS='|' read -r vec_name vec_path <<< "$vt"
    if [ ! -f "$vec_path" ]; then
        echo "WARNING: Vector path not found: $vec_path"
    fi
done
for eval_set in "${EVAL_SETS[@]}"; do
    if [ ! -f "data/trait_data_eval/${eval_set}.json" ]; then
        echo "WARNING: Eval set not found: data/trait_data_eval/${eval_set}.json"
    fi
done

# Counter for progress
current_run=0

# Run ablation over vector types
for vt in "${VECTOR_TYPES[@]}"; do
    IFS='|' read -r vec_name vec_path <<< "$vt"
    
    echo ""
    echo "============================================================"
    echo "Vector Type: $vec_name"
    echo "Vector Path: $vec_path"
    echo "============================================================"
    
    for eval_set in "${EVAL_SETS[@]}"; do
        current_run=$((current_run + 1))
        
        # Create output filename
        output_file="${OUTPUT_DIR}/${vec_name}_${eval_set}.csv"
        
        echo ""
        echo "[$current_run/$total_runs] Running: vec_type=$vec_name, eval_set=$eval_set"
        echo "Output: $output_file"
        
        # Skip if output already exists
        if [ -f "$output_file" ]; then
            echo "Output already exists, skipping..."
            continue
        fi
        
        # Run evaluation
        CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
            --model "$MODEL" \
            --trait "$eval_set" \
            --output_path "$output_file" \
            --judge_model "$JUDGE_MODEL" \
            --version eval \
            --steering_type "$STEERING_TYPE" \
            --coef "$COEF" \
            --vector_path "$vec_path" \
            --layer "$LAYER" \
            --n_per_question "$N_PER_QUESTION" \
            --max_tokens "$MAX_TOKENS" \
            --overwrite False \
            --repetition_penalty "$REPETITION_PENALTY"
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully"
        else
            echo "✗ Error in run"
        fi
    done
done

echo ""
echo "============================================================"
echo "All ablation runs completed!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Run the aggregation script
echo "Aggregating results..."
python analysis/collect_vector_type_results.py --results_dir "$OUTPUT_DIR" --output_file "results/${MODEL_NAME}/vector_type_summary.csv"
echo ""
echo "Summary saved to: results/${MODEL_NAME}/vector_type_summary.csv"
echo ""
