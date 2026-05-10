#!/bin/bash
# ============================================================================
# Baseline ablation Script
# Runs evaluation across a trait for neutral evaluation sets
# Evaluated in the neutral eval set
# ============================================================================

set -e  # Exit on error


# Default configuration
GPU=${GPU:-0}
MODEL=${MODEL:-"allenai/OLMo-3-1025-7B"}
N_PER_QUESTION=${N_PER_QUESTION:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"data/model_responses/eval"}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}


EVAL_SETS=("dialogue" "stories" "character")
EVAL_SUFFIX=${EVAL_SUFFIX:-"_neutral_q"}


# Print configuration
echo ""
echo "============================================================"
echo "       Baseline Study Configuration"
echo "============================================================"
echo ""
echo "Model:          $MODEL"
echo "N per Question: $N_PER_QUESTION"
echo "Max Tokens:     $MAX_TOKENS"
echo "Judge Model:    $JUDGE_MODEL"
echo "GPU:            $GPU"
echo "Repetition Penalty: $REPETITION_PENALTY"
echo ""
echo ""
echo "============================================================"
echo ""

# Create output directory
MODEL_NAME=$(basename $MODEL)
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}/baseline"
mkdir -p "$OUTPUT_DIR"

for trait in "${EVAL_SETS[@]}"; do
    if [ ! -f "data/trait_data_eval/evil_${trait}${EVAL_SUFFIX}.json" ]; then
        echo "WARNING: Eval set not found: data/trait_data_eval/evil_${trait}${EVAL_SUFFIX}.json"
        echo "Please ensure all eval sets exist before running."
    fi
done

# Run ablation
for trait in "${EVAL_SETS[@]}"; do

    echo ""
    echo "============================================================"
    echo "Evaluating trait: $trait"
    echo "============================================================"

    trait_complete="evil_${trait}${EVAL_SUFFIX}"

    output_file="${OUTPUT_DIR}/${trait_complete}.csv"
    
    echo "output_file: $output_file"

    # Skip if output already exists
    if [ -f "$output_file" ]; then
        echo "Output already exists, skipping..."
        continue
    fi

    # Run evaluation
    echo "Running evaluation for ${trait_complete}..."
    if CUDA_VISIBLE_DEVICES=$GPU python -m source.eval_persona \
        --model "$MODEL" \
        --trait "${trait_complete}" \
        --output_path "$output_file" \
        --judge_model "$JUDGE_MODEL" \
        --version eval \
        --n_per_question "$N_PER_QUESTION" \
        --max_tokens "$MAX_TOKENS" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --overwrite False; then
        echo "✓ Completed successfully for ${trait_complete}"
    else
        echo "✗ Error in run for ${trait_complete}"
    fi
    
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
python analysis/collect_baseline_results.py --results_dir "$OUTPUT_DIR" --output_file "results/${MODEL_NAME}/baseline_summary.csv"
echo ""
echo "Summary saved to: results/${MODEL_NAME}/baseline_summary.csv"
echo ""