## This script generates question responses using positive and negative persona instructions
## and saves the outputs to CSV files to judge the model's adherence to the specified trait and extract persona vectors.

# Script arguments (with defaults)
gpu=${GPU:-0}
model=${MODEL:-"allenai/OLMo-3-1025-7B"}
trait=${TRAIT:-"evil"}
revision=${REVISION:-""}

# Configuration variables
judge_model=${JUDGE_MODEL:-"gpt-4.1-mini-2025-04-14"}
n_per_question=${N_PER_QUESTION:-10}
max_tokens=${MAX_TOKENS:-64}
repetition_penalty=${REPETITION_PENALTY:-1.1}

base_output_dir="data/model_responses/extract"
save_dir_base="data/persona_vectors"

# Extract model name for paths (e.g., "Qwen2.5-7B-Instruct" from "Qwen/Qwen2.5-7B-Instruct")
# If model contains "/", use the part after "/", otherwise use the full model name
model_name=$(echo $model | sed 's/.*\///')
model_name_for_paths=${9:-$model_name}
skip_judge=${SKIP_JUDGE:-False}

rev_flag=""
rev_dir=""
if [ "$revision" != "" ]; then
    rev_flag="--revision $revision"
    rev_dir="$revision/"
fi


# Construct paths
pos_output_path="$base_output_dir/$model_name_for_paths/${rev_dir}${trait}_pos_instruct.csv"
neg_output_path="$base_output_dir/$model_name_for_paths/${rev_dir}${trait}_neg_instruct.csv"


# Run positive instruction evaluation
CUDA_VISIBLE_DEVICES=$gpu python -m source.eval_persona \
    --model $model \
    --trait $trait \
    --output_path $pos_output_path \
    --persona_instruction_type pos \
    --assistant_name $trait \
    --judge_model $judge_model \
    --n_per_question $n_per_question \
    --version extract \
    --max_tokens $max_tokens \
    --repetition_penalty $repetition_penalty \
    --batch_process True \
    --skip_judge $skip_judge \
    $rev_flag

# Run negative instruction evaluation
CUDA_VISIBLE_DEVICES=$gpu python -m source.eval_persona \
    --model $model \
    --trait $trait \
    --output_path $neg_output_path \
    --persona_instruction_type neg \
    --assistant_name $anti_trait \
    --judge_model $judge_model \
    --n_per_question $n_per_question \
    --version extract \
    --max_tokens $max_tokens \
    --repetition_penalty $repetition_penalty \
    --batch_process True \
    --skip_judge $skip_judge \
    $rev_flag