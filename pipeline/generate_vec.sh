# This script runs the persona vector generation for a specified model and trait using file naming conventions.

# Script arguments (with defaults)
gpu=${GPU:-0}
model=${MODEL:-"allenai/OLMo-3-1025-7B"}
trait=${TRAIT:-"evil"}
revision=${REVISION:-""}

# Configuration variables
threshold=${THRESHOLD:-50}

base_output_dir="data/model_responses/extract"
save_dir_base="data/persona_vectors"

# Extract model name for paths (e.g., "Qwen2.5-7B-Instruct" from "Qwen/Qwen2.5-7B-Instruct")
# If model contains "/", use the part after "/", otherwise use the full model name
model_name=$(echo $model | sed 's/.*\///')
model_name_for_paths=${MODEL_NAME_FOR_PATHS:-$model_name}

rev_flag=""
rev_dir=""
if [ "$revision" != "" ]; then
    rev_flag="--revision $revision"
    rev_dir="$revision/"
fi

# Construct paths
pos_output_path="$base_output_dir/$model_name_for_paths/${rev_dir}${trait}_pos_instruct.csv"
neg_output_path="$base_output_dir/$model_name_for_paths/${rev_dir}${trait}_neg_instruct.csv"
save_dir="$save_dir_base/$model_name_for_paths/${rev_dir}"

# Generate persona vectors
CUDA_VISIBLE_DEVICES=$gpu python -m source.generate_vec \
    --model_name $model \
    --pos_path $pos_output_path \
    --neg_path $neg_output_path \
    --trait $trait \
    --save_dir $save_dir \
    --threshold $threshold \
    $rev_flag
