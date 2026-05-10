# Tracing Persona Vectors Through LLM Pretraining

![Python](https://img.shields.io/badge/python-3.11-blue)
![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%2C%20Seaborn%2C%20transformers%2C%20vllm-lightgrey)

This is the repository for the paper "Tracing Persona Vectors Through LLM Pretraining". It contains the code and data to reproduce the experiments and results presented in the paper.

## Dependencies and External Code
This repository builds upon the code from the original persona vector paper, which can be found in this [GitHub repository](https://github.com/safety-research/persona_vectors/tree/main). Files adopted from the original repo are indicated in their initial comment.

The code requires Python 3.11. External libraries include google-genai, matplotlib, numpy, openai, pandas, seaborn, transformers, and vllm libraries. All dependencies and version numbers are listed in the [requirements.txt](requirements.txt) file.

## Hardware
We used a single NVIDIA A100 GPU for most experiments. If the compute cluster supports fractional GPU allocation, 24GB VRAM should be sufficient for the 7B and 8B models.


## Repository Structure

```
pretraining_persona/
├── requirements.txt                        # Python dependencies
├── .env.example                            # Template for API keys
├── analysis/                               # Analysis and plotting scripts
├── data/                                   # All data (input and generations)
│   ├── trait_data_extract/                 # Prompt pairs used to extract persona vectors
│   ├── trait_data_eval/                    # Prompt sets used to evaluate steering
│   ├── upstream_trait_data_eval/           # Original eval data from the upstream repo
│   ├── persona_vectors/                    # Extracted persona vectors (.pt), keyed by model and trait
│   └── model_responses/                    # Model responses and LLM-as-judge verdicts (CSV)
├── pipeline/                               # End-to-end orchestration scripts
├── results/                                # Aggregated summary CSVs, keyed by model
└── source/                                 # Core Python library, partially adapted from the original repo
    ├── config.py                           # Shared configuration (paths, model names, hyperparameters)
    ├── generate_vec.py                     # Extract persona vectors from model activations
    ├── activation_steer.py                 # Apply persona vectors via activation addition
    ├── eval_persona.py                     # Run steered generation and collect responses
    ├── judge.py                            # GPT-based judge
    ├── deepseek_judge.py                   # DeepSeek-based trait expression judge
    ├── model_utils.py                      # Model and tokenizer loading helpers
    ├── prompts.py                          # Prompt templates shared across scripts
    └── utils.py                            # General utility functions
```

## Setup
An OpenAI API key is necessary for using GPT4.1 as a judge like in our report. A Gemini API key is necessary for creating the character description extract and eval dataset. A DeepSeek API key is necessary for using the DeepSeek-based trait expression judge. The keys should be specified in the .env file. We recommend to use a Docker container.

1. Create a virtual environment.

```bash
python3.11 -m venv pretraining-persona
source pretraining-persona/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment and fill in your API keys in the .env file
```bash
cp .env.example .env
```

## Reproducibility

## Script Notes

- All scripts should be run from the main directory
- Output directories are created automatically
- Most scripts skip runs if output files already exist (check individual scripts) unless specified overwrite flags
- Ablation scripts automatically aggregate results (see bottom of script files)

## Baseline Evaluation

To reproduce the baseline evaluation results, please run the following script:

```bash
./pipeline/baseline_evaluation.sh
```

**These arguments can be specified via environment variables:**
- `GPU` (default: 0) - GPU device ID
- `MODEL` (default: "allenai/OLMo-3-1025-7B") - Model name or path
- `N_PER_QUESTION` (default: 5) - Number of responses per question
- `MAX_TOKENS` (default: 64) - Maximum number of tokens
- `JUDGE_MODEL` (default: "gpt-4.1-mini-2025-04-14") - Judge model for evaluation
- `OUTPUT_BASE_DIR` (default: "ablation_results") - Base directory for output
- `REPETITION_PENALTY` (default: 1.1) - Repetition penalty

**Output**
- model responses and their judgments in `data/model_responses/{MODEL}/baseline`
- aggregated statistics in `results/{MODEL}baseline_summary.csv`
  


## Response Generation for Instructed Prompts

Run the following script calls to generate responses for the instructed prompts using the base models. The outputs will be saved in `data/model_responses/extract`.

```
TRAIT=evil_character_neutral_q ./pipeline/instructed_continuations.sh
```

**These arguments can be specified via environment variables:**
- `GPU` (default: 0) - GPU device ID
- `MODEL` (default: "allenai/OLMo-3-1025-7B") - Model name or path
- `TRAIT` (default: "evil") - Trait to evaluate
- `REVISION` (default: "") - Model checkpoint
- `JUDGE_MODEL` (default: "gpt-4.1-mini-2025-04-14") - Judge model for evaluation
- `N_PER_QUESTION` (default: 10) - Number of responses per question
- `MAX_TOKENS` (default: 64) - Maximum number of tokens
- `REPETITION_PENALTY` (default: 1.1) - Repetition penalty
- `SKIP_JUDGE` (default: False) - Whether to skip judging the responses for debugging

**Generated files:**
- `data/model_responses/extract/{MODEL}/{REVISION}/{TRAIT}_pos_instruct.csv`
- `data/model_responses/extract/{MODEL}/{REVISION}/{TRAIT}_neg_instruct.csv`

### Merge Model Responses for Combined Vector Extraction

To run the combined persona vector extraction below, execute `python pipeline/merge_persona_vectors.py`.

**Params:**
- `--model`: Model name (e.g., "allenai/OLMo-3-1025-7B")
- `--revision`: Model revision (default: "")
- `--methods`: Comma-separated list of methods to merge (default: all neutral question methods)

**Generated file:**
- `data/trait_data_eval/{MODEL}/evil_combined.json`

## Persona Vector Extraction (Single Checkpoint)

Execute the following script calls to extract persona vectors from a base model. The outputs will be saved in `data/persona_vectors`. 

```
TRAIT=evil_character_neutral_q ./pipeline/generate_vec.sh
```

**These arguments can be specified via environment variables:**
- `GPU` (default: 0) - GPU device ID
- `MODEL` (default: "allenai/OLMo-3-1025-7B") - Model name or path
- `TRAIT` (default: "evil") - Trait to evaluate, options see above
- `THRESHOLD` (default: 50) - Threshold value for evilness

**Generated Files:**
- `data/persona_vectors/{MODEL}/{TRAIT}_prompt_avg_diff.pt`: Average prompt activations difference
- `data/persona_vectors/{MODEL}/{TRAIT}_response_avg_diff.pt`: Average response activations difference
- `data/persona_vectors/{MODEL}/{TRAIT}_prompt_last_diff.pt`: Last prompt token activations difference

Each vector has shape: `[layers × hidden_dim]`


## Shared Checkpoint Grids

`pipeline/checkpoint_grids.sh` defines the canonical checkpoint lists used across all sweep experiments. It is sourced automatically by scripts that loop over checkpoints — do not run it directly.

| Variable | Model | Checkpoints |
|---|---|---|
| `OLMO3_UNIVERSAL_TRANSFER_CHECKPOINT_GRID` | OLMo-3-1025-7B | 16 checkpoints: `stage1-step3000` … `main` |
| `APERTUS_UNIVERSAL_TRANSFER_CHECKPOINT_GRID` | Apertus-8B-2509 | 15 checkpoints: `step50000-tokens210B` … `main` |

It also sets `OLMO3_SHARED_NORM_CALIBRATION_TEXT_FILES` pointing to the shared prompt file used for hidden-state norm calibration.


## Pretraining Checkpoint Sweep (RQ1 — Persona Emergence)

The core experiment of the paper (Figures 1 and 7). For each pretraining checkpoint in the model's grid, persona vectors are extracted and used to steer the **same checkpoint**. The reported metric is the steered-minus-baseline trait-expression delta (∆τ). Per checkpoint the script runs four steps in sequence: generate instructed continuations → extract vectors → baseline eval → steered eval.

Steering hyperparameters are fixed per model–trait pair (Table 7). Apertus evil uses c=0.2 for checkpoints up to 4200B tokens and c=0.5 afterwards.

```bash
# OLMo-3-7B (default)
./pipeline/checkpoint_sweep.sh

# Apertus-8B-2509
MODEL=Apertus/Apertus-8B-2509 MODEL_NAME_FOR_PATHS=Apertus-8B-2509 \
    ./pipeline/checkpoint_sweep.sh
```

**Environment Variables (optional):**
- `GPU` (default: 0)
- `MODEL` (default: `"allenai/OLMo-3-1025-7B"`)
- `MODEL_NAME_FOR_PATHS` — filesystem-safe model name (default: basename of `MODEL`)
- `TRAITS` — comma-separated trait list (default: all four for OLMo-3, three for Apertus)
- `CHECKPOINT_GRID` — override the checkpoint list (default: model-appropriate grid from `checkpoint_grids.sh`)
- `RUN_TAG` (default: `"same_checkpoint_v1"`) — output subdirectory label
- `N_PER_QUESTION_EXTRACT` (default: 10) / `N_PER_QUESTION_EVAL` (default: 10)
- `MAX_TOKENS` (default: 64) / `REPETITION_PENALTY` (default: 1.1)
- `JUDGE_MODEL` (default: `"gpt-4.1-mini-2025-04-14"`)
- `THRESHOLD` (default: 50) — minimum score for a generation to be included in vector extraction
- `SKIP_JUDGE` (default: `False`) — skip judging during extraction (useful for debugging)
- `OVERWRITE` (default: `False`) — re-run steps even if output files already exist

**Steering hyperparameters (Table 7, hard-coded per trait):**

| Model | Trait | Layer | Coef |
|---|---|---|---|
| OLMo-3-7B | evil, sycophantic, impolite | 16 | 0.5 |
| OLMo-3-7B | humorous | 20 | 0.3 |
| Apertus-8B | evil | 16 | 0.2 (≤4200B tokens) / 0.5 (>4200B) |
| Apertus-8B | sycophantic | 16 | 0.2 |
| Apertus-8B | impolite | 20 | 0.15 |

**Output layout** (one directory per checkpoint):
```
data/model_responses/extract/{MODEL}/{revision}/{trait}_{pos|neg}_instruct.csv
data/persona_vectors/{MODEL}/{revision}/{trait}_response_avg_diff.pt
data/model_responses/eval/{MODEL}/{revision}/{RUN_TAG}/baseline_{trait}.csv
data/model_responses/eval/{MODEL}/{revision}/{RUN_TAG}/steered_{trait}_layer{L}_coef{C}.csv
results/{MODEL}/checkpoint_grid/{RUN_TAG}/
```

### Emergence Plots (`analysis/make_emergence_plot.py` / `analysis/make_emergence_table.py`)

Produces the same-checkpoint emergence figure (Figure 1): Δ trait score vs. pretraining tokens, one curve per trait, with significance stars at each point. Also emits a companion LaTeX longtable.

**Prerequisites:** `results/{MODEL}/checkpoint_grid/{RUN_TAG}/combined.csv` from `checkpoint_sweep.sh`.

```bash
python analysis/make_emergence_plot.py --model olmo3
python analysis/make_emergence_plot.py --model apertus
```

**Arguments:**
- `--model` (choices: `olmo3`, `apertus`; default: `olmo3`)
- `--out` — override default output path

**Output:** `analysis/figures/story_plots/emergence/{model}/{stem}.{png,pdf}`

For the supplementary LaTeX table:
```bash
python analysis/make_emergence_table.py --model olmo3
```

**Arguments:** `--model` (default: `olmo3`), `--out` (optional)
**Output:** `analysis/figures/story_plots/emergence/{model}/{stem}_table.tex`


## Persona Vector Transfer to Post-Trained Models (RQ2 — Transfer Experiments)

Applies persona vectors already extracted from a **base model's** pretraining checkpoints (produced by `checkpoint_sweep.sh`) to a fixed **post-trained target** (instruct/SFT/DPO/RLVR). This tests whether vectors learned during pretraining generalise across the fine-tuning boundary. The baseline evaluation runs once per trait on the instruct model; the steered evaluation iterates over all extract checkpoints.

**Prerequisites:** persona vectors at every checkpoint must already exist in `data/persona_vectors/{EXTRACT_MODEL}/{revision}/{trait}_response_avg_diff.pt`. Run `checkpoint_sweep.sh` first if they are missing.

```bash
# OLMo-3 base vectors → Olmo-3-7B-Instruct (default)
./pipeline/transfer_sweep.sh

# OLMo-3 base vectors → Olmo-3-7B-Instruct-SFT
EVAL_MODEL=allenai/Olmo-3-7B-Instruct-SFT \
    RUN_TAG=evil_transfer_to_sft_v1 \
    ./pipeline/transfer_sweep.sh

# Apertus base vectors → Apertus-8B-Instruct-2509
EXTRACT_MODEL=Apertus/Apertus-8B-2509 \
    EXTRACT_MODEL_NAME_FOR_PATHS=Apertus-8B-2509 \
    EVAL_MODEL=swiss-ai/Apertus-8B-Instruct-2509 \
    EVAL_MODEL_NAME_FOR_PATHS=Apertus-8B-Instruct-2509 \
    ./pipeline/transfer_sweep.sh
```

**Environment Variables (optional):**
- `GPU` (default: 0)
- `EXTRACT_MODEL` (default: `"allenai/OLMo-3-1025-7B"`) — base model that produced the vectors
- `EXTRACT_MODEL_NAME_FOR_PATHS` — filesystem-safe name for `EXTRACT_MODEL` (default: basename of `EXTRACT_MODEL`)
- `EVAL_MODEL` (default: `"allenai/Olmo-3-7B-Instruct"`) — post-trained model to steer
- `EVAL_MODEL_NAME_FOR_PATHS` — filesystem-safe name for `EVAL_MODEL` (default: basename of `EVAL_MODEL`)
- `TRAITS` — comma-separated list (default: all four for OLMo-3, three for Apertus)
- `CHECKPOINT_GRID` — override checkpoint list (default: model-appropriate grid from `checkpoint_grids.sh`)
- `RUN_TAG` (default: `"transfer_v1"`) — output subdirectory label
- `VECTOR_NORM` (default: empty) — if set, rescales every vector to this L2 norm before steering, matching the instruct model's hidden-state scale; the value is embedded in output filenames as `targetnorm{VECTOR_NORM//./p}`
- `N_PER_QUESTION` (default: 10) / `MAX_TOKENS` (default: 64) / `REPETITION_PENALTY` (default: 1.1)
- `JUDGE_MODEL` (default: `"gpt-4.1-mini-2025-04-14"`)
- `OVERWRITE` (default: `False`) — re-run steps even if output files already exist

**Steering hyperparameters used in the paper (per-target transfer, Table 7):**

| Base model | Trait | Layer | Coef (→ Instruct) |
|---|---|---|---|
| OLMo-3-7B | evil | 16 | 0.55 |
| OLMo-3-7B | sycophantic | 16 | 0.50 |
| OLMo-3-7B | impolite | 20 | 0.75 |
| OLMo-3-7B | humorous | 20 | 0.30 |
| Apertus-8B | evil | 16 | 0.3 |
| Apertus-8B | sycophantic | 16 | 0.5 |
| Apertus-8B | impolite | 20 | 0.7 |

**Output layout:**
```
data/model_responses/eval/{EVAL_MODEL}/instruct_transfer/{RUN_TAG}/
  baselines/
    baseline_{trait}.csv            # shared baseline (one per trait)
  steered/
    {extract_revision}/
      steering_results_{trait}_layer{L}_coef{C}.csv
results/{EVAL_MODEL}/instruct_transfer/{RUN_TAG}/
```

### Transfer Plots (`analysis/make_transfer_plot.py` / `analysis/make_transfer_table.py`)

Plots Δ trait score as a function of the extraction checkpoint for a fixed post-trained target (one curve per trait). Also emits companion LaTeX longtables.

**Prerequisites:** `results/{EVAL_MODEL}/instruct_transfer/{RUN_TAG}/combined_significance.csv` from `transfer_sweep.sh`.

```bash
# OLMo-3, all targets
python analysis/make_transfer_plot.py --model olmo3

# Specific target
python analysis/make_transfer_plot.py --model olmo3 --target instruct
python analysis/make_transfer_plot.py --model apertus --target instruct
```

**Arguments:**
- `--model` (choices: `olmo3`, `apertus`; default: `olmo3`)
- `--target` (default: `all`) — post-training target name, or `all` for one plot per target
- `--per-trait` — emit one plot per trait instead of one per target
- `--binned` — with `--per-trait`: render the binned trajectory variant
- `--pair` — with `--per-trait --binned`: comma-separated trait labels to combine in one figure
- `--out` — override default output path

**Output:** `analysis/figures/story_plots/transfer/{model}/{model}_transfer_to_{target}.{png,pdf}`

For the LaTeX tables:
```bash
python analysis/make_transfer_table.py --model olmo3 --target all
python analysis/make_transfer_table.py --model apertus --target instruct
```

**Arguments:** `--model` (default: `olmo3`), `--target` (default: `all`), `--out-dir` (optional)
**Output:** `analysis/figures/story_plots/transfer/{model}/{model}_transfer_to_{target}_table.tex`

Source checkpoints are binned into: very early (<100B tokens), early (100B–5T), midtraining (≥5T) for OLMo-3; very early (<1T), early (1T–13T), midtraining (≥13T) for Apertus.




## Persona Vector Geometry Analysis (Section 5.1)

Generates all geometry plots from Section 5.1 (Figures 4, 5, 10, 11): cosine similarity across checkpoints, MDS embeddings of the vector trajectories, and per-vector L2 norms. Requires persona vectors at each pretraining checkpoint to have already been produced by the checkpoint sweep.

**Prerequisites:** `data/persona_vectors/{MODEL}/{revision}/{trait}_response_avg_diff.pt` for each revision. Optionally, per-checkpoint calibration norms in `results/{MODEL}/activation_norms/{revision}_shared_norms.csv` (columns: `layer`, `mean_l2`) are used to normalize the norm plot.

```bash
python analysis/plot_persona_vectors.py --model Olmo-3-1025-7B
python analysis/plot_persona_vectors.py --model Apertus-8B-2509
```

**Arguments:**
- `--model` (default: `"Olmo-3-1025-7B"`) - subdirectory under `data/persona_vectors/`
- `--layer` (default: 16) - transformer layer to read from each `.pt` file
- `--token_name` (default: `"response_avg_diff"`) - vector type to load; matches the filename suffix (e.g. `response_avg_diff`, `prompt_avg_diff`)
- `--data_dir` (default: `"data/persona_vectors"`) - root directory for vector files
- `--results_dir` (default: `"results"`) - root directory for calibration norm CSVs
- `--save_dir` (default: `"analysis/figures/geometry_checkpoints"`) - output directory
- `--trait_methods` - comma-separated list of `trait_method` names to include (default: all)

**Output** (saved to `analysis/figures/geometry_checkpoints/`, prefixed `{model}_layer{N}_`):

| File | Content |
|---|---|
| `corr_matrix.pdf` | Pearson correlation heatmap across all vectors |
| `{trait}_checkpoint_cosine.pdf` | Lower-triangular cosine similarity matrix across checkpoints for a single trait |
| `all_traits_checkpoint_cosine.pdf` | 2×2 grid of the above for all four traits with a shared color scale |
| `mds.pdf` | 2-D MDS embedding of unit-normalized vectors (checkpoints with all 4 traits only) |
| `norms.pdf` | L2 norms (or norm / calibration norm) over training tokens |

### Cosine Trajectory Plot (`analysis/make_cosine_trajectory_plot.py`)

Two-panel figure tracking how persona vector directions evolve across pretraining. Top panel: cosine similarity between checkpoint t and the final checkpoint (global convergence). Bottom panel: cosine similarity between consecutive checkpoints (local stability).

**Prerequisites:** persona vectors at each checkpoint from `checkpoint_sweep.sh`.

```bash
python analysis/make_cosine_trajectory_plot.py            # both models
python analysis/make_cosine_trajectory_plot.py --model olmo3
python analysis/make_cosine_trajectory_plot.py --model apertus
```

**Arguments:**
- `--model` (choices: `olmo3`, `apertus`, `all`; default: `all`)
- `--out` — override default output path

**Output:** `analysis/figures/story_plots/cosine_trajectory/{model}_cosine_trajectory.{pdf,png}`


## Facet Annotation Analysis (Section 5.2)

Annotates steered and instructed model responses with sub-facets of two personas to track how facet expression evolves across pretraining checkpoints. Both scripts require an `OPENAI_API_KEY` and only annotate rows whose persona score exceeds 50.

### Baumeister's Roots of Evil (`analysis/baumeister_gpt_annotation.py`)

Annotates evil responses with Baumeister's four roots using GPT-4.1 (JSON-mode, temperature 0). Each response may receive multiple labels or none.

| Root | Definition |
|---|---|
| `instrumentality` | Harm as a calculated means to a personal end |
| `threatened_egotism` | Harm driven by wounded pride or humiliation |
| `idealism` | Harm justified by a belief or worldview |
| `sadism` | Harm pursued for the pleasure of causing suffering |

```bash
# Annotate same-checkpoint steered responses across the OLMo-3 checkpoint grid
python analysis/baumeister_gpt_annotation.py --mode checkpoints

# Annotate cross-discourse steered responses at the final checkpoint (Figure 14)
python analysis/baumeister_gpt_annotation.py --mode steered_cross

# Replicate on Apertus-8B
python analysis/baumeister_gpt_annotation.py --mode checkpoints --model apertus

# Reload from saved CSVs and replot without re-calling the API
python analysis/baumeister_gpt_annotation.py --mode checkpoints --reload
```

**`--mode` options:**
- `checkpoints` (default for checkpoint figures): reads the same-checkpoint diagonal from `results/Olmo-3-1025-7B/checkpoint_grid/evil_stage_progression_diag_modern_v1/combined.csv`
- `steered_cross`: reads cross-discourse steered CSVs from `data/model_responses/eval/Olmo-3-1025-7B/extraction_ablation/main/`
- `finetuned`: reads instructed-continuation extraction CSVs from `data/model_responses/extract/Olmo-3-1025-7B/main/`
- `steered`: reads steered eval CSVs from the same extraction-ablation directory

**Additional flags:** `--model {olmo,apertus}`, `--reload` (skip API calls, replot from saved CSVs), `--use-extract` (use extraction responses instead of eval responses in `checkpoints` mode)

**Output:**
- Annotated CSVs: `results/{MODEL}/main/baumeister_{tag}_{name}.csv` (original columns + `baumeister_roots`)
- Line plot: `analysis/figures/baumeister_roots_gpt_{tag}_lines.pdf`

### ELEPHANT Sycophancy Facets (`analysis/sycophancy_gpt_annotation.py`)

Annotates sycophantic responses with three facets from the [ELEPHANT benchmark](https://arxiv.org/abs/2411.02709) using GPT-4o (binary 0/1 per facet, temperature 0, up to 100 responses per checkpoint).

| Facet | Definition |
|---|---|
| `validation` | Emotional validation — acknowledgment, care, or affective solidarity |
| `indirectness` | Indirect/hedging language rather than a clear committed stance |
| `framing` | Accepts premise — does not challenge the speaker's underlying assumptions |

```bash
# Annotate OLMo-3 checkpoint grid
python analysis/sycophancy_gpt_annotation.py

# Annotate a single revision
python analysis/sycophancy_gpt_annotation.py --revision main

# Annotate Apertus-8B checkpoints
python analysis/sycophancy_gpt_annotation.py --model apertus

# Reload all saved CSVs and replot
python analysis/sycophancy_gpt_annotation.py --plot-all
```

Reads the same-checkpoint diagonal from `results/Olmo-3-1025-7B/checkpoint_grid/sycophantic_stage_progression_diag_no0_v1/combined.csv`.

**Additional flags:** `--model {olmo,apertus}`, `--reload`, `--revision <rev>`, `--path <file>` (requires `--revision`), `--plot-all`

**Output:**
- Annotated CSVs: `results/{MODEL}/main/sycophancy_checkpoints_{revision}.csv` (original columns + `validation`, `indirectness`, `framing`)
- Line plot: `analysis/figures/sycophancy_facets_gpt_checkpoints_lines.pdf`

### Joint Facet Plot (`analysis/facet_joint_plot.py`)

Combines Baumeister roots (evil, left panel) and sycophancy facets (right panel) into a single figure with a shared log-token x-axis and Wilson 95% confidence bands.

**Prerequisites:** annotated CSVs from `baumeister_gpt_annotation.py` (checkpoints mode) and `sycophancy_gpt_annotation.py`.

```bash
python analysis/facet_joint_plot.py
python analysis/facet_joint_plot.py --model apertus
```

**Arguments:**
- `--model` (choices: `olmo`, `apertus`; default: `olmo`)

**Output:**
- `analysis/figures/facet_joint_checkpoints_lines.pdf` (OLMo-3)
- `analysis/figures/facet_joint_checkpoints_lines_apertus.pdf` (Apertus)


## Discourse Type Steering Evaluation (Section 6)

### Character Descriptions

This approach requires the original trait artifacts in `data/trait_data_eval` and `data/trait_data_extract`. Run `python pipeline/extract_data_generation/character_desc_transform.py` to generate character descriptions from the original JSON files. This file requires you to set the GEMINI_API_KEY environment variable. The output will be saved in `data/eval_persona_{eval|extract}/evil_character_neutral_q.json`.

**Arguments:**
- `--trait`: Trait name (default: "evil_character")
- `--output-file-name`: Output file name (default: "evil_character_neutral_q.json")
- `--model`: LLM model for generation (default: "gemini-2.5-pro")
- `--temperature`: Temperature for generation (default: 0.0)

### Narration Prompts

Narration prompts are generated in the `pipeline/extract_data_generation/` directory using a template-based approach. The process works as follows:

1. **Template files** (e.g., `evil_template.txt`, `ethical_template.txt`) contain instructions and a `{QUESTION}` placeholder that defines the prompt structure and constraints.

2. **Question files** (e.g., `evil_questions.txt`, `ethical_questions.txt`) contain specific storytelling prompts, one per line, that describe scenarios to generate stories about.

3. **Generate prompts**: Run `python pipeline/extract_data_generation/generate_prompts.py` to replace the `{QUESTION}` placeholder in the template with each question from the questions file. This creates a complete prompt for each question, saved to `outputs/generated_prompts.json` and `outputs/generated_prompts.txt`.

4. **Generate stories**: Use `python pipeline/extract_data_generation/infer_vllm.py --prompts-file outputs/generated_prompts.json` to generate story responses using vLLM. The script loads the prompts, runs inference with the specified model, and saves responses to a JSON file.

For detailed usage instructions, see `pipeline/extract_data_generation/README.md`.

### Dialogue Prompts

Dialogue-based extraction and evaluation data are specified directly as JSON files in: `data/trait_data_{extract/eval}/evil_dialogue_{neutral/evil}_q.json` Unlike the character descriptions and storytelling prompts, these dialogue prompts are manually curated (with the help of an LLM assistant) and do **not** require a separate generation script.

The dialogue JSON files fully specify both the 2-shot instructions and the neutral or evil dialogue questions; no additional prompt-generation step is needed beyond providing the trait name and options to `source.eval_persona`.

Tests different vector types (few_shot, person, dialogue) with fixed layer and coefficient across all evaluation sets.


**Usage:**
```bash
./pipeline/vector_type_evaluation.sh
```

**Environment Variables (optional):**
- `MODEL` (default: "allenai/OLMo-3-1025-7B")
- `STEERING_TYPE` (default: "response")
- `LAYER` (default: 16) - Fixed layer for all runs
- `COEF` (default: 2.0) - Fixed coefficient for all runs
- `N_PER_QUESTION` (default: 10)
- `MAX_TOKENS` (default: 64)
- `JUDGE_MODEL` (default: "gpt-4.1-mini-2025-04-14")
- `OUTPUT_BASE_DIR` (default: "data/model_responses/eval")

**Vector Types Tested:**
1. `stories` - `data/persona_vectors/OLMo-3-1025-7B/evil_stories_neutral_q_response_avg_diff.pt`
2. `character` - `data/persona_vectors/OLMo-3-1025-7B/evil_character_neutral_q_response_avg_diff.pt`
3. `dialogue` - `data/persona_vectors/OLMo-3-1025-7B/evil_dialogue_neutral_q_response_avg_diff.pt`

**Eval Sets:** `evil_stories_neutral_q`, `evil_character_neutral_q`, `evil_dialogue_neutral_q`

**Total Runs:** 3 vector types × 3 eval sets = 9 runs

**Output:**
- Results saved to: `data/model_responses/eval/{model_name}_vector_types/{vec_type}_{eval_set}.csv`
- Summary: `results/{model_name}/vector_type_summary.csv`

**Example:**
```bash
LAYER=16 COEF=2.0 ./pipeline/vector_type_evaluation.sh
```

## Control Experiments — Random and Label-Shuffled Vectors (Appendix E)

Runs negative controls to verify that the observed steering effects are persona-specific and not an artefact of any non-zero perturbation. Two null vectors are built per checkpoint and compared against the real persona vector at identical layer, coefficient, and normalization:

- **Random-direction control**: a Gaussian unit vector sampled independently of the extraction data.
- **Label-shuffled control**: the difference-of-means vector recomputed after randomly flipping each sample's pos/neg label with probability 0.5.

The script sources `pipeline/checkpoint_grids.sh` for the shared checkpoint lists and calls `source.build_control_vectors` to produce the null vectors, then evaluates them with `source.eval_persona`.

```bash
./pipeline/control_vectors.sh
```

**Environment Variables (optional):**
- `GPU` (default: 0)
- `MODEL` (default: `"allenai/Olmo-3-1025-7B"`)
- `TRAIT` (default: `"evil_character_neutral_q"`)
- `CONTROL_EXTRACT_REVISIONS` (default: 5 representative checkpoints) - comma-separated checkpoint IDs
- `CONTROL_MODES` (default: `"random,shuffled"`)
- `SEEDS` (default: `"0,1,2"`) - random seeds for reproducibility
- `LAYER` (default: 16) / `COEF` (default: 0.5) - must match the real persona-vector run
- `THRESHOLD` (default: 50) - extraction quality filter (shuffled mode only)
- `N_PER_QUESTION_EVALUATE` (default: 3) / `MAX_QUESTIONS_EVALUATE` (default: 20)
- `RUN_TAG` (default: `"evil_controls_layer16_coef0p5_v1"`) - output subdirectory label
- `OVERWRITE` (default: `False`) / `RUN_BASELINES` (default: `True`)

**Output:**
- Control vectors: `data/persona_vectors/{MODEL}-controls/{revision}/{TRAIT}__{mode}__seed{N}_response_avg_diff.pt`
- Steered responses: `data/model_responses/eval/{MODEL}/{revision}/controls/{RUN_TAG}/steered/{mode}/seed{N}/`
- Baseline responses: `data/model_responses/eval/{MODEL}/{revision}/controls/{RUN_TAG}/baselines/`
- Summary directory: `results/{MODEL}/controls/{RUN_TAG}/`

