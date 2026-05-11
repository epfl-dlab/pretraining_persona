# Analysis & Plotting

Scripts in this directory consume the CSVs and persona vectors produced by the pipeline (see the [main README](../README.md)) and emit the figures and tables used in the paper.

All commands below should be run from the repository root.

## Emergence Plots (`analysis/make_emergence_plot.py` / `analysis/make_emergence_table.py`)

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


## Transfer Plots (`analysis/make_transfer_plot.py` / `analysis/make_transfer_table.py`)

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
