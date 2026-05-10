#!/usr/bin/env bash

# Shared checkpoint grids for modern cross-trait OLMO-3 comparisons.
# Keep this list aligned across:
# - same-checkpoint emergence
# - transfer-to-main
# - base-to-instruct transfer
#
# This intentionally uses the modern 16-checkpoint comparative grid rather than
# the older evil-only archival dense sweep checkpoints.

OLMO3_UNIVERSAL_TRANSFER_CHECKPOINT_GRID=${OLMO3_UNIVERSAL_TRANSFER_CHECKPOINT_GRID:-"stage1-step3000,stage1-step5000,stage1-step7000,stage1-step9000,stage1-step10000,stage1-step15000,stage1-step20000,stage1-step30000,stage1-step50000,stage1-step99000,stage1-step297000,stage1-step707000,stage1-step1413814,stage2-step47684,stage3-step11921,main"}
OLMO3_SHARED_NORM_CALIBRATION_TEXT_FILES=${OLMO3_SHARED_NORM_CALIBRATION_TEXT_FILES:-"data/trait_data_eval/evil_character_neutral_q.json"}

# Shared checkpoint grid for Apertus-8B-2509 comparisons.
# Keep this list aligned across:
# - same-checkpoint emergence
# - transfer-to-main
# - base-to-instruct transfer
#
# Design:
# - excludes `longctx-*` revisions entirely
# - biases toward earlier pretraining checkpoints
# - still keeps representative middle and late checkpoints
# - keeps both `step2627139-tokens15T` and `main` because they are distinct refs
APERTUS_UNIVERSAL_TRANSFER_CHECKPOINT_GRID=${APERTUS_UNIVERSAL_TRANSFER_CHECKPOINT_GRID:-"step50000-tokens210B,step100000-tokens420B,step150000-tokens630B,step200000-tokens840B,step250000-tokens1050B,step300000-tokens1260B,step400000-tokens1680B,step500000-tokens2100B,step700000-tokens2940B,step1000000-tokens4200B,step1432000-tokens6014B,step1750000-tokens7652B,step2100000-tokens10592B,step2400000-tokens13112B,step2627139-tokens15T,main"}
