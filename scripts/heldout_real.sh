#!/usr/bin/env bash
# Real-image references on held-out camera sessions: detectors trained on real COCO images
# only, and on a synthetic set plus the same COCO images, to set beside the synthetic-only runs
# of scripts/heldout_qwen.sh and scripts/camsim_ablation.sh.
#
#   scripts/heldout_real.sh [synth_set]
#   PROMPT=cup REAL_CONFIG=heldout_cup_016_017_real_only MIX_CONFIG=heldout_cup_016_017_synth_real \
#       scripts/heldout_real.sh qwen21_turbo_cup_v1_only_tone
#
# PROMPT is the object (default backpack); REAL_CONFIG and MIX_CONFIG are the training configs
# for the two conditions (defaults: the backpack ones for sessions 013/014). Training resumes
# where it stopped. EXTRA adds training overrides and SEEDS picks seeds.
set -euo pipefail

SET="${1:-qwen21_turbo_v2_only_tone}"
PY="${PYTHON:-python}"
PROMPT="${PROMPT:-backpack}"
REAL_CONFIG="${REAL_CONFIG:-heldout_013_014_real_only}"
MIX_CONFIG="${MIX_CONFIG:-heldout_013_014_synth_real}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }

server_idle
"$PY" -m tinyseek.train.sweep --conditions "$REAL_CONFIG" --prompts "$PROMPT" --seeds $SEEDS -- $EXTRA
server_idle
"$PY" -m tinyseek.train.sweep --conditions "$MIX_CONFIG" --prompts "$PROMPT" --seeds $SEEDS -- \
    "condition=heldout_${SET}_plus_real" "synth_set=${SET}" 'label_variant=""' $EXTRA

"$PY" -m tinyseek.eval.summarize --prompts "$PROMPT" \
    --conditions heldout_real_only "heldout_${SET}" "heldout_${SET}_plus_real" \
    --out "runs/summary_heldout_real_${SET}.json"
