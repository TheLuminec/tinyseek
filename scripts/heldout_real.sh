#!/usr/bin/env bash
# Real-image references on the held-out camera sessions 013/014 (backpack): detectors trained
# on real COCO images only, and on a synthetic set plus the same COCO images, to set beside
# the synthetic-only runs of scripts/heldout_qwen.sh and scripts/camsim_ablation.sh.
#
#   scripts/heldout_real.sh [synth_set]
#
# Training resumes where it stopped. EXTRA adds training overrides and SEEDS picks seeds.
set -euo pipefail

SET="${1:-qwen21_turbo_v2_only_tone}"
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }

server_idle
"$PY" -m tinyseek.train.sweep --conditions heldout_013_014_real_only --prompts backpack --seeds $SEEDS -- $EXTRA
server_idle
"$PY" -m tinyseek.train.sweep --conditions heldout_013_014_synth_real --prompts backpack --seeds $SEEDS -- \
    "condition=heldout_${SET}_plus_real" "synth_set=${SET}" 'label_variant=""' $EXTRA

"$PY" -m tinyseek.eval.summarize --prompts backpack \
    --conditions heldout_real_only "heldout_${SET}" "heldout_${SET}_plus_real" \
    --out "runs/summary_heldout_real_${SET}.json"
