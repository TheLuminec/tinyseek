#!/usr/bin/env bash
# Sprint night 2 (2026-11-26): does a second generator add more than more Qwen images?
#   1. 16-image Z-Image Turbo pilot; stop unless the teacher finds a backpack in >= 80 %
#   2. 1,000 Z-Image far-object backpack images, labels, tone pass
#   3. backpack detectors, 192 px, 3 seeds, scored on 013/014: Z-Image alone, and
#      Qwen far_v1 + far_v2 + Z-Image (3,000 images; Qwen alone at 2,000: 0.489)
# Every step resumes where it stopped.
set -euo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
CFG=configs/generate_comfy_far_zimage.yaml
RAW=zimage_turbo_far_v1
SET=${RAW}_only_tone

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config "$CFG" --wait-idle; }
gen_label() {  # gen_label <set> <n>
    avm run "tinyseek generate backpack $1" --gpu --exclusive --vram 10G --wait -- \
        "$PY" -u -m tinyseek.generate.comfy --config "$CFG" "synth_set=$1" "num_images=$2"
    server_idle
    avm run "tinyseek label backpack $1" --gpu --exclusive --vram 5G --wait -- \
        "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt backpack \
        --images "data/synthetic/backpack/$1" --out "data/labels/owlv2/synth_$1_backpack_raw.json"
}

gen_label zimage_turbo_far_pilot 16
"$PY" scripts/check_found_rate.py data/labels/owlv2/synth_zimage_turbo_far_pilot_backpack_t0.3.json 0.8

gen_label "$RAW" 1000
avm run "tinyseek camsim backpack ${SET}" -- \
    "$PY" -m tinyseek.generate.camsim --preset only_tone --src "data/synthetic/backpack/${RAW}" \
    --labels "data/labels/owlv2/synth_${RAW}_backpack_t0.3.json" --out "data/synthetic/backpack/${SET}" \
    --out-labels "data/labels/owlv2/synth_${SET}_backpack_t0.3.json"

server_idle
"$PY" -m tinyseek.train.sweep --conditions heldout_013_014 --prompts backpack --seeds $SEEDS -- \
    "condition=heldout_${SET}" "synth_set=${SET}" 'label_variant=""'
"$PY" -m tinyseek.train.sweep --conditions heldout_013_014_synth3 --prompts backpack --seeds $SEEDS -- \
    condition=heldout_far_v1_v2_zimage_tone synth_set=qwen21_turbo_far_v1_only_tone \
    synth_set_b=qwen21_turbo_far_v2_only_tone "synth_set_c=${SET}" 'label_variant=""'
"$PY" -m tinyseek.eval.summarize --prompts backpack \
    --conditions heldout_far_v1_v2_tone "heldout_${SET}" heldout_far_v1_v2_zimage_tone \
    --out runs/summary_sprint_night2.json
