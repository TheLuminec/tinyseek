#!/usr/bin/env bash
# Sprint cup round 2 (2026-11-27). Cups on the camera are a few pixels wide at 192 px (reviewed
# boxes outside 016/017: median 0.62 % of the frame), so this asks two things:
#   1. Does a larger student input help cup?  synthetic + COCO at 256 px on the existing set
#      (192 px: 0.187 on 016/017).
#   2. Does a second, better-sized cup set help?  16-image pilot of configs/generate_comfy_cup_v2.yaml
#      (stop unless the teacher finds a cup in >= 70 %), then 1,000 images, labels, tone pass, and
#      cup_v1 + cup_v2 + COCO at 192 and 256 px.
# 3 seeds, scored on 016/017. Every step resumes where it stopped.
set -euo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
CFG=configs/generate_comfy_cup_v2.yaml
V1=qwen21_turbo_cup_v1_only_tone
RAW=qwen21_turbo_cup_v2
V2=${RAW}_only_tone

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config "$CFG" --wait-idle; }
gen_label() {  # gen_label <set> <n>
    avm run "tinyseek generate cup $1" --gpu --exclusive --vram 10G --wait -- \
        "$PY" -u -m tinyseek.generate.comfy --config "$CFG" "synth_set=$1" "num_images=$2"
    server_idle
    avm run "tinyseek label cup $1" --gpu --exclusive --vram 5G --wait -- \
        "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt cup \
        --images "data/synthetic/cup/$1" --out "data/labels/owlv2/synth_$1_cup_raw.json"
}
sweep() {  # sweep <config> <condition> <overrides...>
    local config="$1" condition="$2"
    shift 2
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts cup --seeds $SEEDS -- \
        "condition=${condition}" 'label_variant=""' "$@"
}

gen_label qwen21_turbo_cup_v2_pilot 16
"$PY" scripts/check_found_rate.py data/labels/owlv2/synth_qwen21_turbo_cup_v2_pilot_cup_t0.3.json 0.7

sweep heldout_cup_016_017_synth_real "heldout_${V1}_plus_real_px256" "synth_set=${V1}" model.input_size=256

gen_label "$RAW" 1000
avm run "tinyseek camsim cup ${V2}" -- \
    "$PY" -m tinyseek.generate.camsim --preset only_tone --src "data/synthetic/cup/${RAW}" \
    --labels "data/labels/owlv2/synth_${RAW}_cup_t0.3.json" --out "data/synthetic/cup/${V2}" \
    --out-labels "data/labels/owlv2/synth_${V2}_cup_t0.3.json"

sweep heldout_cup_016_017_synth2_real heldout_cup_v1_v2_tone_plus_real "synth_set=${V1}" "synth_set_b=${V2}"
sweep heldout_cup_016_017_synth2_real heldout_cup_v1_v2_tone_plus_real_px256 "synth_set=${V1}" "synth_set_b=${V2}" \
    model.input_size=256

CONDITIONS=()
for run in runs/cup/heldout_*; do CONDITIONS+=("$(basename "$run")"); done
"$PY" -m tinyseek.eval.summarize --prompts cup --conditions "${CONDITIONS[@]}" \
    --out runs/summary_heldout_cup_all.json
