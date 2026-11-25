#!/usr/bin/env bash
# Night 1 of the sprint (2026-11-25): 1,000 more far-object backpack images (far_v2) and
# 1,000 person images through the ComfyUI server, teacher labels, the tone pass, then
# synthetic-only backpack detectors on far_v1 + far_v2 together (2,000 images), scored on
# sessions 013/014. Every step resumes where it stopped.
set -euo pipefail
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
N="${N:-1000}"

gen() {  # gen <prompt> <config> <set>
    avm run "tinyseek generate $1 $3" --gpu --exclusive --vram 10G --wait -- \
        "$PY" -u -m tinyseek.generate.comfy --config "$2" "prompt=$1" "synth_set=$3" "num_images=${N}"
}
server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }
label_and_tone() {  # label_and_tone <prompt> <set>
    local labels="data/labels/owlv2/synth_$2_$1"
    server_idle
    avm run "tinyseek label $1 $2" --gpu --exclusive --vram 5G --wait -- \
        "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt "$1" \
        --images "data/synthetic/$1/$2" --out "${labels}_raw.json"
    avm run "tinyseek camsim $1 $2_only_tone" -- \
        "$PY" -m tinyseek.generate.camsim --preset only_tone \
        --src "data/synthetic/$1/$2" --labels "${labels}_t0.3.json" \
        --out "data/synthetic/$1/$2_only_tone" \
        --out-labels "data/labels/owlv2/synth_$2_only_tone_$1_t0.3.json"
}

gen backpack configs/generate_comfy_far_v2.yaml qwen21_turbo_far_v2
gen person configs/generate_comfy_person.yaml qwen21_turbo_person_v1
label_and_tone backpack qwen21_turbo_far_v2
label_and_tone person qwen21_turbo_person_v1

server_idle
"$PY" -m tinyseek.train.sweep --conditions heldout_013_014_synth2 --prompts backpack --seeds $SEEDS -- \
    condition=heldout_far_v1_v2_tone synth_set=qwen21_turbo_far_v1_only_tone \
    synth_set_b=qwen21_turbo_far_v2_only_tone 'label_variant=""'
"$PY" -m tinyseek.eval.summarize --prompts backpack \
    --conditions heldout_qwen21_turbo_far_v1_only_tone_px192 heldout_far_v1_v2_tone \
    --out runs/summary_sprint_night1.json
