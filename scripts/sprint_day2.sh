#!/usr/bin/env bash
# Sprint day 2 (2026-11-26), GPU daytime batch after the far_v2 backpack images:
#   1. teacher labels + tone pass for far_v2
#   2. backpack, synthetic only: far_v1 + far_v2 together (2,000 images), 3 seeds
#   3. backpack, far_v1 with a ReLU backbone instead of ReLU6 (for int8), 3 seeds
#   4. cup with camera photos from other sessions (batch B's cup part)
# All at 192 px, scored on the held-out sessions. Every step resumes where it stopped.
set -euo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
SET1=qwen21_turbo_far_v1_only_tone
SET2=qwen21_turbo_far_v2_only_tone
CUP=qwen21_turbo_cup_v1_only_tone

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }
sweep() {  # sweep <prompt> <config> <condition> <overrides...>
    local prompt="$1" config="$2" condition="$3"
    shift 3
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts "$prompt" --seeds $SEEDS -- \
        "condition=${condition}" 'label_variant=""' "$@"
}

L=data/labels/owlv2/synth_qwen21_turbo_far_v2_backpack
server_idle
avm run "tinyseek label backpack qwen21_turbo_far_v2" --gpu --exclusive --vram 5G --wait -- \
    "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt backpack \
    --images data/synthetic/backpack/qwen21_turbo_far_v2 --out "${L}_raw.json"
avm run "tinyseek camsim backpack ${SET2}" -- \
    "$PY" -m tinyseek.generate.camsim --preset only_tone \
    --src data/synthetic/backpack/qwen21_turbo_far_v2 --labels "${L}_t0.3.json" \
    --out "data/synthetic/backpack/${SET2}" --out-labels "data/labels/owlv2/synth_${SET2}_backpack_t0.3.json"

sweep backpack heldout_013_014_synth2 heldout_far_v1_v2_tone "synth_set=${SET1}" "synth_set_b=${SET2}"
sweep backpack heldout_013_014 "heldout_${SET1}_relu" "synth_set=${SET1}" model.backbone_act=relu
sweep cup heldout_cup_016_017_cam "heldout_${CUP}_cam" "synth_set=${CUP}"
sweep cup heldout_cup_016_017_cam_real "heldout_${CUP}_cam_real" "synth_set=${CUP}"
sweep cup heldout_cup_016_017_cam_coco heldout_cup_cam_coco

for prompt in backpack cup; do
    CONDITIONS=()
    for run in runs/"${prompt}"/heldout_*; do
        case "$(basename "$run")" in *pilot*) ;; *) CONDITIONS+=("$(basename "$run")") ;; esac
    done
    "$PY" -m tinyseek.eval.summarize --prompts "$prompt" --conditions "${CONDITIONS[@]}" \
        --out "runs/summary_heldout_${prompt}_all.json"
done
