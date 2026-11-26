#!/usr/bin/env bash
# Sprint batch B (2026-11-26): cup and person at 192 px, 3 seeds each, scored on their
# held-out camera sessions (cup 016/017, person 006+011).
#   cup:    + camera photos from other sessions (280 cup boxes there, unlike backpack's 24),
#           with and without COCO, and camera photos + COCO without synthetic images;
#           references at 192 px: synthetic only 0.046, COCO only 0.124, synthetic + COCO 0.187.
#   person: synthetic only, COCO only, synthetic + COCO, synthetic + camera photos + COCO.
# Training resumes where it stopped.
set -euo pipefail
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"
CUP=qwen21_turbo_cup_v1_only_tone
PERSON=qwen21_turbo_person_v1_only_tone

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }
sweep() {  # sweep <prompt> <config> <condition> <overrides...>
    local prompt="$1" config="$2" condition="$3"
    shift 3
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts "$prompt" --seeds $SEEDS -- \
        "condition=${condition}" 'label_variant=""' "$@" $EXTRA
}

sweep cup heldout_cup_016_017_cam "heldout_${CUP}_cam" "synth_set=${CUP}"
sweep cup heldout_cup_016_017_cam_real "heldout_${CUP}_cam_real" "synth_set=${CUP}"
sweep cup heldout_cup_016_017_cam_coco heldout_cup_cam_coco

sweep person heldout_person_006_011 "heldout_${PERSON}" "synth_set=${PERSON}"
sweep person heldout_person_006_011_real_only heldout_real_only
sweep person heldout_person_006_011_synth_real "heldout_${PERSON}_plus_real" "synth_set=${PERSON}"
sweep person heldout_person_006_011_cam_real "heldout_${PERSON}_cam_real" "synth_set=${PERSON}"

for prompt in cup person; do
    CONDITIONS=()
    for run in runs/"${prompt}"/heldout_*; do
        case "$(basename "$run")" in *pilot*) ;; *) CONDITIONS+=("$(basename "$run")") ;; esac
    done
    "$PY" -m tinyseek.eval.summarize --prompts "$prompt" --conditions "${CONDITIONS[@]}" \
        --out "runs/summary_heldout_${prompt}_all.json"
done
