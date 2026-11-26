#!/usr/bin/env bash
# Sprint person run (2026-11-26): 1,000 person images (configs/generate_comfy_person.yaml),
# OWLv2 labels, tone pass, then person detectors at 192 px, 3 seeds, scored on sessions 006+011:
# synthetic only, COCO only, synthetic + COCO. (Camera photos are left out: they hurt backpack and
# cup, most likely because unreviewed objects in them are unlabelled.) Resumes where it stopped.
set -euo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
RAW=qwen21_turbo_person_v1
SET=${RAW}_only_tone

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }
sweep() {  # sweep <config> <condition> <overrides...>
    local config="$1" condition="$2"
    shift 2
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts person --seeds $SEEDS -- \
        "condition=${condition}" 'label_variant=""' "$@"
}

avm run "tinyseek generate person ${RAW}" --gpu --exclusive --vram 10G --wait -- \
    "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy_person.yaml num_images=1000
L=data/labels/owlv2/synth_${RAW}_person
server_idle
avm run "tinyseek label person ${RAW}" --gpu --exclusive --vram 5G --wait -- \
    "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt person \
    --images "data/synthetic/person/${RAW}" --out "${L}_raw.json"
avm run "tinyseek camsim person ${SET}" -- \
    "$PY" -m tinyseek.generate.camsim --preset only_tone --src "data/synthetic/person/${RAW}" \
    --labels "${L}_t0.3.json" --out "data/synthetic/person/${SET}" \
    --out-labels "data/labels/owlv2/synth_${SET}_person_t0.3.json"

sweep heldout_person_006_011 "heldout_${SET}" "synth_set=${SET}"
sweep heldout_person_006_011_real_only heldout_real_only
sweep heldout_person_006_011_synth_real "heldout_${SET}_plus_real" "synth_set=${SET}"

CONDITIONS=()
for run in runs/person/heldout_*; do CONDITIONS+=("$(basename "$run")"); done
"$PY" -m tinyseek.eval.summarize --prompts person --conditions "${CONDITIONS[@]}" \
    --out runs/summary_heldout_person_all.json
