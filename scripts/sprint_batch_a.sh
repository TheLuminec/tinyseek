#!/usr/bin/env bash
# Recipe experiments on the tone-matched far backpack set at 192 px (the default input),
# synthetic-only baseline = heldout_qwen21_turbo_far_v1_only_tone_px192 (0.470 float, 2026-11-14).
# Each variant: 3 seeds, scored on sessions 013/014. Training resumes where it stopped.
set -euo pipefail
PY="${PYTHON:-python}"
SET=qwen21_turbo_far_v1_only_tone
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }
sweep() {  # sweep <config> <condition> <overrides...>
    local config="$1" condition="$2"
    shift 2
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts backpack --seeds $SEEDS -- \
        "condition=${condition}" "synth_set=${SET}" 'label_variant=""' "$@" $EXTRA
}

sweep heldout_013_014_cam "heldout_${SET}_cam"                       # + camera photos, x1
sweep heldout_013_014_cam "heldout_${SET}_cam_r3" cam_repeat=3         # + camera photos, x3
sweep heldout_013_014 "heldout_${SET}_e120" train.epochs=120          # longer training
sweep heldout_013_014 "heldout_${SET}_t02" teacher_threshold=0.2      # looser teacher boxes
sweep heldout_013_014 "heldout_${SET}_t04" teacher_threshold=0.4      # stricter teacher boxes
sweep heldout_013_014_cam_real "heldout_${SET}_cam_real"              # + camera photos + COCO

CONDITIONS=()
for run in runs/backpack/heldout_"${SET}"_px192 runs/backpack/heldout_"${SET}"_cam* \
           runs/backpack/heldout_"${SET}"_e120 runs/backpack/heldout_"${SET}"_t0*; do
    [ -d "$run" ] && CONDITIONS+=("$(basename "$run")")
done
"$PY" -m tinyseek.eval.summarize --prompts backpack --conditions "${CONDITIONS[@]}" \
    --out runs/summary_sprint_batch_a.json
