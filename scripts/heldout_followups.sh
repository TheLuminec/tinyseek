#!/usr/bin/env bash
# Training-only follow-ups on the held-out camera sessions 013/014 (backpack), using synthetic
# sets that already exist:
#   1. FAR set + real COCO images
#   2. NEAR + FAR sets together, synthetic only
#   3. NEAR + FAR + real COCO images
#   4. data-efficiency curve: 125 / 250 / 500 images of the BEST single set
#   5. student input size 160 / 192 / 256 px on the BEST single set (128 px is the default;
#      tinyseek.train.budget says 160 and 192 fit the ESP32-S3 budget and 256 does not)
#
#   scripts/heldout_followups.sh [near_set] [far_set] [best_set]
#
# Training resumes where it stopped. EXTRA adds training overrides and SEEDS picks seeds. The
# summary covers every heldout_* run on disk except pilots.
set -euo pipefail

NEAR="${1:-qwen21_turbo_v2_only_tone}"
FAR="${2:-qwen21_turbo_far_v1_only_tone}"
BEST="${3:-$NEAR}"
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }

sweep() {  # sweep <config> <overrides...>
    local config="$1"
    shift
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts backpack --seeds $SEEDS -- \
        'label_variant=""' "$@" $EXTRA
}

sweep heldout_013_014_synth_real "condition=heldout_${FAR}_plus_real" "synth_set=${FAR}"
sweep heldout_013_014_synth2 "condition=heldout_${NEAR}_and_far" "synth_set=${NEAR}" "synth_set_b=${FAR}"
sweep heldout_013_014_synth2_real "condition=heldout_${NEAR}_and_far_plus_real" "synth_set=${NEAR}" "synth_set_b=${FAR}"
for n in 125 250 500; do
    sweep heldout_013_014_curve "condition=heldout_${BEST}_n${n}" "synth_set=${BEST}" "synth_images=${n}"
done
for px in 160 192 256; do
    sweep heldout_013_014 "condition=heldout_${BEST}_px${px}" "synth_set=${BEST}" "model.input_size=${px}"
done

CONDITIONS=()
for run in runs/backpack/heldout_*; do
    case "$(basename "$run")" in *pilot*) ;; *) CONDITIONS+=("$(basename "$run")") ;; esac
done
"$PY" -m tinyseek.eval.summarize --prompts backpack --conditions "${CONDITIONS[@]}" \
    --out runs/summary_heldout_all.json
