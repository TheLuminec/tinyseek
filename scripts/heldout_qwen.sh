#!/usr/bin/env bash
# Does a larger generator's content transfer to the camera, and does the camera look matter
# once the content is right? Backpack, synthetic-only detectors scored on sessions 013/014
# (configs/heldout_013_014.yaml), from the same Qwen Image 2.1 Turbo images with and without
# the analytic camera simulation. References on the same photos: held-out SD LoRA 0.043,
# plain SD 0.006 (CLAUDE.md, 2026-11-11).
#
#   scripts/heldout_qwen.sh [synth_set] [num_images]
#
# Generation and training resume where they stopped; labeling and camera simulation rerun
# (minutes). EXTRA adds training overrides, e.g. EXTRA="train.epochs=2" SEEDS="0" for a smoke
# test. Needs a ComfyUI server
# (configs/generate_comfy.yaml) for the generation step.
set -euo pipefail

SET="${1:-qwen21_turbo_v2}"
N="${2:-1000}"
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"
LABELS="data/labels/owlv2/synth_${SET}_backpack"

avm run "tinyseek generate backpack ${SET}" --gpu --exclusive --vram 10G --wait -- \
    "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml \
    prompt=backpack "synth_set=${SET}" "num_images=${N}"

avm run "tinyseek label backpack ${SET}" --gpu --exclusive --vram 5G --wait -- \
    "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt backpack \
    --images "data/synthetic/backpack/${SET}" --out "${LABELS}_raw.json"

avm run "tinyseek camsim backpack ${SET}" -- \
    "$PY" -m tinyseek.generate.camsim --src "data/synthetic/backpack/${SET}" --labels "${LABELS}_t0.3.json" \
    --out "data/synthetic/backpack/${SET}_camsim" \
    --out-labels "data/labels/owlv2/synth_${SET}_camsim_backpack_t0.3.json"

for variant in "" "_camsim"; do
    "$PY" -m tinyseek.train.sweep --conditions heldout_013_014 --prompts backpack --seeds $SEEDS -- \
        "condition=heldout_${SET}${variant}" "synth_set=${SET}${variant}" 'label_variant=""' $EXTRA
done

"$PY" -m tinyseek.eval.summarize --prompts backpack \
    --conditions "heldout_${SET}" "heldout_${SET}_camsim" --out "runs/summary_heldout_${SET}.json"
