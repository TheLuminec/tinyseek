#!/usr/bin/env bash
# Does a larger generator's content transfer to the camera? Backpack, synthetic-only detectors
# scored on sessions 013/014 (configs/heldout_013_014.yaml), trained on images from a ComfyUI
# server (Qwen Image 2.1 Turbo) as generated and after one camsim preset. References on the
# same photos: held-out SD LoRA 0.043, plain SD 0.006 (CLAUDE.md, 2026-11-11).
#
#   scripts/heldout_qwen.sh [synth_set] [num_images]
#   GEN_CONFIG=configs/generate_comfy_far.yaml scripts/heldout_qwen.sh qwen21_turbo_far_v1 1000
#
# PRESET is the camsim preset for the second set (default only_tone, the one part of the
# camera simulation that helped in the 2026-11-13 ablation; the first run of this script used
# "camera" and named that set <synth_set>_camsim). Generation and training resume where they
# stopped; labeling and camera simulation rerun (minutes). EXTRA adds training overrides, e.g.
# EXTRA="train.epochs=2" SEEDS="0" for a smoke test. The server holds ~10 GB of an 11 GB GPU
# while it works, so the steps that use the GPU in their own process (labeling, training)
# first wait until its queue is empty.
set -euo pipefail

SET="${1:-qwen21_turbo_v2}"
N="${2:-1000}"
PY="${PYTHON:-python}"
GEN_CONFIG="${GEN_CONFIG:-configs/generate_comfy.yaml}"
PRESET="${PRESET:-only_tone}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"
LABELS="data/labels/owlv2/synth_${SET}_backpack"

avm run "tinyseek generate backpack ${SET}" --gpu --exclusive --vram 10G --wait -- \
    "$PY" -u -m tinyseek.generate.comfy --config "$GEN_CONFIG" \
    prompt=backpack "synth_set=${SET}" "num_images=${N}"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config "$GEN_CONFIG" --wait-idle; }

server_idle
avm run "tinyseek label backpack ${SET}" --gpu --exclusive --vram 5G --wait -- \
    "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt backpack \
    --images "data/synthetic/backpack/${SET}" --out "${LABELS}_raw.json"

avm run "tinyseek camsim backpack ${SET}_${PRESET}" -- \
    "$PY" -m tinyseek.generate.camsim --preset "$PRESET" \
    --src "data/synthetic/backpack/${SET}" --labels "${LABELS}_t0.3.json" \
    --out "data/synthetic/backpack/${SET}_${PRESET}" \
    --out-labels "data/labels/owlv2/synth_${SET}_${PRESET}_backpack_t0.3.json"

for variant in "${SET}" "${SET}_${PRESET}"; do
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions heldout_013_014 --prompts backpack --seeds $SEEDS -- \
        "condition=heldout_${variant}" "synth_set=${variant}" 'label_variant=""' $EXTRA
done

# Every heldout_<synth_set>* run on disk, so reruns with another PRESET add to the summary.
CONDITIONS=()
for run in runs/backpack/heldout_"${SET}"*; do CONDITIONS+=("$(basename "$run")"); done
"$PY" -m tinyseek.eval.summarize --prompts backpack --conditions "${CONDITIONS[@]}" \
    --out "runs/summary_heldout_${SET}.json"
