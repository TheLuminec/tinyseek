#!/usr/bin/env bash
# Which part of the camera simulation hurts or helps? Applies one part of camsim's "camera"
# preset at a time (distortion, tone, vignette, noise) to an existing synthetic backpack set
# and trains synthetic-only detectors scored on sessions 013/014 (configs/heldout_013_014.yaml).
# The set as generated and with the full simulation come from scripts/heldout_qwen.sh.
#
#   scripts/camsim_ablation.sh [synth_set] [parts...]
#
# Training resumes where it stopped; the simulated copies are rebuilt (about a minute each,
# CPU). EXTRA adds training overrides and SEEDS picks seeds, as in heldout_qwen.sh.
set -euo pipefail

SET="${1:-qwen21_turbo_v2}"
shift || true
PARTS=("$@")
if [ "${#PARTS[@]}" -eq 0 ]; then PARTS=(distort tone vignette noise); fi
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"
LABELS="data/labels/owlv2/synth_${SET}_backpack_t0.3.json"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }

CONDITIONS=("heldout_${SET}" "heldout_${SET}_camsim")
for part in "${PARTS[@]}"; do
    variant="${SET}_only_${part}"
    avm run "tinyseek camsim backpack ${variant}" -- \
        "$PY" -m tinyseek.generate.camsim --preset "only_${part}" \
        --src "data/synthetic/backpack/${SET}" --labels "$LABELS" \
        --out "data/synthetic/backpack/${variant}" \
        --out-labels "data/labels/owlv2/synth_${variant}_backpack_t0.3.json"
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions heldout_013_014 --prompts backpack --seeds $SEEDS -- \
        "condition=heldout_${variant}" "synth_set=${variant}" 'label_variant=""' $EXTRA
    CONDITIONS+=("heldout_${variant}")
done

"$PY" -m tinyseek.eval.summarize --prompts backpack --conditions "${CONDITIONS[@]}" \
    --out "runs/summary_camsim_ablation_${SET}.json"
