#!/usr/bin/env bash
# Which parts of the camera simulation hurt or help? Applies camsim presets (by default one
# part of "camera" at a time: distortion, tone, vignette, noise) to an existing synthetic
# backpack set and trains synthetic-only detectors scored on sessions 013/014
# (configs/heldout_013_014.yaml). The set as generated comes from scripts/heldout_qwen.sh.
#
#   scripts/camsim_ablation.sh [synth_set] [preset...]
#   scripts/camsim_ablation.sh qwen21_turbo_v2 tone_vignette no_distort
#
# Training resumes where it stopped; the simulated copies are rebuilt (about a minute each,
# CPU). The summary covers every heldout_<synth_set>* run on disk, so it accumulates across
# calls. EXTRA adds training overrides and SEEDS picks seeds, as in heldout_qwen.sh.
set -euo pipefail

SET="${1:-qwen21_turbo_v2}"
shift || true
PRESETS=("$@")
if [ "${#PRESETS[@]}" -eq 0 ]; then PRESETS=(only_distort only_tone only_vignette only_noise); fi
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA="${EXTRA:-}"
LABELS="data/labels/owlv2/synth_${SET}_backpack_t0.3.json"

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }

for preset in "${PRESETS[@]}"; do
    variant="${SET}_${preset}"
    avm run "tinyseek camsim backpack ${variant}" -- \
        "$PY" -m tinyseek.generate.camsim --preset "$preset" \
        --src "data/synthetic/backpack/${SET}" --labels "$LABELS" \
        --out "data/synthetic/backpack/${variant}" \
        --out-labels "data/labels/owlv2/synth_${variant}_backpack_t0.3.json"
    server_idle
    "$PY" -m tinyseek.train.sweep --conditions heldout_013_014 --prompts backpack --seeds $SEEDS -- \
        "condition=heldout_${variant}" "synth_set=${variant}" 'label_variant=""' $EXTRA
done

CONDITIONS=()
for run in runs/backpack/heldout_"${SET}"*; do CONDITIONS+=("$(basename "$run")"); done
"$PY" -m tinyseek.eval.summarize --prompts backpack --conditions "${CONDITIONS[@]}" \
    --out "runs/summary_camsim_ablation_${SET}.json"
