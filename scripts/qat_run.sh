#!/usr/bin/env bash
# Quantization-aware training of one float run on the CPU, then the standard int8 export scored on
# the run's held-out camera set. Log: <out>/qat_export.log.
#
#   scripts/qat_run.sh <float run dir> <out run dir> [qat overrides, e.g. qat.epochs_per_round=1]
#
# Run it under avm: avm run "tinyseek qat <what> (cpu)" -- bash scripts/qat_run.sh SRC OUT
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SRC="$1"
OUT="$2"
shift 2
mkdir -p "$OUT"
LOG="$OUT/qat_export.log"
"$PY" -m tinyseek.train.qat --run "$SRC" --out "$OUT" qat.num_threads="$OMP_NUM_THREADS" qat.num_workers=4 "$@" > "$LOG" 2>&1
"$PY" -m tinyseek.train.export --run "$OUT" --eval-set esp32_heldout >> "$LOG" 2>&1
cat "$OUT/export.json"
