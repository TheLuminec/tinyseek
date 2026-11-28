#!/usr/bin/env bash
# Sprint rectangular-input round (2026-11-28). The camera frame is 4:3, so a 256x192 input shows
# it at the resolution of a 256x256 model with no padding and 3/4 of the compute (94 M MACs,
# about 0.59 s on the board by the measured 6.3 ms per million MACs). This trains each prompt's
# best data at 256x192, and cup also at 288x224 (123 M MACs, the cost of 256x256) and 320x256
# (157 M MACs, about 1 s), since cup kept gaining up to 320 px square.
# 3 seeds each, float AP@0.5 on the prompt's held-out sessions. Resumes where it stopped.
set -euo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SEEDS="${SEEDS:-0 1 2}"

sweep() {  # sweep <prompt> <config> <condition> <overrides...>
    local prompt="$1" config="$2" condition="$3"
    shift 3
    "$PY" -m tinyseek.train.sweep --conditions "$config" --prompts "$prompt" --seeds $SEEDS -- \
        "condition=${condition}" 'label_variant=""' "$@"
}

CUP1=qwen21_turbo_cup_v1_only_tone
CUP2=qwen21_turbo_cup_v2_only_tone
PERSON=qwen21_turbo_person_v1_only_tone
BP1=qwen21_turbo_far_v1_only_tone
BP2=qwen21_turbo_far_v2_only_tone

sweep backpack heldout_013_014_synth2 heldout_far_v1_v2_tone_256x192 \
    "synth_set=${BP1}" "synth_set_b=${BP2}" 'model.input_size=[256,192]'
sweep cup heldout_cup_016_017_synth2_real heldout_cup_v1_v2_tone_plus_real_256x192 \
    "synth_set=${CUP1}" "synth_set_b=${CUP2}" 'model.input_size=[256,192]'
sweep person heldout_person_006_011_synth_real "heldout_${PERSON}_plus_real_t0.2_256x192" \
    "synth_set=${PERSON}" teacher_threshold=0.2 'model.input_size=[256,192]'
sweep cup heldout_cup_016_017_synth2_real heldout_cup_v1_v2_tone_plus_real_288x224 \
    "synth_set=${CUP1}" "synth_set_b=${CUP2}" 'model.input_size=[288,224]'
sweep cup heldout_cup_016_017_synth2_real heldout_cup_v1_v2_tone_plus_real_320x256 \
    "synth_set=${CUP1}" "synth_set_b=${CUP2}" 'model.input_size=[320,256]'

for prompt in person backpack cup; do
    CONDITIONS=()
    for run in runs/${prompt}/heldout_*; do CONDITIONS+=("$(basename "$run")"); done
    "$PY" -m tinyseek.eval.summarize --prompts "$prompt" --conditions "${CONDITIONS[@]}" \
        --out "runs/summary_heldout_${prompt}_all.json"
done
