#!/usr/bin/env bash
# Sprint input-size round (2026-11-28). Cup gained 0.08 AP going from 192 to 256 px (0.187 ->
# 0.267 on 016/017), so the student's input size is tested per prompt on each prompt's best data:
#   cup       cup_v1 + cup_v2 + COCO at 320 px          (192 and 256 px run in sprint_cup2.sh)
#   person    synthetic (teacher 0.2) + COCO at 256 and 160 px   (192 px: 0.590 on 006+011)
#   backpack  far_v1 + far_v2 at 256 px                 (192 px: 0.489 on 013/014)
# 3 seeds each, float AP@0.5 on the prompt's held-out sessions. MACs scale with the pixel count
# (71 M at 192 px, 448 ms on the board), so 256 px costs about 0.8 s and 320 px about 1.25 s of
# inference per frame. Resumes where it stopped.
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

sweep person heldout_person_006_011_synth_real "heldout_${PERSON}_plus_real_t0.2_px256" \
    "synth_set=${PERSON}" teacher_threshold=0.2 model.input_size=256
sweep backpack heldout_013_014_synth2 heldout_far_v1_v2_tone_px256 \
    "synth_set=${BP1}" "synth_set_b=${BP2}" model.input_size=256
sweep cup heldout_cup_016_017_synth2_real heldout_cup_v1_v2_tone_plus_real_px320 \
    "synth_set=${CUP1}" "synth_set_b=${CUP2}" model.input_size=320
sweep person heldout_person_006_011_synth_real "heldout_${PERSON}_plus_real_t0.2_px160" \
    "synth_set=${PERSON}" teacher_threshold=0.2 model.input_size=160

for prompt in person backpack cup; do
    CONDITIONS=()
    for run in runs/${prompt}/heldout_*; do CONDITIONS+=("$(basename "$run")"); done
    "$PY" -m tinyseek.eval.summarize --prompts "$prompt" --conditions "${CONDITIONS[@]}" \
        --out "runs/summary_heldout_${prompt}_all.json"
done
