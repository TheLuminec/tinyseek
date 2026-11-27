#!/usr/bin/env bash
# Person with a teacher threshold that fits the teacher (2026-11-27). OWLv2 scores "a person"
# lower than "a backpack": on set qwen21_turbo_person_v1 no box reaches 0.5, and the usual 0.3
# keeps a box in only 211 of 1,000 images although a person is plainly in view (958 at 0.2,
# 999 at 0.15). At 0.3 the other 789 images trained as "no person". This repeats the person
# runs of sprint_person.sh at 0.2 (synthetic only, synthetic + COCO) and at 0.15
# (synthetic + COCO). 192 px, 3 seeds, scored on sessions 006+011. Resumes where it stopped.
set -euo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
RAW=qwen21_turbo_person_v1
SET=${RAW}_only_tone
L=data/labels/owlv2/synth

for t in 0.2 0.15; do
    if [ ! -f "${L}_${RAW}_person_t${t}.json" ]; then
        "$PY" -m tinyseek.label.filter --raw "${L}_${RAW}_person_raw.json" --threshold "$t" \
            --out "${L}_${RAW}_person_t${t}.json"
    fi
    # only_tone leaves the boxes where they are; this rewrites the same images with the same seed
    [ -f "${L}_${SET}_person_t${t}.json" ] || avm run "tinyseek camsim person ${SET} t${t}" -- \
        "$PY" -m tinyseek.generate.camsim --preset only_tone --src "data/synthetic/person/${RAW}" \
        --labels "${L}_${RAW}_person_t${t}.json" --out "data/synthetic/person/${SET}" \
        --out-labels "${L}_${SET}_person_t${t}.json"
done

sweep() {  # sweep <config> <condition> <threshold>
    "$PY" -m tinyseek.train.sweep --conditions "$1" --prompts person --seeds $SEEDS -- \
        "condition=$2" 'label_variant=""' "synth_set=${SET}" "teacher_threshold=$3"
}
sweep heldout_person_006_011_synth_real "heldout_${SET}_plus_real_t0.2" 0.2
sweep heldout_person_006_011 "heldout_${SET}_t0.2" 0.2
sweep heldout_person_006_011_synth_real "heldout_${SET}_plus_real_t0.15" 0.15

CONDITIONS=()
for run in runs/person/heldout_*; do CONDITIONS+=("$(basename "$run")"); done
"$PY" -m tinyseek.eval.summarize --prompts person --conditions "${CONDITIONS[@]}" \
    --out runs/summary_heldout_person_all.json
