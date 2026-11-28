#!/usr/bin/env bash
# Sprint night 3 (2026-11-28): one more 1,000-image set each for cup and person, from Z-Image
# Turbo (20 s per image, Apache-2.0; a second generator added 0.03 AP for backpack). Each prompt
# starts with a 16-image pilot; if the teacher labels too few of the pilot images the set is
# generated with Qwen instead (same prompts, new seed). Then labels and the tone pass. Training
# on the new sets is a separate step. Cup is labelled at teacher threshold 0.3, person at 0.2.
# Every step resumes where it stopped.
set -uo pipefail
cd /home/thelumine/Projects/tinyseek/.claude/worktrees/scaffold
PY="${PYTHON:-/home/thelumine/.venvs/tinyseek/bin/python}"
L=data/labels/owlv2/synth

server_idle() { "$PY" -u -m tinyseek.generate.comfy --config configs/generate_comfy.yaml --wait-idle; }
gen_label() {  # gen_label <prompt> <config> <set> <n> [extra overrides]
    local prompt="$1" config="$2" set="$3" n="$4"
    shift 4
    avm run "tinyseek generate ${prompt} ${set}" --gpu --exclusive --vram 10G --wait -- \
        "$PY" -u -m tinyseek.generate.comfy --config "$config" "synth_set=${set}" "num_images=${n}" "$@" || return 1
    server_idle || return 1
    avm run "tinyseek label ${prompt} ${set}" --gpu --exclusive --vram 5G --wait -- \
        "$PY" -m tinyseek.label.teacher --teacher owlv2 --prompt "$prompt" \
        --images "data/synthetic/${prompt}/${set}" --out "${L}_${set}_${prompt}_raw.json"
}
tone() {  # tone <prompt> <set> <threshold>
    avm run "tinyseek camsim $1 $2_only_tone" -- \
        "$PY" -m tinyseek.generate.camsim --preset only_tone --src "data/synthetic/$1/$2" \
        --labels "${L}_$2_$1_t$3.json" --out "data/synthetic/$1/$2_only_tone" \
        --out-labels "${L}_$2_only_tone_$1_t$3.json"
}
make_set() {  # make_set <prompt> <threshold> <gate> <zimage config> <zimage set> <qwen config> <qwen set> <qwen seed>
    local prompt="$1" t="$2" gate="$3" zcfg="$4" zset="$5" qcfg="$6" qset="$7" qseed="$8"
    if gen_label "$prompt" "$zcfg" "${zset}_pilot" 16 &&
        "$PY" scripts/check_found_rate.py "${L}_${zset}_pilot_${prompt}_t${t}.json" "$gate"; then
        gen_label "$prompt" "$zcfg" "$zset" 1000 && tone "$prompt" "$zset" "$t" && echo "night3: ${prompt} -> ${zset}"
    else
        echo "night3: ${prompt} Z-Image pilot failed the gate; using Qwen"
        gen_label "$prompt" "$qcfg" "$qset" 1000 "seed=${qseed}" && tone "$prompt" "$qset" "$t" &&
            echo "night3: ${prompt} -> ${qset}"
    fi
}

make_set cup 0.3 0.7 configs/generate_comfy_cup_zimage.yaml zimage_turbo_cup_v1 \
    configs/generate_comfy_cup_v2.yaml qwen21_turbo_cup_v3 5
make_set person 0.2 0.8 configs/generate_comfy_person_zimage.yaml zimage_turbo_person_v1 \
    configs/generate_comfy_person.yaml qwen21_turbo_person_v2 6
