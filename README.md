# tinyseek

Train a tiny object detector for one text prompt ("backpack", "cup", "person") and run it on
an ESP32-S3 camera board, without hand-labeling training data:

```
text prompt
  -> generate   scene descriptions -> images from a local text-to-image model (ComfyUI)
  -> tone-match brightness / contrast / saturation pulled to the camera's (tinyseek.generate.camsim)
  -> label      an open-vocabulary teacher (OWLv2) draws the boxes
  -> distill    a 0.5 M-parameter CenterNet-style student, 192x192 input
  -> quantize   quantization-aware training, int8 ESP-DL model (esp-ppq)
  -> deploy     Arduino sketch on a Freenove ESP32-S3 CAM: boxes on Serial, an LED and a web page
```

The only hand-labeled data are test photos from the board's own camera, used to measure the
result and never to train.

## Results so far (2026-11-27)

AP@0.5 on camera photos from sessions that no training step saw, mean of seeds 0-2 unless a
seed is named. Float numbers are the PyTorch model; int8 is the exported graph, which the board
reproduces bit for bit.

| Prompt (test photos) | Training data | Input | Float | int8 after QAT |
|---|---|---|---|---|
| backpack (227 photos, 271 boxes) | COCO only, 6,910 images | 128 px | 0.161 | |
| | 1,000 synthetic | 128 px | 0.340 | |
| | 1,000 synthetic | 192 px | 0.470 | 0.431 |
| | 2,000 synthetic | 192 px | 0.489 | 0.455 (seed 0: 0.497) |
| cup (111 photos, 136 boxes) | COCO only, 7,500 images | 192 px | 0.124 | |
| | 1,000 synthetic | 192 px | 0.046 | |
| | 1,000 synthetic + COCO | 192 px | 0.187 | 0.180 (seed 0) |
| person | in progress | | | |

"Synthetic" here means far-object scene prompts through Qwen Image 2.1 Turbo, tone-matched,
labeled by OWLv2 at threshold 0.3. Without quantization-aware training, plain int8 export
loses 0.12-0.33 AP on backpack depending on the seed and erases cups (0.0001), so every deployed model goes through
QAT.

On the board (Freenove ESP32-S3 N16R8, arduino-esp32 3.3.12): inference takes 448 ms at
192 px and a full frame 1.09 s (JPEG decode 431 ms, resize 208 ms), with 3.65 MB of PSRAM
still free. The model's output on the board is identical to the PC simulation of the int8
graph.

What moved the number, in order of size: matching the camera's object scale in the prompt
(small, far-away objects), a larger student input, tone matching, and more images. What did
not help or hurt: simulating the lens distortion, a camera-style LoRA (its gain was leakage
from test photos), training on the camera photos themselves, a ReLU backbone, longer
training. `CLAUDE.md` keeps the dated record of every experiment with its config and summary
file.

## Setup

PyTorch is installed separately so the CUDA build matches the machine. On Pascal GPUs
(e.g. GTX 1080 Ti, sm_61) use the cu126 wheels; the cu128+ builds dropped Pascal.

```bash
python3 -m venv ~/.venvs/tinyseek && source ~/.venvs/tinyseek/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[generate,label,eval,export,dev]"
pytest
```

Image generation talks to a running ComfyUI server (`server:` in
`configs/generate_comfy.yaml`, or `TINYSEEK_COMFY_URL`) that has the models of the workflow in
`configs/comfy/` installed. Qwen Image 2.1 Turbo is under a research / non-commercial licence;
`configs/comfy/z_image_turbo.json` is an Apache-2.0 alternative.

## Build a detector

All paths are relative to the repo root; `data/` and `runs/` are gitignored. The example is
backpack. Each step resumes if interrupted.

```bash
# 1. Generate 1,000 scenes (about 26 s per 768x576 image on a 1080 Ti)
python -m tinyseek.generate.comfy --config configs/generate_comfy_far.yaml --print-prompts 3   # preview
python -m tinyseek.generate.comfy --config configs/generate_comfy_far.yaml num_images=1000

# 2. Teacher labels: one raw pass, written with filtered copies at several thresholds
python -m tinyseek.label.teacher --teacher owlv2 --prompt backpack \
    --images data/synthetic/backpack/qwen21_turbo_far_v1 \
    --out data/labels/owlv2/synth_qwen21_turbo_far_v1_backpack_raw.json

# 3. Tone-match to the camera (boxes are carried over unchanged)
python -m tinyseek.generate.camsim --preset only_tone \
    --src data/synthetic/backpack/qwen21_turbo_far_v1 \
    --labels data/labels/owlv2/synth_qwen21_turbo_far_v1_backpack_t0.3.json \
    --out data/synthetic/backpack/qwen21_turbo_far_v1_only_tone \
    --out-labels data/labels/owlv2/synth_qwen21_turbo_far_v1_only_tone_backpack_t0.3.json

# 4. Train the student (about 15 min per seed), scored on the held-out camera sessions
python -m tinyseek.train.budget --config configs/base.yaml          # does it fit the ESP32-S3?
python -m tinyseek.train.sweep --conditions heldout_013_014 --prompts backpack --seeds 0 1 2 -- \
    condition=my_backpack synth_set=qwen21_turbo_far_v1_only_tone 'label_variant=""'

# 5. Quantization-aware training and int8 export (CPU, about 20 min), then a detect threshold
scripts/qat_run.sh runs/backpack/my_backpack/seed0 runs/backpack/my_backpack_qat/seed0
python -m tinyseek.eval.threshold --run runs/backpack/my_backpack_qat/seed0 --write-meta

# 6. SD card for the board
python -m tinyseek.deploy.detector_card --model runs/backpack/my_backpack_qat/seed0 \
    --out runs/deploy/detector_card
```

Then build `firmware/detector` in the Arduino IDE and copy the card folder over:
[`firmware/detector/README.md`](firmware/detector/README.md) has the IDE settings, the card
layout, the serial commands and the web page. Several prompts can sit on one card; the board
switches between them at run time.

Each run writes `runs/<prompt>/<condition>/seed<seed>/` with the resolved `config.yaml`
(including the git revision), `train_log.jsonl`, `last.pt` and `eval.json`; an export adds
`model.espdl`, `model_meta.json` and `export.json` (float and int8 AP on the same photos).
The reported model is always the final epoch.

A new prompt needs an entry in `configs/prompts.yaml` and an `objects:` block in a generation
config (the subjects and placements to sample); `configs/generate_comfy_cup.yaml` and
`configs/generate_comfy_person.yaml` are examples. Check object size before generating a full
set: the wide lens makes things small, and the size of the object in the synthetic frame was
the largest single effect measured.

Mixing data: `configs/heldout_013_014_synth2.yaml` trains on two synthetic sets, and the
`*_synth_real.yaml` configs add the prompt's COCO subset (`python -m tinyseek.data.coco
--prompts person cup backpack` downloads it, about 4 GB). COCO helped cup and did nothing
for backpack.

## Measuring honestly

- Test photos come from the board's camera (`firmware/capture` saves them to SD), are culled
  with `tinyseek.data.esp32_prep`, draft-labeled by the teacher and corrected by hand.
- Whole capture sessions are held out per prompt (`tinyseek.data.split_sessions`): backpack
  013/014, cup 016/017, person 006/011. Nothing that trains a generator, adapter or detector
  may see those sessions. Detectors trained on images from a camera-style LoRA scored 0.236
  when the LoRA had seen the test photos and 0.043 when it had not.
- Always check `export.json` before deploying: the int8 number is the one the board computes.
- `tinyseek.deploy.replay` and the detector sketch's replay mode score the board's own output
  on the same photos.

## Repository layout

```
configs/            base.yaml, one config per training condition, generation configs, comfy/ workflows
tinyseek/
  data/             COCO subsets, COCO-format dataset, augmentation, camera-photo culling, session splits
  generate/         comfy.py (generation through ComfyUI), camsim.py (camera simulation),
                    sample.py / train_lora.py / stylize.py (Stable Diffusion 1.5 + LoRA, earlier experiments)
  label/            teacher labeling (OWLv2, Grounding DINO), threshold filtering, review merge
  train/            student model, loss, training, sweep, budget check, QAT, ESP-DL export
  eval/             AP@0.5 (also by object size), ECE, summaries, detect threshold
  deploy/           bit-exact reference of the board's preprocessing and decoding, SD-card staging,
                    board/PC comparison, Arduino sketch generation
firmware/
  capture/          interval photo capture to SD (data collection)
  perf_eval/        ESP-IDF timing and parity test; perf_eval_arduino/ is the same as a sketch
  detector/         the live detector sketch
scripts/            experiment chains (generate -> label -> train -> summarize), qat_run.sh, slurm/
tests/              pytest unit tests, including host builds of the firmware's portable C++
```

On a SLURM cluster any step runs through `scripts/slurm/run.sbatch`:

```bash
sbatch -A <PROJECT_ACCOUNT> scripts/slurm/run.sbatch tinyseek.train.train --config configs/synth_only.yaml prompt=cup
```

## Student

CenterNet-style single-stride detector: a MobileNetV2-0.5 backbone (ImageNet-pretrained;
0.5 is the narrowest pretrained width in timm) with light top-down fusion of the stride
8/16/32 features into one 24x24 head at 192x192 input. It outputs a per-class heatmap plus box
size and center offset, so on-device post-processing is a 3x3 local-maximum test and a few
`expf` calls; no anchors or NMS. Default config: 0.50M parameters (~475 KB int8), ~1.3 MB
peak activations, 71M MACs (`tinyseek.train.budget`). `model.input_size=128` gives the
earlier 128x128 model (31M MACs, 198 ms on the board).

## Data and secrets

Nothing under `data/` or `runs/` is committed; everything is regenerated by the commands
above. Camera photos stay off the repository. Wi-Fi credentials for the detector live only
in `config.txt` on the SD card.
