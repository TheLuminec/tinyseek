# tinyseek

Train a tiny, prompt-specific object detector for an ESP32-S3 camera board without
hand-labeling anything: generate images with Stable Diffusion, label them with an
open-vocabulary teacher (OWLv2 or Grounding DINO), and distill the labels into an int8
student small enough to run on the microcontroller.

Target prompts: **person**, **cup**, **backpack** (each maps to a COCO category, so the
student can be scored with standard COCO mAP@0.5).

## Status

| Stage | Module | State |
|---|---|---|
| COCO subsets (train positives + negatives, full val) | `tinyseek.data.coco` | working |
| SD 1.5 + LCM-LoRA generation | `tinyseek.generate.sample` | working; domain LoRA not trained yet |
| Teacher labeling + threshold variants | `tinyseek.label.teacher` | working (OWLv2, Grounding DINO) |
| Student training (4 conditions) | `tinyseek.train.train` | working |
| mAP@0.5 (overall and by object size), ECE | `tinyseek.eval.evaluate` | working |
| ESP32-S3 budget check | `tinyseek.train.budget` | working |
| int8 ESP-DL export + int8 accuracy check | `tinyseek.train.export` | working |
| ESP32 photo collection (interval capture to SD) | `firmware/capture` | written; not yet flashed |
| ESP32-camera domain LoRA | — | needs ~200-300 photos from the board |
| On-device detection firmware | `firmware/` | not started |

## Setup

PyTorch is installed separately so the CUDA build matches the machine. On Pascal GPUs
(e.g. GTX 1080 Ti, sm_61) use the cu126 wheels; the cu128+ builds dropped Pascal.

```bash
python3 -m venv ~/.venvs/tinyseek && source ~/.venvs/tinyseek/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[generate,label,eval,export,dev]"
pytest
```

## Pipeline

All paths below are relative to the repo root; `data/` and `runs/` are gitignored.

```bash
# 1. Real data: COCO annotations, per-prompt subsets, and only the images they use (~4 GB)
python -m tinyseek.data.coco --prompts person cup backpack

# 2. Synthetic images (resumable; ~2.4 s/image on a 1080 Ti in fp32)
python -m tinyseek.generate.sample prompt=cup num_images=2000 synth_set=sd15_lcm_v0

# 3. Teacher labels: one raw pass, then filtered copies at several thresholds
python -m tinyseek.label.teacher --teacher owlv2 --prompt cup \
    --images data/synthetic/cup/sd15_lcm_v0 --out data/labels/owlv2/synth_sd15_lcm_v0_cup_raw.json
python -m tinyseek.label.teacher --teacher owlv2 --prompt cup \
    --coco data/coco/subsets/train2017_cup.json --images data/coco/train2017 \
    --out data/labels/owlv2/train2017_cup_raw.json
#    Score a teacher against COCO ground truth (pick the teacher and threshold):
python -m tinyseek.label.teacher --teacher grounding_dino --prompt cup \
    --coco data/coco/subsets/val2017_cup.json --images data/coco/val2017 \
    --gt data/coco/subsets/val2017_cup.json --out data/labels/grounding_dino/val2017_cup_raw.json

# 4. Train the student under each condition (config overrides are key=value)
python -m tinyseek.train.train --config configs/real_only.yaml    prompt=cup seed=0
python -m tinyseek.train.train --config configs/teacher_real.yaml prompt=cup seed=0
python -m tinyseek.train.train --config configs/synth_only.yaml   prompt=cup seed=0 synth_set=sd15_lcm_v0
python -m tinyseek.train.train --config configs/synth_real.yaml   prompt=cup seed=0 synth_set=sd15_lcm_v0

# 5. Check the ESP32-S3 budget, then quantize and export for ESP-DL
python -m tinyseek.train.budget --config configs/base.yaml
python -m tinyseek.train.export --run runs/cup/real_only/seed0
```

Each training run writes `runs/<prompt>/<condition>/seed<seed>/` with the resolved
`config.yaml` (including the git revision), `train_log.jsonl`, `last.pt`, and
`eval.json`. The reported model is always the final epoch; COCO val2017 is the test
set and is never used to pick checkpoints.

On a SLURM cluster, any step runs through `scripts/slurm/run.sbatch`:

```bash
sbatch -A <PROJECT_ACCOUNT> scripts/slurm/run.sbatch tinyseek.train.train --config configs/synth_only.yaml prompt=cup
```

## Training conditions

| Config | Training data | Labels |
|---|---|---|
| `real_only` | COCO train2017 subset | human (COCO) |
| `teacher_real` | same COCO images | teacher |
| `synth_only` | Stable Diffusion images | teacher |
| `synth_real` | SD images + COCO subset | teacher + human |

The data-efficiency sweep uses `max_images` on the synthetic source (see `configs/smoke.yaml`
for the syntax).

## Student

CenterNet-style single-stride detector: a MobileNetV2-0.5 backbone (ImageNet-pretrained;
0.5 is the narrowest pretrained width in timm) with light top-down fusion of the stride
8/16/32 features into one 16x16 head at 128x128 input. It outputs a per-class heatmap plus box
size and center offset, so on-device post-processing is a 3x3 local-maximum test and a few
`expf` calls; no anchors or NMS. Default config: 0.50M parameters (~475 KB int8), ~0.6 MB
peak activations, 31M MACs (`tinyseek.train.budget`).

## Data and secrets

Nothing under `data/` or `runs/` is committed. Everything is regenerated by the commands
above. Wi-Fi credentials for the firmware go in a gitignored `secrets.h`.
