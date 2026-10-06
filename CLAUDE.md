# tinyseek

Synthetic-data distillation of prompt-specific object detectors for the ESP32-S3.

Goal: given a text prompt (e.g. "cup", "backpack"), train a tiny detector that runs on an ESP32-S3 camera board and outputs bounding boxes, using Stable Diffusion images labeled by an open-vocabulary teacher instead of manually labeled data.

Owner: Caleb Turner (solo). Started as a CSE 534 (graduate Generative AI, Miami University) final project, but the repo is a standalone project and should read that way: no course-specific naming in code, comments, or docs.

## Pipeline

1. **Generate**: Stable Diffusion via Hugging Face Diffusers, with a LoRA/QLoRA adapter fine-tuned on a few hundred ESP32-camera-style images to close the domain gap (low resolution, sensor noise).
2. **Label**: an open-vocabulary teacher (Grounding DINO or OWL-ViT) converts each prompt into bounding boxes; filter by a confidence threshold.
3. **Distill**: train a MobileNet-class student detector, quantize to int8, deploy to the ESP32-S3 (ESP-DL or TFLite Micro). Keep a laptop-hosted copy of the student as a fallback and for evaluation.
4. **Evaluate** four training conditions: real only, teacher-labeled real, synthetic only, synthetic plus real. Also sweep the number of synthetic images.

## Datasets

- MS COCO: real training images and the held-out evaluation set for 2-3 target prompts that map to COCO categories.
- Synthetic set: roughly 2,000-5,000 generated images per prompt, auto-labeled by the teacher.
- ESP32-S3 camera test set: roughly 200 self-collected photos from the target hardware, used to measure the real deployment domain.

## Metrics

- mAP@0.5
- Expected calibration error (ECE) of detection confidences
- CLIP semantic similarity between the prompt and detected crops
- FID/KID for realism of synthetic images (if time allows)
- Model size and on-device latency

## Expected outputs

- Results table: four training conditions x (mAP, ECE, model size, latency)
- Data-efficiency curve: mAP vs. number of synthetic images
- Gallery of synthetic images, plus success and failure cases from the real camera
- Live demo: ESP32-S3 drawing bounding boxes for a typed prompt
- Stretch goal: detector boxes drive a pan-tilt Nerf turret so it aims at the prompted object

## Constraints to design around

- ESP32-S3 has roughly 8 MB PSRAM; expect 96-160 px inputs and a MobileNet-class or FOMO-style model. Check memory and latency budgets before committing to an architecture.
- Training and generation run on a GPU machine or the OSC cluster (SLURM batch jobs); the board only runs inference. Keep training scripts runnable both interactively and as SLURM jobs, with paths and hyperparameters in config files rather than hardcoded.
- Experiments must be reproducible: fix seeds, log configs, and record which synthetic set and teacher threshold produced each result.

## Suggested layout (adjust as the repo grows)

```
tinyseek/
  CLAUDE.md
  README.md
  configs/        # experiment configs (one per training condition)
  data/           # gitignored; scripts to fetch/generate, not the data itself
  generate/       # Stable Diffusion + LoRA training and sampling
  label/          # teacher auto-labeling and filtering
  train/          # student training, quantization, export
  eval/           # mAP, ECE, CLIP similarity, plotting
  firmware/       # ESP32-S3 inference code and build files
  scripts/        # SLURM job scripts and utilities
```

## Working conventions

- Python for everything off-board (PyTorch, Hugging Face Diffusers/Transformers); C/C++ for firmware. Use type hints and keep functions small and testable.
- Never commit datasets, model checkpoints, generated images, or secrets (API tokens, Wi-Fi credentials). Keep them gitignored and document how to regenerate them.
- Before running anything long (generation, training, SLURM submission), state what will run, the expected duration, and the output location, and prefer a small smoke test first.
- When changing the student architecture, input size, or quantization settings, re-check that the model still fits on the ESP32-S3 before running full experiments.
- Report results with the exact config and seed used. Do not report numbers that were not produced by a run in this repo.
- Keep commits focused, with messages saying what changed and why.

## Open questions

- Final target prompts (must map to COCO categories for the standard mAP evaluation).
- Which teacher (Grounding DINO vs. OWL-ViT) gives cleaner boxes on synthetic images.
- Student architecture: MobileNet-class SSD vs. FOMO-style.
