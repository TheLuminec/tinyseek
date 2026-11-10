"""Generate a synthetic image set for one prompt with Stable Diffusion.

    python -m tinyseek.generate.sample --config configs/generate.yaml prompt=cup num_images=2000

Resumable: existing images are skipped. Every image has its own seed derived from
(config seed, image index), and its text prompt and seed are logged to
``metadata.jsonl``, so any single image can be regenerated exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tinyseek.config import deep_merge, load_config, load_prompts, save_run_config


def image_seed(seed: int, index: int) -> int:
    return seed * 1_000_003 + index


def text_prompt(spec: dict, scenes: list[str], seed: int, index: int) -> str:
    """Deterministically pick a template and scene for image ``index``.

    A prompt's own ``scenes`` list (e.g. parking garages for "car") replaces the shared one.
    """
    rng = np.random.default_rng([seed, index])
    template = spec["sd_templates"][rng.integers(len(spec["sd_templates"]))]
    scenes = spec.get("scenes") or scenes
    return template.format(scene=scenes[rng.integers(len(scenes))])


def with_trigger(text: str, cfg: dict) -> str:
    """Prefix the domain LoRA's trigger phrase, which carries the camera's look."""
    trigger = cfg.get("domain_trigger") if cfg.get("domain_lora") else None
    return f"{trigger}, {text}" if trigger else text


def load_pipeline(cfg: dict, device: torch.device):
    from diffusers import DiffusionPipeline, LCMScheduler

    dtype = getattr(torch, cfg.get("dtype", "float32"))
    pipe = DiffusionPipeline.from_pretrained(cfg["base_model"], torch_dtype=dtype, safety_checker=None)
    adapters, weights = [], []
    if cfg.get("lcm_lora"):
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(cfg["lcm_lora"], adapter_name="lcm")
        adapters.append("lcm")
        weights.append(1.0)
    if cfg.get("domain_lora"):
        pipe.load_lora_weights(cfg["domain_lora"], adapter_name="domain")
        adapters.append("domain")
        weights.append(cfg.get("domain_lora_scale", 1.0))
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)
    pipe.set_progress_bar_config(disable=True)
    # Decoding a batch of 512 px latents at once peaks near 10 GB in fp32; decode one at a time.
    pipe.vae.enable_slicing()
    return pipe.to(device)


def resolve_prompt_overrides(cfg: dict) -> dict:
    """Apply ``prompt_overrides[<prompt>]`` (e.g. a different domain LoRA for one object)."""
    return deep_merge(cfg, cfg.get("prompt_overrides", {}).get(cfg["prompt"], {}))


def generate(cfg: dict) -> Path:
    cfg = resolve_prompt_overrides(cfg)
    prompts = load_prompts()
    spec = prompts["prompts"][cfg["prompt"]]
    out_dir = Path(cfg["data_root"]) / "synthetic" / cfg["prompt"] / cfg["synth_set"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, out_dir)

    todo = [i for i in range(cfg["num_images"]) if not (out_dir / f"{i:06d}.jpg").exists()]
    if not todo:
        print(f"{out_dir}: all {cfg['num_images']} images exist")
        return out_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(cfg, device)

    with (out_dir / "metadata.jsonl").open("a") as meta:
        bs = cfg["batch_size"]
        for start in tqdm(range(0, len(todo), bs), desc=f"SD {cfg['prompt']}", mininterval=30):
            idx = todo[start : start + bs]
            texts = [with_trigger(text_prompt(spec, prompts["scenes"], cfg["seed"], i), cfg) for i in idx]
            gens = [torch.Generator(device).manual_seed(image_seed(cfg["seed"], i)) for i in idx]
            images = pipe(
                prompt=texts,
                negative_prompt=[cfg["negative_prompt"]] * len(idx) if cfg.get("guidance_scale", 0) > 1 else None,
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance_scale"],
                height=cfg["resolution"],
                width=cfg["resolution"],
                generator=gens,
            ).images
            for i, text, image in zip(idx, texts, images):
                image.save(out_dir / f"{i:06d}.jpg", quality=95)
                meta.write(json.dumps({"index": i, "file_name": f"{i:06d}.jpg", "prompt": text,
                                       "seed": image_seed(cfg["seed"], i)}) + "\n")
            meta.flush()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/generate.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. prompt=backpack num_images=100")
    args = parser.parse_args()
    print(generate(load_config(args.config, args.overrides)))


if __name__ == "__main__":
    main()
