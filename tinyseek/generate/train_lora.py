"""Train a LoRA on the SD 1.5 UNet from captioned camera photos.

    python -m tinyseek.generate.train_lora --config configs/lora.yaml [steps=100 name=smoke]

The photo set is small, so everything except the UNet is computed once: each photo gets
``crops_per_image`` random square crops encoded by the VAE, and each caption is encoded
by the text encoder. The VAE and text encoder are then freed and only the LoRA weights
on the UNet's attention projections train (fp32, gradient checkpointing; fits 11 GB).
Checkpoints are written in the Diffusers LoRA format that ``tinyseek.generate.sample``
loads as ``domain_lora``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tinyseek.config import load_config, save_run_config, seed_everything


def random_crops(image: Image.Image, n: int, size: int, scale: tuple[float, float],
                 rng: np.random.Generator) -> list[Image.Image]:
    """``n`` random square crops, side = scale x the short side, resized to ``size``."""
    w, h = image.size
    crops = []
    for _ in range(n):
        side = int(min(w, h) * rng.uniform(*scale))
        x, y = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
        crops.append(image.crop((x, y, x + side, y + side)).resize((size, size), Image.BICUBIC))
    return crops


@torch.no_grad()
def encode_dataset(cfg: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """Returns (latents [N,4,h,w], caption embeddings [C,77,768], caption index per latent, empty-prompt embedding)."""
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    rows = [json.loads(line) for line in Path(cfg["captions"]).read_text().splitlines() if line.strip()]
    tokenizer = CLIPTokenizer.from_pretrained(cfg["base_model"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg["base_model"], subfolder="text_encoder").to(device)

    def embed(encoder: CLIPTextModel, texts: list[str]) -> torch.Tensor:
        ids = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        return encoder(ids)[0].cpu()

    captions = torch.cat([embed(text_encoder, [r["caption"] for r in rows[i : i + 32]])
                          for i in range(0, len(rows), 32)])
    empty = embed(text_encoder, [""])[0]
    del text_encoder

    vae = AutoencoderKL.from_pretrained(cfg["base_model"], subfolder="vae").to(device)
    rng = np.random.default_rng(cfg["seed"])
    latents, owner = [], []
    for i, r in enumerate(rows):
        image = Image.open(Path(cfg["images"]) / r["file"]).convert("RGB")
        crops = random_crops(image, cfg["crops_per_image"], cfg["resolution"], tuple(cfg["crop_scale"]), rng)
        pixels = torch.stack([torch.from_numpy(np.asarray(c, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
                              for c in crops]).to(device)
        latents.append(vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor)
        owner += [i] * len(crops)
    del vae
    torch.cuda.empty_cache()
    return torch.cat(latents).cpu(), captions, owner, empty


def save_lora(unet, out_dir: Path) -> None:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
        safe_serialization=True,
    )


def train(cfg: dict) -> Path:
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig

    seed_everything(cfg["seed"])
    run_dir = Path(cfg["run_root"]) / "lora" / cfg["name"]
    save_run_config(cfg, run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    latents, captions, owner, empty = encode_dataset(cfg, device)
    print(f"cached {len(latents)} latents from {len(captions)} photos in {time.time() - t0:.0f}s", flush=True)

    scheduler = DDPMScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(cfg["base_model"], subfolder="unet").to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], init_lora_weights="gaussian",
                                target_modules=cfg["target_modules"]))
    unet.enable_gradient_checkpointing()
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA parameters: {sum(p.numel() for p in params):,}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-2)

    gen = torch.Generator().manual_seed(cfg["seed"])
    log = (run_dir / "train_log.jsonl").open("a")
    running, t0 = 0.0, time.time()
    for step in range(1, cfg["steps"] + 1):
        lr = cfg["lr"] * min(1.0, step / max(1, cfg["warmup_steps"]))
        lr *= 0.5 * (1 + math.cos(math.pi * max(0, step - cfg["warmup_steps"]) / max(1, cfg["steps"] - cfg["warmup_steps"])))
        for g in optimizer.param_groups:
            g["lr"] = lr

        idx = torch.randint(len(latents), (cfg["batch_size"],), generator=gen)
        x0 = latents[idx]
        flip = torch.rand(len(idx), generator=gen) < 0.5
        x0[flip] = x0[flip].flip(-1)
        cond = captions[[owner[i] for i in idx.tolist()]]
        drop = torch.rand(len(idx), generator=gen) < cfg["caption_dropout"]
        cond[drop] = empty
        x0, cond = x0.to(device), cond.to(device)

        noise = torch.randn(x0.shape, generator=gen).to(device)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (len(idx),), generator=gen).to(device)
        pred = unet(scheduler.add_noise(x0, noise, t), t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        running += loss.item()

        if step % 25 == 0:
            entry = {"step": step, "loss": round(running / 25, 5), "lr": lr,
                     "s_per_step": round((time.time() - t0) / step, 3)}
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            running = 0.0
        if step % cfg["checkpoint_every"] == 0 or step == cfg["steps"]:
            save_lora(unet, run_dir / f"checkpoint-{step}")
    save_lora(unet, run_dir / "final")
    print(f"LoRA -> {run_dir / 'final'}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. steps=100 name=smoke")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
