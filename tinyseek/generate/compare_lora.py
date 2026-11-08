"""Compare domain-LoRA checkpoints against plain SD 1.5 on the same prompts and seeds.

    python -m tinyseek.generate.compare_lora --run runs/lora/esp32cam_v1 \
        --real data/esp32cam/kept_coco.json --real-images data/esp32cam/raw

Images come from the real generation pipeline (configs/generate.yaml: LCM-LoRA, fp32),
with the trigger phrase added for LoRA columns as tinyseek.generate.sample does. Per
column it reports:
  real_sim   mean CLIP image-embedding cosine similarity to the centroid of the real
             camera photos (higher = closer to the camera's look)
  prompt_sim mean CLIP similarity between each image and "a photo of a <object>"
             (should not drop: the object must stay visible for the teacher)
  luma, sat  mean brightness and HSV saturation, beside the same for the real photos
  teacher_found  (with --teacher) per object, the fraction of images where the teacher's
             best box for that object scores at least --teacher-threshold
Images are saved under <run>/compare/<column>/. Writes compare.jpg (rows = prompt x seed, columns = base, checkpoints) and compare.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tinyseek.config import load_config, load_prompts
from tinyseek.generate.sample import image_seed, load_pipeline, text_prompt, with_trigger

CLIP = "openai/clip-vit-base-patch32"


class ClipScorer:
    def __init__(self, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP)

    @torch.no_grad()
    def images(self, images: list[Image.Image]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), 32):
            inputs = self.processor(images=images[i : i + 32], return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats if isinstance(feats, torch.Tensor) else feats.pooler_output
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats if isinstance(feats, torch.Tensor) else feats.pooler_output
        return torch.nn.functional.normalize(feats, dim=-1).cpu()


def pixel_stats(images: list[Image.Image]) -> dict[str, float]:
    luma = [float(np.asarray(im.convert("L"), dtype=np.float32).mean()) for im in images]
    sat = [float(np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean() / 2.55) for im in images]
    return {"luma": round(float(np.mean(luma)), 1), "sat": round(float(np.mean(sat)), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="LoRA run dir with checkpoint-* folders")
    parser.add_argument("--real", type=Path, required=True, help="COCO image list of the real camera photos")
    parser.add_argument("--real-images", type=Path, required=True)
    parser.add_argument("--config", default="configs/generate.yaml")
    parser.add_argument("--objects", nargs="+", default=["person", "cup", "backpack"])
    parser.add_argument("--seeds", type=int, default=8, help="images per object per column")
    parser.add_argument("--grid-seeds", type=int, default=2, help="of those, how many per object go in the grid")
    parser.add_argument("--scale", type=float, help="domain LoRA scale (default: the generate config's)")
    parser.add_argument("--scales", type=float, nargs="+", help="several LoRA scales, one column each")
    parser.add_argument("--checkpoints", type=int, nargs="+", help="only these checkpoint steps")
    parser.add_argument("--extra", nargs="+",
                        help="checkpoints from other runs as <checkpoint dir>:<scale>, e.g. a previous version")
    parser.add_argument("--teacher", choices=["owlv2", "grounding_dino"],
                        help="also report how often the teacher finds the prompted object")
    parser.add_argument("--teacher-threshold", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scale = args.scale if args.scale is not None else cfg.get("domain_lora_scale", 1.0)
    prompts = load_prompts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted(args.run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if args.checkpoints:
        checkpoints = [p for p in checkpoints if int(p.name.split("-")[1]) in args.checkpoints]
    scales = args.scales or [scale]
    extra = []
    for item in args.extra or []:
        path, _, sc = item.rpartition(":")
        ckpt = Path(path)
        extra.append((f"{ckpt.parent.name} {ckpt.name.replace('checkpoint-', 'step ')} x{sc}", ckpt, float(sc)))
    columns = [("base", None, 0.0)] + extra + [
        (f"{p.name.replace('checkpoint-', 'step ')}" + (f" x{sc}" if len(scales) > 1 else ""), p, sc)
        for p in checkpoints for sc in scales
    ]

    pipe = load_pipeline(dict(cfg, domain_lora=None), device)
    jobs = [(obj, k) for obj in args.objects for k in range(args.seeds)]
    generated: dict[str, list[Image.Image]] = {}
    for label, ckpt, sc in columns:
        if ckpt is not None:
            pipe.load_lora_weights(str(ckpt), adapter_name="domain")
            pipe.set_adapters(["lcm", "domain"], adapter_weights=[1.0, sc])
        col_cfg = dict(cfg, domain_lora=str(ckpt) if ckpt else None)
        images = []
        for obj, k in jobs:
            text = with_trigger(text_prompt(prompts["prompts"][obj], prompts["scenes"], cfg["seed"], k), col_cfg)
            images += pipe(prompt=text, num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance_scale"],
                           height=cfg["resolution"], width=cfg["resolution"],
                           generator=torch.Generator(device).manual_seed(image_seed(cfg["seed"], k))).images
        generated[label] = images
        col_dir = args.run / "compare" / label.replace(" ", "_")
        col_dir.mkdir(parents=True, exist_ok=True)
        for (obj, k), im in zip(jobs, images):
            im.save(col_dir / f"{obj}_{k:02d}.png")
        if ckpt is not None:
            pipe.delete_adapters("domain")
        print(f"generated column {label}", flush=True)
    del pipe
    torch.cuda.empty_cache()

    clip = ClipScorer(device)
    real_files = [img["file_name"] for img in json.loads(args.real.read_text())["images"]]
    real = [Image.open(args.real_images / f).convert("RGB") for f in real_files]
    centroid = torch.nn.functional.normalize(clip.images(real).mean(0), dim=0)
    object_text = clip.texts([f"a photo of a {o}" for o in args.objects])
    report = {"real_photos": dict(pixel_stats(real), n=len(real)), "columns": {}}
    obj_idx = torch.tensor([args.objects.index(obj) for obj, _ in jobs])
    for label, images in generated.items():
        feats = clip.images(images)
        prompt_sim = (feats * object_text[obj_idx]).sum(-1)
        report["columns"][label] = {
            "real_sim": round(float((feats @ centroid).mean()), 4),
            "prompt_sim": {o: round(float(prompt_sim[obj_idx == i].mean()), 4) for i, o in enumerate(args.objects)},
            **pixel_stats(images),
        }
    del clip
    if args.teacher:
        from tinyseek.label.teacher import Teacher

        teacher = Teacher(args.teacher, device)
        for label, images in generated.items():
            found = {}
            for i, obj in enumerate(args.objects):
                text = prompts["prompts"][obj]["teacher_text"]
                sel = [im for im, (o, _) in zip(images, jobs) if o == obj]
                top = [float(sc.max()) if len(sc) else 0.0
                       for _, sc in (r for j in range(0, len(sel), 8) for r in teacher(sel[j : j + 8], text, 0.05))]
                found[obj] = round(sum(t >= args.teacher_threshold for t in top) / len(top), 3)
            report["columns"][label]["teacher_found"] = found
        report["teacher"] = {"name": args.teacher, "threshold": args.teacher_threshold}
    print(json.dumps(report, indent=1))
    (args.run / "compare.json").write_text(json.dumps(report, indent=1))

    cell = 192
    rows = [(i, obj) for i, (obj, k) in enumerate(jobs) if k < args.grid_seeds]
    sheet = Image.new("RGB", (cell * (len(columns) + 1), cell * len(rows) + 24), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for c, (label, _, _) in enumerate(columns):
        draw.text((cell * c + 6, 6), label, fill=(255, 255, 255))
    draw.text((cell * len(columns) + 6, 6), "real photo", fill=(255, 255, 255))
    for r, (i, obj) in enumerate(rows):
        for c, (label, _, _) in enumerate(columns):
            sheet.paste(generated[label][i].resize((cell, cell)), (cell * c, 24 + cell * r))
        ref = real[(r * 17) % len(real)]
        side = min(ref.size)
        sheet.paste(ref.crop(((ref.width - side) // 2, 0, (ref.width + side) // 2, side)).resize((cell, cell)),
                    (cell * len(columns), 24 + cell * r))
        draw.text((4, 24 + cell * r + 4), obj, fill=(255, 255, 0))
    sheet.save(args.run / "compare.jpg", quality=90)
    print(f"grid -> {args.run / 'compare.jpg'}")


if __name__ == "__main__":
    main()
