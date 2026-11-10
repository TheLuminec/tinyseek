"""Generate-then-stylize: give clean SD images the camera look without moving their objects.

Plain SD draws the prompted object reliably and the teacher labels clean images best, but a
domain LoRA applied during generation also learns the *content* of the camera photos (mostly
empty rooms and garages) and suppresses the object. So: generate with plain SD, label those
images, then run a low-strength img2img pass with the domain LoRA. At low strength the layout
is kept, so the clean image's boxes stay valid for the stylized one.

    # stylize an existing plain-SD set (same file names, boxes reusable)
    python -m tinyseek.generate.stylize --src data/synthetic/car/sd15_lcm_plain \
        --out data/synthetic/car/sd15_lcm_plain_sty045 --strength 0.45

    # small experiment: plain SD vs direct LoRA vs stylized at several strengths
    python -m tinyseek.generate.stylize --compare --objects car chair --seeds 32 16 \
        --strengths 0.3 0.45 0.6 --out runs/stylize/test1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tinyseek.config import load_config, load_prompts, save_run_config
from tinyseek.generate.sample import image_seed, load_pipeline, text_prompt, with_trigger


def build_pipelines(cfg: dict, device: torch.device):
    """Text-to-image (LCM only) and img2img (LCM + domain LoRA) sharing one set of weights."""
    from diffusers import AutoPipelineForImage2Image

    t2i = load_pipeline(dict(cfg, domain_lora=None), device)
    t2i.load_lora_weights(cfg["domain_lora"], adapter_name="domain")
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    return t2i, i2i


def use_adapters(pipe, domain_scale: float | None) -> None:
    if domain_scale is None:
        pipe.set_adapters(["lcm"], adapter_weights=[1.0])
    else:
        pipe.set_adapters(["lcm", "domain"], adapter_weights=[1.0, domain_scale])


def stylize(i2i, image: Image.Image, text: str, seed: int, cfg: dict, strength: float,
            device: torch.device) -> Image.Image:
    use_adapters(i2i, cfg.get("domain_lora_scale", 1.0))
    return i2i(prompt=text, image=image, strength=strength, num_inference_steps=cfg.get("stylize_steps", 8),
               guidance_scale=cfg["guidance_scale"],
               generator=torch.Generator(device).manual_seed(seed)).images[0]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def box_survival(src_boxes: list[list[float]], new_boxes: list[list[float]]) -> tuple[int, int]:
    """How many source boxes still have a teacher box at IoU >= 0.5 after stylizing."""
    kept = sum(any(iou(s, n) >= 0.5 for n in new_boxes) for s in src_boxes)
    return kept, len(src_boxes)


def teacher_boxes(teacher, images: list[Image.Image], text: str, threshold: float) -> list[list[list[float]]]:
    out = []
    for i in range(0, len(images), 8):
        for boxes, scores in teacher(images[i : i + 8], text, 0.05):
            out.append([[float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
                        for (x0, y0, x1, y1), s in zip(boxes.tolist(), scores.tolist()) if s >= threshold])
    return out


def compare(args: argparse.Namespace, cfg: dict) -> dict:
    from tinyseek.generate.compare_lora import ClipScorer, pixel_stats
    from tinyseek.label.teacher import Teacher

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_prompts()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_run_config(dict(cfg, compare={"objects": args.objects, "seeds": args.seeds,
                                       "strengths": args.strengths}), out)
    jobs = [(obj, k) for obj, n in zip(args.objects, args.seeds) for k in range(n)]
    lora_cfg = dict(cfg)  # with_trigger adds the trigger only when a domain LoRA is set

    t2i, i2i = build_pipelines(cfg, device)
    columns: dict[str, list[Image.Image]] = {"plain SD": [], "direct LoRA": []}
    columns.update({f"stylized {s}": [] for s in args.strengths})
    for obj, k in jobs:
        spec, seed = prompts["prompts"][obj], image_seed(cfg["seed"], k)
        plain_text = text_prompt(spec, prompts["scenes"], cfg["seed"], k)
        lora_text = with_trigger(plain_text, lora_cfg)
        gen = {"num_inference_steps": cfg["steps"], "guidance_scale": cfg["guidance_scale"],
               "height": cfg["resolution"], "width": cfg["resolution"]}
        use_adapters(t2i, None)
        plain = t2i(prompt=plain_text, generator=torch.Generator(device).manual_seed(seed), **gen).images[0]
        use_adapters(t2i, cfg.get("domain_lora_scale", 1.0))
        direct = t2i(prompt=lora_text, generator=torch.Generator(device).manual_seed(seed), **gen).images[0]
        columns["plain SD"].append(plain)
        columns["direct LoRA"].append(direct)
        for s in args.strengths:
            columns[f"stylized {s}"].append(stylize(i2i, plain, lora_text, seed, cfg, s, device))
    del t2i, i2i
    torch.cuda.empty_cache()

    for label, images in columns.items():
        col_dir = out / label.replace(" ", "_")
        col_dir.mkdir(exist_ok=True)
        for (obj, k), im in zip(jobs, images):
            im.save(col_dir / f"{obj}_{k:02d}.png")

    clip = ClipScorer(device)
    real_files = [img["file_name"] for img in json.loads(Path(args.real).read_text())["images"]]
    centroid = torch.nn.functional.normalize(
        clip.images([Image.open(Path(args.real_images) / f).convert("RGB") for f in real_files]).mean(0), dim=0)
    sims = {label: (clip.images(images) @ centroid).numpy() for label, images in columns.items()}
    del clip

    teacher = Teacher("owlv2", device)
    report = {"columns": {}, "objects": args.objects, "teacher_threshold": args.teacher_threshold}
    found_boxes = {}
    for label, images in columns.items():
        per_obj = {}
        for obj in args.objects:
            idx = [i for i, (o, _) in enumerate(jobs) if o == obj]
            boxes = teacher_boxes(teacher, [images[i] for i in idx], prompts["prompts"][obj]["teacher_text"],
                                  args.teacher_threshold)
            found_boxes[(label, obj)] = boxes
            per_obj[obj] = {"found": round(sum(bool(b) for b in boxes) / len(boxes), 3),
                            "real_sim": round(float(np.mean([sims[label][i] for i in idx])), 4)}
        report["columns"][label] = {"real_sim": round(float(sims[label].mean()), 4), **pixel_stats(images),
                                    "per_object": per_obj}
    # Box survival: plain-SD teacher boxes still present (IoU >= 0.5, teacher score >= 0.2) after stylizing.
    for s in args.strengths:
        label = f"stylized {s}"
        for obj in args.objects:
            relaxed = teacher_boxes(teacher, [columns[label][i] for i, (o, _) in enumerate(jobs) if o == obj],
                                    prompts["prompts"][obj]["teacher_text"], 0.2)
            kept = total = 0
            for src, new in zip(found_boxes[("plain SD", obj)], relaxed):
                k, n = box_survival(src, new)
                kept, total = kept + k, total + n
            report["columns"][label]["per_object"][obj]["box_survival"] = round(kept / max(1, total), 3)
    (out / "compare.json").write_text(json.dumps(report, indent=1))

    cell, rows = 160, [i for i, (o, k) in enumerate(jobs) if k < 4]
    sheet = Image.new("RGB", (cell * len(columns), cell * len(rows) + 20), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(columns):
        draw.text((cell * c + 4, 4), label, fill=(255, 255, 255))
        for r, i in enumerate(rows):
            sheet.paste(columns[label][i].resize((cell, cell)), (cell * c, 20 + cell * r))
    sheet.save(out / "compare.jpg", quality=90)
    return report


def stylize_set(args: argparse.Namespace, cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_run_config(dict(cfg, stylize={"src": str(src), "strength": args.strength}), out)
    rows = [json.loads(line) for line in (src / "metadata.jsonl").read_text().splitlines() if line.strip()]
    _, i2i = build_pipelines(cfg, device)
    with (out / "metadata.jsonl").open("a") as meta:
        for r in rows:
            if (out / r["file_name"]).exists():
                continue
            text = with_trigger(r["prompt"], cfg)
            image = stylize(i2i, Image.open(src / r["file_name"]).convert("RGB"), text, r["seed"], cfg,
                            args.strength, device)
            image.save(out / r["file_name"], quality=95)
            meta.write(json.dumps(dict(r, prompt=text, stylized_from=str(src), strength=args.strength)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/generate.yaml")
    parser.add_argument("--compare", action="store_true", help="run the small comparison instead of a set")
    parser.add_argument("--objects", nargs="+", default=["car", "chair"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[32, 16], help="images per object")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.45, 0.6])
    parser.add_argument("--teacher-threshold", type=float, default=0.3)
    parser.add_argument("--real", default="data/esp32cam/kept_coco.json")
    parser.add_argument("--real-images", default="data/esp32cam/raw")
    parser.add_argument("--src", help="set to stylize (a plain-SD set)")
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--out", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if not cfg.get("domain_lora"):
        parser.error("stylizing needs domain_lora in the generate config")
    if args.compare:
        print(json.dumps(compare(args, cfg), indent=1))
    else:
        if not args.src:
            parser.error("--src is required without --compare")
        stylize_set(args, cfg)


if __name__ == "__main__":
    main()
