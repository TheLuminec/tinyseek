"""Open-vocabulary teacher labeling (OWLv2 or Grounding DINO) into COCO-format JSON.

    # Label a COCO subset (real images) or a directory of synthetic images:
    python -m tinyseek.label.teacher --teacher owlv2 --prompt cup \
        --coco data/coco/subsets/val2017_cup.json --images data/coco/val2017 \
        --out data/labels/owlv2/val2017_cup_raw.json --gt data/coco/subsets/val2017_cup.json
    python -m tinyseek.label.teacher --teacher owlv2 --prompt cup \
        --images data/synthetic/cup/sd15_lcm_v0 --out data/labels/owlv2/synth_sd15_lcm_v0_cup_raw.json

The teacher runs once at a low ``--min-score`` and writes every box with its score
(``*_raw.json``). ``--thresholds`` then writes filtered copies (``*_t<thr>.json``) that
training configs reference, so a threshold sweep never re-runs the teacher. With
``--gt``, the teacher itself is scored (AP@0.5) against ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

from tinyseek.config import load_prompts

TEACHERS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def images_from_dir(image_dir: Path) -> list[dict]:
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = []
    for i, p in enumerate(files, start=1):
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": i, "file_name": p.name, "width": w, "height": h})
    return images


class Teacher:
    def __init__(self, name: str, device: torch.device) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name = name
        self.device = device
        self.processor = AutoProcessor.from_pretrained(TEACHERS[name])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(TEACHERS[name]).to(device).eval()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], text: str, min_score: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return per-image (boxes xyxy in original pixels, scores)."""
        sizes = [(im.height, im.width) for im in images]
        if self.name == "grounding_dino":
            # Grounding DINO expects lowercase phrases terminated by a period.
            query = text.lower().rstrip(".") + "."
            inputs = self.processor(images=images, text=[query] * len(images), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=min_score, text_threshold=min_score, target_sizes=sizes
            )
        else:
            inputs = self.processor(text=[[text]] * len(images), images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # OWLv2 pads each image (bottom/right) to a square before resizing, so boxes
            # are relative to the padded square: rescale with the long side for both axes.
            square = [(max(h, w), max(h, w)) for h, w in sizes]
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=min_score, target_sizes=square
            )
        out = []
        for res, (h, w) in zip(results, sizes):
            boxes, scores = res["boxes"].float().cpu(), res["scores"].float().cpu()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
            keep = nms(boxes, scores, 0.5)
            out.append((boxes[keep], scores[keep]))
        return out


def label_images(
    teacher: Teacher, images: list[dict], image_dir: Path, text: str, category_id: int,
    min_score: float, batch_size: int,
) -> list[dict]:
    annotations = []
    for start in tqdm(range(0, len(images), batch_size), desc=f"{teacher.name} '{text}'", mininterval=10):
        chunk = images[start : start + batch_size]
        pil = [Image.open(image_dir / img["file_name"]).convert("RGB") for img in chunk]
        for img, (boxes, scores) in zip(chunk, teacher(pil, text, min_score)):
            for (x0, y0, x1, y1), s in zip(boxes.tolist(), scores.tolist()):
                w, h = x1 - x0, y1 - y0
                if w < 1 or h < 1:
                    continue
                annotations.append({
                    "id": len(annotations) + 1, "image_id": img["id"], "category_id": category_id,
                    "bbox": [x0, y0, w, h], "area": w * h, "iscrowd": 0, "score": s,
                })
    return annotations


def filter_by_score(coco: dict, threshold: float) -> dict:
    return dict(coco, annotations=[a for a in coco["annotations"] if a["score"] >= threshold])


def threshold_path(raw_path: Path, threshold: float) -> Path:
    stem = raw_path.name.removesuffix("_raw.json")
    return raw_path.with_name(f"{stem}_t{threshold}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", choices=sorted(TEACHERS), required=True)
    parser.add_argument("--prompt", required=True, help="key in configs/prompts.yaml")
    parser.add_argument("--images", required=True, help="image directory")
    parser.add_argument("--coco", help="COCO JSON listing the images to label (default: every image in --images)")
    parser.add_argument("--out", required=True, help="output path ending in _raw.json")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--batch-size", type=int, help="default: 8 for owlv2, 2 for grounding_dino (11 GB GPU)")
    parser.add_argument("--gt", help="COCO JSON with ground truth; prints the teacher's own AP@0.5")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.name.endswith("_raw.json"):
        parser.error("--out must end in _raw.json")
    spec = load_prompts()["prompts"][args.prompt]
    image_dir = Path(args.images)
    if args.coco:
        with open(args.coco) as f:
            images = [{k: img[k] for k in ("id", "file_name", "width", "height")} for img in json.load(f)["images"]]
    else:
        images = images_from_dir(image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = Teacher(args.teacher, device)
    batch_size = args.batch_size or {"owlv2": 8, "grounding_dino": 2}[args.teacher]
    annotations = label_images(teacher, images, image_dir, spec["teacher_text"], spec["coco_id"], args.min_score, batch_size)
    coco = {
        "info": {"teacher": TEACHERS[args.teacher], "text": spec["teacher_text"], "min_score": args.min_score},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": spec["coco_id"], "name": args.prompt}],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco))
    print(f"{out}: {len(images)} images, {len(annotations)} boxes")
    for t in args.thresholds:
        filtered = filter_by_score(coco, t)
        threshold_path(out, t).write_text(json.dumps(filtered))
        print(f"  t={t}: {len(filtered['annotations'])} boxes")

    if args.gt:
        from tinyseek.eval.metrics import coco_ap50

        dets = [{"image_id": a["image_id"], "bbox": a["bbox"], "score": a["score"]} for a in annotations]
        ap = coco_ap50(args.gt, dets, spec["coco_id"])
        print(json.dumps({"teacher": args.teacher, "prompt": args.prompt, "ap50_vs_gt": ap}))
        out.with_name(out.name.replace("_raw.json", "_teacher_ap.json")).write_text(
            json.dumps({"teacher": args.teacher, "prompt": args.prompt, "min_score": args.min_score, "ap50": ap})
        )


if __name__ == "__main__":
    main()
