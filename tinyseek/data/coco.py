"""Fetch MS COCO annotations and build per-prompt subsets, downloading only the images they need.

    python -m tinyseek.data.coco --prompts person cup backpack

Writes, under ``data/coco/``:
  annotations/instances_{train,val}2017.json   (official, from the trainval zip)
  subsets/train2017_<prompt>.json  positives (capped) + sampled negatives
  subsets/val2017_<prompt>.json    all 5,000 val images (the test set)
  train2017/, val2017/             only images referenced by the subsets

Subsets keep only the prompt's category; crowd annotations are kept (flagged
``iscrowd``) so pycocotools can ignore detections on them as usual.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tinyseek.config import load_prompts

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch_annotations(coco_root: Path) -> None:
    ann_dir = coco_root / "annotations"
    if (ann_dir / "instances_train2017.json").exists() and (ann_dir / "instances_val2017.json").exists():
        return
    coco_root.mkdir(parents=True, exist_ok=True)
    zip_path = coco_root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"downloading {ANNOTATIONS_URL}")
        urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
            zf.extract(name, coco_root)
    zip_path.unlink()


def build_subset(
    coco: dict,
    category_id: int,
    max_positives: int | None,
    negative_ratio: float | None,
    seed: int,
) -> dict:
    """Images containing the category (capped) plus ``negative_ratio`` x as many without it.

    ``negative_ratio=None`` keeps every image (used for the val/test set).
    """
    anns = [a for a in coco["annotations"] if a["category_id"] == category_id]
    positive_ids = sorted({a["image_id"] for a in anns if not a.get("iscrowd", 0)})
    if negative_ratio is None:
        image_ids = {img["id"] for img in coco["images"]}
    else:
        rng = np.random.default_rng(seed)
        if max_positives is not None and len(positive_ids) > max_positives:
            positive_ids = sorted(rng.choice(positive_ids, max_positives, replace=False).tolist())
        has_category = {a["image_id"] for a in anns}
        negative_pool = sorted(img["id"] for img in coco["images"] if img["id"] not in has_category)
        n_neg = min(len(negative_pool), round(len(positive_ids) * negative_ratio))
        negative_ids = rng.choice(negative_pool, n_neg, replace=False).tolist()
        image_ids = set(positive_ids) | set(negative_ids)
    return {
        "info": {"description": f"COCO subset for category {category_id}", "seed": seed},
        "images": [img for img in coco["images"] if img["id"] in image_ids],
        "annotations": [a for a in anns if a["image_id"] in image_ids],
        "categories": [c for c in coco["categories"] if c["id"] == category_id],
    }


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def fetch_images(subset: dict, image_dir: Path, workers: int = 16) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    todo = [img for img in subset["images"] if not (image_dir / img["file_name"]).exists()]
    if not todo:
        return
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(_download, img["coco_url"], image_dir / img["file_name"]) for img in todo]
        for job in tqdm(jobs, desc=f"images -> {image_dir.name}", mininterval=10):
            job.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", nargs="+", default=["person", "cup", "backpack"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-train-positives", type=int, default=6000)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-images", action="store_true", help="only write subset JSON files")
    args = parser.parse_args()

    coco_root = Path(args.data_root) / "coco"
    fetch_annotations(coco_root)
    prompts = load_prompts()["prompts"]
    (coco_root / "subsets").mkdir(parents=True, exist_ok=True)
    for split in ("val2017", "train2017"):
        with (coco_root / "annotations" / f"instances_{split}.json").open() as f:
            coco = json.load(f)
        for name in args.prompts:
            is_train = split == "train2017"
            subset = build_subset(
                coco, prompts[name]["coco_id"],
                max_positives=args.max_train_positives if is_train else None,
                negative_ratio=args.negative_ratio if is_train else None,
                seed=args.seed,
            )
            out = coco_root / "subsets" / f"{split}_{name}.json"
            out.write_text(json.dumps(subset))
            print(f"{out}: {len(subset['images'])} images, {len(subset['annotations'])} annotations")
            if not args.no_images:
                fetch_images(subset, coco_root / split)
        del coco


if __name__ == "__main__":
    main()
