"""Detection dataset over COCO-format annotation files.

Ground truth, teacher labels, and synthetic sets all use the same COCO JSON layout
(``images``, ``annotations`` with ``bbox`` as xywh, ``categories``), so every training
condition is just a different list of (annotation file, image directory) sources.
Teacher-produced annotations also carry a ``score``; filtering happens upstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tinyseek.data.transforms import Letterbox, augment, letterbox
from tinyseek.train.model import IMAGENET_MEAN, IMAGENET_STD
from tinyseek.train.targets import encode_targets


@dataclass
class Record:
    image_path: Path
    image_id: int
    boxes: np.ndarray  # (N, 4) xyxy in original pixels, crowd boxes excluded


def load_coco_records(ann_file: str | Path, image_dir: str | Path, category_id: int) -> list[Record]:
    with Path(ann_file).open() as f:
        coco = json.load(f)
    by_image: dict[int, list[list[float]]] = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        if ann["category_id"] != category_id or ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        by_image[ann["image_id"]].append([x, y, x + w, y + h])
    return [
        Record(
            image_path=Path(image_dir) / img["file_name"],
            image_id=img["id"],
            boxes=np.asarray(by_image[img["id"]], dtype=np.float32).reshape(-1, 4),
        )
        for img in coco["images"]
    ]


def to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class DetectionDataset(Dataset):
    """Single-class detection samples, letterboxed to ``input_size``.

    In training mode, returns encoded CenterNet targets; in eval mode, returns the
    letterbox transform so predictions can be mapped back to original coordinates.
    """

    def __init__(
        self,
        records: list[Record],
        input_size: int,
        stride: int,
        train: bool,
        augment_cfg: dict | None = None,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.input_size = input_size
        self.stride = stride
        self.train = train
        self.augment_cfg = augment_cfg or {}
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        image = Image.open(rec.image_path).convert("RGB")
        if self.train:
            # Seeded per (seed, epoch, index) so augmentation is reproducible across workers.
            rng = np.random.default_rng([self.seed, self.epoch, i])
            image, boxes = augment(image, rec.boxes, self.input_size, self.augment_cfg, rng)
            targets = encode_targets(
                boxes, np.zeros(len(boxes), np.int64), 1, self.input_size, self.stride
            )
            return {"image": to_tensor(image), **{k: torch.from_numpy(v) for k, v in targets.items()}}
        image, lb = letterbox(image, self.input_size)
        return {"image": to_tensor(image), "image_id": rec.image_id, "letterbox": lb}


def collate_eval(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "letterbox": [b["letterbox"] for b in batch],
    }


__all__ = ["DetectionDataset", "Letterbox", "Record", "collate_eval", "load_coco_records", "to_tensor"]
