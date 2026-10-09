"""Run a student checkpoint on a COCO-format set and report mAP@0.5 and ECE.

    python -m tinyseek.eval.evaluate --run runs/cup/real_only/seed0 [--set coco_val]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tinyseek.config import fill, load_config, load_prompts
from tinyseek.data.dataset import DetectionDataset, Record, collate_eval, load_coco_records
from tinyseek.eval.metrics import (
    average_precision,
    coco_ap50,
    expected_calibration_error,
    match_detections,
)
from tinyseek.train.model import TinyCenterNet, build_model
from tinyseek.train.targets import decode


@torch.no_grad()
def predict(
    model: TinyCenterNet,
    records: list[Record],
    input_size: int,
    device: torch.device,
    score_threshold: float = 0.05,
    max_detections: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
) -> list[dict]:
    """Detections in original-image xywh coordinates, COCO results format."""
    ds = DetectionDataset(records, input_size, model.stride, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_eval)
    model.eval()
    results = []
    for batch in loader:
        out = model(batch["image"].to(device))
        boxes, scores, _ = decode(out["heatmap"], out["size"], out["offset"], model.stride, max_detections)
        for b, (image_id, lb) in enumerate(zip(batch["image_id"], batch["letterbox"])):
            keep = scores[b] >= score_threshold
            xyxy = lb.inverse(boxes[b][keep].cpu().numpy())
            for (x0, y0, x1, y1), s in zip(xyxy, scores[b][keep].cpu().numpy()):
                results.append(
                    {"image_id": image_id, "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "score": float(s)}
                )
    return results


def evaluate_set(
    model: TinyCenterNet,
    ann_file: str,
    image_dir: str,
    category_id: int,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    ev = cfg["eval"]
    records = load_coco_records(ann_file, image_dir, category_id)
    dets = predict(
        model, records, cfg["model"]["input_size"], device,
        score_threshold=ev["score_threshold"], max_detections=ev["max_detections"],
        num_workers=cfg["train"].get("num_workers", 4),
    )
    gt = {r.image_id: np.column_stack([r.boxes[:, :2], r.boxes[:, 2:] - r.boxes[:, :2]]) for r in records}
    scores, tp, num_gt = match_detections(dets, gt, ev["iou_threshold"])
    return {
        "ap50": coco_ap50(ann_file, dets, category_id),
        "ap50_ref": average_precision(scores, tp, num_gt),
        "ece": expected_calibration_error(scores, tp, ev["ece_bins"]),
        "num_images": len(records),
        "num_gt": num_gt,
        "num_detections": len(dets),
    }


def evaluate_all(model: TinyCenterNet, cfg: dict, device: torch.device) -> dict[str, dict]:
    prompts = load_prompts()["prompts"]
    category_id = prompts[cfg["prompt"]]["coco_id"]
    root = Path(cfg["data_root"])
    return {
        name: evaluate_set(
            model, str(root / fill(s["ann"], cfg)), str(root / fill(s["images"], cfg)),
            category_id, cfg, device,
        )
        for name, s in cfg["eval"]["sets"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory containing config.yaml and last.pt")
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. eval.score_threshold=0.01")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dict(cfg["model"], pretrained=False)).to(device)
    model.load_state_dict(torch.load(run / args.checkpoint, map_location=device)["model"])
    metrics = evaluate_all(model, cfg, device)
    print(json.dumps(metrics, indent=2))
    (run / "eval.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
