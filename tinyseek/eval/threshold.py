"""Pick a detection threshold for a deployed model: precision, recall and F1 by score threshold.

    python -m tinyseek.eval.threshold --run runs/backpack/qat_far_v1_v2_tone/seed0 [--eval-set esp32_heldout]
        [--float] [--out <run>/threshold.json]

Scores the exported (quantized, as esp-ppq simulates it bit-exactly for the board) graph by
default, or the float checkpoint with --float. Writes threshold.json with the table and the
F1-maximising threshold; tinyseek.deploy.detector_card reads ``detect_threshold`` from
model_meta.json, so --write-meta copies it there. AP is threshold-free; this only sets the score
at which the board reports a detection (LED, web page).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tinyseek.config import fill, load_config, load_prompts
from tinyseek.data.dataset import load_coco_records
from tinyseek.eval.evaluate import predict
from tinyseek.eval.metrics import match_detections
from tinyseek.train.model import build_model

THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05))


def table(scores: np.ndarray, is_tp: np.ndarray, num_gt: int, thresholds=THRESHOLDS) -> list[dict]:
    """Precision / recall / F1 of the detections at or above each threshold."""
    rows = []
    for t in thresholds:
        keep = scores >= t
        tp = int(is_tp[keep].sum())
        n = int(keep.sum())
        p = tp / n if n else 1.0
        r = tp / num_gt if num_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"threshold": t, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                     "detections": n, "true_positives": tp})
    return rows


def best(rows: list[dict]) -> dict:
    """Highest F1; ties go to the higher threshold (fewer false alarms)."""
    return max(rows, key=lambda r: (r["f1"], r["threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--eval-set", default="esp32_heldout")
    parser.add_argument("--float", action="store_true", help="score the float checkpoint, not the exported graph")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-meta", action="store_true", help="set detect_threshold in model_meta.json")
    args = parser.parse_args()

    cfg = load_config(args.run / "config.yaml")
    category_id = load_prompts()["prompts"][cfg["prompt"]]["coco_id"]
    spec = cfg["eval"]["sets"][args.eval_set]
    root = Path(cfg["data_root"])
    ann = root / fill(spec["ann"], cfg)
    records = load_coco_records(ann, root / fill(spec["images"], cfg), category_id)
    size = cfg["model"]["input_size"]
    if args.float:
        model = build_model(dict(cfg["model"], pretrained=False))
        model.load_state_dict(torch.load(args.run / "last.pt", map_location="cpu")["model"])
        model.eval()
    else:
        from tinyseek.deploy.simulate import rebuild
        from tinyseek.train.export import QuantizedAdapter

        sim = rebuild(args.run, args.run / "threshold_scratch")
        model = QuantizedAdapter(sim.executor, sim.num_classes, sim.stride)
    dets = predict(model, records, size, torch.device("cpu"), 0.05, cfg["eval"]["max_detections"],
                   batch_size=16, num_workers=4)
    gt = {r.image_id: np.asarray([[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in r.boxes]) for r in records}
    scores, is_tp, num_gt = match_detections(dets, gt)
    rows = table(scores, is_tp, num_gt)
    pick = best(rows)
    result = {"run": str(args.run), "eval_set": args.eval_set, "scored": "float" if args.float else "quantized",
              "images": len(records), "num_gt": num_gt, "best": pick, "table": rows}
    out = args.out or args.run / "threshold.json"
    out.write_text(json.dumps(result, indent=1))
    if args.write_meta:
        meta_path = args.run / "model_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["detect_threshold"] = pick["threshold"]
        meta["detect_threshold_source"] = f"F1-max on {args.eval_set} ({result['scored']})"
        meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({"best": pick, "images": len(records), "num_gt": num_gt}))


if __name__ == "__main__":
    main()
