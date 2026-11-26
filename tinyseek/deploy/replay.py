"""Score the detector's replay results (firmware/detector, source=replay) on the PC.

    python -m tinyseek.deploy.replay --card runs/deploy/detector_card \\
        --results runs/deploy/detector_card/results/run_001 [--json out.json]

``--card`` is the folder ``tinyseek.deploy.detector_card`` staged (its card.json maps each
replay file to a COCO image id); ``--results`` is a copy of the card's
``/tinyseek/results/run_NNN`` folder (dets.jsonl, run.txt, summary.txt). The board writes
every detection its decode keeps (score >= the model's score_threshold, at most
max_detections), so the AP@0.5 here is computed exactly as tinyseek.eval.evaluate does for
the PC numbers (pycocotools over the same images, plus the same ECE), and the two are
directly comparable: the difference is the deployed int8 model and the board's own JPEG
decode and preprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tinyseek.deploy.layout import read_kv
from tinyseek.eval.metrics import coco_ap50_by_area, expected_calibration_error, match_detections

STAGES = ("cap", "dec", "pre", "inf", "post", "lat")


def read_results(results: Path) -> list[dict]:
    lines = [ln.strip() for ln in (results / "dets.jsonl").read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    rows = []
    for n, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A reset mid-write can leave a partial last line; anything else is a real problem.
            if n != len(lines):
                raise
    return rows


def score(card: Path, results: Path, iou_threshold: float = 0.5, ece_bins: int = 10) -> dict:
    manifest = json.loads((card / "card.json").read_text())
    if "replay" not in manifest:
        raise SystemExit(f"{card}/card.json has no replay set (stage with --replay-ann)")
    rep = manifest["replay"]
    # FAT may report the staged 8.3 names in upper case ("00001.JPG").
    by_file = {f["file"].lower(): f["image_id"] for f in rep["files"]}
    labels = card / rep["labels"]
    category_id = rep["category_id"]
    run = read_kv(results / "run.txt") if (results / "run.txt").exists() else {}
    rows = read_results(results)

    dets, seen, failed, unknown = [], set(), [], []
    times = {k: [] for k in STAGES}
    for r in rows:
        image_id = by_file.get(r["file"].lower())
        if image_id is None:
            unknown.append(r["file"])
            continue
        seen.add(r["file"].lower())
        if not r.get("ok", True):
            failed.append({"file": r["file"], "error": r.get("error", "")})
            continue
        for k in STAGES:
            times[k].append(float(r["t"][k]))
        for d in r["dets"]:
            dets.append({"image_id": image_id, "bbox": [d["x"], d["y"], d["w"], d["h"]], "score": d["s"]})
    image_ids = [f["image_id"] for f in rep["files"]]
    missing = [f["file"] for f in rep["files"] if f["file"].lower() not in seen]

    coco = json.loads(labels.read_text())
    gt: dict[int, list] = {i: [] for i in image_ids}
    for a in coco["annotations"]:
        if a["category_id"] == category_id and not a.get("iscrowd", 0) and a["image_id"] in gt:
            gt[a["image_id"]].append(a["bbox"])
    gt_arr = {i: np.asarray(b, np.float64).reshape(-1, 4) for i, b in gt.items()}
    scores, tp, num_gt = match_detections(dets, gt_arr, iou_threshold)
    by_area = coco_ap50_by_area(str(labels), dets, category_id, image_ids)
    out = {
        "ap50": by_area["all"],
        "ap50_small": by_area["small"],
        "ap50_medium": by_area["medium"],
        "ap50_large": by_area["large"],
        "ece": expected_calibration_error(scores, tp, ece_bins),
        "num_images": len(image_ids),
        "num_gt": num_gt,
        "num_detections": len(dets),
        "num_results": len(rows),
        "failed": failed,
        "missing": missing,
        "unknown_files": unknown,
        "complete": run.get("complete") == "1",
        "prompt": run.get("prompt"),
        "model_sha256": run.get("model_sha256"),
        "settings": json.loads(run["settings"]) if "settings" in run else None,
        "mean_ms": {k: float(np.mean(v)) if v else None for k, v in times.items()},
    }
    staged = {m["prompt"]: m for m in manifest["models"]}.get(rep["prompt"], {})
    out["warnings"] = []
    if out["prompt"] and out["prompt"] != rep["prompt"]:
        out["warnings"].append(f"the board ran prompt {out['prompt']!r}, the replay set was staged for {rep['prompt']!r}")
    if out["model_sha256"] and staged.get("espdl_sha256") and out["model_sha256"] != staged["espdl_sha256"]:
        out["warnings"].append("the board's model is not the one staged in card.json")
    if missing:
        out["warnings"].append(f"{len(missing)} replay files have no result (interrupted run?): AP counts them as empty")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", type=Path, required=True, help="the detector_card --out folder (has card.json)")
    parser.add_argument("--results", type=Path, required=True, help="a copy of /tinyseek/results/run_NNN")
    parser.add_argument("--json", type=Path, help="also write the result here")
    args = parser.parse_args()
    out = score(args.card, args.results)
    text = json.dumps(out, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
